//! Roadmap step 1: KMS bring-up.
//!
//! Open the DRM device through libseat, pick a connector/mode/CRTC/primary
//! plane, allocate a scanout buffer with GBM, export it as a dma-buf, import
//! that as a KMS framebuffer, show it with an atomic commit, and restore the
//! previous display state on exit (normal return, error, panic unwind, or a
//! termination signal).
//!
//! Usage: vellum [--probe] [--seconds N]
//!   --probe      open the card directly (no seat, no DRM master), do
//!                everything except the modeset, and print what was chosen
//!   --seconds N  how long to show the color; 0 = until a signal (default 10)

mod dmabuf;
mod gbm;
mod kms;
mod session;
mod signals;

use std::os::fd::AsFd;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use drm::buffer::{DrmFourcc, DrmModifier};
use drm::{ClientCapability, Device as DrmDevice, DriverCapability};
use rustix::event::{PollFd, PollFlags, Timespec};

use crate::kms::{Card, Framebuffer, Output};
use crate::session::Session;
use crate::signals::SignalFd;

/// Fill color, 0x00RRGGBB.
const COLOR: u32 = 0x00_28_3c_5a;

struct Args {
    probe: bool,
    seconds: u64,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Args {
            probe: false,
            seconds: 10,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--probe" => args.probe = true,
                "--seconds" => {
                    let n = it.next().context("--seconds needs a value")?;
                    args.seconds = n.parse().context("--seconds expects an integer")?;
                }
                other => bail!("unknown argument {other:?}"),
            }
        }
        Ok(args)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vellum: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    // Before anything else (and before any threads exist).
    let signals = SignalFd::new(&[libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT])
        .context("setting up signalfd")?;
    let path = kms::find_card()?;
    eprintln!("vellum: using {}", path.display());

    if args.probe {
        let card = Card::open_direct(&path)?;
        return probe(&card);
    }

    let mut session = Session::open()?;
    let card = Card::new(session.open_device(&path)?);
    let output = prepare(&card)?;

    let gbm = gbm::Device::new(card.as_fd()).context("gbm_create_device")?;
    let mut bo = allocate_scanout(&gbm, &card, &output)?;
    fill(&mut bo, COLOR)?;
    let dmabuf = bo.export().context("exporting dma-buf")?;
    let fb = Framebuffer::import(&card, &dmabuf)?;

    // Locals drop in reverse order: the display state is restored first, then
    // the framebuffer, buffers, card and finally the seat are released. This
    // also holds when unwinding from a panic.
    let _display = kms::Display::take_over(&card, &output, &fb, session.active_flag())?;
    eprintln!(
        "vellum: showing {}x{} on {}; exiting in {}",
        dmabuf.width,
        dmabuf.height,
        output.connector_name,
        match args.seconds {
            0 => "never (send SIGTERM or switch VT)".to_owned(),
            s => format!("{s}s"),
        }
    );

    let deadline = (args.seconds > 0).then(|| Instant::now() + Duration::from_secs(args.seconds));
    loop {
        let timeout = match deadline {
            Some(d) => {
                let left = d.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                Some(Timespec {
                    tv_sec: left.as_secs() as _,
                    tv_nsec: left.subsec_nanos() as _,
                })
            }
            None => None,
        };

        let (seat_ready, signal_ready) = {
            let seat_fd = session.fd();
            let mut fds = [
                PollFd::new(&seat_fd, PollFlags::IN),
                PollFd::new(&signals, PollFlags::IN),
            ];
            match rustix::event::poll(&mut fds, timeout.as_ref()) {
                Ok(_) => {}
                Err(rustix::io::Errno::INTR) => continue,
                Err(e) => return Err(e).context("poll"),
            }
            (!fds[0].revents().is_empty(), !fds[1].revents().is_empty())
        };

        if signal_ready && let Some(sig) = signals.read()? {
            eprintln!("vellum: received {}, exiting", signals::name(sig));
            break;
        }
        if seat_ready {
            session.dispatch()?;
            if !session.active() {
                // VT switching proper is roadmap step 3; for now just leave.
                eprintln!("vellum: session disabled (VT switched away), exiting");
                break;
            }
        }
    }
    Ok(())
}

/// Enable atomic modesetting and choose the output.
fn prepare(card: &Card) -> Result<Output> {
    card.set_client_capability(ClientCapability::Atomic, true)
        .context("enabling DRM_CLIENT_CAP_ATOMIC (driver lacks atomic modesetting?)")?;
    kms::select_output(card)
}

/// Allocate an XRGB8888 scanout buffer the size of the mode, using a modifier
/// the primary plane advertises when the driver supports explicit modifiers.
fn allocate_scanout<'d>(gbm: &'d gbm::Device, card: &Card, output: &Output) -> Result<gbm::Bo<'d>> {
    let (w, h) = output.mode.size();
    let (w, h) = (u32::from(w), u32::from(h));
    let format = DrmFourcc::Xrgb8888;
    let flags = gbm::BoFlags::SCANOUT | gbm::BoFlags::RENDERING;

    let modifiers_supported = card
        .get_driver_capability(DriverCapability::AddFB2Modifiers)
        .is_ok_and(|v| v != 0);
    let modifiers = if modifiers_supported {
        kms::plane_modifiers(card, &output.plane_props, format)?
    } else {
        Vec::new()
    };

    if !modifiers.is_empty() {
        match gbm.create_bo(w, h, format, &modifiers, flags) {
            Ok(bo) => return Ok(bo),
            Err(e) => eprintln!("vellum: allocation with explicit modifiers failed ({e}); retrying implicit"),
        }
    }
    gbm.create_bo(w, h, format, &[], flags).context("gbm_bo_create")
}

fn fill(bo: &mut gbm::Bo, color: u32) -> Result<()> {
    let (w, h) = (bo.width() as usize, bo.height() as usize);
    let mut map = bo.map_write().context("gbm_bo_map")?;
    let stride = map.stride() as usize;
    let data = map.as_mut_slice();
    // XRGB8888 is a little-endian packed u32: bytes B, G, R, X.
    let pixel = color.to_le_bytes();
    for y in 0..h {
        for px in data[y * stride..y * stride + w * 4].chunks_exact_mut(4) {
            px.copy_from_slice(&pixel);
        }
    }
    Ok(())
}

/// Everything short of the modeset, without a seat. Useful from SSH or from
/// inside another graphical session to check device/plane/modifier selection.
fn probe(card: &Card) -> Result<()> {
    let output = prepare(card)?;
    let (w, h) = output.mode.size();
    println!(
        "output:    {} (connector {:?}), mode {}x{}@{}",
        output.connector_name,
        output.connector,
        w,
        h,
        output.mode.vrefresh()
    );
    println!("crtc:      {:?}", output.crtc);
    println!("plane:     {:?}", output.plane);

    let modifiers = kms::plane_modifiers(card, &output.plane_props, DrmFourcc::Xrgb8888)?;
    println!("plane supports {} XRGB8888 modifiers:", modifiers.len());
    for m in &modifiers {
        println!("  {m:#018x} {:?}", DrmModifier::from(*m));
    }

    let gbm = gbm::Device::new(card.as_fd()).context("gbm_create_device")?;
    let mut bo = allocate_scanout(&gbm, card, &output)?;
    fill(&mut bo, COLOR)?;
    let dmabuf = bo.export().context("exporting dma-buf")?;
    println!(
        "buffer:    {}x{} {:?}, modifier {:#018x} {:?}",
        dmabuf.width,
        dmabuf.height,
        dmabuf.format,
        u64::from(dmabuf.modifier),
        dmabuf.modifier
    );
    for (i, p) in dmabuf.planes.iter().enumerate() {
        println!("  plane {i}: stride {} offset {}", p.stride, p.offset);
    }
    let fb = Framebuffer::import(card, &dmabuf)?;
    println!("framebuffer: {:?}", fb.handle);
    println!("probe ok (modeset skipped: needs DRM master via the seat)");
    Ok(())
}
