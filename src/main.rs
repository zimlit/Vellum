//! Roadmap steps 1–2: KMS + Vulkan bring-up.
//!
//! Open the DRM device through libseat, pick a connector/mode/CRTC/primary
//! plane, and allocate two scanout buffers with GBM using a format modifier
//! both the plane and Vulkan support. Each buffer is exported as a dma-buf
//! and imported twice: as a KMS framebuffer and as a Vulkan image. Vulkan
//! clears the back buffer to an animated color; its completion semaphore is
//! exported as a sync_file and handed to the page flip as `IN_FENCE_FD`, and
//! page-flip events drive the next frame. The previous display state is
//! restored on exit (normal return, error, panic unwind, or a termination
//! signal).
//!
//! Usage: vellum [--probe] [--seconds N]
//!   --probe      open the card directly (no seat, no DRM master), render one
//!                frame without showing it, and print what was chosen
//!   --seconds N  how long to run; 0 = until a signal (default 10)

mod dmabuf;
mod gbm;
mod kms;
mod session;
mod signals;
mod vk;

use std::os::fd::{AsFd, OwnedFd};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use drm::buffer::DrmFourcc;
use drm::{ClientCapability, Device as DrmDevice, DriverCapability};
use rustix::event::{PollFd, PollFlags, Timespec};

use crate::kms::{Card, Framebuffer, Output};
use crate::session::Session;
use crate::signals::SignalFd;

const FORMAT: DrmFourcc = DrmFourcc::Xrgb8888;
const USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
const BUFFER_COUNT: usize = 2;

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

/// One scanout buffer: the same dma-buf seen by KMS and by Vulkan, plus the
/// per-buffer GPU submission state.
struct Buffer<'a> {
    image: Arc<vk::DmaBufImage>,
    fb: Framebuffer<'a>,
    cmd: vk::CommandBuffer,
    rendered: Arc<vk::Semaphore>,
    // The dma-buf keeps the memory alive on its own; holding the BO just
    // keeps GBM's view of it valid for the buffer's lifetime.
    bo: gbm::Bo<'a>,
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
    let instance = vk::Instance::new(cfg!(debug_assertions))?;
    let device = vulkan_device(&instance, &card)?;
    let gbm = gbm::Device::new(card.as_fd()).context("gbm_create_device")?;
    let modifiers = negotiate_modifiers(&card, &output, &device)?;
    let mut buffers = (0..BUFFER_COUNT)
        .map(|_| create_buffer(&card, &gbm, &device, &output, &modifiers))
        .collect::<Result<Vec<_>>>()?;

    // First frame: render, wait on the CPU, and modeset with it.
    let start = Instant::now();
    render(&mut buffers[0], start)?;
    buffers[0].cmd.wait()?;

    // Locals drop in reverse order: the display state is restored first (after
    // any pending flip lands), then GPU work is waited for and buffers, the
    // device, the card and finally the seat are released. This also holds
    // when unwinding from a panic.
    let mut display =
        kms::Display::take_over(&card, &output, &buffers[0].fb, session.active_flag())?;
    eprintln!(
        "vellum: {}x{} on {} via {}; exiting in {}",
        output.mode.size().0,
        output.mode.size().1,
        output.connector_name,
        device.name(),
        match args.seconds {
            0 => "never (send SIGTERM or switch VT)".to_owned(),
            s => format!("{s}s"),
        }
    );

    // Buffer `shown` is on screen; `queued` is flipping to it next.
    let mut shown = 0;
    let mut queued = 1;
    queue_frame(&mut display, &mut buffers[queued], start)?;
    let mut flips = 0u64;

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

        let (seat_ready, signal_ready, card_ready) = {
            let seat_fd = session.fd();
            let mut fds = [
                PollFd::new(&seat_fd, PollFlags::IN),
                PollFd::new(&signals, PollFlags::IN),
                PollFd::new(&card, PollFlags::IN),
            ];
            match rustix::event::poll(&mut fds, timeout.as_ref()) {
                Ok(_) => {}
                Err(rustix::io::Errno::INTR) => continue,
                Err(e) => return Err(e).context("poll"),
            }
            let ready = |i: usize| !fds[i].revents().is_empty();
            (ready(0), ready(1), ready(2))
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
        if card_ready && display.dispatch_events()? {
            // `queued` is on screen now, so the previously shown buffer is
            // no longer scanned out and can be rendered into.
            flips += 1;
            (shown, queued) = (queued, shown);
            queue_frame(&mut display, &mut buffers[queued], start)?;
        }
    }

    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "vellum: {flips} flips in {secs:.1}s ({:.1} Hz)",
        flips as f64 / secs
    );
    Ok(())
}

/// Enable atomic modesetting and choose the output.
fn prepare(card: &Card) -> Result<Output> {
    card.set_client_capability(ClientCapability::Atomic, true)
        .context("enabling DRM_CLIENT_CAP_ATOMIC (driver lacks atomic modesetting?)")?;
    kms::select_output(card)
}

/// The Vulkan device on the same GPU as `card`.
fn vulkan_device(instance: &Arc<vk::Instance>, card: &Card) -> Result<Arc<vk::Device>> {
    let rdev = rustix::fs::fstat(card)?.st_rdev;
    vk::Device::for_drm_node(instance, rustix::fs::major(rdev), rustix::fs::minor(rdev))
}

/// Modifiers both the primary plane can scan out and Vulkan can import for
/// rendering, in the plane's order of preference.
fn negotiate_modifiers(card: &Card, output: &Output, device: &vk::Device) -> Result<Vec<u64>> {
    let explicit = card
        .get_driver_capability(DriverCapability::AddFB2Modifiers)
        .is_ok_and(|v| v != 0);
    if !explicit {
        bail!("KMS driver doesn't support framebuffer modifiers");
    }
    let kms = kms::plane_modifiers(card, &output.plane_props, FORMAT)?;
    let vulkan = device.dmabuf_modifiers(FORMAT, USAGE)?;
    let common: Vec<u64> = kms
        .iter()
        .copied()
        .filter(|m| vulkan.iter().any(|v| v.modifier == *m))
        .collect();
    if common.is_empty() {
        bail!(
            "no format modifier is supported by both the plane ({} candidates) and Vulkan ({})",
            kms.len(),
            vulkan.len()
        );
    }
    Ok(common)
}

fn create_buffer<'a>(
    card: &'a Card,
    gbm: &'a gbm::Device<'_>,
    device: &Arc<vk::Device>,
    output: &Output,
    modifiers: &[u64],
) -> Result<Buffer<'a>> {
    let (w, h) = output.mode.size();
    let bo = gbm
        .create_bo(
            u32::from(w),
            u32::from(h),
            FORMAT,
            modifiers,
            gbm::BoFlags::SCANOUT | gbm::BoFlags::RENDERING,
        )
        .context("gbm_bo_create_with_modifiers2")?;
    let dmabuf = bo.export().context("exporting dma-buf")?;
    Ok(Buffer {
        image: vk::DmaBufImage::import(device, &dmabuf, USAGE)?,
        fb: Framebuffer::import(card, &dmabuf)?,
        cmd: vk::CommandBuffer::new(device)?,
        rendered: vk::Semaphore::new_exportable(device)?,
        bo,
    })
}

/// Record and submit this buffer's frame. Returns a sync_file that signals
/// when rendering is done, or `None` if it already is.
fn render(buf: &mut Buffer, start: Instant) -> Result<Option<OwnedFd>> {
    let color = animated_color(start.elapsed());
    let mut rec = buf.cmd.begin()?;
    rec.acquire_from_foreign(&buf.image);
    rec.clear(&buf.image, color);
    rec.release_to_foreign(&buf.image);
    rec.submit(Some(&buf.rendered))?;
    buf.rendered.export_sync_file()
}

/// Render into `buf` and queue a flip to it, fenced on the rendering.
fn queue_frame(display: &mut kms::Display, buf: &mut Buffer, start: Instant) -> Result<()> {
    let fence = render(buf, start)?;
    display.flip(&buf.fb, fence.as_ref())
}

/// Slowly cycling hue, so dropped or torn frames are visible.
fn animated_color(t: Duration) -> [f32; 4] {
    const PERIOD: f32 = 6.0;
    let h = (t.as_secs_f32() / PERIOD).fract() * 6.0;
    let (s, v) = (0.6, 0.5);
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m, 1.0]
}

/// Everything short of the modeset, without a seat. Useful from SSH or from
/// inside another graphical session to check device/plane/modifier selection
/// and that Vulkan can import and render into the scanout buffers.
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

    let instance = vk::Instance::new(cfg!(debug_assertions))?;
    let device = vulkan_device(&instance, card)?;
    println!("vulkan:    {}", device.name());

    let kms_mods = kms::plane_modifiers(card, &output.plane_props, FORMAT)?;
    let vk_mods = device.dmabuf_modifiers(FORMAT, USAGE)?;
    println!("modifiers (plane / vulkan import for color attachment):");
    for m in &kms_mods {
        let vulkan = vk_mods.iter().find(|v| v.modifier == *m);
        println!(
            "  {m:#018x} {:<12} {}",
            match vulkan {
                Some(v) => format!("{} plane(s)", v.plane_count),
                None => "-".to_owned(),
            },
            if vulkan.is_some() {
                "both"
            } else {
                "plane only"
            }
        );
    }

    let gbm = gbm::Device::new(card.as_fd()).context("gbm_create_device")?;
    let modifiers = negotiate_modifiers(card, &output, &device)?;
    let mut buf = create_buffer(card, &gbm, &device, &output, &modifiers)?;
    let start = Instant::now();
    render(&mut buf, start)?;
    buf.cmd.wait()?;
    println!(
        "buffer:    {}x{} {:?}, modifier {:#018x}, framebuffer {:?}",
        w,
        h,
        FORMAT,
        buf.bo.modifier(),
        buf.fb.handle
    );
    println!("rendered one frame in {:.2?}", start.elapsed());
    println!("probe ok (modeset skipped: needs DRM master via the seat)");
    Ok(())
}
