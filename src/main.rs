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
mod logging;
mod session;
mod signals;
#[cfg(test)]
mod testutil;
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
        Self::parse_from(std::env::args().skip(1))
    }

    /// Parse arguments (without the program name).
    fn parse_from(it: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = Args {
            probe: false,
            seconds: 10,
        };
        let mut it = it.into_iter();
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
    _bo: gbm::Bo<'a>,
}

fn main() -> ExitCode {
    let mut log = logging::init();
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        log_file = ?log.file,
        "starting"
    );
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e:#}");
            if let Some(console) = &mut log.console {
                // Started from a VT, which is back in text mode now that the
                // seat is released: say why we failed where the user is
                // looking. (Best effort; nothing to do if this fails.)
                use std::io::Write;
                let _ = writeln!(console, "vellum: {e:#}");
                if let Some(file) = &log.file {
                    let _ = writeln!(console, "vellum: log: {}", file.display());
                }
            }
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    // Before any threads exist, so they all inherit the blocked mask.
    let signals = SignalFd::new(&[libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT])
        .context("setting up signalfd")?;
    let path = kms::find_card()?;
    tracing::info!(card = %path.display(), "using DRM device");

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
    let (w, h) = output.mode.size();
    tracing::info!(
        connector = %output.connector_name,
        mode = %format_args!("{w}x{h}@{}", output.mode.vrefresh()),
        gpu = device.name(),
        "display taken over; running {}",
        match args.seconds {
            0 => "until SIGTERM or VT switch".to_owned(),
            s => format!("for {s}s"),
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
            tracing::info!(signal = signals::name(sig), "exiting on signal");
            break;
        }
        if seat_ready {
            session.dispatch()?;
            if !session.active() {
                // VT switching proper is roadmap step 7; for now just leave.
                tracing::info!("session disabled (VT switched away), exiting");
                break;
            }
        }
        if card_ready && display.dispatch_events()? {
            // `queued` is on screen now, so the previously shown buffer is
            // no longer scanned out and can be rendered into.
            flips += 1;
            (shown, queued) = (queued, shown);
            tracing::trace!(flips, shown, "page flip completed");
            queue_frame(&mut display, &mut buffers[queued], start)?;
        }
    }

    let secs = start.elapsed().as_secs_f64();
    tracing::info!(
        flips,
        seconds = %format_args!("{secs:.1}"),
        rate_hz = %format_args!("{:.1}", flips as f64 / secs),
        "shutting down"
    );
    Ok(())
}

/// Enable atomic modesetting and choose the output.
fn prepare(card: &Card) -> Result<Output> {
    card.set_client_capability(ClientCapability::Atomic, true)
        .context("enabling DRM_CLIENT_CAP_ATOMIC (driver lacks atomic modesetting?)")?;
    let output = kms::select_output(card)?;
    let (w, h) = output.mode.size();
    tracing::info!(
        connector = %output.connector_name,
        mode = %format_args!("{w}x{h}@{}", output.mode.vrefresh()),
        crtc = ?output.crtc,
        plane = ?output.plane,
        "selected output"
    );
    Ok(output)
}

/// The Vulkan device on the same GPU as `card`.
fn vulkan_device(instance: &Arc<vk::Instance>, card: &Card) -> Result<Arc<vk::Device>> {
    let rdev = rustix::fs::fstat(card)?.st_rdev;
    let (major, minor) = (rustix::fs::major(rdev), rustix::fs::minor(rdev));
    let device = vk::Device::for_drm_node(instance, major, minor)?;
    tracing::info!(gpu = device.name(), drm_node = %format_args!("{major}:{minor}"), "created Vulkan device");
    Ok(device)
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
    for &m in &kms {
        let vk = vulkan.iter().find(|v| v.modifier == m);
        tracing::debug!(
            modifier = %format_args!("{m:#018x}"),
            vulkan_planes = vk.map(|v| v.plane_count),
            "plane modifier{}",
            if vk.is_some() { "" } else { " (not importable by Vulkan)" }
        );
    }
    let vulkan_mods: Vec<u64> = vulkan.iter().map(|v| v.modifier).collect();
    let common = intersect_modifiers(&kms, &vulkan_mods);
    tracing::info!(
        plane = kms.len(),
        vulkan = vulkan.len(),
        common = common.len(),
        "negotiated format modifiers"
    );
    if common.is_empty() {
        bail!(
            "no format modifier is supported by both the plane ({} candidates) and Vulkan ({})",
            kms.len(),
            vulkan.len()
        );
    }
    Ok(common)
}

/// Modifiers in both lists, in `preferred`'s order.
fn intersect_modifiers(preferred: &[u64], other: &[u64]) -> Vec<u64> {
    preferred
        .iter()
        .copied()
        .filter(|m| other.contains(m))
        .collect()
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
    tracing::debug!(
        size = %format_args!("{}x{}", dmabuf.width, dmabuf.height),
        format = ?dmabuf.format,
        modifier = %format_args!("{:#018x}", u64::from(dmabuf.modifier)),
        planes = ?dmabuf.planes.iter().map(|p| (p.offset, p.stride)).collect::<Vec<_>>(),
        "allocated scanout buffer (planes as (offset, stride))"
    );
    Ok(Buffer {
        image: vk::DmaBufImage::import(device, &dmabuf, USAGE)?,
        fb: Framebuffer::import(card, &dmabuf)?,
        cmd: vk::CommandBuffer::new(device)?,
        rendered: vk::Semaphore::new_exportable(device)?,
        _bo: bo,
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
///
/// The interesting details (output, GPU, modifiers, buffer layout) are logged
/// by the shared setup functions; run with `VELLUM_LOG=debug` to see them all.
fn probe(card: &Card) -> Result<()> {
    let output = prepare(card)?;
    let instance = vk::Instance::new(cfg!(debug_assertions))?;
    let device = vulkan_device(&instance, card)?;
    let gbm = gbm::Device::new(card.as_fd()).context("gbm_create_device")?;
    let modifiers = negotiate_modifiers(card, &output, &device)?;
    let mut buf = create_buffer(card, &gbm, &device, &output, &modifiers)?;
    let start = Instant::now();
    render(&mut buf, start)?;
    buf.cmd.wait()?;
    tracing::info!(
        render_time = ?start.elapsed(),
        framebuffer = ?buf.fb.handle,
        "probe ok: rendered one frame (modeset skipped: needs DRM master via the seat)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil;

    fn args(list: &[&str]) -> Result<Args> {
        Args::parse_from(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn args_defaults() {
        let a = args(&[]).unwrap();
        assert!(!a.probe);
        assert_eq!(a.seconds, 10);
    }

    #[test]
    fn args_flags() {
        let a = args(&["--probe", "--seconds", "0"]).unwrap();
        assert!(a.probe);
        assert_eq!(a.seconds, 0);
    }

    #[test]
    fn args_errors() {
        assert!(args(&["--seconds"]).is_err(), "missing value");
        assert!(args(&["--seconds", "ten"]).is_err(), "not a number");
        assert!(args(&["--seconds", "-1"]).is_err(), "negative");
        assert!(args(&["--bogus"]).is_err(), "unknown flag");
    }

    #[test]
    fn intersect_modifiers_keeps_preferred_order() {
        assert_eq!(intersect_modifiers(&[5, 3, 1, 0], &[0, 1, 5, 9]), [5, 1, 0]);
        assert!(intersect_modifiers(&[1, 2], &[3, 4]).is_empty());
        assert!(intersect_modifiers(&[], &[1]).is_empty());
    }

    #[test]
    fn animated_color_is_valid_opaque_rgb() {
        for ms in (0..12_000).step_by(37) {
            let c = animated_color(Duration::from_millis(ms));
            assert!(
                c[..3].iter().all(|v| (0.0..=1.0).contains(v)),
                "{c:?} at {ms}ms"
            );
            assert_eq!(c[3], 1.0);
        }
    }

    #[test]
    fn animated_color_starts_red_and_repeats() {
        let at = |s: f32| animated_color(Duration::from_secs_f32(s));
        // Hue 0: red is the dominant channel, green and blue equal.
        let c = at(0.0);
        assert!(c[0] > c[1] && c[1] == c[2]);
        // 6 s period.
        for (a, b) in at(1.3).iter().zip(at(7.3)) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    /// The real setup path minus the modeset: output selection, modifier
    /// negotiation, GBM allocation, KMS + Vulkan import, and rendering with
    /// an exported fence that actually signals.
    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn scanout_buffer_renders_and_signals_fence() {
        let card = testutil::card();
        let output = prepare(&card).unwrap();
        let instance = vk::Instance::new(cfg!(debug_assertions)).unwrap();
        let device = vulkan_device(&instance, &card).unwrap();
        let gbm = gbm::Device::new(card.as_fd()).unwrap();
        let modifiers = negotiate_modifiers(&card, &output, &device).unwrap();
        assert!(!modifiers.is_empty());
        let mut buf = create_buffer(&card, &gbm, &device, &output, &modifiers).unwrap();
        assert!(modifiers.contains(&buf._bo.modifier()));

        let start = Instant::now();
        for _ in 0..3 {
            if let Some(fence) = render(&mut buf, start).unwrap() {
                let mut fds = [PollFd::new(&fence, PollFlags::IN)];
                let timeout = Timespec {
                    tv_sec: 5,
                    tv_nsec: 0,
                };
                assert_eq!(rustix::event::poll(&mut fds, Some(&timeout)).unwrap(), 1);
            }
        }
        buf.cmd.wait().unwrap();
    }
}
