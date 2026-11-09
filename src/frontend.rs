//! The front end: owns the seat, the display and the input devices, and
//! shows the shell running in the PTY session server.
//!
//! The display is taken over through libseat, then page-flip events drive
//! rendering into the scanout buffers (see [`crate::scanout`]). The previous
//! display state is restored on exit (normal return, error, panic unwind, or
//! a termination signal).
//!
//! The shell runs in the session server (see [`crate::server`]), started on
//! demand; the front end attaches and gets the PTY master. Until text
//! rendering exists (step 4) the shell's output only goes to the log.

use std::os::fd::{AsFd, OwnedFd};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rustix::io::Errno;

use crate::cli::Args;
use crate::event::Poller;
use crate::kms::{self, Card};
use crate::scanout::{
    create_buffer, negotiate_modifiers, prepare, queue_frame, render, vulkan_device,
};
use crate::session::Session;
use crate::signals::{self, SignalFd};
use crate::{gbm, pty, server, vk};

const BUFFER_COUNT: usize = 2;
/// Cell size used for the PTY's size until there is a font (step 4).
const PLACEHOLDER_CELL: (u16, u16) = (8, 16);

// Event loop tokens.
const SEAT: u64 = 0;
const SIGNALS: u64 = 1;
const CARD: u64 = 2;
const PTY: u64 = 3;

pub fn run(args: Args) -> Result<()> {
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

    let (w, h) = output.mode.size();
    let size = pty::WinSize {
        cols: w / PLACEHOLDER_CELL.0,
        rows: h / PLACEHOLDER_CELL.1,
        width: w,
        height: h,
    };
    let socket = server::socket_path()?;
    let shell = server::attach(&socket, size)?;
    rustix::fs::fcntl_setfl(&shell.master, rustix::fs::OFlags::NONBLOCK)
        .context("making the PTY non-blocking")?;
    // Reattaching to a running shell: make sure it has our size.
    pty::set_size(&shell.master, size).context("setting the PTY size")?;
    tracing::info!(
        pid = shell.pid,
        new = shell.spawned,
        size = %format_args!("{}x{}", size.cols, size.rows),
        "attached to shell"
    );

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

    let mut poller = Poller::new().context("creating epoll instance")?;
    poller.add(session.fd(), SEAT)?;
    poller.add(&signals, SIGNALS)?;
    poller.add(&card, CARD)?;
    poller.add(&shell.master, PTY)?;
    let mut ready = Vec::new();
    let mut pty_buf = vec![0u8; 64 * 1024];
    let mut shell_alive = true;

    let deadline = (args.seconds > 0).then(|| Instant::now() + Duration::from_secs(args.seconds));
    'events: loop {
        let timeout = match deadline {
            Some(d) => {
                let left = d.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                Some(left)
            }
            None => None,
        };
        poller.wait(timeout, &mut ready).context("epoll_wait")?;

        for &token in &ready {
            match token {
                SIGNALS => {
                    if let Some(sig) = signals.read()? {
                        tracing::info!(signal = signals::name(sig), "exiting on signal");
                        break 'events;
                    }
                }
                SEAT => {
                    session.dispatch()?;
                    if !session.active() {
                        // VT switching proper is roadmap step 7; for now just leave.
                        tracing::info!("session disabled (VT switched away), exiting");
                        break 'events;
                    }
                }
                CARD => {
                    if display.dispatch_events()? {
                        // `queued` is on screen now, so the previously shown
                        // buffer is no longer scanned out and can be
                        // rendered into.
                        flips += 1;
                        (shown, queued) = (queued, shown);
                        tracing::trace!(flips, shown, "page flip completed");
                        queue_frame(&mut display, &mut buffers[queued], start)?;
                    }
                }
                PTY => {
                    if !read_pty(&shell.master, &mut pty_buf)? {
                        tracing::info!("shell exited");
                        shell_alive = false;
                        break 'events;
                    }
                }
                _ => unreachable!("unknown event token {token}"),
            }
        }
    }

    let secs = start.elapsed().as_secs_f64();
    tracing::info!(
        flips,
        seconds = %format_args!("{secs:.1}"),
        rate_hz = %format_args!("{:.1}", flips as f64 / secs),
        "shutting down"
    );
    // Only an orderly exit ends the shell; after a crash (error return,
    // panic) it keeps running in the session server for the next attach.
    if shell_alive && let Err(e) = shell.close() {
        tracing::warn!("asking the session server to close the shell: {e}");
    }
    Ok(())
}

/// Read what the shell wrote. Until text rendering (step 4) it only goes to
/// the log. Returns false once the shell is gone.
fn read_pty(master: &OwnedFd, buf: &mut [u8]) -> Result<bool> {
    match rustix::io::read(master, &mut *buf) {
        // EIO: every slave fd is closed, i.e. the shell (and anything else
        // using the terminal) has exited.
        Ok(0) | Err(Errno::IO) => Ok(false),
        Ok(n) => {
            tracing::debug!(bytes = n, "shell output");
            tracing::trace!(target: "vellum::pty", "{}", buf[..n].escape_ascii());
            Ok(true)
        }
        Err(Errno::AGAIN | Errno::INTR) => Ok(true),
        Err(e) => Err(e).context("reading the PTY"),
    }
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
