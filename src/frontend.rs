//! The front end: owns the seat, the display and the input devices, and
//! shows the shell running in the PTY session server.
//!
//! The display is taken over through libseat, then page-flip events drive
//! rendering into the scanout buffers (see [`crate::scanout`]). The previous
//! display state is restored on exit (normal return, error, panic unwind, or
//! a termination signal).
//!
//! The shell runs in the session server (see [`crate::server`]), started on
//! demand; the front end attaches, gets the PTY master, and feeds what the
//! shell prints to the terminal core ([`crate::term`]).
//!
//! Keys come from libinput through the seat, go through the xkb keymap
//! ([`crate::input`]) and the key encoder ([`crate::keys`]), and are
//! written to the PTY along with the terminal's replies.
//!
//! Frames are drawn only when the screen changed and no flip is in flight;
//! the flip-completion event frees the other buffer for the next frame.
//! While a program has synchronized output on (DECSET 2026), frames wait
//! until it's turned off, or [`SYNC_TIMEOUT`] passes.

use std::cell::RefCell;
use std::os::fd::{AsFd, OwnedFd};
use std::rc::Rc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rustix::io::Errno;

use crate::cli::Args;
use crate::event::Poller;
use crate::font::{Face, FontSet, fontconfig};
use crate::input::{Binding, Devices, Key, KeyRepeat, Keyboard, Scroll};
use crate::keys::{self, KeyEvent, KeyKind, KeyModes};
use crate::kms::{self, Card, Flip};
use crate::scanout::{
    self, create_buffer, negotiate_modifiers, prepare, queue_frame, render, vulkan_device,
};
use crate::session::{Session, SessionEvent};
use crate::signals::{self, SignalFd};
use crate::term::Terminal;
use crate::text::{self, TextRenderer};
use crate::{gbm, pty, server, vk};

const BUFFER_COUNT: usize = 2;

/// Longest a program's synchronized update can hold back drawing (what
/// other terminals use), in case it never ends it.
const SYNC_TIMEOUT: Duration = Duration::from_millis(150);

// Event loop tokens.
const SEAT: u64 = 0;
const SIGNALS: u64 = 1;
const CARD: u64 = 2;
const PTY: u64 = 3;
const INPUT: u64 = 4;
const REPEAT: u64 = 5;

pub fn run(args: Args) -> Result<()> {
    // Before any threads exist, so they all inherit the blocked mask.
    let signals = SignalFd::new(&[libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT])
        .context("setting up signalfd")?;
    let path = kms::find_card()?;
    tracing::info!(card = %path.display(), "using DRM device");

    if args.probe {
        let card = Card::open_direct(&path)?;
        return probe(&card, &args);
    }

    // Shared with libinput, which opens input devices through it.
    let session = Rc::new(RefCell::new(Session::open()?));
    let card = Card::new(session.borrow_mut().open_device(&path)?);
    let output = prepare(&card)?;
    let font = load_font(&args.font, output.dpi())?;

    let (w, h) = output.mode.size();
    let cell = font.cell();
    let (cols, rows) = text::grid_size(cell, w.into(), h.into());
    let size = pty::WinSize {
        cols,
        rows,
        width: (u32::from(cols) * cell.width).min(u16::MAX.into()) as u16,
        height: (u32::from(rows) * cell.height).min(u16::MAX.into()) as u16,
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
    let format = vk::vk_format(scanout::FORMAT).context("scanout format")?;
    let mut text = TextRenderer::new(&device, font, format, BUFFER_COUNT)?;
    let mut term = Terminal::new(cols, rows);
    term.set_cell_pixels(
        cell.width.min(u16::MAX.into()) as u16,
        cell.height.min(u16::MAX.into()) as u16,
    );

    // First frame: render, wait on the CPU, and modeset with it.
    let start = Instant::now();
    render(&mut buffers[0], |rec, image| {
        text.draw(rec, image, 0, &term)
    })?;
    buffers[0].cmd.wait()?;

    // Locals drop in reverse order: the display state is restored first (after
    // any pending flip lands), then GPU work is waited for and buffers, the
    // device, the card and finally the seat are released. This also holds
    // when unwinding from a panic.
    let active = session.borrow().active_flag();
    let mut display = kms::Display::take_over(&card, &output, &buffers[0].fb, active)?;
    let (w, h) = output.mode.size();
    tracing::info!(
        connector = %output.connector_name,
        mode = %format_args!("{w}x{h}@{}", output.mode.vrefresh()),
        gpu = device.name(),
        "display taken over; running {}",
        match args.seconds {
            0 => "until Ctrl+Alt+Backspace or SIGTERM".to_owned(),
            s => format!("for {s}s"),
        }
    );

    // Buffer `shown` is on screen; `next` is free to render into, or being
    // flipped to while `flip_pending`.
    let mut shown = 0;
    let mut next = 1;
    let mut flip_pending = false;
    let mut dirty = false;
    // When the current synchronized update started.
    let mut sync_since: Option<Instant> = None;
    let mut frames = 1u64;

    let mut keyboard = Keyboard::new(&args.keymap)?;
    let mut repeat = KeyRepeat::new().context("creating the key repeat timer")?;
    // Dropped first, closing the input devices while the seat is still ours.
    let mut devices = Devices::new(session.clone()).context("starting libinput")?;

    let mut poller = Poller::new().context("creating epoll instance")?;
    poller.add(session.borrow().fd(), SEAT)?;
    poller.add(&signals, SIGNALS)?;
    poller.add(&card, CARD)?;
    poller.add(&shell.master, PTY)?;
    poller.add(&devices, INPUT)?;
    poller.add(&repeat, REPEAT)?;
    let mut ready = Vec::new();
    let mut pty_buf = vec![0u8; 64 * 1024];
    // What goes to the shell (keys, and replies to its queries),
    // and whether the PTY is watched for room to write the rest.
    let mut to_shell = pty::Writer::default();
    let mut watching_writable = false;
    let mut shell_alive = true;

    let deadline = (args.seconds > 0).then(|| Instant::now() + Duration::from_secs(args.seconds));
    'events: loop {
        let now = Instant::now();
        if deadline.is_some_and(|d| d <= now) {
            break;
        }
        // Wake for the end of the run, and for a held-back frame.
        let sync_end = sync_since
            .map(|since| since + SYNC_TIMEOUT)
            .filter(|&end| dirty && end > now);
        let timeout = [deadline, sync_end]
            .into_iter()
            .flatten()
            .min()
            .map(|t| t.saturating_duration_since(now));
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
                    let events = session.borrow_mut().dispatch()?;
                    for event in events {
                        match event {
                            SessionEvent::Disabled => {
                                // Stop using the devices, then let the
                                // switch go ahead. The shell keeps running
                                // and its output keeps being read.
                                display.pause();
                                flip_pending = false;
                                session.borrow_mut().ack_disable()?;
                                tracing::info!("session disabled (switched to another VT)");
                            }
                            SessionEvent::Enabled => {
                                // Show the last frame again, then redraw:
                                // the screen may have changed while away.
                                display
                                    .resume(&output, &buffers[shown].fb)
                                    .context("taking the display back")?;
                                flip_pending = false;
                                dirty = true;
                                tracing::info!("session enabled (back on our VT)");
                            }
                        }
                    }
                }
                CARD => {
                    if display.dispatch_events()? {
                        // `next` is on screen now, so the previously shown
                        // buffer is no longer scanned out and can be
                        // rendered into.
                        (shown, next) = (next, shown);
                        flip_pending = false;
                        tracing::trace!(shown, "page flip completed");
                    }
                }
                PTY => match read_pty(&shell.master, &mut pty_buf)? {
                    Some(0) => {}
                    Some(n) => {
                        term.feed(&pty_buf[..n]);
                        let replies = term.take_replies();
                        if !replies.is_empty() {
                            tracing::trace!(target: "vellum::pty", "reply: {}", replies.escape_ascii());
                            to_shell
                                .send(&shell.master, &replies)
                                .context("writing to the PTY")?;
                        }
                        dirty = true;
                        if !term.synchronized() {
                            sync_since = None;
                        } else if sync_since.is_none() {
                            sync_since = Some(Instant::now());
                        }
                    }
                    None => {
                        tracing::info!("shell exited");
                        shell_alive = false;
                        break 'events;
                    }
                },
                INPUT => {
                    let mut keys = Vec::new();
                    devices
                        .dispatch(|key| keys.push(key))
                        .context("reading input events")?;
                    for Key { code, pressed } in keys {
                        let kind = if pressed {
                            repeat.press(code, keyboard.repeats(code))
                        } else {
                            repeat.release(code)
                        };
                        kind.context("setting the key repeat timer")?;
                        let kind = if pressed {
                            KeyKind::Press
                        } else {
                            KeyKind::Release
                        };
                        if let Some(event) = keyboard.key(code, kind) {
                            match on_key(&event, &mut term, &mut to_shell, &shell.master)? {
                                After::Exit => break 'events,
                                After::Redraw => dirty = true,
                                After::Nothing => {}
                            }
                        }
                    }
                }
                REPEAT => {
                    if let Some(code) = repeat.expired().context("reading the repeat timer")?
                        && let Some(event) = keyboard.key(code, KeyKind::Repeat)
                    {
                        match on_key(&event, &mut term, &mut to_shell, &shell.master)? {
                            After::Exit => break 'events,
                            After::Redraw => dirty = true,
                            After::Nothing => {}
                        }
                    }
                }
                _ => unreachable!("unknown event token {token}"),
            }
        }

        // Finish queued writes once there's room, and watch for room only
        // while something is queued (the PTY is nearly always writable).
        if ready.contains(&PTY) && to_shell.pending() {
            to_shell
                .flush(&shell.master)
                .context("writing to the PTY")?;
        }
        if to_shell.pending() != watching_writable {
            watching_writable = to_shell.pending();
            poller.watch_writable(&shell.master, PTY, watching_writable)?;
        }

        let held = sync_since.is_some_and(|since| since.elapsed() < SYNC_TIMEOUT);
        if dirty && !flip_pending && !held && !display.paused() {
            let slot = next;
            let flip = queue_frame(&mut display, &mut buffers[slot], |rec, image| {
                text.draw(rec, image, slot, &term)
            })?;
            // Lost: the seat took DRM master for a switch and its event is
            // on the way. The frame is drawn again after resuming.
            if flip == Flip::Queued {
                flip_pending = true;
                dirty = false;
                frames += 1;
            }
        }
    }

    tracing::info!(
        frames,
        seconds = %format_args!("{:.1}", start.elapsed().as_secs_f64()),
        "shutting down"
    );
    // Only an orderly exit ends the shell; after a crash (error return,
    // panic) it keeps running in the session server for the next attach.
    if shell_alive && let Err(e) = shell.close() {
        tracing::warn!("asking the session server to close the shell: {e}");
    }
    Ok(())
}

/// What the event loop does after a key.
enum After {
    Nothing,
    /// The screen changed (the view moved).
    Redraw,
    Exit,
}

/// Handle one key: vellum's own bindings, else encode it and queue it for
/// the shell.
fn on_key(
    event: &KeyEvent,
    term: &mut Terminal,
    to_shell: &mut pty::Writer,
    master: &OwnedFd,
) -> Result<After> {
    let press = event.kind != KeyKind::Release;
    match crate::input::binding(event) {
        Some(Binding::Exit) if event.kind == KeyKind::Press => {
            tracing::info!("exiting on Ctrl+Alt+Backspace");
            return Ok(After::Exit);
        }
        Some(Binding::SwitchVt(n)) if event.kind == KeyKind::Press => {
            tracing::info!(vt = n, "Ctrl+Alt+F{n}: VT switching is step 7; ignored");
            return Ok(After::Nothing);
        }
        // The alternate screen has no scrollback: the program gets these.
        Some(Binding::Scroll(scroll)) if !term.alternate_screen() => {
            if !press {
                return Ok(After::Nothing);
            }
            let half = (term.rows() / 2).max(1) as isize;
            let before = term.view_offset();
            match scroll {
                Scroll::PageUp => term.scroll_view(half),
                Scroll::PageDown => term.scroll_view(-half),
                Scroll::Top => term.scroll_view(isize::MAX),
                Scroll::Bottom => term.reset_view(),
            }
            return Ok(if term.view_offset() == before {
                After::Nothing
            } else {
                After::Redraw
            });
        }
        Some(Binding::Scroll(_)) => {}
        Some(_) => return Ok(After::Nothing),
        None => {}
    }
    let Some(bytes) = keys::encode(event, &KeyModes::of(term)) else {
        return Ok(After::Nothing);
    };
    tracing::trace!(target: "vellum::pty", "key {:?}: {}", event.sym, bytes.escape_ascii());
    to_shell
        .send(master, &bytes)
        .context("writing to the PTY")?;
    // Typing goes back to the live screen. Not for releases or modifier
    // keys (the kitty protocol reports those), or Shift+PgUp would snap
    // back as soon as Shift was let go.
    if press && !event.sym.is_modifier_key() && term.view_offset() > 0 {
        term.reset_view();
        return Ok(After::Redraw);
    }
    Ok(After::Nothing)
}

/// Read what the shell wrote into `buf`: `Some(bytes read)` (0 if nothing
/// was ready), or `None` once the shell is gone.
fn read_pty(master: &OwnedFd, buf: &mut [u8]) -> Result<Option<usize>> {
    match rustix::io::read(master, &mut *buf) {
        // EIO: every slave fd is closed, i.e. the shell (and anything else
        // using the terminal) has exited.
        Ok(0) | Err(Errno::IO) => Ok(None),
        Ok(n) => {
            tracing::trace!(target: "vellum::pty", "{}", buf[..n].escape_ascii());
            Ok(Some(n))
        }
        Err(Errno::AGAIN | Errno::INTR) => Ok(Some(0)),
        Err(e) => Err(e).context("reading the PTY"),
    }
}

/// Find and load the fonts for `pattern` at the monitor's DPI.
fn load_font(pattern: &str, dpi: f64) -> Result<FontSet> {
    let fonts = FontSet::load(&fontconfig::Config::new()?, pattern, dpi)?;
    let regular = fonts.regular();
    let cell = fonts.cell();
    tracing::info!(
        pattern,
        pixel_size = %format_args!("{:.1}", regular.size()),
        dpi = %format_args!("{dpi:.0}"),
        cell = %format_args!("{}x{}", cell.width, cell.height),
        rendering = ?regular.rendering(),
        "loaded fonts"
    );
    for face in Face::ALL {
        let font = fonts.face(face);
        let s = font.synthetic();
        let synthetic = match (s.embolden > 0.0, s.skew != 0.0) {
            (false, false) => "no",
            (true, false) => "bold",
            (false, true) => "italic",
            (true, true) => "bold italic",
        };
        tracing::info!(
            face = face.name(),
            font = font.name(),
            file = %font.file().display(),
            synthetic,
            "font face"
        );
    }
    Ok(fonts)
}

/// Everything short of the modeset, without a seat. Useful from SSH or from
/// inside another graphical session to check device/plane/modifier selection
/// and that Vulkan can import and render into the scanout buffers.
///
/// The interesting details (output, GPU, modifiers, buffer layout) are logged
/// by the shared setup functions; run with `VELLUM_LOG=debug` to see them all.
fn probe(card: &Card, args: &Args) -> Result<()> {
    let output = prepare(card)?;
    let font = load_font(&args.font, output.dpi())?;
    let (w, h) = output.mode.size();
    let (cols, rows) = text::grid_size(font.cell(), w.into(), h.into());
    let instance = vk::Instance::new(cfg!(debug_assertions))?;
    let device = vulkan_device(&instance, card)?;
    let gbm = gbm::Device::new(card.as_fd()).context("gbm_create_device")?;
    let modifiers = negotiate_modifiers(card, &output, &device)?;
    let mut buf = create_buffer(card, &gbm, &device, &output, &modifiers)?;
    let format = vk::vk_format(scanout::FORMAT).context("scanout format")?;
    let mut text = TextRenderer::new(&device, font, format, 1)?;
    let mut term = Terminal::new(cols, rows);
    term.feed(
        b"\x1b[1mvellum --probe\x1b[m\r\n\
          The quick brown fox jumps over the lazy dog. 0123456789 {}[]()<>\r\n\
          \x1b(0lqqqk\r\nx   x\r\nmqqqj\x1b(B",
    );
    let start = Instant::now();
    render(&mut buf, |rec, image| text.draw(rec, image, 0, &term))?;
    buf.cmd.wait()?;
    tracing::info!(
        render_time = ?start.elapsed(),
        framebuffer = ?buf.fb.handle,
        grid = %format_args!("{cols}x{rows}"),
        "probe ok: rendered one frame of text (modeset skipped: needs DRM master via the seat)"
    );
    Ok(())
}
