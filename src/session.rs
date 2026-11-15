//! Seat/session management through libseat (logind or seatd).
//!
//! libseat gives us DRM master and input device fds without root, and the
//! seat backend puts the VT into graphics mode with the keyboard off while we
//! hold the session (and puts it back when we release it).
//!
//! When the user switches to another VT the seat *disables* the session:
//! DRM master and the input devices are taken away, and the seat waits for
//! us to acknowledge ([`Session::ack_disable`]) once we've stopped using
//! them. Switching back *enables* it again.

use std::cell::{Cell, RefCell};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::path::Path;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use libseat::{Seat, SeatEvent};

/// A change in whether the session is ours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionEvent {
    /// Our VT is in the foreground: devices can be used again.
    Enabled,
    /// Switched away: stop using devices, then [`Session::ack_disable`].
    Disabled,
}

/// What the seat's events mean for us: whether the session is ours, and
/// whether a disable still needs acknowledging. Kept apart from libseat so
/// it can be tested.
#[derive(Debug, Default)]
struct State {
    active: bool,
    awaiting_ack: bool,
}

impl State {
    fn event(&mut self, event: SessionEvent) {
        match event {
            SessionEvent::Enabled => {
                self.active = true;
                self.awaiting_ack = false;
            }
            SessionEvent::Disabled => {
                self.active = false;
                self.awaiting_ack = true;
            }
        }
    }

    /// Whether an acknowledgement is due (and now given).
    fn ack(&mut self) -> bool {
        std::mem::take(&mut self.awaiting_ack)
    }
}

pub struct Session {
    /// Open devices, by the raw number of the fd handed out for each.
    devices: Vec<(RawFd, libseat::Device)>,
    seat: Seat,
    seat_fd: RawFd,
    /// Events from the seat's callback, not yet returned by `dispatch`.
    events: Rc<RefCell<Vec<SessionEvent>>>,
    state: State,
    /// Shared with [`crate::kms::Display`]: whether we still hold the
    /// display. Drops as soon as the seat says so, before the front end
    /// has handled the event.
    active: Rc<Cell<bool>>,
}

impl Session {
    /// Open the seat and block until the session is enabled (i.e. our VT is
    /// in the foreground).
    pub fn open() -> Result<Self> {
        let active = Rc::new(Cell::new(false));
        let events = Rc::new(RefCell::new(Vec::new()));
        let (flag, queue) = (active.clone(), events.clone());
        // Only record here: acknowledging a disable waits until the front
        // end has stopped using the devices.
        let mut seat = Seat::open(move |_seat, event| {
            let event = match event {
                SeatEvent::Enable => SessionEvent::Enabled,
                SeatEvent::Disable => SessionEvent::Disabled,
            };
            flag.set(event == SessionEvent::Enabled);
            queue.borrow_mut().push(event);
        })
        .map_err(|e| anyhow!("could not open seat via libseat: {e:?}"))?;

        let seat_fd = seat
            .get_fd()
            .map_err(|e| anyhow!("libseat_get_fd: {e:?}"))?
            .as_raw_fd();

        let mut session = Self {
            devices: Vec::new(),
            seat,
            seat_fd,
            events,
            state: State::default(),
            active,
        };
        while !session.state.active {
            session
                .seat
                .dispatch(-1)
                .map_err(|e| anyhow!("libseat_dispatch: {e:?}"))?;
            session.take_events();
        }
        Ok(session)
    }

    /// Open a device through the seat. The returned fd is a dup of the seat's
    /// fd (same open file description, so revocation still applies to it).
    /// The device stays open until [`Session::close_device`] or the session
    /// is dropped.
    pub fn open_device(&mut self, path: &Path) -> Result<OwnedFd> {
        self.open_device_io(path)
            .map_err(|e| anyhow!("libseat_open_device({}): {e}", path.display()))
    }

    /// [`Session::open_device`] with the errno kept, for libinput.
    pub fn open_device_io(&mut self, path: &Path) -> io::Result<OwnedFd> {
        let dev = self
            .seat
            .open_device(&path)
            .map_err(|e| io::Error::from_raw_os_error(e.0))?;
        let fd = match dev.as_fd().try_clone_to_owned() {
            Ok(fd) => fd,
            Err(e) => {
                let _ = self.seat.close_device(dev);
                return Err(e);
            }
        };
        // The number is unique while `fd` is open, which it is until
        // `close_device` takes it back.
        self.devices.push((fd.as_raw_fd(), dev));
        Ok(fd)
    }

    /// Close a device opened with [`Session::open_device`], given the fd it
    /// returned.
    pub fn close_device(&mut self, fd: OwnedFd) {
        let raw = fd.as_raw_fd();
        drop(fd);
        match self.devices.iter().position(|&(n, _)| n == raw) {
            Some(i) => {
                let (_, dev) = self.devices.swap_remove(i);
                if let Err(e) = self.seat.close_device(dev) {
                    tracing::warn!("libseat_close_device failed: {e:?}");
                }
            }
            None => tracing::warn!(fd = raw, "closing a device the seat didn't open"),
        }
    }

    /// The seat's name (`seat0`), which libinput needs to find its devices.
    pub fn seat_name(&mut self) -> String {
        self.seat.name().to_owned()
    }

    /// Process pending seat events, and return them in order. After a
    /// [`SessionEvent::Disabled`], stop using the devices and call
    /// [`Session::ack_disable`].
    pub fn dispatch(&mut self) -> Result<Vec<SessionEvent>> {
        self.seat
            .dispatch(0)
            .map_err(|e| anyhow!("libseat_dispatch: {e:?}"))?;
        Ok(self.take_events())
    }

    fn take_events(&mut self) -> Vec<SessionEvent> {
        let events = std::mem::take(&mut *self.events.borrow_mut());
        for &e in &events {
            self.state.event(e);
        }
        events
    }

    /// Tell the seat we've stopped using the devices, so the switch can
    /// go ahead. Does nothing unless a disable is waiting for it.
    pub fn ack_disable(&mut self) -> Result<()> {
        if self.state.ack() {
            self.seat
                .disable()
                .map_err(|e| anyhow!("libseat_disable_seat: {e:?}"))?;
        }
        Ok(())
    }

    /// Ask the seat to switch to session (VT) `n`.
    pub fn switch_session(&mut self, n: i32) -> Result<()> {
        self.seat
            .switch_session(n)
            .map_err(|e| anyhow!("libseat_switch_session({n}): {e:?}"))?;
        Ok(())
    }

    /// The VT in the foreground, from sysfs; `None` if it can't be told
    /// (no VTs, or a seat without them).
    pub fn foreground_vt() -> Option<i32> {
        let active = std::fs::read_to_string("/sys/class/tty/tty0/active").ok()?;
        parse_vt(&active)
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        // SAFETY: the seat fd stays open as long as `self.seat` does.
        unsafe { BorrowedFd::borrow_raw(self.seat_fd) }
    }

    /// Whether the session is ours, as of the events returned so far.
    pub fn active(&self) -> bool {
        self.state.active
    }

    /// Shared view of the active flag, for code that must stop touching KMS
    /// once the session is disabled.
    pub fn active_flag(&self) -> Rc<Cell<bool>> {
        self.active.clone()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for (_, dev) in self.devices.drain(..) {
            if let Err(e) = self.seat.close_device(dev) {
                tracing::warn!("libseat_close_device failed: {e:?}");
            }
        }
        // Dropping `seat` closes it, which releases the VT back to text mode.
    }
}

/// `tty3\n` → 3.
fn parse_vt(active: &str) -> Option<i32> {
    active.trim().strip_prefix("tty")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreground_vt_from_sysfs() {
        assert_eq!(parse_vt("tty3\n"), Some(3));
        assert_eq!(parse_vt("tty12"), Some(12));
        assert_eq!(parse_vt("ttyS0\n"), None);
        assert_eq!(parse_vt(""), None);
    }

    #[test]
    fn disable_waits_for_one_acknowledgement() {
        let mut s = State::default();
        assert!(!s.ack(), "nothing to acknowledge yet");
        s.event(SessionEvent::Enabled);
        assert!(s.active);
        s.event(SessionEvent::Disabled);
        assert!(!s.active);
        assert!(s.ack());
        assert!(!s.ack(), "only once");
        s.event(SessionEvent::Enabled);
        assert!(s.active && !s.ack());
    }

    #[test]
    fn enabling_drops_an_unsent_acknowledgement() {
        let mut s = State::default();
        s.event(SessionEvent::Disabled);
        s.event(SessionEvent::Enabled);
        assert!(!s.ack());
    }
}
