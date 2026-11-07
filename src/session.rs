//! Seat/session management through libseat (logind or seatd).
//!
//! libseat gives us DRM master and input device fds without root, and the
//! seat backend puts the VT into graphics mode with the keyboard off while we
//! hold the session (and puts it back when we release it).

use std::cell::Cell;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow};
use libseat::{Seat, SeatEvent};

pub struct Session {
    devices: Vec<libseat::Device>,
    seat: Seat,
    seat_fd: RawFd,
    active: Rc<Cell<bool>>,
}

impl Session {
    /// Open the seat and block until the session is enabled (i.e. our VT is
    /// in the foreground).
    pub fn open() -> Result<Self> {
        let active = Rc::new(Cell::new(false));
        let flag = active.clone();
        let mut seat = Seat::open(move |seat, event| match event {
            SeatEvent::Enable => flag.set(true),
            SeatEvent::Disable => {
                // We must stop touching devices and acknowledge. Callers see
                // the flag drop and must not issue KMS commits afterwards.
                flag.set(false);
                if let Err(e) = seat.disable() {
                    tracing::error!("libseat_disable_seat failed: {e:?}");
                }
            }
        })
        .map_err(|e| anyhow!("could not open seat via libseat: {e:?}"))?;

        let seat_fd = seat
            .get_fd()
            .map_err(|e| anyhow!("libseat_get_fd: {e:?}"))?
            .as_raw_fd();

        while !active.get() {
            seat.dispatch(-1)
                .map_err(|e| anyhow!("libseat_dispatch: {e:?}"))?;
        }

        Ok(Self {
            devices: Vec::new(),
            seat,
            seat_fd,
            active,
        })
    }

    /// Open a device through the seat. The returned fd is a dup of the seat's
    /// fd (same open file description, so revocation still applies to it).
    pub fn open_device(&mut self, path: &Path) -> Result<OwnedFd> {
        let dev = self
            .seat
            .open_device(&path)
            .map_err(|e| anyhow!("libseat_open_device({}): {e:?}", path.display()))?;
        let fd = dev
            .as_fd()
            .try_clone_to_owned()
            .context("dup of seat device fd")?;
        self.devices.push(dev);
        Ok(fd)
    }

    /// Process pending seat events (enable/disable).
    pub fn dispatch(&mut self) -> Result<()> {
        self.seat
            .dispatch(0)
            .map_err(|e| anyhow!("libseat_dispatch: {e:?}"))?;
        Ok(())
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        // SAFETY: the seat fd stays open as long as `self.seat` does.
        unsafe { BorrowedFd::borrow_raw(self.seat_fd) }
    }

    pub fn active(&self) -> bool {
        self.active.get()
    }

    /// Shared view of the active flag, for code that must stop touching KMS
    /// once the session is disabled.
    pub fn active_flag(&self) -> Rc<Cell<bool>> {
        self.active.clone()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for dev in self.devices.drain(..) {
            if let Err(e) = self.seat.close_device(dev) {
                tracing::warn!("libseat_close_device failed: {e:?}");
            }
        }
        // Dropping `seat` closes it, which releases the VT back to text mode.
    }
}
