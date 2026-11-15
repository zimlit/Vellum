//! A small epoll wrapper: register fds under a token, wait, get back the
//! tokens of the ones that are ready.
//!
//! Registrations are level-triggered and for input: an fd is reported on
//! every wait while it is readable, hung up or in error, so handlers can do
//! one read per wakeup without draining. An fd can also be watched for
//! being writable while there's output queued for it
//! ([`Poller::watch_writable`]). epoll drops an fd by itself once
//! its last reference is closed.

use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::time::Duration;

use rustix::event::{Timespec, epoll};
use rustix::io::Errno;

pub struct Poller {
    epoll: OwnedFd,
    events: Vec<epoll::Event>,
}

impl Poller {
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            epoll: epoll::create(epoll::CreateFlags::CLOEXEC)?,
            events: Vec::with_capacity(16),
        })
    }

    pub fn add(&self, fd: impl AsFd, token: u64) -> io::Result<()> {
        epoll::add(
            &self.epoll,
            fd,
            epoll::EventData::new_u64(token),
            epoll::EventFlags::IN,
        )?;
        Ok(())
    }

    /// Change whether `fd` (already added) is also watched for being
    /// writable. Its token is reported for either.
    pub fn watch_writable(&self, fd: impl AsFd, token: u64, writable: bool) -> io::Result<()> {
        let mut flags = epoll::EventFlags::IN;
        if writable {
            flags |= epoll::EventFlags::OUT;
        }
        epoll::modify(&self.epoll, fd, epoll::EventData::new_u64(token), flags)?;
        Ok(())
    }

    /// Wait until at least one fd is ready or `timeout` passes (`None`:
    /// forever), and put the ready tokens in `ready`. An interrupted wait
    /// counts as a timeout.
    pub fn wait(&mut self, timeout: Option<Duration>, ready: &mut Vec<u64>) -> io::Result<()> {
        ready.clear();
        self.events.clear();
        let timeout = timeout.map(|t| Timespec {
            tv_sec: t.as_secs() as _,
            tv_nsec: t.subsec_nanos() as _,
        });
        let buf = rustix::buffer::spare_capacity(&mut self.events);
        match epoll::wait(&self.epoll, buf, timeout.as_ref()) {
            Ok(_) | Err(Errno::INTR) => {}
            Err(e) => return Err(e.into()),
        }
        ready.extend(self.events.iter().map(|e| e.data.u64()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    use super::*;

    #[test]
    fn reports_only_ready_fds_until_drained() {
        let (mut a_tx, a_rx) = UnixStream::pair().unwrap();
        let (_b_tx, b_rx) = UnixStream::pair().unwrap();
        let mut poller = Poller::new().unwrap();
        poller.add(&a_rx, 1).unwrap();
        poller.add(&b_rx, 2).unwrap();
        let mut ready = vec![99];

        poller.wait(Some(Duration::ZERO), &mut ready).unwrap();
        assert!(ready.is_empty(), "nothing written yet");

        a_tx.write_all(b"x").unwrap();
        poller
            .wait(Some(Duration::from_secs(5)), &mut ready)
            .unwrap();
        assert_eq!(ready, [1]);
        // Level-triggered: still ready while unread.
        poller.wait(Some(Duration::ZERO), &mut ready).unwrap();
        assert_eq!(ready, [1]);

        let mut byte = [0];
        rustix::io::read(&a_rx, &mut byte).unwrap();
        poller.wait(Some(Duration::ZERO), &mut ready).unwrap();
        assert!(ready.is_empty());
    }

    #[test]
    fn writable_only_while_watched() {
        let (a, _b) = UnixStream::pair().unwrap();
        let mut poller = Poller::new().unwrap();
        poller.add(&a, 3).unwrap();
        let mut ready = Vec::new();
        poller.wait(Some(Duration::ZERO), &mut ready).unwrap();
        assert!(ready.is_empty(), "writable, but not watched for it");
        poller.watch_writable(&a, 3, true).unwrap();
        poller.wait(Some(Duration::ZERO), &mut ready).unwrap();
        assert_eq!(ready, [3]);
        poller.watch_writable(&a, 3, false).unwrap();
        poller.wait(Some(Duration::ZERO), &mut ready).unwrap();
        assert!(ready.is_empty());
    }

    #[test]
    fn hangup_counts_as_ready() {
        let (tx, rx) = UnixStream::pair().unwrap();
        let mut poller = Poller::new().unwrap();
        poller.add(&rx, 7).unwrap();
        drop(tx);
        let mut ready = Vec::new();
        poller
            .wait(Some(Duration::from_secs(5)), &mut ready)
            .unwrap();
        assert_eq!(ready, [7]);
    }

    #[test]
    fn closed_fd_is_dropped_from_the_set() {
        let (tx, rx) = UnixStream::pair().unwrap();
        let mut poller = Poller::new().unwrap();
        poller.add(&rx, 1).unwrap();
        drop(rx);
        drop(tx);
        let mut ready = Vec::new();
        poller.wait(Some(Duration::ZERO), &mut ready).unwrap();
        assert!(ready.is_empty());
    }
}
