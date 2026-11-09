//! Termination signals delivered through a signalfd, so they arrive as an
//! ordinary readable fd in the event loop and the normal (restoring) exit path
//! runs instead of the process dying mid-commit.

use std::ffi::c_int;
use std::io;
use std::mem::{self, MaybeUninit};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr;

pub struct SignalFd(OwnedFd);

impl SignalFd {
    /// Block `signals` for this thread and route them to a new signalfd.
    ///
    /// Call before spawning any threads: the mask is inherited, and a thread
    /// with the signal unblocked would receive it instead. Child processes
    /// (shells, later) inherit the mask too and must unblock it after fork.
    pub fn new(signals: &[c_int]) -> io::Result<Self> {
        // SAFETY: plain libc calls on a locally owned sigset.
        unsafe {
            let mut set = MaybeUninit::<libc::sigset_t>::uninit();
            libc::sigemptyset(set.as_mut_ptr());
            let mut set = set.assume_init();
            for &sig in signals {
                libc::sigaddset(&mut set, sig);
            }
            let err = libc::pthread_sigmask(libc::SIG_BLOCK, &set, ptr::null_mut());
            if err != 0 {
                return Err(io::Error::from_raw_os_error(err));
            }
            let fd = libc::signalfd(-1, &set, libc::SFD_CLOEXEC | libc::SFD_NONBLOCK);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self(OwnedFd::from_raw_fd(fd)))
        }
    }

    /// Read one pending signal, if any.
    pub fn read(&self) -> io::Result<Option<c_int>> {
        let mut info = MaybeUninit::<libc::signalfd_siginfo>::uninit();
        let size = mem::size_of::<libc::signalfd_siginfo>();
        // SAFETY: buffer is exactly one signalfd_siginfo.
        let n = unsafe { libc::read(self.0.as_raw_fd(), info.as_mut_ptr().cast(), size) };
        if n < 0 {
            let err = io::Error::last_os_error();
            return match err.kind() {
                io::ErrorKind::WouldBlock => Ok(None),
                _ => Err(err),
            };
        }
        if n as usize != size {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short signalfd read",
            ));
        }
        // SAFETY: the kernel filled the whole struct.
        Ok(Some(unsafe { info.assume_init() }.ssi_signo as c_int))
    }
}

impl AsFd for SignalFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// Unblock every signal for the calling thread.
///
/// For child processes between fork and exec: the blocked mask survives
/// exec, so without this a shell started by vellum would never see the
/// signals blocked for [`SignalFd`] (^C included). Only async-signal-safe
/// calls.
pub fn unblock_all() -> io::Result<()> {
    // SAFETY: plain libc calls on a locally owned sigset.
    unsafe {
        let mut set = MaybeUninit::<libc::sigset_t>::uninit();
        libc::sigemptyset(set.as_mut_ptr());
        if libc::sigprocmask(libc::SIG_SETMASK, set.as_ptr(), ptr::null_mut()) < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

pub fn name(sig: c_int) -> &'static str {
    match sig {
        libc::SIGINT => "SIGINT",
        libc::SIGTERM => "SIGTERM",
        libc::SIGHUP => "SIGHUP",
        libc::SIGQUIT => "SIGQUIT",
        _ => "signal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Signal masks are per thread and each test runs on its own thread, so
    // blocking here doesn't leak into other tests. Each test uses its own
    // signal anyway, to stay independent of scheduling.

    #[test]
    fn read_without_pending_signal_returns_none() {
        let fd = SignalFd::new(&[libc::SIGUSR1]).unwrap();
        assert_eq!(fd.read().unwrap(), None);
    }

    #[test]
    fn raised_signal_is_read_from_the_fd() {
        let fd = SignalFd::new(&[libc::SIGUSR2]).unwrap();
        // raise() targets the calling thread, where the signal is blocked,
        // so it stays pending for the signalfd instead of killing us.
        assert_eq!(unsafe { libc::raise(libc::SIGUSR2) }, 0);
        assert_eq!(fd.read().unwrap(), Some(libc::SIGUSR2));
        assert_eq!(fd.read().unwrap(), None, "consumed");
    }

    #[test]
    fn names_termination_signals() {
        assert_eq!(name(libc::SIGINT), "SIGINT");
        assert_eq!(name(libc::SIGTERM), "SIGTERM");
        assert_eq!(name(libc::SIGHUP), "SIGHUP");
        assert_eq!(name(libc::SIGQUIT), "SIGQUIT");
        assert_eq!(name(libc::SIGUSR1), "signal");
    }
}
