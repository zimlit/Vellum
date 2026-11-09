//! Pseudo-terminals, and starting a shell on one.
//!
//! A PTY is a pair of character devices joined by the kernel's line
//! discipline (echo, line editing, ^C → SIGINT, `\n` → `\r\n`, ...). The
//! *master* side belongs to the terminal emulator: it reads what the program
//! printed and writes what the user typed. The program sees the *slave* side
//! (`/dev/pts/N`) as its terminal on stdin/stdout/stderr.
//!
//! The child also becomes a session leader (`setsid`) with the slave as its
//! controlling terminal (`TIOCSCTTY`). That is what makes job control,
//! keyboard signals and `/dev/tty` work, and it is why the shell gets SIGHUP
//! when the terminal goes away.

use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};

use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
use rustix::termios::{self, tcsetwinsize};

use crate::signals;

/// `$TERM` for programs in the terminal. This is the emulation vellum aims
/// for; until the terminal core (roadmap step 5) exists nothing interprets it.
pub const TERM: &str = "xterm-256color";

/// Terminal size in cells and in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WinSize {
    pub cols: u16,
    pub rows: u16,
    pub width: u16,
    pub height: u16,
}

impl From<WinSize> for termios::Winsize {
    fn from(s: WinSize) -> Self {
        termios::Winsize {
            ws_row: s.rows,
            ws_col: s.cols,
            ws_xpixel: s.width,
            ws_ypixel: s.height,
        }
    }
}

/// Open a new PTY, returning `(master, slave)`. Both are close-on-exec and
/// the slave does not become anyone's controlling terminal yet.
pub fn open() -> io::Result<(OwnedFd, OwnedFd)> {
    let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
    let master = openpt(flags)?;
    // grantpt does nothing on Linux (devpts sets the owner itself), but POSIX
    // requires it before unlockpt.
    grantpt(&master)?;
    unlockpt(&master)?;
    // TIOCGPTPEER opens the slave straight from the master, instead of
    // looking up ptsname() in /dev/pts and opening that path.
    let slave = ioctl_tiocgptpeer(&master, flags)?;
    Ok((master, slave))
}

/// Set the window size. The kernel sends SIGWINCH to the terminal's
/// foreground process group if it changed.
pub fn set_size(pty: impl AsFd, size: WinSize) -> io::Result<()> {
    tcsetwinsize(pty, size.into())?;
    Ok(())
}

/// A program running on a PTY.
pub struct Spawned {
    pub master: OwnedFd,
    pub child: Child,
}

/// Run `cmd` on a new PTY of the given size. Its stdio is replaced by the
/// slave; everything else (program, args, environment, cwd) is up to the
/// caller.
pub fn spawn(mut cmd: Command, size: WinSize) -> io::Result<Spawned> {
    let (master, slave) = open()?;
    set_size(&master, size)?;
    cmd.stdin(Stdio::from(slave.try_clone()?))
        .stdout(Stdio::from(slave.try_clone()?))
        .stderr(Stdio::from(slave));
    // SAFETY: runs between fork and exec, and only makes async-signal-safe
    // system calls. std has already moved the slave to fds 0-2.
    unsafe {
        cmd.pre_exec(|| {
            signals::unblock_all()?;
            rustix::process::setsid()?;
            rustix::process::ioctl_tiocsctty(BorrowedFd::borrow_raw(0))?;
            Ok(())
        });
    }
    let child = cmd.spawn()?;
    // `cmd` holds our copies of the slave. Close them now: the master only
    // reports EIO (the child is gone) once every slave fd is closed.
    drop(cmd);
    Ok(Spawned { master, child })
}

/// The user's shell (`$SHELL`, else `/bin/sh`) with `TERM` set.
pub fn shell_command() -> Command {
    let shell = std::env::var_os("SHELL")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/bin/sh".into());
    let mut cmd = Command::new(shell);
    cmd.env("TERM", TERM);
    cmd
}

#[cfg(test)]
pub mod tests {
    use std::time::{Duration, Instant};

    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    use rustix::io::Errno;
    use rustix::termios::tcgetwinsize;

    use super::*;

    const SIZE: WinSize = WinSize {
        cols: 80,
        rows: 24,
        width: 640,
        height: 384,
    };

    /// Read from a PTY master until `done(output)` or `timeout`. Returns
    /// what was read and whether the slave side hung up (EIO).
    pub fn read_until(
        master: &OwnedFd,
        timeout: Duration,
        done: impl Fn(&[u8]) -> bool,
    ) -> (Vec<u8>, bool) {
        let deadline = Instant::now() + timeout;
        let mut out = Vec::new();
        let mut buf = [0u8; 4096];
        while !done(&out) {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            let ts = Timespec {
                tv_sec: left.as_secs() as _,
                tv_nsec: left.subsec_nanos() as _,
            };
            let mut fds = [PollFd::new(master, PollFlags::IN)];
            match poll(&mut fds, Some(&ts)) {
                Ok(0) | Err(Errno::INTR) => continue,
                Ok(_) => {}
                Err(e) => panic!("poll: {e}"),
            }
            match rustix::io::read(master, &mut buf) {
                Ok(0) | Err(Errno::IO) => return (out, true),
                Ok(n) => out.extend_from_slice(&buf[..n]),
                Err(Errno::AGAIN | Errno::INTR) => {}
                Err(e) => panic!("read: {e}"),
            }
        }
        (out, false)
    }

    fn run(script: &str) -> String {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg(script).env("TERM", TERM);
        let mut spawned = spawn(cmd, SIZE).unwrap();
        let (out, hung_up) = read_until(&spawned.master, Duration::from_secs(10), |_| false);
        assert!(
            hung_up,
            "child output never ended: {:?}",
            out.escape_ascii()
        );
        assert!(spawned.child.wait().unwrap().success());
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn open_gives_a_connected_pair() {
        let (master, slave) = open().unwrap();
        rustix::io::write(&slave, b"hi\n").unwrap();
        let (out, _) = read_until(&master, Duration::from_secs(5), |o| o.ends_with(b"\n"));
        // The line discipline's output processing turns \n into \r\n.
        assert_eq!(out, b"hi\r\n");
    }

    #[test]
    fn size_set_on_master_is_seen_on_slave() {
        let (master, slave) = open().unwrap();
        set_size(&master, SIZE).unwrap();
        let ws = tcgetwinsize(&slave).unwrap();
        assert_eq!(
            (ws.ws_col, ws.ws_row, ws.ws_xpixel, ws.ws_ypixel),
            (80, 24, 640, 384)
        );
    }

    #[test]
    fn child_sees_size_term_and_a_controlling_terminal() {
        // Opening /dev/tty only works with a controlling terminal.
        let out = run("stty size; echo \"$TERM\"; exec 3</dev/tty && echo ctty");
        assert_eq!(out, "24 80\r\nxterm-256color\r\nctty\r\n");
    }

    #[test]
    fn child_is_a_session_leader() {
        // Field 6 of /proc/<pid>/stat is the session id.
        let out = run("read -r stat </proc/$$/stat; set -- $stat; [ \"$6\" = $$ ] && echo leader");
        assert_eq!(out, "leader\r\n");
    }

    #[test]
    fn child_starts_with_no_signals_blocked() {
        // Blocked on this test's thread, which forks the child.
        let _signals = crate::signals::SignalFd::new(&[libc::SIGURG]).unwrap();
        // SigBlk in /proc/<pid>/status is the blocked mask, in hex.
        let out = run("grep SigBlk /proc/$$/status");
        assert_eq!(out.trim(), "SigBlk:\t0000000000000000");
    }

    #[test]
    fn shell_command_sets_term() {
        let cmd = shell_command();
        assert!(
            cmd.get_envs()
                .any(|(k, v)| k == "TERM" && v == Some(TERM.as_ref()))
        );
    }
}
