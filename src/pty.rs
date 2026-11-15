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

/// `$TERM` for programs in the terminal: the emulation the terminal core
/// (`crate::term`) follows.
pub const TERM: &str = "xterm-256color";

/// `$COLORTERM`: SGR 38/48 with RGB colors work (terminfo can't say so).
pub const COLORTERM: &str = "truecolor";

/// Variables other terminals set to identify themselves. They're inherited
/// when vellum is started from inside one (e.g. over SSH with `SendEnv`),
/// and would make programs expect that terminal's features.
const OTHER_TERMINALS: &[&str] = &[
    "VTE_VERSION",
    "KONSOLE_VERSION",
    "KONSOLE_DBUS_SESSION",
    "KITTY_WINDOW_ID",
    "KITTY_PID",
    "KITTY_PUBLIC_KEY",
    "WEZTERM_PANE",
    "WEZTERM_EXECUTABLE",
    "WEZTERM_UNIX_SOCKET",
    "ALACRITTY_WINDOW_ID",
    "ALACRITTY_SOCKET",
    "GHOSTTY_RESOURCES_DIR",
    "WT_SESSION",
    "TERMINAL_EMULATOR",
    "TERM_SESSION_ID",
];

/// Set the environment that describes vellum to programs in it: `TERM`,
/// `COLORTERM`, `TERM_PROGRAM` and `TERM_PROGRAM_VERSION`, and remove what
/// another terminal may have left there.
pub fn terminal_env(cmd: &mut Command) -> &mut Command {
    for var in OTHER_TERMINALS {
        cmd.env_remove(var);
    }
    cmd.env("TERM", TERM)
        .env("COLORTERM", COLORTERM)
        .env("TERM_PROGRAM", "vellum")
        .env("TERM_PROGRAM_VERSION", env!("CARGO_PKG_VERSION"))
}

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

/// Most bytes [`Writer`] holds for a program that isn't reading; more are
/// dropped (with a warning) rather than growing without bound.
const MAX_PENDING: usize = 1 << 20;

/// Writes to a non-blocking PTY master, keeping what it can't take yet in
/// order. When [`Writer::pending`], the owner waits for the fd to be
/// writable and calls [`Writer::flush`].
#[derive(Default)]
pub struct Writer {
    pending: Vec<u8>,
}

impl Writer {
    /// Write `bytes` after anything still pending.
    pub fn send(&mut self, fd: impl AsFd, bytes: &[u8]) -> io::Result<()> {
        let room = MAX_PENDING - self.pending.len();
        if bytes.len() > room {
            tracing::warn!(
                dropped = bytes.len() - room,
                "the program isn't reading its input; dropping some"
            );
        }
        self.pending
            .extend_from_slice(&bytes[..bytes.len().min(room)]);
        self.flush(fd)
    }

    /// Write as much of what's pending as the fd takes.
    pub fn flush(&mut self, fd: impl AsFd) -> io::Result<()> {
        let mut done = 0;
        while done < self.pending.len() {
            match rustix::io::write(&fd, &self.pending[done..]) {
                Ok(n) => done += n,
                Err(rustix::io::Errno::INTR) => {}
                Err(rustix::io::Errno::AGAIN) => break,
                // The program is gone; the reader finds out the same way.
                Err(rustix::io::Errno::IO) => done = self.pending.len(),
                Err(e) => {
                    self.pending.drain(..done);
                    return Err(e.into());
                }
            }
        }
        self.pending.drain(..done);
        Ok(())
    }

    /// Bytes are waiting for the fd to become writable.
    pub fn pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// The user's shell (`$SHELL`, else `/bin/sh`) with the [`terminal_env`].
pub fn shell_command() -> Command {
    let shell = std::env::var_os("SHELL")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/bin/sh".into());
    let mut cmd = Command::new(shell);
    terminal_env(&mut cmd);
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
        terminal_env(cmd.arg("-c").arg(script));
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
    fn shell_command_describes_vellum() {
        let cmd = shell_command();
        let envs: Vec<_> = cmd.get_envs().collect();
        let has = |k: &str, v: &str| envs.contains(&(k.as_ref(), Some(v.as_ref())));
        assert!(has("TERM", TERM));
        assert!(has("COLORTERM", COLORTERM));
        assert!(has("TERM_PROGRAM", "vellum"));
        // Another terminal's markers are removed (None: unset).
        assert!(envs.contains(&("VTE_VERSION".as_ref(), None)));
    }

    #[test]
    fn writer_keeps_what_the_pty_cannot_take() {
        let (master, slave) = open().unwrap();
        rustix::fs::fcntl_setfl(&master, rustix::fs::OFlags::NONBLOCK).unwrap();
        rustix::fs::fcntl_setfl(&slave, rustix::fs::OFlags::NONBLOCK).unwrap();
        // Raw mode: bytes pass through unchanged, and nothing echoes back.
        let mut t = termios::tcgetattr(&slave).unwrap();
        t.make_raw();
        termios::tcsetattr(&slave, termios::OptionalActions::Now, &t).unwrap();

        // More than the PTY buffers, with a recognizable pattern.
        let data: Vec<u8> = (0..256 * 1024).map(|i| (i % 251) as u8).collect();
        let mut w = Writer::default();
        w.send(&master, &data[..100 * 1024]).unwrap();
        assert!(w.pending(), "the slave hasn't read anything yet");
        w.send(&master, &data[100 * 1024..]).unwrap();

        let mut got = Vec::new();
        let mut buf = [0u8; 8192];
        let deadline = Instant::now() + Duration::from_secs(10);
        while got.len() < data.len() && Instant::now() < deadline {
            match rustix::io::read(&slave, &mut buf) {
                Ok(n) => got.extend_from_slice(&buf[..n]),
                Err(Errno::AGAIN) => w.flush(&master).unwrap(),
                Err(e) => panic!("read: {e}"),
            }
        }
        assert!(!w.pending());
        assert!(got == data, "bytes arrived in order, all of them");
    }

    #[test]
    fn child_environment() {
        // VTE_VERSION is unset even when the tests run inside a VTE terminal.
        let out = run("echo \"$TERM $COLORTERM $TERM_PROGRAM ${VTE_VERSION-unset}\"");
        assert_eq!(out, "xterm-256color truecolor vellum unset\r\n");
    }
}
