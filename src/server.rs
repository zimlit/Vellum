//! The PTY session server: a small daemon that owns the shell's PTY, so the
//! shell survives a crash of the front end (the process that owns the
//! display).
//!
//! - The front end connects to a Unix `SOCK_SEQPACKET` socket (one message
//!   per packet, so no framing) at [`socket_path`]. If nothing is listening,
//!   it binds the socket itself and starts the server
//!   (`vellum --session-server PATH`) with the listening socket as fd 3.
//!   That is socket activation: the front end's connection is queued before
//!   the server even runs, so there is no "wait until it's up" race.
//! - `Attach` starts the shell if there isn't one yet and replies with the PTY
//!   master fd (`SCM_RIGHTS`). From then on the front end reads and writes the
//!   PTY itself; the server never touches the data. It only keeps a reference
//!   to the master so the terminal stays open while no front end is attached.
//! - An orderly front-end exit sends `Close`, and the server hangs up the
//!   shell. A front end that goes away without it (crash, SIGKILL) leaves the
//!   shell running for the next `Attach`.
//! - The server exits when the shell does, or when its last client leaves
//!   and there is no shell.
//!
//! Passing the fd instead of relaying the bytes (as tmux does) keeps the data
//! path direct, but the server keeps no screen contents: a front end that
//! reattaches starts blank until the program redraws (a size change sends
//! SIGWINCH; Ctrl+L for a shell).

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{self, DirBuilder};
use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result, bail};
use rustix::net::{
    AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags,
    SendAncillaryBuffer, SendAncillaryMessage, SendFlags, SocketAddrUnix, SocketFlags, SocketType,
};
use rustix::process::{Pid, PidfdFlags, Signal};

use crate::event::Poller;
use crate::pty::{self, ShellSpec, WinSize};
use crate::signals::{self, SignalFd};

/// The listening socket's fd number in the server process.
const LISTEN_FD: RawFd = 3;
/// Largest message either side sends (an attach carries the configured
/// shell command and environment).
const MAX_MESSAGE: usize = 64 * 1024;

/// Where the session socket lives: `$VELLUM_SESSION_SOCKET`, else
/// `$XDG_RUNTIME_DIR/vellum/session.sock`.
pub fn socket_path() -> Result<PathBuf> {
    socket_path_from(|key| std::env::var_os(key))
        .context("neither VELLUM_SESSION_SOCKET nor XDG_RUNTIME_DIR is set")
}

fn socket_path_from(env: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    if let Some(p) = env("VELLUM_SESSION_SOCKET") {
        return Some(p.into());
    }
    env("XDG_RUNTIME_DIR").map(|dir| PathBuf::from(dir).join("vellum/session.sock"))
}

// ---- Protocol ----

#[derive(Debug, PartialEq, Eq)]
enum Request {
    /// Start the shell at this size if needed (as `shell` says), and send
    /// its PTY master.
    Attach { size: WinSize, shell: ShellSpec },
    /// The front end is quitting on purpose: hang up the shell.
    Close,
}

#[derive(Debug, PartialEq, Eq)]
enum Reply {
    /// Carries the PTY master as `SCM_RIGHTS`.
    Attached {
        pid: u32,
        spawned: bool,
    },
    Error(String),
}

const REQ_ATTACH: u8 = 1;
const REQ_CLOSE: u8 = 2;
const REPLY_ERROR: u8 = 0;
const REPLY_ATTACHED: u8 = 1;

impl Request {
    fn encode(&self) -> Vec<u8> {
        match self {
            Request::Attach { size: s, shell } => {
                let mut msg = vec![REQ_ATTACH];
                for v in [s.cols, s.rows, s.width, s.height] {
                    msg.extend_from_slice(&v.to_le_bytes());
                }
                put_count(&mut msg, shell.argv.len());
                for arg in &shell.argv {
                    put_str(&mut msg, arg);
                }
                put_count(&mut msg, shell.env.len());
                for (k, v) in &shell.env {
                    put_str(&mut msg, k);
                    put_str(&mut msg, v);
                }
                msg
            }
            Request::Close => vec![REQ_CLOSE],
        }
    }

    fn decode(msg: &[u8]) -> io::Result<Self> {
        match msg {
            [REQ_ATTACH, rest @ ..] => {
                let mut r = Reader(rest);
                let size = WinSize {
                    cols: r.u16()?,
                    rows: r.u16()?,
                    width: r.u16()?,
                    height: r.u16()?,
                };
                let argv = (0..r.u16()?).map(|_| r.str()).collect::<io::Result<_>>()?;
                let env = (0..r.u16()?)
                    .map(|_| Ok((r.str()?, r.str()?)))
                    .collect::<io::Result<_>>()?;
                if !r.0.is_empty() {
                    return Err(invalid("trailing bytes in attach request".into()));
                }
                Ok(Request::Attach {
                    size,
                    shell: ShellSpec { argv, env },
                })
            }
            [REQ_CLOSE] => Ok(Request::Close),
            _ => Err(invalid(format!("malformed request {msg:02x?}"))),
        }
    }
}

impl Reply {
    fn encode(&self) -> Vec<u8> {
        match self {
            Reply::Attached { pid, spawned } => {
                let mut msg = vec![REPLY_ATTACHED];
                msg.extend_from_slice(&pid.to_le_bytes());
                msg.push(u8::from(*spawned));
                msg
            }
            Reply::Error(text) => {
                let mut end = text.len().min(MAX_MESSAGE - 1);
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                let mut msg = vec![REPLY_ERROR];
                msg.extend_from_slice(&text.as_bytes()[..end]);
                msg
            }
        }
    }

    fn decode(msg: &[u8]) -> io::Result<Self> {
        match *msg {
            [REPLY_ATTACHED, a, b, c, d, spawned @ (0 | 1)] => Ok(Reply::Attached {
                pid: u32::from_le_bytes([a, b, c, d]),
                spawned: spawned == 1,
            }),
            [REPLY_ERROR, ref text @ ..] => {
                Ok(Reply::Error(String::from_utf8_lossy(text).into_owned()))
            }
            _ => Err(invalid(format!("malformed reply {msg:02x?}"))),
        }
    }
}

/// A list's length. Callers check the whole message against
/// [`MAX_MESSAGE`], which keeps lists far below `u16::MAX`.
fn put_count(msg: &mut Vec<u8>, n: usize) {
    msg.extend_from_slice(&(n.min(u16::MAX.into()) as u16).to_le_bytes());
}

fn put_str(msg: &mut Vec<u8>, s: &str) {
    msg.extend_from_slice(&(s.len() as u32).to_le_bytes());
    msg.extend_from_slice(s.as_bytes());
}

/// Reads a message front to back; running out early means it's malformed.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.0.len() < n {
            return Err(invalid("truncated request".into()));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn u16(&mut self) -> io::Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn str(&mut self) -> io::Result<String> {
        let b = self.take(4)?;
        let len = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| invalid("a string in the request isn't UTF-8".into()))
    }
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Send one message, optionally with an fd attached.
fn send(sock: impl AsFd, msg: &[u8], fd: Option<BorrowedFd>) -> io::Result<()> {
    let fds: Vec<BorrowedFd> = fd.into_iter().collect();
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !fds.is_empty() {
        control.push(SendAncillaryMessage::ScmRights(&fds));
    }
    // NOSIGNAL: a peer that went away is an EPIPE error, not a SIGPIPE.
    rustix::net::sendmsg(
        sock,
        &[IoSlice::new(msg)],
        &mut control,
        SendFlags::NOSIGNAL,
    )?;
    Ok(())
}

/// Receive one message into `buf`: its length (0 at end of stream) and the
/// fd that came with it, if any.
fn recv(sock: impl AsFd, buf: &mut [u8]) -> io::Result<(usize, Option<OwnedFd>)> {
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let msg = rustix::net::recvmsg(
        sock,
        &mut [IoSliceMut::new(buf)],
        &mut control,
        RecvFlags::CMSG_CLOEXEC,
    )?;
    let mut fd = None;
    for message in control.drain() {
        if let RecvAncillaryMessage::ScmRights(fds) = message {
            for received in fds {
                // Keep the first; extra ones are closed as they drop.
                fd.get_or_insert(received);
            }
        }
    }
    if msg
        .flags
        .intersects(ReturnFlags::TRUNC | ReturnFlags::CTRUNC)
    {
        return Err(invalid("oversized message".into()));
    }
    Ok((msg.bytes, fd))
}

fn seqpacket_socket() -> io::Result<OwnedFd> {
    Ok(rustix::net::socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC,
        None,
    )?)
}

fn listen(path: &Path) -> io::Result<OwnedFd> {
    let sock = seqpacket_socket()?;
    rustix::net::bind(&sock, &SocketAddrUnix::new(path)?)?;
    rustix::net::listen(&sock, 16)?;
    Ok(sock)
}

fn connect(path: &Path) -> io::Result<OwnedFd> {
    let sock = seqpacket_socket()?;
    rustix::net::connect(&sock, &SocketAddrUnix::new(path)?)?;
    Ok(sock)
}

// ---- Client (front end) ----

/// The front end's connection to the shell.
pub struct Attachment {
    conn: OwnedFd,
    pub master: OwnedFd,
    pub pid: u32,
    /// The shell was started for this attach (rather than already running).
    pub spawned: bool,
}

impl Attachment {
    /// Orderly exit: ask the server to hang up the shell. Not sent on a crash,
    /// which leaves the shell running for the next attach.
    pub fn close(self) -> io::Result<()> {
        send(&self.conn, &Request::Close.encode(), None)
    }
}

/// Attach to the shell behind `path`, starting the server if it isn't
/// running. `shell` is what to run if the shell has to be started.
pub fn attach(path: &Path, size: WinSize, shell: &ShellSpec) -> Result<Attachment> {
    let mut last_error = None;
    for _ in 0..3 {
        match try_attach(path, size, shell) {
            Ok(a) => return Ok(a),
            // A server that was exiting as we connected, or another front end
            // that bound the socket first: try again.
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::UnexpectedEof
                        | io::ErrorKind::ConnectionReset
                        | io::ErrorKind::BrokenPipe
                        | io::ErrorKind::AddrInUse
                ) =>
            {
                tracing::debug!("attach failed, retrying: {e}");
                last_error = Some(e);
            }
            Err(e) => return Err(e).context(format!("attaching to {}", path.display())),
        }
    }
    Err(last_error.unwrap()).context(format!("attaching to {}", path.display()))
}

fn try_attach(path: &Path, size: WinSize, shell: &ShellSpec) -> io::Result<Attachment> {
    let conn = match connect(path) {
        Ok(conn) => conn,
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            ) =>
        {
            start_server(path)?
        }
        Err(e) => return Err(e),
    };
    request_attach(conn, size, shell)
}

fn request_attach(conn: OwnedFd, size: WinSize, shell: &ShellSpec) -> io::Result<Attachment> {
    let request = Request::Attach {
        size,
        shell: shell.clone(),
    }
    .encode();
    if request.len() > MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("the shell command and environment are too long (over {MAX_MESSAGE} bytes)"),
        ));
    }
    send(&conn, &request, None)?;
    let mut buf = vec![0u8; MAX_MESSAGE];
    let (n, fd) = recv(&conn, &mut buf)?;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "session server closed the connection",
        ));
    }
    match Reply::decode(&buf[..n])? {
        Reply::Attached { pid, spawned } => Ok(Attachment {
            conn,
            master: fd.ok_or_else(|| invalid("attach reply without a PTY fd".into()))?,
            pid,
            spawned,
        }),
        Reply::Error(text) => Err(io::Error::other(format!("session server: {text}"))),
    }
}

/// Bind the socket, queue our connection on it, then start the server with
/// the listening socket. Returns our connection.
fn start_server(path: &Path) -> io::Result<OwnedFd> {
    if let Some(dir) = path.parent() {
        DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    }
    match fs::remove_file(path) {
        Ok(()) => tracing::debug!(socket = %path.display(), "removed stale socket"),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let listener = listen(path)?;
    let conn = connect(path)?;
    spawn_server(&listener, path)?;
    tracing::info!(socket = %path.display(), "started session server");
    Ok(conn)
}

fn spawn_server(listener: &OwnedFd, path: &Path) -> io::Result<()> {
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.arg("--session-server")
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let fd = listener.as_raw_fd();
    // SAFETY: runs between fork and exec in a single-threaded child, using
    // only async-signal-safe calls.
    unsafe {
        cmd.pre_exec(move || {
            signals::unblock_all()?;
            // Own session: no controlling terminal, so the VT's hangup
            // doesn't reach the server or the shell.
            rustix::process::setsid()?;
            // Fork again and let the middle process exit: the server is
            // reparented to init and never becomes our zombie. std's
            // exec-error pipe still works; only the grandchild execs.
            match libc::fork() {
                -1 => return Err(io::Error::last_os_error()),
                0 => {}
                _ => libc::_exit(0),
            }
            let ok = if fd == LISTEN_FD {
                libc::fcntl(fd, libc::F_SETFD, 0) // just clear close-on-exec
            } else {
                libc::dup2(fd, LISTEN_FD) // the copy is not close-on-exec
            };
            if ok < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    // Reap the middle process, which exits right away.
    cmd.spawn()?.wait()?;
    Ok(())
}

// ---- Server ----

/// Entry point of `vellum --session-server PATH`.
pub fn main(path: PathBuf) -> Result<()> {
    let signals = SignalFd::new(&[libc::SIGTERM, libc::SIGINT, libc::SIGHUP])
        .context("setting up signalfd")?;
    // SAFETY: fcntl on a possibly unused fd number is harmless.
    if unsafe { libc::fcntl(LISTEN_FD, libc::F_GETFD) } < 0 {
        bail!("no listening socket on fd {LISTEN_FD} (the session server is started by vellum)");
    }
    // SAFETY: by contract with spawn_server, fd 3 is our listening socket and
    // nothing else in this process owns it.
    let listener = unsafe { OwnedFd::from_raw_fd(LISTEN_FD) };
    let stat = rustix::fs::fstat(&listener)?;
    if !rustix::fs::FileType::from_raw_mode(stat.st_mode).is_socket() {
        bail!("fd {LISTEN_FD} is not a socket (the session server is started by vellum)");
    }
    // dup2 cleared close-on-exec; the shell must not inherit the listener.
    rustix::io::fcntl_setfd(&listener, rustix::io::FdFlags::CLOEXEC)?;

    tracing::info!(socket = %path.display(), pid = std::process::id(), "session server running");
    Server::new(
        listener,
        Some(path),
        {
            let info = pty::TermInfo::install();
            tracing::info!(term = info.term(), "shell terminfo");
            Box::new(move |spec: &ShellSpec| pty::shell_command(&info, spec))
        },
        Some(signals),
    )?
    .run()?;
    tracing::info!("session server exiting");
    Ok(())
}

const LISTENER: u64 = 0;
const SIGNALS: u64 = 1;
const SHELL: u64 = 2;
const FIRST_CLIENT: u64 = 3;

struct Shell {
    /// `None` once hung up; the shell is then on its way out.
    master: Option<OwnedFd>,
    child: Child,
    /// Readable when the child exits.
    pidfd: OwnedFd,
}

pub struct Server {
    listener: OwnedFd,
    /// Socket file to remove on exit.
    path: Option<PathBuf>,
    command: Box<dyn Fn(&ShellSpec) -> Command>,
    signals: Option<SignalFd>,
    poller: Poller,
    clients: HashMap<u64, OwnedFd>,
    next_token: u64,
    shell: Option<Shell>,
}

impl Server {
    /// `command` builds the program to run on the PTY, from what the
    /// attaching front end asked for.
    pub fn new(
        listener: OwnedFd,
        path: Option<PathBuf>,
        command: Box<dyn Fn(&ShellSpec) -> Command>,
        signals: Option<SignalFd>,
    ) -> io::Result<Self> {
        let poller = Poller::new()?;
        poller.add(&listener, LISTENER)?;
        if let Some(signals) = &signals {
            poller.add(signals, SIGNALS)?;
        }
        Ok(Self {
            listener,
            path,
            command,
            signals,
            poller,
            clients: HashMap::new(),
            next_token: FIRST_CLIENT,
            shell: None,
        })
    }

    /// Serve until the shell exits, a termination signal arrives, or the last
    /// client leaves with no shell running.
    pub fn run(mut self) -> io::Result<()> {
        let mut ready = Vec::new();
        loop {
            self.poller.wait(None, &mut ready)?;
            for &token in &ready {
                let done = match token {
                    LISTENER => {
                        self.accept();
                        false
                    }
                    SIGNALS => self.signal()?,
                    SHELL => self.reap()?,
                    token => self.client(token),
                };
                if done {
                    return Ok(());
                }
            }
        }
    }

    fn accept(&mut self) {
        let conn = match rustix::net::accept_with(&self.listener, SocketFlags::CLOEXEC) {
            Ok(conn) => conn,
            Err(e) => {
                tracing::warn!("accept: {e}");
                return;
            }
        };
        let token = self.next_token;
        self.next_token += 1;
        if let Err(e) = self.poller.add(&conn, token) {
            tracing::warn!("watching client: {e}");
            return;
        }
        tracing::debug!(client = token, "client connected");
        self.clients.insert(token, conn);
    }

    /// Handle a message from (or the departure of) a client. Returns whether
    /// the server should exit.
    fn client(&mut self, token: u64) -> bool {
        let Some(conn) = self.clients.get(&token) else {
            return false;
        };
        let mut buf = vec![0u8; MAX_MESSAGE];
        let result = match recv(conn, &mut buf) {
            Ok((0, _)) => Ok(None),
            Ok((n, _)) => Request::decode(&buf[..n]).map(Some),
            Err(e) => Err(e),
        };
        match result.and_then(|req| req.map(|r| self.handle(token, r)).transpose()) {
            Ok(Some(())) => {}
            Ok(None) => {
                tracing::debug!(client = token, "client disconnected");
                self.clients.remove(&token);
            }
            Err(e) => {
                tracing::warn!(client = token, "dropping client: {e}");
                self.clients.remove(&token);
            }
        }
        // Nothing to serve and nothing to keep alive.
        self.shell.is_none() && self.clients.is_empty()
    }

    fn handle(&mut self, token: u64, req: Request) -> io::Result<()> {
        match req {
            Request::Attach { size, shell } => {
                let result = self.ensure_shell(size, &shell);
                let conn = &self.clients[&token];
                match result {
                    Ok(spawned) => {
                        let shell = self.shell.as_ref().expect("ensure_shell succeeded");
                        let pid = shell.child.id();
                        tracing::info!(client = token, pid, spawned, "attached");
                        let reply = Reply::Attached { pid, spawned };
                        send(
                            conn,
                            &reply.encode(),
                            shell.master.as_ref().map(|m| m.as_fd()),
                        )
                    }
                    Err(e) => {
                        tracing::warn!(client = token, "attach failed: {e}");
                        send(conn, &Reply::Error(e.to_string()).encode(), None)
                    }
                }
            }
            Request::Close => {
                tracing::info!(
                    client = token,
                    "front end closed the session; hanging up the shell"
                );
                self.hang_up();
                Ok(())
            }
        }
    }

    /// Start the shell if there is none. Returns whether it was started.
    fn ensure_shell(&mut self, size: WinSize, spec: &ShellSpec) -> io::Result<bool> {
        if let Some(shell) = &self.shell {
            return match shell.master {
                Some(_) => Ok(false),
                None => Err(io::Error::other("the shell is exiting")),
            };
        }
        let mut spawned = pty::spawn((self.command)(spec), size)?;
        let pidfd =
            rustix::process::pidfd_open(Pid::from_child(&spawned.child), PidfdFlags::empty())
                .map_err(io::Error::from)
                .and_then(|pidfd| self.poller.add(&pidfd, SHELL).map(|()| pidfd));
        let pidfd = match pidfd {
            Ok(pidfd) => pidfd,
            Err(e) => {
                // Can't watch it, so don't keep it.
                let _ = spawned.child.kill();
                let _ = spawned.child.wait();
                return Err(e);
            }
        };
        tracing::info!(pid = spawned.child.id(), ?size, "started shell");
        self.shell = Some(Shell {
            master: Some(spawned.master),
            child: spawned.child,
            pidfd,
        });
        Ok(true)
    }

    /// Drop our end of the terminal and send the shell SIGHUP, as a real
    /// terminal does when it goes away. (The kernel also hangs up the PTY
    /// once the front end's copy of the master is closed.)
    fn hang_up(&mut self) {
        if let Some(shell) = &mut self.shell {
            shell.master = None;
            if let Err(e) = rustix::process::pidfd_send_signal(&shell.pidfd, Signal::HUP) {
                tracing::debug!("SIGHUP to shell: {e}");
            }
        }
    }

    /// The shell's pidfd is readable. Returns whether it exited.
    fn reap(&mut self) -> io::Result<bool> {
        let Some(shell) = &mut self.shell else {
            return Ok(false);
        };
        let Some(status) = shell.child.try_wait()? else {
            return Ok(false);
        };
        tracing::info!(pid = shell.child.id(), %status, "shell exited");
        self.shell = None;
        Ok(true)
    }

    fn signal(&mut self) -> io::Result<bool> {
        let Some(signals) = &self.signals else {
            return Ok(false);
        };
        let Some(sig) = signals.read()? else {
            return Ok(false);
        };
        tracing::info!(
            signal = signals::name(sig),
            "exiting on signal; hanging up the shell"
        );
        self.hang_up();
        Ok(true)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // Remove the socket file while still listening, so a front end never
        // finds a socket nobody will accept on (it would cope, but slower).
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use crate::pty::tests::read_until;
    use crate::testutil::TempDir;

    const SIZE: WinSize = WinSize {
        cols: 100,
        rows: 30,
        width: 800,
        height: 480,
    };

    #[test]
    fn requests_round_trip() {
        // Encoded only, never run.
        let shell = ShellSpec {
            argv: vec!["/bin/sh".into(), "-l".into()],
            env: vec![
                ("EDITOR".into(), "emacs".into()),
                ("EMPTY".into(), String::new()),
            ],
        };
        for req in [
            Request::Attach {
                size: SIZE,
                shell: ShellSpec::default(),
            },
            Request::Attach { size: SIZE, shell },
            Request::Close,
        ] {
            assert_eq!(Request::decode(&req.encode()).unwrap(), req);
        }
    }

    #[test]
    fn replies_round_trip() {
        for reply in [
            Reply::Attached {
                pid: 123456,
                spawned: true,
            },
            Reply::Attached {
                pid: 1,
                spawned: false,
            },
            Reply::Error("no shell".into()),
        ] {
            assert_eq!(Reply::decode(&reply.encode()).unwrap(), reply);
        }
    }

    #[test]
    fn malformed_messages_are_rejected() {
        assert!(Request::decode(&[]).is_err());
        assert!(
            Request::decode(&[REQ_ATTACH, 1, 2]).is_err(),
            "short attach"
        );
        let mut attach = Request::Attach {
            size: SIZE,
            shell: ShellSpec {
                argv: vec!["sh".into()],
                env: vec![],
            },
        }
        .encode();
        assert!(
            Request::decode(&attach[..attach.len() - 1]).is_err(),
            "string cut short"
        );
        attach.push(0);
        assert!(Request::decode(&attach).is_err(), "trailing byte");
        assert!(Request::decode(&[REQ_CLOSE, 0]).is_err(), "trailing bytes");
        assert!(Request::decode(&[9]).is_err(), "unknown op");
        assert!(
            Reply::decode(&[REPLY_ATTACHED, 1, 0, 0, 0, 2]).is_err(),
            "bad bool"
        );
        assert!(Reply::decode(&[7]).is_err());
    }

    #[test]
    fn long_error_is_truncated_on_a_char_boundary() {
        let text = "é".repeat(MAX_MESSAGE);
        let msg = Reply::Error(text).encode();
        assert!(msg.len() <= MAX_MESSAGE);
        assert!(
            matches!(Reply::decode(&msg).unwrap(), Reply::Error(t) if t.chars().all(|c| c == 'é'))
        );
    }

    #[test]
    fn socket_path_prefers_override() {
        let env = |vars: &'static [(&'static str, &'static str)]| {
            move |k: &str| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.into())
        };
        assert_eq!(
            socket_path_from(env(&[
                ("VELLUM_SESSION_SOCKET", "/x.sock"),
                ("XDG_RUNTIME_DIR", "/run/u")
            ])),
            Some("/x.sock".into())
        );
        assert_eq!(
            socket_path_from(env(&[("XDG_RUNTIME_DIR", "/run/u")])),
            Some("/run/u/vellum/session.sock".into())
        );
        assert_eq!(socket_path_from(env(&[])), None);
    }

    #[test]
    fn fd_passes_over_the_socket() {
        let (a, b) = rustix::net::socketpair(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC,
            None,
        )
        .unwrap();
        let (master, slave) = pty::open().unwrap();
        send(&a, b"x", Some(master.as_fd())).unwrap();
        let mut buf = [0u8; 8];
        let (n, fd) = recv(&b, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"x");
        // The received fd is the same PTY master.
        rustix::io::write(&slave, b"ok\n").unwrap();
        let (out, _) = read_until(&fd.unwrap(), Duration::from_secs(5), |o| o.ends_with(b"\n"));
        assert_eq!(out, b"ok\r\n");
    }

    /// A server on a temporary socket, running `/bin/sh` on its own thread.
    /// The receiver yields its result when it exits.
    fn start(dir: &TempDir) -> (PathBuf, mpsc::Receiver<io::Result<()>>) {
        let path = dir.path().join("session.sock");
        let listener = listen(&path).unwrap();
        let (tx, rx) = mpsc::channel();
        let server_path = path.clone();
        std::thread::spawn(move || {
            let command = Box::new(|spec: &ShellSpec| {
                let mut cmd = Command::new("/bin/sh");
                cmd.envs(spec.env.iter().map(|(k, v)| (k, v)));
                pty::terminal_env(&mut cmd, &pty::TermInfo::Fallback).env("PS1", "$ ");
                cmd
            });
            let result =
                Server::new(listener, Some(server_path), command, None).and_then(Server::run);
            let _ = tx.send(result);
        });
        (path, rx)
    }

    fn wait_exit(rx: &mpsc::Receiver<io::Result<()>>) {
        rx.recv_timeout(Duration::from_secs(10))
            .expect("server did not exit")
            .unwrap();
    }

    fn shell_says(a: &Attachment, command: &str, expect: &str) {
        rustix::io::write(&a.master, format!("{command}\n").as_bytes()).unwrap();
        let (out, _) = read_until(&a.master, Duration::from_secs(10), |o| {
            String::from_utf8_lossy(o).contains(expect)
        });
        assert!(
            String::from_utf8_lossy(&out).contains(expect),
            "{:?}",
            out.escape_ascii().to_string()
        );
    }

    #[test]
    fn attach_starts_shell_once_and_close_ends_it() {
        let dir = TempDir::new();
        let (path, done) = start(&dir);

        let shell = ShellSpec {
            argv: vec![],
            env: vec![("FROM_CONFIG".into(), "first".into())],
        };
        let first = request_attach(connect(&path).unwrap(), SIZE, &shell).unwrap();
        assert!(first.spawned);
        shell_says(&first, "echo $((6*7))", "42");
        shell_says(&first, "stty size", "30 100");
        shell_says(&first, "echo env=$FROM_CONFIG", "env=first");

        // Already running: the second front end's shell settings don't
        // matter.
        let other = ShellSpec {
            argv: vec![],
            env: vec![("FROM_CONFIG".into(), "second".into())],
        };
        let second = request_attach(connect(&path).unwrap(), SIZE, &other).unwrap();
        assert!(!second.spawned);
        assert_eq!(second.pid, first.pid);

        drop(second);
        first.close().unwrap();
        wait_exit(&done);
        assert!(!path.exists(), "socket file removed on exit");
    }

    #[test]
    fn shell_survives_a_vanished_front_end() {
        let dir = TempDir::new();
        let (path, done) = start(&dir);

        let shell = ShellSpec::default();
        let first = request_attach(connect(&path).unwrap(), SIZE, &shell).unwrap();
        shell_says(&first, "x=kept", "$ ");
        // A crash: connection and master go away without Close.
        let pid = first.pid;
        drop(first);

        let again = request_attach(connect(&path).unwrap(), SIZE, &shell).unwrap();
        assert!(!again.spawned);
        assert_eq!(again.pid, pid);
        shell_says(&again, "echo $x", "kept");
        again.close().unwrap();
        wait_exit(&done);
    }

    #[test]
    fn server_exits_when_the_shell_does() {
        let dir = TempDir::new();
        let (path, done) = start(&dir);
        let a = request_attach(connect(&path).unwrap(), SIZE, &ShellSpec::default()).unwrap();
        rustix::io::write(&a.master, b"exit\n").unwrap();
        let (_, hung_up) = read_until(&a.master, Duration::from_secs(10), |_| false);
        assert!(hung_up);
        wait_exit(&done);
    }

    #[test]
    fn server_exits_when_idle_client_leaves() {
        let dir = TempDir::new();
        let (path, done) = start(&dir);
        drop(connect(&path).unwrap());
        wait_exit(&done);
    }
}
