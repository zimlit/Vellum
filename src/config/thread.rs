//! The script thread: the one Lua state lives here, so the event loop never
//! runs Lua and a slow or stuck script can't hold up drawing, output or
//! input.
//!
//! The front end sends [`Request`]s; [`Reply`]s come back over a channel,
//! and an eventfd (registered with the event loop) says some are waiting.
//! Only plain Rust data crosses between the threads; the Lua state never
//! leaves this one.
//!
//! A request that isn't answered within the configured `script_timeout`
//! gets a warning ([`Script::overdue`]). A stuck thread can't be stopped
//! from outside; a reload starts a fresh one and abandons it.

use std::collections::VecDeque;
use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use rustix::event::{EventfdFlags, eventfd};
use rustix::io::Errno;

use super::api;
use super::settings::Settings;

pub enum Request {
    /// Run the config at this path (`None`: no config) in a fresh state,
    /// replacing the current one if it succeeds.
    Load(Option<PathBuf>),
}

#[derive(Debug)]
pub enum Reply {
    /// The config ran (its settings), or why it didn't.
    Loaded(Result<Settings, String>),
}

/// The front end's handle on the script thread.
pub struct Script {
    requests: Sender<Request>,
    replies: Receiver<Reply>,
    /// Readable while replies are waiting.
    wake: OwnedFd,
    /// When each unanswered request was sent, oldest first.
    sent: VecDeque<Instant>,
    /// The oldest unanswered request has been warned about.
    warned: bool,
}

impl Script {
    pub fn start() -> io::Result<Self> {
        let wake = eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK)?;
        let thread_wake = wake.try_clone()?;
        let (requests, thread_requests) = mpsc::channel();
        let (thread_replies, replies) = mpsc::channel();
        // Detached: a thread stuck in a script can't be joined, and must
        // not keep vellum from exiting. It ends when `requests` is dropped.
        thread::Builder::new()
            .name("vellum-script".into())
            .spawn(move || run(&thread_requests, &thread_replies, &thread_wake))?;
        Ok(Self {
            requests,
            replies,
            wake,
            sent: VecDeque::new(),
            warned: false,
        })
    }

    pub fn send(&mut self, request: Request) {
        // A closed channel means the thread died (a panic, already
        // logged); the request then simply never gets an answer.
        if self.requests.send(request).is_ok() {
            self.sent.push_back(Instant::now());
        }
    }

    /// The replies waiting, after the eventfd was readable.
    pub fn replies(&mut self) -> Vec<Reply> {
        let mut count = [0u8; 8];
        match rustix::io::read(&self.wake, &mut count) {
            Ok(_) | Err(Errno::AGAIN | Errno::INTR) => {}
            Err(e) => tracing::warn!("reading the script thread's eventfd: {e}"),
        }
        let replies: Vec<Reply> = self.replies.try_iter().collect();
        self.answered(replies.len());
        replies
    }

    /// Block for the next reply, up to `timeout` (for the first load, which
    /// the display waits on).
    pub fn wait(&mut self, timeout: Duration) -> Option<Reply> {
        let reply = self.replies.recv_timeout(timeout).ok()?;
        self.answered(1);
        Some(reply)
    }

    fn answered(&mut self, n: usize) {
        for _ in 0..n.min(self.sent.len()) {
            self.sent.pop_front();
            self.warned = false;
        }
    }

    /// When the oldest unanswered request becomes overdue, unless it was
    /// already warned about (for the event loop's timeout).
    pub fn deadline(&self, timeout: Option<Duration>) -> Option<Instant> {
        let t = timeout?;
        if self.warned {
            return None;
        }
        self.sent.front().map(|&sent| sent + t)
    }

    /// Whether the oldest request is overdue and not yet warned about;
    /// true once per request.
    pub fn overdue(&mut self, timeout: Option<Duration>, now: Instant) -> bool {
        let due = self.deadline(timeout).is_some_and(|d| d <= now);
        self.warned |= due;
        due
    }
}

impl AsFd for Script {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.wake.as_fd()
    }
}

/// The thread's loop: serve requests until the front end goes away.
fn run(requests: &Receiver<Request>, replies: &Sender<Reply>, wake: &OwnedFd) {
    // The state bindings and hooks will live in.
    let mut _current: Option<mlua::Lua> = None;
    for request in requests {
        let reply = match request {
            Request::Load(path) => {
                let start = Instant::now();
                let result = api::load(path.as_deref()).map(|loaded| {
                    _current = Some(loaded.lua);
                    loaded.settings
                });
                match &result {
                    Ok(_) => {
                        tracing::info!(config = ?path, time = ?start.elapsed(), "config loaded")
                    }
                    Err(e) => tracing::warn!(config = ?path, "config failed: {e}"),
                }
                Reply::Loaded(result)
            }
        };
        if replies.send(reply).is_err() {
            return;
        }
        if let Err(e) = rustix::io::write(wake, &1u64.to_ne_bytes()) {
            tracing::warn!("waking the event loop: {e}");
        }
    }
}
