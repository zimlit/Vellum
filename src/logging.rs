//! Logging setup (tracing).
//!
//! While vellum owns the display the VT is in graphics mode, so anything
//! written to it is invisible (and shows up garbled once text mode returns).
//! Logs therefore always go to a file, and to stderr only when stderr is not
//! a VT (SSH session, pipe, redirect).
//!
//! When stderr *is* a VT, fd 2 itself is pointed at the log file, so output
//! from C libraries that write to stderr directly (Mesa, libseat fallbacks)
//! lands in the log too instead of on the console. The original VT is kept
//! as [`LogTargets::console`] for messages meant for the user after the
//! console is back in text mode.
//!
//! Environment:
//! - `VELLUM_LOG`: filter directives (`EnvFilter` syntax, e.g. `debug` or
//!   `info,vellum::kms=trace`). Default `info`, `info,vellum=debug` in debug
//!   builds.
//! - `VELLUM_LOG_FILE`: log file path. Default
//!   `$XDG_STATE_HOME/vellum/vellum.log` (`~/.local/state/vellum/vellum.log`).
//!   The previous run's log is kept as `<file>.old`.
//!
//! Messages from libseat arrive through the `log` crate and are bridged in.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, IsTerminal};
use std::os::fd::AsFd;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

/// Linux TTY_MAJOR: /dev/tty0../dev/tty63 (VTs) are char devices 4:0..4:63.
const TTY_MAJOR: u32 = 4;

/// Where logs are going.
pub struct LogTargets {
    /// The log file, if one could be opened.
    pub file: Option<PathBuf>,
    /// The VT stderr originally pointed to, when it was one. Logs are kept
    /// off it; write here only once the VT is back in text mode (after the
    /// seat is released).
    pub console: Option<File>,
}

/// Install the global subscriber and panic hook. Problems opening the log
/// file are logged (to stderr, if that is usable) rather than fatal.
pub fn init() -> LogTargets {
    let filter = EnvFilter::try_from_env("VELLUM_LOG").unwrap_or_else(|_| {
        EnvFilter::new(if cfg!(debug_assertions) {
            "info,vellum=debug"
        } else {
            "info"
        })
    });

    let on_vt = is_vt(io::stderr());
    let console = on_vt
        .then(|| io::stderr().as_fd().try_clone_to_owned().ok())
        .flatten()
        .map(File::from);

    let path = log_path(|key| std::env::var_os(key));
    let file = path
        .as_ref()
        .map(|p| open_log(p).map_err(|e| (p.clone(), e)));

    // dup2 makes fd 2 share the log file's open file description (and so
    // its offset), so direct stderr writes and tracing's writes interleave
    // instead of overwriting each other.
    let redirect_error = match &file {
        Some(Ok(f)) if on_vt => rustix::stdio::dup2_stderr(f).err(),
        _ => None,
    };

    let (file_layer, file_error) = match file {
        Some(Ok(f)) => (
            Some(fmt::layer().with_ansi(false).with_writer(Mutex::new(f))),
            None,
        ),
        Some(Err(err)) => (None, Some(err)),
        None => (None, None),
    };

    let stderr_layer = (!on_vt).then(|| {
        fmt::layer()
            .with_ansi(io::stderr().is_terminal())
            .with_writer(io::stderr)
    });

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(stderr_layer)
        .init();

    install_panic_hook();

    let file = match (file_error, path) {
        (Some((path, err)), _) => {
            tracing::warn!(path = %path.display(), "could not open log file: {err}");
            None
        }
        (None, None) => {
            tracing::warn!("no log file: neither VELLUM_LOG_FILE, XDG_STATE_HOME nor HOME is set");
            None
        }
        (None, Some(path)) => Some(path),
    };
    if let Some(err) = redirect_error {
        tracing::warn!("could not redirect stderr to the log file: {err}");
    }
    LogTargets { file, console }
}

/// The log file path, from environment variables looked up through `env`
/// (a parameter so tests needn't touch the process environment).
fn log_path(env: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    if let Some(p) = env("VELLUM_LOG_FILE") {
        return Some(PathBuf::from(p));
    }
    let state = env("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| env("HOME").map(|h| PathBuf::from(h).join(".local/state")))?;
    Some(state.join("vellum/vellum.log"))
}

fn open_log(path: &Path) -> io::Result<File> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut old = path.to_path_buf().into_os_string();
    old.push(".old");
    // Fine if there's no previous log.
    let _ = fs::rename(path, old);
    File::create(path)
}

/// Whether `fd` is a virtual terminal (/dev/tty0../dev/tty63).
fn is_vt(fd: impl AsFd) -> bool {
    let Ok(stat) = rustix::fs::fstat(fd) else {
        return false;
    };
    let is_char = rustix::fs::FileType::from_raw_mode(stat.st_mode).is_char_device();
    is_char && rustix::fs::major(stat.st_rdev) == TTY_MAJOR && rustix::fs::minor(stat.st_rdev) < 64
}

/// Log panics through tracing instead of the default hook, which would write
/// a duplicate to stderr. Unwinding continues afterwards, so destructors
/// still restore the display.
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        tracing::error!("{info}\n{backtrace}");
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |key| {
            vars.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| OsString::from(v))
        }
    }

    #[test]
    fn log_path_prefers_explicit_file() {
        let vars = [
            ("VELLUM_LOG_FILE", "/x/custom.log"),
            ("XDG_STATE_HOME", "/state"),
            ("HOME", "/home/u"),
        ];
        assert_eq!(log_path(env(&vars)), Some("/x/custom.log".into()));
    }

    #[test]
    fn log_path_uses_xdg_state_home() {
        let vars = [("XDG_STATE_HOME", "/state"), ("HOME", "/home/u")];
        assert_eq!(
            log_path(env(&vars)),
            Some("/state/vellum/vellum.log".into())
        );
    }

    #[test]
    fn log_path_falls_back_to_home() {
        let vars = [("HOME", "/home/u")];
        assert_eq!(
            log_path(env(&vars)),
            Some("/home/u/.local/state/vellum/vellum.log".into())
        );
    }

    #[test]
    fn log_path_none_without_any_variable() {
        assert_eq!(log_path(env(&[])), None);
    }

    #[test]
    fn open_log_creates_parent_directories() {
        let tmp = TempDir::new();
        let path = tmp.path().join("a/b/vellum.log");
        open_log(&path).unwrap();
        assert!(path.is_file());
    }

    #[test]
    fn open_log_keeps_previous_run_as_old() {
        let tmp = TempDir::new();
        let path = tmp.path().join("vellum.log");
        fs::write(&path, "previous run\n").unwrap();
        fs::write(tmp.path().join("vellum.log.old"), "run before that\n").unwrap();

        open_log(&path).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "");
        assert_eq!(
            fs::read_to_string(tmp.path().join("vellum.log.old")).unwrap(),
            "previous run\n"
        );
    }

    #[test]
    fn pipes_and_other_char_devices_are_not_vts() {
        let (read, write) = io::pipe().unwrap();
        assert!(!is_vt(&read));
        assert!(!is_vt(&write));
        // /dev/null is a char device too (1:3), just not a tty.
        let null = File::open("/dev/null").unwrap();
        assert!(!is_vt(&null));
        let tmp = TempDir::new();
        let file = File::create(tmp.path().join("f")).unwrap();
        assert!(!is_vt(&file));
    }
}
