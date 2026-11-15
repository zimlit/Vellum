//! Command-line arguments.
//!
//! Usage: vellum [--probe] [--seconds N] [--font PATTERN]
//!   --probe          open the card directly (no seat, no DRM master), render
//!                    one frame without showing it, and print what was chosen
//!   --seconds N      how long to run; 0 = until a signal (default 10)
//!   --font PATTERN   fontconfig pattern (default `monospace:size=11`); point
//!                    sizes use the monitor's DPI
//!
//! Internal: `vellum --session-server PATH` is the session server, started
//! by vellum itself with the listening socket on fd 3.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

pub const DEFAULT_FONT: &str = "monospace:size=11";

pub struct Args {
    pub probe: bool,
    pub seconds: u64,
    pub font: String,
    /// Run as the PTY session server on this socket.
    pub session_server: Option<PathBuf>,
}

impl Args {
    pub fn parse() -> Result<Self> {
        Self::parse_from(std::env::args().skip(1))
    }

    /// Parse arguments (without the program name).
    pub fn parse_from(it: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = Args {
            probe: false,
            seconds: 10,
            font: DEFAULT_FONT.to_owned(),
            session_server: None,
        };
        let mut it = it.into_iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--probe" => args.probe = true,
                "--seconds" => {
                    let n = it.next().context("--seconds needs a value")?;
                    args.seconds = n.parse().context("--seconds expects an integer")?;
                }
                "--font" => {
                    args.font = it.next().context("--font needs a fontconfig pattern")?;
                }
                "--session-server" => {
                    let path = it.next().context("--session-server needs a socket path")?;
                    args.session_server = Some(path.into());
                }
                other => bail!("unknown argument {other:?}"),
            }
        }
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args> {
        Args::parse_from(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn args_defaults() {
        let a = args(&[]).unwrap();
        assert!(!a.probe);
        assert_eq!(a.seconds, 10);
        assert_eq!(a.session_server, None);
        assert_eq!(a.font, DEFAULT_FONT);
    }

    #[test]
    fn args_flags() {
        let a = args(&["--probe", "--seconds", "0"]).unwrap();
        assert!(a.probe);
        assert_eq!(a.seconds, 0);
    }

    #[test]
    fn args_errors() {
        assert!(args(&["--seconds"]).is_err(), "missing value");
        assert!(args(&["--seconds", "ten"]).is_err(), "not a number");
        assert!(args(&["--seconds", "-1"]).is_err(), "negative");
        assert!(args(&["--bogus"]).is_err(), "unknown flag");
        assert!(args(&["--session-server"]).is_err(), "missing path");
        assert!(args(&["--font"]).is_err(), "missing pattern");
    }

    #[test]
    fn args_font() {
        assert_eq!(
            args(&["--font", "Iosevka:size=14"]).unwrap().font,
            "Iosevka:size=14"
        );
    }

    #[test]
    fn args_session_server() {
        let a = args(&["--session-server", "/run/s.sock"]).unwrap();
        assert_eq!(a.session_server, Some("/run/s.sock".into()));
    }
}
