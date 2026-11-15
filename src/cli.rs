//! Command-line arguments.
//!
//! Usage: vellum [--probe] [--seconds N] [--font PATTERN] [--xkb-* NAME]
//!   --probe          open the card directly (no seat, no DRM master), render
//!                    one frame without showing it, and print what was chosen
//!   --seconds N      how long to run; 0 = until a signal (default 10)
//!   --font PATTERN   fontconfig pattern (default `monospace:size=11`); point
//!                    sizes use the monitor's DPI
//!   --xkb-rules, --xkb-model, --xkb-layout, --xkb-variant, --xkb-options
//!                    the keymap; each defaults to `XKB_DEFAULT_*` from the
//!                    environment, then to the system default (`us`)
//!
//! Internal: `vellum --session-server PATH` is the session server, started
//! by vellum itself with the listening socket on fd 3.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::input::RuleNames;

pub const DEFAULT_FONT: &str = "monospace:size=11";

pub struct Args {
    pub probe: bool,
    pub seconds: u64,
    pub font: String,
    pub keymap: RuleNames,
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
            keymap: RuleNames::default(),
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
                "--xkb-rules" | "--xkb-model" | "--xkb-layout" | "--xkb-variant"
                | "--xkb-options" => {
                    let value = it.next().with_context(|| format!("{arg} needs a value"))?;
                    let k = &mut args.keymap;
                    match arg.as_str() {
                        "--xkb-rules" => k.rules = value,
                        "--xkb-model" => k.model = value,
                        "--xkb-layout" => k.layout = value,
                        "--xkb-variant" => k.variant = value,
                        _ => k.options = Some(value),
                    }
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
        assert_eq!(a.keymap, RuleNames::default());
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
        assert!(args(&["--xkb-layout"]).is_err(), "missing layout");
    }

    #[test]
    fn args_keymap() {
        let a = args(&[
            "--xkb-rules",
            "evdev",
            "--xkb-model",
            "pc104",
            "--xkb-layout",
            "us,de",
            "--xkb-variant",
            "intl,",
            "--xkb-options",
            "ctrl:nocaps",
        ])
        .unwrap();
        assert_eq!(
            a.keymap,
            RuleNames {
                rules: "evdev".into(),
                model: "pc104".into(),
                layout: "us,de".into(),
                variant: "intl,".into(),
                options: Some("ctrl:nocaps".into()),
            }
        );
        // An empty option list is not the same as none given.
        assert_eq!(
            args(&["--xkb-options", ""]).unwrap().keymap.options,
            Some(String::new())
        );
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
