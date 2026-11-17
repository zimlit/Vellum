//! Command-line arguments.
//!
//! Usage: vellum [--probe] [--config PATH | --no-config]
//!               [--font PATTERN] [--xkb-* NAME]
//!   --probe          open the card directly (no seat, no DRM master), render
//!                    one frame without showing it, and print what was chosen
//!   --config PATH    the Lua config to run (default
//!                    `$XDG_CONFIG_HOME/vellum/init.lua`, if it exists)
//!   --no-config      run no config: built-in settings
//!   --font PATTERN   fontconfig pattern (default from the config, else
//!                    `monospace:size=11`); point sizes use the monitor's DPI
//!   --xkb-rules, --xkb-model, --xkb-layout, --xkb-variant, --xkb-options
//!                    the keymap; each defaults to the config, then to
//!                    `XKB_DEFAULT_*` from the environment, then to the
//!                    system default (`us`)
//!
//! Flags given here win over the config.
//!
//! Internal: `vellum --session-server PATH` is the session server, started
//! by vellum itself with the listening socket on fd 3.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::input::RuleNames;

/// Which config to run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ConfigFile {
    /// The usual place, if there's a file there.
    #[default]
    Default,
    /// This file (missing is an error).
    Path(PathBuf),
    /// None (`--no-config`).
    None,
}

pub struct Args {
    pub probe: bool,
    pub config: ConfigFile,
    /// Overrides the config's font.
    pub font: Option<String>,
    /// Non-empty fields override the config's keymap.
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
            config: ConfigFile::Default,
            font: None,
            keymap: RuleNames::default(),
            session_server: None,
        };
        let mut it = it.into_iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--probe" => args.probe = true,
                "--font" => {
                    args.font = Some(it.next().context("--font needs a fontconfig pattern")?);
                }
                "--config" => {
                    let path = it.next().context("--config needs a path")?;
                    args.config = ConfigFile::Path(path.into());
                }
                "--no-config" => args.config = ConfigFile::None,
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
        assert_eq!(a.session_server, None);
        assert_eq!(a.font, None);
        assert_eq!(a.config, ConfigFile::Default);
        assert_eq!(a.keymap, RuleNames::default());
    }

    #[test]
    fn args_flags() {
        assert!(args(&["--probe"]).unwrap().probe);
    }

    #[test]
    fn args_errors() {
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
            args(&["--font", "Iosevka:size=14"])
                .unwrap()
                .font
                .as_deref(),
            Some("Iosevka:size=14")
        );
    }

    #[test]
    fn args_config() {
        assert_eq!(
            args(&["--config", "/x/init.lua"]).unwrap().config,
            ConfigFile::Path("/x/init.lua".into())
        );
        assert_eq!(args(&["--no-config"]).unwrap().config, ConfigFile::None);
        assert!(args(&["--config"]).is_err(), "missing path");
    }

    #[test]
    fn args_session_server() {
        let a = args(&["--session-server", "/run/s.sock"]).unwrap();
        assert_eq!(a.session_server, Some("/run/s.sock".into()));
    }
}
