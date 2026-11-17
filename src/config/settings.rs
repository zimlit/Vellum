//! What the configuration sets, as plain Rust data. Nothing outside
//! [`crate::config`] sees Lua; the script thread hands these over.
//!
//! Read with serde from the table the config built with `vellum.set{...}`:
//! unknown keys and wrong types are errors, and anything not set keeps its
//! default (vellum's built-in behavior).

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use serde::Deserialize;
use serde::de::{self, Deserializer, Visitor};

use crate::cli::Args;
use crate::input::RuleNames;
use crate::term::CursorShape;
use crate::term::palette::{self, Palette, Rgb};

pub const DEFAULT_FONT: &str = "monospace:size=11";

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// fontconfig pattern; point sizes use the monitor's DPI.
    pub font: String,
    pub keymap: RuleNames,
    pub colors: Colors,
    /// Milliseconds before a held key repeats.
    pub repeat_delay: u64,
    /// Repeats per second.
    pub repeat_rate: u32,
    /// Lines of scrollback kept.
    pub scrollback: usize,
    pub cursor_style: CursorStyle,
    /// The program run for a new shell, then its arguments. Empty: `$SHELL`.
    pub shell: Vec<String>,
    /// Extra environment variables for a new shell.
    pub env: HashMap<String, String>,
    /// Seconds before a script that hasn't answered gets a warning;
    /// negative: never.
    pub script_timeout: f64,
}

/// Settings a later `vellum.set` replaces whole, instead of merging its
/// table key by key into what earlier calls set.
pub const ATOMIC_SETTINGS: &[&str] = &["shell"];

impl Default for Settings {
    fn default() -> Self {
        Self {
            font: DEFAULT_FONT.to_owned(),
            keymap: RuleNames::default(),
            colors: Colors::default(),
            repeat_delay: 600,
            repeat_rate: 25,
            scrollback: crate::term::scrollback::DEFAULT_LIMIT,
            cursor_style: CursorStyle::default(),
            shell: Vec::new(),
            env: HashMap::new(),
            script_timeout: 5.0,
        }
    }
}

impl Settings {
    /// Checks types can't express. The error names the setting.
    pub fn validate(&self) -> Result<(), String> {
        if self.repeat_rate == 0 {
            return Err("repeat_rate: must be at least 1".into());
        }
        if self.shell.first().is_some_and(|p| p.is_empty()) {
            return Err("shell[1]: the program can't be empty".into());
        }
        if !self.script_timeout.is_finite() {
            return Err("script_timeout: must be a number of seconds".into());
        }
        Ok(())
    }

    /// Command-line flags win over the config.
    pub fn with_cli(mut self, args: &Args) -> Self {
        if let Some(font) = &args.font {
            self.font.clone_from(font);
        }
        let (k, cli) = (&mut self.keymap, &args.keymap);
        for (field, flag) in [
            (&mut k.rules, &cli.rules),
            (&mut k.model, &cli.model),
            (&mut k.layout, &cli.layout),
            (&mut k.variant, &cli.variant),
        ] {
            if !flag.is_empty() {
                field.clone_from(flag);
            }
        }
        if cli.options.is_some() {
            k.options.clone_from(&cli.options);
        }
        self
    }

    pub fn repeat_delay(&self) -> Duration {
        Duration::from_millis(self.repeat_delay)
    }

    pub fn repeat_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / f64::from(self.repeat_rate.max(1)))
    }

    /// `None`: never warn.
    pub fn script_timeout(&self) -> Option<Duration> {
        (self.script_timeout >= 0.0).then(|| Duration::from_secs_f64(self.script_timeout))
    }
}

/// `colors = { foreground = "#d8d8d8", palette = { [0] = "#000000", ... } }`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Colors {
    pub foreground: Option<Color>,
    pub background: Option<Color>,
    pub cursor: Option<Color>,
    /// Any of the 256 colors, by terminal color number (0 black, 1 red,
    /// ...). `{ [0] = "#000000", "#cc0000", ... }` lists them from 0.
    pub palette: HashMap<u8, Color>,
}

impl Colors {
    /// The built-in palette with these colors over it.
    pub fn palette(&self) -> Palette {
        let mut p = Palette::default();
        for (&i, c) in &self.palette {
            p.colors[usize::from(i)] = c.0;
        }
        for (slot, c) in [
            (&mut p.foreground, self.foreground),
            (&mut p.background, self.background),
            (&mut p.cursor, self.cursor),
        ] {
            if let Some(c) = c {
                *slot = c.0;
            }
        }
        p
    }
}

/// A color written as `"#rrggbb"` (or any form OSC 4 accepts: `#rgb`,
/// `rgb:rr/gg/bb`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(pub Rgb);

impl<'de> Deserialize<'de> for Color {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ColorVisitor;
        impl Visitor<'_> for ColorVisitor {
            type Value = Color;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(r##"a color like "#rrggbb""##)
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<Color, E> {
                palette::parse_color(s.as_bytes())
                    .map(Color)
                    .ok_or_else(|| {
                        E::custom(format!(r##"expected a color like "#rrggbb", got {s:?}"##))
                    })
            }
        }
        d.deserialize_str(ColorVisitor)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CursorStyle {
    #[default]
    Block,
    Underline,
    Bar,
}

impl From<CursorStyle> for CursorShape {
    fn from(s: CursorStyle) -> Self {
        match s {
            CursorStyle::Block => CursorShape::Block,
            CursorStyle::Underline => CursorShape::Underline,
            CursorStyle::Bar => CursorShape::Bar,
        }
    }
}
