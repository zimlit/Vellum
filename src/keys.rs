//! Key encoding: what a key press sends to the program.
//!
//! Two encodings:
//! - **Legacy** (xterm): text as typed, Ctrl combinations as C0 controls,
//!   Alt as an ESC prefix, and escape sequences for the other keys, with
//!   modifiers as an extra parameter (`CSI 1;5A`). Cursor keys and the
//!   keypad follow DECCKM and DECKPAM. Releases aren't sent.
//! - The **kitty keyboard protocol**
//!   (<https://sw.kovidgoyal.net/kitty/keyboard-protocol/>), once a program
//!   turns it on with `CSI > flags u`. Every key and modifier combination
//!   gets an unambiguous `CSI ... u` form, and on request, releases,
//!   repeats, alternate keys and the text a key typed.
//!
//! The functions here are pure: the input module (step 6C) turns libinput
//! and xkb events into [`KeyEvent`]s, and the terminal provides the
//! [`KeyModes`].

use std::fmt::Write;

use xkbcommon::xkb::Keysym;

use crate::term::Terminal;

/// Kitty keyboard protocol flags (`CSI > flags u`).
pub mod flags {
    /// Use `CSI u` where the legacy encoding is ambiguous (Esc, Alt and
    /// Ctrl combinations, keypad keys that don't type text).
    pub const DISAMBIGUATE: u8 = 1;
    /// Report repeats and releases too.
    pub const EVENT_TYPES: u8 = 2;
    /// Add the shifted key and the key in the US layout.
    pub const ALTERNATE_KEYS: u8 = 4;
    /// Every key as `CSI u`, text and modifier keys included.
    pub const ALL_KEYS: u8 = 8;
    /// Add the text the key typed (with [`ALL_KEYS`]).
    pub const TEXT: u8 = 16;
}

/// Modifier state, with the kitty protocol's bit values (the legacy
/// encoding uses the low four the same way, with Super as xterm's Meta).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Mods(u8);

impl Mods {
    pub const SHIFT: Self = Self(1 << 0);
    pub const ALT: Self = Self(1 << 1);
    pub const CTRL: Self = Self(1 << 2);
    pub const SUPER: Self = Self(1 << 3);
    pub const HYPER: Self = Self(1 << 4);
    pub const META: Self = Self(1 << 5);
    pub const CAPS_LOCK: Self = Self(1 << 6);
    pub const NUM_LOCK: Self = Self(1 << 7);
    /// Caps Lock and Num Lock.
    pub const LOCKS: Self = Self(Self::CAPS_LOCK.0 | Self::NUM_LOCK.0);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl std::ops::BitOr for Mods {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Mods {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KeyKind {
    #[default]
    Press,
    Repeat,
    Release,
}

/// One key event, after the keymap and compose have had their say.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyEvent {
    /// The keysym with the modifiers applied (`A` for Shift+a,
    /// `ISO_Left_Tab` for Shift+Tab).
    pub sym: Keysym,
    /// The key's keysym with no modifiers (level 1 of the active layout):
    /// what the kitty protocol identifies the key by.
    pub base: Keysym,
    /// The same key in the US layout, when the active layout differs. The
    /// kitty protocol reports it, and the legacy encoding falls back to it
    /// for Ctrl combinations on non-Latin layouts.
    pub us: Option<Keysym>,
    /// The text the key types, as if Ctrl and Alt weren't held (the
    /// encoder decides what they do). Empty for keys that type nothing,
    /// and for keys consumed by a compose sequence in progress.
    pub text: String,
    pub mods: Mods,
    pub kind: KeyKind,
}

impl KeyEvent {
    /// A press of a key that types its keysym's character (tests, and
    /// simple callers).
    pub fn press(sym: Keysym, mods: Mods) -> Self {
        Self {
            sym,
            base: sym,
            us: None,
            text: sym
                .key_char()
                .filter(|c| !c.is_control())
                .map(String::from)
                .unwrap_or_default(),
            mods,
            kind: KeyKind::Press,
        }
    }
}

/// The terminal state that changes what keys send.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyModes {
    /// DECCKM: cursor keys send `SS3 x`.
    pub cursor_keys_app: bool,
    /// DECKPAM: the keypad sends `SS3 x`.
    pub keypad_app: bool,
    /// LNM: Enter sends CR LF.
    pub newline: bool,
    /// Kitty keyboard protocol flags; 0 means the legacy encoding.
    pub kitty: u8,
}

impl KeyModes {
    pub fn of(term: &Terminal) -> Self {
        let m = term.modes();
        Self {
            cursor_keys_app: m.cursor_keys_app,
            keypad_app: m.keypad_app,
            newline: m.newline,
            kitty: term.keyboard_flags(),
        }
    }
}

/// The bytes to send for `event`, or `None` if it sends nothing.
pub fn encode(event: &KeyEvent, modes: &KeyModes) -> Option<Vec<u8>> {
    if modes.kitty == 0 {
        legacy(event, modes)
    } else {
        kitty(event, modes)
    }
}

const ESC: u8 = 0x1b;

// Legacy (xterm).

fn legacy(event: &KeyEvent, modes: &KeyModes) -> Option<Vec<u8>> {
    if event.kind == KeyKind::Release {
        return None;
    }
    let mods = event.mods;
    let alt = mods.contains(Mods::ALT);
    // The modifier parameter: 1 + Shift/Alt/Ctrl/Super bits.
    let param = 1 + (mods.bits() & 0b1111);
    // The same with Shift added, for F13–F24.
    let shifted = 1 + (mods.bits() & 0b1111 | Mods::SHIFT.bits());
    let prefixed = |bytes: &[u8]| {
        let mut out = Vec::with_capacity(bytes.len() + 1);
        if alt {
            out.push(ESC);
        }
        out.extend_from_slice(bytes);
        out
    };

    if event.sym == Keysym::BackSpace && mods.contains(Mods::CTRL) {
        return Some(prefixed(b"\x08"));
    }
    if let Some(key) = legacy_key(event.sym, modes) {
        return Some(match key {
            Legacy::Bytes(b) => prefixed(b),
            // Plain: `SS3 x` in application mode for cursor-like keys.
            Legacy::Letter(c, app) if param == 1 => {
                let intro = if app { b'O' } else { b'[' };
                vec![ESC, intro, c]
            }
            Legacy::Letter(c, _) => format!("\x1b[1;{param}{}", char::from(c)).into_bytes(),
            Legacy::Tilde(n) if param == 1 => format!("\x1b[{n}~").into_bytes(),
            Legacy::Tilde(n) => format!("\x1b[{n};{param}~").into_bytes(),
            Legacy::Shifted(c) => format!("\x1b[1;{shifted}{}", char::from(c)).into_bytes(),
            Legacy::ShiftedTilde(n) => format!("\x1b[{n};{shifted}~").into_bytes(),
        });
    }

    // Text keys.
    let mut chars = event.text.chars();
    let (first, single) = (chars.next()?, chars.next().is_none());
    if mods.contains(Mods::CTRL) && single {
        let control = control_char(first)
            .or_else(|| event.us.and_then(|s| s.key_char()).and_then(control_char));
        if let Some(c) = control {
            return Some(prefixed(&[c]));
        }
    }
    Some(prefixed(event.text.as_bytes()))
}

enum Legacy {
    /// Fixed bytes (Alt adds an ESC prefix).
    Bytes(&'static [u8]),
    /// `CSI x`, or `SS3 x` when the flag says so; `CSI 1;m x` modified.
    Letter(u8, bool),
    /// `CSI n ~`, or `CSI n;m ~` modified.
    Tilde(u16),
    /// F13–F24: F1–F12 with Shift, as xterm sends them.
    Shifted(u8),
    ShiftedTilde(u16),
}

fn legacy_key(sym: Keysym, modes: &KeyModes) -> Option<Legacy> {
    use Legacy::*;
    let cursor = modes.cursor_keys_app;
    let kp = modes.keypad_app;
    Some(match sym {
        Keysym::Return => Bytes(if modes.newline { b"\r\n" } else { b"\r" }),
        Keysym::KP_Enter if kp => Bytes(b"\x1bOM"),
        Keysym::KP_Enter => Bytes(if modes.newline { b"\r\n" } else { b"\r" }),
        Keysym::Tab | Keysym::KP_Tab => Bytes(b"\t"),
        Keysym::ISO_Left_Tab => Bytes(b"\x1b[Z"),
        Keysym::BackSpace => Bytes(b"\x7f"),
        Keysym::Escape => Bytes(b"\x1b"),

        Keysym::Up | Keysym::KP_Up => Letter(b'A', cursor),
        Keysym::Down | Keysym::KP_Down => Letter(b'B', cursor),
        Keysym::Right | Keysym::KP_Right => Letter(b'C', cursor),
        Keysym::Left | Keysym::KP_Left => Letter(b'D', cursor),
        Keysym::Home | Keysym::KP_Home => Letter(b'H', cursor),
        Keysym::End | Keysym::KP_End => Letter(b'F', cursor),
        Keysym::KP_Begin => Letter(b'E', cursor),
        Keysym::Insert | Keysym::KP_Insert => Tilde(2),
        Keysym::Delete | Keysym::KP_Delete => Tilde(3),
        Keysym::Page_Up | Keysym::KP_Page_Up => Tilde(5),
        Keysym::Page_Down | Keysym::KP_Page_Down => Tilde(6),

        Keysym::F1 | Keysym::KP_F1 => Letter(b'P', true),
        Keysym::F2 | Keysym::KP_F2 => Letter(b'Q', true),
        Keysym::F3 | Keysym::KP_F3 => Letter(b'R', true),
        Keysym::F4 | Keysym::KP_F4 => Letter(b'S', true),
        s if (Keysym::F5..=Keysym::F12).contains(&s) => Tilde(F5_TO_F12[fkey(s) - 5]),
        s if (Keysym::F13..=Keysym::F16).contains(&s) => Shifted(b"PQRS"[fkey(s) - 13]),
        s if (Keysym::F17..=Keysym::F24).contains(&s) => ShiftedTilde(F5_TO_F12[fkey(s) - 17]),

        // The keypad in application mode (with Num Lock, xkb gives these;
        // without it, the cursor keys above).
        s if kp => Bytes(match s {
            Keysym::KP_0 => b"\x1bOp",
            Keysym::KP_1 => b"\x1bOq",
            Keysym::KP_2 => b"\x1bOr",
            Keysym::KP_3 => b"\x1bOs",
            Keysym::KP_4 => b"\x1bOt",
            Keysym::KP_5 => b"\x1bOu",
            Keysym::KP_6 => b"\x1bOv",
            Keysym::KP_7 => b"\x1bOw",
            Keysym::KP_8 => b"\x1bOx",
            Keysym::KP_9 => b"\x1bOy",
            Keysym::KP_Decimal => b"\x1bOn",
            Keysym::KP_Separator => b"\x1bOl",
            Keysym::KP_Add => b"\x1bOk",
            Keysym::KP_Subtract => b"\x1bOm",
            Keysym::KP_Multiply => b"\x1bOj",
            Keysym::KP_Divide => b"\x1bOo",
            Keysym::KP_Equal => b"\x1bOX",
            _ => return None,
        }),
        _ => return None,
    })
}

/// The `~` codes for F5–F12 (the gaps are historical).
const F5_TO_F12: [u16; 8] = [15, 17, 18, 19, 20, 21, 23, 24];

/// The number of a function key keysym (F1 = 1).
fn fkey(sym: Keysym) -> usize {
    (sym.raw() - Keysym::F1.raw()) as usize + 1
}

/// What Ctrl turns a character into, as xterm does it.
fn control_char(c: char) -> Option<u8> {
    Some(match c {
        '@' | ' ' | '2' => 0x00,
        'a'..='z' => c as u8 - b'a' + 1,
        'A'..='Z' => c as u8 - b'A' + 1,
        '[' | '3' => 0x1b,
        '\\' | '4' => 0x1c,
        ']' | '5' => 0x1d,
        '^' | '~' | '6' => 0x1e,
        '_' | '/' | '-' | '7' => 0x1f,
        '?' | '8' => 0x7f,
        _ => return None,
    })
}

// Kitty.

/// How the kitty protocol writes a key that isn't text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Functional {
    /// `CSI 1;mods X` (`CSI X` unmodified).
    Letter(u8),
    /// `CSI n;mods ~`.
    Tilde(u16),
    /// `CSI code;mods u`.
    Code(u32),
}

fn functional(sym: Keysym) -> Option<Functional> {
    use Functional::*;
    // Private-use codes, in the order of the protocol's table.
    let code = |n: u32| Code(57344 + n);
    Some(match sym {
        Keysym::Escape => Code(27),
        Keysym::Return => Code(13),
        Keysym::Tab | Keysym::ISO_Left_Tab => Code(9),
        Keysym::BackSpace => Code(127),
        Keysym::Insert => Tilde(2),
        Keysym::Delete => Tilde(3),
        Keysym::Left => Letter(b'D'),
        Keysym::Right => Letter(b'C'),
        Keysym::Up => Letter(b'A'),
        Keysym::Down => Letter(b'B'),
        Keysym::Page_Up => Tilde(5),
        Keysym::Page_Down => Tilde(6),
        Keysym::Home => Letter(b'H'),
        Keysym::End => Letter(b'F'),
        Keysym::Caps_Lock => code(14),
        Keysym::Scroll_Lock => code(15),
        Keysym::Num_Lock => code(16),
        Keysym::Print => code(17),
        Keysym::Pause => code(18),
        Keysym::Menu => code(19),
        Keysym::F1 => Letter(b'P'),
        Keysym::F2 => Letter(b'Q'),
        // Not `CSI R`, which is also a cursor position report.
        Keysym::F3 => Tilde(13),
        Keysym::F4 => Letter(b'S'),
        s if (Keysym::F5..=Keysym::F12).contains(&s) => Tilde(F5_TO_F12[fkey(s) - 5]),
        s if (Keysym::F13..=Keysym::F35).contains(&s) => code(32 + (fkey(s) as u32 - 13)),
        s if (Keysym::KP_0..=Keysym::KP_9).contains(&s) => {
            code(55 + (s.raw() - Keysym::KP_0.raw()))
        }
        Keysym::KP_Decimal => code(65),
        Keysym::KP_Divide => code(66),
        Keysym::KP_Multiply => code(67),
        Keysym::KP_Subtract => code(68),
        Keysym::KP_Add => code(69),
        Keysym::KP_Enter => code(70),
        Keysym::KP_Equal => code(71),
        Keysym::KP_Separator => code(72),
        Keysym::KP_Left => code(73),
        Keysym::KP_Right => code(74),
        Keysym::KP_Up => code(75),
        Keysym::KP_Down => code(76),
        Keysym::KP_Page_Up => code(77),
        Keysym::KP_Page_Down => code(78),
        Keysym::KP_Home => code(79),
        Keysym::KP_End => code(80),
        Keysym::KP_Insert => code(81),
        Keysym::KP_Delete => code(82),
        Keysym::KP_Begin => Letter(b'E'),
        Keysym::XF86_AudioPlay => code(84),
        Keysym::XF86_AudioPause => code(85),
        Keysym::XF86_AudioStop => code(88),
        Keysym::XF86_AudioForward => code(89),
        Keysym::XF86_AudioRewind => code(90),
        Keysym::XF86_AudioNext => code(91),
        Keysym::XF86_AudioPrev => code(92),
        Keysym::XF86_AudioRecord => code(93),
        Keysym::XF86_AudioLowerVolume => code(94),
        Keysym::XF86_AudioRaiseVolume => code(95),
        Keysym::XF86_AudioMute => code(96),
        _ => return modifier_code(sym).map(Code),
    })
}

/// Modifier keys' codes; also what makes a key a modifier key.
fn modifier_code(sym: Keysym) -> Option<u32> {
    Some(
        57344
            + match sym {
                Keysym::Shift_L => 97,
                Keysym::Control_L => 98,
                Keysym::Alt_L => 99,
                Keysym::Super_L => 100,
                Keysym::Hyper_L => 101,
                Keysym::Meta_L => 102,
                Keysym::Shift_R => 103,
                Keysym::Control_R => 104,
                Keysym::Alt_R => 105,
                Keysym::Super_R => 106,
                Keysym::Hyper_R => 107,
                Keysym::Meta_R => 108,
                Keysym::ISO_Level3_Shift => 109,
                Keysym::ISO_Level5_Shift => 110,
                _ => return None,
            },
    )
}

/// Keys that only change state: modifiers and locks. They send nothing
/// unless every key is reported.
fn is_state_key(sym: Keysym) -> bool {
    modifier_code(sym).is_some()
        || matches!(
            sym,
            Keysym::Caps_Lock | Keysym::Num_Lock | Keysym::Scroll_Lock
        )
}

/// The key's Unicode code point for `CSI code u`: its unshifted
/// character, lowercase.
fn key_code(sym: Keysym) -> Option<u32> {
    let c = sym.key_char().filter(|c| !c.is_control())?;
    let mut lower = c.to_lowercase();
    match (lower.next(), lower.next()) {
        (Some(l), None) => Some(l.into()),
        _ => Some(c.into()),
    }
}

fn is_text(text: &str) -> bool {
    !text.is_empty() && !text.chars().any(char::is_control)
}

fn kitty(event: &KeyEvent, modes: &KeyModes) -> Option<Vec<u8>> {
    let f = modes.kitty;
    let all = f & flags::ALL_KEYS != 0;
    let release = event.kind == KeyKind::Release;
    if release && f & flags::EVENT_TYPES == 0 {
        return None;
    }
    let func = functional(event.sym);
    // Keys that type a character: text keys, and keypad digits and
    // operators (with Num Lock).
    let text_key = func.is_none() || (event.sym.is_keypad_key() && key_code(event.sym).is_some());
    // "Lock modifiers are not reported for text producing keys, to keep
    // them usable in legacy programs", unless every key is reported.
    let mods = if all || !text_key {
        event.mods
    } else {
        event.mods.without(Mods::LOCKS)
    };
    let unlocked = event.mods.without(Mods::LOCKS);

    if !all {
        if is_state_key(event.sym) {
            return None;
        }
        let enter_tab_bs = matches!(
            event.sym,
            Keysym::Return | Keysym::Tab | Keysym::ISO_Left_Tab | Keysym::BackSpace
        );
        // Text keys typed with at most Shift: plain text, so no repeat or
        // release events ("events are not supported for them").
        let types_text = text_key && unlocked.without(Mods::SHIFT).is_empty();
        if release && (enter_tab_bs || types_text) {
            return None;
        }
        if !release {
            // Enter, Tab and Backspace "still generate the same bytes as
            // in legacy mode" when unmodified.
            if unlocked.is_empty() && enter_tab_bs {
                return legacy(event, &KeyModes { kitty: 0, ..*modes });
            }
            // Unmodified cursor keys keep DECCKM's form.
            if let Some(Functional::Letter(c @ (b'A' | b'B' | b'C' | b'D' | b'H' | b'F'))) = func
                && unlocked.is_empty()
                && modes.cursor_keys_app
            {
                return Some(vec![ESC, b'O', c]);
            }
            if types_text && is_text(&event.text) {
                return Some(event.text.clone().into_bytes());
            }
        }
    }

    let event_type = match event.kind {
        KeyKind::Press => None,
        KeyKind::Repeat if f & flags::EVENT_TYPES != 0 => Some(2),
        KeyKind::Repeat => None,
        KeyKind::Release => Some(3),
    };

    let (key, final_byte) = match func {
        Some(Functional::Letter(c)) => (String::from("1"), c),
        Some(Functional::Tilde(n)) => (n.to_string(), b'~'),
        Some(Functional::Code(n)) => (n.to_string(), b'u'),
        None => {
            let code = key_code(event.base).or_else(|| key_code(event.sym))?;
            let mut key = code.to_string();
            if f & flags::ALTERNATE_KEYS != 0 {
                let shifted = mods
                    .contains(Mods::SHIFT)
                    .then(|| key_code_exact(event.sym))
                    .flatten()
                    .filter(|&s| s != code);
                let us = event.us.and_then(key_code).filter(|&u| u != code);
                match (shifted, us) {
                    (None, None) => {}
                    (Some(s), None) => write!(key, ":{s}").unwrap(),
                    (s, Some(u)) => {
                        write!(key, ":{}:{u}", s.map(|s| s.to_string()).unwrap_or_default())
                            .unwrap()
                    }
                }
            }
            (key, b'u')
        }
    };

    // The text, only with every key reported, and only text typed with at
    // most Shift (Ctrl and Alt combinations don't type).
    let text = (all
        && f & flags::TEXT != 0
        && !release
        && final_byte == b'u'
        && mods.without(Mods::SHIFT | Mods::LOCKS).is_empty()
        && is_text(&event.text))
    .then(|| {
        event
            .text
            .chars()
            .map(|c| u32::from(c).to_string())
            .collect::<Vec<_>>()
            .join(":")
    });

    let mod_param = 1 + u32::from(mods.bits());
    let mut out = String::from("\x1b[");
    let plain = mod_param == 1 && event_type.is_none() && text.is_none();
    if final_byte != b'u' && plain && key == "1" {
        // `CSI A`, not `CSI 1A`.
    } else {
        out.push_str(&key);
    }
    if !plain {
        out.push(';');
        if mod_param != 1 || event_type.is_some() {
            write!(out, "{mod_param}").unwrap();
        }
        if let Some(t) = event_type {
            write!(out, ":{t}").unwrap();
        }
        if let Some(text) = text {
            write!(out, ";{text}").unwrap();
        }
    }
    out.push(char::from(final_byte));
    Some(out.into_bytes())
}

/// A keysym's code point, without lowercasing (for the shifted key).
fn key_code_exact(sym: Keysym) -> Option<u32> {
    sym.key_char().filter(|c| !c.is_control()).map(u32::from)
}

#[cfg(test)]
mod tests;
