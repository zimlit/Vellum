//! Building replies to queries: the parts that don't depend on terminal
//! state (formats, encodings, the capability table).

use super::palette::Rgb;

/// `ESC \`, which ends DCS and OSC replies unless the query used BEL.
pub const ST: &[u8] = b"\x1b\\";

/// Primary device attributes (DA1): a VT220 (62) with ANSI color (22).
pub const DA1: &[u8] = b"\x1b[?62;22c";

/// Most saved keyboard modes per screen (kitty protocol); the oldest is
/// forgotten when a push goes over.
pub const KEYBOARD_STACK: usize = 16;

/// The kitty keyboard protocol's flag bits that exist.
pub const KEYBOARD_FLAGS: u8 = 0b1_1111;

/// Longest DCS request kept (XTGETTCAP, DECRQSS); longer ones are dropped.
pub const MAX_DCS: usize = 4096;

/// Secondary device attributes (DA2): terminal type 1 (VT220, as foot and
/// others say), our version as `MMmmpp`, no ROM cartridge.
pub fn da2() -> Vec<u8> {
    let part = |s: &str| s.parse::<u32>().unwrap_or(0);
    let version = part(env!("CARGO_PKG_VERSION_MAJOR")) * 10000
        + part(env!("CARGO_PKG_VERSION_MINOR")) * 100
        + part(env!("CARGO_PKG_VERSION_PATCH"));
    format!("\x1b[>1;{version};0c").into_bytes()
}

/// XTVERSION: the name and version, as `name(version)` like xterm and kitty.
pub fn xtversion() -> Vec<u8> {
    let mut out = format!("\x1bP>|vellum({})", env!("CARGO_PKG_VERSION")).into_bytes();
    out.extend_from_slice(ST);
    out
}

/// An OSC color report: `OSC <prefix>;rgb:rrrr/gggg/bbbb` with each 8-bit
/// component scaled to 16 bits as xterm reports them, ended like the query.
pub fn osc_color(prefix: &str, rgb: Rgb, bell: bool) -> Vec<u8> {
    let [r, g, b] = rgb.map(|c| u16::from(c) * 0x101);
    let mut out = format!("\x1b]{prefix};rgb:{r:04x}/{g:04x}/{b:04x}").into_bytes();
    out.extend_from_slice(if bell { b"\x07" } else { ST });
    out
}

/// Whether `bg` is the darker of the two, for the color scheme report.
pub fn is_dark(fg: Rgb, bg: Rgb) -> bool {
    // Rec. 709 luma, near enough for "is the background dark".
    let luma = |[r, g, b]: Rgb| 2126 * u32::from(r) + 7152 * u32::from(g) + 722 * u32::from(b);
    luma(bg) <= luma(fg)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect()
}

fn hex_decode(hex: &[u8]) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    hex.chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect()
}

/// terminfo capabilities answered through XTGETTCAP: name and value, with
/// `None` for booleans. String values are the expanded bytes (real ESC),
/// as xterm sends them.
pub(crate) const CAPABILITIES: &[(&str, Option<&str>)] = &[
    // The session server falls back to xterm-256color if it can't install
    // the entry; this still names the one that describes vellum.
    ("TN", Some(crate::terminfo::NAME)),
    ("name", Some(crate::terminfo::NAME)),
    ("Co", Some("256")),
    ("colors", Some("256")),
    // Direct color: both spellings programs look for.
    ("RGB", None),
    ("Tc", None),
    // Styled underlines (`4:n`) and their color (58).
    ("Smulx", Some("\x1b[4:%p1%dm")),
    (
        "Setulc",
        Some("\x1b[58:2::%p1%{65536}%/%d:%p1%{256}%/%{255}%&%d:%p1%{255}%&%dm"),
    ),
    // Cursor style (DECSCUSR) and its reset.
    ("Ss", Some("\x1b[%p1%d q")),
    ("Se", Some("\x1b[0 q")),
    // Synchronized output (2026).
    ("Sync", Some("\x1b[?2026%?%p1%{1}%-%tl%eh%;")),
];

/// The replies to XTGETTCAP (`DCS + q <hex name>;<hex name>... ST`): one
/// `DCS 1 + r <hex name>[=<hex value>] ST` per known name and
/// `DCS 0 + r <hex name> ST` per unknown one, in order.
pub fn xtgettcap(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for hex in data.split(|&b| b == b';') {
        let found = hex_decode(hex).and_then(|name| {
            CAPABILITIES
                .iter()
                .find(|(n, _)| n.as_bytes() == name.as_slice())
        });
        match found {
            Some((_, value)) => {
                out.extend_from_slice(b"\x1bP1+r");
                out.extend_from_slice(hex);
                if let Some(v) = value {
                    out.push(b'=');
                    out.extend_from_slice(hex_encode(v.as_bytes()).as_bytes());
                }
            }
            None => {
                out.extend_from_slice(b"\x1bP0+r");
                out.extend_from_slice(hex);
            }
        }
        out.extend_from_slice(ST);
    }
    out
}

/// The kitty keyboard protocol's mode for one screen: the current flags
/// and the ones saved by pushes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyboardModes {
    pub flags: u8,
    saved: Vec<u8>,
}

impl KeyboardModes {
    /// `CSI > flags u`: save the current flags and use `flags`.
    pub fn push(&mut self, flags: u8) {
        if self.saved.len() == KEYBOARD_STACK {
            self.saved.remove(0);
        }
        self.saved.push(self.flags);
        self.flags = flags & KEYBOARD_FLAGS;
    }

    /// `CSI < n u`: go back `n` pushes. Popping more than were pushed
    /// leaves all flags off.
    pub fn pop(&mut self, n: u16) {
        for _ in 0..n {
            match self.saved.pop() {
                Some(f) => self.flags = f,
                None => {
                    self.flags = 0;
                    break;
                }
            }
        }
    }

    /// `CSI = flags ; mode u`: replace (1), add (2) or remove (3) bits of
    /// the current flags. Returns false for an unknown mode.
    pub fn set(&mut self, flags: u8, mode: u16) -> bool {
        let flags = flags & KEYBOARD_FLAGS;
        match mode {
            1 => self.flags = flags,
            2 => self.flags |= flags,
            3 => self.flags &= !flags,
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        assert_eq!(hex_encode(b"TN"), "544E");
        assert_eq!(hex_decode(b"544e").unwrap(), b"TN");
        assert_eq!(hex_decode(b"544"), None, "odd length");
        assert_eq!(hex_decode(b"zz"), None);
    }

    #[test]
    fn osc_color_scales_to_16_bits() {
        assert_eq!(
            osc_color("11", [0x12, 0xab, 0xff], true),
            b"\x1b]11;rgb:1212/abab/ffff\x07"
        );
        assert_eq!(
            osc_color("4;1", [0, 0, 0], false),
            b"\x1b]4;1;rgb:0000/0000/0000\x1b\\"
        );
    }

    #[test]
    fn dark_and_light_schemes() {
        assert!(is_dark([0xd8; 3], [0x12; 3]));
        assert!(!is_dark([0; 3], [0xff; 3]));
    }

    #[test]
    fn version_replies() {
        let v = String::from_utf8(da2()).unwrap();
        assert!(v.starts_with("\x1b[>1;") && v.ends_with(";0c"), "{v:?}");
        let v = String::from_utf8(xtversion()).unwrap();
        assert!(
            v.starts_with("\x1bP>|vellum(") && v.ends_with(")\x1b\\"),
            "{v:?}"
        );
    }

    #[test]
    fn keyboard_modes_stack() {
        let mut k = KeyboardModes::default();
        k.push(1);
        k.push(0b11);
        assert_eq!(k.flags, 0b11);
        assert!(k.set(0b100, 2));
        assert_eq!(k.flags, 0b111);
        assert!(k.set(0b1, 3));
        assert_eq!(k.flags, 0b110);
        assert!(!k.set(1, 7));
        k.pop(1);
        assert_eq!(k.flags, 1);
        k.pop(5);
        assert_eq!(k.flags, 0, "popping past the bottom");
        k.push(0xff);
        assert_eq!(k.flags, KEYBOARD_FLAGS, "unknown bits dropped");
    }

    #[test]
    fn keyboard_stack_forgets_the_oldest() {
        let mut k = KeyboardModes::default();
        for f in 1..=KEYBOARD_STACK as u8 + 1 {
            k.push(f);
        }
        // Pushed 1..=17 over the initial 0: 0 was forgotten, so popping
        // everything ends at 1, then off.
        k.pop(KEYBOARD_STACK as u16);
        assert_eq!(k.flags, 1);
        k.pop(1);
        assert_eq!(k.flags, 0);
    }
}
