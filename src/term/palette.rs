//! The 256-color palette and the default colors, which programs can
//! change with OSC 4 / 10 / 11 / 12.

pub type Rgb = [u8; 3];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Palette {
    pub colors: [Rgb; 256],
    pub foreground: Rgb,
    pub background: Rgb,
    pub cursor: Rgb,
}

/// The first 16 colors: the Tango palette (readable on dark and light).
const BASE: [Rgb; 16] = [
    [0x2e, 0x34, 0x36],
    [0xcc, 0x00, 0x00],
    [0x4e, 0x9a, 0x06],
    [0xc4, 0xa0, 0x00],
    [0x34, 0x65, 0xa4],
    [0x75, 0x50, 0x7b],
    [0x06, 0x98, 0x9a],
    [0xd3, 0xd7, 0xcf],
    [0x55, 0x57, 0x53],
    [0xef, 0x29, 0x29],
    [0x8a, 0xe2, 0x34],
    [0xfc, 0xe9, 0x4f],
    [0x72, 0x9f, 0xcf],
    [0xad, 0x7f, 0xa8],
    [0x34, 0xe2, 0xe2],
    [0xee, 0xee, 0xec],
];

impl Default for Palette {
    fn default() -> Self {
        Self {
            colors: std::array::from_fn(default_color),
            foreground: [0xd8, 0xd8, 0xd8],
            background: [0x12, 0x12, 0x14],
            cursor: [0xd8, 0xd8, 0xd8],
        }
    }
}

/// Palette entry `i` before any changes: the 16 base colors, then the
/// 6×6×6 color cube, then 24 grays (the xterm layout).
pub fn default_color(i: usize) -> Rgb {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    match i {
        0..16 => BASE[i],
        16..232 => {
            let i = i - 16;
            [LEVELS[i / 36], LEVELS[i / 6 % 6], LEVELS[i % 6]]
        }
        _ => {
            let v = 8 + 10 * (i - 232) as u8;
            [v, v, v]
        }
    }
}

/// An X11 color spec as OSC sequences carry it: `rgb:R/G/B` with 1–4 hex
/// digits per component, or `#RGB` / `#RRGGBB` / `#RRRGGGBBB` /
/// `#RRRRGGGGBBBB`. Color names aren't supported.
pub fn parse_color(spec: &[u8]) -> Option<Rgb> {
    let spec = std::str::from_utf8(spec).ok()?;
    if let Some(rest) = spec.strip_prefix("rgb:") {
        let mut parts = rest.split('/');
        let mut out = [0; 3];
        for c in &mut out {
            let part = parts.next()?;
            if !(1..=4).contains(&part.len()) {
                return None;
            }
            let v = u32::from_str_radix(part, 16).ok()?;
            let max = (1u32 << (4 * part.len())) - 1;
            *c = ((v * 255 + max / 2) / max) as u8;
        }
        return parts.next().is_none().then_some(out);
    }
    let hex = spec.strip_prefix('#')?;
    if hex.is_empty() || hex.len() % 3 != 0 || hex.len() > 12 {
        return None;
    }
    let n = hex.len() / 3;
    let mut out = [0; 3];
    for (i, c) in out.iter_mut().enumerate() {
        // The most significant digits, as xterm reads them.
        let digits = &hex[i * n..i * n + n.min(2)];
        let v = u8::from_str_radix(digits, 16).ok()?;
        *c = if n == 1 { v << 4 } else { v };
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_palette_layout() {
        let p = Palette::default();
        assert_eq!(p.colors[1], [0xcc, 0, 0]);
        assert_eq!(p.colors[16], [0, 0, 0]);
        assert_eq!(p.colors[21], [0, 0, 255]);
        assert_eq!(p.colors[196], [255, 0, 0]);
        assert_eq!(p.colors[231], [255, 255, 255]);
        assert_eq!(p.colors[232], [8, 8, 8]);
        assert_eq!(p.colors[255], [238, 238, 238]);
    }

    #[test]
    fn color_specs() {
        assert_eq!(parse_color(b"rgb:ff/80/00"), Some([255, 128, 0]));
        assert_eq!(parse_color(b"rgb:f/8/0"), Some([255, 136, 0]));
        assert_eq!(parse_color(b"rgb:ffff/0000/8080"), Some([255, 0, 128]));
        assert_eq!(parse_color(b"#ff8000"), Some([255, 128, 0]));
        assert_eq!(parse_color(b"#f80"), Some([0xf0, 0x80, 0]));
        assert_eq!(parse_color(b"#ffff80800000"), Some([255, 128, 0]));
        for bad in [
            &b"rgb:ff/80"[..],
            b"rgb:ff/80/00/1",
            b"rgb:fffff/0/0",
            b"#ff80",
            b"red",
            b"#gg0000",
        ] {
            assert_eq!(parse_color(bad), None, "{}", bad.escape_ascii());
        }
    }
}
