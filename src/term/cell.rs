//! What one screen cell holds: a character and its style.

use std::ops::{BitOr, BitOrAssign};

/// A color as the program asked for it; the [`Palette`](super::Palette)
/// turns it into RGB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Color {
    /// The default foreground or background (or underline: same as the text).
    #[default]
    Default,
    /// One of the 256 palette colors.
    Indexed(u8),
    Rgb([u8; 3]),
}

/// Boolean text attributes.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags(u8);

impl Flags {
    pub const BOLD: Self = Self(1 << 0);
    pub const DIM: Self = Self(1 << 1);
    pub const ITALIC: Self = Self(1 << 2);
    pub const BLINK: Self = Self(1 << 3);
    pub const INVERSE: Self = Self(1 << 4);
    pub const HIDDEN: Self = Self(1 << 5);
    pub const STRIKE: Self = Self(1 << 6);
    pub const OVERLINE: Self = Self(1 << 7);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
}

impl BitOr for Flags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Flags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::fmt::Debug for Flags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const NAMES: [&str; 8] = [
            "BOLD", "DIM", "ITALIC", "BLINK", "INVERSE", "HIDDEN", "STRIKE", "OVERLINE",
        ];
        let set: Vec<_> = (0..8)
            .filter(|i| self.0 & (1 << i) != 0)
            .map(|i| NAMES[i])
            .collect();
        write!(f, "Flags({})", set.join(" | "))
    }
}

/// Underline style (SGR 4:n).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Underline {
    #[default]
    None,
    Single,
    Double,
    Curly,
    Dotted,
    Dashed,
}

/// How text is drawn: what SGR sets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Style {
    pub fg: Color,
    pub bg: Color,
    pub underline_color: Color,
    pub flags: Flags,
    pub underline: Underline,
}

impl Style {
    /// The style erased cells get: the background only (xterm's
    /// "background color erase").
    pub fn erased(&self) -> Self {
        Self {
            bg: self.bg,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub ch: char,
    pub style: Style,
}

impl Cell {
    pub const fn blank(style: Style) -> Self {
        Self { ch: ' ', style }
    }

    /// A space with no style at all.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::blank(Style::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(size_of::<Color>(), 4);
        assert_eq!(size_of::<Style>(), 14);
        assert_eq!(size_of::<Cell>(), 20);
    }

    #[test]
    fn flags() {
        let mut f = Flags::BOLD | Flags::ITALIC;
        assert!(f.contains(Flags::BOLD) && !f.contains(Flags::DIM));
        f.remove(Flags::BOLD);
        f |= Flags::STRIKE;
        assert_eq!(f, Flags::ITALIC | Flags::STRIKE);
        assert_eq!(format!("{f:?}"), "Flags(ITALIC | STRIKE)");
    }

    #[test]
    fn erasing_keeps_only_the_background() {
        let s = Style {
            fg: Color::Indexed(1),
            bg: Color::Rgb([1, 2, 3]),
            flags: Flags::INVERSE,
            underline: Underline::Single,
            ..Style::default()
        };
        assert_eq!(
            s.erased(),
            Style {
                bg: Color::Rgb([1, 2, 3]),
                ..Style::default()
            }
        );
    }
}
