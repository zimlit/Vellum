//! SGR (`CSI ... m`): changing the pen.

use super::cell::{Color, Flags, Style, Underline};
use super::parser::{Groups, Params};

/// Apply an SGR sequence to `pen`. Returns the codes it didn't understand.
pub fn apply(pen: &mut Style, params: &Params) -> Vec<u16> {
    let mut unknown = Vec::new();
    if params.is_empty() {
        *pen = Style::default();
        return unknown;
    }
    let mut groups = params.iter();
    while let Some(g) = groups.next() {
        match g[0] {
            0 => *pen = Style::default(),
            1 => pen.flags.insert(Flags::BOLD),
            2 => pen.flags.insert(Flags::DIM),
            3 => pen.flags.insert(Flags::ITALIC),
            4 => {
                pen.underline = match g.get(1) {
                    None | Some(1) => Underline::Single,
                    Some(0) => Underline::None,
                    Some(2) => Underline::Double,
                    Some(3) => Underline::Curly,
                    Some(4) => Underline::Dotted,
                    Some(5) => Underline::Dashed,
                    Some(_) => {
                        unknown.push(4);
                        continue;
                    }
                }
            }
            // Slow and rapid blink.
            5 | 6 => pen.flags.insert(Flags::BLINK),
            7 => pen.flags.insert(Flags::INVERSE),
            8 => pen.flags.insert(Flags::HIDDEN),
            9 => pen.flags.insert(Flags::STRIKE),
            21 => pen.underline = Underline::Double,
            22 => pen.flags.remove(Flags::BOLD | Flags::DIM),
            23 => pen.flags.remove(Flags::ITALIC),
            24 => pen.underline = Underline::None,
            25 => pen.flags.remove(Flags::BLINK),
            27 => pen.flags.remove(Flags::INVERSE),
            28 => pen.flags.remove(Flags::HIDDEN),
            29 => pen.flags.remove(Flags::STRIKE),
            n @ 30..=37 => pen.fg = Color::Indexed((n - 30) as u8),
            39 => pen.fg = Color::Default,
            n @ 40..=47 => pen.bg = Color::Indexed((n - 40) as u8),
            49 => pen.bg = Color::Default,
            53 => pen.flags.insert(Flags::OVERLINE),
            55 => pen.flags.remove(Flags::OVERLINE),
            59 => pen.underline_color = Color::Default,
            n @ 90..=97 => pen.fg = Color::Indexed((n - 90 + 8) as u8),
            n @ 100..=107 => pen.bg = Color::Indexed((n - 100 + 8) as u8),
            n @ (38 | 48 | 58) => match extended(g, &mut groups) {
                Some(c) if n == 38 => pen.fg = c,
                Some(c) if n == 48 => pen.bg = c,
                Some(c) => pen.underline_color = c,
                None => unknown.push(n),
            },
            n => unknown.push(n),
        }
    }
    unknown
}

/// The color of a 38/48/58 code: `5;n` / `2;r;g;b` from the following
/// groups, or the colon forms `5:n`, `2::r:g:b` (with a color space id,
/// usually empty) and `2:r:g:b` within the group.
fn extended(group: &[u16], rest: &mut Groups<'_>) -> Option<Color> {
    let byte = |v: u16| u8::try_from(v).ok();
    let rgb = |v: &[u16]| Some(Color::Rgb([byte(v[0])?, byte(v[1])?, byte(v[2])?]));
    if group.len() > 1 {
        return match (group[1], group.len()) {
            (5, 3) => Some(Color::Indexed(byte(group[2])?)),
            (2, 6..) => rgb(&group[3..6]),
            (2, 5) => rgb(&group[2..5]),
            _ => None,
        };
    }
    let mut next = || rest.next().map(|g| g[0]);
    match next()? {
        5 => Some(Color::Indexed(byte(next()?)?)),
        2 => {
            let v = [next()?, next()?, next()?];
            rgb(&v)
        }
        _ => None,
    }
}

/// The SGR parameters that set a fresh pen to `pen` (for DECRQSS), always
/// starting with `0`. Colors use the semicolon forms, which every program
/// that reads them back understands; the underline color uses the colon
/// form, its usual spelling.
pub fn describe(pen: &Style) -> String {
    let mut codes = vec!["0".to_owned()];
    const FLAGS: [(Flags, &str); 8] = [
        (Flags::BOLD, "1"),
        (Flags::DIM, "2"),
        (Flags::ITALIC, "3"),
        (Flags::BLINK, "5"),
        (Flags::INVERSE, "7"),
        (Flags::HIDDEN, "8"),
        (Flags::STRIKE, "9"),
        (Flags::OVERLINE, "53"),
    ];
    for (flag, code) in FLAGS {
        if pen.flags.contains(flag) {
            codes.push(code.to_owned());
        }
    }
    match pen.underline {
        Underline::None => {}
        Underline::Single => codes.push("4".to_owned()),
        Underline::Double => codes.push("4:2".to_owned()),
        Underline::Curly => codes.push("4:3".to_owned()),
        Underline::Dotted => codes.push("4:4".to_owned()),
        Underline::Dashed => codes.push("4:5".to_owned()),
    }
    match pen.fg {
        Color::Default => {}
        Color::Indexed(i @ 0..8) => codes.push((30 + i).to_string()),
        Color::Indexed(i @ 8..16) => codes.push((90 + i - 8).to_string()),
        Color::Indexed(i) => codes.push(format!("38;5;{i}")),
        Color::Rgb([r, g, b]) => codes.push(format!("38;2;{r};{g};{b}")),
    }
    match pen.bg {
        Color::Default => {}
        Color::Indexed(i @ 0..8) => codes.push((40 + i).to_string()),
        Color::Indexed(i @ 8..16) => codes.push((100 + i - 8).to_string()),
        Color::Indexed(i) => codes.push(format!("48;5;{i}")),
        Color::Rgb([r, g, b]) => codes.push(format!("48;2;{r};{g};{b}")),
    }
    match pen.underline_color {
        Color::Default => {}
        Color::Indexed(i) => codes.push(format!("58:5:{i}")),
        Color::Rgb([r, g, b]) => codes.push(format!("58:2::{r}:{g}:{b}")),
    }
    codes.join(";")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::parser::{Parser, Perform};

    #[test]
    fn describe_round_trips() {
        for params in [
            "",
            "1;3;4;7;9;53",
            "2;5;8;4:3;31;42",
            "97;100;4:2",
            "38;5;196;48;2;1;2;3;58:2::4:5:6",
            "58;5;9;4:5",
        ] {
            let s = style(params);
            let described = describe(&s);
            assert!(described.starts_with('0'), "{described}");
            assert_eq!(style(&described), s, "{params} → {described}");
        }
        assert_eq!(describe(&style("1;31")), "0;1;31");
    }

    /// Parse `CSI <params> m` and apply it to `pen`.
    fn sgr(pen: &mut Style, params: &str) -> Vec<u16> {
        struct Apply<'a>(&'a mut Style, Vec<u16>);
        impl Perform for Apply<'_> {
            fn print(&mut self, _: char) {}
            fn execute(&mut self, _: u8) {}
            fn csi_dispatch(&mut self, p: &Params, _: &[u8], _: bool, _: u8) {
                self.1 = apply(self.0, p);
            }
            fn esc_dispatch(&mut self, _: &[u8], _: bool, _: u8) {}
            fn osc_dispatch(&mut self, _: &[&[u8]], _: bool) {}
        }
        let mut a = Apply(pen, Vec::new());
        Parser::new().advance(&mut a, format!("\x1b[{params}m").as_bytes());
        a.1
    }

    fn style(params: &str) -> Style {
        let mut s = Style::default();
        assert_eq!(sgr(&mut s, params), [] as [u16; 0], "{params}");
        s
    }

    #[test]
    fn attributes_set_and_reset() {
        let s = style("1;3;4;7;9;53");
        assert_eq!(
            s.flags,
            Flags::BOLD | Flags::ITALIC | Flags::INVERSE | Flags::STRIKE | Flags::OVERLINE
        );
        assert_eq!(s.underline, Underline::Single);
        let mut s = s;
        sgr(&mut s, "22;23;24;27;29;55");
        assert_eq!(s, Style::default());
        let mut s = style("2;5;8;31");
        sgr(&mut s, "");
        assert_eq!(s, Style::default(), "empty SGR resets");
        sgr(&mut s, "1;0");
        assert_eq!(s, Style::default());
    }

    #[test]
    fn basic_and_bright_colors() {
        let s = style("31;42");
        assert_eq!((s.fg, s.bg), (Color::Indexed(1), Color::Indexed(2)));
        let s = style("97;100");
        assert_eq!((s.fg, s.bg), (Color::Indexed(15), Color::Indexed(8)));
        let mut s = s;
        sgr(&mut s, "39;49");
        assert_eq!(s, Style::default());
    }

    #[test]
    fn extended_colors_in_both_syntaxes() {
        let red = Color::Rgb([255, 0, 0]);
        assert_eq!(style("38;5;196").fg, Color::Indexed(196));
        assert_eq!(style("38:5:196").fg, Color::Indexed(196));
        assert_eq!(style("48;2;255;0;0").bg, red);
        assert_eq!(style("48:2::255:0:0").bg, red);
        assert_eq!(style("48:2:255:0:0").bg, red);
        assert_eq!(style("58;5;3").underline_color, Color::Indexed(3));
        // What follows a semicolon-form color still applies.
        let s = style("38;2;1;2;3;1");
        assert_eq!(s.fg, Color::Rgb([1, 2, 3]));
        assert!(s.flags.contains(Flags::BOLD));
    }

    #[test]
    fn underline_styles() {
        assert_eq!(style("4:3").underline, Underline::Curly);
        assert_eq!(style("4:5").underline, Underline::Dashed);
        assert_eq!(style("21").underline, Underline::Double);
        assert_eq!(style("4;4:0").underline, Underline::None);
    }

    #[test]
    fn bad_codes_are_reported() {
        let mut s = Style::default();
        assert_eq!(sgr(&mut s, "1;73;38;5;300;31"), [73, 38]);
        assert!(s.flags.contains(Flags::BOLD));
        assert_eq!(s.fg, Color::Indexed(1));
        assert_eq!(sgr(&mut s, "38;2;1"), [38], "truncated");
        assert_eq!(sgr(&mut s, "4:9"), [4]);
    }
}
