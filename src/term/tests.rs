//! Terminal behavior, sequence by sequence: feed bytes, look at the screen.

use super::*;

fn term(cols: u16, rows: u16, input: &str) -> Terminal {
    let mut t = Terminal::new(cols, rows);
    t.feed(input.as_bytes());
    t
}

/// All rows' text, trailing empty rows dropped.
fn text(t: &Terminal) -> Vec<String> {
    let mut rows: Vec<String> = (0..t.rows()).map(|r| t.row_text(r)).collect();
    while rows.last().is_some_and(String::is_empty) {
        rows.pop();
    }
    rows
}

fn cursor(t: &Terminal) -> (u16, u16) {
    let c = t.cursor();
    (c.col, c.row)
}

/// `rows` lines `0`, `1`, ... (one per row).
fn numbered(rows: u16) -> String {
    (0..rows)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join("\r\n")
}

#[test]
fn prints_and_moves_with_cr_lf() {
    let mut t = term(10, 3, "hello\r\nworld");
    assert_eq!(text(&t), ["hello", "world"]);
    assert_eq!(cursor(&t), (5, 1));
    t.feed(b"\rW");
    assert_eq!(t.row_text(1), "World");
    // Bare LF keeps the column; with LNM it returns too.
    let t = term(10, 3, "ab\ncd\x1b[20h\nx");
    assert_eq!(text(&t), ["ab", "  cd", "x"]);
}

#[test]
fn pending_wrap() {
    let mut t = term(4, 3, "abcd");
    assert_eq!(cursor(&t), (3, 0), "stays on the last column");
    assert!(!t.line(0).wrapped);
    t.feed(b"e");
    assert_eq!(text(&t), ["abcd", "e"]);
    assert!(t.line(0).wrapped);
    // CR cancels it.
    let t = term(4, 3, "abcd\rX");
    assert_eq!(text(&t), ["Xbcd"]);
    // BS from the pending position goes to the second to last column.
    let t = term(4, 3, "abcd\x08X");
    assert_eq!(text(&t), ["abXd"]);
    // A cursor move cancels it.
    let t = term(4, 3, "abcd\x1b[1;2HX");
    assert_eq!(text(&t), ["aXcd"]);
    // CR LF after a full line doesn't leave an empty line.
    let t = term(4, 3, "abcd\r\nef");
    assert_eq!(text(&t), ["abcd", "ef"]);
}

#[test]
fn no_autowrap_overwrites_the_last_column() {
    let t = term(4, 2, "\x1b[?7labcdefg");
    assert_eq!(text(&t), ["abcg"]);
    assert_eq!(cursor(&t), (3, 0));
}

#[test]
fn scrolls_into_the_scrollback() {
    let t = term(5, 3, &numbered(5));
    assert_eq!(text(&t), ["2", "3", "4"]);
    let sb = t.scrollback();
    assert_eq!(sb.len(), 2);
    assert_eq!(sb.line(0).text(), "0");
    assert_eq!(sb.line(1).text(), "1");
}

#[test]
fn scrollback_is_limited() {
    let mut t = Terminal::with_scrollback(5, 2, 3);
    t.feed(numbered(10).as_bytes());
    assert_eq!(t.scrollback().len(), 3);
    assert_eq!(t.scrollback().line(0).text(), "5");
}

#[test]
fn scroll_region() {
    // Region rows 2-3 (1-based): only they scroll, nothing to scrollback.
    let mut t = term(5, 4, &numbered(4));
    t.feed(b"\x1b[2;3r");
    assert_eq!(cursor(&t), (0, 0), "DECSTBM homes the cursor");
    t.feed(b"\x1b[3;1H\nx\r\ny");
    assert_eq!(text(&t), ["0", "x", "y", "3"]);
    assert_eq!(t.scrollback().len(), 0);
    // Reverse index at the region top scrolls it down.
    t.feed(b"\x1b[2;1H\x1bM");
    assert_eq!(text(&t), ["0", "", "x", "3"]);
    // Below the region, LF moves down to the last row and stops.
    t.feed(b"\x1b[4;1H\n\nz");
    assert_eq!(text(&t), ["0", "", "x", "z"]);
}

#[test]
fn scroll_region_with_top_margin_zero_keeps_scrollback() {
    let mut t = term(5, 4, "\x1b[1;3r");
    t.feed(numbered(4).as_bytes());
    assert_eq!(text(&t), ["1", "2", "3"]);
    assert_eq!(t.scrollback().line(0).text(), "0");
}

#[test]
fn invalid_scroll_regions_are_ignored() {
    let mut t = term(5, 4, "\x1b[3;3r\x1b[4;2r");
    t.feed(numbered(5).as_bytes());
    assert_eq!(text(&t), ["1", "2", "3", "4"]);
    // Bottom past the screen is clamped.
    let mut t = term(5, 4, "\x1b[2;99r");
    t.feed(numbered(5).as_bytes());
    assert_eq!(text(&t), ["0", "2", "3", "4"]);
}

#[test]
fn origin_mode() {
    let mut t = term(5, 5, "\x1b[2;4r\x1b[?6h");
    assert_eq!(cursor(&t), (0, 1), "home is the region's top");
    t.feed(b"\x1b[2;3H");
    assert_eq!(cursor(&t), (2, 2));
    t.feed(b"\x1b[9;1H");
    assert_eq!(cursor(&t), (0, 3), "kept inside the region");
    t.feed(b"\x1b[1d");
    assert_eq!(cursor(&t), (0, 1));
    t.feed(b"\x1b[?6l");
    assert_eq!(cursor(&t), (0, 0));
}

#[test]
fn cursor_movement() {
    let mut t = term(10, 5, "\x1b[3;4H");
    assert_eq!(cursor(&t), (3, 2));
    let moves: &[(&str, (u16, u16))] = &[
        ("\x1b[A", (3, 1)),
        ("\x1b[5A", (3, 0)),
        ("\x1b[2B", (3, 2)),
        ("\x1b[99B", (3, 4)),
        ("\x1b[3C", (6, 4)),
        ("\x1b[99C", (9, 4)),
        ("\x1b[4D", (5, 4)),
        ("\x1b[0D", (4, 4)),
        ("\x1b[2F", (0, 2)),
        ("\x1b[E", (0, 3)),
        ("\x1b[7G", (6, 3)),
        ("\x1b[2`", (1, 3)),
        ("\x1b[3a", (4, 3)),
        ("\x1b[2d", (4, 1)),
        ("\x1b[2e", (4, 3)),
        ("\x1b[H", (0, 0)),
        ("\x1b[;5H", (4, 0)),
        ("\x1b[2;99f", (9, 1)),
    ];
    for &(seq, expected) in moves {
        t.feed(seq.as_bytes());
        assert_eq!(cursor(&t), expected, "{}", seq.escape_debug());
    }
}

#[test]
fn up_and_down_stop_at_the_margins() {
    let mut t = term(5, 6, "\x1b[2;4r\x1b[3;1H\x1b[9A");
    assert_eq!(cursor(&t), (0, 1));
    t.feed(b"\x1b[9B");
    assert_eq!(cursor(&t), (0, 3));
    // Outside the region, the screen edges.
    t.feed(b"\x1b[6;1H\x1b[9A");
    // From below the region, up stops at the top margin (as in xterm).
    assert_eq!(cursor(&t), (0, 1));
    t.feed(b"\x1b[1;1H\x1b[A");
    assert_eq!(cursor(&t), (0, 0));
}

#[test]
fn tabs() {
    let mut t = term(20, 1, "a\tb\tc\td");
    assert_eq!(t.row_text(0), "a       b       c  d");
    // Clear all, set one at column 3.
    t.feed(b"\r\x1b[3g\x1b[3G\x1bH\r\tX");
    assert_eq!(cursor(&t), (3, 0));
    // No stop further right: to the last column.
    t.feed(b"\t");
    assert_eq!(cursor(&t), (19, 0));
    // Back tab and forward tab with counts.
    let mut t = term(40, 1, "\x1b[3I");
    assert_eq!(cursor(&t), (24, 0));
    t.feed(b"\x1b[2Z");
    assert_eq!(cursor(&t), (8, 0));
    t.feed(b"\x1b[9Z");
    assert_eq!(cursor(&t), (0, 0));
    // Clear just the one under the cursor.
    let mut t = term(40, 1, "\x1b[9G\x1b[g\r\t");
    assert_eq!(cursor(&t), (16, 0));
    t.feed(b"\x1b[0g\r\t");
    assert_eq!(cursor(&t), (24, 0));
}

#[test]
fn erase_in_display_and_line() {
    let screen = "abcde\r\nfghij\r\nklmno\x1b[2;3H";
    assert_eq!(
        text(&term(5, 3, &format!("{screen}\x1b[J"))),
        ["abcde", "fg"]
    );
    assert_eq!(
        text(&term(5, 3, &format!("{screen}\x1b[1J"))),
        ["", "   ij", "klmno"]
    );
    assert_eq!(
        text(&term(5, 3, &format!("{screen}\x1b[2J"))),
        [] as [&str; 0]
    );
    assert_eq!(
        text(&term(5, 3, &format!("{screen}\x1b[K"))),
        ["abcde", "fg", "klmno"]
    );
    assert_eq!(
        text(&term(5, 3, &format!("{screen}\x1b[1K"))),
        ["abcde", "   ij", "klmno"]
    );
    assert_eq!(
        text(&term(5, 3, &format!("{screen}\x1b[2K"))),
        ["abcde", "", "klmno"]
    );
    // Erasing doesn't move the cursor.
    assert_eq!(cursor(&term(5, 3, &format!("{screen}\x1b[2J"))), (2, 1));
}

#[test]
fn ed_3_clears_the_scrollback_only() {
    let mut t = term(5, 2, &numbered(4));
    assert_eq!(t.scrollback().len(), 2);
    t.feed(b"\x1b[3J");
    assert!(t.scrollback().is_empty());
    assert_eq!(text(&t), ["2", "3"]);
}

#[test]
fn erasing_uses_the_background_color() {
    let t = term(4, 2, "ab\x1b[41m\x1b[K\x1b[1m\x1b[2;1H\x1b[X");
    let red = Style {
        bg: Color::Indexed(1),
        ..Style::default()
    };
    assert_eq!(t.cell(1, 0).style, Style::default());
    assert_eq!(t.cell(2, 0).style, red);
    assert_eq!(t.cell(3, 0).style, red);
    assert_eq!(t.cell(0, 1).style, red, "ECH too, without the bold");
    assert_eq!(t.cell(1, 1).style, Style::default());
    // And scrolling brings in colored lines.
    let t = term(4, 2, "\x1b[44m\n\n");
    assert_eq!(t.cell(0, 1).style.bg, Color::Indexed(4));
}

#[test]
fn insert_and_delete_characters() {
    assert_eq!(text(&term(6, 1, "abcdef\x1b[3G\x1b[2@")), ["ab  cd"]);
    assert_eq!(text(&term(6, 1, "abcdef\x1b[3G\x1b[2P")), ["abef"]);
    assert_eq!(text(&term(6, 1, "abcdef\x1b[3G\x1b[99P")), ["ab"]);
    assert_eq!(text(&term(6, 1, "abcdef\x1b[3G\x1b[99@")), ["ab"]);
    // At the right edge, after a pending wrap.
    let t = term(4, 2, "abcd\x1b[@X");
    assert_eq!(text(&t), ["abcX"], "ICH cancels the pending wrap");
    // Erase characters: no shifting.
    assert_eq!(text(&term(6, 1, "abcdef\x1b[2G\x1b[2X")), ["a  def"]);
    // Insert mode.
    assert_eq!(
        text(&term(6, 1, "abcd\x1b[2G\x1b[4hXY\x1b[4lZ")),
        ["aXYZcd"]
    );
}

#[test]
fn insert_and_delete_lines() {
    let lines = numbered(5);
    assert_eq!(
        text(&term(5, 5, &format!("{lines}\x1b[2;3H\x1b[2L"))),
        ["0", "", "", "1", "2"]
    );
    assert_eq!(
        text(&term(5, 5, &format!("{lines}\x1b[2;3H\x1b[2M"))),
        ["0", "3", "4"]
    );
    // Within a region; the cursor returns to column 0.
    let t = term(5, 5, &format!("{lines}\x1b[2;4r\x1b[3;3H\x1b[L"));
    assert_eq!(text(&t), ["0", "1", "", "2", "4"]);
    assert_eq!(cursor(&t), (0, 2));
    let t = term(5, 5, &format!("{lines}\x1b[2;4r\x1b[2;1H\x1b[9M"));
    assert_eq!(text(&t), ["0", "", "", "", "4"]);
    // Outside the region: nothing.
    let t = term(5, 5, &format!("{lines}\x1b[2;4r\x1b[5;3H\x1b[L\x1b[M"));
    assert_eq!(text(&t), ["0", "1", "2", "3", "4"]);
    assert_eq!(t.scrollback().len(), 0, "deleted lines aren't scrollback");
}

#[test]
fn scroll_up_and_down() {
    let lines = numbered(4);
    let t = term(5, 4, &format!("{lines}\x1b[2S"));
    assert_eq!(text(&t), ["2", "3"]);
    assert_eq!(t.scrollback().len(), 2);
    assert_eq!(
        text(&term(5, 4, &format!("{lines}\x1b[T"))),
        ["", "0", "1", "2"]
    );
}

#[test]
fn repeat_last_character() {
    assert_eq!(text(&term(10, 1, "ab\x1b[3b")), ["abbbb"]);
    assert_eq!(text(&term(10, 1, "\x1b[3b")), [] as [&str; 0]);
    // Capped at a screenful.
    assert_eq!(text(&term(4, 2, "x\x1b[65535b")), ["xxxx", "x"]);
}

#[test]
fn save_and_restore_cursor() {
    let mut t = term(10, 5, "\x1b[2;3H\x1b[1;31m\x1b7\x1b[H\x1b[0m\x1b8x");
    assert_eq!(cursor(&t), (3, 1));
    let style = t.cell(2, 1).style;
    assert_eq!(style.fg, Color::Indexed(1));
    assert!(style.flags.contains(Flags::BOLD));
    // CSI s / u do the same.
    t.feed(b"\x1b[4;5H\x1b[s\x1b[H\x1b[u");
    assert_eq!(cursor(&t), (4, 3));
    // Restoring with nothing saved: home, default pen.
    let t = term(10, 5, "\x1b[3;3H\x1b[31m\x1b8x");
    assert_eq!(cursor(&t), (1, 0));
    assert_eq!(t.cell(0, 0).style, Style::default());
}

#[test]
fn alternate_screen_1049() {
    let mut t = term(10, 3, "main\x1b[1;31m");
    t.feed(b"\x1b[?1049h");
    assert!(t.alternate_screen());
    assert_eq!(text(&t), [] as [&str; 0], "starts clear");
    t.feed(b"\x1b[0m\x1b[3;1Halt\r\n\r\n");
    assert_eq!(t.scrollback().len(), 0, "the alternate screen has none");
    t.feed(b"\x1b[?1049l");
    assert!(!t.alternate_screen());
    assert_eq!(text(&t), ["main"]);
    assert_eq!(cursor(&t), (4, 0), "cursor restored");
    t.feed(b"x");
    assert_eq!(t.cell(4, 0).style.fg, Color::Indexed(1), "pen restored");
    // Entering again clears what was left there.
    t.feed(b"\x1b[?1049h");
    assert_eq!(text(&t), [] as [&str; 0]);
}

#[test]
fn alternate_screen_47_and_1047() {
    // 47 switches without clearing anything.
    let mut t = term(10, 2, "main\x1b[?47halt\x1b[?47l");
    assert_eq!(text(&t), ["main"]);
    t.feed(b"\x1b[?47h");
    assert_eq!(text(&t), ["    alt"]);
    t.feed(b"\x1b[?47l");
    // 1047 clears the alternate screen when leaving it.
    t.feed(b"\x1b[?1047h");
    assert_eq!(text(&t), ["    alt"]);
    t.feed(b"\x1b[?1047l\x1b[?47h");
    assert_eq!(text(&t), [] as [&str; 0]);
    // 1048 is save/restore cursor alone.
    let t = term(10, 2, "\x1b[2;5H\x1b[?1048h\x1b[H\x1b[?1048l");
    assert_eq!(cursor(&t), (4, 1));
}

#[test]
fn saved_cursors_are_per_screen() {
    let t = term(10, 3, "\x1b[2;2H\x1b7\x1b[?47h\x1b[3;3H\x1b7\x1b[?47l\x1b8");
    assert_eq!(cursor(&t), (1, 1));
}

#[test]
fn line_drawing_charset() {
    let t = term(10, 3, "\x1b(0lqqk\r\nx  x\r\nmqqj\x1b(B q");
    assert_eq!(text(&t), ["┌──┐", "│  │", "└──┘ q"]);
    // Through G1 with SO / SI.
    let t = term(10, 1, "\x1b)0a\x0eqx\x0fqx");
    assert_eq!(text(&t), ["a─│qx"]);
    // UK: just the pound sign.
    let t = term(10, 1, "\x1b(A#3");
    assert_eq!(text(&t), ["£3"]);
}

#[test]
fn modes_are_recorded() {
    let t = term(
        10,
        1,
        "\x1b[?1;25;2004;1002;1006;1004;2026h\x1b=\x1b[?25l\x1b[?12l\x1b[5 q",
    );
    let m = t.modes();
    assert!(m.cursor_keys_app && m.bracketed_paste && m.focus_events && m.keypad_app);
    assert_eq!(m.mouse, MouseTracking::Drag);
    assert_eq!(m.mouse_encoding, MouseEncoding::Sgr);
    assert!(t.synchronized());
    let c = t.cursor();
    assert!(!c.visible);
    assert_eq!((c.shape, c.blink), (CursorShape::Bar, true));
    let t = term(
        10,
        1,
        "\x1b[?1002h\x1b[?1000l\x1b[2 q\x1b[?2026h\x1b[?2026l",
    );
    assert_eq!(t.modes().mouse, MouseTracking::Off);
    assert_eq!(
        (t.cursor().shape, t.cursor().blink),
        (CursorShape::Block, false)
    );
    assert!(!t.synchronized());
}

#[test]
fn soft_reset() {
    let t = term(
        10,
        4,
        "\x1b[2;3r\x1b[?6h\x1b[?7l\x1b[4h\x1b[?25l\x1b[31m\x1b(0\x1b[!p\x1b[4;1Hq",
    );
    let m = t.modes();
    assert!(m.autowrap && !m.origin && !m.insert && m.cursor_visible);
    assert_eq!(text(&t), ["", "", "", "q"]);
    assert_eq!(t.cell(0, 3).style, Style::default());
}

#[test]
fn full_reset() {
    let mut t = term(5, 2, &numbered(3));
    t.feed(b"\x1b[31m\x1b[?1049h\x1b[?25lx\x1bc");
    assert!(!t.alternate_screen());
    assert_eq!(text(&t), [] as [&str; 0]);
    assert_eq!(cursor(&t), (0, 0));
    assert!(t.cursor().visible);
    assert_eq!(t.scrollback().len(), 1, "the scrollback stays");
    t.feed(b"y");
    assert_eq!(t.cell(0, 0).style, Style::default());
}

#[test]
fn screen_alignment_pattern() {
    let t = term(3, 2, "\x1b[2;2r\x1b[2;2H\x1b#8");
    assert_eq!(text(&t), ["EEE", "EEE"]);
    assert_eq!(cursor(&t), (0, 0));
}

#[test]
fn sgr_sets_the_pen() {
    let t = term(10, 1, "\x1b[1;4:3;38;2;1;2;3;48;5;200mA\x1b[mB");
    let a = t.cell(0, 0).style;
    assert!(a.flags.contains(Flags::BOLD));
    assert_eq!(a.underline, Underline::Curly);
    assert_eq!(a.fg, Color::Rgb([1, 2, 3]));
    assert_eq!(a.bg, Color::Indexed(200));
    assert_eq!(t.cell(1, 0).style, Style::default());
}

#[test]
fn osc_title_and_colors() {
    let mut t = term(10, 1, "\x1b]2;my; title\x07\x1b]0;other\x1b\\");
    assert_eq!(t.title(), "other");
    t.feed(b"\x1b]2;my; title\x07");
    assert_eq!(t.title(), "my; title");

    t.feed(b"\x1b]4;1;#010203;2;rgb:ff/ff/ff\x07");
    assert_eq!(t.palette().colors[1], [1, 2, 3]);
    assert_eq!(t.palette().colors[2], [255, 255, 255]);
    t.feed(b"\x1b]104;1\x07");
    assert_eq!(t.palette().colors[1], palette::default_color(1));
    assert_eq!(t.palette().colors[2], [255, 255, 255]);
    t.feed(b"\x1b]104\x07");
    assert_eq!(t.palette().colors[2], palette::default_color(2));

    t.feed(b"\x1b]10;#102030;#405060\x07\x1b]12;#708090\x07");
    let p = t.palette();
    assert_eq!(
        (p.foreground, p.background, p.cursor),
        ([0x10, 0x20, 0x30], [0x40, 0x50, 0x60], [0x70, 0x80, 0x90])
    );
    t.feed(b"\x1b]111\x07");
    assert_eq!(t.palette().background, Palette::default().background);
    // Queries need replies: ignored for now, nothing changes.
    t.feed(b"\x1b]10;?\x07\x1b]4;1;?\x07");
    assert_eq!(t.palette().foreground, [0x10, 0x20, 0x30]);
}

#[test]
fn unhandled_sequences_are_harmless() {
    let t = term(
        20,
        2,
        "a\x1b[c\x1b[>c\x1b[6n\x1b[?1$p\x1b[?u\x1b[18t\x1bP+q544e\x1b\\\x1b]777;x\x07\
         \x1b[?9999h\x1b[73m\x1b%Gb\x1b_kitty\x1b\\c",
    );
    assert_eq!(text(&t), ["abc"]);
    assert_eq!(t.cell(1, 0).style, Style::default());
}

#[test]
fn utf8_and_invalid_input() {
    let mut t = Terminal::new(10, 1);
    for b in "é€😀".as_bytes() {
        t.feed(std::slice::from_ref(b));
    }
    assert_eq!(text(&t), ["é€😀"]);
    let mut t = Terminal::new(10, 1);
    t.feed(b"a\xffb");
    assert_eq!(text(&t), ["a\u{FFFD}b"]);
}

#[test]
fn degenerate_sizes_are_clamped() {
    let t = term(0, 0, "xy\r\nz\x1b[5;5H\x1b[3@\x1b[2L\x1b[9rq");
    assert_eq!((t.cols(), t.rows()), (1, 1));
    assert_eq!(text(&t), ["q"]);
}

#[test]
fn splitting_the_input_changes_nothing() {
    let corpus = super::parser::tests::corpus();
    let whole = {
        let mut t = Terminal::new(20, 5);
        t.feed(&corpus);
        (text(&t), cursor(&t), t.title().to_owned())
    };
    for split in 0..=corpus.len() {
        let mut t = Terminal::new(20, 5);
        t.feed(&corpus[..split]);
        t.feed(&corpus[split..]);
        assert_eq!(
            (text(&t), cursor(&t), t.title().to_owned()),
            whole,
            "split at {split}"
        );
    }
}
