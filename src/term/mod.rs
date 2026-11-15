//! The terminal core: parsing what programs print ([`parser`]) and keeping
//! the screen state it describes ([`Terminal`]).
//!
//! The behavior follows xterm where terminals differ, since that's what
//! `TERM=xterm-256color` promises. Not handled yet: wide characters and
//! combining marks (roadmap step 8) and resize (the size is fixed).
//! Sequences that aren't handled are logged once each at debug level,
//! target `vellum::term`.
//!
//! Queries (device attributes, cursor position, modes, colors, terminfo
//! capabilities, ...) are answered by queuing replies, which the owner
//! sends to the program: see [`Terminal::take_replies`].

pub mod cell;
pub mod palette;
pub mod parser;
mod reply;
pub mod screen;
pub mod scrollback;
mod sgr;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

pub use cell::{Cell, Color, Flags, Style, Underline};
pub use palette::Palette;
pub use parser::{Params, Parser, Perform};
pub use screen::Line;
pub use scrollback::Scrollback;

use screen::Screen;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorShape {
    #[default]
    Block,
    Underline,
    Bar,
}

/// What the renderer needs to draw the cursor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorState {
    pub col: u16,
    pub row: u16,
    pub shape: CursorShape,
    pub blink: bool,
    pub visible: bool,
}

/// Which mouse events the program wants (DECSET 1000 / 1002 / 1003).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MouseTracking {
    #[default]
    Off,
    /// Presses and releases.
    Click,
    /// Plus motion while a button is down.
    Drag,
    /// Plus all motion.
    Motion,
}

/// How mouse events are encoded (DECSET 1005 / 1006 / 1015).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MouseEncoding {
    #[default]
    X10,
    Utf8,
    Sgr,
    Urxvt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Modes {
    /// DECAWM: wrap at the right edge.
    pub autowrap: bool,
    /// DECOM: cursor positions are relative to the scroll region.
    pub origin: bool,
    /// IRM: printing inserts instead of overwriting.
    pub insert: bool,
    /// LNM: line feed also returns the carriage.
    pub newline: bool,
    /// DECTCEM.
    pub cursor_visible: bool,
    /// DECCKM: cursor keys send `ESC O x` instead of `CSI x` (step 6).
    pub cursor_keys_app: bool,
    /// DECKPAM / DECNKM: application keypad (step 6).
    pub keypad_app: bool,
    pub bracketed_paste: bool,
    pub focus_events: bool,
    pub mouse: MouseTracking,
    pub mouse_encoding: MouseEncoding,
    /// Synchronized output (2026): the program is in the middle of an
    /// update and would rather it weren't shown yet.
    pub synchronized: bool,
}

impl Default for Modes {
    fn default() -> Self {
        Self {
            autowrap: true,
            origin: false,
            insert: false,
            newline: false,
            cursor_visible: true,
            cursor_keys_app: false,
            keypad_app: false,
            bracketed_paste: false,
            focus_events: false,
            mouse: MouseTracking::Off,
            mouse_encoding: MouseEncoding::X10,
            synchronized: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Charset {
    #[default]
    Ascii,
    /// DEC Special Graphics: line drawing in place of `_` to `~`.
    DecSpecial,
    /// British: `#` is `£`.
    Uk,
}

impl Charset {
    fn map(self, ch: char) -> char {
        match self {
            Self::Ascii => ch,
            Self::Uk => {
                if ch == '#' {
                    '£'
                } else {
                    ch
                }
            }
            Self::DecSpecial => match ch {
                '_'..='~' => DEC_SPECIAL[ch as usize - '_' as usize],
                _ => ch,
            },
        }
    }
}

/// DEC Special Graphics for `_` (0x5F) to `~` (0x7E).
const DEC_SPECIAL: [char; 32] = [
    ' ', '◆', '▒', '␉', '␌', '␍', '␊', '°', '±', '␤', '␋', '┘', '┐', '┌', '└', '┼', '⎺', '⎻', '─',
    '⎼', '⎽', '├', '┤', '┴', '┬', '│', '≤', '≥', 'π', '≠', '£', '·',
];

#[derive(Clone, Copy, Debug, Default)]
struct Cursor {
    col: u16,
    row: u16,
    /// The last column was just written: the next character wraps first.
    /// `col` stays on the last column meanwhile.
    pending_wrap: bool,
    pen: Style,
    /// G0–G3.
    charsets: [Charset; 4],
    /// Which of G0–G3 is in use (SO/SI switch between G0 and G1).
    gl: u8,
}

/// What DECSC saves.
#[derive(Clone, Copy, Debug)]
struct Saved {
    cursor: Cursor,
    origin: bool,
}

/// Most distinct unhandled sequences to log.
const MAX_REPORTED: usize = 256;

/// The DCS requests that are answered; their data is collected until the
/// string ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DcsRequest {
    /// `DCS $ q`: request a setting (DECRQSS).
    Setting,
    /// `DCS + q`: request terminfo capabilities (XTGETTCAP).
    Capabilities,
}

struct Dcs {
    request: DcsRequest,
    data: Vec<u8>,
    /// Longer than [`reply::MAX_DCS`]: not answered.
    overflow: bool,
}

pub struct Terminal {
    parser: Parser,
    cols: u16,
    rows: u16,
    primary: Screen,
    alternate: Screen,
    /// The alternate screen is shown.
    alt: bool,
    scrollback: Scrollback,
    cursor: Cursor,
    /// DECSC state for the primary and alternate screens.
    saved: [Option<Saved>; 2],
    /// Scroll region: rows `top..bottom`.
    top: u16,
    bottom: u16,
    modes: Modes,
    cursor_shape: CursorShape,
    cursor_blink: bool,
    tabs: Vec<bool>,
    title: String,
    palette: Palette,
    /// For REP.
    last_printed: Option<char>,
    /// Kitty keyboard protocol modes for the primary and alternate screens.
    keyboard: [reply::KeyboardModes; 2],
    /// The DCS string being collected, if it's one we answer.
    dcs: Option<Dcs>,
    /// Replies waiting to be sent to the program.
    replies: Vec<u8>,
    /// A cell's size in pixels, for window size reports (0: unknown).
    cell_pixels: (u16, u16),
    /// Unhandled sequences already logged.
    reported: HashSet<String>,
}

impl Terminal {
    pub fn new(cols: u16, rows: u16) -> Self {
        Self::with_scrollback(cols, rows, scrollback::DEFAULT_LIMIT)
    }

    pub fn with_scrollback(cols: u16, rows: u16, lines: usize) -> Self {
        let (cols, rows) = (cols.max(1), rows.max(1));
        Self {
            parser: Parser::new(),
            cols,
            rows,
            primary: Screen::new(cols, rows),
            alternate: Screen::new(cols, rows),
            alt: false,
            scrollback: Scrollback::new(lines),
            cursor: Cursor::default(),
            saved: [None; 2],
            top: 0,
            bottom: rows,
            modes: Modes::default(),
            cursor_shape: CursorShape::default(),
            cursor_blink: true,
            tabs: default_tabs(cols),
            title: String::new(),
            palette: Palette::default(),
            last_printed: None,
            keyboard: Default::default(),
            dcs: None,
            replies: Vec::new(),
            cell_pixels: (0, 0),
            reported: HashSet::new(),
        }
    }

    /// Process what the program printed.
    pub fn feed(&mut self, bytes: &[u8]) {
        let mut parser = std::mem::take(&mut self.parser);
        parser.advance(self, bytes);
        self.parser = parser;
    }

    /// Take the replies to queries seen so far, to be written to the
    /// program in order.
    pub fn take_replies(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.replies)
    }

    /// Set a cell's size in pixels, for window size reports.
    pub fn set_cell_pixels(&mut self, width: u16, height: u16) {
        self.cell_pixels = (width, height);
    }

    /// The terminal gained or lost focus: reported when the program asked
    /// for focus events (1004).
    pub fn focus(&mut self, focused: bool) {
        if self.modes.focus_events {
            self.replies
                .extend_from_slice(if focused { b"\x1b[I" } else { b"\x1b[O" });
        }
    }

    /// The kitty keyboard protocol flags for the screen being shown.
    pub fn keyboard_flags(&self) -> u8 {
        self.keyboard[usize::from(self.alt)].flags
    }

    pub fn cols(&self) -> u16 {
        self.cols
    }

    pub fn rows(&self) -> u16 {
        self.rows
    }

    /// A cell of the screen being shown.
    pub fn cell(&self, col: u16, row: u16) -> &Cell {
        &self.screen().line(row).cells[usize::from(col)]
    }

    pub fn line(&self, row: u16) -> &Line {
        self.screen().line(row)
    }

    pub fn cursor(&self) -> CursorState {
        CursorState {
            col: self.cursor.col,
            row: self.cursor.row,
            shape: self.cursor_shape,
            blink: self.cursor_blink,
            visible: self.modes.cursor_visible,
        }
    }

    pub fn modes(&self) -> &Modes {
        &self.modes
    }

    pub fn palette(&self) -> &Palette {
        &self.palette
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn scrollback(&self) -> &Scrollback {
        &self.scrollback
    }

    pub fn alternate_screen(&self) -> bool {
        self.alt
    }

    /// The program asked to hold back drawing (synchronized output).
    pub fn synchronized(&self) -> bool {
        self.modes.synchronized
    }

    /// Row `row`'s text, trailing blanks trimmed (tests and debugging).
    pub fn row_text(&self, row: u16) -> String {
        self.line(row).text()
    }

    fn screen(&self) -> &Screen {
        if self.alt {
            &self.alternate
        } else {
            &self.primary
        }
    }

    fn screen_mut(&mut self) -> &mut Screen {
        if self.alt {
            &mut self.alternate
        } else {
            &mut self.primary
        }
    }

    /// What erasing leaves behind: a space with the pen's background.
    fn blank(&self) -> Cell {
        Cell::blank(self.cursor.pen.erased())
    }

    // Printing.

    fn put_char(&mut self, ch: char) {
        if self.cursor.pending_wrap {
            let row = self.cursor.row;
            self.screen_mut().line_mut(row).wrapped = true;
            self.cursor.col = 0;
            self.index();
        }
        let Cursor { col, row, pen, .. } = self.cursor;
        if self.modes.insert {
            let blank = self.blank();
            self.screen_mut().insert_cells(row, col, 1, blank);
        }
        *self.screen_mut().cell_mut(col, row) = Cell { ch, style: pen };
        if col + 1 < self.cols {
            self.cursor.col += 1;
        } else if self.modes.autowrap {
            self.cursor.pending_wrap = true;
        }
        self.last_printed = Some(ch);
    }

    // Cursor movement.

    fn carriage_return(&mut self) {
        self.cursor.col = 0;
        self.cursor.pending_wrap = false;
    }

    /// Move down a line, scrolling the region if at its bottom.
    fn index(&mut self) {
        self.cursor.pending_wrap = false;
        if self.cursor.row + 1 == self.bottom {
            self.scroll_up(1);
        } else if self.cursor.row + 1 < self.rows {
            self.cursor.row += 1;
        }
    }

    fn reverse_index(&mut self) {
        self.cursor.pending_wrap = false;
        if self.cursor.row == self.top {
            self.scroll_down(1);
        } else if self.cursor.row > 0 {
            self.cursor.row -= 1;
        }
    }

    /// Absolute position, clamped to the screen.
    fn goto(&mut self, col: u16, row: u16) {
        self.cursor.col = col.min(self.cols - 1);
        self.cursor.row = row.min(self.rows - 1);
        self.cursor.pending_wrap = false;
    }

    /// Row as the program means it: relative to the scroll region, and
    /// kept inside it, in origin mode.
    fn goto_origin(&mut self, col: u16, row: u16) {
        let row = if self.modes.origin {
            self.top.saturating_add(row).min(self.bottom - 1)
        } else {
            row
        };
        self.goto(col, row);
    }

    fn up(&mut self, n: u16) {
        // Stops at the top margin, unless already above it.
        let limit = if self.cursor.row >= self.top {
            self.top
        } else {
            0
        };
        let row = self.cursor.row.saturating_sub(n).max(limit);
        self.goto(self.cursor.col, row);
    }

    fn down(&mut self, n: u16) {
        let limit = if self.cursor.row < self.bottom {
            self.bottom - 1
        } else {
            self.rows - 1
        };
        let row = self.cursor.row.saturating_add(n).min(limit);
        self.goto(self.cursor.col, row);
    }

    fn tab_forward(&mut self, n: u16) {
        let mut col = self.cursor.col;
        for _ in 0..n {
            col = (col + 1..self.cols)
                .find(|&c| self.tabs[usize::from(c)])
                .unwrap_or(self.cols - 1);
        }
        self.cursor.col = col;
        self.cursor.pending_wrap = false;
    }

    fn tab_backward(&mut self, n: u16) {
        let mut col = self.cursor.col;
        for _ in 0..n {
            col = (0..col)
                .rev()
                .find(|&c| self.tabs[usize::from(c)])
                .unwrap_or(0);
        }
        self.cursor.col = col;
        self.cursor.pending_wrap = false;
    }

    // Scrolling and editing.

    /// Scroll the region up. Lines leaving the top of the primary screen
    /// go to the scrollback.
    fn scroll_up(&mut self, n: u16) {
        let (top, bottom, blank) = (self.top, self.bottom, self.blank());
        let gone = self.screen_mut().scroll_up(top, bottom, n, blank);
        if !self.alt && top == 0 {
            for line in gone {
                self.scrollback.push(line);
            }
        }
    }

    fn scroll_down(&mut self, n: u16) {
        let (top, bottom, blank) = (self.top, self.bottom, self.blank());
        self.screen_mut().scroll_down(top, bottom, n, blank);
    }

    fn insert_lines(&mut self, n: u16) {
        let row = self.cursor.row;
        if (self.top..self.bottom).contains(&row) {
            let (bottom, blank) = (self.bottom, self.blank());
            self.screen_mut().scroll_down(row, bottom, n, blank);
            self.carriage_return();
        }
    }

    fn delete_lines(&mut self, n: u16) {
        let row = self.cursor.row;
        if (self.top..self.bottom).contains(&row) {
            let (bottom, blank) = (self.bottom, self.blank());
            // Deleted, not scrolled off: no scrollback.
            self.screen_mut().scroll_up(row, bottom, n, blank);
            self.carriage_return();
        }
    }

    fn erase_display(&mut self, mode: u16) {
        let Cursor { col, row, .. } = self.cursor;
        let (cols, rows, blank) = (self.cols, self.rows, self.blank());
        let screen = self.screen_mut();
        match mode {
            0 => {
                screen.erase(row, col, cols, blank);
                screen.erase_rows(row + 1, rows, blank);
            }
            1 => {
                screen.erase_rows(0, row, blank);
                screen.erase(row, 0, col + 1, blank);
            }
            2 => screen.erase_rows(0, rows, blank),
            3 => self.scrollback.clear(),
            _ => self.unhandled(format!("ED {mode}")),
        }
    }

    fn erase_line(&mut self, mode: u16) {
        let Cursor { col, row, .. } = self.cursor;
        let (cols, blank) = (self.cols, self.blank());
        match mode {
            0 => self.screen_mut().erase(row, col, cols, blank),
            1 => self.screen_mut().erase(row, 0, col + 1, blank),
            2 => self.screen_mut().erase(row, 0, cols, blank),
            _ => self.unhandled(format!("EL {mode}")),
        }
    }

    fn set_scroll_region(&mut self, top: u16, bottom: u16) {
        let bottom = bottom.min(self.rows);
        // At least two lines.
        if top < bottom {
            self.top = top - 1;
            self.bottom = bottom;
            self.goto_origin(0, 0);
        }
    }

    // Saving, switching and resetting.

    fn save_cursor(&mut self) {
        self.saved[usize::from(self.alt)] = Some(Saved {
            cursor: self.cursor,
            origin: self.modes.origin,
        });
    }

    fn restore_cursor(&mut self) {
        match self.saved[usize::from(self.alt)] {
            Some(saved) => {
                self.cursor = saved.cursor;
                self.modes.origin = saved.origin;
                self.cursor.col = self.cursor.col.min(self.cols - 1);
                self.cursor.row = self.cursor.row.min(self.rows - 1);
            }
            None => {
                // Nothing saved: the power-on state.
                self.cursor = Cursor::default();
                self.modes.origin = false;
            }
        }
    }

    /// Blank the alternate screen for its next user, who also starts with
    /// the kitty keyboard flags off.
    fn clear_alternate(&mut self) {
        let rows = self.rows;
        self.alternate.erase_rows(0, rows, Cell::default());
        self.keyboard[1] = Default::default();
    }

    /// DECSTR: the modes and cursor state, not the screen.
    fn soft_reset(&mut self) {
        self.modes = Modes {
            mouse: self.modes.mouse,
            mouse_encoding: self.modes.mouse_encoding,
            focus_events: self.modes.focus_events,
            bracketed_paste: self.modes.bracketed_paste,
            synchronized: self.modes.synchronized,
            ..Modes::default()
        };
        self.top = 0;
        self.bottom = self.rows;
        let Cursor { col, row, .. } = self.cursor;
        self.cursor = Cursor {
            col,
            row,
            ..Cursor::default()
        };
        self.saved = [None; 2];
        self.keyboard = Default::default();
    }

    /// RIS: back to the power-on state. The scrollback stays, and so do
    /// replies not yet sent.
    fn full_reset(&mut self) {
        let scrollback = std::mem::replace(&mut self.scrollback, Scrollback::new(0));
        let reported = std::mem::take(&mut self.reported);
        let replies = std::mem::take(&mut self.replies);
        let cell_pixels = self.cell_pixels;
        *self = Self::new(self.cols, self.rows);
        self.scrollback = scrollback;
        self.reported = reported;
        self.replies = replies;
        self.cell_pixels = cell_pixels;
    }

    /// DECALN: fill the screen with `E` (a test pattern).
    fn screen_alignment(&mut self) {
        let (cols, rows) = (self.cols, self.rows);
        for row in 0..rows {
            for col in 0..cols {
                *self.screen_mut().cell_mut(col, row) = Cell {
                    ch: 'E',
                    style: Style::default(),
                };
            }
        }
        self.top = 0;
        self.bottom = rows;
        self.goto(0, 0);
    }

    // Modes.

    fn set_private_mode(&mut self, mode: u16, on: bool) {
        let m = &mut self.modes;
        match mode {
            1 => m.cursor_keys_app = on,
            6 => {
                m.origin = on;
                self.goto_origin(0, 0);
            }
            7 => {
                m.autowrap = on;
                if !on {
                    self.cursor.pending_wrap = false;
                }
            }
            12 => self.cursor_blink = on,
            25 => m.cursor_visible = on,
            47 => self.alt = on,
            66 => m.keypad_app = on,
            1000 | 1002 | 1003 => {
                m.mouse = match (on, mode) {
                    (false, _) => MouseTracking::Off,
                    (true, 1000) => MouseTracking::Click,
                    (true, 1002) => MouseTracking::Drag,
                    (true, _) => MouseTracking::Motion,
                }
            }
            1004 => m.focus_events = on,
            1005 | 1006 | 1015 => {
                m.mouse_encoding = match (on, mode) {
                    (false, _) => MouseEncoding::X10,
                    (true, 1005) => MouseEncoding::Utf8,
                    (true, 1006) => MouseEncoding::Sgr,
                    (true, _) => MouseEncoding::Urxvt,
                }
            }
            1047 => {
                if on {
                    self.alt = true;
                } else if self.alt {
                    self.clear_alternate();
                    self.alt = false;
                }
            }
            1048 => {
                if on {
                    self.save_cursor();
                } else {
                    self.restore_cursor();
                }
            }
            1049 => {
                if on && !self.alt {
                    self.save_cursor();
                    self.alt = true;
                    self.clear_alternate();
                } else if !on && self.alt {
                    self.alt = false;
                    self.restore_cursor();
                }
            }
            2004 => m.bracketed_paste = on,
            2026 => m.synchronized = on,
            _ => self.unhandled(format!("DECSET/DECRST {mode}")),
        }
    }

    fn set_mode(&mut self, mode: u16, on: bool) {
        match mode {
            4 => self.modes.insert = on,
            20 => self.modes.newline = on,
            _ => self.unhandled(format!("SM/RM {mode}")),
        }
    }

    fn set_cursor_style(&mut self, style: u16) {
        let (shape, blink) = match style {
            0 | 1 => (CursorShape::Block, true),
            2 => (CursorShape::Block, false),
            3 => (CursorShape::Underline, true),
            4 => (CursorShape::Underline, false),
            5 => (CursorShape::Bar, true),
            6 => (CursorShape::Bar, false),
            _ => return self.unhandled(format!("DECSCUSR {style}")),
        };
        self.cursor_shape = shape;
        self.cursor_blink = blink;
    }

    /// A DEC private mode's state for DECRQM: `Some(set)` for the modes
    /// [`Self::set_private_mode`] knows, `None` for the rest.
    fn private_mode(&self, mode: u16) -> Option<bool> {
        let m = &self.modes;
        Some(match mode {
            1 => m.cursor_keys_app,
            6 => m.origin,
            7 => m.autowrap,
            12 => self.cursor_blink,
            25 => m.cursor_visible,
            47 | 1047 | 1049 => self.alt,
            66 => m.keypad_app,
            1000 => m.mouse == MouseTracking::Click,
            1002 => m.mouse == MouseTracking::Drag,
            1003 => m.mouse == MouseTracking::Motion,
            1004 => m.focus_events,
            1005 => m.mouse_encoding == MouseEncoding::Utf8,
            1006 => m.mouse_encoding == MouseEncoding::Sgr,
            1015 => m.mouse_encoding == MouseEncoding::Urxvt,
            // Saving the cursor isn't a state.
            1048 => false,
            2004 => m.bracketed_paste,
            2026 => m.synchronized,
            _ => return None,
        })
    }

    fn ansi_mode(&self, mode: u16) -> Option<bool> {
        match mode {
            4 => Some(self.modes.insert),
            20 => Some(self.modes.newline),
            _ => None,
        }
    }

    // Replies.

    fn reply(&mut self, bytes: &[u8]) {
        self.replies.extend_from_slice(bytes);
    }

    /// DECRQM (`CSI ? Ps $ p`, or `CSI Ps $ p` for ANSI modes): 0 for an
    /// unknown mode, 1 set, 2 reset.
    fn report_mode(&mut self, mode: u16, private: bool) {
        let state = if private {
            self.private_mode(mode)
        } else {
            self.ansi_mode(mode)
        };
        let value = match state {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        };
        let marker = if private { "?" } else { "" };
        self.reply(format!("\x1b[{marker}{mode};{value}$y").as_bytes());
    }

    /// DSR (`CSI n`) and its DEC form (`CSI ? n`).
    fn device_status(&mut self, params: &Params, private: bool) {
        // The cursor row as the program addresses it: relative to the
        // scroll region in origin mode.
        let row = if self.modes.origin {
            self.cursor.row.saturating_sub(self.top)
        } else {
            self.cursor.row
        } + 1;
        let col = self.cursor.col + 1;
        match (private, param(params, 0, 0)) {
            (false, 5) => self.reply(b"\x1b[0n"),
            // CPR.
            (false, 6) => self.reply(format!("\x1b[{row};{col}R").as_bytes()),
            // DECXCPR, with the page number.
            (true, 6) => self.reply(format!("\x1b[?{row};{col};1R").as_bytes()),
            // Color scheme: 1 dark, 2 light.
            (true, 996) => {
                let p = &self.palette;
                let scheme = if reply::is_dark(p.foreground, p.background) {
                    1
                } else {
                    2
                };
                self.reply(format!("\x1b[?997;{scheme}n").as_bytes());
            }
            _ => {
                let marker = if private { &b"?"[..] } else { b"" };
                self.unhandled(describe_csi(params, marker, b'n'));
            }
        }
    }

    /// Window operations (`CSI t`): only the size reports.
    fn window_op(&mut self, params: &Params) {
        let (cw, ch) = self.cell_pixels;
        let known = cw > 0 && ch > 0;
        let (cols, rows) = (u32::from(self.cols), u32::from(self.rows));
        match param(params, 0, 0) {
            // Text area in pixels.
            14 if known => {
                let (w, h) = (cols * u32::from(cw), rows * u32::from(ch));
                self.reply(format!("\x1b[4;{h};{w}t").as_bytes());
            }
            // A cell in pixels.
            16 if known => self.reply(format!("\x1b[6;{ch};{cw}t").as_bytes()),
            // Text area in cells.
            18 => self.reply(format!("\x1b[8;{rows};{cols}t").as_bytes()),
            _ => self.unhandled(describe_csi(params, b"", b't')),
        }
    }

    /// The kitty keyboard protocol's `CSI u` forms (`>`, `<`, `=`, `?`).
    fn keyboard_mode(&mut self, params: &Params, marker: u8) {
        let keyboard = &mut self.keyboard[usize::from(self.alt)];
        let flags = (param(params, 0, 0) & 0xff) as u8;
        match marker {
            b'>' => keyboard.push(flags),
            b'<' => keyboard.pop(param(params, 0, 1)),
            b'=' => {
                if !keyboard.set(flags, param(params, 1, 1)) {
                    self.unhandled(describe_csi(params, b"=", b'u'));
                }
            }
            _ => {
                let flags = keyboard.flags;
                self.reply(format!("\x1b[?{flags}u").as_bytes());
            }
        }
    }

    /// OSC color queries (`?` in place of a color): the color, in the
    /// form OSC 4/10/11/12 would set it.
    fn report_color(&mut self, prefix: &str, rgb: palette::Rgb, bell: bool) {
        self.reply(&reply::osc_color(prefix, rgb, bell));
    }

    /// DECRQSS: `DCS 1 $ r <setting> ST` for the settings we have, or
    /// `DCS 0 $ r ST`.
    fn report_setting(&mut self, request: &[u8]) {
        let setting = match request {
            b"m" => Some(format!("{}m", sgr::describe(&self.cursor.pen))),
            b"r" => Some(format!("{};{}r", self.top + 1, self.bottom)),
            b" q" => {
                let style = match self.cursor_shape {
                    CursorShape::Block => 1,
                    CursorShape::Underline => 3,
                    CursorShape::Bar => 5,
                } + u8::from(!self.cursor_blink);
                Some(format!("{style} q"))
            }
            // Conformance level: VT200, 7-bit controls.
            b"\"p" => Some("62;1\"p".to_owned()),
            _ => None,
        };
        match setting {
            Some(s) => self.reply(format!("\x1bP1$r{s}").as_bytes()),
            None => {
                self.unhandled(format!("DECRQSS {}", request.escape_ascii()));
                self.reply(b"\x1bP0$r");
            }
        }
        self.reply(reply::ST);
    }

    // OSC.

    /// `bell`: the OSC ended with BEL, so replies do too.
    fn osc(&mut self, params: &[&[u8]], bell: bool) {
        let num = std::str::from_utf8(params[0])
            .ok()
            .and_then(|s| s.parse::<u16>().ok());
        let args = &params[1..];
        match num {
            Some(0 | 2) => {
                let title = args
                    .iter()
                    .map(|a| String::from_utf8_lossy(a))
                    .collect::<Vec<_>>()
                    .join(";");
                if title != self.title {
                    tracing::debug!(target: "vellum::term", %title, "window title");
                    self.title = title;
                }
            }
            // Icon name; working directory; hyperlinks; clipboard; shell
            // integration marks. Nothing to do with them (yet).
            Some(1 | 7 | 8 | 52 | 133) => {}
            Some(4) => {
                for pair in args.chunks(2) {
                    let index = std::str::from_utf8(pair[0])
                        .ok()
                        .and_then(|s| s.parse::<u8>().ok());
                    let spec = pair.get(1).copied().unwrap_or_default();
                    match (index, spec) {
                        (Some(i), b"?") => {
                            let rgb = self.palette.colors[usize::from(i)];
                            self.report_color(&format!("4;{i}"), rgb, bell);
                        }
                        (Some(i), spec) if let Some(rgb) = palette::parse_color(spec) => {
                            self.palette.colors[usize::from(i)] = rgb;
                        }
                        _ => self.unhandled(format!("OSC 4 {}", pair.join(&b';').escape_ascii())),
                    }
                }
            }
            Some(104) => {
                if args.iter().all(|a| a.is_empty()) {
                    self.palette.colors = std::array::from_fn(palette::default_color);
                }
                for a in args {
                    if let Some(i) = std::str::from_utf8(a)
                        .ok()
                        .and_then(|s| s.parse::<u8>().ok())
                    {
                        self.palette.colors[usize::from(i)] = palette::default_color(i.into());
                    }
                }
            }
            // Default foreground, background, cursor; each further argument
            // sets the next one.
            Some(n @ 10..=12) => {
                for (which, spec) in (n..=12).zip(args) {
                    if *spec == b"?" {
                        let rgb = *self.dynamic_color(which);
                        self.report_color(&which.to_string(), rgb, bell);
                        continue;
                    }
                    match palette::parse_color(spec) {
                        Some(rgb) => *self.dynamic_color(which) = rgb,
                        None => self.unhandled(format!("OSC {which} {}", spec.escape_ascii())),
                    }
                }
            }
            Some(n @ 110..=112) => {
                let default = Palette::default();
                let color = self.dynamic_color(n - 100);
                *color = match n {
                    110 => default.foreground,
                    111 => default.background,
                    _ => default.cursor,
                };
            }
            _ => self.unhandled(format!("OSC {}", params[0].escape_ascii())),
        }
    }

    fn dynamic_color(&mut self, which: u16) -> &mut palette::Rgb {
        match which {
            10 => &mut self.palette.foreground,
            11 => &mut self.palette.background,
            _ => &mut self.palette.cursor,
        }
    }

    /// Log a sequence we don't handle, once.
    fn unhandled(&mut self, what: String) {
        if self.reported.len() < MAX_REPORTED && self.reported.insert(what.clone()) {
            tracing::debug!(target: "vellum::term", "unhandled: {what}");
        }
    }
}

fn default_tabs(cols: u16) -> Vec<bool> {
    (0..cols).map(|c| c % 8 == 0 && c > 0).collect()
}

/// Parameter `i` (its first value), with 0 or missing meaning `default`.
fn param(params: &Params, i: usize, default: u16) -> u16 {
    match params.iter().nth(i).map_or(0, |g| g[0]) {
        0 => default,
        v => v,
    }
}

/// `CSI ? 1 ; 2 $ p` style, for logs.
fn describe_csi(params: &Params, intermediates: &[u8], action: u8) -> String {
    let (marker, rest) = match intermediates.first() {
        Some(b'<'..=b'?') => intermediates.split_at(1),
        _ => (&[][..], intermediates),
    };
    let params: Vec<String> = params
        .iter()
        .map(|g| g.iter().map(u16::to_string).collect::<Vec<_>>().join(":"))
        .collect();
    format!(
        "CSI {}{}{}{}",
        marker.escape_ascii(),
        params.join(";"),
        rest.escape_ascii(),
        char::from(action)
    )
}

impl Perform for Terminal {
    fn print(&mut self, ch: char) {
        let charset = self.cursor.charsets[usize::from(self.cursor.gl)];
        self.put_char(charset.map(ch));
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            0x08 => {
                self.cursor.pending_wrap = false;
                self.cursor.col = self.cursor.col.saturating_sub(1);
            }
            b'\t' => self.tab_forward(1),
            // LF, VT, FF.
            0x0a..=0x0c => {
                self.index();
                if self.modes.newline {
                    self.carriage_return();
                }
            }
            b'\r' => self.carriage_return(),
            // SO, SI.
            0x0e => self.cursor.gl = 1,
            0x0f => self.cursor.gl = 0,
            // BEL, NUL, and the rest: nothing to do.
            _ => {}
        }
    }

    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], ignored: bool, action: u8) {
        if ignored {
            return self.unhandled(describe_csi(params, intermediates, action));
        }
        let n = param(params, 0, 1);
        let Cursor { col, row, .. } = self.cursor;
        match (intermediates, action) {
            (b"", b'@') => {
                let blank = self.blank();
                self.cursor.pending_wrap = false;
                self.screen_mut().insert_cells(row, col, n, blank);
            }
            (b"", b'A') => self.up(n),
            (b"", b'B') => self.down(n),
            (b"", b'C' | b'a') => self.goto(col.saturating_add(n), row),
            (b"", b'D') => self.goto(col.saturating_sub(n), row),
            (b"", b'E') => {
                self.down(n);
                self.carriage_return();
            }
            (b"", b'F') => {
                self.up(n);
                self.carriage_return();
            }
            (b"", b'G' | b'`') => self.goto(n - 1, row),
            (b"", b'H' | b'f') => {
                let (r, c) = (param(params, 0, 1), param(params, 1, 1));
                self.goto_origin(c - 1, r - 1);
            }
            (b"", b'I') => self.tab_forward(n),
            (b"", b'J') => self.erase_display(param(params, 0, 0)),
            (b"", b'K') => self.erase_line(param(params, 0, 0)),
            (b"", b'L') => self.insert_lines(n),
            (b"", b'M') => self.delete_lines(n),
            (b"", b'P') => {
                let blank = self.blank();
                self.cursor.pending_wrap = false;
                self.screen_mut().delete_cells(row, col, n, blank);
            }
            (b"", b'S') => self.scroll_up(n),
            // With more parameters, `CSI T` is an old mouse sequence.
            (b"", b'T') if params.len() <= 1 => self.scroll_down(n),
            (b"", b'X') => {
                let blank = self.blank();
                self.screen_mut()
                    .erase(row, col, col.saturating_add(n), blank);
            }
            (b"", b'Z') => self.tab_backward(n),
            (b"", b'b') => {
                if let Some(ch) = self.last_printed {
                    let max = usize::from(self.cols) * usize::from(self.rows);
                    for _ in 0..usize::from(n).min(max) {
                        self.put_char(ch);
                    }
                }
            }
            (b"", b'd') => self.goto_origin(col, n - 1),
            (b"", b'e') => self.down(n),
            (b"", b'g') => match param(params, 0, 0) {
                0 => self.tabs[usize::from(col)] = false,
                3 => self.tabs.fill(false),
                _ => self.unhandled(describe_csi(params, intermediates, action)),
            },
            (b"" | b"?", b'h' | b'l') => {
                let on = action == b'h';
                for g in params.iter() {
                    if intermediates.is_empty() {
                        self.set_mode(g[0], on);
                    } else {
                        self.set_private_mode(g[0], on);
                    }
                }
            }
            (b"", b'm') => {
                for code in sgr::apply(&mut self.cursor.pen, params) {
                    self.unhandled(format!("SGR {code}"));
                }
            }
            (b"", b'r') => {
                let rows = self.rows;
                self.set_scroll_region(param(params, 0, 1), param(params, 1, rows));
            }
            (b"", b's') if params.is_empty() => self.save_cursor(),
            (b"", b'u') if params.is_empty() => self.restore_cursor(),
            (b" ", b'q') => self.set_cursor_style(param(params, 0, 0)),
            (b"!", b'p') => self.soft_reset(),
            // Device attributes: primary and secondary.
            (b"", b'c') if param(params, 0, 0) == 0 => self.reply(reply::DA1),
            (b">", b'c') if param(params, 0, 0) == 0 => self.reply(&reply::da2()),
            (b"" | b"?", b'n') => self.device_status(params, !intermediates.is_empty()),
            (b"$" | b"?$", b'p') => {
                let private = intermediates[0] == b'?';
                for g in params.iter() {
                    self.report_mode(g[0], private);
                }
            }
            (b">", b'q') if param(params, 0, 0) == 0 => self.reply(&reply::xtversion()),
            (b"", b't') => self.window_op(params),
            (&[marker @ (b'>' | b'<' | b'=' | b'?')], b'u') => self.keyboard_mode(params, marker),
            _ => self.unhandled(describe_csi(params, intermediates, action)),
        }
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], ignored: bool, byte: u8) {
        let describe = || format!("ESC {}{}", intermediates.escape_ascii(), char::from(byte));
        if ignored {
            return self.unhandled(describe());
        }
        match (intermediates, byte) {
            (b"", b'7') => self.save_cursor(),
            (b"", b'8') => self.restore_cursor(),
            (b"", b'D') => self.index(),
            (b"", b'E') => {
                self.index();
                self.carriage_return();
            }
            (b"", b'H') => self.tabs[usize::from(self.cursor.col)] = true,
            (b"", b'M') => self.reverse_index(),
            (b"", b'c') => self.full_reset(),
            (b"", b'=') => self.modes.keypad_app = true,
            (b"", b'>') => self.modes.keypad_app = false,
            // ST, ending an OSC or DCS string: already handled.
            (b"", b'\\') => {}
            (b"#", b'8') => self.screen_alignment(),
            (&[g @ b'('..=b'+'], f) => {
                let charset = match f {
                    b'B' => Charset::Ascii,
                    b'0' => Charset::DecSpecial,
                    b'A' => Charset::Uk,
                    _ => return self.unhandled(describe()),
                };
                self.cursor.charsets[usize::from(g - b'(')] = charset;
            }
            // Select UTF-8 / default character set: we're always UTF-8.
            (b"%", b'G' | b'@') => {}
            _ => self.unhandled(describe()),
        }
    }

    fn osc_dispatch(&mut self, params: &[&[u8]], bell_terminated: bool) {
        self.osc(params, bell_terminated);
    }

    fn hook(&mut self, params: &Params, intermediates: &[u8], ignored: bool, action: u8) {
        let request = match (intermediates, action) {
            _ if ignored || !params.is_empty() => None,
            (b"$", b'q') => Some(DcsRequest::Setting),
            (b"+", b'q') => Some(DcsRequest::Capabilities),
            _ => None,
        };
        self.dcs = request.map(|request| Dcs {
            request,
            data: Vec::new(),
            overflow: false,
        });
        if self.dcs.is_none() {
            let csi = describe_csi(params, intermediates, action);
            self.unhandled(format!("DCS{}", &csi[3..]));
        }
    }

    fn put(&mut self, byte: u8) {
        if let Some(dcs) = &mut self.dcs {
            if dcs.data.len() < reply::MAX_DCS {
                dcs.data.push(byte);
            } else {
                dcs.overflow = true;
            }
        }
    }

    // A string abandoned with CAN or SUB ends here too, and is answered
    // as if it had ended normally (the parser doesn't say which it was).
    fn unhook(&mut self) {
        let Some(dcs) = self.dcs.take() else { return };
        if dcs.overflow {
            return self.unhandled(format!(
                "DCS {:?} longer than {}",
                dcs.request,
                reply::MAX_DCS
            ));
        }
        match dcs.request {
            DcsRequest::Setting => self.report_setting(&dcs.data),
            DcsRequest::Capabilities => self.reply(&reply::xtgettcap(&dcs.data)),
        }
    }
}
