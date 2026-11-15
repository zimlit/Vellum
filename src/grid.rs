//! A raw character grid: shell output written cell by cell, no escape
//! sequences (those are roadmap step 5, which replaces this).
//!
//! Handles UTF-8 (including characters split across reads), CR, LF, BS
//! and TAB, wraps at the right edge and scrolls at the bottom. ESC and other
//! control characters are dropped, so escape sequences show up as their
//! printable tail (e.g. `[?2004h`).

/// One screen cell. `' '` is empty.
pub type Cell = char;

pub struct Grid {
    cols: u16,
    rows: u16,
    cells: Vec<Cell>,
    /// Cursor column and row. The column can equal `cols` after writing the
    /// last column ("pending wrap"): the next character wraps first.
    cursor: (u16, u16),
    /// Bytes of a UTF-8 sequence cut off by the end of the last read.
    partial: Vec<u8>,
}

impl Grid {
    pub fn new(cols: u16, rows: u16) -> Self {
        let (cols, rows) = (cols.max(1), rows.max(1));
        Self {
            cols,
            rows,
            cells: vec![' '; usize::from(cols) * usize::from(rows)],
            cursor: (0, 0),
            partial: Vec::new(),
        }
    }

    pub fn cols(&self) -> u16 {
        self.cols
    }

    pub fn rows(&self) -> u16 {
        self.rows
    }

    /// Cursor position, clamped to the grid.
    pub fn cursor(&self) -> (u16, u16) {
        (self.cursor.0.min(self.cols - 1), self.cursor.1)
    }

    pub fn cell(&self, col: u16, row: u16) -> Cell {
        self.cells[usize::from(row) * usize::from(self.cols) + usize::from(col)]
    }

    /// Text of one row, trailing blanks trimmed (for tests and debugging).
    pub fn row_text(&self, row: u16) -> String {
        let start = usize::from(row) * usize::from(self.cols);
        let line: String = self.cells[start..start + usize::from(self.cols)]
            .iter()
            .collect();
        line.trim_end().to_owned()
    }

    /// Write what the program printed.
    pub fn feed(&mut self, bytes: &[u8]) {
        let mut data = std::mem::take(&mut self.partial);
        data.extend_from_slice(bytes);
        let mut rest = &data[..];
        while !rest.is_empty() {
            match std::str::from_utf8(rest) {
                Ok(text) => {
                    self.feed_str(text);
                    break;
                }
                Err(e) => {
                    let (valid, after) = rest.split_at(e.valid_up_to());
                    // from_utf8 reported this prefix valid.
                    self.feed_str(std::str::from_utf8(valid).unwrap());
                    match e.error_len() {
                        // Cut off at the end: wait for the rest.
                        None => {
                            self.partial = after.to_vec();
                            break;
                        }
                        Some(n) => {
                            self.put('\u{FFFD}');
                            rest = &after[n..];
                        }
                    }
                }
            }
        }
    }

    fn feed_str(&mut self, text: &str) {
        for ch in text.chars() {
            match ch {
                '\r' => self.cursor.0 = 0,
                '\n' => self.line_feed(),
                '\x08' => self.cursor.0 = self.cursor().0.saturating_sub(1),
                '\t' => {
                    let next = (self.cursor.0 / 8 + 1) * 8;
                    self.cursor.0 = next.min(self.cols - 1);
                }
                c if c.is_control() => {}
                c => self.put(c),
            }
        }
    }

    fn put(&mut self, ch: char) {
        if self.cursor.0 >= self.cols {
            self.cursor.0 = 0;
            self.line_feed();
        }
        let (col, row) = self.cursor;
        self.cells[usize::from(row) * usize::from(self.cols) + usize::from(col)] = ch;
        self.cursor.0 += 1;
    }

    fn line_feed(&mut self) {
        if self.cursor.1 + 1 < self.rows {
            self.cursor.1 += 1;
        } else {
            let cols = usize::from(self.cols);
            self.cells.copy_within(cols.., 0);
            let len = self.cells.len();
            self.cells[len - cols..].fill(' ');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(g: &Grid) -> Vec<String> {
        (0..g.rows()).map(|r| g.row_text(r)).collect()
    }

    #[test]
    fn prints_and_moves_with_cr_lf() {
        let mut g = Grid::new(10, 3);
        g.feed(b"hello\r\nworld");
        assert_eq!(rows(&g), ["hello", "world", ""]);
        assert_eq!(g.cursor(), (5, 1));
        g.feed(b"\rW");
        assert_eq!(g.row_text(1), "World");
    }

    #[test]
    fn wraps_at_the_edge_and_scrolls_at_the_bottom() {
        let mut g = Grid::new(4, 2);
        g.feed(b"abcd");
        // Pending wrap: cursor stays on the line until the next character.
        assert_eq!(rows(&g), ["abcd", ""]);
        assert_eq!(g.cursor(), (3, 0));
        g.feed(b"efghij");
        assert_eq!(rows(&g), ["efgh", "ij"]);
        g.feed(b"\r\n");
        assert_eq!(rows(&g), ["ij", ""]);
    }

    #[test]
    fn backspace_and_tab() {
        let mut g = Grid::new(20, 1);
        g.feed(b"ab\x08c\tx");
        assert_eq!(g.row_text(0), "ac      x");
        g.feed(b"\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08Z");
        assert_eq!(g.row_text(0), "Zc      x");
        // Tab stops at the last column.
        g.feed(b"\r\t\t\t\t");
        assert_eq!(g.cursor(), (19, 0));
    }

    #[test]
    fn utf8_split_across_reads() {
        let mut g = Grid::new(10, 1);
        let text = "é€😀".as_bytes();
        for b in text {
            g.feed(std::slice::from_ref(b));
        }
        assert_eq!(g.row_text(0), "é€😀");
    }

    #[test]
    fn invalid_utf8_becomes_replacement_characters() {
        let mut g = Grid::new(10, 1);
        g.feed(b"a\xffb\xc3(");
        assert_eq!(g.row_text(0), "a\u{FFFD}b\u{FFFD}(");
    }

    #[test]
    fn escape_and_other_controls_are_dropped() {
        let mut g = Grid::new(20, 1);
        g.feed(b"\x1b[?2004h$ \x07\x00");
        assert_eq!(g.row_text(0), "[?2004h$");
        assert_eq!(g.cursor(), (9, 0));
    }

    #[test]
    fn degenerate_sizes_are_clamped() {
        let mut g = Grid::new(0, 0);
        g.feed(b"xy\r\nz");
        assert_eq!((g.cols(), g.rows()), (1, 1));
        assert_eq!(g.row_text(0), "z");
    }
}
