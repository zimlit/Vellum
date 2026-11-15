//! A grid of lines and the editing operations on it. Knows nothing about
//! the cursor or modes; [`Terminal`](super::Terminal) decides what to do.
//!
//! Row ranges are half-open (`top..bottom`), column indexes 0-based. All
//! operations clamp counts to what's there, so callers can pass the raw
//! parameter.

use super::cell::Cell;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    pub cells: Vec<Cell>,
    /// The text continues on the next line (it was wrapped there, not
    /// broken by a newline). Kept for reflow on resize, later.
    pub wrapped: bool,
}

impl Line {
    pub fn new(cols: u16, blank: Cell) -> Self {
        Self {
            cells: vec![blank; usize::from(cols)],
            wrapped: false,
        }
    }

    /// The characters, trailing blanks trimmed (for tests and debugging).
    pub fn text(&self) -> String {
        let s: String = self.cells.iter().map(|c| c.ch).collect();
        s.trim_end().to_owned()
    }
}

pub struct Screen {
    cols: u16,
    lines: Vec<Line>,
}

impl Screen {
    pub fn new(cols: u16, rows: u16) -> Self {
        Self {
            cols,
            lines: (0..rows)
                .map(|_| Line::new(cols, Cell::default()))
                .collect(),
        }
    }

    pub fn rows(&self) -> u16 {
        self.lines.len() as u16
    }

    pub fn line(&self, row: u16) -> &Line {
        &self.lines[usize::from(row)]
    }

    pub fn line_mut(&mut self, row: u16) -> &mut Line {
        &mut self.lines[usize::from(row)]
    }

    pub fn cell_mut(&mut self, col: u16, row: u16) -> &mut Cell {
        &mut self.lines[usize::from(row)].cells[usize::from(col)]
    }

    /// Move lines `top..bottom` up by `n`. Returns the lines that went off
    /// the top (oldest first); blank lines come in at the bottom.
    pub fn scroll_up(&mut self, top: u16, bottom: u16, n: u16, blank: Cell) -> Vec<Line> {
        let region = &mut self.lines[usize::from(top)..usize::from(bottom)];
        let n = usize::from(n).min(region.len());
        region.rotate_left(n);
        let len = region.len();
        region[len - n..]
            .iter_mut()
            .map(|l| std::mem::replace(l, Line::new(self.cols, blank)))
            .collect()
    }

    /// Move lines `top..bottom` down by `n`; the ones pushed past `bottom`
    /// are gone, blank lines come in at the top.
    pub fn scroll_down(&mut self, top: u16, bottom: u16, n: u16, blank: Cell) {
        let region = &mut self.lines[usize::from(top)..usize::from(bottom)];
        let n = usize::from(n).min(region.len());
        region.rotate_right(n);
        for l in &mut region[..n] {
            *l = Line::new(self.cols, blank);
        }
    }

    /// Insert `n` blanks at `col`, pushing the rest of the line right (and
    /// off the edge).
    pub fn insert_cells(&mut self, row: u16, col: u16, n: u16, blank: Cell) {
        let cells = &mut self.line_mut(row).cells[usize::from(col)..];
        let n = usize::from(n).min(cells.len());
        cells.rotate_right(n);
        cells[..n].fill(blank);
    }

    /// Delete `n` cells at `col`, pulling the rest of the line left; blanks
    /// come in at the right edge.
    pub fn delete_cells(&mut self, row: u16, col: u16, n: u16, blank: Cell) {
        let cells = &mut self.line_mut(row).cells[usize::from(col)..];
        let n = usize::from(n).min(cells.len());
        cells.rotate_left(n);
        let len = cells.len();
        cells[len - n..].fill(blank);
    }

    /// Blank columns `start..end` of `row` (clamped to the line).
    pub fn erase(&mut self, row: u16, start: u16, end: u16, blank: Cell) {
        let line = self.line_mut(row);
        let end = usize::from(end).min(line.cells.len());
        let start = usize::from(start).min(end);
        line.cells[start..end].fill(blank);
        if end == line.cells.len() {
            line.wrapped = false;
        }
    }

    /// Blank whole rows `top..bottom`.
    pub fn erase_rows(&mut self, top: u16, bottom: u16, blank: Cell) {
        for row in top..bottom {
            self.erase(row, 0, self.cols, blank);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::cell::{Color, Flags, Style, Underline};

    fn screen(rows: &[&str]) -> Screen {
        let cols = rows.iter().map(|r| r.len()).max().unwrap() as u16;
        let mut s = Screen::new(cols, rows.len() as u16);
        for (r, text) in rows.iter().enumerate() {
            for (c, ch) in text.chars().enumerate() {
                s.cell_mut(c as u16, r as u16).ch = ch;
            }
        }
        s
    }

    fn text(s: &Screen) -> Vec<String> {
        (0..s.rows()).map(|r| s.line(r).text()).collect()
    }

    /// A blank with a background, as erasing with a colored pen leaves.
    const BLANK: Cell = Cell::blank(Style {
        fg: Color::Default,
        bg: Color::Indexed(4),
        underline_color: Color::Default,
        flags: Flags::empty(),
        underline: Underline::None,
    });

    #[test]
    fn scroll_up_returns_what_went_off() {
        let mut s = screen(&["a", "b", "c", "d"]);
        let gone = s.scroll_up(0, 3, 2, BLANK);
        assert_eq!(gone.iter().map(Line::text).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(text(&s), ["c", "", "", "d"]);
        assert_eq!(s.line(1).cells[0], BLANK);
        // More than the region: everything goes.
        assert_eq!(s.scroll_up(1, 3, 9, BLANK).len(), 2);
    }

    #[test]
    fn scroll_down_within_a_region() {
        let mut s = screen(&["a", "b", "c", "d"]);
        s.scroll_down(1, 4, 1, BLANK);
        assert_eq!(text(&s), ["a", "", "b", "c"]);
        s.scroll_down(0, 4, 10, BLANK);
        assert_eq!(text(&s), ["", "", "", ""]);
    }

    #[test]
    fn insert_and_delete_cells() {
        let mut s = screen(&["abcdef"]);
        s.insert_cells(0, 1, 2, Cell::default());
        assert_eq!(s.line(0).text(), "a  bcd");
        s.delete_cells(0, 0, 3, Cell::default());
        assert_eq!(s.line(0).text(), "bcd");
        s.insert_cells(0, 4, 9, Cell::default());
        assert_eq!(s.line(0).text(), "bcd");
        s.delete_cells(0, 5, 9, BLANK);
        assert_eq!(s.line(0).cells[5], BLANK);
    }

    #[test]
    fn erase_clamps() {
        let mut s = screen(&["abcdef", "ghijkl"]);
        s.erase(0, 2, 4, Cell::default());
        s.erase(1, 4, 99, Cell::default());
        assert_eq!(text(&s), ["ab  ef", "ghij"]);
        s.erase_rows(0, 2, Cell::default());
        assert_eq!(text(&s), ["", ""]);
    }
}
