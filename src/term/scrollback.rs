//! Lines that scrolled off the top of the primary screen.
//!
//! Stored with trailing empty cells trimmed: most lines are short, and a
//! cell is 20 bytes, so 10,000 full lines of a wide terminal would be tens
//! of megabytes. The terminal's view (`Terminal::view_line`) reads it
//! while scrolled back with Shift+PgUp.

use std::collections::VecDeque;

use super::screen::Line;

pub const DEFAULT_LIMIT: usize = 10_000;

pub struct Scrollback {
    lines: VecDeque<Line>,
    limit: usize,
}

impl Scrollback {
    pub fn new(limit: usize) -> Self {
        Self {
            lines: VecDeque::new(),
            limit,
        }
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Line `i`, oldest first. May be shorter than the screen is wide.
    pub fn line(&self, i: usize) -> &Line {
        &self.lines[i]
    }

    pub fn push(&mut self, mut line: Line) {
        if self.limit == 0 {
            return;
        }
        let used = line
            .cells
            .iter()
            .rposition(|c| !c.is_empty())
            .map_or(0, |i| i + 1);
        line.cells.truncate(used);
        line.cells.shrink_to_fit();
        if self.lines.len() == self.limit {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Keep at most `limit` lines from now on, dropping the oldest ones
    /// over it.
    pub fn set_limit(&mut self, limit: usize) {
        self.limit = limit;
        let excess = self.lines.len().saturating_sub(limit);
        self.lines.drain(..excess);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::cell::{Cell, Color, Style};

    fn line(text: &str, cols: u16) -> Line {
        let mut l = Line::new(cols, Cell::default());
        for (c, ch) in l.cells.iter_mut().zip(text.chars()) {
            c.ch = ch;
        }
        l
    }

    #[test]
    fn keeps_the_newest_lines_trimmed() {
        let mut sb = Scrollback::new(2);
        for t in ["one", "two", "three"] {
            sb.push(line(t, 80));
        }
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.line(0).text(), "two");
        assert_eq!(sb.line(1).cells.len(), 5);
        sb.clear();
        assert!(sb.is_empty());
    }

    #[test]
    fn colored_blanks_are_kept() {
        let mut sb = Scrollback::new(10);
        let mut l = line("x", 10);
        l.cells[5].style = Style {
            bg: Color::Indexed(1),
            ..Style::default()
        };
        sb.push(l);
        assert_eq!(sb.line(0).cells.len(), 6);
    }

    #[test]
    fn lowering_the_limit_drops_the_oldest() {
        let mut sb = Scrollback::new(10);
        for t in ["one", "two", "three"] {
            sb.push(line(t, 80));
        }
        sb.set_limit(2);
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.line(0).text(), "two");
        sb.push(line("four", 80));
        assert_eq!(sb.line(0).text(), "three");
    }

    #[test]
    fn zero_limit_keeps_nothing() {
        let mut sb = Scrollback::new(0);
        sb.push(line("x", 10));
        assert!(sb.is_empty());
    }
}
