//! Turning glyph outlines into line segments for the GPU rasterizer.
//!
//! skrifa hands us contours of lines and quadratic/cubic Béziers in pixels,
//! y up. The rasterizer only knows straight lines, so curves are split into
//! enough segments that no point strays more than `tolerance` pixels from
//! the true curve. Output is y down with the bitmap's top-left at (0, 0).
//!
//! Faces a font family lacks can be faked here ([`Synthetic`]): italic by
//! shearing the lines, bold by adding a copy shifted right. With nonzero
//! winding, overlapping copies fill as one shape.

use skrifa::outline::OutlinePen;

/// A line segment in bitmap pixels (y down).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
}

// SAFETY: four f32s, no padding.
unsafe impl crate::vk::Pod for Line {}

/// A flattened glyph: lines in its own bitmap's coordinates, and where that
/// bitmap sits relative to the pen position on the baseline.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outline {
    pub lines: Vec<Line>,
    /// Bitmap size in pixels, including padding. Zero for empty glyphs.
    pub width: u32,
    pub height: u32,
    /// Offset of the bitmap's top-left from the pen position (y down, so
    /// `top` is negative for glyphs that rise above the baseline).
    pub left: i32,
    pub top: i32,
}

/// Empty pixels around the outline, so coverage that spills past the edge
/// pixels (antialiasing) still lands inside the bitmap.
pub const PADDING: i32 = 1;

/// Default maximum distance between a curve and its segments, in pixels.
pub const TOLERANCE: f32 = 0.15;

type Point = [f32; 2];

/// How to fake a missing bold or italic face.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Synthetic {
    /// Widen strokes by this many pixels (to the right).
    pub embolden: f32,
    /// Lean right by this much per pixel above the baseline.
    pub skew: f32,
}

impl Synthetic {
    /// FreeType's slant for synthetic obliques (`FT_GlyphSlot_Oblique`).
    pub const ITALIC_SKEW: f32 = 0.2;

    /// FreeType's emboldening at `size` pixels (`FT_GlyphSlot_Embolden`,
    /// 1/24 em), at least a pixel so it shows.
    pub fn embolden_for(size: f32) -> f32 {
        (size / 24.0).round().max(1.0)
    }

    pub fn is_none(&self) -> bool {
        *self == Self::default()
    }
}

/// An [`OutlinePen`] that flattens what it's given. Feed it a glyph with
/// skrifa's `OutlineGlyph::draw`, then call [`Flattener::finish`].
pub struct Flattener {
    tolerance: f32,
    /// In font orientation (y up); flipped in `finish`.
    lines: Vec<Line>,
    start: Point,
    current: Point,
    open: bool,
    synthetic: Synthetic,
}

impl Flattener {
    pub fn new(tolerance: f32) -> Self {
        Self::with_synthetic(tolerance, Synthetic::default())
    }

    pub fn with_synthetic(tolerance: f32, synthetic: Synthetic) -> Self {
        Self {
            tolerance,
            synthetic,
            lines: Vec::new(),
            start: [0.0; 2],
            current: [0.0; 2],
            open: false,
        }
    }

    fn segment_to(&mut self, p: Point) {
        if p != self.current {
            self.lines.push(Line {
                p0: self.current,
                p1: p,
            });
        }
        self.current = p;
    }

    /// Segments for a curve whose control polygon has largest second
    /// difference `dd` and whose second derivative is at most `k * dd`.
    ///
    /// A chord over a parameter span `h` deviates from the curve by at most
    /// `max|B''| * h² / 8`. With `n` equal steps (h = 1/n), keeping that
    /// under the tolerance needs `n >= sqrt(k * dd / (8 * tol))`.
    fn steps(&self, dd: f32, k: f32) -> u32 {
        ((k * dd / (8.0 * self.tolerance)).sqrt().ceil() as u32).max(1)
    }

    /// The bitmap: fake bold/italic if asked, flip y, pad, and move the
    /// top-left corner to (0, 0).
    pub fn finish(mut self) -> Outline {
        self.close();
        if self.lines.is_empty() {
            return Outline::default();
        }
        let Synthetic { embolden, skew } = self.synthetic;
        if skew != 0.0 {
            // Still y up from the baseline: shear around it.
            for line in &mut self.lines {
                for p in [&mut line.p0, &mut line.p1] {
                    p[0] += skew * p[1];
                }
            }
        }
        if embolden > 0.0 {
            let shifted: Vec<Line> = self
                .lines
                .iter()
                .map(|l| Line {
                    p0: [l.p0[0] + embolden, l.p0[1]],
                    p1: [l.p1[0] + embolden, l.p1[1]],
                })
                .collect();
            self.lines.extend(shifted);
        }
        let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
        for line in &mut self.lines {
            for p in [&mut line.p0, &mut line.p1] {
                p[1] = -p[1];
                for i in 0..2 {
                    min[i] = min[i].min(p[i]);
                    max[i] = max[i].max(p[i]);
                }
            }
        }
        let left = min[0].floor() as i32 - PADDING;
        let top = min[1].floor() as i32 - PADDING;
        let right = max[0].ceil() as i32 + PADDING;
        let bottom = max[1].ceil() as i32 + PADDING;
        let shift = [left as f32, top as f32];
        for line in &mut self.lines {
            for p in [&mut line.p0, &mut line.p1] {
                p[0] -= shift[0];
                p[1] -= shift[1];
            }
        }
        Outline {
            lines: self.lines,
            width: (right - left) as u32,
            height: (bottom - top) as u32,
            left,
            top,
        }
    }
}

fn dist(a: Point, b: Point) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// |a - 2b + c|: how far the middle control point bends the curve.
fn second_difference(a: Point, b: Point, c: Point) -> f32 {
    dist([a[0] + c[0], a[1] + c[1]], [2.0 * b[0], 2.0 * b[1]])
}

impl OutlinePen for Flattener {
    fn move_to(&mut self, x: f32, y: f32) {
        // Contours are closed implicitly if the font doesn't say so.
        self.close();
        self.start = [x, y];
        self.current = [x, y];
        self.open = true;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.segment_to([x, y]);
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (p0, p1, p2) = (self.current, [cx, cy], [x, y]);
        // B''(t) = 2 (p0 - 2 p1 + p2).
        let n = self.steps(second_difference(p0, p1, p2), 2.0);
        for i in 1..=n {
            let t = i as f32 / n as f32;
            let mt = 1.0 - t;
            let point = |k: usize| mt * mt * p0[k] + 2.0 * mt * t * p1[k] + t * t * p2[k];
            self.segment_to([point(0), point(1)]);
        }
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let (p0, p1, p2, p3) = (self.current, [cx0, cy0], [cx1, cy1], [x, y]);
        // B''(t) = 6 ((1 - t)(p0 - 2 p1 + p2) + t (p1 - 2 p2 + p3)).
        let dd = second_difference(p0, p1, p2).max(second_difference(p1, p2, p3));
        let n = self.steps(dd, 6.0);
        for i in 1..=n {
            let t = i as f32 / n as f32;
            let mt = 1.0 - t;
            let point = |k: usize| {
                mt * mt * mt * p0[k]
                    + 3.0 * mt * mt * t * p1[k]
                    + 3.0 * mt * t * t * p2[k]
                    + t * t * t * p3[k]
            };
            self.segment_to([point(0), point(1)]);
        }
    }

    fn close(&mut self) {
        if self.open {
            self.segment_to(self.start);
            self.open = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flatten(draw: impl FnOnce(&mut Flattener)) -> Outline {
        let mut pen = Flattener::new(TOLERANCE);
        draw(&mut pen);
        pen.finish()
    }

    #[test]
    fn square_is_flipped_padded_and_moved_to_the_origin() {
        // 2x3 px box sitting on the baseline, 1 px right of the pen.
        let o = flatten(|p| {
            p.move_to(1.0, 0.0);
            p.line_to(3.0, 0.0);
            p.line_to(3.0, 3.0);
            p.line_to(1.0, 3.0);
            p.close();
        });
        assert_eq!((o.left, o.top), (1 - PADDING, -3 - PADDING));
        assert_eq!(
            (o.width, o.height),
            (2 + 2 * PADDING as u32, 3 + 2 * PADDING as u32)
        );
        let pad = PADDING as f32;
        assert_eq!(o.lines.len(), 4);
        // First edge runs along the baseline: the bitmap's bottom row.
        assert_eq!(
            o.lines[0],
            Line {
                p0: [pad, 3.0 + pad],
                p1: [2.0 + pad, 3.0 + pad]
            }
        );
    }

    fn unit_box(p: &mut Flattener) {
        // 2x4 px box on the baseline at the pen.
        p.move_to(0.0, 0.0);
        p.line_to(2.0, 0.0);
        p.line_to(2.0, 4.0);
        p.line_to(0.0, 4.0);
        p.close();
    }

    #[test]
    fn synthetic_bold_adds_a_shifted_copy() {
        let plain = flatten(unit_box);
        let mut pen = Flattener::with_synthetic(
            TOLERANCE,
            Synthetic {
                embolden: 1.0,
                skew: 0.0,
            },
        );
        unit_box(&mut pen);
        let bold = pen.finish();
        assert_eq!(bold.lines.len(), 2 * plain.lines.len());
        assert_eq!(bold.width, plain.width + 1);
        assert_eq!(
            (bold.left, bold.top, bold.height),
            (plain.left, plain.top, plain.height)
        );
    }

    #[test]
    fn synthetic_italic_leans_right_above_the_baseline() {
        let mut pen = Flattener::with_synthetic(
            TOLERANCE,
            Synthetic {
                embolden: 0.0,
                skew: 0.5,
            },
        );
        unit_box(&mut pen);
        let o = pen.finish();
        let pad = PADDING as f32;
        // The baseline edge stays put; the top edge (4 px up) moves 2 px.
        assert_eq!(o.lines[0].p0, [pad, 4.0 + pad]);
        assert_eq!(o.lines[2].p0, [2.0 + 2.0 + pad, pad]);
        assert_eq!(o.width, 4 + 2 * PADDING as u32);
        assert_eq!(o.left, -PADDING);
    }

    #[test]
    fn embolden_amount_follows_size() {
        assert_eq!(Synthetic::embolden_for(10.0), 1.0);
        assert_eq!(Synthetic::embolden_for(24.0), 1.0);
        assert_eq!(Synthetic::embolden_for(48.0), 2.0);
    }

    #[test]
    fn contours_close_implicitly() {
        let o = flatten(|p| {
            p.move_to(0.0, 0.0);
            p.line_to(4.0, 0.0);
            p.line_to(4.0, 4.0);
            // no close(); a second contour starts
            p.move_to(1.0, 1.0);
            p.line_to(2.0, 1.0);
            p.line_to(2.0, 2.0);
        });
        assert_eq!(o.lines.len(), 6);
        // Every contour ends where it started.
        assert_eq!(o.lines[2].p1, o.lines[0].p0);
        assert_eq!(o.lines[5].p1, o.lines[3].p0);
    }

    #[test]
    fn empty_glyph_has_no_bitmap() {
        assert_eq!(flatten(|_| {}), Outline::default());
        // A lone move_to (a "contour" of one point) draws nothing either.
        assert_eq!(flatten(|p| p.move_to(5.0, 5.0)), Outline::default());
    }

    /// Distance from `p` to the nearest point of the polyline.
    fn distance_to_polyline(p: Point, lines: &[Line]) -> f32 {
        lines
            .iter()
            .map(|l| {
                let d = [l.p1[0] - l.p0[0], l.p1[1] - l.p0[1]];
                let len2 = d[0] * d[0] + d[1] * d[1];
                let t = if len2 == 0.0 {
                    0.0
                } else {
                    (((p[0] - l.p0[0]) * d[0] + (p[1] - l.p0[1]) * d[1]) / len2).clamp(0.0, 1.0)
                };
                dist(p, [l.p0[0] + t * d[0], l.p0[1] + t * d[1]])
            })
            .fold(f32::MAX, f32::min)
    }

    /// Flatten one curve (in pen coordinates) and check sampled curve points
    /// stay within tolerance of the segments.
    fn check_within_tolerance(draw: impl FnOnce(&mut Flattener), curve: impl Fn(f32) -> Point) {
        let mut pen = Flattener::new(TOLERANCE);
        draw(&mut pen);
        // Compare before finish(), in the same (unflipped) coordinates.
        let lines = pen.lines.clone();
        for i in 0..=1000 {
            let p = curve(i as f32 / 1000.0);
            let d = distance_to_polyline(p, &lines);
            assert!(d <= TOLERANCE * 1.01, "t={} off by {d}", i as f32 / 1000.0);
        }
    }

    #[test]
    fn quadratic_stays_within_tolerance() {
        let (p0, p1, p2) = ([0.0, 0.0], [20.0, 40.0], [40.0, 0.0]);
        check_within_tolerance(
            |p| {
                p.move_to(p0[0], p0[1]);
                p.quad_to(p1[0], p1[1], p2[0], p2[1]);
            },
            |t| {
                let mt = 1.0 - t;
                let f = |k: usize| mt * mt * p0[k] + 2.0 * mt * t * p1[k] + t * t * p2[k];
                [f(0), f(1)]
            },
        );
    }

    #[test]
    fn cubic_stays_within_tolerance() {
        let (p0, p1, p2, p3) = ([0.0, 0.0], [0.0, 30.0], [50.0, -30.0], [50.0, 0.0]);
        check_within_tolerance(
            |p| {
                p.move_to(p0[0], p0[1]);
                p.curve_to(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]);
            },
            |t| {
                let mt = 1.0 - t;
                let f = |k: usize| {
                    mt * mt * mt * p0[k]
                        + 3.0 * mt * mt * t * p1[k]
                        + 3.0 * mt * t * t * p2[k]
                        + t * t * t * p3[k]
                };
                [f(0), f(1)]
            },
        );
    }

    #[test]
    fn flat_curves_become_single_segments() {
        let o = flatten(|p| {
            p.move_to(0.0, 0.0);
            p.quad_to(5.0, 0.0, 10.0, 0.0);
            // Collinear, evenly spaced controls: zero second differences.
            p.curve_to(10.0, 3.0, 10.0, 6.0, 10.0, 9.0);
        });
        // quad, cubic, closing line
        assert_eq!(o.lines.len(), 3);
    }

    #[test]
    fn segment_count_grows_as_tolerance_shrinks() {
        let count = |tol| {
            let mut pen = Flattener::new(tol);
            pen.move_to(0.0, 0.0);
            pen.quad_to(50.0, 100.0, 100.0, 0.0);
            pen.lines.len()
        };
        assert!(count(0.01) > count(0.15));
        assert!(count(0.15) > count(2.0));
    }
}
