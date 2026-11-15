//! Where glyphs go in the atlas: a shelf packer.
//!
//! The atlas is cut into horizontal shelves. A glyph goes on the shelf
//! whose height fits it most tightly and that has room left; otherwise a new
//! shelf exactly its height opens below the last. When there's no room for
//! a new shelf, the bottom one may grow taller instead, if the atlas has the
//! height (a short first glyph shouldn't make a shelf useless). Monospace glyphs at one
//! size have similar heights, so shelves fill well. Nothing is freed
//! individually: when the atlas is full, everything is cleared and the
//! glyphs on screen are rasterized again.

/// A rectangle in the atlas, in texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
struct Shelf {
    y: u32,
    height: u32,
    /// Next free column.
    x: u32,
}

#[derive(Debug)]
pub struct Packer {
    width: u32,
    height: u32,
    shelves: Vec<Shelf>,
    /// Top of the unused area below the last shelf.
    next_y: u32,
}

impl Packer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            shelves: Vec::new(),
            next_y: 0,
        }
    }

    /// Room for a `width` x `height` bitmap, or `None` if the atlas is full.
    pub fn alloc(&mut self, width: u32, height: u32) -> Option<Rect> {
        if width == 0 || height == 0 || width > self.width {
            return None;
        }
        let atlas_width = self.width;
        let best = self
            .shelves
            .iter_mut()
            .filter(|s| s.height >= height && s.x + width <= atlas_width)
            .min_by_key(|s| s.height - height);
        let shelf = match best {
            Some(shelf) => shelf,
            None if self.next_y + height <= self.height => {
                self.shelves.push(Shelf {
                    y: self.next_y,
                    height,
                    x: 0,
                });
                self.next_y += height;
                self.shelves.last_mut().unwrap()
            }
            None => {
                // Grow the bottom shelf (nothing is below it).
                let last = self.shelves.last_mut()?;
                if last.x + width > self.width || last.y + height > self.height {
                    return None;
                }
                last.height = height;
                self.next_y = last.y + height;
                last
            }
        };
        let rect = Rect {
            x: shelf.x,
            y: shelf.y,
            width,
            height,
        };
        shelf.x += width;
        Some(rect)
    }

    /// Forget every allocation.
    pub fn clear(&mut self) {
        self.shelves.clear();
        self.next_y = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlaps(a: &Rect, b: &Rect) -> bool {
        a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
    }

    #[test]
    fn fills_a_shelf_then_opens_the_next() {
        let mut p = Packer::new(10, 10);
        assert_eq!(
            p.alloc(4, 3),
            Some(Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 3
            })
        );
        assert_eq!(
            p.alloc(4, 3),
            Some(Rect {
                x: 4,
                y: 0,
                width: 4,
                height: 3
            })
        );
        // Doesn't fit beside them: new shelf.
        assert_eq!(
            p.alloc(4, 3),
            Some(Rect {
                x: 0,
                y: 3,
                width: 4,
                height: 3
            })
        );
        // Fits in the leftover of the first shelf.
        assert_eq!(
            p.alloc(2, 2),
            Some(Rect {
                x: 8,
                y: 0,
                width: 2,
                height: 2
            })
        );
    }

    #[test]
    fn prefers_the_tightest_shelf() {
        let mut p = Packer::new(20, 20);
        p.alloc(5, 8).unwrap(); // shelf 0: height 8, 15 columns left
        p.alloc(16, 4).unwrap(); // too wide for shelf 0: shelf 1, height 4, 4 left
        // Both shelves have room; the exact-height one wins.
        assert_eq!(p.alloc(3, 4).unwrap().y, 8);
        // Too tall for shelf 1: back to shelf 0.
        assert_eq!(p.alloc(3, 6).unwrap().y, 0);
    }

    #[test]
    fn bottom_shelf_grows_when_no_new_shelf_fits() {
        let mut p = Packer::new(10, 6);
        assert_eq!(p.alloc(3, 4).unwrap().y, 0); // shelf 0, height 4
        // A height-6 shelf won't fit below, but shelf 0 can grow to 6.
        assert_eq!(
            p.alloc(3, 6),
            Some(Rect {
                x: 3,
                y: 0,
                width: 3,
                height: 6
            })
        );
        assert_eq!(p.alloc(3, 7), None, "taller than the atlas");
        // Only the bottom shelf grows.
        let mut q = Packer::new(10, 9);
        q.alloc(10, 4).unwrap(); // shelf 0, full width
        q.alloc(2, 4).unwrap(); // shelf 1 at y 4
        assert_eq!(q.alloc(2, 5).unwrap().y, 4, "shelf 1 grows to 5");
        assert_eq!(q.alloc(2, 6), None);
    }

    #[test]
    fn full_atlas_returns_none_until_cleared() {
        let mut p = Packer::new(8, 8);
        assert!(p.alloc(8, 5).is_some());
        assert_eq!(p.alloc(8, 4), None);
        assert!(p.alloc(8, 3).is_some());
        assert_eq!(p.alloc(1, 1), None);
        p.clear();
        assert_eq!(
            p.alloc(8, 8),
            Some(Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8
            })
        );
    }

    #[test]
    fn rejects_empty_and_oversized() {
        let mut p = Packer::new(8, 8);
        assert_eq!(p.alloc(0, 3), None);
        assert_eq!(p.alloc(3, 0), None);
        assert_eq!(p.alloc(9, 1), None);
        assert_eq!(p.alloc(1, 9), None);
    }

    #[test]
    fn allocations_never_overlap_or_leave_the_atlas() {
        let mut p = Packer::new(64, 64);
        let mut rects = Vec::new();
        // Deterministic pseudo-random sizes.
        let mut seed = 12345u32;
        let mut next = |m: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % m + 1
        };
        for _ in 0..500 {
            if let Some(r) = p.alloc(next(12), next(12)) {
                rects.push(r);
            }
        }
        assert!(rects.len() > 20);
        for (i, a) in rects.iter().enumerate() {
            assert!(a.x + a.width <= 64 && a.y + a.height <= 64, "{a:?}");
            for b in &rects[i + 1..] {
                assert!(!overlaps(a, b), "{a:?} {b:?}");
            }
        }
    }
}
