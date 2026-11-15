//! Drawing the terminal's screen: glyph cache, atlas, and the instanced cell pass.
//!
//! Each frame builds one instance per non-empty cell (plus the cursor),
//! rasterizes glyphs not seen before into the atlas (compute, before the
//! render pass), clears the target to the background and draws all
//! instances in one call. See `shaders/cell.vert` and `shaders/cell.frag`.
//!
//! The whole grid is redrawn every frame; damage tracking comes later.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use skrifa::GlyphId;

use crate::font::{CellMetrics, Font};
use crate::raster::{AtlasFull, Rasterizer, Rect};
use crate::spirv;
use crate::term::Terminal;
use crate::vk::{
    self, Blend, Buffer, BufferUsageFlags, DescriptorSet, Device, GraphicsPipeline, Layout,
    Location, Pod, Recorder, RenderTarget,
};

/// Atlas size. 2048² RGBA8 is 16 MiB and holds thousands of glyphs at
/// typical sizes.
const ATLAS_SIZE: u32 = 2048;

/// `FLAG_SOLID` in `shaders/cell.glsl`.
const FLAG_SOLID: u32 = 1;

/// `struct Instance` in `shaders/cell.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Instance {
    cell: u32,
    offset: u32,
    atlas: u32,
    size: u32,
    color: u32,
    flags: u32,
}
unsafe impl Pod for Instance {}

/// The push constant block in `shaders/cell.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Push {
    instances: u64,
    screen: [u32; 2],
    cell: [u32; 2],
    origin: [u32; 2],
}
unsafe impl Pod for Push {}

/// Two 16-bit values in one word, low half first.
fn pack(lo: u16, hi: u16) -> u32 {
    u32::from(lo) | u32::from(hi) << 16
}

fn pack_signed(lo: i32, hi: i32) -> u32 {
    pack(lo as i16 as u16, hi as i16 as u16)
}

/// Straight-alpha RGBA as the shader's `unpackUnorm4x8` expects it.
fn pack_color(rgba: [u8; 4]) -> u32 {
    u32::from_le_bytes(rgba)
}

#[derive(Clone, Copy, Debug)]
pub struct Colors {
    pub foreground: [u8; 4],
    pub background: [f32; 4],
    pub cursor: [u8; 4],
}

impl Default for Colors {
    fn default() -> Self {
        Self {
            foreground: [0xd8, 0xd8, 0xd8, 0xff],
            background: [0.07, 0.07, 0.08, 1.0],
            cursor: [0xd8, 0xd8, 0xd8, 0x99],
        }
    }
}

/// A glyph in the atlas, and where its bitmap goes relative to the cell's
/// top-left corner.
#[derive(Clone, Copy, Debug)]
struct Placed {
    rect: Rect,
    offset: (i32, i32),
}

pub struct TextRenderer {
    device: Arc<Device>,
    font: Font,
    raster: Rasterizer,
    /// `None`: the character draws nothing (e.g. space).
    cache: HashMap<char, Option<Placed>>,
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    /// Scratch for building a frame's instances.
    instances: Vec<Instance>,
    /// One instance buffer per frame slot, so a frame in flight keeps its
    /// own while the next is written.
    buffers: Vec<Option<Arc<Buffer>>>,
}

impl TextRenderer {
    /// `slots`: how many frames can be in flight (one per scanout buffer).
    pub fn new(device: &Arc<Device>, font: Font, format: vk::Format, slots: usize) -> Result<Self> {
        Self::with_atlas(device, font, format, slots, ATLAS_SIZE, ATLAS_SIZE)
    }

    fn with_atlas(
        device: &Arc<Device>,
        font: Font,
        format: vk::Format,
        slots: usize,
        atlas_width: u32,
        atlas_height: u32,
    ) -> Result<Self> {
        let raster = Rasterizer::new(device, atlas_width, atlas_height)?;
        let layout = Layout::new(device, 1, size_of::<Push>() as u32)?;
        let pipeline = GraphicsPipeline::new(
            &layout,
            spirv!("cell.vert"),
            spirv!("cell.frag"),
            format,
            Blend::PremultipliedAlpha,
        )?;
        let set = DescriptorSet::new(&layout, &[raster.atlas()])?;
        Ok(Self {
            device: device.clone(),
            font,
            raster,
            cache: HashMap::new(),
            pipeline,
            set,
            instances: Vec::new(),
            buffers: vec![None; slots],
        })
    }

    pub fn cell(&self) -> CellMetrics {
        self.font.cell()
    }

    /// Record a frame of `term`'s screen into `target`, using instance
    /// buffer `slot`.
    pub fn draw<T: RenderTarget>(
        &mut self,
        rec: &mut Recorder,
        target: &Arc<T>,
        slot: usize,
        term: &Terminal,
        colors: &Colors,
    ) -> Result<()> {
        if let Err(e) = self.build(term, colors) {
            if e.downcast_ref::<AtlasFull>().is_none() {
                return Err(e);
            }
            // Start over with an empty atlas, holding only what's on screen.
            tracing::debug!(glyphs = self.cache.len(), "glyph atlas full; clearing it");
            self.raster.clear();
            self.cache.clear();
            self.build(term, colors)
                .context("the glyphs on screen don't fit in the atlas")?;
        }
        self.raster.record(rec)?;

        let count = self.instances.len() as u32;
        let buffer = self.instance_buffer(slot)?;
        buffer.write(0, &self.instances)?;
        rec.using(&[&buffer]);

        let (width, height) = target.size();
        let cell = self.cell();
        let push = Push {
            instances: buffer.address(),
            screen: [width, height],
            cell: [cell.width, cell.height],
            origin: grid_origin(cell, term, width, height),
        };
        rec.begin_rendering(target, Some(colors.background))?;
        if count > 0 {
            rec.draw(&self.pipeline, Some(&self.set), &push, 6, count)?;
        }
        rec.end_rendering()
    }

    /// Fill `self.instances` for `term`, queueing new glyphs. Fails with
    /// [`AtlasFull`] (inside the anyhow error) when the atlas runs out.
    fn build(&mut self, term: &Terminal, colors: &Colors) -> Result<()> {
        self.instances.clear();
        let fg = pack_color(colors.foreground);
        for row in 0..term.rows() {
            for col in 0..term.cols() {
                let ch = term.cell(col, row).ch;
                if ch == ' ' {
                    continue;
                }
                if let Some(p) = self.glyph(ch)? {
                    self.instances.push(Instance {
                        cell: pack(col, row),
                        offset: pack_signed(p.offset.0, p.offset.1),
                        atlas: pack(p.rect.x as u16, p.rect.y as u16),
                        size: pack(p.rect.width as u16, p.rect.height as u16),
                        color: fg,
                        flags: 0,
                    });
                }
            }
        }
        let cursor = term.cursor();
        let (col, row) = (cursor.col, cursor.row);
        let cell = self.cell();
        self.instances.push(Instance {
            cell: pack(col, row),
            offset: 0,
            atlas: 0,
            size: pack(cell.width as u16, cell.height as u16),
            color: pack_color(colors.cursor),
            flags: FLAG_SOLID,
        });
        Ok(())
    }

    /// The atlas placement for `ch`, rasterizing it if new.
    fn glyph(&mut self, ch: char) -> Result<Option<Placed>> {
        if let Some(&placed) = self.cache.get(&ch) {
            return Ok(placed);
        }
        // Missing characters show the font's .notdef glyph (usually a box).
        let id = self.font.glyph_id(ch).unwrap_or(GlyphId::NOTDEF);
        let glyph = self.font.glyph(id)?;
        let placed = self.raster.queue(&glyph.outline)?.map(|rect| Placed {
            rect,
            offset: (
                glyph.outline.left,
                self.font.cell().baseline as i32 + glyph.outline.top,
            ),
        });
        self.cache.insert(ch, placed);
        Ok(placed)
    }

    /// The instance buffer for `slot`, grown to fit this frame.
    fn instance_buffer(&mut self, slot: usize) -> Result<Arc<Buffer>> {
        let needed = (self.instances.len().max(1) * size_of::<Instance>()) as u64;
        let current = &mut self.buffers[slot];
        match current {
            Some(b) if b.size() >= needed => Ok(b.clone()),
            _ => {
                let b = Buffer::new(
                    &self.device,
                    needed.next_power_of_two(),
                    BufferUsageFlags::empty(),
                    Location::Upload,
                )?;
                *current = Some(b.clone());
                Ok(b)
            }
        }
    }
}

/// How many columns and rows of `cell` fit in `width` x `height` pixels.
pub fn grid_size(cell: CellMetrics, width: u32, height: u32) -> (u16, u16) {
    let cols = (width / cell.width).clamp(1, u32::from(u16::MAX));
    let rows = (height / cell.height).clamp(1, u32::from(u16::MAX));
    (cols as u16, rows as u16)
}

/// Top-left of the grid: centred, splitting the leftover pixels.
fn grid_origin(cell: CellMetrics, term: &Terminal, width: u32, height: u32) -> [u32; 2] {
    let used_w = cell.width * u32::from(term.cols());
    let used_h = cell.height * u32::from(term.rows());
    [
        width.saturating_sub(used_w) / 2,
        height.saturating_sub(used_h) / 2,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::fontconfig;
    use crate::testutil;
    use crate::vk::{Access, CommandBuffer, Format, Image, ImageLayout, ImageUsageFlags};

    const CELL: CellMetrics = CellMetrics {
        width: 10,
        height: 20,
        baseline: 15,
    };

    #[test]
    fn packing_matches_the_shader() {
        assert_eq!(pack(3, 7), 3 | 7 << 16);
        assert_eq!(pack_signed(-1, 2), 0xffff | 2 << 16);
        // The shader sign-extends: (int(v << 16) >> 16, int(v) >> 16).
        let v = pack_signed(-5, -300);
        assert_eq!(((v << 16) as i32 >> 16, v as i32 >> 16), (-5, -300));
        assert_eq!(pack_color([1, 2, 3, 4]), 0x0403_0201);
        assert_eq!(size_of::<Instance>(), 24);
        assert_eq!(size_of::<Push>(), 32);
    }

    #[test]
    fn grid_fits_and_centres() {
        assert_eq!(grid_size(CELL, 105, 41), (10, 2));
        assert_eq!(grid_size(CELL, 3, 3), (1, 1));
        let term = Terminal::new(10, 2);
        assert_eq!(grid_origin(CELL, &term, 105, 41), [2, 0]);
        assert_eq!(grid_origin(CELL, &term, 100, 40), [0, 0]);
    }

    /// White text on black, red cursor, rendered offscreen and read back.
    struct Fixture {
        device: Arc<Device>,
        renderer: TextRenderer,
        cell: CellMetrics,
    }

    struct Frame {
        pixels: Vec<[u8; 4]>,
        width: u32,
        cell: CellMetrics,
    }

    impl Fixture {
        /// `atlas_for`: size the atlas to one shelf as tall as the tallest
        /// glyph in these screens and as wide as the widest screen's glyphs
        /// side by side, so each fits alone but not together.
        fn new(atlas_for: Option<&[&str]>) -> Self {
            let card = testutil::card();
            let rdev = rustix::fs::fstat(&card).unwrap().st_rdev;
            let instance = vk::Instance::new(cfg!(debug_assertions)).unwrap();
            let device =
                Device::for_drm_node(&instance, rustix::fs::major(rdev), rustix::fs::minor(rdev))
                    .unwrap();
            let m = fontconfig::Config::new()
                .unwrap()
                .match_font("monospace:pixelsize=16", 96.0)
                .unwrap();
            let font = Font::load(&m).unwrap();
            let cell = font.cell();
            let (aw, ah) = match atlas_for {
                Some(screens) => {
                    let size = |ch| {
                        let o = font.glyph(font.glyph_id(ch).unwrap()).unwrap().outline;
                        (o.width, o.height)
                    };
                    let width = screens
                        .iter()
                        .map(|s| s.chars().map(|c| size(c).0).sum::<u32>())
                        .max()
                        .unwrap();
                    let height = screens
                        .iter()
                        .flat_map(|s| s.chars())
                        .map(|c| size(c).1)
                        .max()
                        .unwrap();
                    (width, height)
                }
                None => (ATLAS_SIZE, ATLAS_SIZE),
            };
            let renderer =
                TextRenderer::with_atlas(&device, font, Format::R8G8B8A8_UNORM, 1, aw, ah).unwrap();
            Self {
                device,
                renderer,
                cell,
            }
        }

        fn frame(&mut self, text: &[u8], cols: u16, rows: u16) -> Result<Frame> {
            let device = &self.device;
            let (width, height) = (
                self.cell.width * u32::from(cols),
                self.cell.height * u32::from(rows),
            );
            let target = Image::new(
                device,
                width,
                height,
                Format::R8G8B8A8_UNORM,
                ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
            )?;
            let mut term = Terminal::new(cols, rows);
            term.feed(text);
            let colors = Colors {
                foreground: [255, 255, 255, 255],
                background: [0.0, 0.0, 0.0, 1.0],
                cursor: [255, 0, 0, 255],
            };
            let out = Buffer::new(
                device,
                u64::from(width * height) * 4,
                BufferUsageFlags::TRANSFER_DST,
                Location::Readback,
            )?;
            let mut cmd = CommandBuffer::new(device)?;
            let mut rec = cmd.begin()?;
            rec.transition(
                &target,
                ImageLayout::UNDEFINED,
                ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                Access::NONE,
                Access::COLOR_ATTACHMENT,
            );
            self.renderer.draw(&mut rec, &target, 0, &term, &colors)?;
            rec.transition(
                &target,
                ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                Access::COLOR_ATTACHMENT,
                Access::TRANSFER_READ,
            );
            rec.copy_image_to_buffer(&target, &out)?;
            rec.barrier(Access::TRANSFER_WRITE, Access::HOST_READ);
            rec.submit(None)?;
            cmd.wait()?;
            let pixels: Vec<[u8; 4]> = out.read(0, (width * height) as usize)?;
            // VELLUM_DUMP_FRAME=path.ppm: save the frame to look at.
            if let Some(path) = std::env::var_os("VELLUM_DUMP_FRAME") {
                let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
                ppm.extend(pixels.iter().flat_map(|p| [p[0], p[1], p[2]]));
                std::fs::write(path, ppm)?;
            }
            Ok(Frame {
                pixels,
                width,
                cell: self.cell,
            })
        }
    }

    impl Frame {
        /// Sum of the green channel over one cell: glyph coverage, since the
        /// text is white and the cursor red.
        fn ink(&self, col: u32, row: u32) -> u32 {
            let mut sum = 0;
            for y in row * self.cell.height..(row + 1) * self.cell.height {
                for x in col * self.cell.width..(col + 1) * self.cell.width {
                    sum += u32::from(self.pixels[(y * self.width + x) as usize][1]);
                }
            }
            sum
        }

        fn is_cursor(&self, col: u32, row: u32) -> bool {
            let x = col * self.cell.width + self.cell.width / 2;
            let y = row * self.cell.height + self.cell.height / 2;
            self.pixels[(y * self.width + x) as usize] == [255, 0, 0, 255]
        }
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn draws_text_where_the_grid_has_it() {
        let f = Fixture::new(None).frame(b"ab\r\n i", 4, 2).unwrap();
        assert!(f.ink(0, 0) > 0 && f.ink(1, 0) > 0, "a and b drawn");
        assert_eq!(f.ink(2, 0), 0);
        assert_eq!(f.ink(0, 1), 0, "space");
        assert!(f.ink(1, 1) > 0, "i drawn");
        assert!(f.ink(1, 1) < f.ink(0, 0), "i has less ink than a");
        assert!(f.is_cursor(2, 1));
        assert!(!f.is_cursor(0, 0));
    }

    /// A sample screen, mostly for VELLUM_DUMP_FRAME.
    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn draws_a_sample_screen() {
        let text = "$ ls -la ~/projects/vellum\r\nThe quick brown fox jumps over the lazy dog.\r\n\
                    0123456789 {}[]()<> @#$%^&*_+=|\\/?!~`'\":;,.\r\nÄÖÜ é ß ñ → ✓ 漢\r\n$ ";
        let f = Fixture::new(None).frame(text.as_bytes(), 50, 5).unwrap();
        assert!((0..10).all(|c| f.ink(c, 1) > 0 || c == 3 || c == 9));
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn same_text_draws_the_same_from_cache() {
        let mut fixture = Fixture::new(None);
        let first = fixture.frame(b"xyz", 4, 1).unwrap();
        let again = fixture.frame(b"xyz", 4, 1).unwrap();
        assert_eq!(first.pixels, again.pixels);
        assert_eq!(fixture.renderer.cache.len(), 3);
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn full_atlas_is_cleared_for_a_new_screen() {
        let mut fixture = Fixture::new(Some(&["abc", "def"]));
        let abc = fixture.frame(b"abc", 4, 1).unwrap();
        // Three new glyphs don't fit beside the first three: cleared, redrawn.
        let def = fixture.frame(b"def", 4, 1).unwrap();
        assert!((0..3).all(|c| def.ink(c, 0) > 0));
        assert_eq!(fixture.renderer.cache.len(), 3, "only def cached now");
        // And the old screen draws identically after another clear.
        let abc_again = fixture.frame(b"abc", 4, 1).unwrap();
        assert_eq!(abc.pixels, abc_again.pixels);
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn screen_that_cannot_fit_the_atlas_is_an_error() {
        let mut fixture = Fixture::new(Some(&["ab"]));
        let err = fixture.frame(b"abcdef", 7, 1).err().unwrap();
        assert!(
            format!("{err:#}").contains("don't fit in the atlas"),
            "{err:#}"
        );
    }
}
