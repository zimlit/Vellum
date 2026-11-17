//! Drawing the terminal's screen: glyph cache, atlas, and the instanced
//! cell pass.
//!
//! Each frame rasterizes glyphs not seen before into the atlas (compute,
//! before the render pass), clears the target to the default background and
//! draws everything in one instanced call. Blending within a draw happens in
//! primitive order, so instances are layered by their order in the buffer:
//!
//! 1. cell backgrounds that differ from the default (solid quads, one per
//!    run of equal color in a row). Runs at the edge of the grid extend
//!    over the margin to the screen's edge, so a program's background
//!    fills the screen instead of leaving a frame of the default one;
//! 2. glyphs, each in its cell's foreground;
//! 3. decorations: underline, strikeout, overline (solid quads);
//! 4. the cursor (an inverted block redraws the glyph in the cell's
//!    background color on top of it).
//!
//! Rows come from the terminal's view ([`Terminal::view_line`]), which is
//! the scrollback while scrolled back.
//!
//! See `shaders/cell.vert` and `shaders/cell.frag`. The whole screen is
//! redrawn every frame; damage tracking comes later (roadmap step 10).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use skrifa::GlyphId;

use crate::font::{CellMetrics, Face, FontSet, Stroke};
use crate::raster::{AtlasFull, Rasterizer, Rect};
use crate::spirv;
use crate::term::palette::Rgb;
use crate::term::{Color, CursorShape, Flags, Palette, Style, Terminal, Underline};
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

impl Instance {
    /// A filled rectangle `size` pixels big, `offset` from the top-left of
    /// cell (`col`, `row`) (negative reaches into the margin).
    fn solid(col: u16, row: u16, offset: (i32, i32), size: (u32, u32), color: Rgb) -> Self {
        let clamp = |v: u32| v.min(u32::from(u16::MAX)) as u16;
        let signed = |v: i32| v.clamp(i16::MIN.into(), i16::MAX.into());
        Self {
            cell: pack(col, row),
            offset: pack_signed(signed(offset.0), signed(offset.1)),
            atlas: 0,
            size: pack(clamp(size.0), clamp(size.1)),
            color: pack_color(color),
            flags: FLAG_SOLID,
        }
    }

    fn glyph(col: u16, row: u16, placed: Placed, color: Rgb) -> Self {
        let r = placed.rect;
        Self {
            cell: pack(col, row),
            offset: pack_signed(placed.offset.0, placed.offset.1),
            atlas: pack(r.x as u16, r.y as u16),
            size: pack(r.width as u16, r.height as u16),
            color: pack_color(color),
            flags: 0,
        }
    }
}

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

/// Opaque RGB as the shader's `unpackUnorm4x8` expects it.
fn pack_color([r, g, b]: Rgb) -> u32 {
    u32::from_le_bytes([r, g, b, 0xff])
}

/// The RGB of `color`, with `default` for [`Color::Default`].
fn rgb(palette: &Palette, color: Color, default: Rgb) -> Rgb {
    match color {
        Color::Default => default,
        Color::Indexed(i) => palette.colors[usize::from(i)],
        Color::Rgb(c) => c,
    }
}

/// The colors a cell is drawn in: `(foreground, background)`, after
/// inverse and dim.
fn resolve(palette: &Palette, style: &Style) -> (Rgb, Rgb) {
    let mut fg = rgb(palette, style.fg, palette.foreground);
    let mut bg = rgb(palette, style.bg, palette.background);
    if style.flags.contains(Flags::INVERSE) {
        (fg, bg) = (bg, fg);
    }
    if style.flags.contains(Flags::DIM) {
        // Halfway to the background, so it works on any background.
        fg = std::array::from_fn(|i| ((u16::from(fg[i]) + u16::from(bg[i])) / 2) as u8);
    }
    (fg, bg)
}

/// A message from vellum itself (a config error, say), shown in a bar over
/// the bottom row until the front end drops it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notice {
    pub level: Level,
    /// Only the first line is shown (the log has the rest).
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warning,
    Error,
}

impl Level {
    /// Text and bar colors: fixed, not the palette, so a program's colors
    /// can't hide the notice.
    fn colors(self) -> (Rgb, Rgb) {
        match self {
            Level::Info => ([0xee, 0xee, 0xec], [0x20, 0x4a, 0x87]),
            Level::Warning => ([0x2e, 0x34, 0x36], [0xed, 0xd4, 0x00]),
            Level::Error => ([0xee, 0xee, 0xec], [0xa4, 0x00, 0x00]),
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
    fonts: FontSet,
    raster: Rasterizer,
    /// `None`: the character draws nothing (e.g. space).
    cache: HashMap<(Face, char), Option<Placed>>,
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    /// Scratch for building a frame's instances, in drawing order.
    instances: Vec<Instance>,
    /// Scratch for decorations, appended after the glyphs.
    decorations: Vec<Instance>,
    /// One instance buffer per frame slot, so a frame in flight keeps its
    /// own while the next is written.
    buffers: Vec<Option<Arc<Buffer>>>,
}

impl TextRenderer {
    /// `slots`: how many frames can be in flight (one per scanout buffer).
    pub fn new(
        device: &Arc<Device>,
        fonts: FontSet,
        format: vk::Format,
        slots: usize,
    ) -> Result<Self> {
        Self::with_atlas(device, fonts, format, slots, ATLAS_SIZE, ATLAS_SIZE)
    }

    fn with_atlas(
        device: &Arc<Device>,
        fonts: FontSet,
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
            fonts,
            raster,
            cache: HashMap::new(),
            pipeline,
            set,
            instances: Vec::new(),
            decorations: Vec::new(),
            buffers: vec![None; slots],
        })
    }

    pub fn cell(&self) -> CellMetrics {
        self.fonts.cell()
    }

    /// Record a frame of `term`'s screen into `target`, using instance
    /// buffer `slot`, with `notice` over the bottom row.
    pub fn draw<T: RenderTarget>(
        &mut self,
        rec: &mut Recorder,
        target: &Arc<T>,
        slot: usize,
        term: &Terminal,
        notice: Option<&Notice>,
    ) -> Result<()> {
        let (width, height) = target.size();
        let margins = Margins::new(self.cell(), term, width, height);
        if let Err(e) = self.build(term, margins, notice) {
            if e.downcast_ref::<AtlasFull>().is_none() {
                return Err(e);
            }
            // Start over with an empty atlas, holding only what's on screen.
            tracing::debug!(glyphs = self.cache.len(), "glyph atlas full; clearing it");
            self.raster.clear();
            self.cache.clear();
            self.build(term, margins, notice)
                .context("the glyphs on screen don't fit in the atlas")?;
        }
        self.raster.record(rec)?;

        let count = self.instances.len() as u32;
        let buffer = self.instance_buffer(slot)?;
        buffer.write(0, &self.instances)?;
        rec.using(&[&buffer]);

        let cell = self.cell();
        let push = Push {
            instances: buffer.address(),
            screen: [width, height],
            cell: [cell.width, cell.height],
            origin: [margins.left, margins.top],
        };
        let bg = term.palette().background.map(|c| f32::from(c) / 255.0);
        rec.begin_rendering(target, Some([bg[0], bg[1], bg[2], 1.0]))?;
        if count > 0 {
            rec.draw(&self.pipeline, Some(&self.set), &push, 6, count)?;
        }
        rec.end_rendering()
    }

    /// Fill `self.instances` for `term`, queueing new glyphs. Fails with
    /// [`AtlasFull`] (inside the anyhow error) when the atlas runs out.
    fn build(&mut self, term: &Terminal, margins: Margins, notice: Option<&Notice>) -> Result<()> {
        self.instances.clear();
        self.decorations.clear();
        backgrounds(&mut self.instances, term, self.fonts.cell(), margins);
        let palette = term.palette();
        let cell = self.cell();
        for row in 0..term.rows() {
            for col in 0..term.cols() {
                let c = term.view_cell(col, row);
                if c.style.flags.contains(Flags::HIDDEN) {
                    continue;
                }
                let (fg, _) = resolve(palette, &c.style);
                if c.ch != ' '
                    && let Some(p) = self.glyph(face_of(&c.style), c.ch)?
                {
                    self.instances.push(Instance::glyph(col, row, p, fg));
                }
                decorations(&mut self.decorations, cell, palette, &c.style, fg, col, row);
            }
        }
        self.instances.append(&mut self.decorations);
        self.cursor(term)?;
        match notice {
            Some(n) => self.notice(term, margins, n),
            None => Ok(()),
        }
    }

    /// A bar across the bottom row (and the margins beside and below it)
    /// with the first line of the notice's text, cut to fit.
    fn notice(&mut self, term: &Terminal, margins: Margins, notice: &Notice) -> Result<()> {
        let cell = self.cell();
        let row = term.rows() - 1;
        let (fg, bg) = notice.level.colors();
        let width = margins.left + u32::from(term.cols()) * cell.width + margins.right;
        self.instances.push(Instance::solid(
            0,
            row,
            (-(margins.left as i32), 0),
            (width, cell.height + margins.bottom),
            bg,
        ));
        let text = notice.text.lines().next().unwrap_or_default();
        // A space before the text, so it doesn't touch the screen's edge.
        for (col, ch) in (1..term.cols()).zip(text.chars()) {
            if ch != ' '
                && !ch.is_control()
                && let Some(p) = self.glyph(Face::new(false, false), ch)?
            {
                self.instances.push(Instance::glyph(col, row, p, fg));
            }
        }
        Ok(())
    }

    fn cursor(&mut self, term: &Terminal) -> Result<()> {
        let cursor = term.cursor();
        // Scrolled back, the cursor's row isn't where it's drawn.
        if !cursor.visible || term.view_offset() > 0 {
            return Ok(());
        }
        let palette = term.palette();
        let cell = self.cell();
        let (col, row) = (cursor.col, cursor.row);
        let color = palette.cursor;
        // Thin cursors: at least 2 px so they're easy to see.
        let thin = cell.underline.thickness.max(2);
        match cursor.shape {
            CursorShape::Block => {
                self.instances.push(Instance::solid(
                    col,
                    row,
                    (0, 0),
                    (cell.width, cell.height),
                    color,
                ));
                // The character under it, in the cell's background color.
                let c = term.cell(col, row);
                let (_, bg) = resolve(palette, &c.style);
                if c.ch != ' '
                    && !c.style.flags.contains(Flags::HIDDEN)
                    && let Some(p) = self.glyph(face_of(&c.style), c.ch)?
                {
                    self.instances.push(Instance::glyph(col, row, p, bg));
                }
            }
            CursorShape::Underline => {
                let thin = thin.min(cell.height);
                self.instances.push(Instance::solid(
                    col,
                    row,
                    (0, (cell.height - thin) as i32),
                    (cell.width, thin),
                    color,
                ));
            }
            CursorShape::Bar => {
                self.instances.push(Instance::solid(
                    col,
                    row,
                    (0, 0),
                    (thin.min(cell.width), cell.height),
                    color,
                ));
            }
        }
        Ok(())
    }

    /// The atlas placement for `ch`, rasterizing it if new.
    fn glyph(&mut self, face: Face, ch: char) -> Result<Option<Placed>> {
        if let Some(&placed) = self.cache.get(&(face, ch)) {
            return Ok(placed);
        }
        let font = self.fonts.face(face);
        // Missing characters show the font's .notdef glyph (usually a box).
        let id = font.glyph_id(ch).unwrap_or(GlyphId::NOTDEF);
        let glyph = font.glyph(id)?;
        let placed = self.raster.queue(&glyph.outline)?.map(|rect| Placed {
            rect,
            offset: (
                glyph.outline.left,
                self.fonts.cell().baseline as i32 + glyph.outline.top,
            ),
        });
        self.cache.insert((face, ch), placed);
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

/// The face SGR bold and italic select.
fn face_of(style: &Style) -> Face {
    Face::new(
        style.flags.contains(Flags::BOLD),
        style.flags.contains(Flags::ITALIC),
    )
}

/// Solid quads for backgrounds that aren't the default one (that's the
/// clear color), merged into runs along each row. Runs in the first or last
/// column or row also cover the margin beside them, corners included.
fn backgrounds(out: &mut Vec<Instance>, term: &Terminal, cell: CellMetrics, margins: Margins) {
    let palette = term.palette();
    let (cols, rows) = (term.cols(), term.rows());
    for row in 0..rows {
        let top = if row == 0 { margins.top } else { 0 };
        let bottom = if row == rows - 1 { margins.bottom } else { 0 };
        let mut run: Option<(u16, Rgb)> = None;
        for col in 0..=term.cols() {
            let bg = (col < term.cols())
                .then(|| resolve(palette, &term.view_cell(col, row).style).1)
                .filter(|&bg| bg != palette.background);
            if let Some((start, color)) = run {
                if bg == Some(color) {
                    continue;
                }
                let left = if start == 0 { margins.left } else { 0 };
                let right = if col == cols { margins.right } else { 0 };
                let width = u32::from(col - start) * cell.width;
                out.push(Instance::solid(
                    start,
                    row,
                    (-(left as i32), -(top as i32)),
                    (left + width + right, top + cell.height + bottom),
                    color,
                ));
            }
            run = bg.map(|color| (col, color));
        }
    }
}

/// Underline, strikeout and overline quads for a cell in `style`.
/// Curly, dotted and dashed underlines are drawn straight for now.
fn decorations(
    out: &mut Vec<Instance>,
    cell: CellMetrics,
    palette: &Palette,
    style: &Style,
    fg: Rgb,
    col: u16,
    row: u16,
) {
    let mut line = |s: Stroke, color| {
        out.push(Instance::solid(
            col,
            row,
            (0, s.top as i32),
            (cell.width, s.thickness),
            color,
        ));
    };
    let underline_color = rgb(palette, style.underline_color, fg);
    match style.underline {
        Underline::None => {}
        Underline::Double => {
            let u = cell.underline;
            line(u, underline_color);
            line(double_underline(u, cell.height), underline_color);
        }
        _ => line(cell.underline, underline_color),
    }
    if style.flags.contains(Flags::STRIKE) {
        line(cell.strikeout, fg);
    }
    if style.flags.contains(Flags::OVERLINE) {
        line(
            Stroke {
                top: 0,
                thickness: cell.underline.thickness,
            },
            fg,
        );
    }
}

/// The second line of a double underline: one thickness of gap below the
/// first, or above it when there's no room below.
fn double_underline(u: Stroke, height: u32) -> Stroke {
    let below = u.top + 2 * u.thickness;
    let top = if below + u.thickness <= height {
        below
    } else {
        u.top.saturating_sub(2 * u.thickness)
    };
    Stroke { top, ..u }
}

/// How many columns and rows of `cell` fit in `width` x `height` pixels.
pub fn grid_size(cell: CellMetrics, width: u32, height: u32) -> (u16, u16) {
    let cols = (width / cell.width).clamp(1, u32::from(u16::MAX));
    let rows = (height / cell.height).clamp(1, u32::from(u16::MAX));
    (cols as u16, rows as u16)
}

/// The pixels around the grid: it's centred, splitting what's left over
/// (the right and bottom get the odd pixel).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Margins {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

impl Margins {
    fn new(cell: CellMetrics, term: &Terminal, width: u32, height: u32) -> Self {
        let spare_w = width.saturating_sub(cell.width * u32::from(term.cols()));
        let spare_h = height.saturating_sub(cell.height * u32::from(term.rows()));
        Self {
            left: spare_w / 2,
            top: spare_h / 2,
            right: spare_w - spare_w / 2,
            bottom: spare_h - spare_h / 2,
        }
    }
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
        underline: Stroke {
            top: 16,
            thickness: 1,
        },
        strikeout: Stroke {
            top: 10,
            thickness: 1,
        },
    };

    fn rgb_of(i: &Instance) -> Rgb {
        let [r, g, b, _] = i.color.to_le_bytes();
        [r, g, b]
    }

    fn unpack(v: u32) -> (u16, u16) {
        (v as u16, (v >> 16) as u16)
    }

    #[test]
    fn colors_resolve_with_inverse_and_dim() {
        let p = Palette::default();
        let style = |fg, bg, flags| Style {
            fg,
            bg,
            flags,
            ..Style::default()
        };
        assert_eq!(resolve(&p, &Style::default()), (p.foreground, p.background));
        let red = Color::Indexed(1);
        let blue = Color::Rgb([0, 0, 200]);
        assert_eq!(
            resolve(&p, &style(red, blue, Flags::empty())),
            (p.colors[1], [0, 0, 200])
        );
        assert_eq!(
            resolve(&p, &style(Color::Default, blue, Flags::INVERSE)),
            ([0, 0, 200], p.foreground)
        );
        let dim = resolve(
            &p,
            &style(
                Color::Rgb([200, 100, 0]),
                Color::Rgb([0, 0, 100]),
                Flags::DIM,
            ),
        );
        assert_eq!(dim.0, [100, 50, 50]);
    }

    #[test]
    fn edge_backgrounds_cover_the_margins() {
        let mut term = Terminal::new(4, 3);
        // Row 0: red across (both edges); row 1: blue in the middle only;
        // row 2: green at the left edge only.
        term.feed(b"\x1b[41m    \x1b[m\r\n \x1b[44m  \x1b[m \r\n\x1b[42m \x1b[m");
        let margins = Margins {
            left: 2,
            top: 3,
            right: 4,
            bottom: 5,
        };
        let mut out = Vec::new();
        backgrounds(&mut out, &term, CELL, margins);
        let offset = |i: &Instance| ((i.offset << 16) as i32 >> 16, i.offset as i32 >> 16);
        let quads: Vec<_> = out
            .iter()
            .map(|i| (unpack(i.cell), offset(i), unpack(i.size)))
            .collect();
        assert_eq!(
            quads,
            [
                // Top row, full width: the top margin and both sides.
                ((0, 0), (-2, -3), (2 + 40 + 4, 3 + 20)),
                // Inside: unchanged.
                ((1, 1), (0, 0), (20, 20)),
                // Bottom-left: the left and bottom margins.
                ((0, 2), (-2, 0), (2 + 10, 20 + 5)),
            ]
        );
    }

    #[test]
    fn backgrounds_merge_into_runs() {
        let mut term = Terminal::new(8, 2);
        // Red for 3 cells, a default cell, blue for 2, then red again;
        // second row: inverse makes the default foreground a background.
        term.feed(b"\x1b[41m   \x1b[m \x1b[44m  \x1b[41m \r\n\x1b[7mab");
        let mut out = Vec::new();
        backgrounds(&mut out, &term, CELL, Margins::default());
        let p = term.palette();
        let runs: Vec<_> = out
            .iter()
            .map(|i| (unpack(i.cell), unpack(i.size).0, rgb_of(i)))
            .collect();
        assert_eq!(
            runs,
            [
                ((0, 0), 30, p.colors[1]),
                ((4, 0), 20, p.colors[4]),
                ((6, 0), 10, p.colors[1]),
                ((0, 1), 20, p.foreground),
            ]
        );
        assert!(
            out.iter()
                .all(|i| i.flags == FLAG_SOLID && unpack(i.size).1 == 20)
        );
    }

    #[test]
    fn decoration_quads() {
        let p = Palette::default();
        let fg = [1, 2, 3];
        let mut out = Vec::new();
        let style = Style {
            underline: Underline::Double,
            underline_color: Color::Indexed(2),
            flags: Flags::STRIKE | Flags::OVERLINE,
            ..Style::default()
        };
        decorations(&mut out, CELL, &p, &style, fg, 4, 1);
        let quads: Vec<_> = out
            .iter()
            .map(|i| (unpack(i.offset).1, unpack(i.size), rgb_of(i)))
            .collect();
        assert_eq!(
            quads,
            [
                (16, (10, 1), p.colors[2]),
                (18, (10, 1), p.colors[2]),
                (10, (10, 1), fg),
                (0, (10, 1), fg),
            ]
        );
        assert!(out.iter().all(|i| unpack(i.cell) == (4, 1)));
        // Default underline color: the text's.
        out.clear();
        let curly = Style {
            underline: Underline::Curly,
            ..Style::default()
        };
        decorations(&mut out, CELL, &p, &curly, fg, 0, 0);
        assert_eq!(out.len(), 1);
        assert_eq!(rgb_of(&out[0]), fg);
    }

    #[test]
    fn double_underline_goes_above_when_there_is_no_room() {
        let u = Stroke {
            top: 10,
            thickness: 2,
        };
        assert_eq!(double_underline(u, 20).top, 14);
        assert_eq!(double_underline(u, 15).top, 6);
    }

    #[test]
    fn packing_matches_the_shader() {
        assert_eq!(pack(3, 7), 3 | 7 << 16);
        assert_eq!(pack_signed(-1, 2), 0xffff | 2 << 16);
        // The shader sign-extends: (int(v << 16) >> 16, int(v) >> 16).
        let v = pack_signed(-5, -300);
        assert_eq!(((v << 16) as i32 >> 16, v as i32 >> 16), (-5, -300));
        assert_eq!(pack_color([1, 2, 3]), 0xff03_0201);
        assert_eq!(size_of::<Instance>(), 24);
        assert_eq!(size_of::<Push>(), 32);
    }

    #[test]
    fn grid_fits_and_centres() {
        assert_eq!(grid_size(CELL, 105, 41), (10, 2));
        assert_eq!(grid_size(CELL, 3, 3), (1, 1));
        let term = Terminal::new(10, 2);
        assert_eq!(
            Margins::new(CELL, &term, 105, 41),
            Margins {
                left: 2,
                top: 0,
                right: 3,
                bottom: 1
            }
        );
        assert_eq!(Margins::new(CELL, &term, 100, 40), Margins::default());
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
            let config = fontconfig::Config::new().unwrap();
            let fonts = FontSet::load(&config, "monospace:pixelsize=16", 96.0).unwrap();
            let cell = fonts.cell();
            let font = fonts.regular();
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
                TextRenderer::with_atlas(&device, fonts, Format::R8G8B8A8_UNORM, 1, aw, ah)
                    .unwrap();
            Self {
                device,
                renderer,
                cell,
            }
        }

        fn frame(&mut self, text: &[u8], cols: u16, rows: u16) -> Result<Frame> {
            let (width, height) = (
                self.cell.width * u32::from(cols),
                self.cell.height * u32::from(rows),
            );
            self.frame_in(text, cols, rows, width, height)
        }

        /// A frame `width` x `height` pixels, with the grid centred in it.
        fn frame_in(
            &mut self,
            text: &[u8],
            cols: u16,
            rows: u16,
            width: u32,
            height: u32,
        ) -> Result<Frame> {
            let mut term = Terminal::new(cols, rows);
            // White on black, red cursor.
            term.feed(b"\x1b]10;#ffffff;#000000;#ff0000\x07");
            term.feed(text);
            self.render(&term, width, height)
        }

        /// Draw `term` into a `width` x `height` frame and read it back.
        fn render(&mut self, term: &Terminal, width: u32, height: u32) -> Result<Frame> {
            self.render_with(term, width, height, None)
        }

        fn render_with(
            &mut self,
            term: &Terminal,
            width: u32,
            height: u32,
            notice: Option<&Notice>,
        ) -> Result<Frame> {
            let device = &self.device;
            let target = Image::new(
                device,
                width,
                height,
                Format::R8G8B8A8_UNORM,
                ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
            )?;
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
            self.renderer.draw(&mut rec, &target, 0, term, notice)?;
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

    const WHITE: [u8; 4] = [255, 255, 255, 255];
    const BLACK: [u8; 4] = [0, 0, 0, 255];
    const RED: [u8; 4] = [255, 0, 0, 255];

    impl Frame {
        /// The pixel at (`x`, `y`) within cell (`col`, `row`).
        fn at(&self, col: u32, row: u32, x: u32, y: u32) -> [u8; 4] {
            let (x, y) = (col * self.cell.width + x, row * self.cell.height + y);
            self.pixels[(y * self.width + x) as usize]
        }

        /// One cell's pixels as `(x, y, pixel)`, `x` and `y` within the cell.
        fn cell_pixels(&self, col: u32, row: u32) -> Vec<(u32, u32, [u8; 4])> {
            let (w, h) = (self.cell.width, self.cell.height);
            (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| (x, y, self.at(col, row, x, y)))
                .collect()
        }

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

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn a_full_screen_background_reaches_the_edges() {
        // What Emacs does: every cell black-on-black explicitly, on a
        // terminal whose default background is gray. The 2x2 grid sits in
        // the middle of a target 7 px wider and 5 px taller.
        let mut fixture = Fixture::new(None);
        let cell = fixture.cell;
        let (w, h) = (cell.width * 2 + 7, cell.height * 2 + 5);
        let text = b"\x1b]11;#808080\x07\x1b[?25l\x1b[44m    ";
        let f = fixture.frame_in(text, 2, 2, w, h).unwrap();
        let blue = [0x34, 0x65, 0xa4, 255];
        let px = |x: u32, y: u32| f.pixels[(y * w + x) as usize];
        for (x, y) in [
            (0, 0),
            (w - 1, 0),
            (0, h - 1),
            (w - 1, h - 1),
            (w / 2, 0),
            (0, h / 2),
        ] {
            assert_eq!(px(x, y), blue, "({x}, {y})");
        }
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn scrolled_back_shows_history_without_a_cursor() {
        let mut fixture = Fixture::new(None);
        let cell = fixture.cell;
        let mut term = Terminal::new(3, 2);
        term.feed(b"\x1b]10;#ffffff;#000000;#ff0000\x07");
        // "W" scrolls off; the screen has two empty rows, cursor at the top.
        term.feed(b"W\r\n\r\n\x1b[H");
        let (w, h) = (cell.width * 3, cell.height * 2);
        let live = fixture.render(&term, w, h).unwrap();
        assert!(live.is_cursor(0, 0));
        assert_eq!(live.ink(0, 1), 0);

        term.scroll_view(1);
        let back = fixture.render(&term, w, h).unwrap();
        assert!(back.ink(0, 0) > 0, "W, from the scrollback");
        assert!(!back.pixels.contains(&RED), "no cursor");
    }

    /// A sample screen, mostly for VELLUM_DUMP_FRAME.
    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn draws_a_sample_screen() {
        let text = "$ ls -la ~/projects/vellum\r\nThe quick brown fox jumps over the lazy dog.\r\n\
                    0123456789 {}[]()<> @#$%^&*_+=|\\/?!~`'\":;,.\r\nÄÖÜ é ß ñ → ✓ 漢\r\n\
                    \x1b[31mred \x1b[32mgreen \x1b[33myellow \x1b[34mblue \x1b[35mmagenta \x1b[36mcyan\x1b[m\r\n\
                    \x1b[41m bg \x1b[42m bg \x1b[44m bg \x1b[48;2;80;40;120m truecolor \x1b[m \x1b[7m inverse \x1b[m \x1b[2mdim\x1b[m\r\n\
                    \x1b[4munder\x1b[m \x1b[21mdouble\x1b[m \x1b[4:3;58;5;9mcurly\x1b[m \x1b[9mstrike\x1b[m \x1b[53mover\x1b[m\r\n\
                    regular \x1b[1mbold\x1b[m \x1b[3mitalic\x1b[m \x1b[1;3mbold italic\x1b[m \x1b[1;31mbold red\x1b[m\r\n\
                    \x1b(0lqqqwqqqk\r\nx   x   x\r\nmqqqvqqqj\x1b(B $ ";
        let f = Fixture::new(None).frame(text.as_bytes(), 50, 12).unwrap();
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

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn draws_cell_colors() {
        let f = Fixture::new(None)
            .frame(
                b"\x1b[41m \x1b[m\x1b[7mA\x1b[m\x1b[38;2;0;255;0mH\x1b[m\x1b[2mH\x1b[?25l",
                4,
                1,
            )
            .unwrap();
        let [r, g, b] = Palette::default().colors[1];
        let red_bg = [r, g, b, 255];
        assert!(f.cell_pixels(0, 0).iter().all(|&(_, _, px)| px == red_bg));
        // Inverse: white background, dark glyph.
        assert_eq!(f.at(1, 0, 0, 0), WHITE);
        assert!(f.cell_pixels(1, 0).iter().any(|&(_, _, px)| px[0] < 128));
        // Truecolor: pure green where coverage is full, never other hues.
        let green = f.cell_pixels(2, 0);
        assert!(green.iter().any(|&(_, _, px)| px == [0, 255, 0, 255]));
        assert!(green.iter().all(|&(_, _, px)| px[0] == 0 && px[2] == 0));
        // Dim: at most halfway from black to white.
        let brightest = f.cell_pixels(3, 0).iter().map(|p| p.2[1]).max().unwrap();
        assert!((120..=135).contains(&brightest), "{brightest}");
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn draws_decorations_where_the_font_says() {
        let mut fixture = Fixture::new(None);
        let cell = fixture.cell;
        let f = fixture
            .frame(b"\x1b[4m \x1b[24;9m \x1b[29;53m \x1b[?25l", 3, 1)
            .unwrap();
        let overline = Stroke {
            top: 0,
            thickness: cell.underline.thickness,
        };
        for (col, stroke) in [(0, cell.underline), (1, cell.strikeout), (2, overline)] {
            for (_, y, px) in f.cell_pixels(col, 0) {
                let inside = (stroke.top..stroke.top + stroke.thickness).contains(&y);
                assert_eq!(px, if inside { WHITE } else { BLACK }, "col {col} y {y}");
            }
        }
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn cursor_shapes() {
        let mut fixture = Fixture::new(None);
        let cell = fixture.cell;
        let thin = cell.underline.thickness.max(2);

        let bar = fixture.frame(b"\x1b[6 q", 2, 1).unwrap();
        for (x, _, px) in bar.cell_pixels(0, 0) {
            assert_eq!(px, if x < thin { RED } else { BLACK });
        }
        let underline = fixture.frame(b"\x1b[4 q", 2, 1).unwrap();
        for (_, y, px) in underline.cell_pixels(0, 0) {
            assert_eq!(px, if y >= cell.height - thin { RED } else { BLACK });
        }
        let hidden = fixture.frame(b"\x1b[?25l", 2, 1).unwrap();
        assert!(hidden.pixels.iter().all(|&px| px == BLACK));

        // A block over a character shows it in the background color.
        let block = fixture.frame(b"A\x1b[D", 2, 1).unwrap();
        let pixels = block.cell_pixels(0, 0);
        assert_eq!(block.at(0, 0, 0, 0), RED);
        assert!(pixels.iter().all(|&(_, _, px)| px[1] == 0 && px[2] == 0));
        assert!(pixels.iter().any(|&(_, _, px)| px[0] < 64), "glyph visible");
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn bold_and_italic_use_their_own_faces() {
        let mut fixture = Fixture::new(None);
        let f = fixture
            .frame(b"H\x1b[1mH\x1b[22;3mH\x1b[1mH\x1b[?25l", 5, 1)
            .unwrap();
        let (regular, bold) = (f.ink(0, 0), f.ink(1, 0));
        assert!(bold > regular * 11 / 10, "bold {bold} vs regular {regular}");
        // Italic H: a different shape, about the same amount of ink.
        let italic = f.cell_pixels(2, 0);
        assert_ne!(italic, f.cell_pixels(0, 0));
        assert!(f.ink(3, 0) > f.ink(2, 0), "bold italic heavier than italic");
        assert_eq!(fixture.renderer.cache.len(), 4, "one entry per face");
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn notice_bar_covers_the_bottom_row_and_margins() {
        let mut fixture = Fixture::new(None);
        let cell = fixture.cell;
        let mut term = Terminal::new(4, 2);
        term.feed(b"\x1b]10;#ffffff;#000000;#ff0000\x07HHHH\r\nHHHH\x1b[?25l");
        let notice = Notice {
            level: Level::Error,
            text: "E\nsecond line not shown".into(),
        };
        // 3 px of margin on every side.
        let (w, h) = (4 * cell.width + 6, 2 * cell.height + 6);
        let f = fixture.render_with(&term, w, h, Some(&notice)).unwrap();
        let px = |x: u32, y: u32| f.pixels[(y * w + x) as usize];
        let (fg, bg) = Level::Error.colors();
        let bar = [bg[0], bg[1], bg[2], 255];
        let bottom = 3 + cell.height;
        // The bar: the bottom row's first cell (left of the text), and the
        // margins beside and below it, corners included.
        for (x, y) in [(0, h - 1), (w - 1, h - 1), (0, bottom), (3, bottom + 1)] {
            assert_eq!(px(x, y), bar, "({x}, {y})");
        }
        // The top row is untouched: its H is still white ink.
        assert!(f.pixels[..(w * bottom) as usize].contains(&WHITE));
        // The text's E is drawn in the notice color, from the second column.
        let e = (0..cell.height).flat_map(|y| (0..cell.width).map(move |x| (x, y)));
        let text = [fg[0], fg[1], fg[2], 255];
        let inked = e
            .map(|(x, y)| px(3 + cell.width + x, bottom + y))
            .filter(|&p| p == text)
            .count();
        assert!(inked > 0, "E drawn");
    }
}
