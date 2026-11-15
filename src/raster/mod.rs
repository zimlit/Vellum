//! The GPU glyph rasterizer: flattened outlines in, coverage in an atlas out.
//!
//! Glyphs are batched: [`Rasterizer::queue`] reserves atlas space and
//! appends a glyph's segments, [`Rasterizer::record`] uploads the batch and
//! records two compute passes (see `shaders/deposit.comp` and
//! `shaders/resolve.comp`):
//!
//! 1. fill the accumulation grid with zeros;
//! 2. **deposit**: one invocation per segment adds signed area (fixed point,
//!    `atomicAdd`) into its glyph's part of the grid;
//! 3. **resolve**: one workgroup per glyph turns row-wise running sums into
//!    coverage and writes it to the atlas.
//!
//! The atlas stays in `GENERAL` layout: the resolve pass writes it as a
//! storage image and text drawing reads it the same way.

pub mod atlas;

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::font::Outline;
use crate::spirv;
use crate::vk::{
    self, Access, Buffer, BufferUsageFlags, ComputePipeline, DescriptorSet, Device, Image,
    ImageLayout, ImageUsageFlags, Layout, Location, Pod, Recorder,
};
pub use atlas::Rect;

/// Atlas texel format. RGBA leaves room for subpixel (per-channel) coverage
/// and color glyphs later; for now all four channels hold the coverage.
pub const ATLAS_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

const WORKGROUP: u32 = 64;
/// Most workgroups one dispatch dimension is guaranteed to allow.
const MAX_GROUPS: u32 = 65535;

/// `struct Segment` in `shaders/raster.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Segment {
    p0: [f32; 2],
    p1: [f32; 2],
    job: u32,
    _pad: u32,
}
unsafe impl Pod for Segment {}

/// `struct Job` in `shaders/raster.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Job {
    atlas_origin: [u32; 2],
    size: [u32; 2],
    accum_offset: u32,
    _pad: u32,
}
unsafe impl Pod for Job {}

/// The push constant block in `shaders/raster.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Push {
    segments: u64,
    jobs: u64,
    accum: u64,
    segment_count: u32,
    job_base: u32,
}
unsafe impl Pod for Push {}

/// The atlas has no room for a glyph. Clear it ([`Rasterizer::clear`]) and
/// queue the glyphs that are still needed.
#[derive(Debug, PartialEq, Eq)]
pub struct AtlasFull;

impl std::fmt::Display for AtlasFull {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("glyph atlas is full")
    }
}

impl std::error::Error for AtlasFull {}

pub struct Rasterizer {
    device: Arc<Device>,
    deposit: Arc<ComputePipeline>,
    resolve: Arc<ComputePipeline>,
    set: Arc<DescriptorSet>,
    atlas: Arc<Image>,
    packer: atlas::Packer,
    /// The atlas has been moved out of `UNDEFINED`.
    atlas_ready: bool,
    // The pending batch.
    segments: Vec<Segment>,
    jobs: Vec<Job>,
    accum_cells: u32,
}

impl Rasterizer {
    pub fn new(device: &Arc<Device>, atlas_width: u32, atlas_height: u32) -> Result<Self> {
        let atlas = Image::new(
            device,
            atlas_width,
            atlas_height,
            ATLAS_FORMAT,
            // TRANSFER_SRC: for reading the atlas back in tests and debugging.
            ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC,
        )?;
        let layout = Layout::new(device, 1, size_of::<Push>() as u32)?;
        Ok(Self {
            device: device.clone(),
            deposit: ComputePipeline::new(&layout, spirv!("deposit.comp"))?,
            resolve: ComputePipeline::new(&layout, spirv!("resolve.comp"))?,
            set: DescriptorSet::new(&layout, &[&atlas])?,
            atlas,
            packer: atlas::Packer::new(atlas_width, atlas_height),
            atlas_ready: false,
            segments: Vec::new(),
            jobs: Vec::new(),
            accum_cells: 0,
        })
    }

    /// The atlas: storage image in `GENERAL` layout once [`Self::record`]
    /// has run, for set 0 binding 0 of the text pipeline.
    pub fn atlas(&self) -> &Arc<Image> {
        &self.atlas
    }

    /// Add a glyph to the next batch. Returns where it will be in the
    /// atlas, or `None` for an empty glyph (e.g. a space).
    pub fn queue(&mut self, outline: &Outline) -> Result<Option<Rect>, AtlasFull> {
        if outline.lines.is_empty() || outline.width == 0 || outline.height == 0 {
            return Ok(None);
        }
        let rect = self
            .packer
            .alloc(outline.width, outline.height)
            .ok_or(AtlasFull)?;
        let job = self.jobs.len() as u32;
        self.jobs.push(Job {
            atlas_origin: [rect.x, rect.y],
            size: [rect.width, rect.height],
            accum_offset: self.accum_cells,
            _pad: 0,
        });
        self.accum_cells += (rect.width + 2) * rect.height;
        self.segments.extend(outline.lines.iter().map(|l| Segment {
            p0: l.p0,
            p1: l.p1,
            job,
            _pad: 0,
        }));
        Ok(Some(rect))
    }

    /// Forget all atlas contents. Rects from earlier `queue` calls are no
    /// longer valid. Must not be called with a batch pending.
    pub fn clear(&mut self) -> Result<()> {
        if !self.jobs.is_empty() {
            bail!("clearing the atlas with glyphs still queued");
        }
        self.packer.clear();
        Ok(())
    }

    /// Whether glyphs are waiting for [`Self::record`].
    pub fn pending(&self) -> bool {
        !self.jobs.is_empty()
    }

    /// Record the pending batch (if any). Afterwards the atlas is in
    /// `GENERAL` layout and the new glyphs are visible to fragment shaders.
    pub fn record(&mut self, rec: &mut Recorder) -> Result<()> {
        if !self.atlas_ready {
            rec.transition(
                &self.atlas,
                ImageLayout::UNDEFINED,
                ImageLayout::GENERAL,
                Access::NONE,
                Access::COMPUTE_WRITE | Access::FRAGMENT_READ,
            );
            self.atlas_ready = true;
        }
        if self.jobs.is_empty() {
            return Ok(());
        }

        let segments = upload(&self.device, &self.segments)?;
        let jobs = upload(&self.device, &self.jobs)?;
        let accum = Buffer::new(
            &self.device,
            u64::from(self.accum_cells) * 4,
            BufferUsageFlags::TRANSFER_DST,
            Location::Device,
        )?;
        rec.using(&[&segments, &jobs, &accum]);
        let mut push = Push {
            segments: segments.address(),
            jobs: jobs.address(),
            accum: accum.address(),
            segment_count: self.segments.len() as u32,
            job_base: 0,
        };

        rec.fill(&accum, 0);
        rec.barrier(
            Access::TRANSFER_WRITE,
            Access::COMPUTE_READ | Access::COMPUTE_WRITE,
        );
        // 65535 groups of 64 is ~4M segments: far beyond any batch of glyphs.
        let groups = push.segment_count.div_ceil(WORKGROUP);
        if groups > MAX_GROUPS {
            bail!("{} segments in one batch is too many", push.segment_count);
        }
        rec.dispatch(&self.deposit, Some(&self.set), &push, [groups, 1, 1])?;
        rec.barrier(Access::COMPUTE_WRITE, Access::COMPUTE_READ);
        // New glyphs land in unused atlas space, but order the writes after
        // earlier frames' reads anyway: conservative first.
        rec.barrier(Access::FRAGMENT_READ, Access::COMPUTE_WRITE);
        let job_count = self.jobs.len() as u32;
        while push.job_base < job_count {
            let groups = (job_count - push.job_base).min(MAX_GROUPS);
            rec.dispatch(&self.resolve, Some(&self.set), &push, [groups, 1, 1])?;
            push.job_base += groups;
        }
        rec.barrier(
            Access::COMPUTE_WRITE,
            Access::FRAGMENT_READ | Access::COMPUTE_WRITE,
        );

        self.segments.clear();
        self.jobs.clear();
        self.accum_cells = 0;
        Ok(())
    }
}

fn upload<T: Pod>(device: &Arc<Device>, data: &[T]) -> Result<Arc<Buffer>> {
    let buffer = Buffer::new(
        device,
        size_of_val(data) as u64,
        BufferUsageFlags::empty(),
        Location::Upload,
    )?;
    buffer.write(0, data)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::{self, Line};
    use crate::testutil;
    use crate::vk::CommandBuffer;

    /// font-rs's CPU accumulation rasterizer, in f32: the reference the
    /// shaders are checked against.
    fn reference(outline: &Outline) -> Vec<f32> {
        let (w, h) = (outline.width as usize, outline.height as usize);
        let stride = w + 2;
        let mut a = vec![0.0f32; stride * h];
        for l in &outline.lines {
            let (mut p0, mut p1) = (l.p0, l.p1);
            if (p0[1] - p1[1]).abs() <= 1e-7 {
                continue;
            }
            let dir = if p0[1] < p1[1] {
                1.0
            } else {
                std::mem::swap(&mut p0, &mut p1);
                -1.0
            };
            let dxdy = (p1[0] - p0[0]) / (p1[1] - p0[1]);
            let mut x = p0[0];
            let y0 = p0[1].max(0.0) as usize;
            for y in y0..h.min(p1[1].ceil() as usize) {
                let row = y * stride;
                let dy = ((y + 1) as f32).min(p1[1]) - (y as f32).max(p0[1]);
                let xnext = x + dxdy * dy;
                let d = dy * dir;
                let (x0, x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
                let x0floor = x0.floor();
                let x0i = x0floor as usize;
                let x1ceil = x1.ceil();
                let x1i = x1ceil as usize;
                if x1i <= x0i + 1 {
                    let xmf = 0.5 * (x + xnext) - x0floor;
                    a[row + x0i] += d - d * xmf;
                    a[row + x0i + 1] += d * xmf;
                } else {
                    let s = (x1 - x0).recip();
                    let x0f = x0 - x0floor;
                    let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
                    let x1f = x1 - x1ceil + 1.0;
                    let am = 0.5 * s * x1f * x1f;
                    a[row + x0i] += d * a0;
                    if x1i == x0i + 2 {
                        a[row + x0i + 1] += d * (1.0 - a0 - am);
                    } else {
                        let a1 = s * (1.5 - x0f);
                        a[row + x0i + 1] += d * (a1 - a0);
                        for xi in x0i + 2..x1i - 1 {
                            a[row + xi] += d * s;
                        }
                        let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                        a[row + x1i - 1] += d * (1.0 - a2 - am);
                    }
                    a[row + x1i] += d * am;
                }
                x = xnext;
            }
        }
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            let mut sum = 0.0;
            for x in 0..w {
                sum += a[y * stride + x];
                out.push(sum.abs().min(1.0));
            }
        }
        out
    }

    /// A closed polygon as an outline of the given bitmap size.
    fn polygon(width: u32, height: u32, points: &[[f32; 2]]) -> Outline {
        let lines = (0..points.len())
            .map(|i| Line {
                p0: points[i],
                p1: points[(i + 1) % points.len()],
            })
            .collect();
        Outline {
            lines,
            width,
            height,
            left: 0,
            top: 0,
        }
    }

    #[test]
    fn reference_gives_exact_area_for_a_half_pixel_offset_square() {
        // Square from (1.5, 1.5) to (3.5, 3.5) in a 5x5 bitmap.
        let o = polygon(5, 5, &[[1.5, 1.5], [3.5, 1.5], [3.5, 3.5], [1.5, 3.5]]);
        let c = reference(&o);
        let at = |x: usize, y: usize| c[y * 5 + x];
        assert_eq!(at(1, 1), 0.25); // corner: a quarter pixel
        assert_eq!(at(2, 1), 0.5); // edge: half a pixel
        assert_eq!(at(2, 2), 1.0); // inside
        assert_eq!(at(3, 3), 0.25);
        let total: f32 = c.iter().sum();
        assert!((total - 4.0).abs() < 1e-5, "{total}");
        assert_eq!(at(0, 0), 0.0);
    }

    #[test]
    fn reference_handles_both_windings_and_diagonals() {
        let square = [[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]];
        let mut reversed = square;
        reversed.reverse();
        assert_eq!(
            reference(&polygon(6, 6, &square)),
            reference(&polygon(6, 6, &reversed))
        );
        // A right triangle with legs of 4: area 8, half of each diagonal pixel.
        let tri = polygon(6, 6, &[[1.0, 1.0], [5.0, 5.0], [1.0, 5.0]]);
        let c = reference(&tri);
        assert!((c.iter().sum::<f32>() - 8.0).abs() < 1e-5);
        assert!((c[6 + 1] - 0.5).abs() < 1e-6);
    }

    struct Gpu {
        device: Arc<Device>,
        raster: Rasterizer,
    }

    fn gpu() -> Gpu {
        let card = testutil::card();
        let rdev = rustix::fs::fstat(&card).unwrap().st_rdev;
        let instance = vk::Instance::new(cfg!(debug_assertions)).unwrap();
        let device =
            Device::for_drm_node(&instance, rustix::fs::major(rdev), rustix::fs::minor(rdev))
                .unwrap();
        let raster = Rasterizer::new(&device, 256, 256).unwrap();
        Gpu { device, raster }
    }

    impl Gpu {
        /// Rasterize `outlines` in one batch and read back each one's
        /// coverage (red channel, 0..1).
        fn rasterize(&mut self, outlines: &[Outline]) -> Vec<Vec<f32>> {
            let rects: Vec<_> = outlines
                .iter()
                .map(|o| self.raster.queue(o).unwrap().unwrap())
                .collect();
            let atlas = self.raster.atlas().clone();
            let (w, h) = (atlas.width(), atlas.height());
            let out = Buffer::new(
                &self.device,
                u64::from(w * h) * 4,
                BufferUsageFlags::TRANSFER_DST,
                Location::Readback,
            )
            .unwrap();
            let mut cmd = CommandBuffer::new(&self.device).unwrap();
            let mut rec = cmd.begin().unwrap();
            self.raster.record(&mut rec).unwrap();
            rec.transition(
                &atlas,
                ImageLayout::GENERAL,
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                Access::COMPUTE_WRITE,
                Access::TRANSFER_READ,
            );
            rec.copy_image_to_buffer(&atlas, &out).unwrap();
            rec.transition(
                &atlas,
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                ImageLayout::GENERAL,
                Access::TRANSFER_READ,
                Access::COMPUTE_WRITE,
            );
            rec.barrier(Access::TRANSFER_WRITE, Access::HOST_READ);
            rec.submit(None).unwrap();
            cmd.wait().unwrap();
            let texels: Vec<[u8; 4]> = out.read(0, (w * h) as usize).unwrap();
            // VELLUM_DUMP_ATLAS=path.pgm: save the atlas to look at.
            if let Some(path) = std::env::var_os("VELLUM_DUMP_ATLAS") {
                let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
                pgm.extend(texels.iter().map(|t| t[0]));
                std::fs::write(path, pgm).unwrap();
            }
            rects
                .iter()
                .map(|r| {
                    let mut c = Vec::new();
                    for y in r.y..r.y + r.height {
                        for x in r.x..r.x + r.width {
                            let t = texels[(y * w + x) as usize];
                            assert!(t.iter().all(|&v| v == t[0]), "channels differ: {t:?}");
                            c.push(f32::from(t[0]) / 255.0);
                        }
                    }
                    c
                })
                .collect()
        }
    }

    /// GPU result within one 8-bit step (plus fixed-point slack) of the
    /// reference, and treats near-zero reference as zero (the haze clamp).
    fn assert_matches_reference(gpu: &[f32], outline: &Outline) {
        let reference = reference(outline);
        assert_eq!(gpu.len(), reference.len());
        let mut worst = 0.0f32;
        for (i, (&g, &r)) in gpu.iter().zip(&reference).enumerate() {
            let r = if r < 1.0 / 512.0 { 0.0 } else { r };
            let diff = (g - r).abs();
            worst = worst.max(diff);
            assert!(
                diff <= 1.5 / 255.0,
                "pixel ({}, {}): gpu {g} vs reference {r}",
                i as u32 % outline.width,
                i as u32 / outline.width
            );
        }
        eprintln!("worst difference {:.4} ({:.2} steps)", worst, worst * 255.0);
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn gpu_matches_reference_on_shapes() {
        let mut g = gpu();
        let shapes = [
            polygon(5, 5, &[[1.5, 1.5], [3.5, 1.5], [3.5, 3.5], [1.5, 3.5]]),
            polygon(6, 6, &[[1.0, 1.0], [5.0, 5.0], [1.0, 5.0]]),
            // Long shallow edge crossing many columns in one row.
            polygon(40, 4, &[[1.0, 1.2], [38.7, 2.9], [1.0, 2.9]]),
            // Self-overlapping (winding 2 in the middle) stays capped at 1.
            polygon(
                8,
                8,
                &[
                    [1.0, 1.0],
                    [7.0, 1.0],
                    [7.0, 7.0],
                    [1.0, 7.0],
                    [1.0, 1.0],
                    [7.0, 1.0],
                    [7.0, 7.0],
                    [1.0, 7.0],
                ],
            ),
        ];
        let results = g.rasterize(&shapes);
        for (gpu, shape) in results.iter().zip(&shapes) {
            assert_matches_reference(gpu, shape);
        }
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn gpu_matches_reference_on_real_glyphs() {
        let mut g = gpu();
        let config = font::fontconfig::Config::new().unwrap();
        let mut outlines = Vec::new();
        for size in [11, 16, 31] {
            let m = config
                .match_font(&format!("monospace:pixelsize={size}"), 96.0)
                .unwrap();
            let font = font::Font::load(&m).unwrap();
            for ch in "ag@&Q%".chars() {
                outlines.push(font.glyph(font.glyph_id(ch).unwrap()).unwrap().outline);
            }
        }
        let results = g.rasterize(&outlines);
        for (gpu, outline) in results.iter().zip(&outlines) {
            assert_matches_reference(gpu, outline);
            // Clearly drawn (not necessarily fully covering any pixel:
            // slight hinting leaves vertical stems off the pixel grid), and
            // the padding row stays empty.
            assert!(gpu.iter().any(|&c| c > 0.75));
            let w = outline.width as usize;
            assert!(gpu[..w].iter().all(|&c| c == 0.0), "top padding row");
        }
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn batches_accumulate_in_the_atlas() {
        let mut g = gpu();
        let a = polygon(5, 5, &[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]]);
        let b = polygon(4, 4, &[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0]]);
        let first = g.rasterize(std::slice::from_ref(&a));
        let second = g.rasterize(std::slice::from_ref(&b));
        assert_matches_reference(&first[0], &a);
        assert_matches_reference(&second[0], &b);
        assert!(!g.raster.pending());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn empty_glyphs_take_no_space_and_full_atlas_is_reported() {
        let mut g = gpu();
        assert_eq!(g.raster.queue(&Outline::default()), Ok(None));
        let big = polygon(200, 200, &[[1.0, 1.0], [199.0, 1.0], [199.0, 199.0]]);
        g.raster.queue(&big).unwrap().unwrap();
        assert_eq!(g.raster.queue(&big), Err(AtlasFull));
        assert!(g.raster.clear().is_err(), "batch still pending");
        let mut cmd = CommandBuffer::new(&g.device).unwrap();
        let mut rec = cmd.begin().unwrap();
        g.raster.record(&mut rec).unwrap();
        rec.submit(None).unwrap();
        cmd.wait().unwrap();
        g.raster.clear().unwrap();
        assert!(g.raster.queue(&big).unwrap().is_some());
    }
}
