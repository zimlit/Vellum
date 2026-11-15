//! Fonts: finding them (fontconfig), reading them (skrifa), and turning
//! glyphs into line segments for the GPU rasterizer.

pub mod flatten;
pub mod fontconfig;

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, Engine, HintingInstance, HintingOptions, SmoothMode, Target};
use skrifa::{FontRef, GlyphId, MetadataProvider};

pub use flatten::{Line, Outline};
pub use fontconfig::{FontMatch, HintStyle, Rendering};

use flatten::{Flattener, TOLERANCE};

/// The terminal cell size a font implies, in whole pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellMetrics {
    pub width: u32,
    pub height: u32,
    /// Distance from the cell's top edge down to the baseline.
    pub baseline: u32,
    pub underline: Stroke,
    pub strikeout: Stroke,
}

/// A horizontal line across the cell (underline, strikeout), in whole
/// pixels from its top edge. Always inside the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stroke {
    pub top: u32,
    pub thickness: u32,
}

impl Stroke {
    /// A stroke whose top is `top` pixels below the cell's top edge (may be
    /// fractional or out of range), rounded and kept inside the cell.
    fn fit(top: f32, thickness: f32, cell_height: u32) -> Self {
        let thickness = (thickness.round().max(1.0) as u32).min(cell_height);
        let top = (top.round().max(0.0) as u32).min(cell_height - thickness);
        Self { top, thickness }
    }
}

/// A rendered glyph: its outline as lines plus where to place it.
#[derive(Clone, Debug)]
pub struct Glyph {
    pub outline: Outline,
    /// Horizontal advance in pixels (rounded when hinted).
    pub advance: f32,
}

/// One face of a font file at one pixel size, with hinting set up.
pub struct Font {
    data: Vec<u8>,
    index: u32,
    size: f32,
    rendering: Rendering,
    /// `None` when hinting is off.
    hinting: Option<HintingInstance>,
    metrics: CellMetrics,
}

impl Font {
    /// Load the font fontconfig picked, at its pixel size and settings.
    pub fn load(m: &FontMatch) -> Result<Self> {
        let data = read(&m.path)?;
        Self::from_data(data, m.index, m.pixel_size as f32, m.rendering)
            .with_context(|| format!("loading {}", m.path.display()))
    }

    pub fn from_data(data: Vec<u8>, index: u32, size: f32, rendering: Rendering) -> Result<Self> {
        if !(size > 0.0 && size.is_finite()) {
            return Err(anyhow!("invalid font size {size}"));
        }
        let mut font = Self {
            data,
            index,
            size,
            rendering,
            hinting: None,
            metrics: CellMetrics {
                width: 0,
                height: 0,
                baseline: 0,
                underline: Stroke {
                    top: 0,
                    thickness: 0,
                },
                strikeout: Stroke {
                    top: 0,
                    thickness: 0,
                },
            },
        };
        let font_ref = font.font_ref()?;
        font.hinting = hinting_options(rendering)
            .map(|options| {
                HintingInstance::new(
                    &font_ref.outline_glyphs(),
                    Size::new(size),
                    LocationRef::default(),
                    options,
                )
            })
            .transpose()
            .map_err(|e| anyhow!("setting up hinting: {e}"))?;
        font.metrics = font.cell_metrics()?;
        Ok(font)
    }

    fn font_ref(&self) -> Result<FontRef<'_>> {
        FontRef::from_index(&self.data, self.index).map_err(|e| anyhow!("reading font: {e}"))
    }

    pub fn size(&self) -> f32 {
        self.size
    }

    pub fn rendering(&self) -> Rendering {
        self.rendering
    }

    pub fn cell(&self) -> CellMetrics {
        self.metrics
    }

    /// The glyph for `ch`, or `None` if the font doesn't have one.
    pub fn glyph_id(&self, ch: char) -> Option<GlyphId> {
        self.font_ref().ok()?.charmap().map(ch)
    }

    /// The hinted (per the font's settings) outline of `glyph`, flattened.
    pub fn glyph(&self, glyph: GlyphId) -> Result<Glyph> {
        let font = self.font_ref()?;
        let outlines = font.outline_glyphs();
        let outline = outlines
            .get(glyph)
            .ok_or_else(|| anyhow!("font has no outline for glyph {}", glyph.to_u32()))?;
        let settings = match &self.hinting {
            Some(instance) => DrawSettings::hinted(instance, false),
            None => DrawSettings::unhinted(Size::new(self.size), LocationRef::default()),
        };
        let mut pen = Flattener::new(TOLERANCE);
        let adjusted = outline
            .draw(settings, &mut pen)
            .map_err(|e| anyhow!("drawing glyph {}: {e}", glyph.to_u32()))?;
        let advance = match adjusted.advance_width {
            Some(a) => a,
            None => font
                .glyph_metrics(Size::new(self.size), LocationRef::default())
                .advance_width(glyph)
                .unwrap_or(0.0),
        };
        Ok(Glyph {
            outline: pen.finish(),
            advance,
        })
    }

    /// Cell size: the advance of `0` (fonts used in terminals are
    /// monospaced) by ascent + descent + line gap.
    fn cell_metrics(&self) -> Result<CellMetrics> {
        let font = self.font_ref()?;
        let m = font.metrics(Size::new(self.size), LocationRef::default());
        let advance = ['0', 'M', ' ']
            .into_iter()
            .find_map(|ch| self.glyph_id(ch))
            .map(|g| self.glyph(g))
            .transpose()?
            .map_or(self.size / 2.0, |g| g.advance);
        let descent = -m.descent; // skrifa reports it negative (below baseline)
        let baseline = (m.ascent + m.leading / 2.0).ceil();
        let height = (m.ascent + descent + m.leading)
            .ceil()
            .max(baseline + descent.ceil());
        let cell_height = height.max(1.0) as u32;

        // Font decoration offsets are y-up from the baseline. Fallbacks
        // when the font has none: a 1/14 em line halfway down the descent,
        // and a strikeout centred on half the x-height.
        let fallback = self.size / 14.0;
        let underline = m.underline.map_or((-descent / 2.0, fallback), |d| {
            // The post table's position is nominally the top of the line,
            // but fonts set it as the centre (as FreeType notes and most
            // terminals assume).
            (d.offset, d.thickness)
        });
        let (u_center, u_thickness) = underline;
        let underline = Stroke::fit(
            baseline - u_center - u_thickness / 2.0,
            u_thickness,
            cell_height,
        );
        let strikeout = match m.strikeout {
            // OS/2: the top of the stroke.
            Some(d) if d.thickness > 0.0 => {
                Stroke::fit(baseline - d.offset, d.thickness, cell_height)
            }
            _ => {
                let x_height = m.x_height.unwrap_or(m.ascent / 2.0);
                Stroke::fit(
                    baseline - x_height / 2.0 - u_thickness / 2.0,
                    u_thickness,
                    cell_height,
                )
            }
        };
        Ok(CellMetrics {
            width: advance.round().max(1.0) as u32,
            height: cell_height,
            baseline: baseline.max(0.0) as u32,
            underline,
            strikeout,
        })
    }
}

/// skrifa hinting for fontconfig's settings, or `None` for unhinted.
fn hinting_options(r: Rendering) -> Option<HintingOptions> {
    if r.hint_style == HintStyle::None {
        return None;
    }
    let target = if !r.antialias {
        Target::Mono
    } else if r.hint_style == HintStyle::Slight {
        // Vertical-only hinting, like FreeType's light mode.
        SmoothMode::Light.into()
    } else {
        SmoothMode::Normal.into()
    };
    let engine = if r.autohint {
        Engine::Auto(None)
    } else {
        // The font's own hints if it has any, else the autohinter
        // (FreeType's default rule).
        Engine::AutoFallback
    };
    Some(HintingOptions { engine, target })
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monospace(pattern: &str) -> Font {
        let config = fontconfig::Config::new().unwrap();
        Font::load(&config.match_font(pattern, 96.0).unwrap()).unwrap()
    }

    #[test]
    fn hinting_follows_fontconfig_settings() {
        let r = Rendering::default();
        assert!(
            hinting_options(Rendering {
                hint_style: HintStyle::None,
                ..r
            })
            .is_none()
        );
        let slight = hinting_options(r).unwrap();
        assert_eq!(slight.target, SmoothMode::Light.into());
        assert!(matches!(slight.engine, Engine::AutoFallback));
        let full = hinting_options(Rendering {
            hint_style: HintStyle::Full,
            autohint: true,
            ..r
        })
        .unwrap();
        assert_eq!(full.target, SmoothMode::Normal.into());
        assert!(matches!(full.engine, Engine::Auto(None)));
        let mono = hinting_options(Rendering {
            antialias: false,
            ..r
        })
        .unwrap();
        assert_eq!(mono.target, Target::Mono);
    }

    #[test]
    fn cell_metrics_are_plausible_and_scale() {
        let small = monospace("monospace:pixelsize=16").cell();
        let large = monospace("monospace:pixelsize=32").cell();
        for c in [small, large] {
            assert!(c.width > 0 && c.height > c.width, "{c:?}");
            assert!(c.baseline > 0 && c.baseline < c.height, "{c:?}");
        }
        assert!(large.width >= 2 * small.width - 1, "{small:?} {large:?}");
        assert!(large.height >= 2 * small.height - 2, "{small:?} {large:?}");
    }

    #[test]
    fn decorations_sit_where_expected() {
        for size in [8, 16, 32] {
            let c = monospace(&format!("monospace:pixelsize={size}")).cell();
            let (u, s) = (c.underline, c.strikeout);
            for stroke in [u, s] {
                assert!(stroke.thickness >= 1, "{c:?}");
                assert!(stroke.top + stroke.thickness <= c.height, "{c:?}");
            }
            assert!(
                u.top + 1 >= c.baseline,
                "underline at or below the baseline: {c:?}"
            );
            assert!(s.top < c.baseline, "strikeout above the baseline: {c:?}");
            assert!(s.top > c.height / 6, "strikeout below the top: {c:?}");
        }
    }

    #[test]
    fn strokes_are_kept_inside_the_cell() {
        assert_eq!(
            Stroke::fit(-3.0, 2.0, 10),
            Stroke {
                top: 0,
                thickness: 2
            }
        );
        assert_eq!(
            Stroke::fit(9.6, 2.4, 10),
            Stroke {
                top: 8,
                thickness: 2
            }
        );
        assert_eq!(
            Stroke::fit(4.0, 0.2, 10),
            Stroke {
                top: 4,
                thickness: 1
            }
        );
        assert_eq!(
            Stroke::fit(0.0, 30.0, 10),
            Stroke {
                top: 0,
                thickness: 10
            }
        );
    }

    #[test]
    fn letter_outline_fits_its_cell() {
        let font = monospace("monospace:pixelsize=20");
        let cell = font.cell();
        let g = font.glyph(font.glyph_id('H').unwrap()).unwrap();
        let o = &g.outline;
        assert!(!o.lines.is_empty());
        assert!((g.advance - cell.width as f32).abs() < 1.0);
        // Within the cell, allowing for the padding.
        let pad = flatten::PADDING;
        assert!(
            o.left >= -pad && o.left + o.width as i32 <= cell.width as i32 + pad,
            "{o:?}"
        );
        assert!(-o.top <= cell.baseline as i32 + pad, "{o:?}");
        assert!(o.top + o.height as i32 >= -pad, "{o:?}");
        // Every point is inside the bitmap.
        for l in &o.lines {
            for p in [l.p0, l.p1] {
                assert!(p[0] >= 0.0 && p[0] <= o.width as f32);
                assert!(p[1] >= 0.0 && p[1] <= o.height as f32);
            }
        }
    }

    #[test]
    fn space_has_no_outline_and_missing_chars_no_glyph() {
        let font = monospace("monospace:pixelsize=20");
        let space = font.glyph(font.glyph_id(' ').unwrap()).unwrap();
        assert!(space.outline.lines.is_empty());
        assert_eq!(space.outline.width, 0);
        assert!(space.advance > 0.0);
        // A noncharacter no font maps.
        assert_eq!(font.glyph_id('\u{FFFF}'), None);
    }

    #[test]
    fn hinting_changes_outlines() {
        let config = fontconfig::Config::new().unwrap();
        let m = config.match_font("monospace:pixelsize=13", 96.0).unwrap();
        let load = |hint_style| {
            let rendering = Rendering {
                hint_style,
                ..m.rendering
            };
            Font::from_data(read(&m.path).unwrap(), m.index, 13.0, rendering).unwrap()
        };
        let (hinted, unhinted) = (load(HintStyle::Full), load(HintStyle::None));
        let id = hinted.glyph_id('e').unwrap();
        assert_ne!(
            hinted.glyph(id).unwrap().outline.lines,
            unhinted.glyph(id).unwrap().outline.lines
        );
    }

    #[test]
    fn invalid_data_or_size_is_an_error() {
        let r = Rendering::default();
        assert!(Font::from_data(b"not a font".to_vec(), 0, 12.0, r).is_err());
        let path = fontconfig::Config::new()
            .unwrap()
            .match_font("monospace", 96.0)
            .unwrap()
            .path;
        assert!(Font::from_data(read(&path).unwrap(), 0, 0.0, r).is_err());
        assert!(Font::from_data(read(&path).unwrap(), 0, f32::NAN, r).is_err());
    }
}
