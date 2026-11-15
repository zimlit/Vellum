//! Fonts: finding them (fontconfig), reading them (skrifa), and turning
//! glyphs into line segments for the GPU rasterizer.
//!
//! A [`FontSet`] holds the four faces the terminal draws with (regular,
//! bold, italic, bold italic). Faces the family lacks are synthesized from
//! another face (see [`Synthetic`]).

pub mod flatten;
pub mod fontconfig;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, Engine, HintingInstance, HintingOptions, SmoothMode, Target};
use skrifa::{FontRef, GlyphId, MetadataProvider};

pub use flatten::{Line, Outline, Synthetic};
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
    /// Shared with synthetic faces made from this one.
    data: Arc<[u8]>,
    index: u32,
    size: f32,
    rendering: Rendering,
    /// `None` when hinting is off.
    hinting: Option<HintingInstance>,
    /// Applied to every glyph; nonzero for a synthesized face.
    synthetic: Synthetic,
    /// Family and style, for logs.
    name: String,
    /// Where it was loaded from (empty for fonts made from data).
    file: PathBuf,
    metrics: CellMetrics,
}

impl Font {
    /// Load the font fontconfig picked, at its pixel size and settings.
    pub fn load(m: &FontMatch) -> Result<Self> {
        Self::load_at(m, m.pixel_size as f32)
    }

    /// Load the font fontconfig picked at `size` pixels.
    pub fn load_at(m: &FontMatch, size: f32) -> Result<Self> {
        let data = read(&m.path)?;
        let mut font = Self::from_data(data, m.index, size, m.rendering)
            .with_context(|| format!("loading {}", m.path.display()))?;
        font.name = format!("{} {}", m.family, m.style);
        font.file.clone_from(&m.path);
        Ok(font)
    }

    pub fn from_data(
        data: impl Into<Arc<[u8]>>,
        index: u32,
        size: f32,
        rendering: Rendering,
    ) -> Result<Self> {
        if !(size > 0.0 && size.is_finite()) {
            return Err(anyhow!("invalid font size {size}"));
        }
        let mut font = Self {
            data: data.into(),
            index,
            size,
            rendering,
            hinting: None,
            synthetic: Synthetic::default(),
            name: String::new(),
            file: PathBuf::new(),
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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn file(&self) -> &Path {
        &self.file
    }

    pub fn synthetic(&self) -> Synthetic {
        self.synthetic
    }

    /// This face with `synthetic` added on top of what it has.
    pub fn synthesize(&self, synthetic: Synthetic) -> Result<Self> {
        let mut font = Self::from_data(self.data.clone(), self.index, self.size, self.rendering)?;
        font.synthetic = Synthetic {
            embolden: self.synthetic.embolden + synthetic.embolden,
            skew: self.synthetic.skew + synthetic.skew,
        };
        font.name.clone_from(&self.name);
        font.file.clone_from(&self.file);
        Ok(font)
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
        let mut pen = Flattener::with_synthetic(TOLERANCE, self.synthetic);
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

/// Which of the four faces text is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}

impl Face {
    pub const ALL: [Self; 4] = [Self::Regular, Self::Bold, Self::Italic, Self::BoldItalic];

    pub fn new(bold: bool, italic: bool) -> Self {
        match (bold, italic) {
            (false, false) => Self::Regular,
            (true, false) => Self::Bold,
            (false, true) => Self::Italic,
            (true, true) => Self::BoldItalic,
        }
    }

    pub fn is_bold(self) -> bool {
        matches!(self, Self::Bold | Self::BoldItalic)
    }

    pub fn is_italic(self) -> bool {
        matches!(self, Self::Italic | Self::BoldItalic)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Bold => "bold",
            Self::Italic => "italic",
            Self::BoldItalic => "bold italic",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The four faces, all at the regular face's size. The cell size comes
/// from the regular face only.
pub struct FontSet {
    faces: [Font; 4],
}

impl FontSet {
    /// The faces for `pattern` at `dpi`. The regular face is what the
    /// pattern matches; the others are asked for by weight and slant, and
    /// synthesized when the family doesn't have them.
    pub fn load(config: &fontconfig::Config, pattern: &str, dpi: f64) -> Result<Self> {
        let regular_match = config.match_font(pattern, dpi)?;
        let regular = Font::load(&regular_match)?;
        let mut real: [Option<Font>; 4] = [None, None, None, None];
        for face in [Face::Bold, Face::Italic, Face::BoldItalic] {
            let m = config.match_face(pattern, dpi, face.is_bold(), face.is_italic())?;
            if is_real_face(&regular_match, &m, face) {
                real[face.index()] = Some(Font::load_at(&m, regular.size())?);
            }
        }
        real[0] = Some(regular);
        Self::complete(real)
    }

    /// Only a regular face; the others synthesized from it.
    pub fn from_regular(regular: Font) -> Result<Self> {
        Self::complete([Some(regular), None, None, None])
    }

    /// Fill in missing faces from the closest real one: bold italic from
    /// italic (embolden) or bold (slant) before falling back to regular.
    fn complete(mut faces: [Option<Font>; 4]) -> Result<Self> {
        let size = faces[0].as_ref().expect("regular face").size();
        let bold = Synthetic {
            embolden: Synthetic::embolden_for(size),
            skew: 0.0,
        };
        let italic = Synthetic {
            embolden: 0.0,
            skew: Synthetic::ITALIC_SKEW,
        };
        let both = Synthetic {
            embolden: bold.embolden,
            skew: italic.skew,
        };
        let get = |faces: &[Option<Font>; 4], f: Face| faces[f.index()].is_some();
        if !get(&faces, Face::BoldItalic) {
            let (base, add) = if get(&faces, Face::Italic) {
                (Face::Italic, bold)
            } else if get(&faces, Face::Bold) {
                (Face::Bold, italic)
            } else {
                (Face::Regular, both)
            };
            let font = faces[base.index()].as_ref().unwrap().synthesize(add)?;
            faces[Face::BoldItalic.index()] = Some(font);
        }
        for (face, add) in [(Face::Bold, bold), (Face::Italic, italic)] {
            if !get(&faces, face) {
                let font = faces[0].as_ref().unwrap().synthesize(add)?;
                faces[face.index()] = Some(font);
            }
        }
        Ok(Self {
            faces: faces.map(|f| f.expect("every face filled in")),
        })
    }

    pub fn face(&self, face: Face) -> &Font {
        &self.faces[face.index()]
    }

    pub fn regular(&self) -> &Font {
        self.face(Face::Regular)
    }

    pub fn cell(&self) -> CellMetrics {
        self.regular().cell()
    }
}

/// Whether fontconfig's answer `m` for `face` really is that face of the
/// regular face's family: a different face of the same family, heavier
/// exactly when bold was asked for and slanted exactly when italic was.
/// (Variable fonts give the same file for every face; their named
/// instances aren't applied yet, so those count as missing.)
fn is_real_face(regular: &FontMatch, m: &FontMatch, face: Face) -> bool {
    m.family == regular.family
        && (m.path != regular.path || m.index != regular.index)
        && face.is_bold() == (m.weight > regular.weight)
        && face.is_italic() == (m.slant > regular.slant)
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

    fn fake_match(file: &str, family: &str, weight: i32, slant: i32) -> FontMatch {
        FontMatch {
            path: file.into(),
            index: 0,
            family: family.into(),
            style: String::new(),
            pixel_size: 16.0,
            weight,
            slant,
            rendering: Rendering::default(),
        }
    }

    #[test]
    fn real_faces_are_told_from_substitutes() {
        use fontconfig::{SLANT_ITALIC, SLANT_ROMAN, WEIGHT_BOLD, WEIGHT_REGULAR};
        let regular = fake_match("r.ttf", "Mono", WEIGHT_REGULAR, SLANT_ROMAN);
        let real = |m: &FontMatch, face| is_real_face(&regular, m, face);
        let bold = fake_match("b.ttf", "Mono", WEIGHT_BOLD, SLANT_ROMAN);
        let italic = fake_match("i.ttf", "Mono", WEIGHT_REGULAR, SLANT_ITALIC);
        let bold_italic = fake_match("bi.ttf", "Mono", WEIGHT_BOLD, SLANT_ITALIC);
        assert!(real(&bold, Face::Bold));
        assert!(real(&italic, Face::Italic));
        assert!(real(&bold_italic, Face::BoldItalic));
        // The regular face again (no such face in the family).
        assert!(!real(&regular, Face::Bold));
        // Wrong weight or slant for what was asked.
        assert!(!real(&bold_italic, Face::Italic));
        assert!(!real(&italic, Face::BoldItalic));
        assert!(!real(&bold, Face::BoldItalic));
        // Another family.
        let other = fake_match("o.ttf", "Other Mono", WEIGHT_BOLD, SLANT_ROMAN);
        assert!(!real(&other, Face::Bold));
        // One file for every face (a variable font).
        let variable = fake_match("r.ttf", "Mono", WEIGHT_BOLD, SLANT_ROMAN);
        assert!(!real(&variable, Face::Bold));
    }

    #[test]
    fn font_set_has_every_face_at_one_size() {
        let config = fontconfig::Config::new().unwrap();
        let set = FontSet::load(&config, "monospace:pixelsize=20", 96.0).unwrap();
        let regular = set.regular();
        assert!(regular.synthetic().is_none());
        for face in Face::ALL {
            let font = set.face(face);
            assert_eq!(font.size(), regular.size(), "{face:?}");
            let s = font.synthetic();
            // Synthesized only for the missing part.
            if !face.is_bold() {
                assert_eq!(s.embolden, 0.0, "{face:?}");
            }
            if !face.is_italic() {
                assert_eq!(s.skew, 0.0, "{face:?}");
            }
            assert!(!font.name().is_empty(), "{face:?}");
        }
        assert_eq!(set.cell(), regular.cell());
    }

    #[test]
    fn synthesized_faces() {
        let set = FontSet::from_regular(monospace("monospace:pixelsize=20")).unwrap();
        let s = |f| set.face(f).synthetic();
        assert_eq!(s(Face::Bold).embolden, 1.0);
        assert_eq!(s(Face::Italic).skew, Synthetic::ITALIC_SKEW);
        assert_eq!(
            s(Face::BoldItalic),
            Synthetic {
                embolden: 1.0,
                skew: Synthetic::ITALIC_SKEW
            }
        );

        // Italic: every point sheared by its height above the baseline.
        let outline = |face| {
            let font = set.face(face);
            font.glyph(font.glyph_id('l').unwrap()).unwrap().outline
        };
        let (regular, italic) = (outline(Face::Regular), outline(Face::Italic));
        assert_eq!(regular.lines.len(), italic.lines.len());
        // Back to font space: x right, y up from the baseline at the pen.
        let font_space = |o: &Outline, p: [f32; 2]| [p[0] + o.left as f32, -(p[1] + o.top as f32)];
        for (r, i) in regular.lines.iter().zip(&italic.lines) {
            for (rp, ip) in [(r.p0, i.p0), (r.p1, i.p1)] {
                let (rp, ip) = (font_space(&regular, rp), font_space(&italic, ip));
                assert!((ip[1] - rp[1]).abs() < 1e-3, "{rp:?} {ip:?}");
                let expected = rp[0] + Synthetic::ITALIC_SKEW * rp[1];
                assert!((ip[0] - expected).abs() < 1e-3, "{rp:?} {ip:?}");
            }
        }
        // Bold: one pixel wider, twice the lines.
        let bold = outline(Face::Bold);
        assert_eq!(bold.width, regular.width + 1);
        assert_eq!(bold.lines.len(), 2 * regular.lines.len());
    }
}
