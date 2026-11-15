//! Minimal hand-written bindings to fontconfig: match a pattern such as
//! `monospace:size=11` to a font file, and read the rendering settings the
//! user's configuration attaches to it (antialiasing, hinting, subpixel
//! order, LCD filter).

use std::ffi::{CStr, CString, c_char, c_double, c_int};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::ptr::{self, NonNull};

use anyhow::{Context, Result, anyhow, bail};

#[allow(non_camel_case_types)]
mod ffi {
    use std::ffi::{c_char, c_double, c_int};

    #[repr(C)]
    pub struct FcConfig {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub struct FcPattern {
        _opaque: [u8; 0],
    }

    pub type FcBool = c_int;
    pub type FcChar8 = u8;
    /// `enum _FcResult`.
    pub type FcResult = c_int;
    pub const FC_RESULT_MATCH: FcResult = 0;
    pub const FC_RESULT_NO_MATCH: FcResult = 1;
    /// `enum _FcMatchKind`.
    pub const FC_MATCH_PATTERN: c_int = 0;

    #[link(name = "fontconfig")]
    unsafe extern "C" {
        pub fn FcInitLoadConfigAndFonts() -> *mut FcConfig;
        pub fn FcConfigDestroy(config: *mut FcConfig);
        pub fn FcConfigSubstitute(config: *mut FcConfig, p: *mut FcPattern, kind: c_int) -> FcBool;
        pub fn FcDefaultSubstitute(pattern: *mut FcPattern);
        pub fn FcFontMatch(
            config: *mut FcConfig,
            p: *mut FcPattern,
            result: *mut FcResult,
        ) -> *mut FcPattern;

        pub fn FcNameParse(name: *const FcChar8) -> *mut FcPattern;
        pub fn FcPatternDestroy(p: *mut FcPattern);
        pub fn FcPatternAddDouble(p: *mut FcPattern, object: *const c_char, d: c_double) -> FcBool;
        pub fn FcPatternAddInteger(p: *mut FcPattern, object: *const c_char, i: c_int) -> FcBool;
        pub fn FcPatternDel(p: *mut FcPattern, object: *const c_char) -> FcBool;
        pub fn FcPatternGetString(
            p: *const FcPattern,
            object: *const c_char,
            n: c_int,
            s: *mut *mut FcChar8,
        ) -> FcResult;
        pub fn FcPatternGetInteger(
            p: *const FcPattern,
            object: *const c_char,
            n: c_int,
            i: *mut c_int,
        ) -> FcResult;
        pub fn FcPatternGetDouble(
            p: *const FcPattern,
            object: *const c_char,
            n: c_int,
            d: *mut c_double,
        ) -> FcResult;
        pub fn FcPatternGetBool(
            p: *const FcPattern,
            object: *const c_char,
            n: c_int,
            b: *mut FcBool,
        ) -> FcResult;
    }
}

/// How glyphs should be hinted (`hintstyle`, `hinting`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintStyle {
    None,
    /// Vertical only (FreeType's "light").
    Slight,
    Medium,
    Full,
}

/// Physical order of a pixel's color elements (`rgba`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubpixelOrder {
    Unknown,
    Rgb,
    Bgr,
    VerticalRgb,
    VerticalBgr,
    /// Grayscale antialiasing only.
    None,
}

/// Filter for LCD subpixel rendering (`lcdfilter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LcdFilter {
    None,
    Default,
    Light,
    Legacy,
}

/// The rendering settings fontconfig attaches to a matched font.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rendering {
    pub antialias: bool,
    /// `hintstyle`, or `None` when `hinting` is off.
    pub hint_style: HintStyle,
    /// Use the autohinter even if the font has its own hints.
    pub autohint: bool,
    pub subpixel: SubpixelOrder,
    pub lcd_filter: LcdFilter,
}

impl Default for Rendering {
    /// What a stock fontconfig setup gives: grayscale antialiasing with
    /// slight hinting.
    fn default() -> Self {
        Self {
            antialias: true,
            hint_style: HintStyle::Slight,
            autohint: false,
            subpixel: SubpixelOrder::Unknown,
            lcd_filter: LcdFilter::Default,
        }
    }
}

impl HintStyle {
    fn from_fc(v: c_int) -> Option<Self> {
        Some(match v {
            0 => Self::None,
            1 => Self::Slight,
            2 => Self::Medium,
            3 => Self::Full,
            _ => return None,
        })
    }
}

impl SubpixelOrder {
    fn from_fc(v: c_int) -> Option<Self> {
        Some(match v {
            0 => Self::Unknown,
            1 => Self::Rgb,
            2 => Self::Bgr,
            3 => Self::VerticalRgb,
            4 => Self::VerticalBgr,
            5 => Self::None,
            _ => return None,
        })
    }
}

impl LcdFilter {
    fn from_fc(v: c_int) -> Option<Self> {
        Some(match v {
            0 => Self::None,
            1 => Self::Default,
            2 => Self::Light,
            3 => Self::Legacy,
            _ => return None,
        })
    }
}

/// `FC_WEIGHT_REGULAR`, `FC_WEIGHT_BOLD`, `FC_SLANT_ROMAN`, `FC_SLANT_ITALIC`.
pub const WEIGHT_REGULAR: c_int = 80;
pub const WEIGHT_BOLD: c_int = 200;
pub const SLANT_ROMAN: c_int = 0;
pub const SLANT_ITALIC: c_int = 100;

/// A font file chosen by fontconfig.
#[derive(Clone, Debug)]
pub struct FontMatch {
    pub path: PathBuf,
    /// Face index within the file (collections hold several).
    pub index: u32,
    pub family: String,
    pub style: String,
    /// Size in pixels, from the pattern's size and DPI.
    pub pixel_size: f64,
    /// Fontconfig's weight scale: 80 regular, 200 bold.
    pub weight: c_int,
    /// 0 roman, 100 italic, 110 oblique.
    pub slant: c_int,
    pub rendering: Rendering,
}

/// A loaded fontconfig configuration (fonts.conf plus the font cache).
pub struct Config(NonNull<ffi::FcConfig>);

impl Config {
    pub fn new() -> Result<Self> {
        // SAFETY: returns a new reference or null.
        let raw = unsafe { ffi::FcInitLoadConfigAndFonts() };
        NonNull::new(raw)
            .map(Self)
            .ok_or_else(|| anyhow!("loading the fontconfig configuration failed"))
    }

    /// The best font for `pattern` (fontconfig syntax, e.g.
    /// `monospace:size=11` or `Iosevka:style=Bold:pixelsize=24`), with point
    /// sizes converted to pixels at `dpi` unless the pattern sets its own.
    pub fn match_font(&self, pattern: &str, dpi: f64) -> Result<FontMatch> {
        self.match_with(pattern, dpi, |_| {})
    }

    /// Like [`Config::match_font`], asking for a bold and/or italic face.
    /// These replace any weight, slant or style the pattern names. What
    /// comes back may still not be bold or italic, if the family has no
    /// such face: check [`FontMatch::weight`] and [`FontMatch::slant`].
    pub fn match_face(
        &self,
        pattern: &str,
        dpi: f64,
        bold: bool,
        italic: bool,
    ) -> Result<FontMatch> {
        self.match_with(pattern, dpi, |pat| {
            let p = pat.0.as_ptr();
            // SAFETY: p is a live pattern; the object names are static.
            unsafe {
                if bold || italic {
                    ffi::FcPatternDel(p, c"style".as_ptr());
                }
                if bold {
                    ffi::FcPatternDel(p, c"weight".as_ptr());
                    ffi::FcPatternAddInteger(p, c"weight".as_ptr(), WEIGHT_BOLD);
                }
                if italic {
                    ffi::FcPatternDel(p, c"slant".as_ptr());
                    ffi::FcPatternAddInteger(p, c"slant".as_ptr(), SLANT_ITALIC);
                }
            }
        })
    }

    fn match_with(
        &self,
        pattern: &str,
        dpi: f64,
        edit: impl FnOnce(&Pattern),
    ) -> Result<FontMatch> {
        let name = CString::new(pattern).context("font pattern contains a NUL byte")?;
        // SAFETY: name is NUL-terminated; result is a new pattern or null.
        let pat = Pattern::new(unsafe { ffi::FcNameParse(name.as_ptr().cast()) })
            .ok_or_else(|| anyhow!("invalid font pattern {pattern:?}"))?;
        if pat.double(c"dpi").is_none() {
            // SAFETY: pat is a live pattern; the object name is static.
            unsafe { ffi::FcPatternAddDouble(pat.0.as_ptr(), c"dpi".as_ptr(), dpi) };
        }
        edit(&pat);
        // SAFETY: live config and pattern. Substitution applies fonts.conf
        // edits, then fills defaults (including pixelsize = size * dpi / 72).
        let matched = unsafe {
            ffi::FcConfigSubstitute(self.0.as_ptr(), pat.0.as_ptr(), ffi::FC_MATCH_PATTERN);
            ffi::FcDefaultSubstitute(pat.0.as_ptr());
            let mut result = ffi::FC_RESULT_NO_MATCH;
            let m = ffi::FcFontMatch(self.0.as_ptr(), pat.0.as_ptr(), &mut result);
            if result != ffi::FC_RESULT_MATCH {
                // Free a pattern returned alongside a failure code, if any.
                drop(Pattern::new(m));
                bail!("no font matches {pattern:?}");
            }
            Pattern::new(m).ok_or_else(|| anyhow!("no font matches {pattern:?}"))?
        };
        matched.to_font_match()
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: drops the reference FcInitLoadConfigAndFonts gave us.
        unsafe { ffi::FcConfigDestroy(self.0.as_ptr()) }
    }
}

/// An owned `FcPattern`.
struct Pattern(NonNull<ffi::FcPattern>);

impl Pattern {
    fn new(raw: *mut ffi::FcPattern) -> Option<Self> {
        NonNull::new(raw).map(Self)
    }

    fn string(&self, object: &CStr) -> Option<&CStr> {
        let mut s = ptr::null_mut();
        // SAFETY: the returned string points into the pattern and lives as
        // long as it does (tied to &self).
        let r = unsafe { ffi::FcPatternGetString(self.0.as_ptr(), object.as_ptr(), 0, &mut s) };
        (r == ffi::FC_RESULT_MATCH && !s.is_null())
            .then(|| unsafe { CStr::from_ptr(s.cast::<c_char>()) })
    }

    fn int(&self, object: &CStr) -> Option<c_int> {
        let mut v = 0;
        let r = unsafe { ffi::FcPatternGetInteger(self.0.as_ptr(), object.as_ptr(), 0, &mut v) };
        (r == ffi::FC_RESULT_MATCH).then_some(v)
    }

    fn double(&self, object: &CStr) -> Option<c_double> {
        let mut v = 0.0;
        let r = unsafe { ffi::FcPatternGetDouble(self.0.as_ptr(), object.as_ptr(), 0, &mut v) };
        (r == ffi::FC_RESULT_MATCH).then_some(v)
    }

    fn bool(&self, object: &CStr) -> Option<bool> {
        let mut v = 0;
        let r = unsafe { ffi::FcPatternGetBool(self.0.as_ptr(), object.as_ptr(), 0, &mut v) };
        (r == ffi::FC_RESULT_MATCH).then_some(v != 0)
    }

    fn to_font_match(&self) -> Result<FontMatch> {
        let path = self
            .string(c"file")
            .ok_or_else(|| anyhow!("matched font has no file"))?;
        let text = |object| {
            self.string(object)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let defaults = Rendering::default();
        let hinting = self.bool(c"hinting").unwrap_or(true);
        let rendering = Rendering {
            antialias: self.bool(c"antialias").unwrap_or(defaults.antialias),
            hint_style: if hinting {
                self.int(c"hintstyle")
                    .and_then(HintStyle::from_fc)
                    .unwrap_or(defaults.hint_style)
            } else {
                HintStyle::None
            },
            autohint: self.bool(c"autohint").unwrap_or(defaults.autohint),
            subpixel: self
                .int(c"rgba")
                .and_then(SubpixelOrder::from_fc)
                .unwrap_or(defaults.subpixel),
            lcd_filter: self
                .int(c"lcdfilter")
                .and_then(LcdFilter::from_fc)
                .unwrap_or(defaults.lcd_filter),
        };
        Ok(FontMatch {
            path: PathBuf::from(std::ffi::OsStr::from_bytes(path.to_bytes())),
            // The upper 16 bits select a named variable-font instance.
            index: self.int(c"index").unwrap_or(0) as u32 & 0xffff,
            family: text(c"family"),
            style: text(c"style"),
            pixel_size: self
                .double(c"pixelsize")
                .ok_or_else(|| anyhow!("matched font has no pixel size"))?,
            weight: self.int(c"weight").unwrap_or(WEIGHT_REGULAR),
            slant: self.int(c"slant").unwrap_or(SLANT_ROMAN),
            rendering,
        })
    }
}

impl Drop for Pattern {
    fn drop(&mut self) {
        unsafe { ffi::FcPatternDestroy(self.0.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_values_follow_fontconfig() {
        assert_eq!(HintStyle::from_fc(1), Some(HintStyle::Slight));
        assert_eq!(HintStyle::from_fc(3), Some(HintStyle::Full));
        assert_eq!(HintStyle::from_fc(4), None);
        assert_eq!(SubpixelOrder::from_fc(1), Some(SubpixelOrder::Rgb));
        assert_eq!(SubpixelOrder::from_fc(5), Some(SubpixelOrder::None));
        assert_eq!(LcdFilter::from_fc(2), Some(LcdFilter::Light));
        assert_eq!(LcdFilter::from_fc(-1), None);
    }

    // These use the system's fontconfig and fonts. Any Linux system running
    // vellum has both, so they aren't marked #[ignore].

    #[test]
    fn monospace_matches_a_readable_file() {
        let config = Config::new().unwrap();
        let m = config.match_font("monospace", 96.0).unwrap();
        assert!(m.path.is_file(), "{m:?}");
        assert!(!m.family.is_empty());
    }

    #[test]
    fn point_size_scales_with_dpi_but_pixel_size_does_not() {
        let config = Config::new().unwrap();
        let at = |pattern: &str, dpi| config.match_font(pattern, dpi).unwrap().pixel_size;
        assert!((at("monospace:size=12", 72.0) - 12.0).abs() < 1e-6);
        assert!((at("monospace:size=12", 144.0) - 24.0).abs() < 1e-6);
        assert!((at("monospace:pixelsize=17", 144.0) - 17.0).abs() < 1e-6);
        // A dpi in the pattern wins over the argument.
        assert!((at("monospace:size=12:dpi=96", 300.0) - 16.0).abs() < 1e-6);
    }

    #[test]
    fn faces_are_asked_for_by_weight_and_slant() {
        let config = Config::new().unwrap();
        let regular = config.match_face("monospace", 96.0, false, false).unwrap();
        let bold = config.match_face("monospace", 96.0, true, false).unwrap();
        let italic = config.match_face("monospace", 96.0, false, true).unwrap();
        // The system may lack the faces, but never gets lighter or upright
        // for asking.
        assert!(bold.weight >= regular.weight, "{bold:?}");
        assert!(italic.slant >= regular.slant, "{italic:?}");
        // A style in the pattern doesn't win over the request.
        let m = config
            .match_face("monospace:style=Regular", 96.0, true, false)
            .unwrap();
        assert_eq!(m.weight, bold.weight);
    }

    #[test]
    fn bad_patterns_are_errors() {
        let config = Config::new().unwrap();
        assert!(config.match_font("mono\0space", 96.0).is_err());
    }
}
