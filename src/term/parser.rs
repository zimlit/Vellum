//! The escape sequence parser: Paul Williams' DEC/VT500 state machine
//! (<https://vt100.net/emu/dec_ansi_parser>), turning bytes into actions.
//!
//! It knows nothing about what sequences mean; a [`Perform`] implementation
//! gets called for each printable character, control function and complete
//! sequence. Differences from the original diagram:
//!
//! - **UTF-8 only.** Bytes 0x80–0xFF in the ground state are decoded as
//!   UTF-8 (invalid input becomes U+FFFD, one per maximal invalid subpart,
//!   like [`String::from_utf8_lossy`]); 8-bit C1 controls aren't recognized.
//!   Inside OSC and DCS strings they're passed through as data; inside
//!   escape and control sequences they're ignored.
//! - **Colon subparameters** (`CSI 38:2::255:0:0 m`, `CSI 4:3 m`) are kept
//!   instead of ignoring the sequence: see [`Params`].
//! - **OSC can end with BEL** as well as ST (`ESC \`), as in xterm.
//! - CAN and SUB abandon an OSC string without dispatching it.
//! - DEL (0x7F) is ignored everywhere.

/// Most parameters a sequence keeps; any more are dropped.
pub const MAX_PARAMS: usize = 32;
/// Most intermediate bytes (including a private marker like `?`); a
/// sequence with more is dispatched with `ignored` set.
const MAX_INTERMEDIATES: usize = 2;
/// Longest OSC string; longer ones aren't dispatched.
const MAX_OSC: usize = 4096;
/// Most `;`-separated OSC parameters; further `;` stay in the last one.
const MAX_OSC_PARAMS: usize = 16;

const CAN: u8 = 0x18;
const SUB: u8 = 0x1a;
const ESC: u8 = 0x1b;
const BEL: u8 = 0x07;
const DEL: u8 = 0x7f;

/// What the parser found. Called in input order.
pub trait Perform {
    /// A printable character (in the ground state).
    fn print(&mut self, ch: char);
    /// A C0 control byte (0x00–0x1F) to act on.
    fn execute(&mut self, byte: u8);
    /// `CSI` *params* *intermediates* *action*. A private marker (`?`, `>`,
    /// `<`, `=`) comes first in `intermediates`. `ignored`: there were too
    /// many intermediates, so the sequence is probably not understood.
    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], ignored: bool, action: u8);
    /// `ESC` *intermediates* *final*.
    fn esc_dispatch(&mut self, intermediates: &[u8], ignored: bool, byte: u8);
    /// `OSC` string, split on `;`.
    fn osc_dispatch(&mut self, params: &[&[u8]], bell_terminated: bool);
    /// Start of a DCS string with its header; data follows through `put`.
    fn hook(&mut self, _params: &Params, _intermediates: &[u8], _ignored: bool, _action: u8) {}
    /// One byte of DCS data.
    fn put(&mut self, _byte: u8) {}
    /// End of the DCS string (terminated, or abandoned by CAN/SUB).
    fn unhook(&mut self) {}
}

/// Numeric parameters of a control sequence, grouped by subparameters.
///
/// `CSI 1;38:2::10:20:30 m` has two groups: `[1]` and `[38, 2, 0, 10, 20, 30]`.
/// Empty values read as 0 (callers apply their own defaults); values
/// saturate at 65535.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Params {
    values: [u16; MAX_PARAMS],
    /// Bit `i`: value `i` is followed by `:`, so `i + 1` is in its group.
    colon: u32,
    len: usize,
}

impl Params {
    /// Number of values (subparameters count individually).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The parameter groups in order.
    pub fn iter(&self) -> Groups<'_> {
        Groups {
            params: self,
            pos: 0,
        }
    }

    fn clear(&mut self) {
        self.colon = 0;
        self.len = 0;
    }

    fn push(&mut self, value: u16, colon: bool) {
        if self.len == MAX_PARAMS {
            return;
        }
        self.values[self.len] = value;
        if colon {
            self.colon |= 1 << self.len;
        }
        self.len += 1;
    }
}

impl std::fmt::Debug for Params {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

pub struct Groups<'a> {
    params: &'a Params,
    pos: usize,
}

impl<'a> Iterator for Groups<'a> {
    type Item = &'a [u16];

    fn next(&mut self) -> Option<&'a [u16]> {
        let p = self.params;
        let start = self.pos;
        if start >= p.len {
            return None;
        }
        let mut end = start;
        while end + 1 < p.len && p.colon & (1 << end) != 0 {
            end += 1;
        }
        self.pos = end + 1;
        Some(&p.values[start..=end])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Ground,
    Escape,
    EscapeIntermediate,
    CsiEntry,
    CsiParam,
    CsiIntermediate,
    CsiIgnore,
    DcsEntry,
    DcsParam,
    DcsIntermediate,
    DcsPassthrough,
    DcsIgnore,
    OscString,
    /// SOS, PM and APC strings: consumed and ignored.
    SosPmApcString,
}

/// Incremental UTF-8 decoding, so characters can be split across reads.
#[derive(Default)]
struct Utf8 {
    code: u32,
    /// Continuation bytes still needed; 0 when not inside a character.
    need: u8,
    /// Allowed range of the next continuation byte (it's narrower than
    /// 0x80–0xBF after some lead bytes, to reject overlong encodings,
    /// surrogates and code points past U+10FFFF).
    lower: u8,
    upper: u8,
}

enum Decoded {
    /// Byte consumed, character not complete yet.
    Pending,
    Char(char),
    /// Invalid: print U+FFFD. With `reprocess`, the byte wasn't consumed
    /// (it ended an incomplete character) and must be handled again.
    Invalid {
        reprocess: bool,
    },
}

impl Utf8 {
    fn active(&self) -> bool {
        self.need > 0
    }

    fn reset(&mut self) {
        self.need = 0;
    }

    fn lead(&mut self, b: u8) -> Decoded {
        let (need, mask, lower, upper) = match b {
            0xc2..=0xdf => (1, 0x1f, 0x80, 0xbf),
            0xe0 => (2, 0x0f, 0xa0, 0xbf),
            0xe1..=0xec | 0xee..=0xef => (2, 0x0f, 0x80, 0xbf),
            0xed => (2, 0x0f, 0x80, 0x9f),
            0xf0 => (3, 0x07, 0x90, 0xbf),
            0xf1..=0xf3 => (3, 0x07, 0x80, 0xbf),
            0xf4 => (3, 0x07, 0x80, 0x8f),
            _ => return Decoded::Invalid { reprocess: false },
        };
        *self = Self {
            code: u32::from(b & mask),
            need,
            lower,
            upper,
        };
        Decoded::Pending
    }

    fn continuation(&mut self, b: u8) -> Decoded {
        if !(self.lower..=self.upper).contains(&b) {
            self.reset();
            return Decoded::Invalid { reprocess: true };
        }
        self.code = self.code << 6 | u32::from(b & 0x3f);
        self.need -= 1;
        (self.lower, self.upper) = (0x80, 0xbf);
        if self.need > 0 {
            return Decoded::Pending;
        }
        // The lead and range checks only let valid scalar values through.
        Decoded::Char(char::from_u32(self.code).unwrap_or(char::REPLACEMENT_CHARACTER))
    }
}

pub struct Parser {
    state: State,
    params: Params,
    /// The parameter value being read.
    param: u16,
    /// Whether the current sequence has any parameter bytes, so `CSI m`
    /// has no parameters but `CSI ; m` has two empty ones.
    has_params: bool,
    intermediates: [u8; MAX_INTERMEDIATES],
    intermediate_len: usize,
    /// Too many intermediates.
    ignoring: bool,
    osc: Vec<u8>,
    /// End of each OSC parameter but the last, in `osc`.
    osc_ends: [usize; MAX_OSC_PARAMS - 1],
    osc_params: usize,
    osc_overflow: bool,
    utf8: Utf8,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    pub fn new() -> Self {
        Self {
            state: State::Ground,
            params: Params::default(),
            param: 0,
            has_params: false,
            intermediates: [0; MAX_INTERMEDIATES],
            intermediate_len: 0,
            ignoring: false,
            osc: Vec::new(),
            osc_ends: [0; MAX_OSC_PARAMS - 1],
            osc_params: 0,
            osc_overflow: false,
            utf8: Utf8::default(),
        }
    }

    /// Parse `bytes`, calling `perform` for what they contain. Sequences
    /// (and UTF-8 characters) may be split across calls.
    pub fn advance<P: Perform>(&mut self, perform: &mut P, bytes: &[u8]) {
        for &b in bytes {
            self.byte(perform, b);
        }
    }

    fn byte<P: Perform>(&mut self, p: &mut P, b: u8) {
        if self.utf8.active() {
            match self.utf8.continuation(b) {
                Decoded::Pending => return,
                Decoded::Char(ch) => return p.print(ch),
                Decoded::Invalid { reprocess } => {
                    p.print(char::REPLACEMENT_CHARACTER);
                    if !reprocess {
                        return;
                    }
                }
            }
        }

        // Transitions from anywhere.
        match b {
            CAN | SUB => {
                // A cancelled OSC string is dropped without dispatching.
                if self.state == State::DcsPassthrough {
                    p.unhook();
                }
                p.execute(b);
                self.state = State::Ground;
                return;
            }
            ESC => {
                self.exit(p);
                self.clear();
                self.state = State::Escape;
                return;
            }
            _ => {}
        }

        match self.state {
            State::Ground => match b {
                0x00..=0x1f => p.execute(b),
                0x20..=0x7e => p.print(char::from(b)),
                DEL => {}
                0x80..=0xff => match self.utf8.lead(b) {
                    Decoded::Pending => {}
                    _ => p.print(char::REPLACEMENT_CHARACTER),
                },
            },

            State::Escape => match b {
                0x00..=0x1f => p.execute(b),
                0x20..=0x2f => {
                    self.collect(b);
                    self.state = State::EscapeIntermediate;
                }
                b'[' => self.state = State::CsiEntry,
                b']' => self.osc_start(),
                b'P' => self.state = State::DcsEntry,
                b'X' | b'^' | b'_' => self.state = State::SosPmApcString,
                0x30..=0x7e => self.esc_dispatch(p, b),
                _ => {}
            },

            State::EscapeIntermediate => match b {
                0x00..=0x1f => p.execute(b),
                0x20..=0x2f => self.collect(b),
                0x30..=0x7e => self.esc_dispatch(p, b),
                _ => {}
            },

            State::CsiEntry => match b {
                0x00..=0x1f => p.execute(b),
                0x20..=0x2f => {
                    self.collect(b);
                    self.state = State::CsiIntermediate;
                }
                b'0'..=b'9' | b':' | b';' => {
                    self.param_byte(b);
                    self.state = State::CsiParam;
                }
                // Private marker.
                b'<'..=b'?' => {
                    self.collect(b);
                    self.state = State::CsiParam;
                }
                0x40..=0x7e => self.csi_dispatch(p, b),
                _ => {}
            },

            State::CsiParam => match b {
                0x00..=0x1f => p.execute(b),
                b'0'..=b'9' | b':' | b';' => self.param_byte(b),
                b'<'..=b'?' => self.state = State::CsiIgnore,
                0x20..=0x2f => {
                    self.collect(b);
                    self.state = State::CsiIntermediate;
                }
                0x40..=0x7e => self.csi_dispatch(p, b),
                _ => {}
            },

            State::CsiIntermediate => match b {
                0x00..=0x1f => p.execute(b),
                0x20..=0x2f => self.collect(b),
                0x30..=0x3f => self.state = State::CsiIgnore,
                0x40..=0x7e => self.csi_dispatch(p, b),
                _ => {}
            },

            State::CsiIgnore => match b {
                0x00..=0x1f => p.execute(b),
                0x40..=0x7e => self.state = State::Ground,
                _ => {}
            },

            State::DcsEntry => match b {
                0x20..=0x2f => {
                    self.collect(b);
                    self.state = State::DcsIntermediate;
                }
                b'0'..=b'9' | b':' | b';' => {
                    self.param_byte(b);
                    self.state = State::DcsParam;
                }
                b'<'..=b'?' => {
                    self.collect(b);
                    self.state = State::DcsParam;
                }
                0x40..=0x7e => self.hook(p, b),
                _ => {}
            },

            State::DcsParam => match b {
                b'0'..=b'9' | b':' | b';' => self.param_byte(b),
                b'<'..=b'?' => self.state = State::DcsIgnore,
                0x20..=0x2f => {
                    self.collect(b);
                    self.state = State::DcsIntermediate;
                }
                0x40..=0x7e => self.hook(p, b),
                _ => {}
            },

            State::DcsIntermediate => match b {
                0x20..=0x2f => self.collect(b),
                0x30..=0x3f => self.state = State::DcsIgnore,
                0x40..=0x7e => self.hook(p, b),
                _ => {}
            },

            State::DcsPassthrough => match b {
                DEL => {}
                _ => p.put(b),
            },

            State::OscString => match b {
                BEL => {
                    self.osc_dispatch(p, true);
                    self.state = State::Ground;
                }
                0x00..=0x1f | DEL => {}
                _ => self.osc_put(b),
            },

            State::DcsIgnore | State::SosPmApcString => {}
        }
    }

    /// Leave the current state because of ESC: strings end here.
    fn exit<P: Perform>(&mut self, p: &mut P) {
        match self.state {
            State::OscString => self.osc_dispatch(p, false),
            State::DcsPassthrough => p.unhook(),
            _ => {}
        }
    }

    fn clear(&mut self) {
        self.params.clear();
        self.param = 0;
        self.has_params = false;
        self.intermediate_len = 0;
        self.ignoring = false;
    }

    fn collect(&mut self, b: u8) {
        if self.intermediate_len == MAX_INTERMEDIATES {
            self.ignoring = true;
        } else {
            self.intermediates[self.intermediate_len] = b;
            self.intermediate_len += 1;
        }
    }

    fn param_byte(&mut self, b: u8) {
        self.has_params = true;
        match b {
            b';' | b':' => {
                self.params.push(self.param, b == b':');
                self.param = 0;
            }
            _ => {
                self.param = self
                    .param
                    .saturating_mul(10)
                    .saturating_add(u16::from(b - b'0'));
            }
        }
    }

    /// The parameters with the one in progress added.
    fn finish_params(&mut self) {
        if self.has_params {
            self.params.push(self.param, false);
        }
    }

    fn esc_dispatch<P: Perform>(&mut self, p: &mut P, b: u8) {
        p.esc_dispatch(
            &self.intermediates[..self.intermediate_len],
            self.ignoring,
            b,
        );
        self.state = State::Ground;
    }

    fn csi_dispatch<P: Perform>(&mut self, p: &mut P, b: u8) {
        self.finish_params();
        p.csi_dispatch(
            &self.params,
            &self.intermediates[..self.intermediate_len],
            self.ignoring,
            b,
        );
        self.state = State::Ground;
    }

    fn hook<P: Perform>(&mut self, p: &mut P, b: u8) {
        self.finish_params();
        p.hook(
            &self.params,
            &self.intermediates[..self.intermediate_len],
            self.ignoring,
            b,
        );
        self.state = State::DcsPassthrough;
    }

    fn osc_start(&mut self) {
        self.osc.clear();
        self.osc_params = 0;
        self.osc_overflow = false;
        self.state = State::OscString;
    }

    fn osc_put(&mut self, b: u8) {
        if b == b';' && self.osc_params < MAX_OSC_PARAMS - 1 {
            self.osc_ends[self.osc_params] = self.osc.len();
            self.osc_params += 1;
        } else if self.osc.len() < MAX_OSC {
            self.osc.push(b);
        } else {
            self.osc_overflow = true;
        }
    }

    fn osc_dispatch<P: Perform>(&mut self, p: &mut P, bell_terminated: bool) {
        if self.osc_overflow {
            tracing::debug!(target: "vellum::term", "OSC string over {MAX_OSC} bytes dropped");
            return;
        }
        let mut params: [&[u8]; MAX_OSC_PARAMS] = [&[]; MAX_OSC_PARAMS];
        let mut start = 0;
        for (i, &end) in self.osc_ends[..self.osc_params].iter().enumerate() {
            params[i] = &self.osc[start..end];
            start = end;
        }
        params[self.osc_params] = &self.osc[start..];
        p.osc_dispatch(&params[..=self.osc_params], bell_terminated);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Action {
        /// Consecutive printed characters.
        Text(String),
        Execute(u8),
        Csi {
            params: Vec<Vec<u16>>,
            inter: String,
            ignored: bool,
            action: char,
        },
        Esc {
            inter: String,
            ignored: bool,
            byte: char,
        },
        Osc {
            params: Vec<String>,
            bell: bool,
        },
        Hook {
            params: Vec<Vec<u16>>,
            inter: String,
            action: char,
        },
        Put(u8),
        Unhook,
    }
    use Action::*;

    #[derive(Default)]
    struct Recorder(Vec<Action>);

    fn groups(params: &Params) -> Vec<Vec<u16>> {
        params.iter().map(<[u16]>::to_vec).collect()
    }

    fn string(bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).into_owned()
    }

    impl Perform for Recorder {
        fn print(&mut self, ch: char) {
            if let Some(Text(s)) = self.0.last_mut() {
                s.push(ch);
            } else {
                self.0.push(Text(ch.into()));
            }
        }
        fn execute(&mut self, byte: u8) {
            self.0.push(Execute(byte));
        }
        fn csi_dispatch(&mut self, params: &Params, inter: &[u8], ignored: bool, action: u8) {
            self.0.push(Csi {
                params: groups(params),
                inter: string(inter),
                ignored,
                action: action.into(),
            });
        }
        fn esc_dispatch(&mut self, inter: &[u8], ignored: bool, byte: u8) {
            self.0.push(Esc {
                inter: string(inter),
                ignored,
                byte: byte.into(),
            });
        }
        fn osc_dispatch(&mut self, params: &[&[u8]], bell: bool) {
            self.0.push(Osc {
                params: params.iter().map(|p| string(p)).collect(),
                bell,
            });
        }
        fn hook(&mut self, params: &Params, inter: &[u8], _ignored: bool, action: u8) {
            self.0.push(Hook {
                params: groups(params),
                inter: string(inter),
                action: action.into(),
            });
        }
        fn put(&mut self, byte: u8) {
            self.0.push(Put(byte));
        }
        fn unhook(&mut self) {
            self.0.push(Unhook);
        }
    }

    fn parse(bytes: &[u8]) -> Vec<Action> {
        let mut r = Recorder::default();
        Parser::new().advance(&mut r, bytes);
        r.0
    }

    fn text(s: &str) -> Action {
        Text(s.into())
    }

    fn csi(params: &[&[u16]], inter: &str, action: char) -> Action {
        Csi {
            params: params.iter().map(|g| g.to_vec()).collect(),
            inter: inter.into(),
            ignored: false,
            action,
        }
    }

    fn esc(inter: &str, byte: char) -> Action {
        Esc {
            inter: inter.into(),
            ignored: false,
            byte,
        }
    }

    fn osc(params: &[&str], bell: bool) -> Action {
        Osc {
            params: params.iter().map(|&p| p.into()).collect(),
            bell,
        }
    }

    #[test]
    fn text_and_controls() {
        assert_eq!(
            parse(b"hi\r\n\x07x"),
            [
                text("hi"),
                Execute(b'\r'),
                Execute(b'\n'),
                Execute(0x07),
                text("x")
            ]
        );
        assert_eq!(parse(b"a\x7fb"), [text("ab")], "DEL is ignored");
    }

    #[test]
    fn csi_parameters() {
        assert_eq!(parse(b"\x1b[1;31m"), [csi(&[&[1], &[31]], "", 'm')]);
        assert_eq!(parse(b"\x1b[m"), [csi(&[], "", 'm')]);
        assert_eq!(parse(b"\x1b[;5H"), [csi(&[&[0], &[5]], "", 'H')]);
        assert_eq!(parse(b"\x1b[5;H"), [csi(&[&[5], &[0]], "", 'H')]);
        assert_eq!(parse(b"\x1b[0042A"), [csi(&[&[42]], "", 'A')]);
        assert_eq!(parse(b"\x1b[99999m"), [csi(&[&[65535]], "", 'm')]);
    }

    #[test]
    fn csi_private_markers_and_intermediates() {
        assert_eq!(parse(b"\x1b[?25h"), [csi(&[&[25]], "?", 'h')]);
        assert_eq!(parse(b"\x1b[>c"), [csi(&[], ">", 'c')]);
        assert_eq!(parse(b"\x1b[2 q"), [csi(&[&[2]], " ", 'q')]);
        assert_eq!(parse(b"\x1b[!p"), [csi(&[], "!", 'p')]);
        assert_eq!(parse(b"\x1b[?1$p"), [csi(&[&[1]], "?$", 'p')]);
        assert_eq!(
            parse(b"\x1b[1 !\"m"),
            [Csi {
                params: vec![vec![1]],
                inter: " !".into(),
                ignored: true,
                action: 'm'
            }]
        );
    }

    #[test]
    fn colon_subparameters_group() {
        assert_eq!(
            parse(b"\x1b[38:2::10:20:30m"),
            [csi(&[&[38, 2, 0, 10, 20, 30]], "", 'm')]
        );
        assert_eq!(
            parse(b"\x1b[4:3;38;5;1m"),
            [csi(&[&[4, 3], &[38], &[5], &[1]], "", 'm')]
        );
        assert_eq!(parse(b"\x1b[1:m"), [csi(&[&[1, 0]], "", 'm')]);
    }

    #[test]
    fn too_many_parameters_are_dropped() {
        let seq = format!("\x1b[{}m", vec!["7"; 40].join(";"));
        let [Csi { params, .. }] = &parse(seq.as_bytes())[..] else {
            panic!()
        };
        assert_eq!(params.len(), MAX_PARAMS);
    }

    #[test]
    fn malformed_csi_is_ignored_to_its_end() {
        // A private marker after parameters, a parameter after an
        // intermediate: consumed up to the final byte, not dispatched.
        assert_eq!(parse(b"\x1b[1?mok"), [text("ok")]);
        assert_eq!(parse(b"\x1b[1 2mok"), [text("ok")]);
        // High bytes inside a sequence are ignored.
        assert_eq!(parse(b"\x1b[1\xc3m"), [csi(&[&[1]], "", 'm')]);
    }

    #[test]
    fn controls_inside_a_sequence_execute() {
        assert_eq!(
            parse(b"\x1b[1\n2m"),
            [Execute(b'\n'), csi(&[&[12]], "", 'm')]
        );
        assert_eq!(parse(b"\x1b\r7"), [Execute(b'\r'), esc("", '7')]);
    }

    #[test]
    fn escape_sequences() {
        assert_eq!(parse(b"\x1b7"), [esc("", '7')]);
        assert_eq!(parse(b"\x1b(0"), [esc("(", '0')]);
        assert_eq!(parse(b"\x1b#8"), [esc("#", '8')]);
        assert_eq!(parse(b"\x1bc"), [esc("", 'c')]);
    }

    #[test]
    fn can_and_sub_abort_esc_restarts() {
        assert_eq!(parse(b"\x1b[1\x18m"), [Execute(CAN), text("m")]);
        assert_eq!(parse(b"\x1b(\x1a0"), [Execute(SUB), text("0")]);
        assert_eq!(parse(b"\x1b[1\x1b[2m"), [csi(&[&[2]], "", 'm')]);
        assert_eq!(parse(b"\x1b]0;x\x18y"), [Execute(CAN), text("y")]);
    }

    #[test]
    fn osc_strings() {
        assert_eq!(parse(b"\x1b]0;title\x07"), [osc(&["0", "title"], true)]);
        assert_eq!(
            parse(b"\x1b]2;a;b\x1b\\"),
            [osc(&["2", "a", "b"], false), esc("", '\\')]
        );
        assert_eq!(parse("\x1b]2;é\x07".as_bytes()), [osc(&["2", "é"], true)]);
        assert_eq!(parse(b"\x1b]\x07"), [osc(&[""], true)]);
        // Controls other than BEL are dropped from the string.
        assert_eq!(parse(b"\x1b]2;a\nb\x07"), [osc(&["2", "ab"], true)]);
    }

    #[test]
    fn osc_parameter_count_and_length_are_limited() {
        let many = format!("\x1b]{}\x07", vec!["x"; 20].join(";"));
        let [Osc { params, .. }] = &parse(many.as_bytes())[..] else {
            panic!()
        };
        assert_eq!(params.len(), MAX_OSC_PARAMS);
        assert_eq!(params.last().unwrap(), "x;x;x;x;x");

        let long = format!("\x1b]2;{}\x07after", "y".repeat(MAX_OSC + 1));
        assert_eq!(parse(long.as_bytes()), [text("after")]);
    }

    #[test]
    fn dcs_strings() {
        assert_eq!(
            parse(b"\x1bP1$qm\x1b\\"),
            [
                Hook {
                    params: vec![vec![1]],
                    inter: "$".into(),
                    action: 'q'
                },
                Put(b'm'),
                Unhook,
                esc("", '\\')
            ]
        );
        assert_eq!(
            parse(b"\x1bPq#\x18x"),
            [
                Hook {
                    params: vec![],
                    inter: "".into(),
                    action: 'q'
                },
                Put(b'#'),
                Unhook,
                Execute(CAN),
                text("x")
            ]
        );
        // Malformed header: ignored to the end of the string.
        assert_eq!(parse(b"\x1bP1?q data\x1b\\x"), [esc("", '\\'), text("x")]);
    }

    #[test]
    fn sos_pm_apc_are_ignored() {
        assert_eq!(
            parse(b"\x1b_Gf=24;AAAA\x1b\\after"),
            [esc("", '\\'), text("after")]
        );
        assert_eq!(parse(b"\x1b^pm\x1b\\"), [esc("", '\\')]);
        assert_eq!(parse(b"\x1bXsos\x1b\\"), [esc("", '\\')]);
    }

    #[test]
    fn utf8_matches_lossy_decoding() {
        let cases: &[&[u8]] = &[
            "héllo €😀".as_bytes(),
            b"a\xffb\xc3(",
            b"\xed\xa0\x80x",     // surrogate
            b"\xe0\x80\xafx",     // overlong
            b"\xf4\x90\x80\x80x", // past U+10FFFF
            b"\xf0\x9f\x98x",     // truncated 4-byte
            b"\x80\xbf\xc0\xc1x", // stray continuations, invalid leads
            b"\xe2\x82\xac\xe2\x82",
        ];
        for &case in cases {
            let mut expected = String::from_utf8_lossy(case).into_owned();
            // A character cut off at the very end is still pending.
            if case.ends_with(b"\xe2\x82") {
                expected.truncate(expected.len() - '\u{FFFD}'.len_utf8());
            }
            assert_eq!(parse(case), [Text(expected)], "{}", case.escape_ascii());
        }
    }

    #[test]
    fn controls_end_an_incomplete_character() {
        assert_eq!(
            parse(b"\xc3\x1b[m\xe2\x82\n"),
            [
                text("\u{FFFD}"),
                csi(&[], "", 'm'),
                text("\u{FFFD}"),
                Execute(b'\n')
            ]
        );
    }

    /// A bit of everything, including invalid UTF-8.
    pub(crate) fn corpus() -> Vec<u8> {
        let mut c = "plain é€😀\r\n\x1b[1;38:2::1:2:3mbold\x1b[0m\x1b[?1049h\
            \x1b]0;title ✓\x07\x1b]8;;http://x\x1b\\link\x1b]8;;\x1b\\\
            \x1bP1$qm\x1b\\\x1b(0lqk\x1b(B\x1b[2 q\x1b_apc\x1b\\"
            .as_bytes()
            .to_vec();
        c.extend_from_slice(b"\xff\xc3(\x1b[1\x18x\tend");
        c
    }

    #[test]
    fn splitting_the_input_changes_nothing() {
        let corpus = corpus();
        let whole = parse(&corpus);
        assert!(whole.len() > 20);
        for split in 0..=corpus.len() {
            let mut r = Recorder::default();
            let mut parser = Parser::new();
            parser.advance(&mut r, &corpus[..split]);
            parser.advance(&mut r, &corpus[split..]);
            assert_eq!(r.0, whole, "split at {split}");
        }
        let mut r = Recorder::default();
        let mut parser = Parser::new();
        for b in &corpus {
            parser.advance(&mut r, std::slice::from_ref(b));
        }
        assert_eq!(r.0, whole, "byte at a time");
    }
}
