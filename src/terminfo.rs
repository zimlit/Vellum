//! vellum's terminfo entry, `xterm-vellum` (`terminfo/xterm-vellum.ti`).
//!
//! build.rs compiles it with `tic`, and the compiled entry is embedded here.
//! The session server installs it under `$XDG_DATA_HOME/vellum/terminfo`
//! and points `TERMINFO` there for the shell; ncurses still searches the
//! system directories after it, so other `TERM`s keep working inside.
//!
//! Hosts reached over SSH don't have the entry. Copy it once with
//! `infocmp -x xterm-vellum | ssh HOST tic -x -`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The entry's name, and `$TERM` when it's installed.
pub const NAME: &str = "xterm-vellum";

/// `$TERM` when it couldn't be installed: the closest entry everyone has.
pub const FALLBACK: &str = "xterm-256color";

/// The compiled entry.
pub static COMPILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/terminfo/x/xterm-vellum"));

/// Where the entry goes: `$XDG_DATA_HOME/vellum/terminfo`, by default
/// `~/.local/share/vellum/terminfo`.
pub fn default_dir() -> Option<PathBuf> {
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|d| Path::new(d).is_absolute())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .map(|h| Path::new(&h).join(".local/share"))
        })?;
    Some(data.join("vellum/terminfo"))
}

/// Make sure `dir` (a terminfo directory) holds this build's entry,
/// writing it if it's missing or different. The write goes through a
/// temporary file and a rename, so a program reading the entry never sees
/// half of it.
pub fn install(dir: &Path) -> io::Result<()> {
    // Linux ncurses keeps entries under their first letter.
    let path = dir.join("x").join(NAME);
    if fs::read(&path).is_ok_and(|have| have == COMPILED) {
        return Ok(());
    }
    let parent = path.parent().expect("entry path has a parent");
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".{NAME}.{}", std::process::id()));
    fs::write(&tmp, COMPILED)?;
    fs::rename(&tmp, &path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })?;
    tracing::info!(path = %path.display(), "installed terminfo entry");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::process::Command;

    use xkbcommon::xkb::Keysym;

    use super::*;
    use crate::keys::{self, KeyEvent, KeyModes, Mods};
    use crate::term::Terminal;
    use crate::testutil::TempDir;

    /// The entry installed in a temporary directory.
    struct Entry(TempDir);

    impl Entry {
        fn new() -> Self {
            let dir = TempDir::new();
            install(dir.path()).unwrap();
            Self(dir)
        }

        /// `tput` with this entry: the capability expanded with `params`.
        fn tput(&self, args: &[&str]) -> Vec<u8> {
            let out = Command::new("tput")
                .env("TERMINFO", self.0.path())
                .arg("-T")
                .arg(NAME)
                .args(args)
                .output()
                .unwrap();
            assert!(out.status.success(), "tput {args:?}: {out:?}");
            out.stdout
        }

        /// Every capability as infocmp lists it: flags with no value,
        /// numbers in decimal, strings as written (`\E` turned into ESC).
        fn capabilities(&self) -> Vec<(String, Option<String>)> {
            let out = Command::new("infocmp")
                .args(["-x", "-1", "-A"])
                .arg(self.0.path())
                .arg(NAME)
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            // One per line: `\tname=value,`, `\tname#number,` or `\tflag,`.
            String::from_utf8(out.stdout)
                .unwrap()
                .lines()
                .filter_map(|l| l.strip_prefix('\t')?.strip_suffix(','))
                .map(|c| {
                    if let Some((name, value)) = c.split_once('=') {
                        (name.to_owned(), Some(value.replace("\\E", "\x1b")))
                    } else if let Some((name, n)) = c.split_once('#') {
                        // Numbers may be in hex (`colors#0x100`).
                        let n = match n.strip_prefix("0x") {
                            Some(hex) => u32::from_str_radix(hex, 16).unwrap(),
                            None => n.parse().unwrap(),
                        };
                        (name.to_owned(), Some(n.to_string()))
                    } else {
                        (c.to_owned(), None)
                    }
                })
                .collect()
        }

        /// The names of the string capabilities.
        fn strings(&self) -> BTreeSet<String> {
            self.capabilities()
                .into_iter()
                .filter(|(_, v)| v.as_ref().is_some_and(|v| v.parse::<u32>().is_err()))
                .map(|(n, _)| n)
                .collect()
        }
    }

    /// What `bytes` leaves a fresh terminal with: its replies, and the
    /// sequences it didn't handle.
    fn feed(term: &mut Terminal, bytes: &[u8]) -> (Vec<u8>, Vec<String>) {
        term.feed(bytes);
        let unhandled = term.unhandled_sequences().iter().cloned().collect();
        (term.take_replies(), unhandled)
    }

    #[test]
    fn fits_narrow_ncurses() {
        // build.rs checks this too; see `MAX_TERMINFO` there.
        assert!(COMPILED.len() <= 4096, "{} bytes", COMPILED.len());
    }

    #[test]
    fn installs_once_and_repairs() {
        let dir = TempDir::new();
        let path = dir.path().join("x").join(NAME);
        install(dir.path()).unwrap();
        assert_eq!(fs::read(&path).unwrap(), COMPILED);
        install(dir.path()).unwrap();
        fs::write(&path, b"stale").unwrap();
        install(dir.path()).unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            COMPILED,
            "a stale entry is replaced"
        );
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, [NAME], "no temporary file left behind");
    }

    #[test]
    fn rgb_colors_are_always_direct() {
        // What Emacs does with setb24: black must not become palette 0.
        let e = Entry::new();
        assert_eq!(e.tput(&["setb24", "0"]), b"\x1b[48;2;0;0;0m");
        assert_eq!(
            e.tput(&["setf24", &0x12_34_56.to_string()]),
            b"\x1b[38;2;18;52;86m"
        );
        // `RGB` would make ncurses read setaf as direct color.
        assert!(!e.capabilities().iter().any(|(n, _)| n == "RGB"));
        assert_eq!(e.tput(&["colors"]), b"256\n");
    }

    /// Capabilities the terminal answers or sends rather than receives:
    /// the shapes of replies, and the keys.
    fn sent_by_vellum(name: &str) -> bool {
        matches!(name, "u6" | "u8" | "rv" | "xr" | "PS" | "PE" | "acsc") || name.starts_with('k')
    }

    /// Every capability a program sends, with parameters that reach each
    /// branch of its format.
    const SENT_TO_VELLUM: &[(&str, &[&str])] = &[
        ("cr", &[]),
        ("cub1", &[]),
        ("cud1", &[]),
        ("cuf1", &[]),
        ("cuu1", &[]),
        ("cub", &["3"]),
        ("cud", &["3"]),
        ("cuf", &["3"]),
        ("cuu", &["3"]),
        ("cup", &["5", "10"]),
        ("home", &[]),
        ("hpa", &["4"]),
        ("vpa", &["4"]),
        ("sc", &[]),
        ("rc", &[]),
        ("nel", &[]),
        ("ind", &[]),
        ("ri", &[]),
        ("indn", &["2"]),
        ("rin", &["2"]),
        ("csr", &["2", "20"]),
        ("ht", &[]),
        ("cbt", &[]),
        ("hts", &[]),
        ("tbc", &[]),
        ("bel", &[]),
        ("clear", &[]),
        ("ed", &[]),
        ("el", &[]),
        ("el1", &[]),
        ("E3", &[]),
        ("dch1", &[]),
        ("dch", &["2"]),
        ("ich", &["2"]),
        ("ech", &["2"]),
        ("dl1", &[]),
        ("dl", &["2"]),
        ("il1", &[]),
        ("il", &["2"]),
        ("smir", &[]),
        ("rmir", &[]),
        ("rep", &["120", "5"]),
        ("smam", &[]),
        ("rmam", &[]),
        ("smcup", &[]),
        ("rmcup", &[]),
        ("smkx", &[]),
        ("rmkx", &[]),
        ("civis", &[]),
        ("cnorm", &[]),
        ("cvvis", &[]),
        ("is2", &[]),
        ("rs1", &[]),
        ("rs2", &[]),
        ("sgr0", &[]),
        ("sgr", &["1", "1", "1", "1", "1", "1", "1", "0", "1"]),
        ("sgr", &["0", "0", "0", "0", "0", "0", "0", "0", "0"]),
        ("bold", &[]),
        ("dim", &[]),
        ("blink", &[]),
        ("invis", &[]),
        ("rev", &[]),
        ("sitm", &[]),
        ("ritm", &[]),
        ("smso", &[]),
        ("rmso", &[]),
        ("smul", &[]),
        ("rmul", &[]),
        ("smxx", &[]),
        ("rmxx", &[]),
        ("smacs", &[]),
        ("rmacs", &[]),
        ("setaf", &["1"]),
        ("setaf", &["9"]),
        ("setaf", &["200"]),
        ("setab", &["1"]),
        ("setab", &["9"]),
        ("setab", &["200"]),
        ("op", &[]),
        ("initc", &["1", "500", "250", "1000"]),
        ("oc", &[]),
        ("setf24", &["1193046"]),
        ("setb24", &["0"]),
        ("Smulx", &["3"]),
        ("Setulc", &["16744448"]),
        ("Ss", &["5"]),
        ("Se", &[]),
        ("Cs", &["#ff8000"]),
        ("Cr", &[]),
        ("BE", &[]),
        ("BD", &[]),
        ("fe", &[]),
        ("fd", &[]),
        ("Sync", &["1"]),
        ("Sync", &["2"]),
        ("u7", &[]),
        ("u9", &[]),
        ("RV", &[]),
        ("XR", &[]),
    ];

    #[test]
    fn vellum_handles_every_sequence() {
        let e = Entry::new();
        let listed: BTreeSet<String> = SENT_TO_VELLUM.iter().map(|(n, _)| n.to_string()).collect();
        let entry: BTreeSet<String> = e
            .strings()
            .into_iter()
            .filter(|n| !sent_by_vellum(n))
            .collect();
        assert_eq!(listed, entry, "every capability is tested, and only those");

        for (name, params) in SENT_TO_VELLUM {
            let mut args = vec![*name];
            args.extend_from_slice(params);
            let bytes = e.tput(&args);
            let mut term = Terminal::new(80, 24);
            let (_, unhandled) = feed(&mut term, &bytes);
            assert!(
                unhandled.is_empty(),
                "{name} {params:?} = {}: {unhandled:?}",
                bytes.escape_ascii()
            );
        }
    }

    #[test]
    fn replies_have_the_promised_shapes() {
        let e = Entry::new();
        let mut term = Terminal::new(80, 24);
        let reply = |term: &mut Terminal, cap: &str| {
            String::from_utf8(feed(term, &e.tput(&[cap])).0).unwrap()
        };
        // u6 is `\E[%i%d;%dR`: CPR, 1-based.
        assert_eq!(reply(&mut term, "u7"), "\x1b[1;1R");
        // u8 is `\E[?%[;0123456789]c`.
        let da1 = reply(&mut term, "u9");
        let inner = da1
            .strip_prefix("\x1b[?")
            .unwrap()
            .strip_suffix('c')
            .unwrap();
        assert!(
            inner.chars().all(|c| c == ';' || c.is_ascii_digit()),
            "{da1:?}"
        );
        // rv is `\E\[>1;[0-9]+;0c`.
        let da2 = reply(&mut term, "RV");
        let version = da2
            .strip_prefix("\x1b[>1;")
            .unwrap()
            .strip_suffix(";0c")
            .unwrap();
        assert!(version.chars().all(|c| c.is_ascii_digit()), "{da2:?}");
        // xr is `\EP>\|vellum\(([0-9.]+)\)\E\\`.
        let xt = reply(&mut term, "XR");
        let version = xt
            .strip_prefix("\x1bP>|vellum(")
            .unwrap()
            .strip_suffix(")\x1b\\")
            .unwrap();
        assert!(
            version.chars().all(|c| c == '.' || c.is_ascii_digit()),
            "{xt:?}"
        );
        // kxIN/kxOUT are the focus events fe turns on.
        feed(&mut term, &e.tput(&["fe"]));
        term.focus(true);
        assert_eq!(term.take_replies(), e.tput(&["kxIN"]));
        term.focus(false);
        assert_eq!(term.take_replies(), e.tput(&["kxOUT"]));
    }

    #[test]
    fn line_drawing_covers_acsc() {
        let e = Entry::new();
        let acsc = e.tput(&["acsc"]);
        let mut term = Terminal::new(80, 24);
        // Pairs of (what the program means, what it sends under smacs).
        let sent: Vec<u8> = acsc.chunks(2).map(|p| p[1]).collect();
        let mut bytes = e.tput(&["smacs"]);
        bytes.extend_from_slice(&sent);
        feed(&mut term, &bytes);
        for (col, &b) in sent.iter().enumerate() {
            let ch = term.cell(col as u16, 0).ch;
            assert_ne!(
                ch,
                char::from(b),
                "{:?} isn't a line-drawing character",
                char::from(b)
            );
        }
    }

    /// The key each key capability names, and its modifiers.
    fn keys() -> Vec<(String, Keysym, Mods)> {
        const S: Mods = Mods::SHIFT;
        let none = Mods::empty();
        let fkey = |n: u32| Keysym::new(Keysym::F1.raw() + n - 1);
        let mut keys: Vec<(String, Keysym, Mods)> = [
            ("kbs", Keysym::BackSpace, none),
            ("kcbt", Keysym::ISO_Left_Tab, S),
            ("kent", Keysym::KP_Enter, none),
            ("kcuu1", Keysym::Up, none),
            ("kcud1", Keysym::Down, none),
            ("kcuf1", Keysym::Right, none),
            ("kcub1", Keysym::Left, none),
            ("khome", Keysym::Home, none),
            ("kend", Keysym::End, none),
            ("kbeg", Keysym::KP_Begin, none),
            ("kich1", Keysym::Insert, none),
            ("kdch1", Keysym::Delete, none),
            ("kpp", Keysym::Page_Up, none),
            ("knp", Keysym::Page_Down, none),
            ("kri", Keysym::Up, S),
            ("kind", Keysym::Down, S),
            ("kUP", Keysym::Up, S),
            ("kDN", Keysym::Down, S),
            ("kRIT", Keysym::Right, S),
            ("kLFT", Keysym::Left, S),
            ("kHOM", Keysym::Home, S),
            ("kEND", Keysym::End, S),
            ("kIC", Keysym::Insert, S),
            ("kDC", Keysym::Delete, S),
            ("kPRV", Keysym::Page_Up, S),
            ("kNXT", Keysym::Page_Down, S),
            ("ka1", Keysym::KP_7, none),
            ("ka2", Keysym::KP_8, none),
            ("ka3", Keysym::KP_9, none),
            ("kb1", Keysym::KP_4, none),
            ("kb2", Keysym::KP_5, none),
            ("kb3", Keysym::KP_6, none),
            ("kc1", Keysym::KP_1, none),
            ("kc2", Keysym::KP_2, none),
            ("kc3", Keysym::KP_3, none),
            ("kp5", Keysym::KP_Begin, none),
            ("kpADD", Keysym::KP_Add, none),
            ("kpSUB", Keysym::KP_Subtract, none),
            ("kpMUL", Keysym::KP_Multiply, none),
            ("kpDIV", Keysym::KP_Divide, none),
            ("kpDOT", Keysym::KP_Decimal, none),
            ("kpCMA", Keysym::KP_Separator, none),
            ("kpZRO", Keysym::KP_0, none),
        ]
        .into_iter()
        .map(|(n, k, m)| (n.to_owned(), k, m))
        .collect();
        // F1-F63: F1-F12 with none, Shift, Ctrl, Ctrl+Shift, Alt, Alt+Shift.
        let groups = [
            none,
            S,
            Mods::CTRL,
            Mods::CTRL | S,
            Mods::ALT,
            Mods::ALT | S,
        ];
        for n in 1..=63 {
            keys.push((
                format!("kf{n}"),
                fkey((n - 1) % 12 + 1),
                groups[(n as usize - 1) / 12],
            ));
        }
        // Editing and cursor keys with xterm's modifier numbers.
        let mods = [
            (3, Mods::ALT),
            (4, Mods::ALT | S),
            (5, Mods::CTRL),
            (6, Mods::CTRL | S),
            (7, Mods::CTRL | Mods::ALT),
        ];
        for (name, sym) in [
            ("kUP", Keysym::Up),
            ("kDN", Keysym::Down),
            ("kRIT", Keysym::Right),
            ("kLFT", Keysym::Left),
            ("kHOM", Keysym::Home),
            ("kEND", Keysym::End),
            ("kIC", Keysym::Insert),
            ("kDC", Keysym::Delete),
            ("kPRV", Keysym::Page_Up),
            ("kNXT", Keysym::Page_Down),
        ] {
            for (n, m) in mods {
                keys.push((format!("{name}{n}"), sym, m));
            }
        }
        keys
    }

    #[test]
    fn keys_are_what_vellum_sends() {
        let e = Entry::new();
        let keys = keys();
        let listed: BTreeSet<String> = keys.iter().map(|(n, ..)| n.clone()).collect();
        let entry: BTreeSet<String> = e
            .strings()
            .into_iter()
            .filter(|n| n.starts_with('k') && !matches!(n.as_str(), "kxIN" | "kxOUT"))
            .collect();
        assert_eq!(listed, entry, "every key is tested, and only those");

        // Programs reading these keys have sent smkx first.
        let mut term = Terminal::new(80, 24);
        feed(&mut term, &e.tput(&["smkx"]));
        let modes = KeyModes::of(&term);
        for (name, sym, mods) in keys {
            let sent = keys::encode(&KeyEvent::press(sym, mods), &modes).unwrap_or_default();
            assert_eq!(
                sent.escape_ascii().to_string(),
                e.tput(&[&name]).escape_ascii().to_string(),
                "{name}: {sym:?} with {mods:?}"
            );
        }
    }

    #[test]
    fn xtgettcap_agrees_with_the_entry() {
        let caps = Entry::new().capabilities();
        for &(name, value) in crate::term::reply::CAPABILITIES {
            // Names, the RGB flag (left out on purpose), and Co (termcap's
            // name for `colors`, which infocmp prints).
            if matches!(name, "TN" | "name" | "RGB" | "Co") {
                continue;
            }
            let want = (name.to_owned(), value.map(str::to_owned));
            assert!(caps.contains(&want), "{want:?} not in the entry");
        }
    }
}
