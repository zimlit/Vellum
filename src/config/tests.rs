//! The config end to end: Lua source in, [`Settings`] or an error message
//! out, and the script thread around it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rustix::event::{PollFd, PollFlags, Timespec, poll};

use super::settings::{Color, CursorStyle};
use super::*;
use crate::cli::Args;
use crate::input::RuleNames;
use crate::term::palette::{Palette, default_color};
use crate::testutil::TempDir;

/// Write `init.lua` (and any other files) into a temp dir.
fn config_dir(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
    let dir = TempDir::new();
    for (name, text) in files {
        let path = dir.path().join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    let init = dir.path().join("init.lua");
    (dir, init)
}

fn load(source: &str) -> Result<Settings, String> {
    let (_dir, init) = config_dir(&[("init.lua", source)]);
    api::load(Some(&init)).map(|l| l.settings)
}

fn load_err(source: &str) -> String {
    load(source).expect_err("the config should fail")
}

#[test]
fn no_config_is_the_defaults() {
    assert_eq!(api::load(None).unwrap().settings, Settings::default());
    assert_eq!(load("").unwrap(), Settings::default());
}

#[test]
fn every_setting() {
    let s = load(
        r##"
        vellum.set{
            font = "Iosevka:size=14",
            keymap = { layout = "us,de", options = "ctrl:nocaps" },
            colors = {
                foreground = "#ffffff", background = "#000", cursor = "rgb:ff/00/00",
                palette = { [0] = "#010101", "#020202", [208] = "#ff8700" },
            },
            repeat_delay = 250,
            repeat_rate = 50,
            scrollback = 100,
            cursor_style = "bar",
            shell = { "/bin/zsh", "-l" },
            env = { EDITOR = "emacs" },
            script_timeout = -1,
        }
        "##,
    )
    .unwrap();
    assert_eq!(s.font, "Iosevka:size=14");
    assert_eq!(
        s.keymap,
        RuleNames {
            layout: "us,de".into(),
            options: Some("ctrl:nocaps".into()),
            ..RuleNames::default()
        }
    );
    let p = s.colors.palette();
    assert_eq!(p.foreground, [255, 255, 255]);
    assert_eq!(p.background, [0, 0, 0]);
    assert_eq!(p.cursor, [255, 0, 0]);
    assert_eq!(p.colors[0], [1, 1, 1]);
    assert_eq!(p.colors[1], [2, 2, 2], "listed after [0]");
    assert_eq!(p.colors[2], default_color(2), "unset: built in");
    assert_eq!(p.colors[208], [0xff, 0x87, 0]);
    assert_eq!(s.repeat_delay(), Duration::from_millis(250));
    assert_eq!(s.repeat_interval(), Duration::from_millis(20));
    assert_eq!(s.scrollback, 100);
    assert_eq!(s.cursor_style, CursorStyle::Bar);
    assert_eq!(s.shell, ["/bin/zsh", "-l"]);
    assert_eq!(s.env["EDITOR"], "emacs");
    assert_eq!(s.script_timeout(), None);
}

#[test]
fn later_calls_merge_except_atomic_settings() {
    let s = load(
        r##"
        vellum.set{ colors = { palette = { [0] = "#010101", "#020202" } }, shell = { "a", "-x" } }
        vellum.set{ colors = { palette = { [1] = "#030303" }, foreground = "#040404" } }
        vellum.set{ shell = { "b" }, env = { A = "1" } }
        vellum.set{ env = { B = "2" } }
        "##,
    )
    .unwrap();
    let p = s.colors.palette();
    assert_eq!(p.colors[0], [1, 1, 1], "kept from the first call");
    assert_eq!(p.colors[1], [3, 3, 3], "replaced by the second");
    assert_eq!(p.foreground, [4, 4, 4]);
    assert_eq!(s.shell, ["b"], "replaced whole, not merged into a, -x");
    assert_eq!((s.env["A"].as_str(), s.env["B"].as_str()), ("1", "2"));
}

#[test]
fn the_users_table_is_copied() {
    let s = load(
        r##"
        local colors = { foreground = "#010101" }
        vellum.set{ colors = colors }
        colors.foreground = "not even a color"
        "##,
    )
    .unwrap();
    assert_eq!(s.colors.foreground, Some(Color([1, 1, 1])));
}

#[test]
fn errors_name_the_line_and_the_setting() {
    let e = load_err("\n\nvellum.set{ colours = {} }");
    assert!(e.starts_with("init.lua:3: "), "{e}");
    assert!(e.contains("unknown field `colours`"), "{e}");

    let e = load_err("vellum.set{ colors = { palette = { [3] = \"red\" } } }");
    assert!(e.starts_with("init.lua:1: colors.palette"), "{e}");
    assert!(
        e.contains(r##"expected a color like "#rrggbb", got "red""##),
        "{e}"
    );

    let e = load_err("vellum.set{ colors = { palette = { [300] = \"#000000\" } } }");
    assert!(e.contains("colors.palette"), "{e}");

    let e = load_err("vellum.set{ cursor_style = \"beam\" }");
    assert!(e.contains("cursor_style") && e.contains("beam"), "{e}");

    let e = load_err("vellum.set{ repeat_rate = 0 }");
    assert!(e.ends_with("repeat_rate: must be at least 1"), "{e}");

    let e = load_err("vellum.set{ scrollback = \"lots\" }");
    assert!(e.contains("scrollback"), "{e}");
}

#[test]
fn lua_errors_keep_their_location() {
    let e = load_err("local x = \n\n)");
    assert!(e.starts_with("init.lua:3:"), "syntax: {e}");
    // Runtime errors carry a traceback after the message (for the log).
    let e = load_err("\nerror('nope')");
    assert_eq!(e.lines().next(), Some("init.lua:2: nope"), "{e}");
}

#[test]
fn a_missing_config_is_an_error() {
    let e = api::load(Some(Path::new("/nonexistent/vellum/init.lua")))
        .err()
        .unwrap();
    assert!(e.contains("/nonexistent/vellum/init.lua"), "{e}");
}

#[test]
fn require_finds_modules_next_to_the_config() {
    let (_dir, init) = config_dir(&[
        ("init.lua", "vellum.set{ font = require('theme').font }"),
        ("theme.lua", "return { font = 'from theme' }"),
        ("extra/init.lua", "return 1"),
    ]);
    assert_eq!(api::load(Some(&init)).unwrap().settings.font, "from theme");
}

#[test]
fn only_safe_libraries() {
    let s = load(
        r##"
        assert(string and table and math and os and io and package and jit)
        assert(ffi == nil and debug == nil)
        vellum.log.info("hello", 1, true)
        "##,
    );
    assert!(s.is_ok(), "{s:?}");
}

#[test]
fn command_line_flags_win() {
    let mut args = Args::parse_from(["--font".to_owned(), "Flag".to_owned()]).unwrap();
    args.keymap.layout = "de".into();
    let s = Settings {
        font: "Config".into(),
        keymap: RuleNames {
            layout: "us".into(),
            variant: "intl".into(),
            ..RuleNames::default()
        },
        ..Settings::default()
    }
    .with_cli(&args);
    assert_eq!(s.font, "Flag");
    assert_eq!(s.keymap.layout, "de");
    assert_eq!(s.keymap.variant, "intl", "not given on the command line");
}

#[test]
fn default_config_path() {
    let env = |vars: &'static [(&'static str, &'static str)]| {
        move |k: &str| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.into())
    };
    assert_eq!(
        default_path(env(&[("XDG_CONFIG_HOME", "/c"), ("HOME", "/h")])),
        Some("/c/vellum/init.lua".into())
    );
    assert_eq!(
        default_path(env(&[("XDG_CONFIG_HOME", ""), ("HOME", "/h")])),
        Some("/h/.config/vellum/init.lua".into())
    );
    assert_eq!(default_path(env(&[])), None);
}

#[test]
fn the_default_palette_is_the_terminals() {
    assert_eq!(Settings::default().colors.palette(), Palette::default());
}

fn readable(fd: &impl std::os::fd::AsFd, timeout: Duration) -> bool {
    let mut fds = [PollFd::new(fd, PollFlags::IN)];
    let ts = Timespec {
        tv_sec: timeout.as_secs() as _,
        tv_nsec: timeout.subsec_nanos().into(),
    };
    poll(&mut fds, Some(&ts)).unwrap() == 1
}

#[test]
fn the_thread_answers_and_wakes_the_loop() {
    let (_dir, init) = config_dir(&[("init.lua", "vellum.set{ scrollback = 7 }")]);
    let mut script = Script::start().unwrap();
    script.send(Request::Load(Some(init)));
    assert!(readable(&script, Duration::from_secs(10)));
    let replies = script.replies();
    assert!(
        matches!(&replies[..], [Reply::Loaded(Ok(s))] if s.scrollback == 7),
        "{replies:?}"
    );
    assert!(!readable(&script, Duration::ZERO), "eventfd drained");

    script.send(Request::Load(Some("/nonexistent/init.lua".into())));
    assert!(matches!(
        script.wait(Duration::from_secs(10)),
        Some(Reply::Loaded(Err(_)))
    ));
}

#[test]
fn a_stuck_script_is_overdue_once() {
    let (_dir, init) = config_dir(&[("init.lua", "while true do end")]);
    let mut script = Script::start().unwrap();
    let timeout = Some(Duration::from_millis(50));
    assert_eq!(script.deadline(timeout), None, "nothing asked yet");
    script.send(Request::Load(Some(init)));
    assert!(script.wait(Duration::from_millis(200)).is_none());
    let now = Instant::now();
    assert!(script.deadline(timeout).is_some_and(|d| d <= now));
    assert!(script.overdue(timeout, now));
    assert!(!script.overdue(timeout, now), "warned once");
    assert_eq!(script.deadline(timeout), None);
    assert!(!script.overdue(None, now), "never with no timeout");
    // The thread spins on; the test process exits regardless.
}
