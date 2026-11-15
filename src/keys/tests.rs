//! Key encoding, table by table. Events are built by hand here; turning
//! real keymaps into events is tested with the input module (step 6C).

use super::*;

const NONE: Mods = Mods::empty();
const SHIFT: Mods = Mods::SHIFT;
const CTRL: Mods = Mods::CTRL;
const ALT: Mods = Mods::ALT;

fn sym(c: char) -> Keysym {
    xkbcommon::xkb::utf32_to_keysym(c.into())
}

/// A key typing `c` (its keysym), with `mods`.
fn ch(c: char, mods: Mods) -> KeyEvent {
    KeyEvent::press(sym(c), mods)
}

/// Shift+`base`, typing `shifted`, with `mods` besides Shift.
fn shifted(base: char, shifted: char, mods: Mods) -> KeyEvent {
    KeyEvent {
        base: sym(base),
        ..ch(shifted, mods | SHIFT)
    }
}

fn key(s: Keysym, mods: Mods) -> KeyEvent {
    KeyEvent::press(s, mods)
}

fn kind(mut e: KeyEvent, kind: KeyKind) -> KeyEvent {
    e.kind = kind;
    if kind == KeyKind::Release {
        e.text.clear();
    }
    e
}

fn legacy_modes() -> KeyModes {
    KeyModes::default()
}

fn kitty_modes(flags: u8) -> KeyModes {
    KeyModes {
        kitty: flags,
        ..KeyModes::default()
    }
}

/// Encode and show as text, `None` as "(nothing)", for readable failures.
fn enc(e: &KeyEvent, modes: KeyModes) -> String {
    match encode(e, &modes) {
        Some(b) => String::from_utf8(b).unwrap(),
        None => "(nothing)".to_owned(),
    }
}

fn check(cases: &[(KeyEvent, &str)], modes: KeyModes) {
    for (e, want) in cases {
        assert_eq!(enc(e, modes), *want, "{e:?} with {modes:?}");
    }
}

#[test]
fn legacy_text_and_control_characters() {
    let mut russian = ch('ф', CTRL);
    russian.us = Some(sym('a'));
    check(
        &[
            (ch('a', NONE), "a"),
            (shifted('a', 'A', NONE), "A"),
            (ch('é', NONE), "é"),
            (ch('a', CTRL), "\x01"),
            (shifted('a', 'A', CTRL), "\x01"),
            (ch('a', ALT), "\x1ba"),
            (ch('a', CTRL | ALT), "\x1b\x01"),
            (ch(' ', CTRL), "\0"),
            (ch('2', CTRL), "\0"),
            (ch('[', CTRL), "\x1b"),
            (ch('\\', CTRL), "\x1c"),
            (ch(']', CTRL), "\x1d"),
            (shifted('6', '^', CTRL), "\x1e"),
            (ch('/', CTRL), "\x1f"),
            (shifted('/', '?', CTRL), "\x7f"),
            (ch('8', CTRL), "\x7f"),
            // No control character for it: the text as typed.
            (ch('1', CTRL), "1"),
            (ch('é', CTRL), "é"),
            // Non-Latin layouts: the US key's control character.
            (russian, "\x01"),
        ],
        legacy_modes(),
    );
}

#[test]
fn legacy_editing_keys() {
    check(
        &[
            (key(Keysym::Return, NONE), "\r"),
            (key(Keysym::Return, ALT), "\x1b\r"),
            (key(Keysym::Tab, NONE), "\t"),
            (key(Keysym::ISO_Left_Tab, SHIFT), "\x1b[Z"),
            (key(Keysym::BackSpace, NONE), "\x7f"),
            (key(Keysym::BackSpace, CTRL), "\x08"),
            (key(Keysym::BackSpace, ALT), "\x1b\x7f"),
            (key(Keysym::Escape, NONE), "\x1b"),
            (key(Keysym::Escape, ALT), "\x1b\x1b"),
        ],
        legacy_modes(),
    );
    let lnm = KeyModes {
        newline: true,
        ..legacy_modes()
    };
    assert_eq!(enc(&key(Keysym::Return, NONE), lnm), "\r\n");
}

#[test]
fn legacy_cursor_and_function_keys() {
    let cases = [
        (key(Keysym::Up, NONE), "\x1b[A", "\x1bOA"),
        (key(Keysym::End, NONE), "\x1b[F", "\x1bOF"),
        // Modified keys are the same in both modes.
        (key(Keysym::Up, CTRL), "\x1b[1;5A", "\x1b[1;5A"),
        (key(Keysym::Right, SHIFT | ALT), "\x1b[1;4C", "\x1b[1;4C"),
        (key(Keysym::Home, Mods::SUPER), "\x1b[1;9H", "\x1b[1;9H"),
        (key(Keysym::Delete, NONE), "\x1b[3~", "\x1b[3~"),
        (key(Keysym::Delete, CTRL), "\x1b[3;5~", "\x1b[3;5~"),
        (key(Keysym::Page_Up, NONE), "\x1b[5~", "\x1b[5~"),
        (key(Keysym::Insert, SHIFT), "\x1b[2;2~", "\x1b[2;2~"),
        (key(Keysym::F1, NONE), "\x1bOP", "\x1bOP"),
        (key(Keysym::F1, SHIFT), "\x1b[1;2P", "\x1b[1;2P"),
        (key(Keysym::F4, NONE), "\x1bOS", "\x1bOS"),
        (key(Keysym::F5, NONE), "\x1b[15~", "\x1b[15~"),
        (key(Keysym::F12, CTRL), "\x1b[24;5~", "\x1b[24;5~"),
        // F13 and up: F1 and up with Shift.
        (key(Keysym::F13, NONE), "\x1b[1;2P", "\x1b[1;2P"),
        (key(Keysym::F13, CTRL), "\x1b[1;6P", "\x1b[1;6P"),
        (key(Keysym::F17, NONE), "\x1b[15;2~", "\x1b[15;2~"),
    ];
    let app = KeyModes {
        cursor_keys_app: true,
        ..legacy_modes()
    };
    for (e, normal, application) in cases {
        assert_eq!(enc(&e, legacy_modes()), normal, "{e:?}");
        assert_eq!(enc(&e, app), application, "{e:?} with DECCKM");
    }
}

#[test]
fn legacy_keypad() {
    let app = KeyModes {
        keypad_app: true,
        ..legacy_modes()
    };
    let kp1 = KeyEvent {
        text: "1".into(),
        ..key(Keysym::KP_1, Mods::NUM_LOCK)
    };
    assert_eq!(enc(&kp1, legacy_modes()), "1");
    assert_eq!(enc(&kp1, app), "\x1bOq");
    // Without Num Lock: a cursor key.
    assert_eq!(enc(&key(Keysym::KP_End, NONE), legacy_modes()), "\x1b[F");
    assert_eq!(enc(&key(Keysym::KP_Enter, NONE), legacy_modes()), "\r");
    assert_eq!(enc(&key(Keysym::KP_Enter, NONE), app), "\x1bOM");
    let plus = KeyEvent {
        text: "+".into(),
        ..key(Keysym::KP_Add, NONE)
    };
    assert_eq!(enc(&plus, legacy_modes()), "+");
    assert_eq!(enc(&plus, app), "\x1bOk");
}

#[test]
fn legacy_sends_presses_and_repeats_only() {
    check(
        &[
            (kind(ch('a', NONE), KeyKind::Repeat), "a"),
            (kind(ch('a', NONE), KeyKind::Release), "(nothing)"),
            (key(Keysym::Shift_L, SHIFT), "(nothing)"),
            (key(Keysym::Caps_Lock, NONE), "(nothing)"),
        ],
        legacy_modes(),
    );
}

#[test]
fn kitty_disambiguate() {
    let kp1 = KeyEvent {
        text: "1".into(),
        ..key(Keysym::KP_1, Mods::NUM_LOCK)
    };
    check(
        &[
            // Text with at most Shift stays text.
            (ch('a', NONE), "a"),
            (shifted('a', 'A', NONE), "A"),
            (ch('a', Mods::CAPS_LOCK), "a"),
            (kp1, "1"),
            // Ambiguous ones become CSI u.
            (key(Keysym::Escape, NONE), "\x1b[27u"),
            (ch('a', CTRL), "\x1b[97;5u"),
            (ch('a', ALT), "\x1b[97;3u"),
            (shifted('a', 'A', CTRL), "\x1b[97;6u"),
            (ch(' ', CTRL), "\x1b[32;5u"),
            (ch('é', CTRL), "\x1b[233;5u"),
            // Enter, Tab, Backspace: legacy unless modified.
            (key(Keysym::Return, NONE), "\r"),
            (key(Keysym::Return, CTRL), "\x1b[13;5u"),
            (key(Keysym::ISO_Left_Tab, SHIFT), "\x1b[9;2u"),
            (key(Keysym::BackSpace, ALT), "\x1b[127;3u"),
            // Function keys: `CSI x` (no parameters) unmodified.
            (key(Keysym::Up, NONE), "\x1b[A"),
            (key(Keysym::Up, CTRL), "\x1b[1;5A"),
            (key(Keysym::F1, NONE), "\x1b[P"),
            (key(Keysym::F3, NONE), "\x1b[13~"),
            (key(Keysym::F3, CTRL), "\x1b[13;5~"),
            (key(Keysym::F5, SHIFT), "\x1b[15;2~"),
            (key(Keysym::F13, NONE), "\x1b[57376u"),
            // Keypad keys that don't type text.
            (key(Keysym::KP_End, NONE), "\x1b[57424u"),
            (key(Keysym::KP_Enter, NONE), "\x1b[57414u"),
            // Modifier keys and releases: nothing.
            (key(Keysym::Shift_L, SHIFT), "(nothing)"),
            (kind(ch('a', CTRL), KeyKind::Release), "(nothing)"),
            // Locks aren't reported for text keys, but are for the others.
            (ch('a', CTRL | Mods::NUM_LOCK), "\x1b[97;5u"),
            (key(Keysym::Up, CTRL | Mods::NUM_LOCK), "\x1b[1;133A"),
            (key(Keysym::Escape, Mods::CAPS_LOCK), "\x1b[27;65u"),
            (key(Keysym::Return, Mods::NUM_LOCK), "\r"),
        ],
        kitty_modes(flags::DISAMBIGUATE),
    );
    // DECCKM still applies to unmodified cursor keys, and only those.
    let app = KeyModes {
        cursor_keys_app: true,
        ..kitty_modes(flags::DISAMBIGUATE)
    };
    assert_eq!(enc(&key(Keysym::Up, NONE), app), "\x1bOA");
    assert_eq!(enc(&key(Keysym::Up, CTRL), app), "\x1b[1;5A");
    assert_eq!(enc(&key(Keysym::F1, NONE), app), "\x1b[P");
}

#[test]
fn kitty_event_types() {
    use KeyKind::*;
    check(
        &[
            // Text keys type text, and have no events of their own.
            (kind(ch('a', NONE), Repeat), "a"),
            (kind(ch('a', NONE), Release), "(nothing)"),
            (kind(shifted('a', 'A', NONE), Release), "(nothing)"),
            (kind(ch('a', CTRL), Repeat), "\x1b[97;5:2u"),
            (kind(ch('a', CTRL), Release), "\x1b[97;5:3u"),
            (kind(key(Keysym::Up, NONE), Release), "\x1b[1;1:3A"),
            (kind(key(Keysym::Up, SHIFT), Repeat), "\x1b[1;2:2A"),
            (kind(key(Keysym::Delete, NONE), Release), "\x1b[3;1:3~"),
            (kind(key(Keysym::Escape, NONE), Release), "\x1b[27;1:3u"),
            // Releases of Enter, Tab and Backspace only with every key.
            (kind(key(Keysym::Return, NONE), Release), "(nothing)"),
            (kind(key(Keysym::BackSpace, NONE), Release), "(nothing)"),
        ],
        kitty_modes(flags::DISAMBIGUATE | flags::EVENT_TYPES),
    );
}

#[test]
fn kitty_alternate_keys() {
    let mut russian = ch('ф', CTRL);
    russian.us = Some(sym('a'));
    let mut russian_shifted = shifted('ф', 'Ф', ALT);
    russian_shifted.us = Some(sym('a'));
    check(
        &[
            (shifted('a', 'A', CTRL), "\x1b[97:65;6u"),
            (shifted('2', '@', ALT), "\x1b[50:64;4u"),
            (ch('a', CTRL), "\x1b[97;5u"),
            (russian, "\x1b[1092::97;5u"),
            (russian_shifted, "\x1b[1092:1060:97;4u"),
            // Function keys have no alternates.
            (key(Keysym::Up, CTRL), "\x1b[1;5A"),
        ],
        kitty_modes(flags::DISAMBIGUATE | flags::ALTERNATE_KEYS),
    );
}

#[test]
fn kitty_all_keys() {
    check(
        &[
            (ch('a', NONE), "\x1b[97u"),
            (shifted('a', 'A', NONE), "\x1b[97;2u"),
            (key(Keysym::Return, NONE), "\x1b[13u"),
            (key(Keysym::Tab, NONE), "\x1b[9u"),
            (key(Keysym::Up, NONE), "\x1b[A"),
            (key(Keysym::F3, NONE), "\x1b[13~"),
            (key(Keysym::Shift_L, SHIFT), "\x1b[57441;2u"),
            (key(Keysym::Control_R, CTRL), "\x1b[57448;5u"),
            (key(Keysym::Caps_Lock, Mods::CAPS_LOCK), "\x1b[57358;65u"),
            // Locks count when every key is reported.
            (ch('a', Mods::CAPS_LOCK), "\x1b[97;65u"),
            (ch('a', Mods::NUM_LOCK), "\x1b[97;129u"),
        ],
        kitty_modes(flags::DISAMBIGUATE | flags::ALL_KEYS),
    );
    let both = kitty_modes(flags::ALL_KEYS | flags::EVENT_TYPES);
    assert_eq!(
        enc(&kind(key(Keysym::Return, NONE), KeyKind::Release), both),
        "\x1b[13;1:3u"
    );
    assert_eq!(
        enc(&kind(key(Keysym::Shift_L, NONE), KeyKind::Release), both),
        "\x1b[57441;1:3u"
    );
}

#[test]
fn kitty_associated_text() {
    let modes = kitty_modes(flags::DISAMBIGUATE | flags::ALL_KEYS | flags::TEXT);
    check(
        &[
            (ch('a', NONE), "\x1b[97;;97u"),
            (shifted('a', 'A', NONE), "\x1b[97;2;65u"),
            (ch('é', NONE), "\x1b[233;;233u"),
            // Ctrl and Alt combinations type nothing.
            (ch('a', CTRL), "\x1b[97;5u"),
            (ch('a', ALT), "\x1b[97;3u"),
            // Nor do Enter and the like.
            (key(Keysym::Return, NONE), "\x1b[13u"),
        ],
        modes,
    );
    // Composed text: several code points.
    let e = KeyEvent {
        text: "e\u{301}".into(),
        ..ch('e', NONE)
    };
    assert_eq!(enc(&e, modes), "\x1b[101;;101:769u");
    // Without ALL_KEYS the flag does nothing: text is sent as text.
    assert_eq!(enc(&ch('a', NONE), kitty_modes(flags::TEXT | 1)), "a");
}

#[test]
fn modes_come_from_the_terminal() {
    let mut t = Terminal::new(10, 2);
    assert_eq!(KeyModes::of(&t), KeyModes::default());
    t.feed(b"\x1b[?1h\x1b=\x1b[20h\x1b[>5u");
    assert_eq!(
        KeyModes::of(&t),
        KeyModes {
            cursor_keys_app: true,
            keypad_app: true,
            newline: true,
            kitty: 5,
        }
    );
}

#[test]
fn kitty_keypad_text_keys() {
    let kp1 = KeyEvent {
        text: "1".into(),
        ..key(Keysym::KP_1, Mods::NUM_LOCK)
    };
    let modes = kitty_modes(flags::DISAMBIGUATE | flags::EVENT_TYPES);
    assert_eq!(enc(&kp1, modes), "1");
    assert_eq!(
        enc(&kind(kp1.clone(), KeyKind::Release), modes),
        "(nothing)"
    );
    // Modified: its keypad code, without the lock (it types text).
    let ctrl = KeyEvent {
        mods: CTRL | Mods::NUM_LOCK,
        ..kp1.clone()
    };
    assert_eq!(enc(&ctrl, modes), "\x1b[57400;5u");
    // Every key reported: locks included.
    assert_eq!(enc(&kp1, kitty_modes(flags::ALL_KEYS)), "\x1b[57400;129u");
}
