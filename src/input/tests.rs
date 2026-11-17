//! The keyboard driven by evdev keycodes with real xkeyboard-config
//! keymaps (no devices), the repeat state machine, and the bindings.

use super::*;

// evdev keycodes (linux/input-event-codes.h).
const KEY_BACKSPACE: u32 = 14;
const KEY_Q: u32 = 16;
const KEY_E: u32 = 18;
const KEY_A: u32 = 30;
const KEY_LEFTCTRL: u32 = 29;
const KEY_APOSTROPHE: u32 = 40;
const KEY_LEFTSHIFT: u32 = 42;
const KEY_LEFTALT: u32 = 56;
const KEY_CAPSLOCK: u32 = 58;
const KEY_F1: u32 = 59;
const KEY_ENTER: u32 = 28;

fn layout(layout: &str, variant: &str) -> RuleNames {
    RuleNames {
        rules: "evdev".to_owned(),
        model: "pc105".to_owned(),
        layout: layout.to_owned(),
        variant: variant.to_owned(),
        options: Some(String::new()),
    }
}

fn us() -> Keyboard {
    Keyboard::with_compose(&layout("us", ""), "").unwrap()
}

fn press(kb: &mut Keyboard, code: u32) -> Option<KeyEvent> {
    kb.key(code, KeyKind::Press)
}

fn release(kb: &mut Keyboard, code: u32) -> Option<KeyEvent> {
    kb.key(code, KeyKind::Release)
}

#[test]
fn plain_and_shifted_letters() {
    let mut kb = us();
    let a = press(&mut kb, KEY_A).unwrap();
    assert_eq!(
        (a.sym, a.base, a.us, a.text.as_str(), a.mods),
        (Keysym::a, Keysym::a, None, "a", Mods::empty())
    );
    let up = release(&mut kb, KEY_A).unwrap();
    assert_eq!((up.kind, up.text.as_str()), (KeyKind::Release, ""));

    press(&mut kb, KEY_LEFTSHIFT);
    let a = press(&mut kb, KEY_A).unwrap();
    assert_eq!(
        (a.sym, a.base, a.text.as_str(), a.mods),
        (Keysym::A, Keysym::a, "A", Mods::SHIFT)
    );
    release(&mut kb, KEY_A);
    release(&mut kb, KEY_LEFTSHIFT);
    assert_eq!(press(&mut kb, KEY_A).unwrap().text, "a", "Shift released");
}

#[test]
fn modifier_keys_report_their_own_state() {
    // The kitty protocol wants a Shift press with Shift set and its release
    // without.
    let mut kb = us();
    let down = press(&mut kb, KEY_LEFTSHIFT).unwrap();
    assert_eq!((down.sym, down.mods), (Keysym::Shift_L, Mods::SHIFT));
    let up = release(&mut kb, KEY_LEFTSHIFT).unwrap();
    assert_eq!((up.sym, up.mods), (Keysym::Shift_L, Mods::empty()));
}

#[test]
fn ctrl_leaves_the_text_alone() {
    let mut kb = us();
    press(&mut kb, KEY_LEFTCTRL);
    press(&mut kb, KEY_LEFTALT);
    let a = press(&mut kb, KEY_A).unwrap();
    assert_eq!(
        (a.sym, a.text.as_str(), a.mods),
        (Keysym::a, "a", Mods::CTRL | Mods::ALT)
    );
    assert_eq!(binding(&a), None);
}

#[test]
fn caps_lock_is_reported_and_types_capitals() {
    let mut kb = us();
    press(&mut kb, KEY_CAPSLOCK);
    release(&mut kb, KEY_CAPSLOCK);
    let a = press(&mut kb, KEY_A).unwrap();
    assert_eq!((a.text.as_str(), a.mods), ("A", Mods::CAPS_LOCK));
}

#[test]
fn control_keys_have_no_text() {
    let mut kb = us();
    let enter = press(&mut kb, KEY_ENTER).unwrap();
    assert_eq!((enter.sym, enter.text.as_str()), (Keysym::Return, ""));
}

#[test]
fn other_layouts_carry_the_us_key() {
    let mut kb = Keyboard::with_compose(&layout("ru", ""), "").unwrap();
    let q = press(&mut kb, KEY_Q).unwrap();
    assert_eq!(q.text, "й");
    assert_eq!(q.base, q.sym);
    assert_eq!(q.us, Some(Keysym::q));
}

#[test]
fn dead_keys_compose() {
    let compose = "<dead_acute> <e> : \"é\" eacute\n";
    let mut kb = Keyboard::with_compose(&layout("us", "intl"), compose).unwrap();
    assert_eq!(press(&mut kb, KEY_APOSTROPHE), None, "dead key swallowed");
    assert_eq!(release(&mut kb, KEY_APOSTROPHE), None, "and its release");
    let e = press(&mut kb, KEY_E).unwrap();
    assert_eq!((e.sym, e.text.as_str()), (Keysym::eacute, "é"));
    assert!(release(&mut kb, KEY_E).is_some());
    // The sequence is over: e is e again.
    assert_eq!(press(&mut kb, KEY_E).unwrap().text, "e");
}

#[test]
fn a_cancelled_sequence_sends_nothing() {
    let compose = "<dead_acute> <e> : \"é\" eacute\n";
    let mut kb = Keyboard::with_compose(&layout("us", "intl"), compose).unwrap();
    press(&mut kb, KEY_APOSTROPHE);
    release(&mut kb, KEY_APOSTROPHE);
    assert_eq!(press(&mut kb, KEY_Q), None, "no sequence for ´q");
    assert_eq!(press(&mut kb, KEY_Q).unwrap().text, "q");
}

#[test]
fn modifiers_pass_through_a_sequence() {
    let compose = "<dead_acute> <E> : \"É\" Eacute\n";
    let mut kb = Keyboard::with_compose(&layout("us", "intl"), compose).unwrap();
    press(&mut kb, KEY_APOSTROPHE);
    release(&mut kb, KEY_APOSTROPHE);
    assert!(press(&mut kb, KEY_LEFTSHIFT).is_some(), "Shift isn't fed");
    assert_eq!(press(&mut kb, KEY_E).unwrap().text, "É");
}

#[test]
fn repeat_reads_the_current_modifiers() {
    let mut kb = us();
    press(&mut kb, KEY_A);
    press(&mut kb, KEY_LEFTSHIFT);
    let r = kb.key(KEY_A, KeyKind::Repeat).unwrap();
    assert_eq!((r.kind, r.text.as_str()), (KeyKind::Repeat, "A"));
}

#[test]
fn reset_lets_go_of_held_modifiers() {
    let mut kb = us();
    press(&mut kb, KEY_LEFTCTRL);
    press(&mut kb, KEY_LEFTALT);
    assert_eq!(press(&mut kb, KEY_F1).unwrap().mods, Mods::CTRL | Mods::ALT);
    // Switched away: the releases happen on another VT.
    kb.reset();
    let a = press(&mut kb, KEY_A).unwrap();
    assert_eq!((a.text.as_str(), a.mods), ("a", Mods::empty()));
}

#[test]
fn reset_keeps_locks() {
    let mut kb = us();
    press(&mut kb, KEY_CAPSLOCK);
    release(&mut kb, KEY_CAPSLOCK);
    press(&mut kb, KEY_LEFTSHIFT);
    kb.reset();
    let a = press(&mut kb, KEY_A).unwrap();
    assert_eq!((a.text.as_str(), a.mods), ("A", Mods::CAPS_LOCK));
}

#[test]
fn reset_drops_a_compose_sequence() {
    let compose = "<dead_acute> <e> : \"é\" eacute\n";
    let mut kb = Keyboard::with_compose(&layout("us", "intl"), compose).unwrap();
    assert_eq!(press(&mut kb, KEY_APOSTROPHE), None);
    kb.reset();
    // Its release isn't swallowed any more (it happened elsewhere), and
    // e is plain e.
    assert_eq!(press(&mut kb, KEY_E).unwrap().text, "e");
    assert!(release(&mut kb, KEY_E).is_some());
}

#[test]
fn which_keys_repeat() {
    let kb = us();
    assert!(kb.repeats(KEY_A));
    assert!(kb.repeats(KEY_BACKSPACE));
    assert!(!kb.repeats(KEY_LEFTSHIFT));
}

#[test]
fn repeat_state() {
    let mut r = RepeatState::default();
    assert_eq!(r.press(KEY_A, true), Timer::Start);
    assert_eq!(r.key(), Some(KEY_A));
    // Shift doesn't repeat and doesn't stop a: it repeats shifted.
    assert_eq!(r.press(KEY_LEFTSHIFT, false), Timer::Unchanged);
    assert_eq!(r.key(), Some(KEY_A));
    // Another repeating key takes over.
    assert_eq!(r.press(KEY_E, true), Timer::Start);
    assert_eq!(r.key(), Some(KEY_E));
    // Releasing a key that isn't repeating changes nothing.
    assert_eq!(r.release(KEY_A), Timer::Unchanged);
    assert_eq!(r.key(), Some(KEY_E));
    assert_eq!(r.release(KEY_E), Timer::Stop);
    assert_eq!(r.key(), None);
    assert_eq!(r.release(KEY_LEFTSHIFT), Timer::Unchanged);
}

#[test]
fn repeat_state_clears() {
    let mut r = RepeatState::default();
    assert_eq!(r.clear(), Timer::Unchanged, "nothing repeating");
    r.press(KEY_A, true);
    assert_eq!(r.clear(), Timer::Stop);
    assert_eq!(r.key(), None);
    // a's release, if it ever comes, changes nothing.
    assert_eq!(r.release(KEY_A), Timer::Unchanged);
}

#[test]
fn repeat_timer_disarms() {
    let mut r = KeyRepeat::new(Duration::from_millis(600), Duration::from_millis(40)).unwrap();
    r.press(KEY_A, true).unwrap();
    r.release(KEY_A).unwrap();
    assert_eq!(r.expired().unwrap(), None, "nothing expired, nothing held");
    r.press(KEY_A, true).unwrap();
    r.stop().unwrap();
    assert_eq!(r.expired().unwrap(), None, "stopped");
}

#[test]
fn bindings() {
    let mut kb = us();
    press(&mut kb, KEY_LEFTCTRL);
    press(&mut kb, KEY_LEFTALT);
    let bs = press(&mut kb, KEY_BACKSPACE).unwrap();
    assert_eq!(binding(&bs), Some(Binding::Exit));
    let f1 = press(&mut kb, KEY_F1).unwrap();
    assert_eq!(binding(&f1), Some(Binding::SwitchVt(1)), "{:?}", f1.sym);

    let mut kb = us();
    let bs = press(&mut kb, KEY_BACKSPACE).unwrap();
    assert_eq!(binding(&bs), None);
    let f1 = press(&mut kb, KEY_F1).unwrap();
    assert_eq!(binding(&f1), None);
}

#[test]
fn through_the_encoder() {
    use crate::keys::{KeyModes, encode, flags};
    let legacy = KeyModes::default();
    let kitty = KeyModes {
        kitty: flags::DISAMBIGUATE | flags::EVENT_TYPES | flags::ALL_KEYS,
        ..KeyModes::default()
    };
    let mut kb = us();
    let ctrl = press(&mut kb, KEY_LEFTCTRL).unwrap();
    assert_eq!(encode(&ctrl, &legacy), None);
    assert_eq!(encode(&ctrl, &kitty).unwrap(), b"\x1b[57442;5u");
    let a = press(&mut kb, KEY_A).unwrap();
    assert_eq!(encode(&a, &legacy).unwrap(), b"\x01");
    assert_eq!(encode(&a, &kitty).unwrap(), b"\x1b[97;5u");
    let up = release(&mut kb, KEY_LEFTCTRL).unwrap();
    assert_eq!(encode(&up, &kitty).unwrap(), b"\x1b[57442;1:3u");
}

#[test]
fn scroll_bindings() {
    use crate::keys::KeyEvent;
    let shift = Mods::SHIFT;
    for (sym, want) in [
        (Keysym::Page_Up, Scroll::PageUp),
        (Keysym::Page_Down, Scroll::PageDown),
        (Keysym::Home, Scroll::Top),
        (Keysym::End, Scroll::Bottom),
        (Keysym::KP_Page_Up, Scroll::PageUp),
    ] {
        assert_eq!(
            binding(&KeyEvent::press(sym, shift)),
            Some(Binding::Scroll(want))
        );
        // Lock keys don't get in the way.
        let locked = KeyEvent::press(sym, shift | Mods::NUM_LOCK | Mods::CAPS_LOCK);
        assert_eq!(binding(&locked), Some(Binding::Scroll(want)));
        // Without Shift, or with more than Shift, the key is the program's.
        assert_eq!(binding(&KeyEvent::press(sym, Mods::empty())), None);
        assert_eq!(binding(&KeyEvent::press(sym, shift | Mods::CTRL)), None);
    }
}
