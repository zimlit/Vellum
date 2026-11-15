//! Keyboard input: libinput devices opened through the seat, xkbcommon for
//! keymaps and compose, key repeat, and the bindings vellum keeps for
//! itself.
//!
//! libinput reports physical keys (evdev codes) from every keyboard on the
//! seat. [`Keyboard`] runs them through the xkb keymap and compose table
//! and produces [`KeyEvent`]s for [`crate::keys::encode`]. The console
//! keyboard is off while the seat is ours (`K_OFF`), so the kernel sees
//! none of this, including Ctrl+Alt+Fn (step 7).

use std::cell::RefCell;
use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::Path;
use std::rc::Rc;
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use input::event::keyboard::{KeyState, KeyboardEvent, KeyboardEventTrait};
use input::event::{DeviceEvent, Event, EventTrait};
use input::{DeviceCapability, Libinput, LibinputInterface};
use rustix::io::Errno;
use rustix::time::{
    Itimerspec, TimerfdClockId, TimerfdFlags, TimerfdTimerFlags, Timespec, timerfd_create,
    timerfd_settime,
};
use xkbcommon::xkb::{self, Keycode, Keysym, compose};

use crate::keys::{KeyEvent, KeyKind, Mods};
use crate::session::Session;

/// Delay before a held key starts repeating.
pub const REPEAT_DELAY: Duration = Duration::from_millis(600);
/// Time between repeats (25 per second).
pub const REPEAT_INTERVAL: Duration = Duration::from_millis(40);

/// evdev keycodes are xkb keycodes minus 8 (a leftover from X11, where
/// keycodes below 8 were reserved).
const EVDEV_OFFSET: u32 = 8;

/// The xkb RMLVO names (rules, model, layout, variant, options). Empty
/// fields, and `options: None`, let libxkbcommon use `XKB_DEFAULT_*` from
/// the environment, then the system default (`evdev`, `pc105`, `us`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuleNames {
    pub rules: String,
    pub model: String,
    pub layout: String,
    pub variant: String,
    pub options: Option<String>,
}

impl RuleNames {
    fn keymap(&self, context: &xkb::Context) -> Option<xkb::Keymap> {
        xkb::Keymap::new_from_names(
            context,
            &self.rules,
            &self.model,
            &self.layout,
            &self.variant,
            self.options.clone(),
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
    }
}

/// Indices of the modifiers [`Mods`] reports, in the compiled keymap.
struct ModIndices {
    shift: xkb::ModIndex,
    ctrl: xkb::ModIndex,
    alt: xkb::ModIndex,
    logo: xkb::ModIndex,
    caps: xkb::ModIndex,
    num: xkb::ModIndex,
}

impl ModIndices {
    fn of(keymap: &xkb::Keymap) -> Self {
        Self {
            shift: keymap.mod_get_index(xkb::MOD_NAME_SHIFT),
            ctrl: keymap.mod_get_index(xkb::MOD_NAME_CTRL),
            alt: keymap.mod_get_index(xkb::MOD_NAME_ALT),
            logo: keymap.mod_get_index(xkb::MOD_NAME_LOGO),
            caps: keymap.mod_get_index(xkb::MOD_NAME_CAPS),
            num: keymap.mod_get_index(xkb::MOD_NAME_NUM),
        }
    }
}

/// The keymap and its state: evdev keycodes in, [`KeyEvent`]s out.
pub struct Keyboard {
    keymap: xkb::Keymap,
    state: xkb::State,
    mods: ModIndices,
    /// The US layout, for [`KeyEvent::us`].
    us: Option<xkb::Keymap>,
    compose: Option<compose::State>,
    /// Keys whose press a compose sequence swallowed; their releases are
    /// swallowed too.
    swallowed: Vec<u32>,
}

impl Keyboard {
    /// Compile the keymap for `names`, with the compose table for the
    /// locale (`LC_ALL`, `LC_CTYPE`, `LANG`, else `C`).
    pub fn new(names: &RuleNames) -> Result<Self> {
        let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
            .into_iter()
            .filter_map(std::env::var_os)
            .find(|v| !v.is_empty())
            .unwrap_or_else(|| OsString::from("C"));
        Self::build(names, |context| {
            match compose::Table::new_from_locale(context, &locale, compose::COMPILE_NO_FLAGS) {
                Ok(table) => Some(table),
                Err(()) => {
                    tracing::warn!(
                        locale = %locale.to_string_lossy(),
                        "no compose table for the locale; dead keys won't compose"
                    );
                    None
                }
            }
        })
    }

    /// A keyboard with a compose table from `source` (Compose file syntax)
    /// instead of the locale's.
    pub fn with_compose(names: &RuleNames, source: &str) -> Result<Self> {
        let keyboard = Self::build(names, |context| {
            compose::Table::new_from_buffer(
                context,
                source,
                "C",
                compose::FORMAT_TEXT_V1,
                compose::COMPILE_NO_FLAGS,
            )
            .ok()
        })?;
        if keyboard.compose.is_none() {
            bail!("the compose table didn't compile");
        }
        Ok(keyboard)
    }

    fn build(
        names: &RuleNames,
        compose: impl FnOnce(&xkb::Context) -> Option<compose::Table>,
    ) -> Result<Self> {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = names
            .keymap(&context)
            .ok_or_else(|| anyhow!("xkb couldn't compile a keymap for {names:?}"))?;
        let us = RuleNames {
            layout: "us".to_owned(),
            options: Some(String::new()),
            ..RuleNames::default()
        }
        .keymap(&context);
        if us.is_none() {
            tracing::warn!("no US keymap; non-Latin layouts lose the US key fallback");
        }
        let layouts = (0..keymap.num_layouts())
            .map(|i| keymap.layout_get_name(i).to_owned())
            .collect::<Vec<_>>();
        tracing::info!(?names, ?layouts, "keymap compiled");
        Ok(Self {
            state: xkb::State::new(&keymap),
            mods: ModIndices::of(&keymap),
            compose: compose(&context).map(|t| compose::State::new(&t, compose::STATE_NO_FLAGS)),
            keymap,
            us,
            swallowed: Vec::new(),
        })
    }

    /// Whether holding the key with this evdev code repeats it.
    pub fn repeats(&self, evdev: u32) -> bool {
        self.keymap.key_repeats(Keycode::new(evdev + EVDEV_OFFSET))
    }

    /// Feed one key event. Presses and releases update the modifier state,
    /// so they must all come here, even for keys handled elsewhere; a
    /// repeat reads the current state. `None` means the key sends nothing
    /// (a compose sequence took it, or it has no keysym).
    pub fn key(&mut self, evdev: u32, kind: KeyKind) -> Option<KeyEvent> {
        let code = Keycode::new(evdev + EVDEV_OFFSET);
        // The keysym comes from the state before this key changes it: Shift
        // itself is `Shift_L`, and `a` under Shift is `A`.
        let mut sym = self.state.key_get_one_sym(code);
        let layout = self.state.key_get_layout(code);
        let level1 = |keymap: &xkb::Keymap, layout| {
            keymap
                .key_get_syms_by_level(code, layout, 0)
                .first()
                .copied()
        };
        let base = level1(&self.keymap, layout).unwrap_or(sym);
        let us = self
            .us
            .as_ref()
            .and_then(|us| level1(us, 0))
            .filter(|&us| us != base);

        // The modifiers come from after it: the kitty protocol reports a
        // Shift press with Shift held, and its release without.
        match kind {
            KeyKind::Press => {
                self.state.update_key(code, xkb::KeyDirection::Down);
            }
            KeyKind::Release => {
                self.state.update_key(code, xkb::KeyDirection::Up);
            }
            KeyKind::Repeat => {}
        }
        let mods = self.mods();

        let mut composed = None;
        if kind == KeyKind::Release {
            if let Some(i) = self.swallowed.iter().position(|&k| k == evdev) {
                self.swallowed.swap_remove(i);
                return None;
            }
        } else if let Some(c) = &mut self.compose
            && c.feed(sym) == compose::FeedResult::Accepted
        {
            match c.status() {
                status @ (compose::Status::Composing | compose::Status::Cancelled) => {
                    if status == compose::Status::Cancelled {
                        c.reset();
                    }
                    if kind == KeyKind::Press {
                        self.swallowed.push(evdev);
                    }
                    return None;
                }
                compose::Status::Composed => {
                    composed = c.utf8();
                    if let Some(s) = c.keysym() {
                        sym = s;
                    }
                    c.reset();
                }
                compose::Status::Nothing => {}
            }
        }
        if sym == Keysym::NoSymbol {
            return None;
        }

        // Not `state.key_get_utf8`, which applies Ctrl (Ctrl+a → 0x01):
        // the encoder decides what Ctrl does.
        // Keys like Return and Escape have control "text"; they're keys,
        // not text, to the encoder.
        let text = match kind {
            KeyKind::Release => String::new(),
            _ => composed
                .unwrap_or_else(|| xkb::keysym_to_utf8(sym))
                .chars()
                .filter(|c| !c.is_control())
                .collect(),
        };
        Some(KeyEvent {
            sym,
            base,
            us,
            text,
            mods,
            kind,
        })
    }

    /// Forget every held key, after the session was disabled: the keys
    /// released on another VT never reach us, so xkb would keep Ctrl and
    /// Alt held from the Ctrl+Alt+Fn that switched away. Locks (Caps Lock,
    /// a locked layout) stay as they were. An unfinished compose sequence
    /// is dropped.
    pub fn reset(&mut self) {
        let locked = self.state.serialize_mods(xkb::STATE_MODS_LOCKED);
        let layout = self.state.serialize_layout(xkb::STATE_LAYOUT_LOCKED);
        self.state = xkb::State::new(&self.keymap);
        self.state.update_mask(0, 0, locked, 0, 0, layout);
        if let Some(c) = &mut self.compose {
            c.reset();
        }
        self.swallowed.clear();
    }

    fn mods(&self) -> Mods {
        let m = &self.mods;
        let active = |idx| {
            self.state
                .mod_index_is_active(idx, xkb::STATE_MODS_EFFECTIVE)
        };
        [
            (m.shift, Mods::SHIFT),
            (m.ctrl, Mods::CTRL),
            (m.alt, Mods::ALT),
            (m.logo, Mods::SUPER),
            (m.caps, Mods::CAPS_LOCK),
            (m.num, Mods::NUM_LOCK),
        ]
        .into_iter()
        .filter(|&(idx, _)| active(idx))
        .fold(Mods::empty(), |acc, (_, bit)| acc | bit)
    }
}

/// What the repeat timer should do after a key event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timer {
    /// (Re)start: the delay, then the interval.
    Start,
    Stop,
    Unchanged,
}

/// Which key is repeating, separate from the timer so it can be tested.
///
/// Pressing a repeating key makes it the one that repeats; pressing a key
/// that doesn't repeat (Shift) leaves it repeating, now shifted. Only
/// releasing the repeating key stops it.
#[derive(Debug, Default)]
pub struct RepeatState {
    key: Option<u32>,
}

impl RepeatState {
    pub fn press(&mut self, key: u32, repeats: bool) -> Timer {
        if repeats {
            self.key = Some(key);
            Timer::Start
        } else {
            Timer::Unchanged
        }
    }

    pub fn release(&mut self, key: u32) -> Timer {
        if self.key == Some(key) {
            self.key = None;
            Timer::Stop
        } else {
            Timer::Unchanged
        }
    }

    /// Stop repeating whatever was, as if every key were released.
    pub fn clear(&mut self) -> Timer {
        match self.key.take() {
            Some(_) => Timer::Stop,
            None => Timer::Unchanged,
        }
    }

    /// The key that's repeating.
    pub fn key(&self) -> Option<u32> {
        self.key
    }
}

/// Key repeat on a timerfd, for the event loop.
pub struct KeyRepeat {
    timer: OwnedFd,
    state: RepeatState,
}

impl KeyRepeat {
    pub fn new() -> io::Result<Self> {
        let timer = timerfd_create(
            TimerfdClockId::Monotonic,
            TimerfdFlags::NONBLOCK | TimerfdFlags::CLOEXEC,
        )?;
        Ok(Self {
            timer,
            state: RepeatState::default(),
        })
    }

    pub fn press(&mut self, key: u32, repeats: bool) -> io::Result<()> {
        let t = self.state.press(key, repeats);
        self.apply(t)
    }

    pub fn release(&mut self, key: u32) -> io::Result<()> {
        let t = self.state.release(key);
        self.apply(t)
    }

    /// Stop repeating (the session was disabled; releases won't come).
    pub fn stop(&mut self) -> io::Result<()> {
        let t = self.state.clear();
        self.apply(t)
    }

    fn apply(&self, timer: Timer) -> io::Result<()> {
        let (value, interval) = match timer {
            Timer::Unchanged => return Ok(()),
            Timer::Start => (REPEAT_DELAY, REPEAT_INTERVAL),
            // A zero value disarms the timer.
            Timer::Stop => (Duration::ZERO, Duration::ZERO),
        };
        let spec = Itimerspec {
            it_interval: timespec(interval),
            it_value: timespec(value),
        };
        timerfd_settime(&self.timer, TimerfdTimerFlags::empty(), &spec)?;
        Ok(())
    }

    /// After the timer fd is readable: the key to repeat once, if it's
    /// still held. Expirations missed while busy aren't replayed.
    pub fn expired(&mut self) -> io::Result<Option<u32>> {
        let mut count = [0u8; 8];
        match rustix::io::read(&self.timer, &mut count) {
            Ok(_) => Ok(self.state.key()),
            Err(Errno::AGAIN | Errno::INTR) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

impl AsFd for KeyRepeat {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.timer.as_fd()
    }
}

fn timespec(d: Duration) -> Timespec {
    Timespec {
        tv_sec: d.as_secs() as _,
        tv_nsec: d.subsec_nanos().into(),
    }
}

/// libinput opens and closes devices through this, so they come from the
/// seat (no root needed, and logind revokes them when we lose the VT).
struct SeatInterface(Rc<RefCell<Session>>);

impl LibinputInterface for SeatInterface {
    fn open_restricted(&mut self, path: &Path, _flags: i32) -> Result<OwnedFd, i32> {
        // The seat opens devices read-write and non-blocking, which is what
        // libinput asks for.
        self.0.borrow_mut().open_device_io(path).map_err(|e| {
            tracing::warn!(path = %path.display(), "opening input device: {e}");
            e.raw_os_error().unwrap_or(libc::EIO)
        })
    }

    fn close_restricted(&mut self, fd: OwnedFd) {
        self.0.borrow_mut().close_device(fd);
    }
}

/// A physical key going down or up, on any keyboard on the seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    /// evdev keycode (`KEY_A` = 30).
    pub code: u32,
    pub pressed: bool,
}

/// The seat's input devices, through libinput.
pub struct Devices {
    libinput: Libinput,
}

impl Devices {
    /// Start libinput on the session's seat. Devices are opened through
    /// the session; this must not be called with the session borrowed.
    pub fn new(session: Rc<RefCell<Session>>) -> Result<Self> {
        let seat = session.borrow_mut().seat_name();
        let mut libinput = Libinput::new_with_udev(SeatInterface(session));
        libinput
            .udev_assign_seat(&seat)
            .map_err(|()| anyhow!("libinput couldn't use seat {seat:?}"))?;
        let mut devices = Self { libinput };
        // Device-added events are already queued; log them now.
        devices
            .dispatch(|_| {})
            .context("first libinput dispatch")?;
        Ok(devices)
    }

    /// Read what libinput has and pass on the keys. Called when its fd is
    /// readable.
    pub fn dispatch(&mut self, mut on_key: impl FnMut(Key)) -> io::Result<()> {
        self.libinput.dispatch()?;
        for event in &mut self.libinput {
            match event {
                Event::Keyboard(KeyboardEvent::Key(k)) => {
                    let pressed = k.key_state() == KeyState::Pressed;
                    // With two keyboards, the same key can be down on both;
                    // xkb must see it go down once and up once.
                    let count = k.seat_key_count();
                    if (pressed && count == 1) || (!pressed && count == 0) {
                        on_key(Key {
                            code: k.key(),
                            pressed,
                        });
                    }
                }
                Event::Device(DeviceEvent::Added(e)) => log_device("input device added", &e),
                Event::Device(DeviceEvent::Removed(e)) => log_device("input device removed", &e),
                _ => {}
            }
        }
        Ok(())
    }

    /// Close every device through the seat, before acknowledging that the
    /// session is disabled. libinput reports them removed (logged by the
    /// next [`Devices::dispatch`]). Must not be called with the session
    /// borrowed.
    pub fn suspend(&mut self) {
        self.libinput.suspend();
    }

    /// Reopen the devices once the session is enabled again; they're
    /// reported added. Must not be called with the session borrowed.
    pub fn resume(&mut self) -> Result<()> {
        self.libinput
            .resume()
            .map_err(|()| anyhow!("libinput couldn't resume (reopening the input devices)"))
    }
}

impl AsFd for Devices {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.libinput.as_fd()
    }
}

fn log_device(what: &str, event: &impl EventTrait) {
    let dev = event.device();
    let caps = [
        (DeviceCapability::Keyboard, "keyboard"),
        (DeviceCapability::Pointer, "pointer"),
        (DeviceCapability::Touch, "touch"),
        (DeviceCapability::TabletTool, "tablet"),
        (DeviceCapability::Gesture, "gesture"),
        (DeviceCapability::Switch, "switch"),
    ]
    .into_iter()
    .filter(|&(cap, _)| dev.has_capability(cap))
    .map(|(_, name)| name)
    .collect::<Vec<_>>();
    tracing::info!(name = %dev.name(), sysname = dev.sysname(), ?caps, "{what}");
}

/// Keys vellum keeps for itself instead of sending to the shell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// Ctrl+Alt+Backspace: leave, closing the shell.
    Exit,
    /// Ctrl+Alt+F1..F12: switch to that VT.
    SwitchVt(u8),
    /// Shift+PgUp/PgDn/Home/End: move the view through the scrollback.
    /// Only on the primary screen; on the alternate screen the program gets
    /// the key.
    Scroll(Scroll),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scroll {
    /// Half a screen back.
    PageUp,
    /// Half a screen forward.
    PageDown,
    /// The oldest line.
    Top,
    /// The live screen.
    Bottom,
}

/// The binding for `event`, if it's one of vellum's own keys.
pub fn binding(event: &KeyEvent) -> Option<Binding> {
    let ctrl_alt = event.mods.contains(Mods::CTRL | Mods::ALT);
    let shift_only = event.mods.without(Mods::LOCKS) == Mods::SHIFT;
    let raw = event.sym.raw();
    let f1 = Keysym::F1.raw();
    let vt1 = Keysym::XF86_Switch_VT_1.raw();
    match event.sym {
        // With the `terminate:ctrl_alt_bksp` option, xkb makes it a keysym.
        Keysym::Terminate_Server => Some(Binding::Exit),
        Keysym::BackSpace if ctrl_alt => Some(Binding::Exit),
        // The default keymaps turn Ctrl+Alt+Fn into XF86Switch_VT_n.
        _ if (vt1..vt1 + 12).contains(&raw) => Some(Binding::SwitchVt((raw - vt1 + 1) as u8)),
        _ if ctrl_alt && (f1..f1 + 12).contains(&raw) => {
            Some(Binding::SwitchVt((raw - f1 + 1) as u8))
        }
        Keysym::Page_Up | Keysym::KP_Page_Up if shift_only => Some(Binding::Scroll(Scroll::PageUp)),
        Keysym::Page_Down | Keysym::KP_Page_Down if shift_only => {
            Some(Binding::Scroll(Scroll::PageDown))
        }
        Keysym::Home | Keysym::KP_Home if shift_only => Some(Binding::Scroll(Scroll::Top)),
        Keysym::End | Keysym::KP_End if shift_only => Some(Binding::Scroll(Scroll::Bottom)),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
