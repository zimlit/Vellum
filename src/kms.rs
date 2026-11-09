//! KMS: device discovery, output selection, framebuffer import, and the
//! save/commit/restore cycle for taking over a CRTC with atomic modesetting.

use std::cell::Cell;
use std::collections::HashMap;
use std::fs;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, bail};
use drm::buffer::{self, DrmFourcc, DrmModifier, PlanarBuffer};
use drm::control::{
    AtomicCommitFlags, Device as ControlDevice, Event, FbCmd2Flags, Mode, ModeTypeFlags,
    ResourceHandle, ResourceHandles, atomic::AtomicModeReq, connector, crtc, framebuffer, plane,
    property,
};

use crate::dmabuf::DmaBuf;

const DRM_PLANE_TYPE_PRIMARY: u64 = 1;

/// An open DRM card node.
pub struct Card(OwnedFd);

impl Card {
    pub fn new(fd: OwnedFd) -> Self {
        Self(fd)
    }

    /// Open the node directly, without a seat. Works for read-only queries and
    /// buffer/framebuffer creation if the user can open the node (`video`
    /// group); modesetting still needs DRM master.
    pub fn open_direct(path: &Path) -> Result<Self> {
        use rustix::fs::{Mode, OFlags};
        let fd = rustix::fs::open(path, OFlags::RDWR | OFlags::CLOEXEC, Mode::empty())
            .with_context(|| format!("open {}", path.display()))?;
        Ok(Self(fd))
    }
}

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl drm::Device for Card {}
impl ControlDevice for Card {}

/// Pick the card node to drive: `$VELLUM_DRM_DEVICE` if set, otherwise the
/// first card (in name order) that sysfs reports a connected connector on.
pub fn find_card() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("VELLUM_DRM_DEVICE") {
        return Ok(PathBuf::from(path));
    }
    let cards = connected_cards(Path::new("/sys/class/drm"))?;
    let card = cards
        .first()
        .ok_or_else(|| anyhow!("no DRM card with a connected display (set VELLUM_DRM_DEVICE)"))?;
    Ok(Path::new("/dev/dri").join(card))
}

/// Names of cards ("card0", ...) with at least one connected connector,
/// from a sysfs `class/drm` directory, ordered by card number.
fn connected_cards(sysfs_drm: &Path) -> Result<Vec<String>> {
    let card_number = |card: &str| card.strip_prefix("card")?.parse::<u32>().ok();
    let mut cards: Vec<String> = fs::read_dir(sysfs_drm)
        .with_context(|| format!("reading {}", sysfs_drm.display()))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            // Connector entries look like "card0-HDMI-A-1".
            let (card, _connector) = name.split_once('-')?;
            card_number(card)?;
            let status = fs::read_to_string(entry.path().join("status")).ok()?;
            (status.trim() == "connected").then(|| card.to_owned())
        })
        .collect();
    // Numeric, not lexicographic: card2 before card10.
    cards.sort_by_key(|c| card_number(c));
    cards.dedup();
    Ok(cards)
}

/// An object's properties by name: handle plus current raw value.
pub struct Props(HashMap<String, (property::Handle, u64)>);

impl Props {
    pub fn load<H: ResourceHandle>(card: &Card, handle: H) -> Result<Self> {
        let set = card
            .get_properties(handle)
            .context("DRM_IOCTL_MODE_OBJ_GETPROPERTIES")?;
        let (ids, values) = set.as_props_and_values();
        let mut map = HashMap::with_capacity(ids.len());
        for (&id, &value) in ids.iter().zip(values) {
            let info = card
                .get_property(id)
                .context("DRM_IOCTL_MODE_GETPROPERTY")?;
            map.insert(info.name().to_string_lossy().into_owned(), (id, value));
        }
        Ok(Self(map))
    }

    pub fn handle(&self, name: &str) -> Result<property::Handle> {
        self.0
            .get(name)
            .map(|&(h, _)| h)
            .ok_or_else(|| anyhow!("object has no {name:?} property"))
    }

    pub fn value(&self, name: &str) -> Option<u64> {
        self.0.get(name).map(|&(_, v)| v)
    }
}

/// A connector → CRTC → primary plane path with the mode to drive it at.
pub struct Output {
    pub connector: connector::Handle,
    pub connector_name: String,
    pub crtc: crtc::Handle,
    pub plane: plane::Handle,
    pub plane_props: Props,
    pub mode: Mode,
}

/// Find the first connected connector we can drive, and a CRTC and primary
/// plane for it. Prefers the CRTC the connector is already using (fbcon's),
/// which avoids stealing a CRTC from another display.
pub fn select_output(card: &Card) -> Result<Output> {
    let res = card
        .resource_handles()
        .context("DRM_IOCTL_MODE_GETRESOURCES")?;
    let planes = card
        .plane_handles()
        .context("DRM_IOCTL_MODE_GETPLANERESOURCES")?;

    for &conn_handle in res.connectors() {
        let mut conn = card.get_connector(conn_handle, false)?;
        if conn.state() != connector::State::Connected {
            continue;
        }
        if conn.modes().is_empty() {
            // Cached info may be stale if nothing has probed yet.
            conn = card.get_connector(conn_handle, true)?;
        }
        let Some(mode) = conn
            .modes()
            .iter()
            .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
            .or_else(|| conn.modes().first())
            .copied()
        else {
            continue;
        };
        let connector_name = format!("{}-{}", conn.interface().as_str(), conn.interface_id());

        let mut crtcs = Vec::new();
        if let Some(enc) = conn.current_encoder() {
            crtcs.extend(card.get_encoder(enc)?.crtc());
        }
        for &enc in conn.encoders() {
            crtcs.extend(res.filter_crtcs(card.get_encoder(enc)?.possible_crtcs()));
        }

        for crtc in crtcs {
            if let Some((plane, plane_props)) = find_primary_plane(card, &res, &planes, crtc)? {
                return Ok(Output {
                    connector: conn_handle,
                    connector_name,
                    crtc,
                    plane,
                    plane_props,
                    mode,
                });
            }
        }
    }
    bail!("no connected connector with a usable CRTC and primary plane")
}

fn find_primary_plane(
    card: &Card,
    res: &ResourceHandles,
    planes: &[plane::Handle],
    crtc: crtc::Handle,
) -> Result<Option<(plane::Handle, Props)>> {
    for &plane in planes {
        let info = card.get_plane(plane)?;
        if !res.filter_crtcs(info.possible_crtcs()).contains(&crtc) {
            continue;
        }
        if !info.formats().contains(&(DrmFourcc::Xrgb8888 as u32)) {
            continue;
        }
        let props = Props::load(card, plane)?;
        if props.value("type") == Some(DRM_PLANE_TYPE_PRIMARY) {
            return Ok(Some((plane, props)));
        }
    }
    Ok(None)
}

/// Modifiers the plane can scan out `format` with, from its `IN_FORMATS`
/// blob. Empty if the plane doesn't advertise any (implicit modifiers only).
pub fn plane_modifiers(card: &Card, plane_props: &Props, format: DrmFourcc) -> Result<Vec<u64>> {
    let Some(blob_id) = plane_props.value("IN_FORMATS").filter(|&id| id != 0) else {
        return Ok(Vec::new());
    };
    let blob = card
        .get_property_blob(blob_id)
        .context("reading IN_FORMATS blob")?;
    parse_in_formats(&blob, format as u32)
}

/// Parse a `struct drm_format_modifier_blob`:
///
/// ```c
/// struct drm_format_modifier_blob {
///     __u32 version, flags;
///     __u32 count_formats, formats_offset;     // -> __u32 formats[]
///     __u32 count_modifiers, modifiers_offset; // -> struct drm_format_modifier[]
/// };
/// struct drm_format_modifier {
///     __u64 formats;  // bit i: formats[offset + i] supports this modifier
///     __u32 offset, pad;
///     __u64 modifier;
/// };
/// ```
fn parse_in_formats(blob: &[u8], format: u32) -> Result<Vec<u64>> {
    let u32_at = |off: usize| -> Result<u32> {
        let bytes = blob
            .get(off..off + 4)
            .ok_or_else(|| anyhow!("IN_FORMATS blob truncated"))?;
        Ok(u32::from_ne_bytes(bytes.try_into().unwrap()))
    };
    let u64_at = |off: usize| -> Result<u64> {
        let bytes = blob
            .get(off..off + 8)
            .ok_or_else(|| anyhow!("IN_FORMATS blob truncated"))?;
        Ok(u64::from_ne_bytes(bytes.try_into().unwrap()))
    };

    let count_formats = u32_at(8)? as usize;
    let formats_offset = u32_at(12)? as usize;
    let count_modifiers = u32_at(16)? as usize;
    let modifiers_offset = u32_at(20)? as usize;

    let Some(index) = (0..count_formats)
        .map(|i| u32_at(formats_offset + 4 * i))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .position(|&f| f == format)
    else {
        return Ok(Vec::new());
    };

    let mut modifiers = Vec::new();
    for i in 0..count_modifiers {
        let base = modifiers_offset + 24 * i;
        let mask = u64_at(base)?;
        let offset = u32_at(base + 8)? as usize;
        let modifier = u64_at(base + 16)?;
        // The mask covers formats[offset .. offset + 64].
        let supported =
            index >= offset && index < offset + 64 && mask & (1 << (index - offset)) != 0;
        // Drivers may list a modifier more than once (amdgpu repeats LINEAR).
        if supported && !modifiers.contains(&modifier) {
            modifiers.push(modifier);
        }
    }
    Ok(modifiers)
}

/// GEM handles, closed on drop.
struct GemHandles<'a> {
    card: &'a Card,
    handles: Vec<buffer::Handle>,
}

impl Drop for GemHandles<'_> {
    fn drop(&mut self) {
        for &h in &self.handles {
            if let Err(e) = self.card.close_buffer(h) {
                tracing::warn!("closing GEM handle failed: {e}");
            }
        }
    }
}

/// A KMS framebuffer created from a dma-buf.
pub struct Framebuffer<'a> {
    pub handle: framebuffer::Handle,
    card: &'a Card,
    // Dropped after the framebuffer is removed (see Drop below).
    _gem: GemHandles<'a>,
}

struct Planes {
    size: (u32, u32),
    format: DrmFourcc,
    modifier: Option<DrmModifier>,
    handles: [Option<buffer::Handle>; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
}

impl PlanarBuffer for Planes {
    fn size(&self) -> (u32, u32) {
        self.size
    }
    fn format(&self) -> DrmFourcc {
        self.format
    }
    fn modifier(&self) -> Option<DrmModifier> {
        self.modifier
    }
    fn pitches(&self) -> [u32; 4] {
        self.pitches
    }
    fn handles(&self) -> [Option<buffer::Handle>; 4] {
        self.handles
    }
    fn offsets(&self) -> [u32; 4] {
        self.offsets
    }
}

impl<'a> Framebuffer<'a> {
    /// Import each dma-buf plane as a GEM handle (PRIME) and register a
    /// framebuffer with `DRM_IOCTL_MODE_ADDFB2`, with modifiers when known.
    pub fn import(card: &'a Card, buf: &DmaBuf) -> Result<Self> {
        if buf.planes.len() > 4 {
            bail!(
                "dma-buf has {} planes; KMS supports at most 4",
                buf.planes.len()
            );
        }
        let mut gem = GemHandles {
            card,
            handles: Vec::new(),
        };
        let mut planes = Planes {
            size: (buf.width, buf.height),
            format: buf.format,
            modifier: (buf.modifier != DrmModifier::Invalid).then_some(buf.modifier),
            handles: [None; 4],
            pitches: [0; 4],
            offsets: [0; 4],
        };
        for (i, plane) in buf.planes.iter().enumerate() {
            let handle = card
                .prime_fd_to_buffer(plane.fd.as_fd())
                .context("DRM_IOCTL_PRIME_FD_TO_HANDLE")?;
            // Planes of one BO import to the same handle; close it only once.
            if !gem.handles.contains(&handle) {
                gem.handles.push(handle);
            }
            planes.handles[i] = Some(handle);
            planes.pitches[i] = plane.stride;
            planes.offsets[i] = plane.offset;
        }
        let flags = if planes.modifier.is_some() {
            FbCmd2Flags::MODIFIERS
        } else {
            FbCmd2Flags::empty()
        };
        let handle = card
            .add_planar_framebuffer(&planes, flags)
            .context("DRM_IOCTL_MODE_ADDFB2")?;
        Ok(Self {
            handle,
            card,
            _gem: gem,
        })
    }
}

impl Drop for Framebuffer<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.card.destroy_framebuffer(self.handle) {
            tracing::warn!("DRM_IOCTL_MODE_RMFB failed: {e}");
        }
    }
}

/// Wrap a raw property value. The kernel takes every property as a u64; the
/// typed `property::Value` variants only exist on the Rust side.
fn raw(value: u64) -> property::Value<'static> {
    property::Value::UnsignedRange(value)
}

fn create_mode_blob(card: &Card, mode: &Mode) -> Result<u64> {
    match card
        .create_property_blob(mode)
        .context("creating MODE_ID blob")?
    {
        property::Value::Blob(id) => Ok(id),
        other => bail!("unexpected blob value {other:?}"),
    }
}

const CONNECTOR_PROPS: &[&str] = &["CRTC_ID"];
const CRTC_PROPS: &[&str] = &["ACTIVE"];
const PLANE_PROPS: &[&str] = &[
    "FB_ID", "CRTC_ID", "SRC_X", "SRC_Y", "SRC_W", "SRC_H", "CRTC_X", "CRTC_Y", "CRTC_W", "CRTC_H",
];

/// The pre-takeover values of every property we're about to change.
struct SavedState {
    connector: (connector::Handle, Vec<(property::Handle, u64)>),
    crtc: (crtc::Handle, Vec<(property::Handle, u64)>),
    plane: (plane::Handle, Vec<(property::Handle, u64)>),
    mode_id: property::Handle,
    /// Saved as the mode itself rather than the MODE_ID blob id: the old blob
    /// may be freed once our commit drops the last reference to it.
    mode: Option<Mode>,
}

fn save<H: ResourceHandle>(
    card: &Card,
    h: H,
    names: &[&str],
) -> Result<Vec<(property::Handle, u64)>> {
    let props = Props::load(card, h)?;
    names
        .iter()
        .map(|&name| {
            let value = props
                .value(name)
                .ok_or_else(|| anyhow!("missing {name:?} property"))?;
            Ok((props.handle(name)?, value))
        })
        .collect()
}

impl SavedState {
    fn capture(card: &Card, output: &Output) -> Result<Self> {
        let crtc_props = Props::load(card, output.crtc)?;
        Ok(Self {
            connector: (
                output.connector,
                save(card, output.connector, CONNECTOR_PROPS)?,
            ),
            crtc: (output.crtc, save(card, output.crtc, CRTC_PROPS)?),
            plane: (output.plane, save(card, output.plane, PLANE_PROPS)?),
            mode_id: crtc_props.handle("MODE_ID")?,
            mode: card.get_crtc(output.crtc)?.mode(),
        })
    }

    fn restore(&self, card: &Card) -> Result<()> {
        let mut req = AtomicModeReq::new();
        for &(prop, value) in &self.connector.1 {
            req.add_property(self.connector.0, prop, raw(value));
        }
        for &(prop, value) in &self.crtc.1 {
            req.add_property(self.crtc.0, prop, raw(value));
        }
        for &(prop, value) in &self.plane.1 {
            req.add_property(self.plane.0, prop, raw(value));
        }
        let blob = self
            .mode
            .as_ref()
            .map(|m| create_mode_blob(card, m))
            .transpose()?;
        req.add_property(
            self.crtc.0,
            self.mode_id,
            property::Value::Blob(blob.unwrap_or(0)),
        );

        let result = card.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req);
        if let Some(id) = blob {
            let _ = card.destroy_property_blob(id);
        }
        result.context("restore commit")?;
        Ok(())
    }
}

/// Ownership of an output's CRTC. Created by committing a framebuffer to it;
/// dropping it restores whatever was there before (typically fbcon).
pub struct Display<'a> {
    card: &'a Card,
    crtc: crtc::Handle,
    plane: plane::Handle,
    fb_id_prop: property::Handle,
    in_fence_prop: Option<property::Handle>,
    flip_pending: bool,
    saved: SavedState,
    session_active: Rc<Cell<bool>>,
}

impl<'a> Display<'a> {
    /// Save the current state, then modeset `output` to show `fb`
    /// full-screen on its primary plane.
    pub fn take_over(
        card: &'a Card,
        output: &Output,
        fb: &Framebuffer,
        session_active: Rc<Cell<bool>>,
    ) -> Result<Self> {
        let saved = SavedState::capture(card, output).context("saving CRTC state")?;

        let conn_props = Props::load(card, output.connector)?;
        let crtc_props = Props::load(card, output.crtc)?;
        let plane_props = &output.plane_props;
        let (w, h) = output.mode.size();
        let (w, h) = (u64::from(w), u64::from(h));

        let mode_blob = create_mode_blob(card, &output.mode)?;
        let mut req = AtomicModeReq::new();
        req.add_property(
            output.connector,
            conn_props.handle("CRTC_ID")?,
            property::Value::CRTC(Some(output.crtc)),
        );
        req.add_property(
            output.crtc,
            crtc_props.handle("MODE_ID")?,
            property::Value::Blob(mode_blob),
        );
        req.add_property(
            output.crtc,
            crtc_props.handle("ACTIVE")?,
            property::Value::Boolean(true),
        );
        let p = output.plane;
        req.add_property(
            p,
            plane_props.handle("FB_ID")?,
            property::Value::Framebuffer(Some(fb.handle)),
        );
        req.add_property(
            p,
            plane_props.handle("CRTC_ID")?,
            property::Value::CRTC(Some(output.crtc)),
        );
        // Source rectangle is in 16.16 fixed point, destination in pixels.
        req.add_property(p, plane_props.handle("SRC_X")?, raw(0));
        req.add_property(p, plane_props.handle("SRC_Y")?, raw(0));
        req.add_property(p, plane_props.handle("SRC_W")?, raw(w << 16));
        req.add_property(p, plane_props.handle("SRC_H")?, raw(h << 16));
        req.add_property(
            p,
            plane_props.handle("CRTC_X")?,
            property::Value::SignedRange(0),
        );
        req.add_property(
            p,
            plane_props.handle("CRTC_Y")?,
            property::Value::SignedRange(0),
        );
        req.add_property(p, plane_props.handle("CRTC_W")?, raw(w));
        req.add_property(p, plane_props.handle("CRTC_H")?, raw(h));

        // TEST_ONLY first so a rejected configuration fails before anything
        // on screen changes.
        let result = card
            .atomic_commit(
                AtomicCommitFlags::ALLOW_MODESET | AtomicCommitFlags::TEST_ONLY,
                req.clone(),
            )
            .context("atomic test commit rejected")
            .and_then(|()| {
                card.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req)
                    .context("atomic commit")
            });
        // The committed state holds its own reference to the blob.
        let _ = card.destroy_property_blob(mode_blob);
        result?;

        Ok(Self {
            card,
            crtc: output.crtc,
            plane: output.plane,
            fb_id_prop: plane_props.handle("FB_ID")?,
            in_fence_prop: plane_props.handle("IN_FENCE_FD").ok(),
            flip_pending: false,
            saved,
            session_active,
        })
    }

    /// Queue `fb` to be shown at the next vblank, without blocking. If
    /// `in_fence` (a sync_file) is given, the kernel waits for it to signal
    /// before scanning out. Completion is reported by a page-flip event; see
    /// [`Display::dispatch_events`]. Only one flip may be pending at a time.
    pub fn flip(&mut self, fb: &Framebuffer, in_fence: Option<&OwnedFd>) -> Result<()> {
        if self.flip_pending {
            bail!("a page flip is already pending");
        }
        let mut req = AtomicModeReq::new();
        req.add_property(
            self.plane,
            self.fb_id_prop,
            property::Value::Framebuffer(Some(fb.handle)),
        );
        if let Some(fence) = in_fence {
            let prop = self
                .in_fence_prop
                .ok_or_else(|| anyhow!("plane has no IN_FENCE_FD property"))?;
            // The kernel takes its own reference during the ioctl; the fd can
            // be closed as soon as the commit returns.
            req.add_property(
                self.plane,
                prop,
                property::Value::SignedRange(i64::from(fence.as_raw_fd())),
            );
        }
        self.card
            .atomic_commit(
                AtomicCommitFlags::NONBLOCK | AtomicCommitFlags::PAGE_FLIP_EVENT,
                req,
            )
            .context("page flip commit")?;
        self.flip_pending = true;
        Ok(())
    }

    /// Read pending DRM events from the card fd (call when it's readable).
    /// Returns true if our pending flip completed, i.e. the last flipped
    /// framebuffer is now on screen and the previous one is free to reuse.
    pub fn dispatch_events(&mut self) -> Result<bool> {
        let mut flipped = false;
        for event in self.card.receive_events().context("reading DRM events")? {
            if let Event::PageFlip(e) = event
                && e.crtc == self.crtc
            {
                flipped = true;
            }
        }
        if flipped {
            self.flip_pending = false;
        }
        Ok(flipped)
    }

    /// Block (up to a timeout) until a pending flip completes. A new commit
    /// while one is in flight fails with EBUSY, including the restore commit.
    fn wait_for_pending_flip(&mut self) -> Result<()> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while self.flip_pending {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                bail!("timed out waiting for page flip");
            }
            let timeout = rustix::event::Timespec {
                tv_sec: left.as_secs() as _,
                tv_nsec: left.subsec_nanos() as _,
            };
            let mut fds = [rustix::event::PollFd::new(
                self.card,
                rustix::event::PollFlags::IN,
            )];
            match rustix::event::poll(&mut fds, Some(&timeout)) {
                Ok(0) | Err(rustix::io::Errno::INTR) => continue,
                Ok(_) => {
                    self.dispatch_events()?;
                }
                Err(e) => return Err(e).context("poll"),
            }
        }
        Ok(())
    }
}

impl Drop for Display<'_> {
    fn drop(&mut self) {
        // If the session was disabled we no longer hold DRM master; whoever
        // owns the VT now owns the display.
        if !self.session_active.get() {
            tracing::debug!("session inactive; leaving display state to the new VT owner");
            return;
        }
        if let Err(e) = self.wait_for_pending_flip() {
            tracing::warn!("{e:#}");
        }
        match self.saved.restore(self.card) {
            Ok(()) => tracing::debug!("restored previous display state"),
            Err(e) => tracing::error!("failed to restore display state: {e:#}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gbm;
    use crate::testutil::{self, TempDir};

    // --- connected_cards -------------------------------------------------

    fn connector(sysfs: &Path, name: &str, status: &str) {
        let dir = sysfs.join(name);
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("status"), format!("{status}\n")).unwrap();
    }

    #[test]
    fn connected_cards_returns_cards_with_a_connected_connector() {
        let sysfs = TempDir::new();
        connector(sysfs.path(), "card0-DP-1", "disconnected");
        connector(sysfs.path(), "card0-Writeback-1", "unknown");
        connector(sysfs.path(), "card1-HDMI-A-1", "connected");
        connector(sysfs.path(), "card1-DP-2", "connected");
        assert_eq!(connected_cards(sysfs.path()).unwrap(), ["card1"]);
    }

    #[test]
    fn connected_cards_sorts_by_card_number() {
        let sysfs = TempDir::new();
        connector(sysfs.path(), "card10-DP-1", "connected");
        connector(sysfs.path(), "card2-DP-1", "connected");
        connector(sysfs.path(), "card1-DP-1", "connected");
        assert_eq!(
            connected_cards(sysfs.path()).unwrap(),
            ["card1", "card2", "card10"]
        );
    }

    #[test]
    fn connected_cards_ignores_non_connector_entries() {
        let sysfs = TempDir::new();
        // Nodes and files that live in /sys/class/drm alongside connectors.
        fs::create_dir(sysfs.path().join("card0")).unwrap();
        fs::create_dir(sysfs.path().join("renderD128")).unwrap();
        fs::write(sysfs.path().join("version"), "drm 1.1.0\n").unwrap();
        // Look like connectors but aren't cardN-*.
        connector(sysfs.path(), "cardX-DP-1", "connected");
        connector(sysfs.path(), "renderD128-foo", "connected");
        // Connector without a status file.
        fs::create_dir(sysfs.path().join("card3-DP-1")).unwrap();
        assert!(connected_cards(sysfs.path()).unwrap().is_empty());
    }

    #[test]
    fn connected_cards_errors_on_missing_directory() {
        let sysfs = TempDir::new();
        assert!(connected_cards(&sysfs.path().join("nope")).is_err());
    }

    // --- parse_in_formats ------------------------------------------------

    const XR24: u32 = DrmFourcc::Xrgb8888 as u32;
    const AR24: u32 = DrmFourcc::Argb8888 as u32;
    const NV12: u32 = DrmFourcc::Nv12 as u32;
    const MOD_A: u64 = 0x0100_0000_0000_0001;
    const MOD_B: u64 = 0x0200_0000_0000_0002;

    /// Build a `drm_format_modifier_blob` the way the kernel lays it out.
    /// `modifiers` entries are `(format mask, mask offset, modifier)`.
    fn in_formats_blob(formats: &[u32], modifiers: &[(u64, u32, u64)]) -> Vec<u8> {
        const HEADER: usize = 24;
        let formats_offset = HEADER;
        // The modifier array is 8-byte aligned.
        let modifiers_offset = (formats_offset + 4 * formats.len()).next_multiple_of(8);
        let mut blob = Vec::new();
        for v in [
            1, // version
            0, // flags
            formats.len() as u32,
            formats_offset as u32,
            modifiers.len() as u32,
            modifiers_offset as u32,
        ] {
            blob.extend_from_slice(&u32::to_ne_bytes(v));
        }
        for f in formats {
            blob.extend_from_slice(&f.to_ne_bytes());
        }
        blob.resize(modifiers_offset, 0);
        for &(mask, offset, modifier) in modifiers {
            blob.extend_from_slice(&mask.to_ne_bytes());
            blob.extend_from_slice(&offset.to_ne_bytes());
            blob.extend_from_slice(&0u32.to_ne_bytes()); // pad
            blob.extend_from_slice(&modifier.to_ne_bytes());
        }
        blob
    }

    #[test]
    fn in_formats_selects_modifiers_whose_mask_covers_the_format() {
        let blob = in_formats_blob(
            &[XR24, AR24, NV12],
            &[
                (0b011, 0, 0),     // LINEAR: XR24 + AR24
                (0b010, 0, MOD_A), // AR24 only
                (0b111, 0, MOD_B), // all three
            ],
        );
        assert_eq!(parse_in_formats(&blob, XR24).unwrap(), [0, MOD_B]);
        assert_eq!(parse_in_formats(&blob, AR24).unwrap(), [0, MOD_A, MOD_B]);
        assert_eq!(parse_in_formats(&blob, NV12).unwrap(), [MOD_B]);
    }

    #[test]
    fn in_formats_unlisted_format_has_no_modifiers() {
        let blob = in_formats_blob(&[AR24], &[(0b1, 0, 0)]);
        assert!(parse_in_formats(&blob, XR24).unwrap().is_empty());
    }

    #[test]
    fn in_formats_respects_mask_offset() {
        // 70 formats; ours is at index 66, beyond the first 64-bit window.
        let mut formats: Vec<u32> = (0..70).map(|i| 0x1000 + i).collect();
        formats[66] = XR24;
        let blob = in_formats_blob(
            &formats,
            &[
                (u64::MAX, 0, MOD_A), // covers indices 0..64 only
                (1 << 2, 64, MOD_B),  // bit 2 of window 64.. = index 66
            ],
        );
        assert_eq!(parse_in_formats(&blob, XR24).unwrap(), [MOD_B]);
    }

    #[test]
    fn in_formats_deduplicates_repeated_modifiers() {
        // amdgpu lists LINEAR twice.
        let blob = in_formats_blob(&[XR24], &[(1, 0, 0), (1, 0, MOD_A), (1, 0, 0)]);
        assert_eq!(parse_in_formats(&blob, XR24).unwrap(), [0, MOD_A]);
    }

    #[test]
    fn in_formats_rejects_truncated_blobs() {
        let blob = in_formats_blob(&[XR24], &[(1, 0, MOD_A)]);
        // Shorter than the header.
        assert!(parse_in_formats(&blob[..16], XR24).is_err());
        // Header intact, modifier array cut off.
        assert!(parse_in_formats(&blob[..blob.len() - 4], XR24).is_err());
    }

    #[test]
    fn in_formats_rejects_out_of_range_offsets() {
        let mut blob = in_formats_blob(&[XR24], &[(1, 0, MOD_A)]);
        // formats_offset far past the end.
        blob[12..16].copy_from_slice(&u32::MAX.to_ne_bytes());
        assert!(parse_in_formats(&blob, XR24).is_err());
    }

    // --- hardware --------------------------------------------------------

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn select_output_finds_a_primary_plane_and_mode() {
        let card = testutil::card();
        let output = select_output(&card).unwrap();
        assert_eq!(
            output.plane_props.value("type"),
            Some(DRM_PLANE_TYPE_PRIMARY)
        );
        let (w, h) = output.mode.size();
        assert!(w > 0 && h > 0);
        assert!(!output.connector_name.is_empty());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn props_exposes_atomic_crtc_properties() {
        let card = testutil::card();
        let output = select_output(&card).unwrap();
        let props = Props::load(&card, output.crtc).unwrap();
        assert!(props.handle("ACTIVE").is_ok());
        assert!(props.handle("MODE_ID").is_ok());
        assert!(props.handle("NOT_A_PROPERTY").is_err());
        assert_eq!(props.value("NOT_A_PROPERTY"), None);
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn framebuffer_import_registers_and_removes_fb() {
        let card = testutil::card();
        let output = select_output(&card).unwrap();
        let modifiers = plane_modifiers(&card, &output.plane_props, DrmFourcc::Xrgb8888).unwrap();
        let gbm = gbm::Device::new(card.as_fd()).unwrap();
        let bo = gbm
            .create_bo(
                256,
                128,
                DrmFourcc::Xrgb8888,
                &modifiers,
                gbm::BoFlags::SCANOUT,
            )
            .unwrap();
        let dmabuf = bo.export().unwrap();

        let fb = Framebuffer::import(&card, &dmabuf).unwrap();
        let handle = fb.handle;
        let info = card.get_framebuffer(handle).unwrap();
        assert_eq!(info.size(), (256, 128));
        drop(fb);
        assert!(card.get_framebuffer(handle).is_err(), "RMFB on drop");
    }
}
