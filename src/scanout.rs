//! Scanout buffers: memory the display engine shows and Vulkan renders into.
//!
//! Pick a connector/mode/CRTC/primary plane, and allocate buffers with GBM
//! using a format modifier both the plane and Vulkan support. Each buffer is
//! exported as a dma-buf and imported twice: as a KMS framebuffer and as a
//! Vulkan image. Rendering signals a semaphore that is exported as a
//! sync_file, which the page flip waits on as `IN_FENCE_FD`.

use std::os::fd::OwnedFd;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use drm::buffer::DrmFourcc;
use drm::{ClientCapability, Device as DrmDevice, DriverCapability};

use crate::gbm;
use crate::kms::{self, Card, Framebuffer, Output};
use crate::vk;

pub const FORMAT: DrmFourcc = DrmFourcc::Xrgb8888;
pub const USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;

/// One scanout buffer: the same dma-buf seen by KMS and by Vulkan, plus the
/// per-buffer GPU submission state.
pub struct Buffer<'a> {
    pub image: Arc<vk::DmaBufImage>,
    pub fb: Framebuffer<'a>,
    pub cmd: vk::CommandBuffer,
    pub rendered: Arc<vk::Semaphore>,
    // The dma-buf keeps the memory alive on its own; holding the BO just
    // keeps GBM's view of it valid for the buffer's lifetime.
    pub bo: gbm::Bo<'a>,
}

/// Enable atomic modesetting and choose the output.
pub fn prepare(card: &Card) -> Result<Output> {
    card.set_client_capability(ClientCapability::Atomic, true)
        .context("enabling DRM_CLIENT_CAP_ATOMIC (driver lacks atomic modesetting?)")?;
    let output = kms::select_output(card)?;
    let (w, h) = output.mode.size();
    tracing::info!(
        connector = %output.connector_name,
        mode = %format_args!("{w}x{h}@{}", output.mode.vrefresh()),
        crtc = ?output.crtc,
        plane = ?output.plane,
        "selected output"
    );
    Ok(output)
}

/// The Vulkan device on the same GPU as `card`.
pub fn vulkan_device(instance: &Arc<vk::Instance>, card: &Card) -> Result<Arc<vk::Device>> {
    let rdev = rustix::fs::fstat(card)?.st_rdev;
    let (major, minor) = (rustix::fs::major(rdev), rustix::fs::minor(rdev));
    let device = vk::Device::for_drm_node(instance, major, minor)?;
    tracing::info!(gpu = device.name(), drm_node = %format_args!("{major}:{minor}"), "created Vulkan device");
    Ok(device)
}

/// Modifiers both the primary plane can scan out and Vulkan can import for
/// rendering, in the plane's order of preference.
pub fn negotiate_modifiers(card: &Card, output: &Output, device: &vk::Device) -> Result<Vec<u64>> {
    let explicit = card
        .get_driver_capability(DriverCapability::AddFB2Modifiers)
        .is_ok_and(|v| v != 0);
    if !explicit {
        bail!("KMS driver doesn't support framebuffer modifiers");
    }
    let kms = kms::plane_modifiers(card, &output.plane_props, FORMAT)?;
    let vulkan = device.dmabuf_modifiers(FORMAT, USAGE)?;
    for &m in &kms {
        let vk = vulkan.iter().find(|v| v.modifier == m);
        tracing::debug!(
            modifier = %format_args!("{m:#018x}"),
            vulkan_planes = vk.map(|v| v.plane_count),
            "plane modifier{}",
            if vk.is_some() { "" } else { " (not importable by Vulkan)" }
        );
    }
    let vulkan_mods: Vec<u64> = vulkan.iter().map(|v| v.modifier).collect();
    let common = intersect_modifiers(&kms, &vulkan_mods);
    tracing::info!(
        plane = kms.len(),
        vulkan = vulkan.len(),
        common = common.len(),
        "negotiated format modifiers"
    );
    if common.is_empty() {
        bail!(
            "no format modifier is supported by both the plane ({} candidates) and Vulkan ({})",
            kms.len(),
            vulkan.len()
        );
    }
    Ok(common)
}

/// Modifiers in both lists, in `preferred`'s order.
fn intersect_modifiers(preferred: &[u64], other: &[u64]) -> Vec<u64> {
    preferred
        .iter()
        .copied()
        .filter(|m| other.contains(m))
        .collect()
}

pub fn create_buffer<'a>(
    card: &'a Card,
    gbm: &'a gbm::Device<'_>,
    device: &Arc<vk::Device>,
    output: &Output,
    modifiers: &[u64],
) -> Result<Buffer<'a>> {
    let (w, h) = output.mode.size();
    let bo = gbm
        .create_bo(
            u32::from(w),
            u32::from(h),
            FORMAT,
            modifiers,
            gbm::BoFlags::SCANOUT | gbm::BoFlags::RENDERING,
        )
        .context("gbm_bo_create_with_modifiers2")?;
    let dmabuf = bo.export().context("exporting dma-buf")?;
    tracing::debug!(
        size = %format_args!("{}x{}", dmabuf.width, dmabuf.height),
        format = ?dmabuf.format,
        modifier = %format_args!("{:#018x}", u64::from(dmabuf.modifier)),
        planes = ?dmabuf.planes.iter().map(|p| (p.offset, p.stride)).collect::<Vec<_>>(),
        "allocated scanout buffer (planes as (offset, stride))"
    );
    Ok(Buffer {
        image: vk::DmaBufImage::import(device, &dmabuf, USAGE)?,
        fb: Framebuffer::import(card, &dmabuf)?,
        cmd: vk::CommandBuffer::new(device)?,
        rendered: vk::Semaphore::new_exportable(device)?,
        bo,
    })
}

/// Record and submit this buffer's frame: `draw` renders into the image,
/// which is in `COLOR_ATTACHMENT_OPTIMAL` and owned by Vulkan for the call.
/// Returns a sync_file that signals when rendering is done, or `None` if it
/// already is.
pub fn render(
    buf: &mut Buffer,
    draw: impl FnOnce(&mut vk::Recorder, &Arc<vk::DmaBufImage>) -> Result<()>,
) -> Result<Option<OwnedFd>> {
    let mut rec = buf.cmd.begin()?;
    rec.acquire_from_foreign(&buf.image);
    draw(&mut rec, &buf.image)?;
    rec.release_to_foreign(&buf.image);
    rec.submit(Some(&buf.rendered))?;
    buf.rendered.export_sync_file()
}

/// Render into `buf` and queue a flip to it, fenced on the rendering.
pub fn queue_frame(
    display: &mut kms::Display,
    buf: &mut Buffer,
    draw: impl FnOnce(&mut vk::Recorder, &Arc<vk::DmaBufImage>) -> Result<()>,
) -> Result<kms::Flip> {
    let fence = render(buf, draw)?;
    display.flip(&buf.fb, fence.as_ref())
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsFd;

    use rustix::event::{PollFd, PollFlags, Timespec};

    use super::*;
    use crate::testutil;

    #[test]
    fn intersect_modifiers_keeps_preferred_order() {
        assert_eq!(intersect_modifiers(&[5, 3, 1, 0], &[0, 1, 5, 9]), [5, 1, 0]);
        assert!(intersect_modifiers(&[1, 2], &[3, 4]).is_empty());
        assert!(intersect_modifiers(&[], &[1]).is_empty());
    }

    /// The real setup path minus the modeset: output selection, modifier
    /// negotiation, GBM allocation, KMS + Vulkan import, and rendering with
    /// an exported fence that actually signals.
    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn scanout_buffer_renders_and_signals_fence() {
        let card = testutil::card();
        let output = prepare(&card).unwrap();
        let instance = vk::Instance::new(cfg!(debug_assertions)).unwrap();
        let device = vulkan_device(&instance, &card).unwrap();
        let gbm = gbm::Device::new(card.as_fd()).unwrap();
        let modifiers = negotiate_modifiers(&card, &output, &device).unwrap();
        assert!(!modifiers.is_empty());
        let mut buf = create_buffer(&card, &gbm, &device, &output, &modifiers).unwrap();
        assert!(modifiers.contains(&buf.bo.modifier()));

        for i in 0..3 {
            let clear = |rec: &mut vk::Recorder, image: &Arc<vk::DmaBufImage>| {
                rec.clear(image, [0.1 * i as f32, 0.0, 0.0, 1.0])
            };
            if let Some(fence) = render(&mut buf, clear).unwrap() {
                let mut fds = [PollFd::new(&fence, PollFlags::IN)];
                let timeout = Timespec {
                    tv_sec: 5,
                    tv_nsec: 0,
                };
                assert_eq!(rustix::event::poll(&mut fds, Some(&timeout)).unwrap(), 1);
            }
        }
        buf.cmd.wait().unwrap();
    }
}
