//! Safe wrapper over the parts of Vulkan (via ash) that vellum uses.
//!
//! All `unsafe` Vulkan calls live in this module. The public API is safe:
//! - Objects hold an `Arc` of their parent (image → device → instance), so a
//!   parent is never destroyed before its children.
//! - A [`CommandBuffer`] keeps every resource it references alive until the
//!   GPU has finished with it (tracked by its own fence), and waits for that
//!   fence before it is re-recorded or destroyed.
//! - [`Device`] waits for the GPU to go idle before it is destroyed.

mod command;
mod device;
mod image;
mod instance;
mod sync;

pub use ash::vk::ImageUsageFlags;
pub use command::CommandBuffer;
pub use command::Recorder;
pub use device::Device;
pub use image::DmaBufImage;
pub use image::Modifier;
pub use instance::Instance;
pub use sync::Semaphore;

/// Tests needing a real GPU. Pure helpers are tested next to their code.
#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;
    use std::time::Duration;

    use drm::buffer::{DrmFourcc, DrmModifier};
    use rustix::event::{PollFd, PollFlags, Timespec};

    use super::*;
    use crate::dmabuf::{DmaBuf, DmaBufPlane};
    use crate::testutil;

    fn device() -> Arc<Device> {
        let card = testutil::card();
        let rdev = rustix::fs::fstat(&card).unwrap().st_rdev;
        let instance = Instance::new(cfg!(debug_assertions)).unwrap();
        Device::for_drm_node(&instance, rustix::fs::major(rdev), rustix::fs::minor(rdev)).unwrap()
    }

    fn wait_readable(fd: &impl std::os::fd::AsFd, timeout: Duration) -> bool {
        let mut fds = [PollFd::new(fd, PollFlags::IN)];
        let timeout = Timespec {
            tv_sec: timeout.as_secs() as _,
            tv_nsec: timeout.subsec_nanos() as _,
        };
        rustix::event::poll(&mut fds, Some(&timeout)).unwrap() == 1
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn instance_creates_with_and_without_validation() {
        Instance::new(false).unwrap();
        // Falls back to no validation if the layer isn't installed.
        Instance::new(true).unwrap();
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn device_is_found_for_the_kms_card() {
        assert!(!device().name().is_empty());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn unknown_drm_node_is_an_error() {
        let instance = Instance::new(false).unwrap();
        assert!(Device::for_drm_node(&instance, 4095, 4095).is_err());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn xrgb_color_attachment_has_importable_modifiers() {
        let mods = device()
            .dmabuf_modifiers(DrmFourcc::Xrgb8888, ImageUsageFlags::COLOR_ATTACHMENT)
            .unwrap();
        assert!(!mods.is_empty());
        assert!(mods.iter().all(|m| m.plane_count >= 1));
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn modifiers_for_unmapped_fourcc_is_an_error() {
        let result = device().dmabuf_modifiers(DrmFourcc::Nv12, ImageUsageFlags::COLOR_ATTACHMENT);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn import_rejects_unmapped_fourcc_before_touching_vulkan() {
        let buf = DmaBuf {
            width: 16,
            height: 16,
            format: DrmFourcc::Nv12,
            modifier: DrmModifier::Linear,
            planes: vec![DmaBufPlane {
                fd: File::open("/dev/null").unwrap().into(),
                offset: 0,
                stride: 16,
            }],
        };
        assert!(DmaBufImage::import(&device(), &buf, ImageUsageFlags::COLOR_ATTACHMENT).is_err());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn command_buffer_can_be_recorded_repeatedly() {
        let device = device();
        let mut cmd = CommandBuffer::new(&device).unwrap();
        cmd.wait().unwrap(); // nothing submitted yet: no-op
        for _ in 0..3 {
            cmd.begin().unwrap().submit(None).unwrap();
        }
        cmd.wait().unwrap();
        // Abandoning a recording is fine too; the next begin resets it.
        let _ = cmd.begin().unwrap();
        cmd.begin().unwrap().submit(None).unwrap();
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn exporting_semaphore_without_pending_signal_fails() {
        let sem = Semaphore::new_exportable(&device()).unwrap();
        assert!(sem.export_sync_file().is_err());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn semaphore_cannot_be_signaled_twice_without_export() {
        let device = device();
        let sem = Semaphore::new_exportable(&device).unwrap();
        let mut a = CommandBuffer::new(&device).unwrap();
        let mut b = CommandBuffer::new(&device).unwrap();

        a.begin().unwrap().submit(Some(&sem)).unwrap();
        assert!(b.begin().unwrap().submit(Some(&sem)).is_err());

        // Export consumes the pending signal, so it can be signaled again.
        sem.export_sync_file().unwrap();
        b.begin().unwrap().submit(Some(&sem)).unwrap();
        sem.export_sync_file().unwrap();
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn exported_sync_file_signals_when_work_completes() {
        let device = device();
        let sem = Semaphore::new_exportable(&device).unwrap();
        let mut cmd = CommandBuffer::new(&device).unwrap();
        cmd.begin().unwrap().submit(Some(&sem)).unwrap();
        // None means it had already signaled, which is also correct.
        if let Some(fence) = sem.export_sync_file().unwrap() {
            assert!(
                wait_readable(&fence, Duration::from_secs(5)),
                "sync_file never signaled"
            );
        }
        cmd.wait().unwrap();
    }
}
