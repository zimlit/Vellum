//! Safe wrapper over the parts of Vulkan (via ash) that vellum uses.
//!
//! All `unsafe` Vulkan calls live in this module. The public API is safe:
//! - Objects hold an `Arc` of their parent (image → device → instance), so a
//!   parent is never destroyed before its children.
//! - A [`CommandBuffer`] keeps every resource it references alive until the
//!   GPU has finished with it (tracked by its own fence), and waits for that
//!   fence before it is re-recorded or destroyed.
//! - [`Device`] waits for the GPU to go idle before it is destroyed.
//!
//! GPU-side correctness (barriers, image layouts) is not enforced by types;
//! the validation layers check it in debug builds. See [`pipeline`] for how
//! shaders get their inputs.

mod buffer;
mod command;
mod device;
mod image;
mod instance;
mod memory;
mod pipeline;
mod sync;

pub use ash::vk::{BufferUsageFlags, Format, ImageLayout, ImageUsageFlags};
pub use buffer::{Buffer, Pod, bytes_of};
pub use command::{Access, CommandBuffer, Recorder};
pub use device::Device;
pub use image::{DmaBufImage, Image, Modifier, RenderTarget};
pub use instance::Instance;
pub use memory::Location;
pub use pipeline::{Blend, ComputePipeline, DescriptorSet, GraphicsPipeline, Layout};
pub use sync::Semaphore;

/// The SPIR-V for a shader in `shaders/`, compiled by `build.rs`, e.g.
/// `spirv!("cell.vert")`.
#[macro_export]
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name, ".spv"))
    };
}

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

    /// The device for the KMS card. When dropped (at the end of a test that
    /// didn't otherwise fail), asserts that the validation layers reported
    /// no errors. (The count is process-wide, so with tests running in
    /// parallel the failure may land on a neighbor; the message is printed
    /// by the test that caused it.)
    struct TestDevice(Arc<Device>);

    impl std::ops::Deref for TestDevice {
        type Target = Arc<Device>;
        fn deref(&self) -> &Arc<Device> {
            &self.0
        }
    }

    impl Drop for TestDevice {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(instance::validation_errors(), 0, "Vulkan validation errors");
            }
        }
    }

    fn device() -> TestDevice {
        let card = testutil::card();
        let rdev = rustix::fs::fstat(&card).unwrap().st_rdev;
        let instance = Instance::new(cfg!(debug_assertions)).unwrap();
        TestDevice(
            Device::for_drm_node(&instance, rustix::fs::major(rdev), rustix::fs::minor(rdev))
                .unwrap(),
        )
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

    // ---- Buffers, pipelines, rendering ----

    fn upload(device: &Arc<Device>, size: u64) -> Arc<Buffer> {
        Buffer::new(
            device,
            size,
            BufferUsageFlags::TRANSFER_DST,
            Location::Upload,
        )
        .unwrap()
    }

    fn readback(device: &Arc<Device>, size: u64) -> Arc<Buffer> {
        Buffer::new(
            device,
            size,
            BufferUsageFlags::TRANSFER_DST,
            Location::Readback,
        )
        .unwrap()
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn mapped_buffer_round_trips_and_checks_bounds() {
        let device = device();
        let buf = upload(&device, 64);
        assert_ne!(buf.address(), 0);
        buf.write(8, &[1u32, 2, 3]).unwrap();
        assert_eq!(buf.read::<u32>(8, 3).unwrap(), [1, 2, 3]);
        assert!(buf.write(60, &[0u32, 0]).is_err(), "past the end");
        assert!(buf.read::<u32>(u64::MAX, 1).is_err(), "offset overflow");
        let vram = Buffer::new(&device, 64, BufferUsageFlags::empty(), Location::Device).unwrap();
        assert!(vram.read::<u8>(0, 1).is_err(), "device-local isn't mapped");
        assert!(Buffer::new(&device, 0, BufferUsageFlags::empty(), Location::Device).is_err());
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FillPush {
        out: u64,
        count: u32,
        mul: u32,
    }
    unsafe impl Pod for FillPush {}

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn compute_writes_through_a_buffer_address() {
        let device = device();
        let layout = Layout::new(&device, 0, 16).unwrap();
        let pipeline = ComputePipeline::new(&layout, spirv!("test/fill.comp")).unwrap();
        let count = 1000u32;
        let out = readback(&device, u64::from(count) * 4);
        let mut cmd = CommandBuffer::new(&device).unwrap();
        let mut rec = cmd.begin().unwrap();
        rec.fill(&out, 0xdead_beef);
        rec.barrier(Access::TRANSFER_WRITE, Access::COMPUTE_WRITE);
        let push = FillPush {
            out: out.address(),
            count: count - 1, // leave the last word as filled
            mul: 3,
        };
        rec.dispatch(&pipeline, None, &push, [count.div_ceil(64), 1, 1])
            .unwrap();
        rec.using(&[&out]);
        rec.barrier(Access::COMPUTE_WRITE, Access::HOST_READ);
        rec.submit(None).unwrap();
        cmd.wait().unwrap();

        let data = out.read::<u32>(0, count as usize).unwrap();
        for (i, &v) in data[..count as usize - 1].iter().enumerate() {
            assert_eq!(v, i as u32 * 3);
        }
        assert_eq!(data[count as usize - 1], 0xdead_beef);
    }

    /// Run `record` on a fresh command buffer, then copy `image` (left in
    /// `from` layout by `record`) to the CPU as RGBA bytes.
    fn render_and_read(
        device: &Arc<Device>,
        image: &Arc<Image>,
        from: ImageLayout,
        src: Access,
        record: impl FnOnce(&mut Recorder),
    ) -> Vec<[u8; 4]> {
        let texels = (image.width() * image.height()) as usize;
        let out = readback(device, texels as u64 * 4);
        let mut cmd = CommandBuffer::new(device).unwrap();
        let mut rec = cmd.begin().unwrap();
        record(&mut rec);
        rec.transition(
            image,
            from,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            src,
            Access::TRANSFER_READ,
        );
        rec.copy_image_to_buffer(image, &out).unwrap();
        rec.barrier(Access::TRANSFER_WRITE, Access::HOST_READ);
        rec.submit(None).unwrap();
        cmd.wait().unwrap();
        out.read(0, texels).unwrap()
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn compute_writes_a_storage_image() {
        let device = device();
        let image = Image::new(
            &device,
            20,
            12,
            Format::R8G8B8A8_UNORM,
            ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
        let layout = Layout::new(&device, 1, 0).unwrap();
        let set = DescriptorSet::new(&layout, &[&image]).unwrap();
        let pipeline = ComputePipeline::new(&layout, spirv!("test/image_write.comp")).unwrap();
        let pixels = render_and_read(
            &device,
            &image,
            ImageLayout::GENERAL,
            Access::COMPUTE_WRITE,
            |rec| {
                rec.transition(
                    &image,
                    ImageLayout::UNDEFINED,
                    ImageLayout::GENERAL,
                    Access::NONE,
                    Access::COMPUTE_WRITE,
                );
                rec.dispatch(&pipeline, Some(&set), &(), [3, 2, 1]).unwrap();
            },
        );
        for y in 0..12 {
            for x in 0..20 {
                assert_eq!(pixels[y * 20 + x], [x as u8, y as u8, 0, 255], "at {x},{y}");
            }
        }
    }

    fn solid_pipeline(device: &Arc<Device>, blend: Blend) -> Arc<GraphicsPipeline> {
        let layout = Layout::new(device, 0, 16).unwrap();
        GraphicsPipeline::new(
            &layout,
            spirv!("test/solid.vert"),
            spirv!("test/solid.frag"),
            Format::R8G8B8A8_UNORM,
            blend,
        )
        .unwrap()
    }

    fn target(device: &Arc<Device>) -> Arc<Image> {
        Image::new(
            device,
            8,
            8,
            Format::R8G8B8A8_UNORM,
            ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
        )
        .unwrap()
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn draw_fills_target_and_blends_premultiplied() {
        let device = device();
        let replace = solid_pipeline(&device, Blend::Replace);
        let over = solid_pipeline(&device, Blend::PremultipliedAlpha);
        let image = target(&device);
        let pixels = render_and_read(
            &device,
            &image,
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Access::COLOR_ATTACHMENT,
            |rec| {
                rec.transition(
                    &image,
                    ImageLayout::UNDEFINED,
                    ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    Access::NONE,
                    Access::COLOR_ATTACHMENT,
                );
                rec.begin_rendering(&image, Some([0.0, 0.0, 1.0, 1.0]))
                    .unwrap();
                rec.draw(&replace, None, &[1.0f32, 0.0, 0.0, 1.0], 3, 1)
                    .unwrap();
                // Half-transparent green over red, premultiplied.
                rec.draw(&over, None, &[0.0f32, 0.5, 0.0, 0.5], 3, 1)
                    .unwrap();
                rec.end_rendering().unwrap();
            },
        );
        for p in pixels {
            assert!(p[0].abs_diff(128) <= 1 && p[1].abs_diff(128) <= 1, "{p:?}");
            assert_eq!((p[2], p[3]), (0, 255));
        }
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn recorder_rejects_misuse() {
        let device = device();
        let pipeline = solid_pipeline(&device, Blend::Replace);
        let image = target(&device);
        let mut cmd = CommandBuffer::new(&device).unwrap();
        let mut rec = cmd.begin().unwrap();
        let color = [0.0f32; 4];
        assert!(
            rec.draw(&pipeline, None, &color, 3, 1).is_err(),
            "outside rendering"
        );
        assert!(rec.end_rendering().is_err());
        rec.begin_rendering(&image, None).unwrap();
        assert!(rec.begin_rendering(&image, None).is_err(), "nested");
        assert!(
            rec.draw(&pipeline, None, &[0.0f32; 8], 3, 1).is_err(),
            "push too large"
        );
        let other = Layout::new(&device, 0, 16).unwrap();
        let set = DescriptorSet::new(&other, &[]).unwrap();
        assert!(
            rec.draw(&pipeline, Some(&set), &color, 3, 1).is_err(),
            "foreign set"
        );
        // A recording abandoned mid-pass is reset by the next begin.
        let _ = rec;
        cmd.begin().unwrap().submit(None).unwrap();

        let layout = Layout::new(&device, 1, 0).unwrap();
        assert!(DescriptorSet::new(&layout, &[]).is_err(), "image count");
        assert!(Layout::new(&device, 0, 132).is_err());
        assert!(Layout::new(&device, 0, 6).is_err());
        assert!(ComputePipeline::new(&layout, b"not spir-v").is_err());
    }
}
