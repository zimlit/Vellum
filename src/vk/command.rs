use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, bail};
use ash::vk;

use super::image::COLOR_RANGE;
use super::sync::Fence;
use super::{Device, DmaBufImage, Semaphore};

/// A primary command buffer with its own pool and completion fence.
///
/// Resources recorded into it are kept alive until the submission completes,
/// and it waits for its previous submission before being recorded again or
/// dropped, so nothing it references can be freed while the GPU uses it.
pub struct CommandBuffer {
    pool: vk::CommandPool,
    raw: vk::CommandBuffer,
    fence: Fence,
    pending: bool,
    keep_alive: Vec<Arc<dyn Any + Send + Sync>>,
    device: Arc<Device>,
}

impl CommandBuffer {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let dev = &device.raw;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool =
            unsafe { dev.create_command_pool(&pool_info, None) }.context("vkCreateCommandPool")?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let raw = match unsafe { dev.allocate_command_buffers(&alloc) } {
            Ok(buffers) => buffers[0],
            Err(e) => {
                unsafe { dev.destroy_command_pool(pool, None) };
                return Err(e).context("vkAllocateCommandBuffers");
            }
        };
        let fence = match Fence::new(device) {
            Ok(f) => f,
            Err(e) => {
                unsafe { dev.destroy_command_pool(pool, None) };
                return Err(e);
            }
        };
        Ok(Self {
            pool,
            raw,
            fence,
            pending: false,
            keep_alive: Vec::new(),
            device: device.clone(),
        })
    }

    /// Block until the last submission (if any) has finished on the GPU.
    pub fn wait(&mut self) -> Result<()> {
        if self.pending {
            self.fence.wait()?;
            self.fence.reset()?;
            self.pending = false;
            self.keep_alive.clear();
        }
        Ok(())
    }

    /// Start recording, after waiting for the previous submission.
    pub fn begin(&mut self) -> Result<Recorder<'_>> {
        self.wait()?;
        let dev = &self.device.raw;
        unsafe {
            // Also resets a buffer abandoned mid-recording.
            dev.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .context("vkResetCommandPool")?;
            let info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(self.raw, &info)
                .context("vkBeginCommandBuffer")?;
        }
        Ok(Recorder { cb: self })
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if self.wait().is_err() {
            // Can't tell what's still running; wait for everything.
            let _ = unsafe { self.device.raw.device_wait_idle() };
        }
        unsafe { self.device.raw.destroy_command_pool(self.pool, None) };
    }
}

/// A command buffer in the recording state. Consumed by [`Recorder::submit`].
///
/// Scanout images follow this layout protocol per frame:
/// `acquire_from_foreign` → (rendering, e.g. `clear`) → `release_to_foreign`.
pub struct Recorder<'a> {
    cb: &'a mut CommandBuffer,
}

impl Recorder<'_> {
    fn barrier(&mut self, barrier: vk::ImageMemoryBarrier2) {
        let barriers = [barrier];
        let info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { self.cb.device.raw.cmd_pipeline_barrier2(self.cb.raw, &info) };
    }

    fn keep(&mut self, image: &Arc<DmaBufImage>) {
        self.cb.keep_alive.push(image.clone());
    }

    /// Take ownership of an image from outside Vulkan (KMS, another API) for
    /// use as a color attachment. Previous contents are discarded.
    pub fn acquire_from_foreign(&mut self, image: &Arc<DmaBufImage>) {
        let qf = self.cb.device.queue_family;
        self.barrier(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(qf)
                .image(image.image)
                .subresource_range(COLOR_RANGE),
        );
        self.keep(image);
    }

    /// Clear the whole image (in `COLOR_ATTACHMENT_OPTIMAL`) to `rgba`.
    pub fn clear(&mut self, image: &Arc<DmaBufImage>, rgba: [f32; 4]) {
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: rgba },
            })];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: image.extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);
        let dev = &self.cb.device.raw;
        unsafe {
            dev.cmd_begin_rendering(self.cb.raw, &info);
            dev.cmd_end_rendering(self.cb.raw);
        }
        self.keep(image);
    }

    /// Hand a color attachment back to the outside world (e.g. for scanout).
    pub fn release_to_foreign(&mut self, image: &Arc<DmaBufImage>) {
        let qf = self.cb.device.queue_family;
        self.barrier(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(qf)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(image.image)
                .subresource_range(COLOR_RANGE),
        );
        self.keep(image);
    }

    /// End recording and submit, optionally signaling `signal` on completion.
    pub fn submit(self, signal: Option<&Arc<Semaphore>>) -> Result<()> {
        if signal.is_some_and(|s| s.signal_pending.load(Ordering::Acquire)) {
            bail!("semaphore already has a pending signal");
        }
        let cb = self.cb;
        let dev = &cb.device.raw;
        unsafe { dev.end_command_buffer(cb.raw) }.context("vkEndCommandBuffer")?;

        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb.raw)];
        let signal_infos: Vec<_> = signal
            .iter()
            .map(|s| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s.raw)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal_infos);
        unsafe { dev.queue_submit2(cb.device.queue, &[submit], cb.fence.raw) }
            .context("vkQueueSubmit2")?;

        cb.pending = true;
        if let Some(s) = signal {
            s.signal_pending.store(true, Ordering::Release);
            cb.keep_alive.push(s.clone());
        }
        Ok(())
    }
}
