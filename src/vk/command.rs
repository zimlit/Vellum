use std::any::Any;
use std::ops::BitOr;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, bail};
use ash::vk;

use super::buffer::{Pod, bytes_of};
use super::image::{COLOR_RANGE, RenderTarget};
use super::pipeline::Layout;
use super::sync::Fence;
use super::{
    Buffer, ComputePipeline, DescriptorSet, Device, DmaBufImage, GraphicsPipeline, Image, Semaphore,
};

/// A pipeline stage together with how it touches memory there: the two
/// halves of a barrier. Combine with `|`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

impl Access {
    const fn new(stage: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> Self {
        Self { stage, access }
    }

    pub const NONE: Self = Self::new(vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
    pub const TRANSFER_READ: Self = Self::new(
        vk::PipelineStageFlags2::ALL_TRANSFER,
        vk::AccessFlags2::TRANSFER_READ,
    );
    pub const TRANSFER_WRITE: Self = Self::new(
        vk::PipelineStageFlags2::ALL_TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
    );
    pub const COMPUTE_READ: Self = Self::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    pub const COMPUTE_WRITE: Self = Self::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
    );
    pub const VERTEX_READ: Self = Self::new(
        vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    pub const FRAGMENT_READ: Self = Self::new(
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    pub const COLOR_ATTACHMENT: Self = Self::new(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::from_raw(
            vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw(),
        ),
    );
    pub const HOST_READ: Self =
        Self::new(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ);
}

impl BitOr for Access {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self::new(self.stage | rhs.stage, self.access | rhs.access)
    }
}

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
        Ok(Recorder {
            cb: self,
            rendering: false,
        })
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
///
/// What is checked here: resource lifetimes, draws only inside rendering and
/// dispatches only outside it, push constant sizes, and that descriptor sets
/// match the pipeline. What is not: barriers and image layouts (GPU-side
/// correctness, not CPU memory safety); the validation layers check those.
pub struct Recorder<'a> {
    cb: &'a mut CommandBuffer,
    rendering: bool,
}

impl Recorder<'_> {
    fn image_barrier(&mut self, barrier: vk::ImageMemoryBarrier2) {
        let barriers = [barrier];
        let info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { self.cb.device.raw.cmd_pipeline_barrier2(self.cb.raw, &info) };
    }

    fn keep<T: Any + Send + Sync>(&mut self, resource: &Arc<T>) {
        self.cb.keep_alive.push(resource.clone());
    }

    /// Make writes from `src` visible to accesses in `dst`, for all memory.
    pub fn barrier(&mut self, src: Access, dst: Access) {
        let barriers = [vk::MemoryBarrier2::default()
            .src_stage_mask(src.stage)
            .src_access_mask(src.access)
            .dst_stage_mask(dst.stage)
            .dst_access_mask(dst.access)];
        let info = vk::DependencyInfo::default().memory_barriers(&barriers);
        unsafe { self.cb.device.raw.cmd_pipeline_barrier2(self.cb.raw, &info) };
    }

    /// Change an image's layout, ordered after `src` and before `dst`.
    /// From `UNDEFINED`, the contents are discarded.
    pub fn transition(
        &mut self,
        image: &Arc<Image>,
        from: vk::ImageLayout,
        to: vk::ImageLayout,
        src: Access,
        dst: Access,
    ) {
        self.image_barrier(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src.stage)
                .src_access_mask(src.access)
                .dst_stage_mask(dst.stage)
                .dst_access_mask(dst.access)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(COLOR_RANGE),
        );
        self.keep(image);
    }

    /// Fill the whole buffer with a repeated 32-bit `value` (a transfer
    /// write). The buffer needs `TRANSFER_DST` usage.
    pub fn fill(&mut self, buffer: &Arc<Buffer>, value: u32) {
        unsafe {
            self.cb
                .device
                .raw
                .cmd_fill_buffer(self.cb.raw, buffer.raw, 0, vk::WHOLE_SIZE, value)
        };
        self.keep(buffer);
    }

    /// Copy a 4-byte-per-texel image (in `TRANSFER_SRC_OPTIMAL`) into
    /// `buffer`, tightly packed row by row. The buffer needs `TRANSFER_DST`.
    pub fn copy_image_to_buffer(&mut self, image: &Arc<Image>, buffer: &Arc<Buffer>) -> Result<()> {
        let needed = u64::from(image.width()) * u64::from(image.height()) * 4;
        if buffer.size() < needed {
            bail!("buffer of {} bytes can't hold {needed}", buffer.size());
        }
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: image.width(),
                height: image.height(),
                depth: 1,
            });
        unsafe {
            self.cb.device.raw.cmd_copy_image_to_buffer(
                self.cb.raw,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.raw,
                &[region],
            )
        };
        self.keep(image);
        self.keep(buffer);
        Ok(())
    }

    /// Bind the descriptor set (if any) and push constants for `layout`.
    fn bind_inputs<P: Pod>(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: &Arc<Layout>,
        set: Option<&Arc<DescriptorSet>>,
        push: &P,
    ) -> Result<()> {
        let bytes = bytes_of(slice::from_ref(push));
        if bytes.len() as u32 > layout.push_constant_bytes {
            bail!(
                "{} bytes of push constants for a layout with {}",
                bytes.len(),
                layout.push_constant_bytes
            );
        }
        let device = self.cb.device.clone();
        let dev = &device.raw;
        if let Some(set) = set {
            if !Arc::ptr_eq(&set.layout, layout) {
                bail!("descriptor set was made for a different layout");
            }
            unsafe {
                dev.cmd_bind_descriptor_sets(
                    self.cb.raw,
                    bind_point,
                    layout.raw,
                    0,
                    &[set.raw],
                    &[],
                )
            };
            self.keep(set);
        }
        if !bytes.is_empty() {
            unsafe {
                dev.cmd_push_constants(self.cb.raw, layout.raw, vk::ShaderStageFlags::ALL, 0, bytes)
            };
        }
        Ok(())
    }

    /// Run a compute shader over `groups` workgroups. Buffers it uses are
    /// passed by address in `push`; keep them alive with [`Recorder::using`].
    pub fn dispatch<P: Pod>(
        &mut self,
        pipeline: &Arc<ComputePipeline>,
        set: Option<&Arc<DescriptorSet>>,
        push: &P,
        groups: [u32; 3],
    ) -> Result<()> {
        if self.rendering {
            bail!("dispatch inside a rendering pass");
        }
        unsafe {
            self.cb.device.raw.cmd_bind_pipeline(
                self.cb.raw,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.raw,
            )
        };
        self.bind_inputs(vk::PipelineBindPoint::COMPUTE, &pipeline.layout, set, push)?;
        unsafe {
            self.cb
                .device
                .raw
                .cmd_dispatch(self.cb.raw, groups[0], groups[1], groups[2])
        };
        self.keep(pipeline);
        Ok(())
    }

    /// Keep buffers that shaders reach by address alive until this
    /// submission completes (the recorder can't see inside push constants).
    pub fn using(&mut self, buffers: &[&Arc<Buffer>]) {
        for b in buffers {
            self.keep(b);
        }
    }

    /// Start rendering into `target` (in `COLOR_ATTACHMENT_OPTIMAL`),
    /// clearing it first if `clear` is given. Viewport and scissor cover the
    /// whole target.
    pub fn begin_rendering<T: RenderTarget>(
        &mut self,
        target: &Arc<T>,
        clear: Option<[f32; 4]>,
    ) -> Result<()> {
        if self.rendering {
            bail!("already rendering");
        }
        let (view, extent) = target.view_and_extent();
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(match clear {
                Some(_) => vk::AttachmentLoadOp::CLEAR,
                None => vk::AttachmentLoadOp::LOAD,
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear.unwrap_or_default(),
                },
            })];
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&attachments);
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let dev = &self.cb.device.raw;
        unsafe {
            dev.cmd_begin_rendering(self.cb.raw, &info);
            dev.cmd_set_viewport(self.cb.raw, 0, &[viewport]);
            dev.cmd_set_scissor(self.cb.raw, 0, &[area]);
        }
        self.rendering = true;
        self.keep(target);
        Ok(())
    }

    /// Draw `vertices` vertices `instances` times (triangle list).
    pub fn draw<P: Pod>(
        &mut self,
        pipeline: &Arc<GraphicsPipeline>,
        set: Option<&Arc<DescriptorSet>>,
        push: &P,
        vertices: u32,
        instances: u32,
    ) -> Result<()> {
        if !self.rendering {
            bail!("draw outside a rendering pass");
        }
        unsafe {
            self.cb.device.raw.cmd_bind_pipeline(
                self.cb.raw,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw,
            )
        };
        self.bind_inputs(vk::PipelineBindPoint::GRAPHICS, &pipeline.layout, set, push)?;
        unsafe {
            self.cb
                .device
                .raw
                .cmd_draw(self.cb.raw, vertices, instances, 0, 0)
        };
        self.keep(pipeline);
        Ok(())
    }

    pub fn end_rendering(&mut self) -> Result<()> {
        if !self.rendering {
            bail!("not rendering");
        }
        unsafe { self.cb.device.raw.cmd_end_rendering(self.cb.raw) };
        self.rendering = false;
        Ok(())
    }

    /// Take ownership of an image from outside Vulkan (KMS, another API) for
    /// use as a color attachment. Previous contents are discarded.
    pub fn acquire_from_foreign(&mut self, image: &Arc<DmaBufImage>) {
        let qf = self.cb.device.queue_family;
        self.image_barrier(
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

    /// Clear the whole target (in `COLOR_ATTACHMENT_OPTIMAL`) to `rgba`.
    pub fn clear<T: RenderTarget>(&mut self, target: &Arc<T>, rgba: [f32; 4]) -> Result<()> {
        self.begin_rendering(target, Some(rgba))?;
        self.end_rendering()
    }

    /// Hand a color attachment back to the outside world (e.g. for scanout).
    pub fn release_to_foreign(&mut self, image: &Arc<DmaBufImage>) {
        let qf = self.cb.device.queue_family;
        self.image_barrier(
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
        if self.rendering {
            bail!("submit inside a rendering pass");
        }
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
