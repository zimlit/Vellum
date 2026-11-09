//! Pipelines and their inputs.
//!
//! The binding model is deliberately small:
//! - **Buffers** are passed as device addresses (`u64`) inside push
//!   constants; shaders declare them with `GL_EXT_buffer_reference`. No
//!   descriptors.
//! - **Images** go through one descriptor set of storage images
//!   (`set = 0, binding = i`), used in `GENERAL` layout.

use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;

use super::{Device, Image};

/// Push constants are guaranteed to be at least this large.
pub const MAX_PUSH_CONSTANTS: u32 = 128;

/// What a pipeline takes: `storage_images` images in set 0 and
/// `push_constant_bytes` of push constants, visible to all stages.
pub struct Layout {
    pub(super) raw: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pub(super) push_constant_bytes: u32,
    storage_images: u32,
    pub(super) device: Arc<Device>,
}

impl Layout {
    pub fn new(
        device: &Arc<Device>,
        storage_images: u32,
        push_constant_bytes: u32,
    ) -> Result<Arc<Self>> {
        if push_constant_bytes > MAX_PUSH_CONSTANTS || !push_constant_bytes.is_multiple_of(4) {
            bail!("push constants must be a multiple of 4 up to {MAX_PUSH_CONSTANTS} bytes");
        }
        let dev = &device.raw;
        let bindings: Vec<_> = (0..storage_images)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::ALL)
            })
            .collect();
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set_layout = unsafe { dev.create_descriptor_set_layout(&set_info, None) }
            .context("vkCreateDescriptorSetLayout")?;
        let set_layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .size(push_constant_bytes)];
        let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if push_constant_bytes > 0 {
            info = info.push_constant_ranges(&ranges);
        }
        let raw = match unsafe { dev.create_pipeline_layout(&info, None) } {
            Ok(raw) => raw,
            Err(e) => {
                unsafe { dev.destroy_descriptor_set_layout(set_layout, None) };
                return Err(e).context("vkCreatePipelineLayout");
            }
        };
        Ok(Arc::new(Self {
            raw,
            set_layout,
            push_constant_bytes,
            storage_images,
            device: device.clone(),
        }))
    }
}

impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: pipelines and descriptor sets hold an Arc<Layout>.
        unsafe {
            self.device.raw.destroy_pipeline_layout(self.raw, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Set 0 for a [`Layout`]: one storage image per binding, which must be in
/// `GENERAL` layout whenever the set is used.
pub struct DescriptorSet {
    pool: vk::DescriptorPool,
    pub(super) raw: vk::DescriptorSet,
    _images: Vec<Arc<Image>>,
    pub(super) layout: Arc<Layout>,
}

impl DescriptorSet {
    pub fn new(layout: &Arc<Layout>, images: &[&Arc<Image>]) -> Result<Arc<Self>> {
        if images.len() != layout.storage_images as usize {
            bail!(
                "layout has {} storage images, got {}",
                layout.storage_images,
                images.len()
            );
        }
        let dev = &layout.device.raw;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(layout.storage_images.max(1))];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        let pool = unsafe { dev.create_descriptor_pool(&pool_info, None) }
            .context("vkCreateDescriptorPool")?;
        let set_layouts = [layout.set_layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        let raw = match unsafe { dev.allocate_descriptor_sets(&alloc) } {
            Ok(sets) => sets[0],
            Err(e) => {
                unsafe { dev.destroy_descriptor_pool(pool, None) };
                return Err(e).context("vkAllocateDescriptorSets");
            }
        };
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = images
            .iter()
            .map(|image| {
                [vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(raw)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(info)
            })
            .collect();
        // SAFETY: the set is new, so not in use by any command buffer.
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        Ok(Arc::new(Self {
            pool,
            raw,
            _images: images.iter().map(|&i| i.clone()).collect(),
            layout: layout.clone(),
        }))
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        // SAFETY: command buffers hold an Arc while in flight; destroying the
        // pool frees the set.
        unsafe {
            self.layout
                .device
                .raw
                .destroy_descriptor_pool(self.pool, None)
        };
    }
}

struct ShaderModule<'a> {
    raw: vk::ShaderModule,
    device: &'a Device,
}

impl<'a> ShaderModule<'a> {
    fn new(device: &'a Device, spirv: &[u8]) -> Result<Self> {
        // read_spv checks the magic number and copies into aligned words.
        let code = ash::util::read_spv(&mut Cursor::new(spirv)).context("invalid SPIR-V")?;
        let info = vk::ShaderModuleCreateInfo::default().code(&code);
        let raw = unsafe { device.raw.create_shader_module(&info, None) }
            .context("vkCreateShaderModule")?;
        Ok(Self { raw, device })
    }
}

impl Drop for ShaderModule<'_> {
    fn drop(&mut self) {
        // Pipelines don't need their modules after creation.
        unsafe { self.device.raw.destroy_shader_module(self.raw, None) };
    }
}

pub struct ComputePipeline {
    pub(super) raw: vk::Pipeline,
    pub(super) layout: Arc<Layout>,
}

impl ComputePipeline {
    /// `spirv` is a compiled compute shader with entry point `main`.
    pub fn new(layout: &Arc<Layout>, spirv: &[u8]) -> Result<Arc<Self>> {
        let device = &layout.device;
        let module = ShaderModule::new(device, spirv)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module.raw)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout.raw);
        let raw = unsafe {
            device
                .raw
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, e)| e)
        .context("vkCreateComputePipelines")?[0];
        Ok(Arc::new(Self {
            raw,
            layout: layout.clone(),
        }))
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.layout.device.raw.destroy_pipeline(self.raw, None) };
    }
}

/// How fragments combine with what's already in the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// Overwrite.
    Replace,
    /// Standard "over" with premultiplied alpha:
    /// `dst = src + dst * (1 - src.a)`.
    PremultipliedAlpha,
}

/// A graphics pipeline for dynamic rendering into one color attachment.
/// No vertex buffers: the vertex shader builds positions from
/// `gl_VertexIndex` / `gl_InstanceIndex` and buffer addresses. Triangle
/// list, no culling, viewport and scissor set per draw.
pub struct GraphicsPipeline {
    pub(super) raw: vk::Pipeline,
    pub(super) layout: Arc<Layout>,
}

impl GraphicsPipeline {
    pub fn new(
        layout: &Arc<Layout>,
        vertex_spirv: &[u8],
        fragment_spirv: &[u8],
        color_format: vk::Format,
        blend: Blend,
    ) -> Result<Arc<Self>> {
        let device = &layout.device;
        let vs = ShaderModule::new(device, vertex_spirv)?;
        let fs = ShaderModule::new(device, fragment_spirv)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs.raw)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs.raw)
                .name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachment = match blend {
            Blend::Replace => vk::PipelineColorBlendAttachmentState::default(),
            Blend::PremultipliedAlpha => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD),
        }
        .color_write_mask(vk::ColorComponentFlags::RGBA);
        let attachments = [attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [color_format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout.raw)
            .push_next(&mut rendering);
        let raw = unsafe {
            device
                .raw
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, e)| e)
        .context("vkCreateGraphicsPipelines")?[0];
        Ok(Arc::new(Self {
            raw,
            layout: layout.clone(),
        }))
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe { self.layout.device.raw.destroy_pipeline(self.raw, None) };
    }
}
