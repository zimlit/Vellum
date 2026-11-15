use std::os::fd::{AsRawFd, IntoRawFd};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use drm::buffer::DrmFourcc;

use super::Device;
use super::memory::{Dedicated, Location, Memory};
use crate::dmabuf::DmaBuf;

/// The Vulkan format with the same memory layout as a DRM fourcc. DRM
/// fourccs name components from the most significant bit of a little-endian
/// word, Vulkan from the lowest byte, so XRGB8888 (bytes B,G,R,X) is B8G8R8A8.
pub fn vk_format(fourcc: DrmFourcc) -> Option<vk::Format> {
    Some(match fourcc {
        DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => vk::Format::B8G8R8A8_UNORM,
        DrmFourcc::Xbgr8888 | DrmFourcc::Abgr8888 => vk::Format::R8G8B8A8_UNORM,
        _ => return None,
    })
}

/// A DRM format modifier Vulkan can import dma-bufs with.
#[derive(Clone, Copy, Debug)]
pub struct Modifier {
    pub modifier: u64,
    /// Memory planes a buffer with this modifier has (may include
    /// compression metadata planes).
    pub plane_count: u32,
}

fn required_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
    let mut f = vk::FormatFeatureFlags::empty();
    if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        f |= vk::FormatFeatureFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(vk::ImageUsageFlags::SAMPLED) {
        f |= vk::FormatFeatureFlags::SAMPLED_IMAGE;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        f |= vk::FormatFeatureFlags::TRANSFER_DST;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        f |= vk::FormatFeatureFlags::TRANSFER_SRC;
    }
    f
}

impl Device {
    /// Modifiers for which a `fourcc` dma-buf can be imported as an image
    /// with `usage`.
    pub fn dmabuf_modifiers(
        &self,
        fourcc: DrmFourcc,
        usage: vk::ImageUsageFlags,
    ) -> Result<Vec<Modifier>> {
        let format = vk_format(fourcc).ok_or_else(|| anyhow!("no Vulkan format for {fourcc:?}"))?;
        let inst = &self.instance.raw;

        // Two-call idiom: first the count, then the list.
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
        unsafe { inst.get_physical_device_format_properties2(self.physical, format, &mut props2) };
        let mut props = vec![
            vk::DrmFormatModifierPropertiesEXT::default();
            list.drm_format_modifier_count as usize
        ];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut props);
        let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
        unsafe { inst.get_physical_device_format_properties2(self.physical, format, &mut props2) };

        let needed = required_features(usage);
        let mut out = Vec::new();
        for p in &props {
            if p.drm_format_modifier_tiling_features.contains(needed)
                && self.importable(format, usage, p.drm_format_modifier)?
            {
                out.push(Modifier {
                    modifier: p.drm_format_modifier,
                    plane_count: p.drm_format_modifier_plane_count,
                });
            }
        }
        Ok(out)
    }

    fn importable(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        modifier: u64,
    ) -> Result<bool> {
        let mut drm_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(modifier)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .push_next(&mut drm_info)
            .push_next(&mut external_info);
        let mut external_props = vk::ExternalImageFormatProperties::default();
        let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
        let result = unsafe {
            self.instance
                .raw
                .get_physical_device_image_format_properties2(self.physical, &info, &mut props)
        };
        match result {
            Ok(()) => Ok(external_props
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)),
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(false),
            Err(e) => Err(e).context("vkGetPhysicalDeviceImageFormatProperties2"),
        }
    }
}

/// A 2D image whose memory is an imported dma-buf.
pub struct DmaBufImage {
    pub(super) image: vk::Image,
    pub(super) view: vk::ImageView,
    memory: vk::DeviceMemory,
    pub(super) extent: vk::Extent2D,
    device: Arc<Device>,
}

impl DmaBufImage {
    /// Import `buf` (explicit modifier, all planes in one dma-buf).
    pub fn import(
        device: &Arc<Device>,
        buf: &DmaBuf,
        usage: vk::ImageUsageFlags,
    ) -> Result<Arc<Self>> {
        let format = vk_format(buf.format)
            .ok_or_else(|| anyhow!("no Vulkan format for {:?}", buf.format))?;
        let first = buf
            .planes
            .first()
            .ok_or_else(|| anyhow!("dma-buf has no planes"))?;

        // Disjoint images (planes in separate allocations) would need one
        // memory import per plane; GBM doesn't produce them for our formats.
        let first_stat = rustix::fs::fstat(&first.fd)?;
        for plane in &buf.planes[1..] {
            let stat = rustix::fs::fstat(&plane.fd)?;
            if (stat.st_dev, stat.st_ino) != (first_stat.st_dev, first_stat.st_ino) {
                bail!("disjoint multi-plane dma-bufs are not supported");
            }
        }

        let dev = &device.raw;
        let extent = vk::Extent2D {
            width: buf.width,
            height: buf.height,
        };
        // Null handles are valid to destroy, so a partially built image can
        // clean up after itself on any early return.
        let mut this = Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            memory: vk::DeviceMemory::null(),
            extent,
            device: device.clone(),
        };

        let layouts: Vec<vk::SubresourceLayout> = buf
            .planes
            .iter()
            .map(|p| vk::SubresourceLayout {
                offset: u64::from(p.offset),
                row_pitch: u64::from(p.stride),
                ..Default::default()
            })
            .collect();
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(u64::from(buf.modifier))
            .plane_layouts(&layouts);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: buf.width,
                height: buf.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut explicit)
            .push_next(&mut external);
        this.image = unsafe { dev.create_image(&info, None) }.context("vkCreateImage (dma-buf)")?;

        // Memory types usable for both this image and this dma-buf.
        let fd = first.fd.try_clone().context("dup dma-buf fd")?;
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        unsafe {
            device.memory_fd.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut fd_props,
            )
        }
        .context("vkGetMemoryFdPropertiesKHR")?;
        let reqs = unsafe { dev.get_image_memory_requirements(this.image) };
        let types = reqs.memory_type_bits & fd_props.memory_type_bits;
        if types == 0 {
            bail!("no memory type fits both the image and the dma-buf");
        }

        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(this.image);
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(types.trailing_zeros())
            .push_next(&mut import)
            .push_next(&mut dedicated);
        this.memory = unsafe { dev.allocate_memory(&alloc, None) }
            .context("vkAllocateMemory (dma-buf import)")?;
        // A successful import takes ownership of the fd; on failure `fd`
        // would have been dropped (closed) by the `?` above.
        let _ = fd.into_raw_fd();

        unsafe { dev.bind_image_memory(this.image, this.memory, 0) }
            .context("vkBindImageMemory")?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(this.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(COLOR_RANGE);
        this.view =
            unsafe { dev.create_image_view(&view_info, None) }.context("vkCreateImageView")?;

        Ok(Arc::new(this))
    }
}

/// A 2D image in its own device-local memory (atlas, offscreen targets).
pub struct Image {
    pub(super) image: vk::Image,
    pub(super) view: vk::ImageView,
    pub(super) extent: vk::Extent2D,
    format: vk::Format,
    /// Freed after Drop::drop has destroyed the image and view.
    memory: Option<Memory>,
    device: Arc<Device>,
}

impl Image {
    /// An optimally tiled image, initially in `UNDEFINED` layout.
    pub fn new(
        device: &Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Arc<Self>> {
        let dev = &device.raw;
        let extent = vk::Extent2D { width, height };
        let mut this = Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            extent,
            format,
            memory: None,
            device: device.clone(),
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        this.image = unsafe { dev.create_image(&info, None) }
            .with_context(|| format!("vkCreateImage ({width}x{height} {format:?})"))?;
        let reqs = unsafe { dev.get_image_memory_requirements(this.image) };
        let memory = Memory::allocate(
            device,
            reqs,
            Location::Device,
            Dedicated::Image(this.image),
            false,
        )?;
        unsafe { dev.bind_image_memory(this.image, memory.raw, 0) }.context("vkBindImageMemory")?;
        this.memory = Some(memory);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(this.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(COLOR_RANGE);
        this.view =
            unsafe { dev.create_image_view(&view_info, None) }.context("vkCreateImageView")?;
        Ok(Arc::new(this))
    }

    pub fn width(&self) -> u32 {
        self.extent.width
    }

    pub fn height(&self) -> u32 {
        self.extent.height
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: command buffers keep an Arc while in flight. Null handles
        // (partly built image) are valid to destroy. `memory` drops after.
        unsafe {
            let dev = &self.device.raw;
            dev.destroy_image_view(self.view, None);
            dev.destroy_image(self.image, None);
        }
    }
}

/// An image that can be rendered into with [`super::Recorder::begin_rendering`].
pub trait RenderTarget: Send + Sync + 'static {
    #[doc(hidden)]
    fn view_and_extent(&self) -> (vk::ImageView, vk::Extent2D);

    /// Width and height in pixels.
    fn size(&self) -> (u32, u32) {
        let (_, extent) = self.view_and_extent();
        (extent.width, extent.height)
    }
}

impl RenderTarget for Image {
    fn view_and_extent(&self) -> (vk::ImageView, vk::Extent2D) {
        (self.view, self.extent)
    }
}

impl RenderTarget for DmaBufImage {
    fn view_and_extent(&self) -> (vk::ImageView, vk::Extent2D) {
        (self.view, self.extent)
    }
}

pub(super) const COLOR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

impl Drop for DmaBufImage {
    fn drop(&mut self) {
        // SAFETY: command buffers keep an Arc to images they use until their
        // fence signals, so the GPU is done with this image.
        unsafe {
            let dev = &self.device.raw;
            dev.destroy_image_view(self.view, None);
            dev.destroy_image(self.image, None);
            dev.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vk_format_matches_drm_byte_order() {
        // XRGB8888 is bytes B,G,R,X in memory: B8G8R8A8 in Vulkan terms.
        assert_eq!(
            vk_format(DrmFourcc::Xrgb8888),
            Some(vk::Format::B8G8R8A8_UNORM)
        );
        assert_eq!(
            vk_format(DrmFourcc::Argb8888),
            Some(vk::Format::B8G8R8A8_UNORM)
        );
        assert_eq!(
            vk_format(DrmFourcc::Xbgr8888),
            Some(vk::Format::R8G8B8A8_UNORM)
        );
        assert_eq!(
            vk_format(DrmFourcc::Abgr8888),
            Some(vk::Format::R8G8B8A8_UNORM)
        );
        assert_eq!(vk_format(DrmFourcc::Nv12), None);
    }

    #[test]
    fn required_features_follow_usage() {
        use vk::{FormatFeatureFlags as F, ImageUsageFlags as U};
        assert_eq!(required_features(U::empty()), F::empty());
        assert_eq!(required_features(U::COLOR_ATTACHMENT), F::COLOR_ATTACHMENT);
        assert_eq!(required_features(U::SAMPLED), F::SAMPLED_IMAGE);
        assert_eq!(
            required_features(U::TRANSFER_SRC | U::TRANSFER_DST),
            F::TRANSFER_SRC | F::TRANSFER_DST
        );
        assert_eq!(
            required_features(U::COLOR_ATTACHMENT | U::SAMPLED),
            F::COLOR_ATTACHMENT | F::SAMPLED_IMAGE
        );
    }
}
