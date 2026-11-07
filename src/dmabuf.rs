//! A driver-independent description of a dma-buf backed image.
//!
//! This is the hand-off format between allocators (GBM today, Vulkan's
//! `VK_EXT_image_drm_format_modifier` or Wayland clients later) and consumers
//! (KMS framebuffers, Vulkan imports).

use std::os::fd::OwnedFd;

use drm::buffer::{DrmFourcc, DrmModifier};

pub struct DmaBuf {
    pub width: u32,
    pub height: u32,
    pub format: DrmFourcc,
    /// Memory layout. `DrmModifier::Invalid` means "implicit": the layout is
    /// known only to the driver that allocated it.
    pub modifier: DrmModifier,
    pub planes: Vec<DmaBufPlane>,
}

pub struct DmaBufPlane {
    pub fd: OwnedFd,
    pub offset: u32,
    pub stride: u32,
}
