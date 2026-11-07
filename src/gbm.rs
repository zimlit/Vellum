//! Minimal hand-written bindings to libgbm (Mesa's Generic Buffer Manager).
//!
//! GBM is used purely as an allocator: it hands out buffers that the display
//! engine can scan out, which we immediately export as dma-buf fds. Everything
//! downstream (KMS framebuffers now, Vulkan import later) consumes the dma-buf,
//! never the GBM object itself.

use std::ffi::c_void;
use std::io;
use std::marker::PhantomData;
use std::ops::BitOr;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::{self, NonNull};

use drm::buffer::{DrmFourcc, DrmModifier};

use crate::dmabuf::{DmaBuf, DmaBufPlane};

#[allow(non_camel_case_types)]
mod ffi {
    use std::ffi::{c_int, c_uint, c_void};

    #[repr(C)]
    pub struct gbm_device {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub struct gbm_bo {
        _opaque: [u8; 0],
    }

    pub const GBM_BO_TRANSFER_WRITE: u32 = 1 << 1;

    #[link(name = "gbm")]
    unsafe extern "C" {
        pub fn gbm_create_device(fd: c_int) -> *mut gbm_device;
        pub fn gbm_device_destroy(gbm: *mut gbm_device);

        pub fn gbm_bo_create(
            gbm: *mut gbm_device,
            width: u32,
            height: u32,
            format: u32,
            flags: u32,
        ) -> *mut gbm_bo;
        pub fn gbm_bo_create_with_modifiers2(
            gbm: *mut gbm_device,
            width: u32,
            height: u32,
            format: u32,
            modifiers: *const u64,
            count: c_uint,
            flags: u32,
        ) -> *mut gbm_bo;
        pub fn gbm_bo_destroy(bo: *mut gbm_bo);

        pub fn gbm_bo_get_width(bo: *mut gbm_bo) -> u32;
        pub fn gbm_bo_get_height(bo: *mut gbm_bo) -> u32;
        pub fn gbm_bo_get_format(bo: *mut gbm_bo) -> u32;
        pub fn gbm_bo_get_modifier(bo: *mut gbm_bo) -> u64;
        pub fn gbm_bo_get_plane_count(bo: *mut gbm_bo) -> c_int;
        pub fn gbm_bo_get_stride_for_plane(bo: *mut gbm_bo, plane: c_int) -> u32;
        pub fn gbm_bo_get_offset(bo: *mut gbm_bo, plane: c_int) -> u32;
        pub fn gbm_bo_get_fd_for_plane(bo: *mut gbm_bo, plane: c_int) -> c_int;

        pub fn gbm_bo_map(
            bo: *mut gbm_bo,
            x: u32,
            y: u32,
            width: u32,
            height: u32,
            flags: u32,
            stride: *mut u32,
            map_data: *mut *mut c_void,
        ) -> *mut c_void;
        pub fn gbm_bo_unmap(bo: *mut gbm_bo, map_data: *mut c_void);
    }
}

/// `enum gbm_bo_flags` usage bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoFlags(u32);

impl BoFlags {
    /// Buffer will be presented on screen by the display engine.
    pub const SCANOUT: Self = Self(1 << 0);
    /// Buffer will be a GPU render target.
    pub const RENDERING: Self = Self(1 << 2);
    /// Force a linear layout. Not allowed together with an explicit modifier list.
    #[allow(dead_code)]
    pub const LINEAR: Self = Self(1 << 4);
}

impl BitOr for BoFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// A GBM device. It borrows the DRM fd (GBM does not dup it), so it must not
/// outlive the card.
pub struct Device<'fd> {
    raw: NonNull<ffi::gbm_device>,
    _fd: PhantomData<BorrowedFd<'fd>>,
}

impl<'fd> Device<'fd> {
    pub fn new(fd: BorrowedFd<'fd>) -> io::Result<Self> {
        // SAFETY: fd is a valid open DRM fd for at least 'fd.
        let raw = unsafe { ffi::gbm_create_device(fd.as_raw_fd()) };
        NonNull::new(raw)
            .map(|raw| Self {
                raw,
                _fd: PhantomData,
            })
            .ok_or_else(io::Error::last_os_error)
    }

    /// Allocate a buffer object. With an empty `modifiers` list the driver
    /// picks the layout implicitly (and the BO reports `DRM_FORMAT_MOD_INVALID`
    /// unless the driver knows better); otherwise it picks one of `modifiers`.
    pub fn create_bo(
        &self,
        width: u32,
        height: u32,
        format: DrmFourcc,
        modifiers: &[u64],
        flags: BoFlags,
    ) -> io::Result<Bo<'_>> {
        // SAFETY: self.raw is a live device; the modifier slice outlives the call.
        let raw = unsafe {
            if modifiers.is_empty() {
                ffi::gbm_bo_create(self.raw.as_ptr(), width, height, format as u32, flags.0)
            } else {
                ffi::gbm_bo_create_with_modifiers2(
                    self.raw.as_ptr(),
                    width,
                    height,
                    format as u32,
                    modifiers.as_ptr(),
                    modifiers.len() as _,
                    flags.0,
                )
            }
        };
        NonNull::new(raw)
            .map(|raw| Bo {
                raw,
                _dev: PhantomData,
            })
            .ok_or_else(io::Error::last_os_error)
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        // SAFETY: all Bos borrow self, so none are alive here.
        unsafe { ffi::gbm_device_destroy(self.raw.as_ptr()) }
    }
}

/// A GBM buffer object.
pub struct Bo<'dev> {
    raw: NonNull<ffi::gbm_bo>,
    _dev: PhantomData<&'dev Device<'dev>>,
}

impl Bo<'_> {
    pub fn width(&self) -> u32 {
        unsafe { ffi::gbm_bo_get_width(self.raw.as_ptr()) }
    }

    pub fn height(&self) -> u32 {
        unsafe { ffi::gbm_bo_get_height(self.raw.as_ptr()) }
    }

    pub fn format(&self) -> u32 {
        unsafe { ffi::gbm_bo_get_format(self.raw.as_ptr()) }
    }

    pub fn modifier(&self) -> u64 {
        unsafe { ffi::gbm_bo_get_modifier(self.raw.as_ptr()) }
    }

    /// Export every memory plane as a dma-buf fd. The dma-buf keeps the
    /// underlying memory alive independently of this BO.
    pub fn export(&self) -> io::Result<DmaBuf> {
        let bo = self.raw.as_ptr();
        let format = DrmFourcc::try_from(self.format())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}")))?;
        let count = unsafe { ffi::gbm_bo_get_plane_count(bo) };
        let mut planes = Vec::with_capacity(count as usize);
        for plane in 0..count {
            // SAFETY: plane < plane_count; a non-negative return is a new fd we own.
            let fd = unsafe { ffi::gbm_bo_get_fd_for_plane(bo, plane) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            planes.push(DmaBufPlane {
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
                offset: unsafe { ffi::gbm_bo_get_offset(bo, plane) },
                stride: unsafe { ffi::gbm_bo_get_stride_for_plane(bo, plane) },
            });
        }
        Ok(DmaBuf {
            width: self.width(),
            height: self.height(),
            format,
            modifier: DrmModifier::from(self.modifier()),
            planes,
        })
    }

    /// Map the whole buffer for CPU writes. For tiled layouts the driver maps a
    /// linear staging copy and blits it back on unmap, so this is slow but
    /// layout-agnostic. Only meant for bring-up; real rendering goes via Vulkan.
    pub fn map_write(&mut self) -> io::Result<Mapping<'_>> {
        let (width, height) = (self.width(), self.height());
        let mut stride = 0;
        let mut map_data = ptr::null_mut();
        // SAFETY: valid BO; out-pointers are valid for writes.
        let ptr = unsafe {
            ffi::gbm_bo_map(
                self.raw.as_ptr(),
                0,
                0,
                width,
                height,
                ffi::GBM_BO_TRANSFER_WRITE,
                &mut stride,
                &mut map_data,
            )
        };
        let ptr = NonNull::new(ptr.cast::<u8>()).ok_or_else(io::Error::last_os_error)?;
        Ok(Mapping {
            bo: self.raw,
            ptr,
            map_data,
            stride,
            height,
            _bo: PhantomData,
        })
    }
}

impl Drop for Bo<'_> {
    fn drop(&mut self) {
        unsafe { ffi::gbm_bo_destroy(self.raw.as_ptr()) }
    }
}

/// A CPU mapping of a BO; unmapped (and flushed to the BO) on drop.
pub struct Mapping<'bo> {
    bo: NonNull<ffi::gbm_bo>,
    ptr: NonNull<u8>,
    map_data: *mut c_void,
    stride: u32,
    height: u32,
    _bo: PhantomData<&'bo mut Bo<'bo>>,
}

impl Mapping<'_> {
    /// Bytes per row of the mapping (may differ from the BO's own stride).
    pub fn stride(&self) -> u32 {
        self.stride
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: gbm_bo_map returns a region of at least stride * height bytes,
        // exclusively ours until unmap.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.stride as usize * self.height as usize,
            )
        }
    }
}

impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        unsafe { ffi::gbm_bo_unmap(self.bo.as_ptr(), self.map_data) }
    }
}
