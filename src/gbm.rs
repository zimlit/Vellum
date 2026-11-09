//! Minimal hand-written bindings to libgbm (Mesa's Generic Buffer Manager).
//!
//! GBM is used purely as an allocator: it hands out buffers that the display
//! engine can scan out, which we immediately export as dma-buf fds. Everything
//! downstream (KMS framebuffers now, Vulkan import later) consumes the dma-buf,
//! never the GBM object itself.

use std::io;
use std::marker::PhantomData;
use std::ops::BitOr;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;

use drm::buffer::{DrmFourcc, DrmModifier};

use crate::dmabuf::{DmaBuf, DmaBufPlane};

#[allow(non_camel_case_types)]
mod ffi {
    use std::ffi::{c_int, c_uint};

    #[repr(C)]
    pub struct gbm_device {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub struct gbm_bo {
        _opaque: [u8; 0],
    }

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
}

impl Drop for Bo<'_> {
    fn drop(&mut self) {
        unsafe { ffi::gbm_bo_destroy(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsFd;

    use super::*;
    use crate::testutil;

    #[test]
    fn bo_flags_combine_bits() {
        let flags = BoFlags::SCANOUT | BoFlags::RENDERING;
        assert_eq!(flags.0, (1 << 0) | (1 << 2));
        assert_eq!(BoFlags::LINEAR | BoFlags::LINEAR, BoFlags::LINEAR);
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn implicit_bo_reports_size_and_format() {
        let card = testutil::card();
        let gbm = Device::new(card.as_fd()).unwrap();
        let bo = gbm
            .create_bo(
                640,
                480,
                DrmFourcc::Xrgb8888,
                &[],
                BoFlags::SCANOUT | BoFlags::RENDERING,
            )
            .unwrap();
        assert_eq!((bo.width(), bo.height()), (640, 480));
        assert_eq!(bo.format(), DrmFourcc::Xrgb8888 as u32);
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn linear_bo_exports_one_plane_dmabuf() {
        let card = testutil::card();
        let gbm = Device::new(card.as_fd()).unwrap();
        let linear = u64::from(DrmModifier::Linear);
        let bo = gbm
            .create_bo(640, 480, DrmFourcc::Xrgb8888, &[linear], BoFlags::SCANOUT)
            .unwrap();
        assert_eq!(bo.modifier(), linear);

        let buf = bo.export().unwrap();
        assert_eq!((buf.width, buf.height), (640, 480));
        assert_eq!(buf.format, DrmFourcc::Xrgb8888);
        assert_eq!(buf.modifier, DrmModifier::Linear);
        assert_eq!(buf.planes.len(), 1);
        assert!(buf.planes[0].stride >= 640 * 4);
        // Exported fds outlive the BO.
        drop(bo);
        assert!(rustix::fs::fstat(&buf.planes[0].fd).is_ok());
    }

    #[test]
    #[ignore = "needs a GPU with a connected display; run with --ignored"]
    fn zero_sized_bo_is_an_error() {
        let card = testutil::card();
        let gbm = Device::new(card.as_fd()).unwrap();
        assert!(
            gbm.create_bo(0, 0, DrmFourcc::Xrgb8888, &[], BoFlags::RENDERING)
                .is_err()
        );
    }
}
