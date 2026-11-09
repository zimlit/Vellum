use std::mem::size_of;
use std::ptr;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;

use super::Device;
use super::memory::{Dedicated, Location, Memory};

/// Plain data that can be copied to and from GPU memory byte for byte.
///
/// # Safety
/// The type must have no padding, no pointers or references, and every bit
/// pattern must be a valid value. Structs need `#[repr(C)]` with fields laid
/// out to match the shader's (std430 / scalar) layout.
pub unsafe trait Pod: Copy + Send + Sync + 'static {}

/// No push constants.
unsafe impl Pod for () {}
unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for f32 {}
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// The bytes of a slice of plain data.
pub fn bytes_of<T: Pod>(data: &[T]) -> &[u8] {
    // SAFETY: Pod types have no padding, so every byte is initialized.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

/// A buffer with its own memory. Shaders reach it through its device
/// address ([`Buffer::address`]); host-visible ones are mapped for their
/// whole lifetime and accessed with [`Buffer::write`] / [`Buffer::read`].
///
/// The CPU and GPU can race on a mapped buffer: writing one the GPU is still
/// reading is a logic error (wrong pixels), not memory unsafety, because the
/// copies go through raw pointers into memory Rust never hands out
/// references to.
pub struct Buffer {
    pub(super) raw: vk::Buffer,
    size: u64,
    address: u64,
    memory: Memory,
    device: Arc<Device>,
}

impl Buffer {
    /// A buffer of `size` bytes. Storage and device-address usage are always
    /// included; add e.g. `TRANSFER_DST` for fills and copies.
    pub fn new(
        device: &Arc<Device>,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: Location,
    ) -> Result<Arc<Self>> {
        if size == 0 {
            bail!("zero-sized buffer");
        }
        let dev = &device.raw;
        let usage = usage
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let raw = unsafe { dev.create_buffer(&info, None) }.context("vkCreateBuffer")?;
        let requirements = unsafe { dev.get_buffer_memory_requirements(raw) };
        let memory =
            match Memory::allocate(device, requirements, location, Dedicated::Buffer(raw), true) {
                Ok(m) => m,
                Err(e) => {
                    unsafe { dev.destroy_buffer(raw, None) };
                    return Err(e);
                }
            };
        let address = unsafe {
            if let Err(e) = dev.bind_buffer_memory(raw, memory.raw, 0) {
                dev.destroy_buffer(raw, None);
                return Err(e).context("vkBindBufferMemory");
            }
            dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
        };
        Ok(Arc::new(Self {
            raw,
            size,
            address,
            memory,
            device: device.clone(),
        }))
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// The GPU address shaders use for this buffer (`buffer_reference`).
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The mapping, checked against `[offset, offset + len)`.
    fn span(&self, offset: u64, len: usize) -> Result<*mut u8> {
        let Some(base) = self.memory.mapped() else {
            bail!("buffer is not host-visible");
        };
        match offset.checked_add(len as u64) {
            Some(end) if end <= self.size => {}
            _ => bail!("range {offset}+{len} outside buffer of {} bytes", self.size),
        }
        // SAFETY: in bounds of the mapping, checked above.
        Ok(unsafe { base.as_ptr().add(offset as usize) })
    }

    /// Copy `data` into the buffer at byte `offset`.
    pub fn write<T: Pod>(&self, offset: u64, data: &[T]) -> Result<()> {
        let bytes = bytes_of(data);
        let dst = self.span(offset, bytes.len())?;
        // SAFETY: dst has room for bytes.len() (span); the mapping never
        // overlaps Rust memory.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
        Ok(())
    }

    /// Copy `count` values out of the buffer from byte `offset`.
    pub fn read<T: Pod>(&self, offset: u64, count: usize) -> Result<Vec<T>> {
        let len = count
            .checked_mul(size_of::<T>())
            .context("read size overflows")?;
        let src = self.span(offset, len)?;
        let mut out = Vec::<T>::with_capacity(count);
        // SAFETY: src holds len bytes; out has capacity for count values,
        // and any bytes are a valid T (Pod). Byte copies need no alignment.
        unsafe {
            ptr::copy_nonoverlapping(src, out.as_mut_ptr().cast::<u8>(), len);
            out.set_len(count);
        }
        Ok(out)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: command buffers keep an Arc while in flight, so the GPU is
        // done with it. `memory` is freed after this, as a later field.
        unsafe { self.device.raw.destroy_buffer(self.raw, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_of_is_little_endian_and_unpadded() {
        assert_eq!(bytes_of(&[1u32, 0x0302]), [1, 0, 0, 0, 2, 3, 0, 0]);
        assert_eq!(bytes_of(&[[1u16, 2]; 2]).len(), 8);
        assert!(bytes_of::<u64>(&[]).is_empty());
    }
}
