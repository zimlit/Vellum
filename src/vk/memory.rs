//! Device memory: choosing a memory type and making dedicated allocations.
//!
//! Vellum has few GPU resources and they live long, so every buffer and
//! image gets its own `VkDeviceMemory` (a dedicated allocation) instead of
//! being suballocated from a larger block. Drivers handle a few dozen
//! allocations without trouble; revisit if counts grow.

use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use ash::vk;

use super::Device;

/// What the memory is for, which decides the memory type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    /// Only the GPU touches it (VRAM).
    Device,
    /// Written by the CPU, read by the GPU. Mapped for the allocation's
    /// lifetime. Prefers VRAM the CPU can see (resizable BAR), so the GPU
    /// doesn't read across PCIe.
    Upload,
    /// Written by the GPU, read by the CPU. Mapped, and preferably cached on
    /// the CPU side, since uncached reads are very slow.
    Readback,
}

impl Location {
    /// `(required, preferred, avoided)` property flags.
    fn flags(
        self,
    ) -> (
        vk::MemoryPropertyFlags,
        vk::MemoryPropertyFlags,
        vk::MemoryPropertyFlags,
    ) {
        use vk::MemoryPropertyFlags as F;
        match self {
            // Leave CPU-visible VRAM (often a small window) to uploads.
            Location::Device => (F::DEVICE_LOCAL, F::empty(), F::HOST_VISIBLE),
            // Coherent: no explicit flushes or invalidates.
            Location::Upload => (
                F::HOST_VISIBLE | F::HOST_COHERENT,
                F::DEVICE_LOCAL,
                F::HOST_CACHED,
            ),
            Location::Readback => (
                F::HOST_VISIBLE | F::HOST_COHERENT,
                F::HOST_CACHED,
                F::empty(),
            ),
        }
    }

    fn mapped(self) -> bool {
        self != Location::Device
    }
}

/// The best memory type among `type_bits` (from `VkMemoryRequirements`)
/// for `location`: it must have all required flags; among those, the most
/// preferred flags and fewest avoided ones win, then the lowest index
/// (drivers list better types first).
pub fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    location: Location,
) -> Option<u32> {
    let (required, preferred, avoided) = location.flags();
    let types = &props.memory_types[..props.memory_type_count as usize];
    let score = |flags: vk::MemoryPropertyFlags| {
        (flags & preferred).as_raw().count_ones() as i32
            - (flags & avoided).as_raw().count_ones() as i32
    };
    (0..types.len() as u32)
        .filter(|&i| type_bits & (1 << i) != 0)
        .filter(|&i| types[i as usize].property_flags.contains(required))
        // max_by_key keeps the last maximum; reverse so the first wins.
        .rev()
        .max_by_key(|&i| score(types[i as usize].property_flags))
}

/// The resource a dedicated allocation is for.
pub(super) enum Dedicated {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

/// One `VkDeviceMemory`, mapped for its whole lifetime if host-visible.
pub(super) struct Memory {
    pub(super) raw: vk::DeviceMemory,
    mapped: Option<NonNull<u8>>,
    device: Arc<Device>,
}

// SAFETY: the mapping is plain memory owned by this allocation; access
// through it is synchronized by the callers (see `Buffer`).
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

impl Memory {
    pub(super) fn allocate(
        device: &Arc<Device>,
        requirements: vk::MemoryRequirements,
        location: Location,
        dedicated: Dedicated,
        device_address: bool,
    ) -> Result<Self> {
        let type_index = find_memory_type(
            &device.memory_properties,
            requirements.memory_type_bits,
            location,
        )
        .ok_or_else(|| anyhow!("no memory type for {location:?}"))?;
        let mut dedicated_info = match dedicated {
            Dedicated::Buffer(b) => vk::MemoryDedicatedAllocateInfo::default().buffer(b),
            Dedicated::Image(i) => vk::MemoryDedicatedAllocateInfo::default().image(i),
        };
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(type_index)
            .push_next(&mut dedicated_info);
        if device_address {
            info = info.push_next(&mut flags_info);
        }
        let dev = &device.raw;
        // SAFETY: info references live locals.
        let raw = unsafe { dev.allocate_memory(&info, None) }.with_context(|| {
            format!(
                "vkAllocateMemory ({} bytes, {location:?})",
                requirements.size
            )
        })?;
        let mut memory = Self {
            raw,
            mapped: None,
            device: device.clone(),
        };
        if location.mapped() {
            // SAFETY: host-visible memory, mapped once, never mapped twice.
            let ptr =
                unsafe { dev.map_memory(raw, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                    .context("vkMapMemory")?;
            memory.mapped = NonNull::new(ptr.cast());
        }
        Ok(memory)
    }

    /// The CPU address of the mapping, if host-visible.
    pub(super) fn mapped(&self) -> Option<NonNull<u8>> {
        self.mapped
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: the owning resource is destroyed first (field order in the
        // owners); freeing also unmaps.
        unsafe { self.device.raw.free_memory(self.raw, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::MemoryPropertyFlags as F;

    /// Memory types like a discrete AMD GPU with resizable BAR.
    fn discrete() -> vk::PhysicalDeviceMemoryProperties {
        let flags = [
            F::DEVICE_LOCAL,                                      // 0: VRAM
            F::HOST_VISIBLE | F::HOST_COHERENT,                   // 1: system RAM, uncached
            F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT, // 2: BAR
            F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,  // 3: system RAM, cached
        ];
        let mut props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: flags.len() as u32,
            ..Default::default()
        };
        for (t, f) in props.memory_types.iter_mut().zip(flags) {
            t.property_flags = f;
        }
        props
    }

    #[test]
    fn device_prefers_vram_the_cpu_cannot_see() {
        assert_eq!(
            find_memory_type(&discrete(), 0b1111, Location::Device),
            Some(0)
        );
        // Only BAR allowed: still acceptable.
        assert_eq!(
            find_memory_type(&discrete(), 0b0100, Location::Device),
            Some(2)
        );
    }

    #[test]
    fn upload_prefers_bar() {
        assert_eq!(
            find_memory_type(&discrete(), 0b1111, Location::Upload),
            Some(2)
        );
        assert_eq!(
            find_memory_type(&discrete(), 0b1011, Location::Upload),
            Some(1)
        );
    }

    #[test]
    fn readback_prefers_cached() {
        assert_eq!(
            find_memory_type(&discrete(), 0b1111, Location::Readback),
            Some(3)
        );
    }

    #[test]
    fn no_type_meets_requirements() {
        assert_eq!(
            find_memory_type(&discrete(), 0b0001, Location::Upload),
            None
        );
        assert_eq!(find_memory_type(&discrete(), 0, Location::Device), None);
    }

    #[test]
    fn ties_go_to_the_lowest_index() {
        let mut props = discrete();
        props.memory_types[1].property_flags = F::DEVICE_LOCAL;
        assert_eq!(find_memory_type(&props, 0b0011, Location::Device), Some(0));
    }
}
