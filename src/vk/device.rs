use std::ffi::{CStr, c_char};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::{ext, khr, vk};

use super::Instance;

const DEVICE_EXTENSIONS: &[&CStr] = &[
    khr::external_memory_fd::NAME,
    ext::external_memory_dma_buf::NAME,
    ext::image_drm_format_modifier::NAME,
    ext::queue_family_foreign::NAME,
    khr::external_semaphore_fd::NAME,
];

/// A logical device with one graphics queue.
pub struct Device {
    pub(super) raw: ash::Device,
    pub(super) physical: vk::PhysicalDevice,
    pub(super) queue: vk::Queue,
    pub(super) queue_family: u32,
    pub(super) memory_fd: khr::external_memory_fd::Device,
    pub(super) semaphore_fd: khr::external_semaphore_fd::Device,
    pub(super) instance: Arc<Instance>,
    name: String,
}

impl Device {
    /// Create a device on the GPU whose DRM primary node is `major:minor`
    /// (the card we do KMS on), so rendered buffers can be scanned out
    /// without crossing GPUs.
    pub fn for_drm_node(instance: &Arc<Instance>, major: u32, minor: u32) -> Result<Arc<Self>> {
        let inst = &instance.raw;
        let physicals =
            unsafe { inst.enumerate_physical_devices() }.context("vkEnumeratePhysicalDevices")?;

        for physical in physicals {
            let extensions = unsafe { inst.enumerate_device_extension_properties(physical) }?;
            let has = |name: &CStr| {
                extensions
                    .iter()
                    .any(|e| e.extension_name_as_c_str() == Ok(name))
            };
            if !has(ext::physical_device_drm::NAME) {
                continue;
            }

            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
            // SAFETY: the extension is supported, so the chained struct is valid.
            unsafe { inst.get_physical_device_properties2(physical, &mut props2) };
            let props = props2.properties;
            let name = props
                .device_name_as_c_str()
                .unwrap_or(c"<unnamed>")
                .to_string_lossy()
                .into_owned();

            if drm.has_primary == vk::FALSE
                || drm.primary_major != i64::from(major)
                || drm.primary_minor != i64::from(minor)
            {
                continue;
            }
            if props.api_version < vk::API_VERSION_1_3 {
                bail!(
                    "{name} only supports Vulkan {}.{}; need 1.3",
                    vk::api_version_major(props.api_version),
                    vk::api_version_minor(props.api_version)
                );
            }
            if let Some(missing) = DEVICE_EXTENSIONS.iter().find(|e| !has(e)) {
                bail!("{name} lacks {}", missing.to_string_lossy());
            }
            return Self::create(instance, physical, name);
        }
        bail!("no Vulkan device matches DRM node {major}:{minor}")
    }

    fn create(
        instance: &Arc<Instance>,
        physical: vk::PhysicalDevice,
        name: String,
    ) -> Result<Arc<Self>> {
        let inst = &instance.raw;
        let families = unsafe { inst.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| anyhow!("{name} has no graphics queue"))?
            as u32;

        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let extensions: Vec<*const c_char> = DEVICE_EXTENSIONS.iter().map(|e| e.as_ptr()).collect();
        // Both are mandatory in Vulkan 1.3.
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features13);

        // SAFETY: physical belongs to inst; info references live locals.
        let raw = unsafe { inst.create_device(physical, &info, None) }
            .with_context(|| format!("vkCreateDevice on {name}"))?;
        let queue = unsafe { raw.get_device_queue(queue_family, 0) };

        Ok(Arc::new(Self {
            memory_fd: khr::external_memory_fd::Device::new(inst, &raw),
            semaphore_fd: khr::external_semaphore_fd::Device::new(inst, &raw),
            raw,
            physical,
            queue,
            queue_family,
            instance: instance.clone(),
            name,
        }))
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: children hold Arc<Device>, so none remain; wait for any
        // work still queued before tearing down.
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.raw.destroy_device(None);
        }
    }
}
