use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::{ext, vk};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

pub struct Instance {
    debug: Option<Debug>,
    pub(super) raw: ash::Instance,
    // Keeps libvulkan loaded; dropped after `raw` is destroyed.
    _entry: ash::Entry,
}

struct Debug {
    loader: ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Instance {
    /// Create a Vulkan 1.3 instance. With `validation`, enables the Khronos
    /// validation layer with synchronization validation if it is installed,
    /// and routes its messages to stderr.
    pub fn new(validation: bool) -> Result<Arc<Self>> {
        // SAFETY: loading libvulkan runs its initializers; nothing else is
        // racing with us on it.
        let entry = unsafe { ash::Entry::load() }.context("loading libvulkan")?;

        let version =
            unsafe { entry.try_enumerate_instance_version() }?.unwrap_or(vk::API_VERSION_1_0);
        if version < vk::API_VERSION_1_3 {
            bail!(
                "Vulkan loader only supports {}.{}; need 1.3",
                vk::api_version_major(version),
                vk::api_version_minor(version)
            );
        }

        let validation = validation && {
            let layers = unsafe { entry.enumerate_instance_layer_properties() }?;
            let found = layers
                .iter()
                .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
            if !found {
                eprintln!(
                    "vellum: {} is not installed; running without validation",
                    VALIDATION_LAYER.to_string_lossy()
                );
            }
            found
        };

        let app = vk::ApplicationInfo::default()
            .application_name(c"vellum")
            .engine_name(c"vellum")
            .api_version(vk::API_VERSION_1_3);
        let mut layers: Vec<*const c_char> = Vec::new();
        let mut extensions: Vec<*const c_char> = Vec::new();
        if validation {
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(ext::debug_utils::NAME.as_ptr());
            extensions.push(ext::validation_features::NAME.as_ptr());
        }
        let enabled = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled);
        let mut info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        if validation {
            info = info.push_next(&mut features);
        }

        // SAFETY: all pointers in `info` reference locals alive for the call.
        let raw = unsafe { entry.create_instance(&info, None) }.context("vkCreateInstance")?;

        let debug = if validation {
            let loader = ext::debug_utils::Instance::new(&entry, &raw);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            match unsafe { loader.create_debug_utils_messenger(&info, None) } {
                Ok(messenger) => Some(Debug { loader, messenger }),
                Err(e) => {
                    unsafe { raw.destroy_instance(None) };
                    return Err(e).context("vkCreateDebugUtilsMessengerEXT");
                }
            }
        } else {
            None
        };

        Ok(Arc::new(Self {
            debug,
            raw,
            _entry: entry,
        }))
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: every child object holds an Arc<Instance>, so none remain.
        unsafe {
            if let Some(debug) = self.debug.take() {
                debug
                    .loader
                    .destroy_debug_utils_messenger(debug.messenger, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "error"
    } else {
        "warning"
    };
    // SAFETY: the layer passes a valid callback-data struct whose message is
    // a NUL-terminated string (or null).
    let message = unsafe {
        data.as_ref()
            .filter(|d| !d.p_message.is_null())
            .map(|d| CStr::from_ptr(d.p_message).to_string_lossy())
    };
    eprintln!(
        "vulkan {level}: {}",
        message.as_deref().unwrap_or("<no message>")
    );
    vk::FALSE
}
