//! Safe wrapper over the parts of Vulkan (via ash) that vellum uses.
//!
//! All `unsafe` Vulkan calls live in this module. The public API is safe:
//! - Objects hold an `Arc` of their parent (image → device → instance), so a
//!   parent is never destroyed before its children.
//! - A [`CommandBuffer`] keeps every resource it references alive until the
//!   GPU has finished with it (tracked by its own fence), and waits for that
//!   fence before it is re-recorded or destroyed.
//! - [`Device`] waits for the GPU to go idle before it is destroyed.

mod command;
mod device;
mod image;
mod instance;
mod sync;

pub use ash::vk::ImageUsageFlags;
pub use command::CommandBuffer;
pub use device::Device;
pub use image::DmaBufImage;
pub use instance::Instance;
pub use sync::Semaphore;
// Returned by public methods; exported so callers can name them.
#[allow(unused_imports)]
pub use {command::Recorder, image::Modifier};
