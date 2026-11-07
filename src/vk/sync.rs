use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use ash::vk;

use super::Device;

/// CPU-waitable completion signal for a queue submission.
pub(super) struct Fence {
    pub(super) raw: vk::Fence,
    device: Arc<Device>,
}

impl Fence {
    pub(super) fn new(device: &Arc<Device>) -> Result<Self> {
        let raw = unsafe {
            device
                .raw
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("vkCreateFence")?;
        Ok(Self {
            raw,
            device: device.clone(),
        })
    }

    pub(super) fn wait(&self) -> Result<()> {
        unsafe { self.device.raw.wait_for_fences(&[self.raw], true, u64::MAX) }
            .context("vkWaitForFences")
    }

    pub(super) fn reset(&self) -> Result<()> {
        unsafe { self.device.raw.reset_fences(&[self.raw]) }.context("vkResetFences")
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // SAFETY: the owning CommandBuffer waits on this fence before drop.
        unsafe { self.device.raw.destroy_fence(self.raw, None) }
    }
}

/// A binary semaphore that can be exported as a sync_file, for handing GPU
/// completion to the kernel (KMS `IN_FENCE_FD`).
pub struct Semaphore {
    pub(super) raw: vk::Semaphore,
    /// Set by a submission that signals it, cleared by export. Exporting a
    /// sync_file requires a pending signal, and signaling twice without a
    /// wait in between is invalid, so both are checked against this.
    pub(super) signal_pending: AtomicBool,
    device: Arc<Device>,
}

impl Semaphore {
    pub fn new_exportable(device: &Arc<Device>) -> Result<Arc<Self>> {
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        let raw =
            unsafe { device.raw.create_semaphore(&info, None) }.context("vkCreateSemaphore")?;
        Ok(Arc::new(Self {
            raw,
            signal_pending: AtomicBool::new(false),
            device: device.clone(),
        }))
    }

    /// Export the pending signal as a sync_file fd. Sync-file export has copy
    /// semantics and leaves the semaphore unsignaled, ready for reuse.
    /// Returns `None` if the work already finished (the driver may return no
    /// fd for an already-signaled payload).
    pub fn export_sync_file(&self) -> Result<Option<OwnedFd>> {
        if !self.signal_pending.load(Ordering::Acquire) {
            bail!("semaphore has no pending signal to export");
        }
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.raw)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let fd = unsafe { self.device.semaphore_fd.get_semaphore_fd(&info) }
            .context("vkGetSemaphoreFdKHR")?;
        self.signal_pending.store(false, Ordering::Release);
        // SAFETY: a non-negative return is a new fd owned by us.
        Ok((fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) }))
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        // SAFETY: submissions keep an Arc to the semaphores they signal until
        // their fence completes.
        unsafe { self.device.raw.destroy_semaphore(self.raw, None) }
    }
}
