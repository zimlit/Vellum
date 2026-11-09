//! Shared helpers for unit tests.
//!
//! Tests that need real hardware are marked `#[ignore = "needs a GPU ..."]`
//! so a plain `cargo test` stays hermetic; run them with
//! `cargo test -- --ignored` (or `--include-ignored`). They open the card
//! directly (no seat, no DRM master), so they work over SSH or from inside
//! another graphical session, but never modeset.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use drm::{ClientCapability, Device as _};

use crate::kms::{self, Card};

/// A fresh, empty temporary directory, removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("vellum-test-{}-{n}", std::process::id()));
        // Leftover from a crashed run with the same pid.
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("creating temp dir");
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The card vellum would drive, opened directly with atomic modesetting
/// enabled (needed to see all planes and atomic properties).
pub fn card() -> Card {
    let path = kms::find_card().expect("finding a DRM card");
    let card = Card::open_direct(&path).expect("opening the DRM card");
    card.set_client_capability(ClientCapability::Atomic, true)
        .expect("enabling atomic modesetting");
    card
}
