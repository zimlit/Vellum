//! Vellum: a terminal emulator that runs directly on the Linux console
//! (DRM/KMS + Vulkan, no X11 or Wayland), with its shells in a separate
//! PTY session process. See CLAUDE.md for the architecture and roadmap.
//!
//! The binary (`main.rs`) only parses arguments and picks a role: the front
//! end ([`frontend`]) or the session server ([`server`]).

pub mod cli;
pub mod dmabuf;
pub mod event;
pub mod font;
pub mod frontend;
pub mod gbm;
pub mod grid;
pub mod kms;
pub mod logging;
pub mod pty;
pub mod raster;
pub mod scanout;
pub mod server;
pub mod session;
pub mod signals;
#[cfg(test)]
mod testutil;
pub mod text;
pub mod vk;
