//! Configuration: a Lua program (`init.lua`) that sets vellum's settings.
//!
//! - [`settings`]: what can be set, as plain Rust data. The rest of vellum
//!   sees only [`Settings`].
//! - `api`: everything Lua sees (the `vellum` table) and running the
//!   config.
//! - [`thread`]: the script thread the Lua state lives on, so the event
//!   loop never runs Lua.
//!
//! Lua comes from `mlua` (LuaJIT): hand-written bindings would have to get
//! Lua's stack discipline, longjmp errors and GC-owned values right, which
//! is what mlua is for.

mod api;
pub mod settings;
#[cfg(test)]
mod tests;
pub mod thread;

use std::ffi::OsString;
use std::path::PathBuf;

pub use settings::Settings;
pub use thread::{Reply, Request, Script};

use crate::cli::ConfigFile;

/// The config to run for `choice`: the usual `init.lua` only if it exists
/// (no config is fine), a path given on the command line even if it
/// doesn't (so the user hears about it).
pub fn path(choice: &ConfigFile) -> Option<PathBuf> {
    match choice {
        ConfigFile::Default => default_path(|k| std::env::var_os(k)).filter(|p| p.exists()),
        ConfigFile::Path(p) => Some(p.clone()),
        ConfigFile::None => None,
    }
}

/// `$XDG_CONFIG_HOME/vellum/init.lua`, else `~/.config/vellum/init.lua`.
fn default_path(env: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let base = env("XDG_CONFIG_HOME")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .or_else(|| env("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("vellum/init.lua"))
}
