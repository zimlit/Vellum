//! Running the config: a fresh Lua state with the `vellum` API, then the
//! user's `init.lua`. Only the script thread calls this.
//!
//! The API so far:
//! - `vellum.set{...}`: settings (see [`Settings`]). Tables merge key by
//!   key into what earlier calls set, except [`ATOMIC_SETTINGS`], which are
//!   replaced whole. Each call is checked at once, so an error points at
//!   the call that caused it.
//! - `vellum.log.info/warn/error(...)`: lines in vellum's log.
//!
//! Only Lua's safe standard libraries are loaded (no `ffi` or `debug`): a
//! mistake in the config can't crash vellum while it holds the display.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use mlua::{Lua, LuaOptions, StdLib, Table, Value, Variadic};

use super::settings::{ATOMIC_SETTINGS, Settings};

/// A config that ran to the end, and the state it left (bindings and hooks
/// will live in it).
pub struct Loaded {
    pub lua: Lua,
    pub settings: Settings,
}

/// Run the config at `path` (`None`: no config, just the API). The error is
/// a message for the user, with the file and line when Lua knows them.
pub fn load(path: Option<&Path>) -> Result<Loaded, String> {
    let lua = Lua::new_with(StdLib::ALL_SAFE, LuaOptions::default()).map_err(|e| describe(&e))?;
    let settings = Rc::new(RefCell::new(Settings::default()));
    install_api(&lua, settings.clone(), path).map_err(|e| describe(&e))?;
    if let Some(path) = path {
        let source = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        lua.load(source)
            .set_name(format!("@{name}"))
            .exec()
            .map_err(|e| describe(&e))?;
    }
    let settings = settings.borrow().clone();
    Ok(Loaded { lua, settings })
}

fn install_api(lua: &Lua, out: Rc<RefCell<Settings>>, path: Option<&Path>) -> mlua::Result<()> {
    // `require` looks next to the config first.
    if let Some(dir) = path.and_then(Path::parent) {
        let package: Table = lua.globals().get("package")?;
        let old: String = package.get("path")?;
        let dir = dir.to_string_lossy();
        package.set("path", format!("{dir}/?.lua;{dir}/?/init.lua;{old}"))?;
    }

    let vellum = lua.create_table()?;
    // Everything `vellum.set` has been given, merged.
    let merged = lua.create_table()?;
    vellum.set(
        "set",
        lua.create_function(move |lua, table: Table| {
            merge(lua, &merged, &table, true)?;
            let settings = read_settings(&merged)
                .map_err(|msg| mlua::Error::runtime(format!("{}{msg}", caller_location(lua))))?;
            *out.borrow_mut() = settings;
            Ok(())
        })?,
    )?;

    let log = lua.create_table()?;
    for level in ["info", "warn", "error"] {
        log.set(
            level,
            lua.create_function(move |_, args: Variadic<Value>| {
                let text = args
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<mlua::Result<Vec<_>>>()?
                    .join(" ");
                match level {
                    "info" => tracing::info!(target: "vellum::config", "{text}"),
                    "warn" => tracing::warn!(target: "vellum::config", "{text}"),
                    _ => tracing::error!(target: "vellum::config", "{text}"),
                }
                Ok(())
            })?,
        )?;
    }
    vellum.set("log", log)?;
    lua.globals().set("vellum", vellum)
}

/// `init.lua:12: ` for the Lua code calling the current Rust function, as
/// Lua's own errors start; empty if unknown.
fn caller_location(lua: &Lua) -> String {
    lua.inspect_stack(1, |d| {
        let src = d.source().short_src?.into_owned();
        Some(format!("{src}:{}: ", d.current_line()?))
    })
    .flatten()
    .unwrap_or_default()
}

/// Merge `src` into `dst`: tables key by key, recursively, except atomic
/// settings (at the top level), which replace what was there. Tables are
/// copied, so later changes to the user's own tables don't leak in.
fn merge(lua: &Lua, dst: &Table, src: &Table, top: bool) -> mlua::Result<()> {
    for pair in src.pairs::<Value, Value>() {
        let (key, value) = pair?;
        let atomic =
            top && matches!(&key, Value::String(s) if ATOMIC_SETTINGS.iter().any(|a| s == a));
        match value {
            Value::Table(t) if atomic => dst.set(key, copy(lua, &t)?)?,
            Value::Table(t) => {
                let sub = match dst.get::<Value>(key.clone())? {
                    Value::Table(sub) => sub,
                    _ => {
                        let sub = lua.create_table()?;
                        dst.set(key, sub.clone())?;
                        sub
                    }
                };
                merge(lua, &sub, &t, false)?;
            }
            other => dst.set(key, other)?,
        }
    }
    Ok(())
}

fn copy(lua: &Lua, t: &Table) -> mlua::Result<Table> {
    let out = lua.create_table()?;
    for pair in t.pairs::<Value, Value>() {
        let (k, v) = pair?;
        match v {
            Value::Table(sub) => out.set(k, copy(lua, &sub)?)?,
            v => out.set(k, v)?,
        }
    }
    Ok(out)
}

/// The settings `table` describes, or what's wrong with it, naming the key
/// (`colors.palette.1: expected a color like "#rrggbb", got "red"`).
fn read_settings(table: &Table) -> Result<Settings, String> {
    let de = mlua::serde::Deserializer::new(Value::Table(table.clone()));
    let settings: Settings = serde_path_to_error::deserialize(de).map_err(|e| {
        let path = e.path().to_string();
        let inner = match e.into_inner() {
            mlua::Error::DeserializeError(msg) => msg,
            other => other.to_string(),
        };
        if path == "." {
            inner
        } else {
            format!("{path}: {inner}")
        }
    })?;
    settings.validate()?;
    Ok(settings)
}

/// A Lua error as one message for the user: the innermost cause, which
/// carries the file and line.
pub fn describe(e: &mlua::Error) -> String {
    match e {
        mlua::Error::CallbackError { cause, .. } | mlua::Error::WithContext { cause, .. } => {
            describe(cause)
        }
        mlua::Error::RuntimeError(msg) => msg.clone(),
        mlua::Error::SyntaxError { message, .. } => message.clone(),
        other => other.to_string(),
    }
}
