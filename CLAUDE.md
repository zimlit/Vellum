# CLAUDE.md

## Project overview

**Vellum** is a terminal emulator that runs **directly on the Linux console**, using DRM/KMS with no X11 or Wayland session underneath. It is a display server for one app: it owns the display, input devices and VT, renders with **Vulkan**, and runs shells through PTYs. Later it will embed a **minimal, single-surface Wayland compositor** so graphical apps (Emacs pgtk, mpv, SDL games) can run fullscreen from the shell without a full desktop compositor.

Linux only. Learning the stack all the way down is an explicit goal: prefer understanding and control over convenience libraries when there's a choice.

Language: **Rust** (all crate choices below assume it).

## Architecture

```
┌──────────────────────── front-end process ────────────────────────┐
│ input: libinput + xkbcommon → key encoding → PTY master           │
│ session: libseat (DRM master, input devices, VT switching)        │
│ display: own KMS (atomic commits, page flips, planes)             │
│ render: Vulkan (ash) → dma-buf images → KMS framebuffers          │
│ text: fontconfig → HarfRust (shaping) → skrifa (outlines)         │
│       → own GPU compute rasterizer → glyph atlas                  │
│ term core: VT parser + screen grid + scrollback                   │
│ (later) Wayland compositor: socket-activated, single surface      │
└───────────────────────────────────────────────────────────────────┘
              │ (socket)
┌──────── PTY session process (tmux-style) ────────┐
│ owns PTYs/shells; front end reattaches after a crash │
└──────────────────────────────────────────────────┘
```

## Technology decisions

| Area | Choice | Notes |
|---|---|---|
| GPU API | **Vulkan via `ash`** | Chosen over wgpu for control over display timing and VT handling. Performance would be equal for a terminal either way; the reason is control. All `unsafe` Vulkan lives in a safe wrapper (`src/vk/`). |
| GPU memory | **Own small allocator** | Few, long-lived allocations: choose the memory type ourselves, one dedicated allocation per resource, persistent mapping for upload buffers. Revisit (suballocation) if counts grow, e.g. with Wayland clients. |
| Shaders | **GLSL → SPIR-V with `glslc`** in `build.rs` | Needs shaderc (`glslc`) at build time. |
| Display | **Own KMS, atomic modesetting** (libdrm / `drm` crate) | Do NOT use `VK_KHR_display` / `VK_EXT_acquire_drm_display`: it gives KMS control to the driver, VT switching is poorly tested, and driver support varies. |
| Render targets | GBM-allocated (or Vulkan `VK_EXT_image_drm_format_modifier`) **dma-bufs**, imported into Vulkan with `VK_EXT_external_memory_dma_buf`, registered with `drmModeAddFB2WithModifiers` | The same import path is reused later for Wayland client buffers. |
| Input | `libinput` (`input` crate) + `xkbcommon` | evdev gives physical keycodes; xkbcommon turns them into keysyms and UTF-8. libinput opens devices through the libseat session (`src/input.rs`). Keymap from `--xkb-*` flags, else `XKB_DEFAULT_*`; compose from the locale. |
| Key encoding | **Legacy xterm + kitty keyboard protocol** (`src/keys.rs`) | Pure function from key event + modes to bytes. Kitty flags have a push/pop stack per screen. Text comes from the keysym, not xkb's Ctrl transform; Ctrl combinations use xterm's table, falling back to the US layout's key. |
| Session / VT | **libseat** (logind or seatd) | Lets it run without root; handles DRM master, device open/revoke and VT switching. |
| Configuration | **Lua (LuaJIT) via `mlua`**, `init.lua` in `$XDG_CONFIG_HOME/vellum/` (`src/config/`) | A deliberate exception to own bindings: Lua's stack discipline, longjmp errors and GC-owned values are what mlua makes safe. System LuaJIT (pkg-config), safe libraries only (no `ffi`/`debug`). The Lua state lives on a **script thread**: the event loop never runs Lua; requests and replies are plain Rust data over channels plus an eventfd. No instruction hook (it would keep LuaJIT in its interpreter); a request unanswered after `script_timeout` gets a warning. Settings are read with serde from the merged `vellum.set{...}` tables. CLI flags win over the config. Errors show in a notice bar over the bottom row. |
| Font discovery | **system fontconfig**, own FFI bindings | Also read per-font settings: `hintstyle`, `antialias`, `rgba`, `lcdfilter`. Hand-written bindings like `gbm.rs`. |
| Shaping | **HarfRust** | HarfBuzz's official Rust port, built on `read-fonts`. Preferred over swash's shaper, which covers less. |
| Outlines / hinting | **skrifa** (fontations) | FreeType-quality hinted outlines (TrueType bytecode, CFF, autohinter). It does NOT rasterize. |
| Rasterization | **Own, as a GPU compute shader** | See "Glyph rasterizer" below. |
| Terminal core | **Own parser and screen** (`src/term/`) | Paul Williams' DEC/VT500 state machine (vt100.net), UTF-8 only, colon subparameters kept. Behavior follows xterm where terminals differ. |
| Terminfo | **Own entry `xterm-vellum`** (`terminfo/xterm-vellum.ti`) | Compiled by `tic` in `build.rs`, embedded, installed by the session server under `$XDG_DATA_HOME/vellum/terminfo` with `TERMINFO` pointing there (falls back to `xterm-256color`). Lists only what vellum implements; tests feed every capability through the terminal core and every key through the encoder. **Must compile to ≤ 4096 bytes**: narrow ncurses (`libtinfo.so`, used by Emacs and bash) treats larger entries as undefined; `build.rs` enforces it. Keep the XTGETTCAP table (`src/term/reply.rs`) in step. Remote hosts need `infocmp -x xterm-vellum \| ssh HOST tic -x -`. |

**Dependency warning:** HarfRust, skrifa and read-fonts often make breaking changes in *minor* releases by bumping `read-fonts`. Pin all three to versions that share one `read-fonts`, and check `cargo tree -d` after every upgrade. Currently skrifa `=0.49.0`, which matches harfrust 0.15 (read-fonts 0.46).

## Text pipeline design

### Shaping (terminal-specific)
- **Fast path:** if ligatures are off and a run is plain ASCII in the primary font, map character → glyph through the cmap and skip shaping. This covers most output.
- Shape **per run** (consecutive cells in one row with the same font and style), never per cell, or ligatures and combining marks break.
- **Ignore shaper advances.** Use cluster values to map glyphs back to cells; the grid decides placement. A ligature glyph spans several cells.
- Do font fallback (via fontconfig) **before** shaping: split runs where the primary font lacks a character.
- Cache shaped runs keyed by `(text, font_id, style, features)`.
- Wide characters (CJK, emoji) take 2 cells.
- Draw **box-drawing, block and Powerline characters procedurally**, not from the font, so lines join seamlessly between cells (as kitty, Ghostty and WezTerm do).

### Glyph rasterizer (GPU compute, signed-area accumulation)
A GPU port of Raph Levien's **font-rs** `draw_line` and accumulate algorithm. Exact area coverage, no supersampling.

CPU side per glyph:
1. Get the hinted outline from skrifa at the target size (`OutlinePen`: move/line/quad/curve/close).
2. Flatten curves into lines at ~0.1–0.2 px tolerance (quads: `n = ceil(sqrt(dist / tol))` segments).
3. Flip y (font units are y-up), shift by **1 px padding**, emit the closing segment on `close()`.
4. Append segments to a shared line buffer plus a job record `{atlas_origin, size, accum_offset}`.

GPU side, two dispatches per batch:
1. **Deposit:** one thread per line segment; adds signed area into an accumulation grid with `atomicAdd` on **fixed-point `i32`** (scale 65536), since there are no float atomics in core. Row stride = `width + 2` (room for the spill cell).
2. **Resolve:** one workgroup per glyph, threads stride over rows; running sum across the row, `coverage = min(|sum|, 1)` (nonzero winding); write to the atlas.

Notes:
- Clear the accumulation buffer before each batch.
- Atlas format **RGBA8**: leaves room for LCD subpixel RGB coverage and color emoji in the same atlas.
- Keep coverage **linear** in the atlas; apply gamma, contrast and stem darkening in the text fragment shader so they can be tuned without re-rasterizing.
- Clamp coverage below ~1/512 to 0 if fixed-point haze shows up.
- **LCD subpixel (later):** rasterize at 3× horizontal, add a 5-tap FIR filter pass, store RGB. No dual-source blending needed: the terminal knows each cell's exact background, so blend `mix(bg, fg, coverage_rgb)` in the shader.
- Possible later: GPU curve flattening; merge passes into a tiled pass using workgroup shared memory (Vello-style).

### Cell renderer
- Glyph cache key: `(font_id, glyph_id, style, subpixel_offset)` → atlas rect + bearing.
- Per frame: pass 1 draws cell backgrounds; pass 2 is **one instanced draw** for glyphs (per-instance: cell index, atlas UV, fg/bg, flags; the vertex shader computes position from the cell index); then cursor and decorations.
- Redraw only changed cells (damage tracking).

## Display / session details

- **Restore the console on every exit path** (normal exit, panic, signals): KD_TEXT mode, keyboard mode, CRTC state. Otherwise the console is left black or frozen. Keep an **SSH session open** while developing.
- Without libseat, manual VT handling is: `KDSETMODE KD_GRAPHICS`, `KDSKBMODE K_OFF`, and `VT_SETMODE VT_PROCESS` with release/acquire signal handlers.
- **VT switch away:** stop atomic commits, drop DRM master, pause input. **Switch back:** take master, re-commit, full redraw. Build a switch-away/switch-back test early.
- Use flip-completion events for pacing; aim to render as late as possible before vblank (input latency matters more than throughput).
- Use plane-aware atomic commits from the start (primary plane for the terminal; later a cursor/overlay plane, and direct scanout of fullscreen Wayland client buffers).
- Pass KMS fences via sync files, Vulkan side via external semaphores.
- Resize/mode change → update the PTY window size (`TIOCSWINSZ`) so the kernel sends SIGWINCH to the child.

## Input and terminal behavior

- **Vellum's own keys:** Ctrl+Alt+Backspace exits cleanly (closing the shell). Shift+PgUp/PgDn (half a screen) and Shift+Home/End scroll the view on the primary screen; on the alternate screen the program gets them. Ctrl+Alt+F1–F12 switch VTs through the seat (the seat sets the console keyboard to `K_OFF`, so the kernel never sees them); repeats and releases are swallowed, and the VT already in front is left alone.
- **Across VT switches:** on disable, the display pauses, libinput is suspended (closing devices through the seat), the xkb state is reset (locks kept) and key repeat stops, *then* the disable is acknowledged. Releases made on the other VT never arrive, so without the reset Ctrl and Alt would stay held. On enable: full modeset, libinput resumed, full redraw. The PTY is read throughout.
- **Scrollback view:** stays on the same lines as output arrives; any key sent to the shell snaps back to live, except releases and modifier keys (else letting go of Shift under the kitty protocol would). Replies don't snap back. Reset by the alternate screen, ED 3 and RIS. The cursor is hidden while scrolled back.
- **Key repeat** is ours (timerfd, 600 ms then 25/s); holding a repeating key and pressing Shift keeps repeating, shifted.
- **Replies** to queries go through the same PTY write queue as keys, so order is kept: DA1/DA2, DSR/CPR, DECRQM, XTVERSION, OSC 4/10/11/12 color queries, kitty flags query, XTGETTCAP, DECRQSS, window size reports (CSI 14/16/18 t), focus events (1004: out when switched away, in when back).
- **Edge cell backgrounds** extend over the margin around the centred grid, so a program's background reaches the screen edge.
- **Known, not ours:** Emacs never writes the bottom-right cell when the terminal has `am` (its `tty_write_glyphs` rule), so that cell keeps vellum's default background. Workaround on the Emacs side: set the terminal background with OSC 11 to match the theme.

## Vulkan conventions

- **Validation layers plus synchronization validation always on** in debug builds.
- Sync and image-layout mistakes are the expected class of bugs; be explicit and conservative with barriers first, then optimize.
- Shaders are GLSL in `shaders/`, compiled to SPIR-V by `build.rs` with `glslc` and embedded in the binary.

## Planned: embedded Wayland compositor (after the terminal works)

- **In-process**, single fullscreen surface; no window management.
- **Socket-activated:** create the listening socket at startup (`$XDG_RUNTIME_DIR/wayland-N`) and export `WAYLAND_DISPLAY` to child shells; set up protocol state only on the first connection; free it when the last client leaves. Idle cost is one fd.
- Only a client that creates an `xdg_toplevel` and attaches a buffer takes over the screen (clipboard tools like `wl-copy` must not).
- Protocols: `wl_compositor`, `wl_subcompositor`, `wl_output`, `xdg_wm_base` (always fullscreen), `wl_shm`, `zwp_linux_dmabuf_v1`, `wl_seat` (forward existing libinput/xkb input), `wl_data_device_manager` (clipboard), and for games `relative_pointer`, `pointer_constraints`, `presentation_time`.
- Client dma-bufs: import into Vulkan, or **scan out directly on a KMS plane** when fullscreen and in a compatible format (gamescope's approach; best latency).
- **Lazy Xwayland:** hold the X11 sockets (`/tmp/.X11-unix/X<n>` + abstract), export `DISPLAY`, start Xwayland with `-listenfd` on first X connection. Do this last; it's the largest piece.
- Libraries: Smithay (use only its protocol parts, keep the ash renderer) or raw `wayland-server`.
- Test order: foot, mpv, Emacs pgtk, SDL3 games, then Xwayland.
- Crash isolation: the separate PTY session process keeps shells alive if the front end/compositor crashes.
- Workaround until then: run `cage <app>` or `gamescope` on another VT.

## Roadmap

1. **KMS bring-up:** open the device via libseat, find connector/mode/CRTC/plane, allocate dma-buf, atomic commit of a solid color, clean restore on exit.
2. **Vulkan bring-up:** instance/device with dma-buf import, render a clear color into the imported image, page-flip with fences.
3. **PTY:** `forkpty`/`openpty` into the session process; one `poll`/epoll loop over PTY, libinput, DRM and signalfd fds.
4. **Text v0:** fontconfig + skrifa outlines + the **GPU compute rasterizer** (see "Glyph rasterizer"; no temporary CPU rasterizer) into the atlas; instanced cell renderer; raw shell output. Validate the rasterizer against a CPU reference of the same algorithm in tests, and later by diffing against FreeType/swash output for the same font, size and hinting.
5. **Terminal core:** VT parser, grid, scrollback, alternate screen, SGR attributes.
6. **Input:** libinput + xkbcommon, key → escape-sequence encoding.
7. **VT switching:** survive Ctrl+Alt+Fn away and back repeatedly. Comes after input because the seat sets the console keyboard to `K_OFF`: the kernel never sees Ctrl+Alt+Fn, so we must detect it via libinput/xkb and call `libseat_switch_session`. Tested from SSH with `sudo scripts/vt-cycle.sh`.
8. **Configuration and extensibility:** user settings (fonts, colors, keymap, key bindings, scrollback, ...) and a way to extend vellum. Technology and design are decided when this step is planned.
9. **HarfRust shaping**, fallback fonts, ligatures, wide chars, procedural box drawing.
10. Polish: damage tracking, latency-tuned frame pacing, honoring fontconfig hinting/AA, LCD subpixel, color emoji (COLR via skrifa paint). A bigger notice area: today's notice bar is one line, cut to the screen width, which long messages (serde's "expected one of ..." lists grow with every setting) outgrow; make it multi-line and scrollable, like Emacs's minibuffer/echo area.
11. Embedded Wayland compositor (see above).

## References

- David Herrmann's `modeset` examples (modeset.c, -double-buffered, -vsync): KMS basics
- kmscon, fbterm, yaft: prior art for console terminals
- font-rs (Raph Levien): CPU accumulation rasterizer the compute shader is based on
- Vello: wgpu compute 2D renderer (tiled fine rasterization)
- Slug (Eric Lengyel): curve evaluation in the fragment shader, a different approach
- vt100.net DEC parser diagram: VT parser state machine
- wlroots Vulkan renderer, Smithay, gamescope, cage: dma-buf/KMS/compositor references
- Arcan: display server treating terminals and graphical clients the same way
