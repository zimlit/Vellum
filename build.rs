//! Compile the GLSL shaders in `shaders/` to SPIR-V with `glslc` (shaderc).
//! Output goes to `$OUT_DIR/shaders/<path>.spv`, embedded by `vk::spirv!`.
//!
//! The stage comes from the extension (`.comp`, `.vert`, `.frag`); files
//! with other extensions (e.g. `.glsl` includes) are not compiled on their
//! own. Set `GLSLC` to use a specific compiler binary.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const STAGES: &[&str] = &["comp", "vert", "frag"];

fn main() {
    let src_dir = Path::new("shaders");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("shaders");
    // A directory makes cargo rerun when anything under it changes.
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    let glslc = env::var_os("GLSLC").unwrap_or_else(|| "glslc".into());

    let mut sources = Vec::new();
    collect(src_dir, &mut sources);
    sources.sort();
    for src in sources {
        let rel = src.strip_prefix(src_dir).unwrap();
        let mut dst = out_dir.join(rel).into_os_string();
        dst.push(".spv");
        let dst = PathBuf::from(dst);
        fs::create_dir_all(dst.parent().unwrap()).unwrap();

        let output = Command::new(&glslc)
            .arg("--target-env=vulkan1.3")
            .arg("-O")
            .arg("-Werror")
            .arg("-I")
            .arg(src_dir)
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "running {}: {e}\nInstall shaderc (Gentoo: media-libs/shaderc) or set GLSLC.",
                    Path::new(&glslc).display()
                )
            });
        if !output.status.success() {
            panic!(
                "glslc failed on {}:\n{}",
                src.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| STAGES.contains(&e))
        {
            out.push(path);
        }
    }
}
