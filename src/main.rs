//! Entry point: parse arguments, set up logging, and run as the front end
//! or (internally) as the PTY session server. See the library crate docs.

use std::process::ExitCode;

use vellum::cli::Args;
use vellum::{frontend, logging, server};

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("vellum: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(path) = args.session_server {
        logging::init(logging::SESSION_SERVER);
        return match server::main(path) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                tracing::error!("{e:#}");
                ExitCode::FAILURE
            }
        };
    }

    let mut log = logging::init(logging::FRONT_END);
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        log_file = ?log.file,
        "starting"
    );
    match frontend::run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e:#}");
            if let Some(console) = &mut log.console {
                // Started from a VT, which is back in text mode now that the
                // seat is released: say why we failed where the user is
                // looking. (Best effort; nothing to do if this fails.)
                use std::io::Write;
                let _ = writeln!(console, "vellum: {e:#}");
                if let Some(file) = &log.file {
                    let _ = writeln!(console, "vellum: log: {}", file.display());
                }
            }
            ExitCode::FAILURE
        }
    }
}
