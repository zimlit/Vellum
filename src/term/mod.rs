//! The terminal core: parsing what programs print and keeping the screen
//! state it describes.

pub mod parser;

pub use parser::{Params, Parser, Perform};
