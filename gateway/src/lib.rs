//! Model-agnostic gateway in front of Triton. Each task is served through an
//! existing public API ([`endpoints`]): Ultralytics-compatible `/predict` for
//! detection, Google Cloud Vision-compatible `/v1/images:annotate` for OCR.
//! All model-specific logic lives in `backends/<family>/` crates and is
//! selected at startup through [`registry`].

pub mod config;
pub mod endpoints;
pub mod error;
pub mod imaging;
pub mod input;
pub mod ocr;
pub mod registry;
pub mod response;
pub mod sources;
pub mod state;
pub mod triton;
