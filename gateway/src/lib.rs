//! Model-agnostic, Ultralytics-compatible `/predict` gateway in front of
//! Triton. All model-specific logic lives in `backends/<family>/` crates and
//! is selected at startup through [`registry`].

pub mod api;
pub mod config;
pub mod error;
pub mod imaging;
pub mod registry;
pub mod response;
pub mod sources;
pub mod triton;
