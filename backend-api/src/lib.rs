//! The contract between the model-agnostic gateway and a model backend.
//!
//! The gateway owns everything client-facing (HTTP API, auth, image fetching
//! and decoding, Triton transport, response formatting). A backend owns
//! everything model-specific:
//!
//! 1. reading its own metadata from the Triton model config ([`Factory`]),
//! 2. turning a decoded RGB image into the model's input tensor
//!    ([`Backend::preprocess`]),
//! 3. decoding the raw output tensors into detections in original-image pixel
//!    coordinates ([`Backend::postprocess`]).
//!
//! A new model family is a new crate under `backends/<name>/` that exports a
//! `FAMILY` string and a `create` function matching [`Factory`], registered
//! in `gateway/src/registry.rs`.

use serde::Deserialize;
use serde_json::Value;

pub use image;
use image::RgbImage;

pub mod util;

/// Errors a backend can report. The gateway maps `InvalidModel` to a startup
/// failure and `BadOutput` to an upstream (503) error.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The Triton model config / metadata is not usable by this backend.
    #[error("{0}")]
    InvalidModel(String),
    /// Triton returned tensors that do not match what the backend expects.
    #[error("{0}")]
    BadOutput(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;

/// What the gateway knows about the served model when constructing a backend.
pub struct ModelInfo<'a> {
    /// Triton model name.
    pub name: &'a str,
    /// Parsed JSON of `parameters.metadata.string_value` from `config.pbtxt`
    /// (written by the backend's own exporter).
    pub metadata: &'a Value,
    /// Triton `max_batch_size` (0 = the graph carries an explicit batch dim).
    pub max_batch_size: u64,
}

/// Constructor every backend crate exports as `pub fn create`.
pub type Factory = fn(&ModelInfo) -> Result<Box<dyn Backend>>;

/// One FP32 input tensor ready to send to Triton.
#[derive(Debug, Clone)]
pub struct InputTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub data: Vec<f32>,
}

/// One output tensor returned by Triton, converted to f32.
#[derive(Debug, Clone)]
pub struct OutputTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub data: Vec<f32>,
}

/// Affine mapping from network-input pixel space back to original-image
/// pixels: `orig_x = (net_x - dx) / sx`, `orig_y = (net_y - dy) / sy`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxMap {
    pub sx: f64,
    pub sy: f64,
    pub dx: f64,
    pub dy: f64,
    pub orig_w: u32,
    pub orig_h: u32,
}

impl BoxMap {
    /// Map an `xyxy` box from network pixel space to original pixels.
    pub fn to_original(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64, f64, f64) {
        (
            (x1 - self.dx) / self.sx,
            (y1 - self.dy) / self.sy,
            (x2 - self.dx) / self.sx,
            (y2 - self.dy) / self.sy,
        )
    }
}

/// Output of [`Backend::preprocess`].
#[derive(Debug, Clone)]
pub struct Prepared {
    pub input: InputTensor,
    pub map: BoxMap,
}

/// Request-level decode parameters.
#[derive(Debug, Clone, Copy)]
pub struct DecodeParams {
    /// Minimum confidence to keep a detection.
    pub conf: f64,
    /// IoU threshold (only meaningful for backends that run NMS).
    pub iou: f64,
    /// Maximum detections to return.
    pub max_det: usize,
}

/// A detection in ORIGINAL image pixel coordinates, clamped to the image.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub class: i64,
    pub confidence: f64,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// A model backend. Implementations must be cheap to share across threads;
/// `preprocess` runs on a blocking worker thread.
pub trait Backend: Send + Sync {
    /// Family identifier (matches the `family` metadata field).
    fn family(&self) -> &'static str;
    /// Class names indexed by class id.
    fn class_names(&self) -> &[String];
    /// Triton output tensor names to request, in order.
    fn output_names(&self) -> &[String];
    /// Native network input size `(width, height)`.
    fn input_size(&self) -> (u32, u32);
    /// Decoded RGB image -> model input tensor + box mapping.
    fn preprocess(&self, image: &RgbImage) -> Prepared;
    /// Raw Triton outputs -> detections in original pixels, sorted by
    /// descending confidence and capped at `params.max_det`.
    fn postprocess(
        &self,
        outputs: &[OutputTensor],
        map: &BoxMap,
        params: &DecodeParams,
    ) -> Result<Vec<Detection>>;

    /// Class name for an id (falls back to `class_<id>`).
    fn class_name(&self, class: i64) -> String {
        usize::try_from(class)
            .ok()
            .and_then(|i| self.class_names().get(i))
            .cloned()
            .unwrap_or_else(|| format!("class_{class}"))
    }
}

/// Metadata fields shared by all backends (each backend may read more).
#[derive(Debug, Clone, Deserialize)]
pub struct CommonMeta {
    #[serde(default)]
    pub names: Vec<String>,
    pub imgsz: u32,
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub outputs: Option<Vec<String>>,
}

impl CommonMeta {
    pub fn parse(info: &ModelInfo) -> Result<Self> {
        let meta: CommonMeta = serde_json::from_value(info.metadata.clone()).map_err(|e| {
            BackendError::InvalidModel(format!("model '{}' metadata: {e}", info.name))
        })?;
        if meta.imgsz == 0 {
            return Err(BackendError::InvalidModel(format!(
                "model '{}' metadata has imgsz=0",
                info.name
            )));
        }
        Ok(meta)
    }
}
