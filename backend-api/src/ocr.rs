//! The `ocr` task contract: text detection + recognition.
//!
//! OCR is two Triton models chained by CPU work in between, so an OCR backend
//! exposes the pipeline as stages and the gateway drives it (and owns the
//! transport, as for detection):
//!
//! 1. [`OcrBackend::preprocess_detect`] → infer on [`OcrBackend::detector`] →
//!    [`OcrBackend::postprocess_detect`]: text regions (quads, scores);
//! 2. optionally (full OCR, not "regions only"):
//!    [`OcrBackend::preprocess_recognize`] crops the regions into batches →
//!    infer each on [`OcrBackend::recognizer`] →
//!    [`OcrBackend::postprocess_recognize`]: text per region.
//!
//! The detector is the Triton model named in the gateway config; it lists the
//! recognizer in its `companions` metadata, so it arrives in
//! [`ModelInfo::companions`].

use image::RgbImage;

use crate::{BoxMap, InputTensor, ModelInfo, OutputTensor, Prepared, Result};

/// Constructor every `ocr` backend crate exports as `pub fn create`.
pub type OcrFactory = fn(&ModelInfo) -> Result<Box<dyn OcrBackend>>;

/// Four corners in original-image pixels in *reading* order: top-left,
/// top-right, bottom-right, bottom-left of the text as it is read. So
/// `q[0] → q[1]` is the reading direction, also for rotated or vertical text.
pub type Quad = [(f64, f64); 4];

/// One Triton model the pipeline calls.
#[derive(Debug, Clone)]
pub struct Stage {
    pub model: String,
    /// Output tensor names to request, in order.
    pub outputs: Vec<String>,
}

/// Request-level OCR parameters.
#[derive(Debug, Clone, Copy)]
pub struct OcrParams {
    /// Minimum detector score to keep a region (`None`: the model's default).
    pub min_region_score: Option<f64>,
    /// Maximum regions kept per image.
    pub max_regions: usize,
}

/// A detected text region.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRegion {
    pub quad: Quad,
    /// Detector confidence, 0..1.
    pub score: f64,
}

/// One recognizer input batch built from some of the regions.
#[derive(Debug, Clone)]
pub struct RecBatch {
    pub input: InputTensor,
    /// Index into the regions slice for each batch row, in row order.
    pub regions: Vec<usize>,
    /// Per row: fraction of the tensor width holding the crop (the rest is
    /// padding), so glyph positions can be mapped back onto the region.
    pub extent: Vec<f64>,
}

/// Recognized text of one region.
#[derive(Debug, Clone, PartialEq)]
pub struct TextLine {
    pub text: String,
    /// Recognition confidence, 0..1.
    pub confidence: f64,
    /// Characters of `text` in order (whitespace included).
    pub glyphs: Vec<Glyph>,
}

/// One recognized character and where it sits along its region.
#[derive(Debug, Clone, PartialEq)]
pub struct Glyph {
    pub text: String,
    pub confidence: f64,
    /// `(start, end)` as fractions 0..1 along the region's reading direction.
    pub span: (f64, f64),
}

/// An `ocr` model backend (detector + recognizer). Implementations must be
/// cheap to share across threads; the `preprocess_*` methods run on a
/// blocking worker thread.
pub trait OcrBackend: Send + Sync {
    /// Family identifier (matches the `family` metadata field).
    fn family(&self) -> &'static str;
    /// The text detection model.
    fn detector(&self) -> &Stage;
    /// The text recognition model.
    fn recognizer(&self) -> &Stage;
    /// Decoded RGB image -> detector input tensor + mapping back to pixels.
    fn preprocess_detect(&self, image: &RgbImage) -> Prepared;
    /// Detector outputs -> text regions in original pixels, sorted by
    /// descending score and capped at `params.max_regions`.
    fn postprocess_detect(
        &self,
        outputs: &[OutputTensor],
        map: &BoxMap,
        params: &OcrParams,
    ) -> Result<Vec<TextRegion>>;
    /// Crop `regions` out of `image` into recognizer batches. Every region
    /// index appears in exactly one batch.
    fn preprocess_recognize(&self, image: &RgbImage, regions: &[TextRegion]) -> Vec<RecBatch>;
    /// Recognizer outputs for `batch` -> one line per `batch.regions` entry.
    fn postprocess_recognize(
        &self,
        outputs: &[OutputTensor],
        batch: &RecBatch,
    ) -> Result<Vec<TextLine>>;
}

/// The part of `q` between fractions `a..b` of its reading direction.
pub fn sub_quad(q: &Quad, a: f64, b: f64) -> Quad {
    let lerp =
        |p: (f64, f64), r: (f64, f64), t: f64| (p.0 + (r.0 - p.0) * t, p.1 + (r.1 - p.1) * t);
    [
        lerp(q[0], q[1], a),
        lerp(q[0], q[1], b),
        lerp(q[3], q[2], b),
        lerp(q[3], q[2], a),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_quad_interpolates_both_edges() {
        let q: Quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)];
        assert_eq!(
            sub_quad(&q, 0.25, 0.5),
            [(25.0, 0.0), (50.0, 0.0), (50.0, 10.0), (25.0, 10.0)]
        );
    }
}
