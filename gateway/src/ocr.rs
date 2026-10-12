//! The OCR task pipeline, independent of any wire API: decode → detect text
//! regions → (optionally) recognize each region. Endpoints format the result.

use std::sync::Arc;

use backend_api::ocr::{OcrBackend, OcrParams, Quad, TextLine, TextRegion};

use crate::error::{ApiError, Result};
use crate::input;
use crate::state::AppState;

/// One image's OCR result. Regions are in reading order.
#[derive(Debug, Clone)]
pub struct OcrOutput {
    pub width: u32,
    pub height: u32,
    pub regions: Vec<TextRegion>,
    /// Recognized lines (reading order), when recognition was requested.
    /// Lines below the confidence floor or without text are dropped.
    pub lines: Option<Vec<Line>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub quad: Quad,
    pub text: TextLine,
}

/// Run OCR on encoded image `bytes`. With `recognize == false` only the
/// detector runs (text regions, no text).
pub async fn run(
    state: &AppState,
    backend: Arc<dyn OcrBackend>,
    bytes: Vec<u8>,
    params: OcrParams,
    min_text_confidence: f64,
    recognize: bool,
) -> Result<OcrOutput> {
    let b = backend.clone();
    let ((image, prepared), _) =
        input::decode_then(bytes, state.cfg.limits.max_image_pixels, move |image| {
            let prepared = b.preprocess_detect(&image);
            (Arc::new(image), prepared)
        })
        .await?;
    let (width, height) = image.dimensions();

    let det = backend.detector();
    let outputs = state
        .triton
        .infer(&det.model, &prepared.input, &det.outputs)
        .await?;

    // Region decoding and cropping are CPU-bound: keep them off the workers.
    let b = backend.clone();
    let img = image.clone();
    let (regions, batches) = tokio::task::spawn_blocking(move || {
        let mut regions = b
            .postprocess_detect(&outputs, &prepared.map, &params)
            .map_err(|e| ApiError::Upstream(e.to_string()))?;
        sort_reading_order(&mut regions);
        let batches = if recognize && !regions.is_empty() {
            b.preprocess_recognize(&img, &regions)
        } else {
            Vec::new()
        };
        Ok::<_, ApiError>((regions, batches))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("ocr task failed: {e}")))??;

    if !recognize {
        return Ok(OcrOutput {
            width,
            height,
            regions,
            lines: None,
        });
    }

    let rec = backend.recognizer();
    let results = futures_util::future::try_join_all(batches.iter().map(|batch| async {
        let outputs = state
            .triton
            .infer(&rec.model, &batch.input, &rec.outputs)
            .await?;
        backend
            .postprocess_recognize(&outputs, batch)
            .map_err(|e| ApiError::Upstream(e.to_string()))
    }))
    .await?;

    let mut texts: Vec<Option<TextLine>> = vec![None; regions.len()];
    for (batch, lines) in batches.iter().zip(results) {
        if lines.len() != batch.regions.len() {
            return Err(ApiError::Upstream(format!(
                "recognizer returned {} lines for {} regions",
                lines.len(),
                batch.regions.len()
            )));
        }
        for (&i, line) in batch.regions.iter().zip(lines) {
            if let Some(slot) = texts.get_mut(i) {
                *slot = Some(line);
            }
        }
    }
    let lines = regions
        .iter()
        .zip(texts)
        .filter_map(|(r, t)| {
            let t = t?;
            (!t.text.trim().is_empty() && t.confidence >= min_text_confidence).then_some(Line {
                quad: r.quad,
                text: t,
            })
        })
        .collect();
    Ok(OcrOutput {
        width,
        height,
        regions,
        lines: Some(lines),
    })
}

/// Top-to-bottom, then left-to-right; regions whose tops are within
/// `SAME_LINE_PX` are treated as one row (PaddleOCR `sorted_boxes`).
fn sort_reading_order(regions: &mut [TextRegion]) {
    const SAME_LINE_PX: f64 = 10.0;
    regions.sort_by(|a, b| {
        (a.quad[0].1, a.quad[0].0)
            .partial_cmp(&(b.quad[0].1, b.quad[0].0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in 0..regions.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            let (a, b) = (regions[j].quad[0], regions[j + 1].quad[0]);
            if (b.1 - a.1).abs() < SAME_LINE_PX && b.0 < a.0 {
                regions.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(x: f64, y: f64) -> TextRegion {
        TextRegion {
            quad: [(x, y), (x + 10.0, y), (x + 10.0, y + 5.0), (x, y + 5.0)],
            score: 1.0,
        }
    }

    #[test]
    fn reading_order_groups_rows_with_small_vertical_jitter() {
        let mut r = vec![region(50.0, 3.0), region(0.0, 40.0), region(0.0, 5.0)];
        sort_reading_order(&mut r);
        let xy: Vec<_> = r.iter().map(|r| r.quad[0]).collect();
        assert_eq!(xy, vec![(0.0, 5.0), (50.0, 3.0), (0.0, 40.0)]);
    }
}
