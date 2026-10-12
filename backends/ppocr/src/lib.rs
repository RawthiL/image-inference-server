//! PaddleOCR (PP-OCR) OCR backend: DB text detector + CTC recognizer.
//!
//! Model contract (written by `backends/ppocr/export.py`), two Triton models:
//! - detector (`"stage": "det"`, the configured model): input
//!   `[1, 3, H, W]` FP32, H/W multiples of 32 (resized per `limit_side_len` /
//!   `limit_type`), mean/std normalized in `channel_order`; output
//!   `[1, 1, H, W]` text probability map. Lists the recognizer in
//!   `companions`.
//! - recognizer (`"stage": "rec"`): input `[N, 3, imgh, W]` FP32, crops
//!   height-resized and right-padded with zeros, `(x/255 - 0.5) / 0.5`;
//!   output `[N, T, len(charset) + 1]` per-step probabilities, class 0 =
//!   CTC blank, class `k` = `charset[k - 1]`.
//!
//! Post-processing follows the PaddleOCR reference (`DBPostProcess`,
//! `get_rotate_crop_image`, `CTCLabelDecode`) without OpenCV; see
//! [`geometry`].
//!
//! LICENSE NOTE: this crate is MIT. PP-OCR weights are Apache-2.0 — see
//! `backends/ppocr/README.md`.

use backend_api::image::{RgbImage, imageops};
use backend_api::ocr::{Glyph, OcrBackend, OcrParams, Quad, RecBatch, Stage, TextLine, TextRegion};
use backend_api::util;
use backend_api::{BackendError, BoxMap, InputTensor, ModelInfo, OutputTensor, Prepared, Result};
use serde::Deserialize;

pub mod geometry;

use geometry::{Point, dist};

pub const FAMILY: &str = "ppocr";

/// PaddleOCR drops boxes whose short side is below this (in map pixels).
const MIN_SIZE: f64 = 3.0;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
enum LimitType {
    /// Downscale so the longest side is at most `limit_side_len`.
    #[default]
    Max,
    /// Upscale so the shortest side is at least `limit_side_len`.
    Min,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
enum ChannelOrder {
    /// PaddleOCR reads images with OpenCV: BGR.
    #[default]
    Bgr,
    Rgb,
}

fn d_input() -> String {
    "x".into()
}
fn d_limit_side_len() -> u32 {
    960
}
fn d_max_side_limit() -> u32 {
    4000
}
fn d_mean() -> [f32; 3] {
    [0.485, 0.456, 0.406]
}
fn d_std() -> [f32; 3] {
    [0.229, 0.224, 0.225]
}
fn d_thresh() -> f64 {
    0.3
}
fn d_box_thresh() -> f64 {
    0.6
}
fn d_unclip_ratio() -> f64 {
    1.5
}
fn d_max_candidates() -> usize {
    1000
}
fn d_imgh() -> u32 {
    48
}
fn d_imgw() -> u32 {
    320
}
fn d_max_width() -> u32 {
    3200
}
fn d_batch() -> usize {
    6
}

#[derive(Debug, Clone, Deserialize)]
struct DetMeta {
    stage: String,
    #[serde(default = "d_input")]
    input: String,
    #[serde(default)]
    outputs: Option<Vec<String>>,
    #[serde(default = "d_limit_side_len")]
    limit_side_len: u32,
    #[serde(default)]
    limit_type: LimitType,
    #[serde(default = "d_max_side_limit")]
    max_side_limit: u32,
    #[serde(default = "d_mean")]
    mean: [f32; 3],
    #[serde(default = "d_std")]
    std: [f32; 3],
    #[serde(default)]
    channel_order: ChannelOrder,
    /// Probability-map binarization threshold.
    #[serde(default = "d_thresh")]
    thresh: f64,
    /// Default minimum region score.
    #[serde(default = "d_box_thresh")]
    box_thresh: f64,
    #[serde(default = "d_unclip_ratio")]
    unclip_ratio: f64,
    #[serde(default = "d_max_candidates")]
    max_candidates: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RecMeta {
    #[serde(default = "d_input")]
    input: String,
    #[serde(default)]
    outputs: Option<Vec<String>>,
    #[serde(default = "d_imgh")]
    imgh: u32,
    #[serde(default = "d_imgw")]
    imgw: u32,
    #[serde(default = "d_max_width")]
    max_width: u32,
    #[serde(default)]
    channel_order: ChannelOrder,
    /// Rows per recognizer request (capped by the model's max_batch_size).
    #[serde(default = "d_batch")]
    batch: usize,
    charset: Vec<String>,
}

pub struct PpOcr {
    det: DetMeta,
    rec: RecMeta,
    det_stage: Stage,
    rec_stage: Stage,
    rec_batch: usize,
}

fn invalid(model: &str, msg: impl std::fmt::Display) -> BackendError {
    BackendError::InvalidModel(format!("model '{model}': {msg}"))
}

pub fn create(info: &ModelInfo) -> Result<Box<dyn OcrBackend>> {
    let det: DetMeta =
        serde_json::from_value(info.metadata.clone()).map_err(|e| invalid(info.name, e))?;
    if det.stage != "det" {
        return Err(invalid(
            info.name,
            format!(
                "stage is '{}'; configure the ppocr *detector* model (stage 'det')",
                det.stage
            ),
        ));
    }
    if det.limit_side_len == 0 || det.std.contains(&0.0) {
        return Err(invalid(
            info.name,
            "limit_side_len and std must be non-zero",
        ));
    }
    let rec_info = info
        .companions
        .iter()
        .find(|c| c.metadata.get("stage").and_then(|s| s.as_str()) == Some("rec"))
        .ok_or_else(|| {
            invalid(
                info.name,
                "metadata 'companions' must name the ppocr recognizer (stage 'rec')",
            )
        })?;
    let rec: RecMeta =
        serde_json::from_value(rec_info.metadata.clone()).map_err(|e| invalid(rec_info.name, e))?;
    if rec.charset.is_empty() || rec.imgh == 0 || rec.imgw == 0 || rec.batch == 0 {
        return Err(invalid(
            rec_info.name,
            "charset, imgh, imgw and batch must be non-empty / non-zero",
        ));
    }
    let rec_batch = match rec_info.max_batch_size {
        0 => rec.batch,
        n => rec.batch.min(n as usize),
    };
    let det_stage = Stage {
        model: info.name.to_string(),
        outputs: det
            .outputs
            .clone()
            .unwrap_or_else(|| vec!["sigmoid_0.tmp_0".into()]),
    };
    let rec_stage = Stage {
        model: rec_info.name.to_string(),
        outputs: rec
            .outputs
            .clone()
            .unwrap_or_else(|| vec!["softmax_0.tmp_0".into()]),
    };
    Ok(Box::new(PpOcr {
        det,
        rec,
        det_stage,
        rec_stage,
        rec_batch,
    }))
}

/// Detector input size for a `w x h` image (PaddleOCR `DetResizeForTest`
/// type 0, plus the PaddleOCR 3.x `max_side_limit`).
fn det_size(w: u32, h: u32, limit: u32, kind: LimitType, max_side: u32) -> (u32, u32) {
    let (wf, hf, lf) = (w as f64, h as f64, limit as f64);
    let ratio = match kind {
        LimitType::Max if wf.max(hf) > lf => lf / wf.max(hf),
        LimitType::Min if wf.min(hf) < lf => lf / wf.min(hf),
        _ => 1.0,
    };
    let (mut rw, mut rh) = ((wf * ratio).trunc(), (hf * ratio).trunc());
    if rw.max(rh) > max_side as f64 {
        let r = max_side as f64 / rw.max(rh);
        (rw, rh) = ((rw * r).trunc(), (rh * r).trunc());
    }
    // Python `round` (ties to even), as in the reference.
    let snap = |v: f64| (((v / 32.0).round_ties_even() * 32.0) as u32).max(32);
    (snap(rw), snap(rh))
}

/// Write `img` (RGB8) into a planar CHW buffer of row stride `stride`,
/// reordering channels and applying `f(model_channel, value)`.
fn write_planes(
    img: &RgbImage,
    out: &mut [f32],
    stride: usize,
    plane: usize,
    order: ChannelOrder,
    f: impl Fn(usize, u8) -> f32,
) {
    let w = img.width() as usize;
    for (i, px) in img.as_raw().as_chunks::<3>().0.iter().enumerate() {
        let (x, y) = (i % w, i / w);
        for c in 0..3 {
            let src = match order {
                ChannelOrder::Rgb => c,
                ChannelOrder::Bgr => 2 - c,
            };
            out[c * plane + y * stride + x] = f(c, px[src]);
        }
    }
}

/// Mean of `pred` inside `quad` (PaddleOCR `box_score_fast`).
fn box_score(pred: &[f32], w: usize, h: usize, quad: &[Point; 4]) -> f64 {
    let clampx = |v: f64| v.clamp(0.0, (w - 1) as f64);
    let clampy = |v: f64| v.clamp(0.0, (h - 1) as f64);
    let xs = quad.iter().map(|p| p.0);
    let ys = quad.iter().map(|p| p.1);
    let x0 = clampx(xs.clone().fold(f64::MAX, f64::min).floor()) as usize;
    let x1 = clampx(xs.fold(f64::MIN, f64::max).ceil()) as usize;
    let y0 = clampy(ys.clone().fold(f64::MAX, f64::min).floor()) as usize;
    let y1 = clampy(ys.fold(f64::MIN, f64::max).ceil()) as usize;
    let (mut sum, mut n) = (0.0, 0usize);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if geometry::inside(quad, (x as f64, y as f64)) {
                sum += pred[y * w + x] as f64;
                n += 1;
            }
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Turn a clockwise top-left-first box into a reading-order quad: text that
/// is at least 1.5x taller than wide is read top to bottom (PaddleOCR rotates
/// such crops by 90°).
fn reading_quad(b: [Point; 4]) -> Quad {
    let (w, h) = (
        dist(b[0], b[1]).max(dist(b[3], b[2])),
        dist(b[0], b[3]).max(dist(b[1], b[2])),
    );
    if w > 0.0 && h / w >= 1.5 {
        [b[1], b[2], b[3], b[0]]
    } else {
        b
    }
}

/// Crop size of a reading quad (`get_rotate_crop_image`).
fn crop_size(q: &Quad) -> (u32, u32) {
    let w = dist(q[0], q[1]).max(dist(q[3], q[2]));
    let h = dist(q[0], q[3]).max(dist(q[1], q[2]));
    ((w as u32).max(1), (h as u32).max(1))
}

/// Greedy CTC decode of one `[T, C]` row: `(class, prob, step)` per kept step.
fn ctc_greedy(probs: &[f32], t: usize, c: usize) -> Vec<(usize, f64, usize)> {
    let mut out = Vec::new();
    let mut prev = usize::MAX;
    for step in 0..t {
        let row = &probs[step * c..(step + 1) * c];
        let (k, p) = row
            .iter()
            .enumerate()
            .fold((0, f32::MIN), |b, (i, &p)| if p > b.1 { (i, p) } else { b });
        if k != 0 && k != prev {
            out.push((k, p as f64, step));
        }
        prev = k;
    }
    out
}

/// Glyph spans (fractions of the crop) from CTC step centers.
fn glyph_spans(centers: &[f64], step: f64) -> Vec<(f64, f64)> {
    let n = centers.len();
    (0..n)
        .map(|i| {
            let start = if i > 0 {
                (centers[i - 1] + centers[i]) / 2.0
            } else if n > 1 {
                centers[0] - (centers[1] - centers[0]) / 2.0
            } else {
                centers[0] - step
            };
            let end = if i + 1 < n {
                (centers[i] + centers[i + 1]) / 2.0
            } else if n > 1 {
                centers[i] + (centers[i] - centers[i - 1]) / 2.0
            } else {
                centers[0] + step
            };
            (start.clamp(0.0, 1.0), end.clamp(0.0, 1.0))
        })
        .collect()
}

impl OcrBackend for PpOcr {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn detector(&self) -> &Stage {
        &self.det_stage
    }

    fn recognizer(&self) -> &Stage {
        &self.rec_stage
    }

    fn preprocess_detect(&self, image: &RgbImage) -> Prepared {
        let (w, h) = image.dimensions();
        let d = &self.det;
        let (rw, rh) = det_size(w, h, d.limit_side_len, d.limit_type, d.max_side_limit);
        let resized = imageops::resize(image, rw, rh, imageops::FilterType::Triangle);
        let plane = (rw * rh) as usize;
        let mut data = vec![0f32; 3 * plane];
        write_planes(
            &resized,
            &mut data,
            rw as usize,
            plane,
            d.channel_order,
            |c, v| (v as f32 / 255.0 - d.mean[c]) / d.std[c],
        );
        Prepared {
            input: InputTensor {
                name: d.input.clone(),
                shape: vec![1, 3, rh as u64, rw as u64],
                data,
            },
            map: BoxMap {
                sx: rw as f64 / w as f64,
                sy: rh as f64 / h as f64,
                dx: 0.0,
                dy: 0.0,
                orig_w: w,
                orig_h: h,
            },
        }
    }

    fn postprocess_detect(
        &self,
        outputs: &[OutputTensor],
        map: &BoxMap,
        params: &OcrParams,
    ) -> Result<Vec<TextRegion>> {
        let out = util::output(outputs, &self.det_stage.outputs[0], 0)?;
        let (h, w) = match out.shape.as_slice() {
            [.., h, w] if out.shape.len() >= 2 => (*h as usize, *w as usize),
            _ => {
                return Err(BackendError::BadOutput(format!(
                    "ppocr detector output shape {:?} is not [.., H, W]",
                    out.shape
                )));
            }
        };
        util::require_len(out, w * h)?;
        let pred = &out.data[..w * h];
        let d = &self.det;
        let mask: Vec<bool> = pred.iter().map(|&p| p as f64 > d.thresh).collect();
        let min_score = params.min_region_score.unwrap_or(d.box_thresh);
        let (ow, oh) = (map.orig_w as f64, map.orig_h as f64);

        let mut regions = Vec::new();
        for points in geometry::components(&mask, w, h, d.max_candidates) {
            let Some(rect) = geometry::min_area_rect(&points) else {
                continue;
            };
            if rect.short_side < MIN_SIZE {
                continue;
            }
            let score = box_score(pred, w, h, &rect.corners);
            if score < min_score {
                continue;
            }
            let Some(grown) = geometry::unclip(&rect, d.unclip_ratio) else {
                continue;
            };
            if grown.short_side < MIN_SIZE + 2.0 {
                continue;
            }
            let b = grown.corners.map(|(x, y)| {
                (
                    ((x - map.dx) / map.sx).round().clamp(0.0, ow),
                    ((y - map.dy) / map.sy).round().clamp(0.0, oh),
                )
            });
            // `filter_tag_det_res`: drop slivers after mapping back.
            if dist(b[0], b[1]) <= MIN_SIZE || dist(b[0], b[3]) <= MIN_SIZE {
                continue;
            }
            regions.push(TextRegion {
                quad: reading_quad(b),
                score,
            });
        }
        regions.sort_by(|a, b| b.score.total_cmp(&a.score));
        regions.truncate(params.max_regions);
        Ok(regions)
    }

    fn preprocess_recognize(&self, image: &RgbImage, regions: &[TextRegion]) -> Vec<RecBatch> {
        let r = &self.rec;
        let crops: Vec<RgbImage> = regions
            .iter()
            .map(|reg| {
                let (w, h) = crop_size(&reg.quad);
                geometry::warp_quad(image, &reg.quad, w, h)
            })
            .collect();
        let ratio = |i: usize| crops[i].width() as f64 / crops[i].height() as f64;
        // Similar aspect ratios together keep padding small.
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by(|&a, &b| ratio(a).total_cmp(&ratio(b)));

        let imgh = r.imgh;
        order
            .chunks(self.rec_batch)
            .map(|idx| {
                let max_ratio = idx
                    .iter()
                    .map(|&i| ratio(i))
                    .fold(r.imgw as f64 / imgh as f64, f64::max);
                let bw = ((imgh as f64 * max_ratio) as u32).clamp(1, r.max_width.max(1));
                let plane = (bw * imgh) as usize;
                let mut data = vec![0f32; idx.len() * 3 * plane];
                let mut extent = Vec::with_capacity(idx.len());
                for (row, &i) in idx.iter().enumerate() {
                    let rw = ((imgh as f64 * ratio(i)).ceil() as u32).clamp(1, bw);
                    let resized =
                        imageops::resize(&crops[i], rw, imgh, imageops::FilterType::Triangle);
                    let out = &mut data[row * 3 * plane..(row + 1) * 3 * plane];
                    write_planes(
                        &resized,
                        out,
                        bw as usize,
                        plane,
                        r.channel_order,
                        |_, v| (v as f32 / 255.0 - 0.5) / 0.5,
                    );
                    extent.push(rw as f64 / bw as f64);
                }
                RecBatch {
                    input: InputTensor {
                        name: r.input.clone(),
                        shape: vec![idx.len() as u64, 3, imgh as u64, bw as u64],
                        data,
                    },
                    regions: idx.to_vec(),
                    extent,
                }
            })
            .collect()
    }

    fn postprocess_recognize(
        &self,
        outputs: &[OutputTensor],
        batch: &RecBatch,
    ) -> Result<Vec<TextLine>> {
        let out = util::output(outputs, &self.rec_stage.outputs[0], 0)?;
        let n = batch.regions.len();
        let (t, c) = match out.shape.as_slice() {
            [b, t, c] if *b as usize == n => (*t as usize, *c as usize),
            _ => {
                return Err(BackendError::BadOutput(format!(
                    "ppocr recognizer output shape {:?}, expected [{n}, T, C]",
                    out.shape
                )));
            }
        };
        let charset = &self.rec.charset;
        if c != charset.len() + 1 {
            return Err(BackendError::BadOutput(format!(
                "ppocr recognizer has {c} classes, but the charset has {} (+1 blank)",
                charset.len()
            )));
        }
        util::require_len(out, n * t * c)?;
        Ok((0..n)
            .map(|row| {
                let kept = ctc_greedy(&out.data[row * t * c..(row + 1) * t * c], t, c);
                let extent = batch.extent[row].max(f64::EPSILON);
                let centers: Vec<f64> = kept
                    .iter()
                    .map(|&(_, _, s)| (s as f64 + 0.5) / t as f64 / extent)
                    .collect();
                let spans = glyph_spans(&centers, 1.0 / t as f64 / extent);
                let glyphs: Vec<Glyph> = kept
                    .iter()
                    .zip(spans)
                    .map(|(&(k, p, _), span)| Glyph {
                        text: charset[k - 1].clone(),
                        confidence: p,
                        span,
                    })
                    .collect();
                let confidence = if glyphs.is_empty() {
                    0.0
                } else {
                    glyphs.iter().map(|g| g.confidence).sum::<f64>() / glyphs.len() as f64
                };
                TextLine {
                    text: glyphs.iter().map(|g| g.text.as_str()).collect(),
                    confidence,
                    glyphs,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_api::image::Rgb;
    use serde_json::json;

    fn backend() -> Box<dyn OcrBackend> {
        let det = json!({"family": "ppocr", "task": "ocr", "stage": "det",
                         "companions": ["rec"], "outputs": ["map"]});
        let rec = json!({"family": "ppocr", "task": "ocr", "stage": "rec",
                         "outputs": ["probs"], "charset": ["A", "B", " "], "batch": 4});
        let companions = [ModelInfo {
            name: "rec",
            metadata: &rec,
            max_batch_size: 2,
            companions: &[],
        }];
        create(&ModelInfo {
            name: "det",
            metadata: &det,
            max_batch_size: 1,
            companions: &companions,
        })
        .unwrap()
    }

    #[test]
    fn create_requires_a_recognizer_companion() {
        let det = json!({"stage": "det"});
        let err = create(&ModelInfo {
            name: "det",
            metadata: &det,
            max_batch_size: 1,
            companions: &[],
        })
        .err()
        .unwrap();
        assert!(err.to_string().contains("recognizer"), "{err}");
    }

    #[test]
    fn det_size_matches_paddleocr() {
        // 1280x720, max 960 -> ratio .75 -> 960x540 -> snapped 960x544.
        assert_eq!(det_size(1280, 720, 960, LimitType::Max, 4000), (960, 544));
        // Small images are not upscaled with 'max', only snapped.
        assert_eq!(det_size(100, 20, 960, LimitType::Max, 4000), (96, 32));
        // 'min' upscales the short side.
        assert_eq!(det_size(200, 100, 736, LimitType::Min, 4000), (1472, 736));
        // Capped at max_side_limit: 8000x800 -> 4000x400 -> 400/32 = 12.5 -> 12.
        assert_eq!(det_size(1000, 100, 800, LimitType::Min, 4000), (4000, 384));
    }

    #[test]
    fn detects_a_bright_rectangle_and_maps_it_back() {
        let b = backend();
        // 64x32 prob map; text block at x 8..=39, y 8..=15 in map pixels.
        let (w, h) = (64usize, 32usize);
        let mut data = vec![0f32; w * h];
        for y in 8..=15 {
            for x in 8..=39 {
                data[y * w + x] = 0.9;
            }
        }
        let out = OutputTensor {
            name: "map".into(),
            shape: vec![1, 1, h as u64, w as u64],
            data,
        };
        // Original image 2x the map size.
        let map = BoxMap {
            sx: 0.5,
            sy: 0.5,
            dx: 0.0,
            dy: 0.0,
            orig_w: 128,
            orig_h: 64,
        };
        let params = OcrParams {
            min_region_score: None,
            max_regions: 10,
        };
        let r = b.postprocess_detect(&[out], &map, &params).unwrap();
        assert_eq!(r.len(), 1);
        assert!((r[0].score - 0.9).abs() < 1e-6);
        // rect 31x7, d = 31*7*1.5/76 = 4.28 -> (3.72, 3.72)..(43.28, 19.28) in
        // map pixels, x2 and rounded.
        assert_eq!(
            r[0].quad,
            [(7.0, 7.0), (87.0, 7.0), (87.0, 39.0), (7.0, 39.0)]
        );
        let strict = OcrParams {
            min_region_score: Some(0.95),
            max_regions: 10,
        };
        let out = OutputTensor {
            name: "map".into(),
            shape: vec![1, 1, h as u64, w as u64],
            data: vec![0.9; w * h],
        };
        assert!(
            b.postprocess_detect(&[out], &map, &strict)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn tall_regions_are_read_top_to_bottom() {
        let b = [(0.0, 0.0), (10.0, 0.0), (10.0, 40.0), (0.0, 40.0)];
        assert_eq!(
            reading_quad(b),
            [(10.0, 0.0), (10.0, 40.0), (0.0, 40.0), (0.0, 0.0)]
        );
        let wide = [(0.0, 0.0), (40.0, 0.0), (40.0, 10.0), (0.0, 10.0)];
        assert_eq!(reading_quad(wide), wide);
    }

    #[test]
    fn recognition_batches_are_capped_and_padded() {
        let b = backend();
        let img = RgbImage::from_pixel(200, 50, Rgb([255, 255, 255]));
        let region = |w: f64| TextRegion {
            quad: [(0.0, 0.0), (w, 0.0), (w, 24.0), (0.0, 24.0)],
            score: 1.0,
        };
        let regions = [region(48.0), region(192.0), region(24.0)];
        let batches = b.preprocess_recognize(&img, &regions);
        // rec max_batch_size 2 caps meta batch 4.
        assert_eq!(batches.len(), 2);
        // Sorted by aspect ratio: 24 (1:1), 48 (2:1) | 192 (8:1).
        assert_eq!(batches[0].regions, vec![2, 0]);
        // Width: imgh 48 * max(320/48, 2) = 320.
        assert_eq!(batches[0].input.shape, vec![2, 3, 48, 320]);
        assert_eq!(batches[0].extent, vec![48.0 / 320.0, 96.0 / 320.0]);
        assert_eq!(batches[1].input.shape, vec![1, 3, 48, 384]);
        // White -> 1.0, padding -> 0.
        assert_eq!(batches[0].input.data[0], 1.0);
        assert_eq!(batches[0].input.data[100], 0.0);
    }

    #[test]
    fn ctc_decode_collapses_repeats_and_blanks() {
        let b = backend();
        // T=6, C=4 (blank, A, B, space): A A _ B _ A
        let steps = [1usize, 1, 0, 2, 3, 1];
        let mut data = vec![0f32; 6 * 4];
        for (t, &k) in steps.iter().enumerate() {
            data[t * 4 + k] = 0.8;
        }
        let out = OutputTensor {
            name: "probs".into(),
            shape: vec![1, 6, 4],
            data,
        };
        let batch = RecBatch {
            input: InputTensor {
                name: "x".into(),
                shape: vec![],
                data: vec![],
            },
            regions: vec![0],
            extent: vec![0.5],
        };
        let lines = b.postprocess_recognize(&[out], &batch).unwrap();
        assert_eq!(lines[0].text, "AB A");
        assert!((lines[0].confidence - 0.8).abs() < 1e-6);
        // Step 0 center: 0.5/6 / 0.5 = 1/6 of the crop.
        let g = &lines[0].glyphs;
        assert!((g[0].span.0 - 0.0).abs() < 1e-9 && g[0].span.1 > 1.0 / 6.0);
        assert_eq!(g.last().unwrap().span.1, 1.0);
    }
}
