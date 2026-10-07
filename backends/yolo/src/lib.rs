//! YOLO end-to-end (NMS-free) detection backend.
//!
//! Model contract (written by `backends/yolo/export.py`):
//! - input: `[1, 3, imgsz, imgsz]` FP32 RGB in 0..1, letterboxed with gray
//!   (114) padding, Ultralytics `LetterBox(auto=False)` semantics;
//! - output: `[1, N, 6]` rows of `[x1, y1, x2, y2, confidence, class_id]` in
//!   letterboxed pixel space; padding rows have `class_id < 0` or zero conf.
//!
//! LICENSE NOTE: this crate only implements tensor pre/post-processing (MIT).
//! Ultralytics YOLO weights are AGPL-3.0 — see `backends/yolo/README.md`.

use backend_api::image::{RgbImage, imageops};
use backend_api::util::{self, clamped, finish};
use backend_api::{
    Backend, BackendError, BoxMap, CommonMeta, DecodeParams, Detection, InputTensor, ModelInfo,
    OutputTensor, Prepared, Result,
};
use serde::Deserialize;

pub const FAMILY: &str = "yolo";

const PAD: f32 = 114.0 / 255.0;

#[derive(Deserialize)]
struct YoloMeta {
    #[serde(default)]
    end2end: Option<bool>,
}

pub struct Yolo {
    names: Vec<String>,
    imgsz: u32,
    input: String,
    outputs: Vec<String>,
}

pub fn create(info: &ModelInfo) -> Result<Box<dyn Backend>> {
    let common = CommonMeta::parse(info)?;
    let extra: YoloMeta = serde_json::from_value(info.metadata.clone())
        .map_err(|e| BackendError::InvalidModel(format!("model '{}': {e}", info.name)))?;
    if extra.end2end != Some(true) {
        return Err(BackendError::InvalidModel(format!(
            "model '{}' is not an end-to-end (NMS-free) YOLO head; only end2end exports are served",
            info.name
        )));
    }
    Ok(Box::new(Yolo {
        names: common.names,
        imgsz: common.imgsz,
        input: common.input.unwrap_or_else(|| "images".into()),
        outputs: common.outputs.unwrap_or_else(|| vec!["output0".into()]),
    }))
}

/// Letterbox geometry: resized size and integer padding offsets.
fn letterbox_geometry(w: u32, h: u32, size: u32) -> (f64, u32, u32, u32, u32) {
    let r = (size as f64 / w as f64).min(size as f64 / h as f64);
    let new_w = ((w as f64 * r).round() as u32).clamp(1, size);
    let new_h = ((h as f64 * r).round() as u32).clamp(1, size);
    // Ultralytics: top/left = round(pad/2 - 0.1)
    let left = (((size - new_w) as f64 / 2.0) - 0.1).round().max(0.0) as u32;
    let top = (((size - new_h) as f64 / 2.0) - 0.1).round().max(0.0) as u32;
    (r, new_w, new_h, left, top)
}

impl Backend for Yolo {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn class_names(&self) -> &[String] {
        &self.names
    }

    fn output_names(&self) -> &[String] {
        &self.outputs
    }

    fn input_size(&self) -> (u32, u32) {
        (self.imgsz, self.imgsz)
    }

    fn preprocess(&self, image: &RgbImage) -> Prepared {
        let size = self.imgsz;
        let (w, h) = image.dimensions();
        let (r, new_w, new_h, left, top) = letterbox_geometry(w, h, size);

        let mut data = vec![PAD; 3 * (size as usize) * (size as usize)];
        let scaled;
        let src = if (new_w, new_h) == (w, h) {
            image
        } else {
            scaled = imageops::resize(image, new_w, new_h, imageops::FilterType::Triangle);
            &scaled
        };
        util::write_chw(src, &mut data, size, size, left, top, |_, v| {
            v as f32 / 255.0
        });

        Prepared {
            input: InputTensor {
                name: self.input.clone(),
                shape: vec![1, 3, size as u64, size as u64],
                data,
            },
            map: BoxMap {
                sx: r,
                sy: r,
                dx: left as f64,
                dy: top as f64,
                orig_w: w,
                orig_h: h,
            },
        }
    }

    fn postprocess(
        &self,
        outputs: &[OutputTensor],
        map: &BoxMap,
        params: &DecodeParams,
    ) -> Result<Vec<Detection>> {
        let out = util::output(outputs, &self.outputs[0], 0)?;
        let dims = util::per_image_dims(out, 2)?;
        if dims[1] != 6 {
            return Err(BackendError::BadOutput(format!(
                "yolo output has {} columns, expected 6",
                dims[1]
            )));
        }
        let rows = dims[0] as usize;
        util::require_len(out, rows * 6)?;

        let mut dets = Vec::new();
        for &[x1, y1, x2, y2, conf, class] in out.data[..rows * 6].as_chunks::<6>().0 {
            let (conf, class) = (conf as f64, class as i64);
            if class < 0 || conf <= 0.0 || conf < params.conf {
                continue;
            }
            let b = map.to_original(x1 as f64, y1 as f64, x2 as f64, y2 as f64);
            dets.push(clamped(class, conf, b, map.orig_w, map.orig_h));
        }
        Ok(finish(dets, params.max_det))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yolo() -> Yolo {
        Yolo {
            names: vec!["person".into(), "car".into()],
            imgsz: 640,
            input: "images".into(),
            outputs: vec!["output0".into()],
        }
    }

    #[test]
    fn letterbox_never_exceeds_canvas() {
        // Regression: stride-rounded upscaling used to overflow the canvas.
        for (w, h) in [
            (300, 300),
            (1, 1),
            (17, 911),
            (1000, 750),
            (641, 639),
            (5000, 3),
        ] {
            let (_, nw, nh, left, top) = letterbox_geometry(w, h, 640);
            assert!(left + nw <= 640 && top + nh <= 640, "{w}x{h}");
            let img = RgbImage::new(w, h);
            let p = yolo().preprocess(&img);
            assert_eq!(p.input.data.len(), 3 * 640 * 640);
        }
    }

    #[test]
    fn letterbox_matches_ultralytics_geometry() {
        // 1000x750 -> r=0.64, 640x480, top pad 80 (no stride rounding).
        assert_eq!(letterbox_geometry(1000, 750, 640), (0.64, 640, 480, 0, 80));
        // 1280x720 -> r=0.5, 640x360, top pad 140.
        assert_eq!(letterbox_geometry(1280, 720, 640), (0.5, 640, 360, 0, 140));
    }

    #[test]
    fn padding_is_gray_and_image_is_scaled() {
        let img = RgbImage::from_pixel(640, 480, backend_api::image::Rgb([255, 0, 0]));
        let p = yolo().preprocess(&img);
        let n = 640 * 640;
        assert_eq!(p.input.data[0], PAD); // top padding row
        assert_eq!(p.input.data[80 * 640], 1.0); // first image row, R
        assert_eq!(p.input.data[n + 80 * 640], 0.0); // G
        assert_eq!(p.map.dy, 80.0);
    }

    #[test]
    fn decode_filters_and_maps_back() {
        // 1280x720 into 640: r=0.5, dx=0, dy=140
        let map = BoxMap {
            sx: 0.5,
            sy: 0.5,
            dx: 0.0,
            dy: 140.0,
            orig_w: 1280,
            orig_h: 720,
        };
        let mut data = vec![0f32; 300 * 6];
        data[0..6].copy_from_slice(&[0.0, 140.0, 320.0, 500.0, 0.9, 1.0]);
        data[6..12].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.1, 0.0]); // below conf
        data[12..18].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]); // padding
        data[18..24].copy_from_slice(&[100.0, 200.0, 300.0, 400.0, 0.95, 0.0]);
        let out = OutputTensor {
            name: "output0".into(),
            shape: vec![1, 300, 6],
            data,
        };
        let params = DecodeParams {
            conf: 0.25,
            iou: 0.7,
            max_det: 300,
        };
        let d = yolo().postprocess(&[out], &map, &params).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].class, 0);
        assert_eq!(
            (d[0].x1, d[0].y1, d[0].x2, d[0].y2),
            (200.0, 120.0, 600.0, 520.0)
        );
        assert_eq!(d[1].class, 1);
    }
}
