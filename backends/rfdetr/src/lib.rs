//! RF-DETR detection backend.
//!
//! Model contract (written by `backends/rfdetr/export.py`):
//! - input: `[1, 3, imgsz, imgsz]` FP32 RGB, stretch-resized (no aspect
//!   preservation), scaled to 0..1 and ImageNet-normalized;
//! - outputs: `dets` `[1, Q, 4]` normalized `cxcywh` and `labels`
//!   `[1, Q, slots]` raw logits; optional background slot (`bg_slot`).
//!
//! Decoding follows the rfdetr reference `PostProcess`: per-class sigmoid,
//! top-k over all (query, class) pairs, so one query may yield more than one
//! class.
//!
//! LICENSE NOTE: this crate is MIT. RF-DETR N/S/M/B/L weights are Apache-2.0;
//! XLarge/2XLarge weights are not — see `backends/rfdetr/README.md`.

use backend_api::image::{RgbImage, imageops};
use backend_api::util::{self, clamped, finish, sigmoid};
use backend_api::{
    Backend, BackendError, BoxMap, CommonMeta, DecodeParams, Detection, InputTensor, ModelInfo,
    OutputTensor, Prepared, Result,
};
use serde::Deserialize;

pub const FAMILY: &str = "rfdetr";

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Deserialize)]
struct RfdetrMeta {
    #[serde(default)]
    bg_slot: Option<i64>,
}

pub struct Rfdetr {
    names: Vec<String>,
    imgsz: u32,
    bg_slot: Option<i64>,
    input: String,
    outputs: Vec<String>,
}

pub fn create(info: &ModelInfo) -> Result<Box<dyn Backend>> {
    let common = CommonMeta::parse(info)?;
    let extra: RfdetrMeta = serde_json::from_value(info.metadata.clone())
        .map_err(|e| BackendError::InvalidModel(format!("model '{}': {e}", info.name)))?;
    let outputs = common
        .outputs
        .unwrap_or_else(|| vec!["dets".into(), "labels".into()]);
    if outputs.len() != 2 {
        return Err(BackendError::InvalidModel(format!(
            "model '{}': rfdetr needs exactly two outputs (dets, labels), got {outputs:?}",
            info.name
        )));
    }
    Ok(Box::new(Rfdetr {
        names: common.names,
        imgsz: common.imgsz,
        bg_slot: extra.bg_slot,
        input: common.input.unwrap_or_else(|| "input".into()),
        outputs,
    }))
}

impl Backend for Rfdetr {
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
        let scaled;
        let src = if (w, h) == (size, size) {
            image
        } else {
            scaled = imageops::resize(image, size, size, imageops::FilterType::Triangle);
            &scaled
        };
        let mut data = vec![0f32; 3 * (size as usize) * (size as usize)];
        util::write_chw(src, &mut data, size, size, 0, 0, |c, v| {
            (v as f32 / 255.0 - MEAN[c]) / STD[c]
        });
        Prepared {
            input: InputTensor {
                name: self.input.clone(),
                shape: vec![1, 3, size as u64, size as u64],
                data,
            },
            map: BoxMap {
                sx: size as f64 / w as f64,
                sy: size as f64 / h as f64,
                dx: 0.0,
                dy: 0.0,
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
        let dets = util::output(outputs, &self.outputs[0], 0)?;
        let labels = util::output(outputs, &self.outputs[1], 1)?;
        let ddims = util::per_image_dims(dets, 2)?;
        let ldims = util::per_image_dims(labels, 2)?;
        let (queries, slots) = (ddims[0] as usize, ldims[1] as usize);
        if ddims[1] != 4 || ldims[0] as usize != queries || slots == 0 {
            return Err(BackendError::BadOutput(format!(
                "rfdetr outputs mismatch: dets {:?}, labels {:?}",
                dets.shape, labels.shape
            )));
        }
        util::require_len(dets, queries * 4)?;
        util::require_len(labels, queries * slots)?;

        let n = slots as i64;
        let bg = self.bg_slot.map(|b| b.rem_euclid(n) as usize);
        let (ow, oh) = (map.orig_w as f64, map.orig_h as f64);

        let mut out = Vec::new();
        for q in 0..queries {
            let logits = &labels.data[q * slots..(q + 1) * slots];
            let b = &dets.data[q * 4..q * 4 + 4];
            let (cx, cy, bw, bh) = (b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64);
            let xyxy = (
                (cx - bw / 2.0) * ow,
                (cy - bh / 2.0) * oh,
                (cx + bw / 2.0) * ow,
                (cy + bh / 2.0) * oh,
            );
            for (s, &l) in logits.iter().enumerate() {
                if bg == Some(s) {
                    continue;
                }
                let score = sigmoid(l as f64);
                if score >= params.conf {
                    out.push(clamped(s as i64, score, xyxy, map.orig_w, map.orig_h));
                }
            }
        }
        Ok(finish(out, params.max_det))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensors(queries: usize, slots: usize) -> (Vec<f32>, Vec<f32>) {
        (vec![0f32; queries * 4], vec![-20f32; queries * slots])
    }

    fn run(m: &Rfdetr, dets: Vec<f32>, labels: Vec<f32>, q: u64, s: u64) -> Vec<Detection> {
        let outs = [
            OutputTensor {
                name: "dets".into(),
                shape: vec![1, q, 4],
                data: dets,
            },
            OutputTensor {
                name: "labels".into(),
                shape: vec![1, q, s],
                data: labels,
            },
        ];
        let map = BoxMap {
            sx: 1.0,
            sy: 1.0,
            dx: 0.0,
            dy: 0.0,
            orig_w: 1000,
            orig_h: 500,
        };
        let params = DecodeParams {
            conf: 0.25,
            iou: 0.7,
            max_det: 300,
        };
        m.postprocess(&outs, &map, &params).unwrap()
    }

    fn model(bg_slot: Option<i64>) -> Rfdetr {
        Rfdetr {
            names: vec!["a".into(), "b".into(), "c".into()],
            imgsz: 560,
            bg_slot,
            input: "input".into(),
            outputs: vec!["dets".into(), "labels".into()],
        }
    }

    #[test]
    fn sigmoid_bg_slot_and_denorm() {
        let (mut dets, mut labels) = tensors(2, 4); // 3 fg + 1 bg
        dets[0..4].copy_from_slice(&[0.5, 0.5, 0.5, 0.25]);
        labels[1] = 5.0;
        dets[4..8].copy_from_slice(&[0.1, 0.1, 0.1, 0.1]);
        labels[7] = 9.0; // bg slot (-1 == 3) only -> dropped
        let d = run(&model(Some(-1)), dets, labels, 2, 4);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].class, 1);
        assert_eq!(
            (d[0].x1, d[0].y1, d[0].x2, d[0].y2),
            (250.0, 187.5, 750.0, 312.5)
        );
    }

    #[test]
    fn topk_over_query_class_pairs() {
        // One query confidently matching two classes yields two detections,
        // like the rfdetr reference post-processor.
        let (mut dets, mut labels) = tensors(1, 3);
        dets.copy_from_slice(&[0.5, 0.5, 0.2, 0.2]);
        labels[0] = 3.0;
        labels[2] = 1.0;
        let d = run(&model(None), dets, labels, 1, 3);
        assert_eq!(d.iter().map(|d| d.class).collect::<Vec<_>>(), vec![0, 2]);
    }

    #[test]
    fn preprocess_normalizes() {
        let img = RgbImage::from_pixel(100, 50, backend_api::image::Rgb([255, 255, 255]));
        let p = model(None).preprocess(&img);
        assert_eq!(p.input.shape, vec![1, 3, 560, 560]);
        assert!((p.input.data[0] - (1.0 - MEAN[0]) / STD[0]).abs() < 1e-5);
        assert_eq!(p.map.orig_w, 100);
    }
}
