use serde::{Deserialize, Serialize};

use backend_api::{Backend, Detection};

/// A detection formatted exactly as the Ultralytics Platform
/// `PredictResponse.images[].results[]` element. `box` always carries
/// x1,y1,x2,y2 for detect tasks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResultItem {
    pub name: String,
    #[serde(rename = "class")]
    pub class: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#box: Option<BoxJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BoxJson {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Speed {
    pub preprocess: f64,
    pub inference: f64,
    pub postprocess: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageResult {
    pub shape: [u32; 2],
    pub speed: Speed,
    pub results: Vec<ResultItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Metadata {
    #[serde(rename = "imageCount")]
    pub image_count: u32,
    #[serde(rename = "classNames", skip_serializing_if = "Option::is_none")]
    pub class_names: Option<Vec<String>>,
    #[serde(rename = "functionTimeAlive")]
    pub function_time_alive: f64,
    #[serde(rename = "functionTimeCall")]
    pub function_time_call: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    pub version: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictResponse {
    pub images: Vec<ImageResult>,
    pub metadata: Metadata,
}

fn round(x: f64, decimals: u32) -> f64 {
    let f = 10f64.powi(decimals as i32);
    (x * f).round() / f
}

/// Convert raw detections into wire `ResultItem`s. When `normalize` is true,
/// coordinates are divided by original width/height and rounded to 5 decimals
/// (matching Ultralytics normalized output); otherwise rounded to `decimals`.
pub fn to_results(
    dets: &[Detection],
    backend: &dyn Backend,
    orig_w: u32,
    orig_h: u32,
    normalize: bool,
    decimals: u32,
) -> Vec<ResultItem> {
    let dec = if normalize { 5 } else { decimals };
    let (sw, sh) = (orig_w as f64, orig_h as f64);
    dets.iter()
        .map(|d| {
            let (x1, y1, x2, y2) = if normalize {
                (
                    round(d.x1 / sw, dec),
                    round(d.y1 / sh, dec),
                    round(d.x2 / sw, dec),
                    round(d.y2 / sh, dec),
                )
            } else {
                (
                    round(d.x1, dec),
                    round(d.y1, dec),
                    round(d.x2, dec),
                    round(d.y2, dec),
                )
            };
            ResultItem {
                name: backend.class_name(d.class),
                class: d.class,
                confidence: Some(round(d.confidence, dec.max(3))),
                r#box: Some(BoxJson { x1, y1, x2, y2 }),
            }
        })
        .collect()
}
