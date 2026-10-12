//! Object detection, Ultralytics-compatible: `POST /predict` returning the
//! Ultralytics Platform `PredictResponse`.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::routing::post;
use axum::{Json, Router};

use backend_api::DecodeParams;

use crate::config::Defaults;
use crate::error::{ApiError, Result};
use crate::input::{self, elapsed_ms};
use crate::response::{self, ImageResult, Metadata, PredictResponse, Speed};
use crate::sources;
use crate::state::AppState;

/// Maximum detections returned per image (Ultralytics `max_det` default).
pub const MAX_DET: usize = 300;

const KNOWN_PARAMS: [&str; 7] = [
    "conf",
    "iou",
    "imgsz",
    "normalize",
    "decimals",
    "bits",
    "vid_stride",
];

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/predict", post(predict)).route(
        "/api/deployments/{owner}/{deployment}/predict",
        post(predict),
    )
}

/// Parsed `predict` form parameters (all optional with config defaults).
struct PredictParams {
    conf: f64,
    iou: f64,
    normalize: bool,
    decimals: u32,
}

async fn predict(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<PredictResponse>> {
    let cfg = state.cfg.endpoints.detect.as_ref().expect("detect config");
    input::authorize(&headers, &cfg.api_keys)?;
    let slot = state
        .detect
        .as_ref()
        .expect("detect route mounted without its slot");
    let backend = slot.get()?.clone();
    let call_start = Instant::now();

    let (file, source, fields) =
        input::collect_form(&headers, req, &state.cfg, &KNOWN_PARAMS).await?;
    let params = parse_params(&cfg.defaults, &fields)?;

    let bytes = match (file, source) {
        (Some(b), None) => b,
        (None, Some(s)) => sources::resolve_source(&state.fetch, &s, &state.cfg.limits).await?,
        (Some(_), Some(_)) => {
            return Err(ApiError::BadRequest(
                "provide either 'file' or 'source', not both".into(),
            ));
        }
        (None, None) => {
            return Err(ApiError::BadRequest(
                "one of 'file' or 'source' is required".into(),
            ));
        }
    };
    if bytes.is_empty() {
        return Err(ApiError::BadRequest("'file' is empty".into()));
    }

    let b = backend.clone();
    let (prepared, preprocess_ms) =
        input::decode_then(bytes, state.cfg.limits.max_image_pixels, move |image| {
            b.preprocess(&image)
        })
        .await?;
    let (orig_w, orig_h) = (prepared.map.orig_w, prepared.map.orig_h);

    // --- inference (timed) ---
    let t1 = Instant::now();
    let outputs = state
        .triton
        .infer(&slot.triton_model, &prepared.input, backend.output_names())
        .await?;
    let inference_ms = elapsed_ms(t1);

    // --- postprocess (timed) ---
    let t2 = Instant::now();
    let dets = backend
        .postprocess(
            &outputs,
            &prepared.map,
            &DecodeParams {
                conf: params.conf,
                iou: params.iou,
                max_det: MAX_DET,
            },
        )
        .map_err(|e| ApiError::Upstream(e.to_string()))?;
    let postprocess_ms = elapsed_ms(t2);

    let results = response::to_results(
        &dets,
        backend.as_ref(),
        orig_w,
        orig_h,
        params.normalize,
        params.decimals,
    );

    let mut version = serde_json::Map::new();
    version.insert(
        "gateway".into(),
        serde_json::json!(env!("CARGO_PKG_VERSION")),
    );
    version.insert("backend".into(), serde_json::json!("triton"));
    if let Some(Some(v)) = state.triton_version.get() {
        version.insert("triton".into(), serde_json::json!(v));
    }

    let names = backend.class_names();
    let class_names = (!names.is_empty()).then(|| names.to_vec());

    Ok(Json(PredictResponse {
        images: vec![ImageResult {
            shape: [orig_h, orig_w],
            speed: Speed {
                preprocess: preprocess_ms,
                inference: inference_ms,
                postprocess: postprocess_ms,
            },
            results,
        }],
        metadata: Metadata {
            image_count: 1,
            class_names,
            function_time_alive: state.started.elapsed().as_secs_f64(),
            function_time_call: call_start.elapsed().as_secs_f64(),
            model: Some(slot.model.clone()),
            task: Some("detect".into()),
            version,
        },
    }))
}

fn parse_params(defaults: &Defaults, fields: &[(String, String)]) -> Result<PredictParams> {
    let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
    let conf = match get("conf") {
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| ApiError::BadRequest("conf must be a number".into()))?,
        None => defaults.conf,
    };
    if !(0.01..=1.0).contains(&conf) {
        return Err(ApiError::BadRequest(
            "conf must be between 0.01 and 1".into(),
        ));
    }
    let iou = match get("iou") {
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| ApiError::BadRequest("iou must be a number".into()))?,
        None => defaults.iou,
    };
    if !(0.0..=0.95).contains(&iou) {
        return Err(ApiError::BadRequest(
            "iou must be between 0 and 0.95".into(),
        ));
    }
    if let Some(v) = get("imgsz") {
        let imgsz: i64 = v
            .parse()
            .map_err(|_| ApiError::BadRequest("imgsz must be an integer".into()))?;
        if !(32..=1280).contains(&imgsz) {
            return Err(ApiError::BadRequest(
                "imgsz must be between 32 and 1280".into(),
            ));
        }
        // Accepted for API compatibility; inference always runs at the model's
        // native resolution and boxes are mapped back to original pixels.
    }
    let normalize = match get("normalize") {
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => {
                return Err(ApiError::BadRequest(
                    "normalize must be true or false".into(),
                ));
            }
        },
        None => false,
    };
    let decimals = match get("decimals") {
        Some(v) => {
            let d: i64 = v
                .parse()
                .map_err(|_| ApiError::BadRequest("decimals must be an integer".into()))?;
            if !(0..=10).contains(&d) {
                return Err(ApiError::BadRequest(
                    "decimals must be between 0 and 10".into(),
                ));
            }
            d as u32
        }
        None => defaults.decimals.unwrap_or(3),
    };
    Ok(PredictParams {
        conf,
        iou,
        normalize,
        decimals,
    })
}
