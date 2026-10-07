use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use subtle::ConstantTimeEq;

use backend_api::{Backend, DecodeParams};

use crate::config::Config;
use crate::error::{ApiError, Result};
use crate::response::{self, ImageResult, Metadata, PredictResponse, Speed};
use crate::triton::TritonClient;
use crate::{imaging, registry, sources};

/// Maximum detections returned per image (Ultralytics `max_det` default).
pub const MAX_DET: usize = 300;

pub struct AppState {
    pub cfg: Arc<Config>,
    pub triton: Arc<TritonClient>,
    pub fetch: reqwest::Client,
    pub started: Instant,
    /// Set once the model is loaded from Triton (see [`spawn_model_loader`]).
    /// Until then every request gets a 503 JSON error instead of no answer.
    pub ready: OnceLock<Ready>,
    /// Why the model is not ready yet (last loader error), for 503 bodies.
    pub load_error: Mutex<Option<String>>,
}

/// The loaded model.
pub struct Ready {
    /// The model backend selected from the Triton model's metadata.
    pub backend: Arc<dyn Backend>,
    pub triton_version: Option<String>,
}

impl AppState {
    pub fn new(cfg: Arc<Config>, triton: Arc<TritonClient>, fetch: reqwest::Client) -> Self {
        Self {
            cfg,
            triton,
            fetch,
            started: Instant::now(),
            ready: OnceLock::new(),
            load_error: Mutex::new(None),
        }
    }

    /// The loaded model, or a 503 explaining why it is not available yet.
    pub fn ready(&self) -> Result<&Ready> {
        self.ready.get().ok_or_else(|| {
            let reason = self
                .load_error
                .lock()
                .ok()
                .and_then(|e| e.clone())
                .unwrap_or_else(|| "still loading".into());
            ApiError::Upstream(format!(
                "model '{}' is not ready: {reason}",
                self.cfg.default_model
            ))
        })
    }
}

pub const MAX_SOURCE_LEN: usize = 4096;

pub fn router(state: Arc<AppState>) -> Router {
    // Body limit slightly above max_upload so an over-limit *file* is caught by
    // the explicit `bytes.len() > max_upload` check (clean 413) rather than by
    // axum's multipart parser (400). Framing overhead lives in this margin.
    let limit = state.cfg.max_upload_bytes() + 2 * 1024 * 1024;
    Router::new()
        .route("/predict", post(predict))
        .route(
            "/api/deployments/{owner}/{deployment}/predict",
            post(predict),
        )
        .route("/health", get(health))
        .layer(axum::extract::DefaultBodyLimit::max(limit))
        .with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let model = &state.cfg.default_model;
    if let Err(e) = state.ready() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                serde_json::json!({ "status": "loading", "model": model, "error": e.to_string() }),
            ),
        );
    }
    // Readiness: confirm Triton still serves the model.
    match state.triton.model_config(state.cfg.triton_model()).await {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "healthy", "model": model })),
        ),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                serde_json::json!({ "status": "unavailable", "model": model, "error": e.to_string() }),
            ),
        ),
    }
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
    authorize(&headers, &state.cfg)?;
    let ready = state.ready()?;
    let call_start = Instant::now();

    // The Ultralytics endpoint (FastAPI `Form`) accepts both multipart and
    // urlencoded bodies; support both for a true drop-in.
    let ctype = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let (file, source, fields) = if ctype.starts_with("multipart/form-data") {
        collect_multipart(req, &state.cfg).await?
    } else if ctype.starts_with("application/x-www-form-urlencoded") {
        collect_urlencoded(req, &state.cfg).await?
    } else {
        return Err(ApiError::BadRequest(
            "expected a multipart/form-data or x-www-form-urlencoded body".into(),
        ));
    };

    let params = parse_params(&state.cfg, &fields)?;

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

    // Decode + preprocess are CPU-bound: keep them off the async workers.
    let backend = ready.backend.clone();
    let max_pixels = state.cfg.limits.max_image_pixels;
    let (prepared, preprocess_ms) = tokio::task::spawn_blocking(move || {
        let image = imaging::decode(&bytes, max_pixels)?;
        let t0 = Instant::now();
        let prepared = backend.preprocess(&image);
        Ok::<_, ApiError>((prepared, elapsed_ms(t0)))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("preprocess task failed: {e}")))??;
    let (orig_w, orig_h) = (prepared.map.orig_w, prepared.map.orig_h);

    // --- inference (timed) ---
    let t1 = Instant::now();
    let outputs = state
        .triton
        .infer(
            state.cfg.triton_model(),
            &prepared.input,
            ready.backend.output_names(),
        )
        .await?;
    let inference_ms = elapsed_ms(t1);

    // --- postprocess (timed) ---
    let t2 = Instant::now();
    let dets = ready
        .backend
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
        ready.backend.as_ref(),
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
    if let Some(v) = &ready.triton_version {
        version.insert("triton".into(), serde_json::json!(v));
    }

    let names = ready.backend.class_names();
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
            model: Some(state.cfg.default_model.clone()),
            task: Some("detect".into()),
            version,
        },
    }))
}

type Collected = (Option<Vec<u8>>, Option<String>, Vec<(String, String)>);

const KNOWN_PARAMS: [&str; 7] = [
    "conf",
    "iou",
    "imgsz",
    "normalize",
    "decimals",
    "bits",
    "vid_stride",
];

async fn collect_multipart(req: Request, cfg: &Config) -> Result<Collected> {
    let max_upload = cfg.max_upload_bytes();
    let mut file: Option<Vec<u8>> = None;
    let mut source: Option<String> = None;
    let mut fields: Vec<(String, String)> = Vec::new();

    let mut multipart = Multipart::from_request(req, &())
        .await
        .map_err(|e| ApiError::BadRequest(format!("expected multipart/form-data: {e}")))?;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::BadRequest(format!("malformed multipart body: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let bytes = field.bytes().await.map_err(|e| {
                    let msg = e.to_string();
                    if msg.contains("length limit") {
                        ApiError::TooLarge(format!(
                            "file too large (max {} MB)",
                            cfg.limits.max_upload_mb
                        ))
                    } else {
                        ApiError::BadRequest(format!("cannot read file field: {msg}"))
                    }
                })?;
                if bytes.len() > max_upload {
                    return Err(ApiError::TooLarge(format!(
                        "file too large (max {} MB)",
                        cfg.limits.max_upload_mb
                    )));
                }
                file = Some(bytes.to_vec());
            }
            "source" => {
                let text = field
                    .text()
                    .await
                    .map_err(|_| ApiError::BadRequest("source must be text".into()))?;
                source = Some(check_source(text)?);
            }
            n if KNOWN_PARAMS.contains(&n) => {
                let text = field
                    .text()
                    .await
                    .map_err(|_| ApiError::BadRequest(format!("{name} must be text")))?;
                fields.push((name, text));
            }
            _ => {
                // Unknown fields are ignored (forward compatible with the spec).
                let _ = field.bytes().await;
            }
        }
    }
    Ok((file, source, fields))
}

async fn collect_urlencoded(req: Request, cfg: &Config) -> Result<Collected> {
    let body = axum::body::to_bytes(req.into_body(), cfg.max_upload_bytes() + 16 * 1024 * 1024)
        .await
        .map_err(|_| ApiError::TooLarge("form body too large".into()))?;
    let file: Option<Vec<u8>> = None; // 'file' requires multipart (rejected above)
    let mut source: Option<String> = None;
    let mut fields: Vec<(String, String)> = Vec::new();
    // Percent-decoding matters: base64 `+`/`/` are encoded as %2B/%2F by clients.
    for (key, value) in form_urlencoded::parse(&body) {
        match key.as_ref() {
            "file" => {
                return Err(ApiError::BadRequest(
                    "'file' must be sent as a multipart/form-data part".into(),
                ));
            }
            "source" => source = Some(check_source(value.into_owned())?),
            n if KNOWN_PARAMS.contains(&n) => fields.push((key.into_owned(), value.into_owned())),
            _ => {}
        }
    }
    Ok((file, source, fields))
}

fn check_source(text: String) -> Result<String> {
    if text.len() > MAX_SOURCE_LEN {
        return Err(ApiError::BadRequest(format!(
            "source too long (max {MAX_SOURCE_LEN} characters, per API spec)"
        )));
    }
    Ok(text)
}

fn authorize(headers: &HeaderMap, cfg: &Config) -> Result<()> {
    let raw = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let Some(token) = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
    else {
        return Err(ApiError::Unauthorized);
    };
    // Any order: iterate all keys with constant-time comparison.
    let mut ok = 0u8;
    for key in &cfg.api_keys {
        ok |= u8::from(bool::from(token.as_bytes().ct_eq(key.as_bytes())));
    }
    if ok == 1 {
        Ok(())
    } else {
        Err(ApiError::Unauthorized)
    }
}

fn parse_params(cfg: &Config, fields: &[(String, String)]) -> Result<PredictParams> {
    let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
    let conf = match get("conf") {
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| ApiError::BadRequest("conf must be a number".into()))?,
        None => cfg.defaults.conf,
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
        None => cfg.defaults.iou,
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
        None => cfg.defaults.decimals.unwrap_or(3),
    };
    Ok(PredictParams {
        conf,
        iou,
        normalize,
        decimals,
    })
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

/// Load the model in the background, retrying with backoff until Triton
/// serves it. The HTTP server runs meanwhile and answers 503 with the reason.
pub fn spawn_model_loader(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let model = state.cfg.triton_model().to_string();
        let mut delay = Duration::from_millis(500);
        loop {
            let attempt = match state.triton.model_config(&model).await {
                Ok(v) => registry::backend_for(&model, &v),
                Err(e) => Err(e),
            };
            match attempt {
                Ok(backend) => {
                    tracing::info!(
                        "model '{model}' ready -> backend '{}' (input {:?}, {} classes, outputs {:?})",
                        backend.family(),
                        backend.input_size(),
                        backend.class_names().len(),
                        backend.output_names()
                    );
                    let triton_version = state.triton.server_metadata().await;
                    let _ = state.ready.set(Ready {
                        backend,
                        triton_version,
                    });
                    if let Ok(mut e) = state.load_error.lock() {
                        *e = None;
                    }
                    return;
                }
                Err(e) => {
                    let msg = e.to_string();
                    if let Ok(mut last) = state.load_error.lock()
                        && last.as_deref() != Some(msg.as_str())
                    {
                        tracing::warn!("model '{model}' not ready (retrying): {msg}");
                        *last = Some(msg);
                    }
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(Duration::from_secs(10));
                }
            }
        }
    })
}
