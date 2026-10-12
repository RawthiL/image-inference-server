//! The client-facing APIs. Each submodule is one task's wire contract,
//! compatible with an existing public API; only endpoints enabled in the
//! config are mounted.

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use serde_json::{Map, Value, json};

use crate::state::{AppState, ModelSlot};

pub mod detect;
pub mod vision;

pub fn router(state: Arc<AppState>) -> Router {
    let mut app = Router::new().route("/health", get(health));
    if state.detect.is_some() {
        // Body limit slightly above max_upload so an over-limit *file* is
        // caught by the explicit `bytes.len() > max_upload` check (clean 413)
        // rather than by axum's multipart parser (400). Framing overhead
        // lives in this margin.
        let limit = state.cfg.max_upload_bytes() + 2 * 1024 * 1024;
        app = app.merge(detect::routes().layer(axum::extract::DefaultBodyLimit::max(limit)));
    }
    if state.ocr.is_some() {
        // Reads its JSON body itself, with a base64-sized cap.
        app = app.merge(vision::routes());
    }
    app.with_state(state)
}

/// `200` when every enabled endpoint's models are loaded and Triton still
/// serves them; `503` otherwise. Per-endpoint detail under `endpoints`.
async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut endpoints = Map::new();
    if let Some(slot) = &state.detect {
        let models = slot
            .get()
            .map(|_| vec![slot.triton_model.clone()])
            .unwrap_or_default();
        endpoints.insert("detect".into(), check(&state, slot, models).await);
    }
    if let Some(slot) = &state.ocr {
        let models = slot
            .get()
            .map(|b| vec![b.detector().model.clone(), b.recognizer().model.clone()])
            .unwrap_or_default();
        endpoints.insert("ocr".into(), check(&state, slot, models).await);
    }
    let statuses: Vec<&str> = endpoints
        .values()
        .filter_map(|e| e["status"].as_str())
        .collect();
    let status = ["unavailable", "loading"]
        .into_iter()
        .find(|s| statuses.contains(s))
        .unwrap_or("healthy");
    let body = json!({ "status": status, "endpoints": endpoints });
    let code = if status == "healthy" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(body))
}

/// Readiness of one endpoint: loaded, and Triton still serves `models`.
async fn check<B: ?Sized>(state: &AppState, slot: &ModelSlot<B>, models: Vec<String>) -> Value {
    if let Err(e) = slot.get() {
        return json!({ "status": "loading", "model": slot.model, "error": e.to_string() });
    }
    for m in &models {
        if let Err(e) = state.triton.model_config(m).await {
            return json!({ "status": "unavailable", "model": slot.model, "error": e.to_string() });
        }
    }
    json!({ "status": "healthy", "model": slot.model })
}
