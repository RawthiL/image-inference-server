//! Shared server state: config, clients, and one model slot per enabled
//! endpoint, each loaded from Triton in the background.

use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use backend_api::Backend;
use backend_api::ocr::OcrBackend;

use crate::config::Config;
use crate::error::{ApiError, Result};
use crate::registry;
use crate::triton::TritonClient;

pub struct AppState {
    pub cfg: Arc<Config>,
    pub triton: Arc<TritonClient>,
    pub fetch: reqwest::Client,
    pub started: Instant,
    /// Triton server version, fetched once the first model loads.
    pub triton_version: OnceLock<Option<String>>,
    /// `Some` iff the detect endpoint is enabled.
    pub detect: Option<ModelSlot<dyn Backend>>,
    /// `Some` iff the OCR endpoint is enabled.
    pub ocr: Option<ModelSlot<dyn OcrBackend>>,
}

/// The model behind one endpoint. Until it loads, requests get a 503 JSON
/// error with the last load error instead of no answer.
pub struct ModelSlot<B: ?Sized> {
    /// Public model name (config `model`).
    pub model: String,
    /// Triton model name.
    pub triton_model: String,
    ready: OnceLock<Arc<B>>,
    load_error: Mutex<Option<String>>,
}

impl<B: ?Sized> ModelSlot<B> {
    fn new(model: &str, triton_model: &str) -> Self {
        Self {
            model: model.to_string(),
            triton_model: triton_model.to_string(),
            ready: OnceLock::new(),
            load_error: Mutex::new(None),
        }
    }

    /// The loaded backend, or a 503 explaining why it is not available yet.
    pub fn get(&self) -> Result<&Arc<B>> {
        self.ready.get().ok_or_else(|| {
            ApiError::Upstream(format!(
                "model '{}' is not ready: {}",
                self.model,
                self.load_error().unwrap_or_else(|| "still loading".into())
            ))
        })
    }

    pub fn is_ready(&self) -> bool {
        self.ready.get().is_some()
    }

    /// Last loader error, if the model is not loaded yet.
    pub fn load_error(&self) -> Option<String> {
        self.load_error.lock().ok().and_then(|e| e.clone())
    }
}

impl AppState {
    pub fn new(cfg: Arc<Config>, triton: Arc<TritonClient>, fetch: reqwest::Client) -> Self {
        let detect = cfg
            .endpoints
            .detect
            .as_ref()
            .map(|e| ModelSlot::new(&e.model, e.triton_model()));
        let ocr = cfg
            .endpoints
            .ocr
            .as_ref()
            .map(|e| ModelSlot::new(&e.model, e.triton_model()));
        Self {
            cfg,
            triton,
            fetch,
            started: Instant::now(),
            triton_version: OnceLock::new(),
            detect,
            ocr,
        }
    }

    /// True once every enabled endpoint's model is loaded.
    pub fn is_ready(&self) -> bool {
        self.detect.as_ref().is_none_or(ModelSlot::is_ready)
            && self.ocr.as_ref().is_none_or(ModelSlot::is_ready)
    }

    /// Current load errors of models that are not loaded yet.
    pub fn load_errors(&self) -> Vec<String> {
        let detect = self.detect.as_ref().and_then(ModelSlot::load_error);
        let ocr = self.ocr.as_ref().and_then(ModelSlot::load_error);
        detect.into_iter().chain(ocr).collect()
    }
}

/// Load every enabled endpoint's model in the background, retrying with
/// backoff until Triton serves it. The HTTP server runs meanwhile and
/// answers 503 with the reason.
pub fn spawn_model_loaders(state: Arc<AppState>) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::new();
    if state.detect.is_some() {
        let st = state.clone();
        handles.push(tokio::spawn(async move {
            let slot = st.detect.as_ref().expect("detect slot");
            load_until_ready(&st, slot, "detect", || {
                registry::load_detect(&st.triton, &slot.triton_model)
            })
            .await;
            if let Some(b) = slot.ready.get() {
                tracing::info!(
                    "detect model '{}' ready -> backend '{}' (input {:?}, {} classes, outputs {:?})",
                    slot.triton_model,
                    b.family(),
                    b.input_size(),
                    b.class_names().len(),
                    b.output_names()
                );
            }
        }));
    }
    if state.ocr.is_some() {
        let st = state.clone();
        handles.push(tokio::spawn(async move {
            let slot = st.ocr.as_ref().expect("ocr slot");
            load_until_ready(&st, slot, "ocr", || {
                registry::load_ocr(&st.triton, &slot.triton_model)
            })
            .await;
            if let Some(b) = slot.ready.get() {
                tracing::info!(
                    "ocr model '{}' ready -> backend '{}' (detector '{}', recognizer '{}')",
                    slot.triton_model,
                    b.family(),
                    b.detector().model,
                    b.recognizer().model
                );
            }
        }));
    }
    handles
}

async fn load_until_ready<B: ?Sized, F, Fut>(
    state: &AppState,
    slot: &ModelSlot<B>,
    task: &str,
    load: F,
) where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<Arc<B>>>,
{
    let model = &slot.triton_model;
    let mut delay = Duration::from_millis(500);
    loop {
        match load().await {
            Ok(backend) => {
                if state.triton_version.get().is_none() {
                    let _ = state
                        .triton_version
                        .set(state.triton.server_metadata().await);
                }
                let _ = slot.ready.set(backend);
                if let Ok(mut e) = slot.load_error.lock() {
                    *e = None;
                }
                return;
            }
            Err(e) => {
                let msg = e.to_string();
                if let Ok(mut last) = slot.load_error.lock()
                    && last.as_deref() != Some(msg.as_str())
                {
                    tracing::warn!("{task} model '{model}' not ready (retrying): {msg}");
                    *last = Some(msg);
                }
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(10));
            }
        }
    }
}
