//! Compile-time registry of model backends, one table per task.
//!
//! To add a backend: create `backends/<name>/` (see `backends/README.md`),
//! add it to the workspace + a `<name>` feature in `gateway/Cargo.toml`, and
//! add one line to its task's table below.

use std::sync::Arc;

use backend_api::ocr::{OcrBackend, OcrFactory};
use backend_api::{Backend, Factory, ModelInfo};
use serde_json::Value;

use crate::error::{ApiError, Result};
use crate::triton::TritonClient;

/// `(family, constructor)` for every `detect` backend compiled in.
pub const DETECT_BACKENDS: &[(&str, Factory)] = &[
    #[cfg(feature = "yolo")]
    (backend_yolo::FAMILY, backend_yolo::create),
    #[cfg(feature = "rfdetr")]
    (backend_rfdetr::FAMILY, backend_rfdetr::create),
];

/// `(family, constructor)` for every `ocr` backend compiled in.
pub const OCR_BACKENDS: &[(&str, OcrFactory)] = &[
    #[cfg(feature = "ppocr")]
    (backend_ppocr::FAMILY, backend_ppocr::create),
];

/// Load a `detect` backend for a Triton model.
pub async fn load_detect(triton: &TritonClient, model: &str) -> Result<Arc<dyn Backend>> {
    load(triton, model, "detect", DETECT_BACKENDS).await
}

/// Load an `ocr` backend for a Triton model (its detector) and companions.
pub async fn load_ocr(triton: &TritonClient, model: &str) -> Result<Arc<dyn OcrBackend>> {
    load(triton, model, "ocr", OCR_BACKENDS).await
}

/// A backend constructor returning task trait object `B`.
type FactoryOf<B> = fn(&ModelInfo) -> backend_api::Result<Box<B>>;

/// A Triton model's config, reduced to what backends see.
struct Fetched {
    name: String,
    metadata: Value,
    max_batch_size: u64,
}

impl Fetched {
    fn info<'a>(&'a self, companions: &'a [ModelInfo<'a>]) -> ModelInfo<'a> {
        ModelInfo {
            name: &self.name,
            metadata: &self.metadata,
            max_batch_size: self.max_batch_size,
            companions,
        }
    }
}

async fn fetch(triton: &TritonClient, model: &str) -> Result<Fetched> {
    let config = triton.model_config(model).await?;
    Ok(Fetched {
        name: model.to_string(),
        metadata: metadata(model, &config)?,
        max_batch_size: config
            .get("max_batch_size")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Fetch `model` (and its `companions`), check it serves `task`, and build
/// the backend named by its `family` metadata from `table`.
async fn load<B: ?Sized>(
    triton: &TritonClient,
    model: &str,
    task: &str,
    table: &[(&str, FactoryOf<B>)],
) -> Result<Arc<B>> {
    let main = fetch(triton, model).await?;
    let model_task = main
        .metadata
        .get("task")
        .and_then(Value::as_str)
        .unwrap_or("detect");
    if model_task != task {
        return Err(ApiError::Internal(format!(
            "model '{model}' is a '{model_task}' model, but it is configured for the \
             '{task}' endpoint"
        )));
    }
    let factory = factory_for(model, &main.metadata, task, table)?;

    let mut companions = Vec::new();
    for name in companion_names(model, &main.metadata)? {
        companions.push(fetch(triton, &name).await?);
    }
    let companion_infos: Vec<ModelInfo> = companions.iter().map(|c| c.info(&[])).collect();
    let backend =
        factory(&main.info(&companion_infos)).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(Arc::from(backend))
}

/// The JSON embedded in `parameters.metadata.string_value` of a model config.
fn metadata(model: &str, config: &Value) -> Result<Value> {
    let raw = config
        .pointer("/parameters/metadata/string_value")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::Internal(format!(
                "model '{model}' has no parameters.metadata string_value; \
                 re-export it with backends/<family>/export.py"
            ))
        })?;
    serde_json::from_str(raw)
        .map_err(|e| ApiError::Internal(format!("model '{model}' metadata is not valid JSON: {e}")))
}

fn factory_for<F: Copy>(
    model: &str,
    metadata: &Value,
    task: &str,
    table: &[(&str, F)],
) -> Result<F> {
    let family = metadata
        .get("family")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::Internal(format!("model '{model}' metadata has no 'family'")))?
        .to_ascii_lowercase();
    table
        .iter()
        .find(|(f, _)| *f == family)
        .map(|(_, factory)| *factory)
        .ok_or_else(|| {
            let names: Vec<&str> = table.iter().map(|(f, _)| *f).collect();
            ApiError::Internal(format!(
                "model '{model}' has family '{family}', but this gateway was built with \
                 {task} backends {names:?}"
            ))
        })
}

fn companion_names(model: &str, metadata: &Value) -> Result<Vec<String>> {
    match metadata.get("companions") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| {
            ApiError::Internal(format!(
                "model '{model}' metadata 'companions' must be a list of model names"
            ))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_family_names_the_task_and_compiled_backends() {
        let table: &[(&str, u8)] = &[("yolo", 1)];
        assert_eq!(
            factory_for("m", &json!({"family": "YOLO"}), "detect", table).unwrap(),
            1
        );
        let err = factory_for("m", &json!({"family": "x"}), "ocr", table).unwrap_err();
        assert!(err.to_string().contains("ocr backends [\"yolo\"]"), "{err}");
    }

    #[test]
    fn companions_are_optional_but_typed() {
        assert!(companion_names("m", &json!({})).unwrap().is_empty());
        assert_eq!(
            companion_names("m", &json!({"companions": ["rec"]})).unwrap(),
            vec!["rec"]
        );
        assert!(companion_names("m", &json!({"companions": "rec"})).is_err());
    }
}
