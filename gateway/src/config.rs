use std::path::Path;

use serde::Deserialize;

use crate::error::{ApiError, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerCfg,
    pub triton: TritonCfg,
    /// The APIs (services) this deployment serves. Only enabled endpoints
    /// are routed, each with its own API keys.
    pub endpoints: Endpoints,
    #[serde(default)]
    pub limits: Limits,
}

/// One block per API. A missing block or `enabled: false` means the
/// endpoint's routes are not mounted.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Endpoints {
    /// Ultralytics-compatible object detection (`POST /predict`).
    pub detect: Option<DetectEndpoint>,
    /// Google Cloud Vision-compatible OCR (`POST /v1/images:annotate`).
    pub ocr: Option<OcrEndpoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectEndpoint {
    #[serde(default = "enabled")]
    pub enabled: bool,
    /// Public model name (== Triton model name unless `triton_model_name`).
    pub model: String,
    #[serde(default)]
    pub triton_model_name: Option<String>,
    /// API keys accepted by this endpoint only.
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub defaults: Defaults,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrEndpoint {
    #[serde(default = "enabled")]
    pub enabled: bool,
    /// Public model name of the text *detector*; its recognizer is listed in
    /// the detector's `companions` metadata.
    pub model: String,
    #[serde(default)]
    pub triton_model_name: Option<String>,
    /// API keys accepted by this endpoint only.
    pub api_keys: Vec<String>,
    /// Most images accepted in one `images:annotate` call.
    #[serde(default = "default_max_images")]
    pub max_images_per_request: usize,
    #[serde(default)]
    pub defaults: OcrDefaults,
}

fn enabled() -> bool {
    true
}

fn default_max_images() -> usize {
    16
}

impl DetectEndpoint {
    pub fn triton_model(&self) -> &str {
        self.triton_model_name.as_deref().unwrap_or(&self.model)
    }
}

impl OcrEndpoint {
    pub fn triton_model(&self) -> &str {
        self.triton_model_name.as_deref().unwrap_or(&self.model)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerCfg {
    pub listen: String,
}

impl Default for ServerCfg {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TritonCfg {
    pub url: String,
    /// Optional bearer key for an authenticated Triton front (e.g. the legacy
    /// nginx gateway). Not required when Triton is network-internal.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    30_000
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_upload_mb: u64,
    /// Accept `source` as an http(s) URL (the gateway then downloads it).
    /// Off by default: the server only accepts images in the request payload.
    pub allow_url_sources: bool,
    pub url_timeout_ms: u64,
    /// Allow `source` URLs resolving to private/loopback ranges (dev only).
    pub allow_private_urls: bool,
    /// Largest accepted image (width x height), checked before decoding.
    pub max_image_pixels: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_upload_mb: 20,
            allow_url_sources: false,
            url_timeout_ms: 10_000,
            allow_private_urls: false,
            max_image_pixels: 40_000_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Defaults {
    pub conf: f64,
    pub iou: f64,
    /// Coordinate decimals for pixel-space boxes (normalized boxes always use 5
    /// unless overridden).
    pub decimals: Option<u32>,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            conf: 0.25,
            iou: 0.7,
            decimals: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OcrDefaults {
    /// Minimum text-region score (unset: the model's own default).
    pub min_region_score: Option<f64>,
    /// Recognized lines below this confidence are dropped.
    pub min_text_confidence: f64,
    /// Most text regions kept per image.
    pub max_regions: usize,
}

impl Default for OcrDefaults {
    fn default() -> Self {
        Self {
            min_region_score: None,
            min_text_confidence: 0.5,
            max_regions: 1000,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            ApiError::Internal(format!("cannot read config {}: {e}", path.display()))
        })?;
        let cfg: Config = serde_yaml::from_str(&text)
            .map_err(|e| ApiError::Internal(format!("invalid config {}: {e}", path.display())))?;
        cfg.validate()
            .map_err(|e| ApiError::Internal(format!("invalid config {}: {e}", path.display())))
    }

    /// Drop disabled endpoints (so the rest of the gateway only sees served
    /// ones) and check what serde can't.
    fn validate(mut self) -> std::result::Result<Self, String> {
        self.endpoints.detect = self.endpoints.detect.filter(|e| e.enabled);
        self.endpoints.ocr = self.endpoints.ocr.filter(|e| e.enabled);
        if self.endpoints.detect.is_none() && self.endpoints.ocr.is_none() {
            return Err("no endpoint enabled; configure `endpoints.detect` and/or \
                        `endpoints.ocr`"
                .into());
        }
        let keyed = [
            (
                "detect",
                self.endpoints.detect.as_ref().map(|e| &e.api_keys),
            ),
            ("ocr", self.endpoints.ocr.as_ref().map(|e| &e.api_keys)),
        ];
        for (name, keys) in keyed {
            let Some(keys) = keys else { continue };
            if keys.is_empty() {
                return Err(format!(
                    "endpoints.{name}.api_keys must not be empty; refuse to serve \
                     unauthenticated"
                ));
            }
            if keys.iter().any(String::is_empty) {
                return Err(format!("endpoints.{name}.api_keys contains an empty key"));
            }
        }
        if let Some(ocr) = &self.endpoints.ocr
            && ocr.max_images_per_request == 0
        {
            return Err("endpoints.ocr.max_images_per_request must be at least 1".into());
        }
        Ok(self)
    }

    pub fn max_upload_bytes(&self) -> usize {
        self.limits.max_upload_mb as usize * 1024 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> std::result::Result<Config, String> {
        let base = "triton: { url: \"http://t\" }\n";
        serde_yaml::from_str::<Config>(&format!("{base}{yaml}"))
            .map_err(|e| e.to_string())?
            .validate()
    }

    #[test]
    fn endpoints_are_enabled_independently_with_their_own_keys() {
        let cfg = parse(
            "endpoints:\n  detect: { model: d, api_keys: [a], enabled: false }\n  \
             ocr: { model: o, api_keys: [b] }\n",
        )
        .unwrap();
        assert!(cfg.endpoints.detect.is_none());
        let ocr = cfg.endpoints.ocr.unwrap();
        assert_eq!(
            (ocr.triton_model(), ocr.api_keys.as_slice()),
            ("o", &["b".to_string()][..])
        );
    }

    #[test]
    fn rejects_unkeyed_empty_or_legacy_configs() {
        assert!(parse("endpoints: { ocr: { model: o, api_keys: [] } }\n").is_err());
        assert!(parse("endpoints: { ocr: { model: o, api_keys: [\"\"] } }\n").is_err());
        assert!(parse("endpoints: { ocr: { model: o } }\n").is_err());
        assert!(
            parse("endpoints: { ocr: { model: o, api_keys: [k], enabled: false } }\n").is_err()
        );
        assert!(parse("default_model: m\napi_keys: [k]\n").is_err());
        assert!(parse("").is_err());
    }
}
