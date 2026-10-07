use std::path::Path;

use serde::Deserialize;

use crate::error::{ApiError, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerCfg,
    pub triton: TritonCfg,
    /// Public model served by this deployment (== Triton model name unless
    /// `triton_model_name` overrides it).
    pub default_model: String,
    /// Bearer API keys accepted on every request (flat list).
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub defaults: Defaults,
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
    /// Optional explicit Triton model name if it differs from `default_model`.
    #[serde(default)]
    pub triton_model_name: Option<String>,
}

fn default_timeout_ms() -> u64 {
    30_000
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_upload_mb: u64,
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

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            ApiError::Internal(format!("cannot read config {}: {e}", path.display()))
        })?;
        let cfg: Config = serde_yaml::from_str(&text)
            .map_err(|e| ApiError::Internal(format!("invalid config {}: {e}", path.display())))?;
        if cfg.api_keys.is_empty() {
            return Err(ApiError::Internal(
                "api_keys must not be empty; refuse to serve unauthenticated".into(),
            ));
        }
        for key in &cfg.api_keys {
            if key.is_empty() {
                return Err(ApiError::Internal("api_keys contains an empty key".into()));
            }
        }
        Ok(cfg)
    }

    pub fn triton_model(&self) -> &str {
        self.triton
            .triton_model_name
            .as_deref()
            .unwrap_or(&self.default_model)
    }

    pub fn max_upload_bytes(&self) -> usize {
        self.limits.max_upload_mb as usize * 1024 * 1024
    }
}
