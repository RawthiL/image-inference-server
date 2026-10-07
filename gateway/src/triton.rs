use serde_json::{Value, json};
use std::time::Duration;

use backend_api::{InputTensor, OutputTensor};

use crate::error::{ApiError, Result};

pub struct TritonClient {
    http: reqwest::Client,
    base: String,
    auth_header: Option<String>,
}

impl TritonClient {
    pub fn new(base: &str, api_key: Option<&str>, timeout: Duration) -> Self {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("failed to build reqwest client");
        Self {
            http,
            base: base.trim_end_matches('/').to_string(),
            auth_header: api_key.map(|k| format!("Bearer {k}")),
        }
    }

    fn with_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth_header {
            Some(v) => req.header("Authorization", v),
            None => req,
        }
    }

    pub async fn model_config(&self, model: &str) -> Result<Value> {
        let url = format!("{}/v2/models/{}/config", self.base, model);
        let req = self.http.get(&url).header("Accept", "application/json");
        let resp = self
            .with_auth(req)
            .send()
            .await
            .map_err(|e| ApiError::Upstream(format!("triton unreachable: {e}")))?;
        let status = resp.status();
        let body: Value = resp.json().await.map_err(|e| {
            ApiError::Upstream(format!("triton config for '{model}' is not JSON: {e}"))
        })?;
        if !status.is_success() {
            let msg = body
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown triton error");
            if status.as_u16() == 404 {
                return Err(ApiError::NotFound(format!(
                    "model '{model}' not found on Triton"
                )));
            }
            return Err(ApiError::Upstream(format!(
                "triton config for '{model}': {msg}"
            )));
        }
        Ok(body)
    }

    /// Best-effort server metadata for `metadata.version` (`GET /v2`).
    pub async fn server_metadata(&self) -> Option<String> {
        let url = format!("{}/v2", self.base);
        let resp = self.with_auth(self.http.get(&url)).send().await.ok()?;
        let v: Value = resp.json().await.ok()?;
        v.get("version")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
    }

    /// Run inference: send `input` as a binary FP32 tensor and request
    /// `outputs` as binary tensors (Triton binary tensor data extension).
    pub async fn infer(
        &self,
        model: &str,
        input: &InputTensor,
        outputs: &[String],
    ) -> Result<Vec<OutputTensor>> {
        let bytes_len = input.data.len() * 4;
        let header = json!({
            "inputs": [{
                "name": input.name,
                "shape": input.shape,
                "datatype": "FP32",
                "parameters": { "binary_data_size": bytes_len }
            }],
            "outputs": outputs.iter().map(|o| json!({
                "name": o,
                "parameters": { "binary_data": true }
            })).collect::<Vec<_>>()
        });

        let mut body =
            serde_json::to_vec(&header).map_err(|e| ApiError::Internal(e.to_string()))?;
        // Pad the JSON header with spaces to a multiple of 4 so the following
        // binary data is 4-byte aligned (tritonclient convention).
        while !body.len().is_multiple_of(4) {
            body.push(b' ');
        }
        let hcl = body.len();
        body.reserve(bytes_len);
        for v in &input.data {
            body.extend_from_slice(&v.to_le_bytes());
        }

        let url = format!("{}/v2/models/{}/infer", self.base, model);
        let req = self
            .http
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .header("Inference-Header-Content-Length", hcl.to_string())
            .body(body);
        let resp = self
            .with_auth(req)
            .send()
            .await
            .map_err(|e| ApiError::Upstream(format!("triton inference request failed: {e}")))?;

        let status = resp.status();
        let hcl_hdr = resp
            .headers()
            .get("Inference-Header-Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<usize>().ok());
        let full = resp
            .bytes()
            .await
            .map_err(|e| ApiError::Upstream(format!("triton response read failed: {e}")))?;

        if !status.is_success() {
            let msg = String::from_utf8_lossy(&full);
            return Err(ApiError::Upstream(format!(
                "triton infer error ({status}): {msg}"
            )));
        }

        let json_slice = match hcl_hdr {
            Some(n) if n <= full.len() => &full[..n],
            _ => &full[..],
        };
        let v: Value = serde_json::from_slice(json_slice)
            .map_err(|e| ApiError::Upstream(format!("unparseable triton response header: {e}")))?;
        let outs = v
            .get("outputs")
            .and_then(Value::as_array)
            .ok_or_else(|| ApiError::Upstream("triton response has no outputs".into()))?;

        let mut binary = &full[json_slice.len()..];
        let mut result = Vec::with_capacity(outs.len());
        for o in outs {
            let name = o
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let datatype = o
                .get("datatype")
                .and_then(Value::as_str)
                .unwrap_or("FP32")
                .to_string();
            let shape: Vec<u64> = o
                .get("shape")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|s| s.as_u64()).collect())
                .unwrap_or_default();
            let bds = o
                .pointer("/parameters/binary_data_size")
                .and_then(Value::as_u64)
                .map(|s| s as usize);
            let data = match bds {
                Some(n) => {
                    if binary.len() < n {
                        return Err(ApiError::Upstream("triton binary output truncated".into()));
                    }
                    let (chunk, rest) = binary.split_at(n);
                    binary = rest;
                    decode_binary(chunk, &datatype)?
                }
                None => o
                    .get("data")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_f64())
                            .map(|x| x as f32)
                            .collect()
                    })
                    .unwrap_or_default(),
            };
            result.push(OutputTensor { name, shape, data });
        }
        Ok(result)
    }
}

fn decode_binary(chunk: &[u8], datatype: &str) -> Result<Vec<f32>> {
    match datatype {
        "FP32" => {
            if !chunk.len().is_multiple_of(4) {
                return Err(ApiError::Upstream(
                    "FP32 tensor size not multiple of 4".into(),
                ));
            }
            Ok(chunk
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(b[..].try_into().unwrap()))
                .collect())
        }
        "FP16" => Ok(chunk
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16_as_f32(u16::from_le_bytes(b[..].try_into().unwrap())))
            .collect()),
        "UINT8" => Ok(chunk.iter().map(|b| *b as f32).collect()),
        other => Err(ApiError::Upstream(format!(
            "unsupported triton output datatype '{other}'"
        ))),
    }
}

/// IEEE half -> f32 bit expansion (no external crate).
fn f16_as_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal: normalize
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            let m = m & 0x3ff;
            let exp = (127 - 15 - e) as u32;
            (sign << 31) | (exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}
