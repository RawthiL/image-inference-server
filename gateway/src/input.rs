//! Request plumbing shared by every endpoint: API-key auth, form bodies
//! (multipart / urlencoded), and decoding images off the async workers.

use std::time::Instant;

use axum::extract::{FromRequest, Multipart, Request};
use axum::http::HeaderMap;
use backend_api::image::RgbImage;
use subtle::ConstantTimeEq;

use crate::config::Config;
use crate::error::{ApiError, Result};
use crate::imaging;

/// Max length of a form `source` field (Ultralytics API spec).
pub const MAX_SOURCE_LEN: usize = 4096;

/// `(file, source, known fields)` collected from a form body.
pub type Collected = (Option<Vec<u8>>, Option<String>, Vec<(String, String)>);

/// Require `Authorization: Bearer <key>` matching one of the endpoint's
/// `api_keys`.
pub fn authorize(headers: &HeaderMap, keys: &[String]) -> Result<()> {
    if bearer(headers).is_some_and(|t| key_matches(t, keys)) {
        Ok(())
    } else {
        Err(ApiError::Unauthorized)
    }
}

/// The token of an `Authorization: Bearer <token>` header.
pub fn bearer(headers: &HeaderMap) -> Option<&str> {
    let raw = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    raw.strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
}

/// Constant-time match against every key of the list (any order).
pub fn key_matches(token: &str, keys: &[String]) -> bool {
    let mut ok = 0u8;
    for key in keys {
        ok |= u8::from(bool::from(token.as_bytes().ct_eq(key.as_bytes())));
    }
    ok == 1
}

/// Collect `file`, `source` and the `known` fields of a multipart or
/// urlencoded form body (the Ultralytics endpoint, FastAPI `Form`, accepts
/// both). Unknown fields are ignored.
pub async fn collect_form(
    headers: &HeaderMap,
    req: Request,
    cfg: &Config,
    known: &[&str],
) -> Result<Collected> {
    let ctype = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if ctype.starts_with("multipart/form-data") {
        collect_multipart(req, cfg, known).await
    } else if ctype.starts_with("application/x-www-form-urlencoded") {
        collect_urlencoded(req, cfg, known).await
    } else {
        Err(ApiError::BadRequest(
            "expected a multipart/form-data or x-www-form-urlencoded body".into(),
        ))
    }
}

async fn collect_multipart(req: Request, cfg: &Config, known: &[&str]) -> Result<Collected> {
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
            n if known.contains(&n) => {
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

async fn collect_urlencoded(req: Request, cfg: &Config, known: &[&str]) -> Result<Collected> {
    let body = axum::body::to_bytes(req.into_body(), cfg.max_upload_bytes() + 16 * 1024 * 1024)
        .await
        .map_err(|_| ApiError::TooLarge("form body too large".into()))?;
    let file: Option<Vec<u8>> = None; // 'file' requires multipart (rejected below)
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
            n if known.contains(&n) => fields.push((key.into_owned(), value.into_owned())),
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

/// Decode image bytes and run `prepare` on the result, both on a blocking
/// worker (CPU-bound). Returns the image, `prepare`'s output, and its time.
pub async fn decode_then<T, F>(bytes: Vec<u8>, max_pixels: u64, prepare: F) -> Result<(T, f64)>
where
    T: Send + 'static,
    F: FnOnce(RgbImage) -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let image = imaging::decode(&bytes, max_pixels)?;
        let t0 = Instant::now();
        let out = prepare(image);
        Ok::<_, ApiError>((out, elapsed_ms(t0)))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("preprocess task failed: {e}")))?
}

pub fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}
