//! OCR, Google Cloud Vision-compatible: `POST /v1/images:annotate`.
//!
//! Supported feature types:
//! - `TEXT_DETECTION`, `DOCUMENT_TEXT_DETECTION`: full OCR, answered with
//!   `textAnnotations` + `fullTextAnnotation` exactly as Cloud Vision shapes
//!   them (one block/paragraph per detected line);
//! - `TEXT_REGION_DETECTION` (extension, not in Cloud Vision): text regions
//!   only, no recognition, answered with `textRegionAnnotations`.
//!
//! Auth accepts every Cloud Vision form: `?key=`, `x-goog-api-key`, or
//! `Authorization: Bearer`. Errors use the Google error envelope; per-image
//! failures go in `responses[i].error` with HTTP 200, as Cloud Vision does.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use backend_api::ocr::{Glyph, OcrParams, Quad, sub_quad};
use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::ocr::{self, Line, OcrOutput};
use crate::state::AppState;
use crate::{input, sources};

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/images:annotate", post(annotate))
}

/// An [`ApiError`] rendered as a Google API error envelope.
pub struct GoogleError(ApiError);

impl From<ApiError> for GoogleError {
    fn from(e: ApiError) -> Self {
        Self(e)
    }
}

impl IntoResponse for GoogleError {
    fn into_response(self) -> Response {
        let status = self.0.status();
        let body = json!({ "error": {
            "code": status.as_u16(),
            "message": self.0.to_string(),
            "status": rpc_status(&self.0).1,
        }});
        (status, Json(body)).into_response()
    }
}

/// `(google.rpc.Code, name)` for an error.
fn rpc_status(e: &ApiError) -> (i32, &'static str) {
    match e {
        ApiError::BadRequest(_) | ApiError::TooLarge(_) => (3, "INVALID_ARGUMENT"),
        ApiError::NotFound(_) => (5, "NOT_FOUND"),
        ApiError::Unauthorized => (16, "UNAUTHENTICATED"),
        ApiError::Upstream(_) => (14, "UNAVAILABLE"),
        ApiError::Internal(_) => (13, "INTERNAL"),
    }
}

// --- request -----------------------------------------------------------------

#[derive(Deserialize)]
struct BatchRequest {
    #[serde(default)]
    requests: Vec<ImageRequest>,
}

#[derive(Deserialize)]
struct ImageRequest {
    image: Option<ImageIn>,
    #[serde(default)]
    features: Vec<FeatureIn>,
    // `imageContext` (e.g. `languageHints`) is accepted and ignored: the
    // served model defines the languages.
}

#[derive(Deserialize)]
struct ImageIn {
    content: Option<String>,
    source: Option<ImageSourceIn>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImageSourceIn {
    image_uri: Option<String>,
    // `gcsImageUri` is not supported (no GCS access); such requests fail with
    // a clear error instead of being ignored.
}

#[derive(Deserialize)]
struct FeatureIn {
    /// Enum name, or its number (Google REST clients send
    /// `$alt=json;enum-encoding=int`).
    #[serde(rename = "type", default)]
    kind: Value,
}

const TEXT_DETECTION: &str = "TEXT_DETECTION";
const DOCUMENT_TEXT_DETECTION: &str = "DOCUMENT_TEXT_DETECTION";
const TEXT_REGION_DETECTION: &str = "TEXT_REGION_DETECTION";

/// What one image request asks for.
#[derive(Debug, Default, PartialEq)]
struct Wanted {
    text: bool,
    regions: bool,
}

fn wanted(features: &[FeatureIn]) -> Result<Wanted, String> {
    if features.is_empty() {
        return Err("at least one feature is required".into());
    }
    let mut w = Wanted::default();
    for f in features {
        match &f.kind {
            Value::String(s) if s == TEXT_DETECTION || s == DOCUMENT_TEXT_DETECTION => {
                w.text = true
            }
            Value::String(s) if s == TEXT_REGION_DETECTION => w.regions = true,
            // Feature.Type numbers of TEXT_DETECTION / DOCUMENT_TEXT_DETECTION.
            Value::Number(n) if matches!(n.as_i64(), Some(5 | 11)) => w.text = true,
            other => {
                return Err(format!(
                    "feature type {other} is not supported by this server (supported: \
                     {TEXT_DETECTION}, {DOCUMENT_TEXT_DETECTION}, {TEXT_REGION_DETECTION})"
                ));
            }
        }
    }
    Ok(w)
}

// --- response ----------------------------------------------------------------

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct ImageResponse {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    text_annotations: Vec<EntityAnnotation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_text_annotation: Option<TextAnnotation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_region_annotations: Option<Vec<RegionAnnotation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Status>,
}

#[derive(Serialize)]
struct Status {
    code: i32,
    message: String,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq)]
struct Vertex {
    x: i64,
    y: i64,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct BoundingPoly {
    vertices: Vec<Vertex>,
}

impl From<&Quad> for BoundingPoly {
    fn from(q: &Quad) -> Self {
        Self {
            vertices: q
                .iter()
                .map(|&(x, y)| Vertex {
                    x: x.round() as i64,
                    y: y.round() as i64,
                })
                .collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EntityAnnotation {
    description: String,
    bounding_poly: BoundingPoly,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RegionAnnotation {
    bounding_poly: BoundingPoly,
    score: f64,
}

#[derive(Serialize)]
struct TextAnnotation {
    pages: Vec<Page>,
    text: String,
}

#[derive(Serialize)]
struct Page {
    width: u32,
    height: u32,
    blocks: Vec<Block>,
    confidence: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Block {
    bounding_box: BoundingPoly,
    paragraphs: Vec<Paragraph>,
    block_type: &'static str,
    confidence: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Paragraph {
    bounding_box: BoundingPoly,
    words: Vec<Word>,
    confidence: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Word {
    bounding_box: BoundingPoly,
    symbols: Vec<Symbol>,
    confidence: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Symbol {
    #[serde(skip_serializing_if = "Option::is_none")]
    property: Option<Value>,
    bounding_box: BoundingPoly,
    text: String,
    confidence: f64,
}

// --- handler -----------------------------------------------------------------

async fn annotate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<Value>, GoogleError> {
    let cfg = state.cfg.endpoints.ocr.as_ref().expect("ocr config");
    authorize(&headers, req.uri().query(), &cfg.api_keys)?;
    let slot = state
        .ocr
        .as_ref()
        .expect("ocr route mounted without its slot");
    let backend = slot.get()?.clone();

    // base64 inflates by 4/3; the JSON envelope gets 1 MiB on top.
    let limit = state.cfg.max_upload_bytes() / 3 * 4 + (1 << 20);
    let body = axum::body::to_bytes(req.into_body(), limit)
        .await
        .map_err(|_| {
            ApiError::TooLarge(format!(
                "request too large (max {} MB of image data)",
                state.cfg.limits.max_upload_mb
            ))
        })?;
    let batch: BatchRequest = serde_json::from_slice(&body)
        .map_err(|e| ApiError::BadRequest(format!("invalid JSON request: {e}")))?;
    if batch.requests.is_empty() {
        return Err(ApiError::BadRequest("'requests' must not be empty".into()).into());
    }
    if batch.requests.len() > cfg.max_images_per_request {
        return Err(ApiError::BadRequest(format!(
            "at most {} images per request",
            cfg.max_images_per_request
        ))
        .into());
    }

    let params = OcrParams {
        min_region_score: cfg.defaults.min_region_score,
        max_regions: cfg.defaults.max_regions,
    };
    let min_text = cfg.defaults.min_text_confidence;
    let responses = futures_util::future::join_all(batch.requests.into_iter().map(|r| {
        let (state, backend) = (&state, backend.clone());
        async move {
            match annotate_one(state, backend, r, params, min_text).await {
                Ok(resp) => resp,
                Err(e) => ImageResponse {
                    error: Some(Status {
                        code: rpc_status(&e).0,
                        message: e.to_string(),
                    }),
                    ..Default::default()
                },
            }
        }
    }))
    .await;
    Ok(Json(json!({ "responses": responses })))
}

fn authorize(headers: &HeaderMap, query: Option<&str>, keys: &[String]) -> Result<(), ApiError> {
    let query_key = query.and_then(|q| {
        form_urlencoded::parse(q.as_bytes())
            .find(|(k, _)| k == "key")
            .map(|(_, v)| v.into_owned())
    });
    let header_key = headers.get("x-goog-api-key").and_then(|v| v.to_str().ok());
    let token = query_key
        .as_deref()
        .or(header_key)
        .or_else(|| input::bearer(headers));
    if token.is_some_and(|t| input::key_matches(t, keys)) {
        Ok(())
    } else {
        Err(ApiError::Unauthorized)
    }
}

async fn annotate_one(
    state: &AppState,
    backend: Arc<dyn backend_api::ocr::OcrBackend>,
    req: ImageRequest,
    params: OcrParams,
    min_text: f64,
) -> Result<ImageResponse, ApiError> {
    let want = wanted(&req.features).map_err(ApiError::BadRequest)?;
    let bytes = image_bytes(state, req.image).await?;
    let out = ocr::run(state, backend, bytes, params, min_text, want.text).await?;
    Ok(to_response(&out, &want))
}

async fn image_bytes(state: &AppState, image: Option<ImageIn>) -> Result<Vec<u8>, ApiError> {
    let image = image.ok_or_else(|| ApiError::BadRequest("'image' is required".into()))?;
    let bytes = match (image.content, image.source) {
        (Some(content), _) => {
            let content = content.trim();
            STANDARD
                .decode(content)
                .or_else(|_| URL_SAFE.decode(content))
                .map_err(|_| ApiError::BadRequest("image.content is not valid base64".into()))?
        }
        (
            None,
            Some(ImageSourceIn {
                image_uri: Some(uri),
                ..
            }),
        ) if uri.starts_with("http://") || uri.starts_with("https://") => {
            sources::resolve_source(&state.fetch, &uri, &state.cfg.limits).await?
        }
        (None, Some(_)) => {
            return Err(ApiError::BadRequest(
                "only http(s) image.source.imageUri is supported (no gs:// URIs)".into(),
            ));
        }
        (None, None) => {
            return Err(ApiError::BadRequest(
                "image.content or image.source is required".into(),
            ));
        }
    };
    if bytes.is_empty() {
        return Err(ApiError::BadRequest("image is empty".into()));
    }
    if bytes.len() > state.cfg.max_upload_bytes() {
        return Err(ApiError::TooLarge(format!(
            "image too large (max {} MB)",
            state.cfg.limits.max_upload_mb
        )));
    }
    Ok(bytes)
}

fn to_response(out: &OcrOutput, want: &Wanted) -> ImageResponse {
    let mut resp = ImageResponse::default();
    if want.regions {
        resp.text_region_annotations = Some(
            out.regions
                .iter()
                .map(|r| RegionAnnotation {
                    bounding_poly: (&r.quad).into(),
                    score: round(r.score),
                })
                .collect(),
        );
    }
    if let Some(lines) = out.lines.as_ref().filter(|l| want.text && !l.is_empty()) {
        let (annotations, full) = text_annotations(lines, out.width, out.height);
        resp.text_annotations = annotations;
        resp.full_text_annotation = Some(full);
    }
    resp
}

/// Words of a line: runs of non-whitespace glyphs.
fn words(glyphs: &[Glyph]) -> Vec<&[Glyph]> {
    glyphs
        .split(|g| g.text.trim().is_empty())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Even per-character spans for a line whose backend gave no glyphs.
fn even_glyphs(line: &Line) -> Vec<Glyph> {
    let chars: Vec<char> = line.text.text.chars().collect();
    let n = chars.len().max(1) as f64;
    chars
        .iter()
        .enumerate()
        .map(|(i, c)| Glyph {
            text: c.to_string(),
            confidence: line.text.confidence,
            span: (i as f64 / n, (i + 1) as f64 / n),
        })
        .collect()
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = xs.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { 0.0 } else { round(sum / n as f64) }
}

fn round(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// Cloud Vision `textAnnotations` (full text first, then one per word) and
/// `fullTextAnnotation` (one block + paragraph per line).
fn text_annotations(
    lines: &[Line],
    width: u32,
    height: u32,
) -> (Vec<EntityAnnotation>, TextAnnotation) {
    let mut text = String::new();
    let mut word_annotations = Vec::new();
    let mut blocks = Vec::new();
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);

    for line in lines {
        for &(x, y) in &line.quad {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        let owned;
        let glyphs = if line.text.glyphs.is_empty() {
            owned = even_glyphs(line);
            &owned
        } else {
            &line.text.glyphs
        };
        let spans = words(glyphs);
        let mut line_words = Vec::new();
        for (wi, w) in spans.iter().enumerate() {
            let (first, last) = (&w[0], &w[w.len() - 1]);
            let quad = sub_quad(&line.quad, first.span.0, last.span.1);
            let word_text: String = w.iter().map(|g| g.text.as_str()).collect();
            let last_word = wi + 1 == spans.len();
            let symbols = w
                .iter()
                .enumerate()
                .map(|(gi, g)| {
                    let property = (gi + 1 == w.len()).then(|| {
                        let kind = if last_word { "LINE_BREAK" } else { "SPACE" };
                        json!({ "detectedBreak": { "type": kind } })
                    });
                    Symbol {
                        property,
                        bounding_box: (&sub_quad(&line.quad, g.span.0, g.span.1)).into(),
                        text: g.text.clone(),
                        confidence: round(g.confidence),
                    }
                })
                .collect();
            word_annotations.push(EntityAnnotation {
                description: word_text,
                bounding_poly: (&quad).into(),
            });
            line_words.push(Word {
                bounding_box: (&quad).into(),
                symbols,
                confidence: mean(w.iter().map(|g| g.confidence)),
            });
        }
        let line_text: Vec<String> = spans
            .iter()
            .map(|w| w.iter().map(|g| g.text.as_str()).collect())
            .collect();
        text.push_str(&line_text.join(" "));
        text.push('\n');
        let confidence = round(line.text.confidence);
        blocks.push(Block {
            bounding_box: (&line.quad).into(),
            paragraphs: vec![Paragraph {
                bounding_box: (&line.quad).into(),
                words: line_words,
                confidence,
            }],
            block_type: "TEXT",
            confidence,
        });
    }

    let page_confidence = mean(lines.iter().map(|l| l.text.confidence));
    let mut annotations = vec![EntityAnnotation {
        description: text.clone(),
        bounding_poly: (&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).into(),
    }];
    annotations.extend(word_annotations);
    let full = TextAnnotation {
        pages: vec![Page {
            width,
            height,
            blocks,
            confidence: page_confidence,
        }],
        text,
    };
    (annotations, full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend_api::ocr::TextLine;

    fn feature(v: Value) -> FeatureIn {
        FeatureIn { kind: v }
    }

    #[test]
    fn features_accept_names_numbers_and_the_region_extension() {
        assert_eq!(
            wanted(&[feature(json!("DOCUMENT_TEXT_DETECTION"))]).unwrap(),
            Wanted {
                text: true,
                regions: false
            }
        );
        assert_eq!(
            wanted(&[feature(json!(5)), feature(json!("TEXT_REGION_DETECTION"))]).unwrap(),
            Wanted {
                text: true,
                regions: true
            }
        );
        assert!(wanted(&[feature(json!("LABEL_DETECTION"))]).is_err());
        assert!(wanted(&[]).is_err());
    }

    fn glyph(c: &str, a: f64, b: f64) -> Glyph {
        Glyph {
            text: c.into(),
            confidence: 0.9,
            span: (a, b),
        }
    }

    #[test]
    fn lines_become_words_symbols_and_full_text() {
        let line = Line {
            quad: [(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)],
            text: TextLine {
                text: "AB C".into(),
                confidence: 0.9,
                glyphs: vec![
                    glyph("A", 0.0, 0.2),
                    glyph("B", 0.2, 0.4),
                    glyph(" ", 0.4, 0.6),
                    glyph("C", 0.6, 1.0),
                ],
            },
        };
        let (ann, full) = text_annotations(&[line], 200, 50);
        assert_eq!(ann[0].description, "AB C\n");
        assert_eq!(ann[1].description, "AB");
        assert_eq!(ann[1].bounding_poly.vertices[1], Vertex { x: 40, y: 0 });
        assert_eq!(ann[2].description, "C");
        assert_eq!(full.text, "AB C\n");
        let words = &full.pages[0].blocks[0].paragraphs[0].words;
        assert_eq!(words.len(), 2);
        assert_eq!(
            words[0].symbols[1].property,
            Some(json!({"detectedBreak": {"type": "SPACE"}}))
        );
        assert_eq!(
            words[1].symbols[0].property,
            Some(json!({"detectedBreak": {"type": "LINE_BREAK"}}))
        );
    }
}
