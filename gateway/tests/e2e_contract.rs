//! End-to-end contract tests: real gateway router + protocol-accurate mock
//! Triton (binary tensor extension per docs/protocol/extension_binary_data.md).
//! Proves the gateway emits an identical Ultralytics `PredictResponse` shape
//! for both YOLO (end-to-end) and RF-DETR backends, and the Cloud Vision
//! `images:annotate` shape for the PP-OCR backend.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

struct Seen {
    inputs: Vec<Value>,
    /// Model name of every inference request, in order.
    models: Vec<String>,
}

#[derive(Clone)]
struct Mock {
    seen: Arc<Mutex<Seen>>,
}

// --- fixture tensors -------------------------------------------------------

fn yolo_config() -> Value {
    let names: Vec<String> = (0..80).map(|i| format!("c{i}")).collect();
    let meta = json!({
        "family": "yolo", "task": "detect", "names": names, "imgsz": 640,
        "stride": 32, "end2end": true, "input": "images", "outputs": ["output0"],
        "queries": 300
    });
    json!({
        "name": "yolo26n", "backend": "onnxruntime", "max_batch_size": 8,
        "parameters": { "metadata": { "string_value": meta.to_string() } }
    })
}

fn rfdetr_config() -> Value {
    // sparse COCO: 91 slots, id == slot, no background slot
    let mut names: Vec<String> = (0..91).map(|i| format!("class_{i}")).collect();
    names[17] = "cat".into();
    names[16] = "bird".into();
    let meta = json!({
        "family": "rfdetr", "task": "detect", "names": names, "imgsz": 560,
        "bg_slot": null, "input": "input", "outputs": ["dets", "labels"],
        "queries": 300
    });
    json!({
        "name": "rfdetr-base", "backend": "onnxruntime", "max_batch_size": 8,
        "parameters": { "metadata": { "string_value": meta.to_string() } }
    })
}

const OCR_CHARSET: [&str; 6] = ["H", "I", " ", "Y", "O", "U"];

fn ppocr_det_config() -> Value {
    let meta = json!({
        "family": "ppocr", "task": "ocr", "stage": "det", "input": "x",
        "outputs": ["sigmoid_0.tmp_0"], "companions": ["ppocr-rec"]
    });
    json!({
        "name": "ppocr-det", "backend": "onnxruntime", "max_batch_size": 1,
        "parameters": { "metadata": { "string_value": meta.to_string() } }
    })
}

fn ppocr_rec_config() -> Value {
    let meta = json!({
        "family": "ppocr", "task": "ocr", "stage": "rec", "input": "x",
        "outputs": ["softmax_0.tmp_0"], "imgh": 48, "imgw": 320, "charset": OCR_CHARSET
    });
    json!({
        "name": "ppocr-rec", "backend": "onnxruntime", "max_batch_size": 8,
        "parameters": { "metadata": { "string_value": meta.to_string() } }
    })
}

/// Text "probability" map: 0.9 where the input's first channel is bright.
fn ppocr_det_tensor(input: &[f32], shape: &[u64]) -> (Vec<f32>, Vec<u64>) {
    let (h, w) = (shape[2], shape[3]);
    let plane = (h * w) as usize;
    let map = input[..plane]
        .iter()
        .map(|&v| if v > 0.0 { 0.9 } else { 0.02 })
        .collect();
    (map, vec![1, 1, h, w])
}

/// Every row decodes to "HI YOU": one char every 3 steps, blanks between.
fn ppocr_rec_tensor(shape: &[u64]) -> (Vec<f32>, Vec<u64>) {
    let (n, t, c) = (
        shape[0] as usize,
        (shape[3] / 8) as usize,
        OCR_CHARSET.len() + 1,
    );
    let mut data = vec![0f32; n * t * c];
    for row in 0..n {
        for step in 0..t {
            let k = match step {
                s if s % 3 == 1 && s / 3 < 6 => s / 3 + 1,
                _ => 0,
            };
            data[(row * t + step) * c + k] = 0.95;
        }
    }
    (data, vec![n as u64, t as u64, c as u64])
}

/// output0 (300,6): two valid rows, rest padding (cls -1, conf 0).
fn yolo_fixture_tensor() -> Vec<f32> {
    let mut v = vec![0f32; 300 * 6];
    for i in 0..300 {
        v[i * 6 + 5] = -1.0;
    }
    // row 0: person-ish full frame, conf .87 class 16 (bird in fixture names)
    let r0 = [100.0, 140.0, 320.0, 460.0, 0.87, 16.0];
    v[0..6].copy_from_slice(&r0);
    // row 1: conf .51 class 0, coords overflowing the frame (clamp check)
    let r1 = [-5.0, -5.0, 900.0, 900.0, 0.51, 0.0];
    v[6..12].copy_from_slice(&r1);
    // row 2: below conf threshold
    let r2 = [10.0, 10.0, 20.0, 20.0, 0.10, 2.0];
    v[12..18].copy_from_slice(&r2);
    v
}

/// dets (300,4) cxcywh normalized + labels (300,91) logits. query 0 -> cat.
fn rfdetr_fixture_tensors() -> (Vec<f32>, Vec<f32>) {
    let mut dets = vec![0f32; 300 * 4];
    dets[0..4].copy_from_slice(&[0.5, 0.5, 0.25, 0.125]); // -> (160,180,480,300) on 640x480
    let mut labels = vec![-20f32; 300 * 91];
    labels[17] = 8.0; // query 0, slot 17 ("cat"): sigmoid ~0.99966
    (dets, labels)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// --- mock triton -----------------------------------------------------------

async fn mock_config(
    State(m): State<Mock>,
    axum::extract::Path(model): axum::extract::Path<String>,
) -> Json<Value> {
    let _ = &m;
    Json(match model.as_str() {
        "yolo26n" => yolo_config(),
        "ppocr-det" => ppocr_det_config(),
        "ppocr-rec" => ppocr_rec_config(),
        _ => rfdetr_config(),
    })
}

async fn mock_metadata() -> Json<Value> {
    Json(json!({"name": "triton", "version": "2.99-mock", "extensions": ["binary_tensor_data"]}))
}

async fn mock_image() -> Response {
    // 640x480 JPEG fixture used as a `source` URL body
    let bytes = std::fs::read(test_image_path()).expect("test image");
    ([(axum::http::header::CONTENT_TYPE, "image/jpeg")], bytes).into_response()
}

async fn mock_infer(
    State(m): State<Mock>,
    headers: HeaderMap,
    axum::extract::Path(model): axum::extract::Path<String>,
    body: Bytes,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let hcl = headers
        .get("Inference-Header-Content-Length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "missing Inference-Header-Content-Length"})),
        ))?;
    if hcl % 4 != 0 || hcl > body.len() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "bad header length"})),
        ));
    }
    let header: Value = serde_json::from_slice(&body[..hcl]).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("bad json header: {e}")})),
        )
    })?;
    let input = header
        .get("inputs")
        .and_then(|i| i.get(0))
        .ok_or((StatusCode::BAD_REQUEST, Json(json!({"error": "no inputs"}))))?
        .clone();
    let bds = input
        .pointer("/parameters/binary_data_size")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if bds != body[hcl..].len() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("binary_data_size {bds} != trailing bytes {}", body[hcl..].len())
            })),
        ));
    }
    {
        let mut seen = m.seen.lock().unwrap();
        seen.inputs.push(input.clone());
        seen.models.push(model.clone());
    }
    let input_shape: Vec<u64> = input["shape"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    let input_data: Vec<f32> = body[hcl..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let want_outputs: Vec<String> = header
        .get("outputs")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|o| o.get("name").and_then(Value::as_str).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let mut outs_json = Vec::new();
    let mut binary: Vec<u8> = Vec::new();
    let tensors: Vec<(String, Vec<f32>, Vec<u64>)> = match model.as_str() {
        "yolo26n" => vec![(
            "output0".to_string(),
            yolo_fixture_tensor(),
            vec![1, 300, 6],
        )],
        "ppocr-det" => {
            let (d, shape) = ppocr_det_tensor(&input_data, &input_shape);
            vec![("sigmoid_0.tmp_0".to_string(), d, shape)]
        }
        "ppocr-rec" => {
            let (d, shape) = ppocr_rec_tensor(&input_shape);
            vec![("softmax_0.tmp_0".to_string(), d, shape)]
        }
        _ => {
            let (d, l) = rfdetr_fixture_tensors();
            vec![
                ("dets".to_string(), d, vec![1, 300, 4]),
                ("labels".to_string(), l, vec![1, 300, 91]),
            ]
        }
    };
    for (name, data, shape) in tensors {
        if !want_outputs.is_empty() && !want_outputs.contains(&name) {
            continue;
        }
        let bytes = f32_bytes(&data);
        outs_json.push(json!({
            "name": name, "shape": shape, "datatype": "FP32",
            "parameters": { "binary_data_size": bytes.len() }
        }));
        binary.extend_from_slice(&bytes);
    }

    let mut json_bytes =
        serde_json::to_vec(&json!({"model_name": model, "outputs": outs_json})).unwrap();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let hcl_out = json_bytes.len();
    let mut body = json_bytes;
    body.extend_from_slice(&binary);
    let mut hdrs = HeaderMap::new();
    hdrs.insert(
        axum::http::header::CONTENT_TYPE,
        "application/octet-stream".parse().unwrap(),
    );
    hdrs.insert(
        axum::http::HeaderName::from_lowercase(b"inference-header-content-length").unwrap(),
        hcl_out.to_string().parse().unwrap(),
    );
    Ok((hdrs, body).into_response())
}

fn test_image_path() -> PathBuf {
    std::env::temp_dir().join(format!("gateway-test-image-{}.jpg", std::process::id()))
}

// --- harness ---------------------------------------------------------------

async fn spawn_mock() -> (SocketAddr, Mock) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    spawn_mock_on(listener)
}

fn spawn_mock_on(listener: tokio::net::TcpListener) -> (SocketAddr, Mock) {
    let m = Mock {
        seen: Arc::new(Mutex::new(Seen {
            inputs: vec![],
            models: vec![],
        })),
    };
    let app = Router::new()
        .route("/v2", get(mock_metadata))
        .route("/v2/models/{model}/config", get(mock_config))
        .route("/v2/models/{model}/infer", post(mock_infer))
        .route("/image", get(mock_image))
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(m.clone());
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (addr, m)
}

static CFG_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

async fn spawn_gateway(triton_addr: SocketAddr, model: &str) -> SocketAddr {
    spawn_gateway_with(triton_addr, model, true).await
}

/// Gateway with URL sources disabled (the production default).
async fn spawn_gateway_no_urls(triton_addr: SocketAddr, model: &str) -> SocketAddr {
    let (addr, state) = start_gateway_cfg(triton_addr, model, true, false).await;
    wait_ready(&state).await;
    addr
}

async fn spawn_gateway_with(
    triton_addr: SocketAddr,
    model: &str,
    allow_private: bool,
) -> SocketAddr {
    let (addr, state) = start_gateway(triton_addr, model, allow_private).await;
    wait_ready(&state).await;
    addr
}

/// Start a gateway (URL sources enabled) without waiting for its model to load.
async fn start_gateway(
    triton_addr: SocketAddr,
    model: &str,
    allow_private: bool,
) -> (SocketAddr, Arc<gateway::state::AppState>) {
    start_gateway_cfg(triton_addr, model, allow_private, true).await
}

async fn start_gateway_cfg(
    triton_addr: SocketAddr,
    model: &str,
    allow_private: bool,
    allow_url: bool,
) -> (SocketAddr, Arc<gateway::state::AppState>) {
    let cfg_text = format!(
        r#"
server: {{ listen: "127.0.0.1:0" }}
triton:
  url: "http://{triton_addr}"
  timeout_ms: 5000
endpoints:
  detect:
    model: "{model}"
    api_keys: ["test-key-1", "test-key-2"]
    defaults: {{ conf: 0.25, iou: 0.7 }}
limits:
  max_upload_mb: 4
  url_timeout_ms: 3000
  allow_url_sources: {allow_url}
  allow_private_urls: {allow_private}
"#
    );
    let n = CFG_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!(
        "gateway-test-config-{}-{model}-{n}.yaml",
        std::process::id()
    ));
    std::fs::write(&path, &cfg_text).unwrap();
    let cfg = Arc::new(gateway::config::Config::load(&path).unwrap());
    let triton = Arc::new(gateway::triton::TritonClient::new(
        &cfg.triton.url,
        cfg.triton.api_key.as_deref(),
        std::time::Duration::from_millis(cfg.triton.timeout_ms),
    ));
    let fetch = gateway::sources::fetch_client(&cfg.limits).unwrap();
    // Same startup path as the binary: serve now, load the model in background.
    let state = Arc::new(gateway::state::AppState::new(cfg.clone(), triton, fetch));
    gateway::state::spawn_model_loaders(state.clone());
    let app = gateway::endpoints::router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (addr, state)
}

/// Start a gateway serving the given `endpoints:` YAML block (indented by
/// two spaces) and wait for its models.
async fn spawn_gateway_endpoints(
    triton_addr: SocketAddr,
    endpoints: &str,
) -> (SocketAddr, Arc<gateway::state::AppState>) {
    let (addr, state) = start_gateway_endpoints(triton_addr, endpoints).await;
    wait_ready(&state).await;
    (addr, state)
}

async fn start_gateway_endpoints(
    triton_addr: SocketAddr,
    endpoints: &str,
) -> (SocketAddr, Arc<gateway::state::AppState>) {
    let cfg_text = format!(
        r#"
server: {{ listen: "127.0.0.1:0" }}
triton: {{ url: "http://{triton_addr}", timeout_ms: 5000 }}
limits: {{ max_upload_mb: 4, allow_url_sources: true, allow_private_urls: true }}
endpoints:
{endpoints}
"#
    );
    let n = CFG_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!(
        "gateway-test-config-{}-endpoints-{n}.yaml",
        std::process::id()
    ));
    std::fs::write(&path, &cfg_text).unwrap();
    let cfg = Arc::new(gateway::config::Config::load(&path).unwrap());
    let triton = Arc::new(gateway::triton::TritonClient::new(
        &cfg.triton.url,
        None,
        std::time::Duration::from_millis(cfg.triton.timeout_ms),
    ));
    let fetch = gateway::sources::fetch_client(&cfg.limits).unwrap();
    let state = Arc::new(gateway::state::AppState::new(cfg.clone(), triton, fetch));
    gateway::state::spawn_model_loaders(state.clone());
    let app = gateway::endpoints::router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (addr, state)
}

async fn wait_ready(state: &gateway::state::AppState) {
    for _ in 0..200 {
        if state.is_ready() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("model never became ready: {:?}", state.load_errors());
}

fn make_test_jpeg() -> Vec<u8> {
    use image::{Rgb, RgbImage};
    // 640x480 gradient — deterministic content
    let mut img = RgbImage::new(640, 480);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
    }
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

fn make_small_jpeg() -> Vec<u8> {
    use image::{Rgb, RgbImage};
    // 64x64 flat image -> a few hundred bytes, base64 fits the 4096-char cap.
    let img = RgbImage::from_fn(64, 64, |x, y| Rgb([x as u8, y as u8, 90]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

async fn post_predict(
    addr: SocketAddr,
    key: &str,
    multipart: reqwest::multipart::Form,
) -> (StatusCode, Value) {
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/predict"))
        .header("Authorization", format!("Bearer {key}"))
        .multipart(multipart)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    (status, resp.json::<Value>().await.unwrap_or(Value::Null))
}

fn mp_file(jpeg: &[u8]) -> reqwest::multipart::Form {
    reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(jpeg.to_vec())
            .file_name("image.jpg")
            .mime_str("image/jpeg")
            .unwrap(),
    )
}

/// Shared structural assertions for the OpenAPI PredictResponse contract.
fn assert_openapi_contract(v: &Value) {
    assert!(
        v.get("images").and_then(Value::as_array).is_some(),
        "images[]"
    );
    assert!(v.get("metadata").is_some(), "metadata");
    let img = &v["images"][0];
    assert_eq!(img["shape"].as_array().unwrap().len(), 2);
    for f in ["preprocess", "inference", "postprocess"] {
        assert!(img["speed"][f].is_number(), "speed.{f}");
    }
    let meta = &v["metadata"];
    assert!(meta["imageCount"].is_number());
    assert!(meta["functionTimeAlive"].is_number());
    assert!(meta["functionTimeCall"].is_number());
    assert!(meta["version"].is_object());
    assert_eq!(meta["task"], "detect");
    for r in img["results"].as_array().unwrap() {
        assert!(r["name"].is_string());
        assert!(r["class"].is_number());
        assert!(r["confidence"].is_number());
        for c in ["x1", "y1", "x2", "y2"] {
            assert!(r["box"][c].is_number(), "box.{c}");
        }
        // only detect fields: no segmentation keys
        assert!(r.get("segments").is_none());
    }
}

// --- tests -----------------------------------------------------------------

#[tokio::test]
async fn yolo_and_rfdetr_emit_identical_contract() {
    let jpeg = make_test_jpeg();
    std::fs::write(test_image_path(), &jpeg).unwrap();

    let (mock_addr, _mock) = spawn_mock().await;
    let yolo_addr = spawn_gateway(mock_addr, "yolo26n").await;
    let (mock_addr2, _mock2) = spawn_mock().await;
    let rfdetr_addr = spawn_gateway(mock_addr2, "rfdetr-base").await;

    let (status, yolo) = post_predict(yolo_addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::OK, "yolo predict: {yolo}");
    let (status, rfdetr) = post_predict(rfdetr_addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::OK, "rfdetr predict: {rfdetr}");

    // 1. Same structure for both families.
    assert_openapi_contract(&yolo);
    assert_openapi_contract(&rfdetr);
    let top_yolo: Vec<&String> = yolo.as_object().unwrap().keys().collect();
    let top_rf: Vec<&String> = rfdetr.as_object().unwrap().keys().collect();
    assert_eq!(top_yolo, top_rf, "top-level keys identical");
    let img_keys: Vec<&String> = yolo["images"][0].as_object().unwrap().keys().collect();
    let img_keys2: Vec<&String> = rfdetr["images"][0].as_object().unwrap().keys().collect();
    assert_eq!(img_keys, img_keys2, "image result keys identical");

    // 2. YOLO e2e decode + inverse letterbox: 640x480 -> r=1, dw=0, dh=80.
    assert_eq!(yolo["images"][0]["shape"], json!([480, 640]));
    let rs = yolo["images"][0]["results"].as_array().unwrap();
    assert_eq!(rs.len(), 2, "conf filter drops row2 + padding rows: {rs:?}");
    assert_eq!(rs[0]["name"], "c16");
    assert_eq!(rs[0]["class"], 16);
    assert!((rs[0]["confidence"].as_f64().unwrap() - 0.87).abs() < 1e-9);
    // (100-0)/1, (140-80)/1, (320)/1, (460-80)/1
    assert_eq!(
        rs[0]["box"],
        json!({"x1": 100.0, "y1": 60.0, "x2": 320.0, "y2": 380.0})
    );
    // clamped to [0, 640] x [0, 480]
    assert_eq!(
        rs[1]["box"],
        json!({"x1": 0.0, "y1": 0.0, "x2": 640.0, "y2": 480.0})
    );
    assert_eq!(yolo["metadata"]["classNames"].as_array().unwrap().len(), 80);
    assert_eq!(yolo["metadata"]["model"], "yolo26n");

    // 3. RF-DETR decode: cat at query0 -> normalized cxcywh to pixels.
    let rs = rfdetr["images"][0]["results"].as_array().unwrap();
    assert_eq!(rs.len(), 1, "{rs:?}");
    assert_eq!(rs[0]["name"], "cat");
    assert_eq!(rs[0]["class"], 17);
    // (0.5±0.125)*640, (0.5±0.0625)*480
    assert_eq!(
        rs[0]["box"],
        json!({"x1": 240.0, "y1": 210.0, "x2": 400.0, "y2": 270.0})
    );
    assert_eq!(
        rfdetr["metadata"]["classNames"].as_array().unwrap().len(),
        91
    );

    // 4. Both ran at native resolution (option a): YOLO 640x640, RF-DETR 560x560.
    //    (mock validated shapes; no error returned => pass)
}

#[tokio::test]
async fn triton_binary_framing_is_valid() {
    let jpeg = make_test_jpeg();
    let (mock_addr, mock) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    let (status, v) = post_predict(addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let seen = mock.seen.lock().unwrap();
    let input = &seen.inputs[0];
    assert_eq!(input["name"], "images");
    assert_eq!(input["datatype"], "FP32");
    assert_eq!(input["shape"], json!([1, 3, 640, 640]));
    assert_eq!(
        input["parameters"]["binary_data_size"].as_u64().unwrap(),
        (3 * 640 * 640 * 4) as u64
    );
    // The mock would have 400'd on malformed framing; the client surfaced it as
    // 502/503 instead — reaching here with OK proves the wire is correct.
}

#[tokio::test]
async fn imgsz_accepted_but_native_resolution_used() {
    let jpeg = make_test_jpeg();
    let (mock_addr, mock) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "rfdetr-base").await;
    let mut mp = mp_file(&jpeg);
    mp = mp
        .text("conf", "0.25")
        .text("iou", "0.7")
        .text("imgsz", "1280");
    let (status, _v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(status, StatusCode::OK);
    let seen = mock.seen.lock().unwrap();
    // native 560 despite imgsz=1280 (drop-in swap rule (a))
    assert_eq!(seen.inputs[0]["shape"], json!([1, 3, 560, 560]));
}

#[tokio::test]
async fn imgsz_out_of_range_is_400() {
    let jpeg = make_test_jpeg();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    let mp = mp_file(&jpeg).text("imgsz", "20");
    let (status, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("imgsz"));
}

#[tokio::test]
async fn auth_flat_key_list() {
    let jpeg = make_test_jpeg();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    // missing header
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/predict"))
        .multipart(mp_file(&jpeg))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v, json!({"error": "Missing or invalid API key"}));
    // wrong key
    let (status, _) = post_predict(addr, "nope", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // second key in the flat list works
    let (status, _) = post_predict(addr, "test-key-2", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn base64_source_matches_file_response() {
    let jpeg = make_small_jpeg();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;

    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
    let mp =
        reqwest::multipart::Form::new().text("source", format!("data:image/jpeg;base64,{b64}"));
    let (s1, r1) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(s1, StatusCode::OK, "{r1}");

    let (s2, r2) = post_predict(addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(r1["images"][0]["results"], r2["images"][0]["results"]);
    assert_eq!(r1["images"][0]["shape"], r2["images"][0]["shape"]);
}

#[tokio::test]
async fn url_source_fetch() {
    let jpeg = make_test_jpeg();
    std::fs::write(test_image_path(), &jpeg).unwrap();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    let mp = reqwest::multipart::Form::new().text("source", format!("http://{mock_addr}/image"));
    let (status, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["images"][0]["shape"], json!([480, 640]));
}

#[tokio::test]
async fn file_source_exclusivity_and_errors() {
    let jpeg = make_test_jpeg();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;

    let mp = mp_file(&jpeg).text("source", "whatever");
    let (s, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("not both"));

    let mp = reqwest::multipart::Form::new().text("conf", "0.5");
    let (s, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("required"));

    // conf out of range
    let mp = mp_file(&jpeg).text("conf", "1.5");
    let (s, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("conf"));

    // non-image file
    let mp = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(b"hello".to_vec())
            .file_name("a.txt")
            .mime_str("text/plain")
            .unwrap(),
    );
    let (s, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"].as_str().unwrap().contains("image"));

    // too large
    let big = vec![0u8; 5 * 1024 * 1024];
    let mp = mp_file(&big);
    let (s, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
}

#[tokio::test]
async fn normalize_and_decimals() {
    let jpeg = make_test_jpeg();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    let mp = mp_file(&jpeg)
        .text("normalize", "true")
        .text("decimals", "2");
    let (status, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(status, StatusCode::OK);
    let b = &v["images"][0]["results"][0]["box"];
    assert_eq!(b["x1"], 100.0 / 640.0); // normalized coords use 5 decimals
    let expected = ((100.0f64 / 640.0) * 1e5).round() / 1e5;
    assert_eq!(b["x1"].as_f64().unwrap(), expected);
}

#[tokio::test]
async fn health_endpoint() {
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["status"], "healthy");
}

#[tokio::test]
async fn urlencoded_source_accepted_like_fastapi() {
    let jpeg = make_small_jpeg();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;

    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
    // reqwest .form() => application/x-www-form-urlencoded (what `requests`
    // sends for data={"source": ...} without files=). Percent-encoding must
    // round-trip base64 '+' and '/' correctly.
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/predict"))
        .header("Authorization", "Bearer test-key-1")
        .form(&[
            ("source", b64.as_str()),
            ("conf", "0.25"),
            ("normalize", "true"),
        ])
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let v: Value = resp.json().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_openapi_contract(&v);
    assert!(v["images"][0]["results"][0]["box"]["x1"].as_f64().unwrap() <= 1.0);

    // oversized source -> 400 (spec maxLength 4096)
    let big = "A".repeat(5000);
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/predict"))
        .header("Authorization", "Bearer test-key-1")
        .form(&[("source", big.as_str())])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(
        resp.json::<Value>().await.unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("source too long")
    );
}

#[tokio::test]
async fn urlencoded_file_and_bad_content_type_rejected() {
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/predict"))
        .header("Authorization", "Bearer test-key-1")
        .form(&[("file", "not-a-real-file")])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(
        resp.json::<Value>().await.unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("multipart")
    );

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/predict"))
        .header("Authorization", "Bearer test-key-1")
        .json(&serde_json::json!({"source": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn small_images_are_upscaled_without_panicking() {
    // Regression: stride-rounded letterbox upscaling overflowed the canvas.
    use image::{Rgb, RgbImage};
    let (mock_addr, mock) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    for (w, h) in [(300, 300), (1, 1), (33, 517)] {
        let img = RgbImage::from_pixel(w, h, Rgb([10, 20, 30]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let (status, v) = post_predict(addr, "test-key-1", mp_file(buf.get_ref())).await;
        assert_eq!(status, StatusCode::OK, "{w}x{h}: {v}");
        assert_eq!(v["images"][0]["shape"], json!([h, w]));
    }
    assert_eq!(
        mock.seen.lock().unwrap().inputs[0]["shape"],
        json!([1, 3, 640, 640])
    );
}

#[tokio::test]
async fn private_url_sources_are_blocked() {
    let jpeg = make_test_jpeg();
    std::fs::write(test_image_path(), &jpeg).unwrap();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway_with(mock_addr, "yolo26n", false).await;
    let port = mock_addr.port();
    for url in [
        format!("http://127.0.0.1:{port}/image"), // IP literal
        format!("http://localhost:{port}/image"), // name -> loopback (resolver)
        format!("http://[::ffff:127.0.0.1]:{port}/image"), // mapped IPv6
        "http://169.254.169.254/latest/meta-data/".to_string(),
    ] {
        let mp = reqwest::multipart::Form::new().text("source", url.clone());
        let (status, v) = post_predict(addr, "test-key-1", mp).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{url}: {v}");
        assert!(
            v["error"].as_str().unwrap().contains("private"),
            "{url}: {v}"
        );
    }
}

#[tokio::test]
async fn url_source_body_is_capped_without_content_length() {
    // Chunked response larger than max_upload_mb (4 MB in the harness).
    let app = Router::new().route(
        "/big",
        get(|| async {
            let chunks = futures_util::stream::iter(
                (0..6).map(|_| Ok::<_, std::io::Error>(Bytes::from(vec![0u8; 1024 * 1024]))),
            );
            axum::body::Body::from_stream(chunks)
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let big_addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway(mock_addr, "yolo26n").await;
    let mp = reqwest::multipart::Form::new().text("source", format!("http://{big_addr}/big"));
    let (status, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
}

#[tokio::test]
async fn triton_down_returns_json_503_then_recovers() {
    // Reserve a port with nothing listening: Triton is "down".
    let reserved = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let triton_addr = reserved.local_addr().unwrap();
    drop(reserved);

    let (addr, state) = start_gateway(triton_addr, "yolo26n", true).await;
    let jpeg = make_test_jpeg();

    // The gateway is up and explains why it cannot serve yet.
    for _ in 0..100 {
        if !state.load_errors().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let (status, v) = post_predict(addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    let err = v["error"].as_str().unwrap();
    assert!(
        err.contains("model 'yolo26n' is not ready") && err.contains("triton unreachable"),
        "{err}"
    );

    let resp = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let h: Value = resp.json().await.unwrap();
    assert_eq!(h["status"], "loading");
    assert!(
        h["endpoints"]["detect"]["error"]
            .as_str()
            .unwrap()
            .contains("triton unreachable"),
        "{h}"
    );

    // Auth is still checked first: no info leaks to unauthenticated callers.
    let (status, _) = post_predict(addr, "wrong", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Triton comes up on that address: the loader picks it up by itself.
    let listener = tokio::net::TcpListener::bind(triton_addr).await.unwrap();
    let _mock = spawn_mock_on(listener);
    for _ in 0..600 {
        if state.is_ready() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    let (status, v) = post_predict(addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let resp = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn triton_inference_error_is_json_503() {
    // A Triton that serves the model config but fails every inference.
    let app = Router::new()
        .route("/v2", get(mock_metadata))
        .route(
            "/v2/models/{model}/config",
            get(|| async { Json(yolo_config()) }),
        )
        .route(
            "/v2/models/{model}/infer",
            post(|| async {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "CUDA out of memory"})),
                )
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let triton_addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let addr = spawn_gateway(triton_addr, "yolo26n").await;
    let (status, v) = post_predict(addr, "test-key-1", mp_file(&make_test_jpeg())).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    assert!(
        v["error"].as_str().unwrap().contains("CUDA out of memory"),
        "{v}"
    );
}

#[tokio::test]
async fn url_sources_disabled_by_config() {
    let jpeg = make_test_jpeg();
    std::fs::write(test_image_path(), &jpeg).unwrap();
    let (mock_addr, _m) = spawn_mock().await;
    let addr = spawn_gateway_no_urls(mock_addr, "yolo26n").await;

    // URL source -> 400, without any outbound fetch.
    let mp = reqwest::multipart::Form::new().text("source", format!("http://{mock_addr}/image"));
    let (status, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert!(
        v["error"]
            .as_str()
            .unwrap()
            .contains("image URLs are disabled"),
        "{v}"
    );

    // Payload images still work: file upload and base64 source.
    let (status, _) = post_predict(addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::OK);
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(make_small_jpeg());
    let mp = reqwest::multipart::Form::new().text("source", b64);
    let (status, v) = post_predict(addr, "test-key-1", mp).await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

#[test]
fn url_sources_are_off_by_default() {
    assert!(!gateway::config::Limits::default().allow_url_sources);
}

// --- OCR (Cloud Vision images:annotate) ----------------------------------------

const OCR_ONLY: &str = "  ocr: { model: ppocr-det, api_keys: [test-key-1, test-key-2] }";

/// 320x160 black PNG with one white "text line" at x 40..=199, y 60..=89.
fn make_text_png() -> Vec<u8> {
    use image::{Rgb, RgbImage};
    let img = RgbImage::from_fn(320, 160, |x, y| {
        if (40..=199).contains(&x) && (60..=89).contains(&y) {
            Rgb([255, 255, 255])
        } else {
            Rgb([0, 0, 0])
        }
    });
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

fn annotate_body(image: &[u8], features: Value) -> Value {
    use base64::Engine;
    let content = base64::engine::general_purpose::STANDARD.encode(image);
    json!({ "requests": [{ "image": { "content": content }, "features": features }] })
}

async fn post_annotate(addr: SocketAddr, query: &str, body: &Value) -> (StatusCode, Value) {
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/images:annotate{query}"))
        .json(body)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    (status, resp.json::<Value>().await.unwrap_or(Value::Null))
}

fn vertices(poly: &Value) -> Vec<(i64, i64)> {
    poly["vertices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| (v["x"].as_i64().unwrap(), v["y"].as_i64().unwrap()))
        .collect()
}

#[tokio::test]
async fn ocr_text_detection_matches_cloud_vision_shape() {
    let (mock_addr, mock) = spawn_mock().await;
    let (addr, _) = spawn_gateway_endpoints(mock_addr, OCR_ONLY).await;

    let body = annotate_body(&make_text_png(), json!([{ "type": "TEXT_DETECTION" }]));
    let (status, v) = post_annotate(addr, "?key=test-key-1", &body).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let r = &v["responses"][0];
    assert!(r.get("error").is_none(), "{r}");

    let ann = r["textAnnotations"].as_array().unwrap();
    assert_eq!(ann[0]["description"], "HI YOU\n");
    assert_eq!(ann[1]["description"], "HI");
    assert_eq!(ann[2]["description"], "YOU");
    assert_eq!(ann.len(), 3);
    // Line box: component 159x29 px grown by DB unclip (d = 18.39).
    let line = vertices(&ann[0]["boundingPoly"]);
    assert_eq!(line, vec![(22, 42), (217, 42), (217, 107), (22, 107)]);
    // Words sit inside the line, in reading order.
    let (hi, you) = (
        vertices(&ann[1]["boundingPoly"]),
        vertices(&ann[2]["boundingPoly"]),
    );
    assert!(
        22 <= hi[0].0 && hi[1].0 <= you[0].0 && you[1].0 <= 217,
        "{hi:?} {you:?}"
    );

    let full = &r["fullTextAnnotation"];
    assert_eq!(full["text"], "HI YOU\n");
    let page = &full["pages"][0];
    assert_eq!(
        (page["width"].as_u64(), page["height"].as_u64()),
        (Some(320), Some(160))
    );
    let words = page["blocks"][0]["paragraphs"][0]["words"]
        .as_array()
        .unwrap();
    assert_eq!(words.len(), 2);
    let last = &words[1]["symbols"][2];
    assert_eq!(last["text"], "U");
    assert_eq!(last["property"]["detectedBreak"]["type"], "LINE_BREAK");
    assert_eq!(page["blocks"][0]["blockType"], "TEXT");

    // Detector then recognizer, one batch.
    assert_eq!(
        mock.seen.lock().unwrap().models,
        vec!["ppocr-det", "ppocr-rec"]
    );
}

#[tokio::test]
async fn ocr_region_only_mode_skips_recognition() {
    let (mock_addr, mock) = spawn_mock().await;
    let (addr, _) = spawn_gateway_endpoints(mock_addr, OCR_ONLY).await;

    let body = annotate_body(
        &make_text_png(),
        json!([{ "type": "TEXT_REGION_DETECTION" }]),
    );
    let (status, v) = post_annotate(addr, "?key=test-key-1", &body).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let r = &v["responses"][0];
    assert!(r.get("textAnnotations").is_none(), "{r}");
    let regions = r["textRegionAnnotations"].as_array().unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(
        vertices(&regions[0]["boundingPoly"]),
        vec![(22, 42), (217, 42), (217, 107), (22, 107)]
    );
    assert!((regions[0]["score"].as_f64().unwrap() - 0.9).abs() < 1e-3);
    assert_eq!(mock.seen.lock().unwrap().models, vec!["ppocr-det"]);

    // Both modes at once: regions *and* text (enum numbers as REST clients send).
    let body = annotate_body(
        &make_text_png(),
        json!([{ "type": 5 }, { "type": "TEXT_REGION_DETECTION" }]),
    );
    let (_, v) = post_annotate(addr, "?key=test-key-1", &body).await;
    let r = &v["responses"][0];
    assert_eq!(r["textAnnotations"][0]["description"], "HI YOU\n");
    assert_eq!(r["textRegionAnnotations"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn ocr_auth_and_errors_use_google_envelopes() {
    let (mock_addr, _m) = spawn_mock().await;
    let (addr, _) = spawn_gateway_endpoints(mock_addr, OCR_ONLY).await;
    let body = annotate_body(&make_text_png(), json!([{ "type": "TEXT_DETECTION" }]));

    // No key / wrong key: 401 with the Google error envelope.
    let (status, v) = post_annotate(addr, "", &body).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(v["error"]["code"], 401);
    assert_eq!(v["error"]["status"], "UNAUTHENTICATED");
    let (status, _) = post_annotate(addr, "?key=nope", &body).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // x-goog-api-key header and Bearer both work.
    for (h, val) in [
        ("x-goog-api-key", "test-key-2"),
        ("authorization", "Bearer test-key-1"),
    ] {
        let resp = reqwest::Client::new()
            .post(format!("http://{addr}/v1/images:annotate"))
            .header(h, val)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{h}");
    }

    // Per-image failures: HTTP 200, error in responses[i] (INVALID_ARGUMENT).
    let (_, v) = post_annotate(
        addr,
        "?key=test-key-1",
        &json!({ "requests": [
            { "image": { "content": "not base64!" }, "features": [{ "type": "TEXT_DETECTION" }] },
            { "image": { "source": { "gcsImageUri": "gs://b/o" } }, "features": [{ "type": "TEXT_DETECTION" }] },
            { "image": { "content": "" }, "features": [{ "type": "LABEL_DETECTION" }] },
            { "image": { "content": "" }, "features": [] }
        ]}),
    )
    .await;
    let rs = v["responses"].as_array().unwrap();
    assert_eq!(rs.len(), 4);
    for r in rs {
        assert_eq!(r["error"]["code"], 3, "{r}");
    }
    assert!(
        rs[2]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("LABEL_DETECTION")
    );

    // A blank image yields an empty response object, as Cloud Vision does.
    let blank = {
        let mut buf = std::io::Cursor::new(Vec::new());
        image::RgbImage::new(64, 64)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    };
    let (_, v) = post_annotate(
        addr,
        "?key=test-key-1",
        &annotate_body(&blank, json!([{ "type": "TEXT_DETECTION" }])),
    )
    .await;
    assert_eq!(v["responses"][0], json!({}));

    // Batch-level problems fail the whole call.
    let (status, v) = post_annotate(addr, "?key=test-key-1", &json!({ "requests": [] })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["status"], "INVALID_ARGUMENT");
    let many = json!({ "requests": vec![body["requests"][0].clone(); 17] });
    let (status, _) = post_annotate(addr, "?key=test-key-1", &many).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn endpoints_are_mounted_only_when_enabled() {
    let (mock_addr, _m) = spawn_mock().await;
    let jpeg = make_test_jpeg();

    // OCR only: /predict is not served.
    let (addr, _) = spawn_gateway_endpoints(mock_addr, OCR_ONLY).await;
    let (status, _) = post_predict(addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Detect only (explicit form): OCR is not served.
    let (addr, _) = spawn_gateway_endpoints(
        mock_addr,
        "  detect: { model: yolo26n, api_keys: [test-key-1] }\n  \
         ocr: { model: ppocr-det, api_keys: [test-key-1], enabled: false }",
    )
    .await;
    let body = annotate_body(&make_text_png(), json!([{ "type": "TEXT_DETECTION" }]));
    let (status, _) = post_annotate(addr, "?key=test-key-1", &body).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // Both from one gateway; /health reports each endpoint.
    let (addr, _) = spawn_gateway_endpoints(
        mock_addr,
        "  detect: { model: yolo26n, api_keys: [test-key-1] }\n  \
         ocr: { model: ppocr-det, api_keys: [test-key-1] }",
    )
    .await;
    let (status, v) = post_predict(addr, "test-key-1", mp_file(&jpeg)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_openapi_contract(&v);
    let (status, v) = post_annotate(addr, "?key=test-key-1", &body).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(
        v["responses"][0]["textAnnotations"][0]["description"],
        "HI YOU\n"
    );
    let h: Value = reqwest::get(format!("http://{addr}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(h["status"], "healthy");
    assert_eq!(h["endpoints"]["detect"]["status"], "healthy");
    assert_eq!(h["endpoints"]["ocr"]["model"], "ppocr-det");
}

#[tokio::test]
async fn model_task_must_match_its_endpoint() {
    let (mock_addr, _m) = spawn_mock().await;
    let (addr, state) = start_gateway_endpoints(
        mock_addr,
        "  ocr: { model: yolo26n, api_keys: [test-key-1] }",
    )
    .await;
    for _ in 0..100 {
        if !state.load_errors().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let err = state.load_errors().join("");
    assert!(err.contains("is a 'detect' model"), "{err}");
    let body = annotate_body(&make_text_png(), json!([{ "type": "TEXT_DETECTION" }]));
    let (status, v) = post_annotate(addr, "?key=test-key-1", &body).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    assert_eq!(v["error"]["status"], "UNAVAILABLE");
}

#[tokio::test]
async fn each_service_accepts_only_its_own_keys() {
    let (mock_addr, _m) = spawn_mock().await;
    let (addr, _) = spawn_gateway_endpoints(
        mock_addr,
        "  detect: { model: yolo26n, api_keys: [detect-key, shared-key] }\n  \
         ocr: { model: ppocr-det, api_keys: [ocr-key, shared-key] }",
    )
    .await;
    let jpeg = make_test_jpeg();
    let body = annotate_body(&make_text_png(), json!([{ "type": "TEXT_DETECTION" }]));

    for (key, detect, ocr) in [
        ("detect-key", StatusCode::OK, StatusCode::UNAUTHORIZED),
        ("ocr-key", StatusCode::UNAUTHORIZED, StatusCode::OK),
        ("shared-key", StatusCode::OK, StatusCode::OK),
    ] {
        let (status, _) = post_predict(addr, key, mp_file(&jpeg)).await;
        assert_eq!(status, detect, "/predict with {key}");
        let (status, _) = post_annotate(addr, &format!("?key={key}"), &body).await;
        assert_eq!(status, ocr, "/v1/images:annotate with {key}");
    }
}
