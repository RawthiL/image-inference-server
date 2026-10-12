# image-inference-server

An **image-inference backend for [Pocket Network](https://pocket.network)**.
It serves any supported model through [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server), behind existing public APIs:

- **Object detection**: an **Ultralytics-compatible `/predict` API**.
- **OCR**: a **Google Cloud Vision-compatible `/v1/images:annotate` API**.

The operator chooses which endpoints a deployment serves in `config.yaml`.

The service is **model-agnostic**. Clients always see the same request and response format, whichever model the operator chooses to run. Each model
family is a self-contained plugin under [`backends/`](backends/). Adding amodel means adding a folder.

> ⚠️ **Some supported models are not free for commercial use.** Serving a
> model on Pocket Network (staking a supplier for this service) is
> commercial use. **You must hold a valid license for the model you serve.**
> See [Model licensing](#model-licensing) before you stake.

---

## Architecture

```
                      ┌──────────────────────── gateway/ (interface) ────────────────────────┐
 client / Pocket      │  HTTP API (Ultralytics-compatible) · auth · image fetch + decode      │
 RelayMiner ─POST────▶│  SSRF guard · limits · Triton transport · response formatting         │
 /predict             │                         │                       ▲                     │
                      │            Backend::preprocess        Backend::postprocess            │
                      └─────────────────────────┼───────────────────────┼─────────────────────┘
                                                ▼                       │
                      ┌──── backends/<family>/ (one folder per model family) ────┐
                      │  yolo/    letterbox → [N,6] end-to-end head               │
                      │  rfdetr/  resize+ImageNet norm → dets/labels + sigmoid    │
                      │  <new>/   …                                               │
                      └───────────────────────────────────────────────────────────┘
                                                │ input tensor     ▲ output tensors
                                                ▼                  │
                                  Triton (onnxruntime, dynamic batching)
```

The code is split into three layers:

| Layer | Path | Responsibility | Knows about models? |
|-------|------|----------------|---------------------|
| **Interface** | [`gateway/`](gateway/) | HTTP API, auth, input handling, Triton client, JSON response | **No** |
| **Contract** | [`backend-api/`](backend-api/) | One backend trait per task (`Backend` for detect, `OcrBackend` for OCR), shared types/helpers | No |
| **Backends** | [`backends/<family>/`](backends/) | Pre-processing, output decoding, ONNX exporter, fixtures, license notes | Only its own family |

At startup the gateway reads each enabled endpoint's model `config.pbtxt`
from Triton. The `task` field in its embedded metadata must match the
endpoint (`detect` or `ocr`), and the `family` field (written by that
backend's exporter) selects the backend. A model can list `companions`
(e.g. an OCR detector names its recognizer), which are loaded with it.
Nothing in `gateway/` branches on model type.

Each endpoint is one *task* with its own wire contract, copied from an
existing public API so existing clients work unchanged:

| Endpoint | Task | Compatible with | Code |
|----------|------|-----------------|------|
| `POST /predict` | `detect` | Ultralytics Platform `predict` | `gateway/src/endpoints/detect.rs` |
| `POST /v1/images:annotate` | `ocr` | Google Cloud Vision `images:annotate` | `gateway/src/endpoints/vision.rs` |

## Supported backends

| Backend | Folder | Models | Code license | **Weights license** |
|---------|--------|--------|--------------|---------------------|
| YOLO (end-to-end, NMS-free) | [`backends/yolo`](backends/yolo/) | Ultralytics YOLO11 / YOLO26 detect (`end2end` head) | MIT | **AGPL-3.0, Enterprise License needed for commercial use** |
| RF-DETR | [`backends/rfdetr`](backends/rfdetr/) | RF-DETR Nano / Small / Base / Medium / Large | MIT | Apache-2.0 ✅ |
| RF-DETR (large tiers) | [`backends/rfdetr`](backends/rfdetr/) | RF-DETR XLarge / 2XLarge | MIT | **Roboflow Platform Model License, not open source** |
| PP-OCR (OCR) | [`backends/ppocr`](backends/ppocr/) | PaddleOCR PP-OCR det + rec (v3 to v6; v6 medium recommended) | MIT | Apache-2.0 ✅ |

Each backend is a Cargo feature, so you can build a gateway that contains
only the families you are licensed to run:

```bash
cargo build --release -p gateway --no-default-features --features rfdetr
```

## Model licensing

**Read this before serving a model on Pocket Network.**

All code in this repository is MIT-licensed (see [`LICENSE`](LICENSE)).
**That license does not cover model weights or third-party software.** A
model checkpoint keeps its own license no matter which server runs it.

Staking a Pocket Network supplier for this service, or running it in any
other paid or commercial setting, is a **commercial network service**. Some
model licenses forbid that unless you obtain a separate license:

- **Ultralytics YOLO (all official weights, and the `ultralytics` package
  used by `backends/yolo/export.py`) is AGPL-3.0.** Ultralytics requires an
  **[Ultralytics Enterprise License](https://www.ultralytics.com/license)**
  for commercial use that does not open-source the whole service under
  AGPL-3.0. **Bring your own license if you stake with a YOLO model.**
- **RF-DETR XLarge and 2XLarge weights** are released under Roboflow's
  **Platform Model License (PML)**, which is not an open-source license.
  **Bring your own Roboflow license if you stake with these models.**
- **RF-DETR Nano / Small / Base / Medium / Large** weights and the `rfdetr`
  package are **Apache-2.0** and can be served commercially, subject to the
  usual Apache-2.0 notice requirements.
- **Fine-tuned or third-party checkpoints** (including Hugging Face
  conversions) inherit the license of their base model and training data.
  Check the source of every checkpoint you deploy.
- **NVIDIA Triton** is BSD-3-Clause. The prebuilt `nvcr.io` container used in
  [`docker/`](docker/) bundles CUDA libraries under NVIDIA's container license.

**The operator is solely responsible for holding the rights to serve the
model they deploy.** The maintainers of this repository do not grant, and
cannot grant, any rights to third-party models. This section is guidance,
not legal advice.

"Ultralytics" and "YOLO" are used only to describe API compatibility. This
project is not affiliated with or endorsed by Ultralytics or Roboflow.

## Repository layout

| Path | What |
|------|------|
| `gateway/` | Rust (axum) HTTP interface: endpoints, auth, sources, Triton client |
| `backend-api/` | Rust crate defining the `Backend` trait every model implements |
| `backends/yolo/` | YOLO backend: Rust pre/post-processing, `export.py`, fixtures, license notes |
| `backends/rfdetr/` | RF-DETR backend: same structure |
| `backends/ppocr/` | PP-OCR (OCR) backend: same structure, two Triton models (detector + recognizer) |
| `backends/README.md` | **How to add a new backend** |
| `tools/` | Backend-agnostic Python tooling: `config.pbtxt` writer, fixture generator, Triton schema validator |
| `triton/model_repository/` | Triton model repository (generated) |
| `docker/` | Compose stack: Triton + gateway |
| `tests/` | Mock Triton (ONNX Runtime) + end-to-end API contract tests |
| `config.example.yaml` | Gateway configuration reference |

## Quick start

### 1. Put a model into the Triton repository

Smoke-test fixtures (no weights, no torch; constant outputs, useful for dev
and CI):

```bash
uv run --extra onnx python tools/make_fixtures.py      # one fixture per backend
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
```

Real weights (needs torch; check the license first):

```bash
uv sync --extra rfdetr

# -> triton/model_repository/rfdetr-base
uv run python backends/rfdetr/export.py --variant base         

# -> Export from cloned hugingface project
uv run python backends/rfdetr/export.py \
    --checkpoint /huggingface-hub/rf-detr-small \
    --name rfdetr-small

# AGPL-3.0: bring your license
uv sync --extra yolo                                           
# -> triton/model_repository/yolo26n
uv run python backends/yolo/export.py yolo26n                  

# OCR: a PP-OCR detector + recognizer ONNX pair (see backends/ppocr/README.md)
# -> triton/model_repository/ppocr-det + ppocr-rec
uv run --extra onnx python backends/ppocr/export.py \
    --det det.onnx --rec rec.onnx --dict ppocr_keys_v1.txt --name ppocr
```

Each exporter writes `triton/model_repository/<name>/{config.pbtxt,1/model.onnx}`.
Run `tools/verify_repo.py` after every export.

### 2. Run Triton + gateway

```bash
cp config.example.yaml config.yaml     # set endpoints and their api_keys
cd docker && docker compose up --build  # gateway on :8080, Triton stays internal
# only compile the backends you serve:
GATEWAY_FEATURES=rfdetr docker compose up --build
```

To develop on a CPU without the Triton image, use the protocol-faithful
ONNX Runtime mock:

```bash
uv run --extra test python tests/mock_triton.py --repo triton/model_repository --port 8000 &
cargo run -p gateway -- --config config.yaml     # config: triton.url = http://127.0.0.1:8000
```

### 3. Call it

```bash
curl -s http://localhost:8080/predict \
  -H "Authorization: Bearer change-me-client-a" \
  -F "file=@image.jpg" -F conf=0.25 -F iou=0.7 -F imgsz=640
```

Existing Ultralytics client code works unchanged. Only the URL changes:

```python
import requests
r = requests.post("https://your-endpoint/predict",
                  headers={"Authorization": f"Bearer {api_key}"},
                  data={"conf": 0.25, "iou": 0.7, "imgsz": 640},
                  files={"file": open("image.jpg", "rb")})
print(r.json())   # Ultralytics PredictResponse JSON
```

OCR, with the `ocr` endpoint enabled. Cloud Vision clients work unchanged:

```bash
curl -s "http://localhost:8080/v1/images:annotate?key=change-me-client-a" \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"image": {"content": "'"$(base64 < page.png | tr -d '\n')"'"},
                     "features": [{"type": "TEXT_DETECTION"}]}]}'
```

```python
from google.api_core.client_options import ClientOptions
from google.cloud import vision

client = vision.ImageAnnotatorClient(
    transport="rest",   # the gateway speaks REST, not gRPC
    client_options=ClientOptions(api_endpoint="https://your-endpoint",
                                 api_key=api_key))
r = client.text_detection(image=vision.Image(content=open("page.png", "rb").read()))
print(r.full_text_annotation.text)
```

## API

| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Detection on one image (`endpoints.detect`) |
| `POST /api/deployments/{owner}/{deployment}/predict` | Same, with the Ultralytics Platform path shape |
| `POST /v1/images:annotate` | OCR, Cloud Vision-compatible (`endpoints.ocr`) |
| `GET /health` | `200` when Triton serves every enabled endpoint's models, `503` otherwise (no auth). Per-endpoint detail under `endpoints` |

Endpoints that are not enabled in the config are not mounted (`404`).

### Detection: `POST /predict`

The request body is `multipart/form-data` or `application/x-www-form-urlencoded`:

| Field | Notes |
|-------|-------|
| `file` | Image upload (multipart only). JPEG, PNG, WebP, BMP, TIFF |
| `source` | Alternative to `file`: base64 string or `data:` URI (max 4096 characters). `http(s)` URLs only if the operator enables `limits.allow_url_sources` |
| `conf` | Confidence threshold, 0.01 to 1 (default from config) |
| `iou` | Accepted for compatibility, 0 to 0.95. Both current backends are NMS-free, so it is ignored |
| `imgsz` | Accepted for compatibility, 32 to 1280. Inference always runs at the model's native size |
| `normalize` | `true` returns boxes in 0 to 1 (5 decimals) |
| `decimals` | Coordinate precision, 0 to 10 (default 3) |

The response is the Ultralytics `PredictResponse`: `images[].results[]` with
`name`, `class`, `confidence` and `box {x1,y1,x2,y2}` in original-image
pixels, plus `speed` and `metadata`. Every error is returned as
`{"error": "..."}` with a fitting HTTP status.

### OCR: `POST /v1/images:annotate`

The request and response follow Google Cloud Vision
[`images:annotate`](https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate):
`{"requests": [{"image": {...}, "features": [{"type": ...}]}]}`, up to
`endpoints.ocr.max_images_per_request` images per call (default 16).

| Feature `type` | Result |
|----------------|--------|
| `TEXT_DETECTION`, `DOCUMENT_TEXT_DETECTION` | Full OCR: `textAnnotations` (full text first, then one entry per word) and `fullTextAnnotation` (pages → blocks → paragraphs → words → symbols, one block per text line, with `detectedBreak`s) |
| `TEXT_REGION_DETECTION` | **Extension, not in Cloud Vision.** Text regions only, without recognition (only the detector runs): `textRegionAnnotations: [{boundingPoly, score}]` |

Notes:

- Images: `image.content` (base64), or `image.source.imageUri` (http(s), only
  with `limits.allow_url_sources`). `gcsImageUri` is not supported.
- Auth: `?key=<api key>`, `x-goog-api-key: <api key>`, or `Authorization: Bearer <api key>`.
- Feature types are accepted as names or as their enum numbers (Google REST
  clients send numbers). `imageContext` (e.g. `languageHints`) is ignored:
  the served model defines the languages. No `locale` is returned.
- Polygons are in original-image pixels, in reading order (vertex 0 is the
  top-left of the text as read), as in Cloud Vision.
- Errors use the Google envelope `{"error": {"code", "message", "status"}}`.
  A failure that concerns one image (bad base64, unsupported feature, Triton
  error) goes in `responses[i].error` with HTTP 200, as Cloud Vision does.
  An image without text yields `{}`.

### Resolution and batching

Every model runs at its native square resolution (for example YOLO 640,
RF-DETR Base 560). The backend maps boxes back to original-image pixels.
The gateway sends one image per Triton request, and Triton's dynamic
batcher (configured by the exporter) groups concurrent requests on the GPU.

### Security and limits

- API keys are per endpoint (`endpoints.<name>.api_keys`): a key opens only
  the endpoints that list it. Keys are compared in constant time, and the
  gateway refuses to start if an enabled endpoint has no keys.
- **Image URLs are off by default.** Only images in the request payload are
  accepted (`file`, or base64 in `source`); an `http(s)` URL in `source`
  returns `400`. Fetching URLs would let any caller make the node download
  arbitrary public content from its own IP, costs egress bandwidth, and can
  make answers differ between suppliers if the content behind a URL changes.
  Operators can opt in with `limits.allow_url_sources: true`.
- When enabled, URL fetching is guarded against server-side request forgery
  (SSRF). Every connection, including each redirect hop, is resolved through
  a resolver that rejects private, loopback, link-local, CGNAT, reserved and
  IPv4-embedding IPv6 addresses. IP-literal URLs and redirects are checked
  too, and environment proxies are ignored. Disable the guard only in
  development (`limits.allow_private_urls: true`).
- Upload and URL body size are capped (`limits.max_upload_mb`). URL bodies
  are streamed with a running cap.
- Image dimensions are checked against `limits.max_image_pixels` *before*
  decoding (decompression-bomb guard).
- Decoding and pre-processing run on a blocking thread pool, so large images
  don't stall the async server.

## Configuration

See [`config.example.yaml`](config.example.yaml). Key fields:

- `endpoints.<name>`: which APIs (services) this deployment serves (`detect`,
  `ocr`) and the Triton model behind each. A missing block or
  `enabled: false` leaves the endpoint unmounted. Swap between any exported
  models of the right task freely; the backend is selected automatically.
- `endpoints.<name>.api_keys`: the keys accepted by that endpoint only, so
  each service can have its own clients. List a key under several endpoints
  (or use a YAML anchor) to share it. The gateway refuses to start if an
  enabled endpoint has no keys.
- `triton.url` / `triton.api_key` / `triton.timeout_ms`: Triton endpoint. The `TRITON_URL` env var overrides `triton.url`; `docker/docker-compose.yaml` sets it to `http://triton:8000`, so a dev `config.yaml` pointing at localhost still works in containers.
- `limits.*`: upload cap, pixel cap, URL sources (`allow_url_sources`, off by default), URL fetch timeout, SSRF guard.
- `endpoints.detect.defaults.*`: `conf` / `iou` / `decimals` used when a request omits them.
- `endpoints.ocr.defaults.*`: `min_region_score` (detector floor; unset uses the model's), `min_text_confidence`, `max_regions`.

## Adding a model backend

1. Create `backends/<name>/` with a Rust crate exporting `FAMILY` and
   `create()`, implementing its task's trait: `backend_api::Backend` for
   detection (`preprocess`: image to tensor, `postprocess`: tensors to
   detections) or `backend_api::ocr::OcrBackend` for OCR (detect and
   recognize stages; the gateway drives the Triton calls).
2. Add `export.py` (weights to ONNX plus `config.pbtxt` with
   `"family": "<name>"`), `triton_repo.py` (fixture and config check) and a
   `README.md` **stating the model's license**.
3. Register it: one workspace dependency, one Cargo feature, and one line in
   its task's table in `gateway/src/registry.rs`.

A new *task* (a new API) is a new trait in `backend-api`, a registry table,
a config block in `Endpoints`, and an `endpoints/<api>.rs` module that
implements the public API it is compatible with.

The step-by-step guide is in [`backends/README.md`](backends/README.md).
`gateway/` does not change beyond the registry line.

## Tests

```bash
bash scripts/test-all.sh   # runs everything below
```

```bash
# Rust: contract crate, every backend, gateway e2e against a mock Triton
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build -p gateway --no-default-features --features rfdetr   # backend isolation

# Python: release gateway + ONNX Runtime mock Triton, validated against the
# vendored PredictResponse schema for every fixture backend
cargo build --release -p gateway
uv run --extra test python tests/test_predict_integration.py
# OCR: Cloud Vision images:annotate contract (fixture + real PP-OCR if exported)
uv run --extra test python tests/test_vision_integration.py

# Triton config schema validation (real model_config.proto)
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
```

## License

The code in this repository is released under the [MIT License](LICENSE).
Model weights and third-party components keep their own licenses. See
[Model licensing](#model-licensing).
