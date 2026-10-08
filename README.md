# image-inference-server

An **object-detection backend for [Pocket Network](https://pocket.network)**.
It exposes an **Ultralytics-compatible `/predict` API** and serves any supported detection model through [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).

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
| **Contract** | [`backend-api/`](backend-api/) | The `Backend` trait and shared types/helpers | No |
| **Backends** | [`backends/<family>/`](backends/) | Pre-processing, output decoding, ONNX exporter, fixtures, license notes | Only its own family |

At startup the gateway reads the served model's `config.pbtxt` from Triton.
The `family` field in its embedded metadata (written by that backend's
exporter) selects the backend. Nothing in `gateway/` branches on model type.

## Supported backends

| Backend | Folder | Models | Code license | **Weights license** |
|---------|--------|--------|--------------|---------------------|
| YOLO (end-to-end, NMS-free) | [`backends/yolo`](backends/yolo/) | Ultralytics YOLO11 / YOLO26 detect (`end2end` head) | MIT | **AGPL-3.0, Enterprise License needed for commercial use** |
| RF-DETR | [`backends/rfdetr`](backends/rfdetr/) | RF-DETR Nano / Small / Base / Medium / Large | MIT | Apache-2.0 ✅ |
| RF-DETR (large tiers) | [`backends/rfdetr`](backends/rfdetr/) | RF-DETR XLarge / 2XLarge | MIT | **Roboflow Platform Model License, not open source** |

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
| `gateway/` | Rust (axum) HTTP interface: `/predict`, auth, sources, Triton client |
| `backend-api/` | Rust crate defining the `Backend` trait every model implements |
| `backends/yolo/` | YOLO backend: Rust pre/post-processing, `export.py`, fixtures, license notes |
| `backends/rfdetr/` | RF-DETR backend: same structure |
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
```

Each exporter writes `triton/model_repository/<name>/{config.pbtxt,1/model.onnx}`.
Run `tools/verify_repo.py` after every export.

### 2. Run Triton + gateway

```bash
cp config.example.yaml config.yaml     # set default_model + api_keys
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

## API

| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Detection on one image |
| `POST /api/deployments/{owner}/{deployment}/predict` | Same, with the Ultralytics Platform path shape |
| `GET /health` | `200` when Triton serves the model, `503` otherwise (no auth) |

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
`{"error": "..."}` with a fitting HTTP status. Only object detection is served.

### Resolution and batching

Every model runs at its native square resolution (for example YOLO 640,
RF-DETR Base 560). The backend maps boxes back to original-image pixels.
The gateway sends one image per Triton request, and Triton's dynamic
batcher (configured by the exporter) groups concurrent requests on the GPU.

### Security and limits

- Bearer API keys are compared in constant time. The gateway refuses to start
  with an empty key list.
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

- `default_model`: the Triton model this deployment serves. Swap between any
  exported models freely; the backend is selected automatically.
- `api_keys`: accepted `Bearer` keys.
- `triton.url` / `triton.api_key` / `triton.timeout_ms`: Triton endpoint. The `TRITON_URL` env var overrides `triton.url`; `docker/docker-compose.yaml` sets it to `http://triton:8000`, so a dev `config.yaml` pointing at localhost still works in containers.
- `limits.*`: upload cap, pixel cap, URL sources (`allow_url_sources`, off by default), URL fetch timeout, SSRF guard.
- `defaults.*`: `conf` / `iou` / `decimals` used when a request omits them.

## Adding a model backend

1. Create `backends/<name>/` with a Rust crate exporting `FAMILY` and
   `create()`, implementing `backend_api::Backend`. The trait has two jobs:
   `preprocess` (image to tensor) and `postprocess` (tensors to detections).
2. Add `export.py` (weights to ONNX plus `config.pbtxt` with
   `"family": "<name>"`), `triton_repo.py` (fixture and config check) and a
   `README.md` **stating the model's license**.
3. Register it: one workspace dependency, one Cargo feature, and one line in
   `gateway/src/registry.rs`.

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

# Triton config schema validation (real model_config.proto)
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
```

## License

The code in this repository is released under the [MIT License](LICENSE).
Model weights and third-party components keep their own licenses. See
[Model licensing](#model-licensing).
