# Model backends

Each folder here is one **model family**: everything specific to that model,
and nothing else. The gateway (`../gateway/`) is model-agnostic. It only talks
to backends through the `Backend` trait in [`../backend-api`](../backend-api/src/lib.rs).

```
backends/<name>/
├── Cargo.toml        # crate `backend-<name>`, depends on backend-api
├── src/lib.rs        # FAMILY + create() + impl Backend (pre/post-processing)
├── export.py         # weights -> ONNX + Triton config.pbtxt (metadata.family = "<name>")
├── triton_repo.py    # FAMILY, check(conf, meta), make_fixture(repo, workdir)
└── README.md         # model contract + LICENSE of the weights
```

## The contract

### Rust (`src/lib.rs`)

```rust
pub const FAMILY: &str = "<name>";                         // == metadata "family"
pub fn create(info: &ModelInfo) -> Result<Box<dyn Backend>>; // parse metadata, validate

impl Backend for MyModel {
    fn family(&self) -> &'static str;
    fn class_names(&self) -> &[String];
    fn output_names(&self) -> &[String];      // Triton outputs to request
    fn input_size(&self) -> (u32, u32);
    fn preprocess(&self, image: &RgbImage) -> Prepared;     // tensor + BoxMap
    fn postprocess(&self, outputs: &[OutputTensor], map: &BoxMap, params: &DecodeParams)
        -> Result<Vec<Detection>>;            // original-pixel boxes, sorted, <= max_det
}
```

- `preprocess` runs on a blocking thread. It receives the decoded RGB8 image
  and returns the FP32 input tensor plus a `BoxMap` (the affine mapping from
  network pixels back to original pixels).
- `postprocess` decodes the raw output tensors. If the model's head needs
  NMS, run it here (`params.iou` carries the client's IoU threshold).
- `backend_api::util` has shared helpers: CHW packing, output lookup with
  optional batch dim, sigmoid, clamping, sort-and-cap.

### Metadata (`config.pbtxt` → `parameters.metadata.string_value`)

The exporter embeds a JSON blob. These fields are common to all backends
(`backend_api::CommonMeta`):

```json
{ "family": "<name>", "task": "detect", "names": ["..."], "imgsz": 640,
  "input": "images", "outputs": ["output0"] }
```

Add any extra fields your backend needs (e.g. `end2end` for YOLO, `bg_slot`
for RF-DETR) and parse them in `create`. Use `tools/common.py`
(`build_config_pbtxt`, `inspect_onnx`, `write_repo`) to write the repository.

### Python hooks (`triton_repo.py`)

`tools/make_fixtures.py` and `tools/verify_repo.py` discover every
`backends/*/triton_repo.py` automatically:

```python
FAMILY = "<name>"
def check(conf, meta) -> None: ...                # assert family-specific config.pbtxt shape
def make_fixture(repo: Path, workdir: Path) -> str: ...   # constant-output ONNX, returns model name
```

## Registering a new backend

The workspace already includes `backends/*`. Then:

1. Root `Cargo.toml`, under `[workspace.dependencies]`:
   `backend-<name> = { path = "backends/<name>" }`
2. `gateway/Cargo.toml`: add `backend-<name> = { workspace = true, optional = true }`,
   a feature `<name> = ["dep:backend-<name>"]`, and add it to `default` if it
   should be built in by default.
3. `gateway/src/registry.rs`, in `BACKENDS`:
   ```rust
   #[cfg(feature = "<name>")]
   (backend_<name>::FAMILY, backend_<name>::create),
   ```
4. Optionally add a `<name>` extra in `pyproject.toml` for the exporter's
   dependencies.

Before opening a PR, document the **weights license** in your backend's README,
and add it to the "Supported backends" and "Model licensing" sections of the
root README. Pocket Network suppliers rely on that information.
