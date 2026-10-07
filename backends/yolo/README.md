# YOLO backend (`family: "yolo"`)

Serves Ultralytics YOLO detection models exported with the **end-to-end
(NMS-free) head**, e.g. YOLO26 and YOLO11 with the one-to-one head. No NMS
runs anywhere: the ONNX graph already emits final detections.

## ⚠️ License: bring your own

| Component | License |
|-----------|---------|
| This folder's code (`src/`, `export.py`, `triton_repo.py`) | MIT |
| `ultralytics` Python package (used by `export.py`) | **AGPL-3.0** |
| Official Ultralytics YOLO weights (`yolo11*.pt`, `yolo26*.pt`, ...) | **AGPL-3.0** |

Serving YOLO weights as a network service, **including staking a Pocket
Network supplier**, triggers AGPL-3.0's network clause. Ultralytics requires
an **[Ultralytics Enterprise License](https://www.ultralytics.com/license)**
for commercial use that does not release the entire service under AGPL-3.0.
**You are responsible for holding that license before you stake with a YOLO
model.** Fine-tuned YOLO weights are derivatives and carry the same terms.

To ship a gateway with no YOLO code at all, build it without this backend:
`cargo build -p gateway --no-default-features --features rfdetr`.

## Model contract

| | |
|-|-|
| Input | `images` `[1, 3, imgsz, imgsz]` FP32 RGB in 0..1 |
| Pre-processing | Letterbox as in Ultralytics `LetterBox(auto=False)`: scale `r = min(S/w, S/h)`, centered, gray (114) padding |
| Output | `output0` `[1, N, 6]`: `x1, y1, x2, y2, confidence, class_id` in letterboxed pixels; padding rows have `class_id < 0` or `confidence = 0` |
| Post-processing | Confidence filter, inverse letterbox, clamp, sort, cap at 300 |
| Extra metadata | `end2end: true` (required), `stride` (informational) |
| Class ids | Dense `0..nc-1` (`names[i]`) |

## Export

```bash
uv sync --extra yolo
uv run python backends/yolo/export.py yolo26n                      # alias or .pt path
uv run python backends/yolo/export.py models/custom.pt --name my-yolo --imgsz 640
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
```

The exporter refuses non-end-to-end heads. The Triton config uses
`max_batch_size: 8` with dynamic batching (`--max-batch`, `--queue-delay-us`).
