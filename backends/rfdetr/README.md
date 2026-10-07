# RF-DETR backend (`family: "rfdetr"`)

Serves [RF-DETR](https://github.com/roboflow/rf-detr) detection models: a
DETR-style transformer that is NMS-free by design.

## License

| Component | License |
|-----------|---------|
| This folder's code (`src/`, `export.py`, `triton_repo.py`) | MIT |
| `rfdetr` Python package | Apache-2.0 |
| RF-DETR **Nano / Small / Base / Medium / Large** weights | **Apache-2.0** ✅ commercial use OK |
| RF-DETR **XLarge / 2XLarge** weights | **Roboflow Platform Model License, not open source** |

The XLarge and 2XLarge checkpoints are supported, but **serving them
commercially, including staking a Pocket Network supplier, requires a
license from Roboflow. You are responsible for holding it.** `export.py`
prints a warning when you export these variants.

Custom fine-tunes inherit the license of the base checkpoint they started
from. Hugging Face conversions carry the license stated in their model card.
Check it before you deploy.

## Model contract

| | |
|-|-|
| Input | `[1, 3, imgsz, imgsz]` FP32 RGB (`input`, or `pixel_values` for HF exports) |
| Pre-processing | Stretch-resize to `imgsz x imgsz` (no aspect preservation), /255, ImageNet mean/std |
| Outputs | `dets` `[1, Q, 4]` normalized `cxcywh`; `labels` `[1, Q, slots]` raw logits |
| Post-processing | Per-class sigmoid, drop `bg_slot`, top-k over all (query, class) pairs like the rfdetr reference `PostProcess`, threshold, cap at 300, scale to original pixels |
| Extra metadata | `bg_slot`: `null` (official sparse-COCO checkpoints), `-1` (contiguous fine-tunes), `0` (HF exports with an `N/A` slot 0) |
| Class ids | Slot index. Official COCO checkpoints use the sparse COCO id (91 slots) |

## Export

`export.py` supports three checkpoint sources:

| Input | Loaded via | Triton batching |
|-------|------------|-----------------|
| `--variant nano\|small\|base\|medium\|large\|xlarge\|2xlarge` | `rfdetr` (downloads pretrained weights) | dynamic batch, `max_batch_size: 8` |
| `--checkpoint my.pth` (rfdetr training output) | `rfdetr` (`RFDETR.from_checkpoint`) | dynamic batch |
| `--checkpoint <HF dir or model.safetensors>` (`model_type: rf_detr`) | `transformers.RfDetrForObjectDetection` | batch-1 graph: `max_batch_size: 0` + 4 instances |

```bash
uv sync --extra rfdetr
uv run python backends/rfdetr/export.py --variant base
uv run python backends/rfdetr/export.py --checkpoint models/my_rfdetr.pth \
    --class-names cup mug bottle --bg-slot -1 --name rfdetr-mycups
uv run python backends/rfdetr/export.py --checkpoint /path/to/rf-detr-small   # HF format
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
```

For HF exports, the positional-embedding interpolation (not exportable to
ONNX) is pinned at the native resolution. This is bit-exact because at that
resolution the interpolation is a no-op.
