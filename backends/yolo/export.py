#!/usr/bin/env python3
"""Export a YOLO detection model (end-to-end head) to ONNX + Triton repo.

Requires the `yolo` extra (torch + ultralytics):

    uv run --extra yolo python backends/yolo/export.py models/yolo26n.pt
    uv run --extra yolo python backends/yolo/export.py yolo26n

LICENSE: the `ultralytics` package and Ultralytics YOLO weights are AGPL-3.0.
Serving them in a commercial service — including staking a Pocket Network
supplier — requires either full AGPL-3.0 compliance or an Ultralytics
Enterprise License. You are responsible for that. See backends/yolo/README.md.

Only end-to-end (NMS-free, output [batch, 300, 6]) heads are served; the Rust
gateway postprocesses exactly that layout (no NMS). The written config.pbtxt
embeds a JSON `metadata` string used by the gateway for model-family
detection, class names and native resolution.

Serving shape is FIXED to imgsz x imgsz (gateway always letterboxes to the
model's native resolution and maps boxes back to original pixels), while
max_batch_size + dynamic_batching let Triton batch concurrent requests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common  # noqa: E402

LICENSE_WARNING = (
    "WARNING: Ultralytics YOLO weights are AGPL-3.0. Commercial serving (including a "
    "Pocket Network supplier) needs AGPL compliance or an Ultralytics Enterprise License."
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", help=".pt path or alias resolvable by Ultralytics (e.g. yolo26n)")
    p.add_argument("--repo", type=Path, default=ROOT / "triton" / "model_repository")
    p.add_argument("--name", default=None, help="Triton model name (default: weights stem)")
    p.add_argument("--imgsz", type=int, default=640, help="square export/serving size")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--queue-delay-us", type=int, default=4000)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is required: uv sync --extra yolo")
    print(LICENSE_WARNING, file=sys.stderr)

    model = YOLO(args.weights)
    if model.task != "detect":
        sys.exit(f"only 'detect' models are served (got task={model.task!r})")
    name = args.name or Path(str(args.weights)).stem.removesuffix(".pt")

    captured: list[dict] = []
    model.add_callback("on_export_end", lambda e: captured.append(dict(e.metadata or {})))

    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True))
    meta = captured[0] if captured else {}
    end2end = bool(meta.get("end2end", False))
    if not end2end:
        sys.exit(
            "exported head is not end-to-end (NMS-free). Use yolo11/yolo26 with the "
            "one-to-one head; the gateway intentionally does not run NMS."
        )

    info = common.inspect_onnx(onnx_path)
    oname = info["outputs"][0]["name"]
    if not common.has_dynamic_batch(info["input"]["dims"]):
        sys.exit(f"exported ONNX input {info['input']['dims']} has a static batch dim; "
                 "Triton batching needs a dynamic one (export uses dynamic=True — check ultralytics)")

    # Confirm the [batch, N, 6] layout with an actual ORT run when available.
    rows, cols = None, None
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        res = sess.run(None, {info["input"]["name"]: np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)})
        rows, cols = res[0].shape[-2], res[0].shape[-1]
    except Exception:
        d = info["outputs"][0]["dims"]
        rows, cols = d[-2], d[-1]
        if not (isinstance(rows, int) and isinstance(cols, int)):
            sys.exit("cannot determine output dims; install onnxruntime or re-export dynamic=False")
    if cols != 6:
        sys.exit(f"unexpected end2end output dims [.., {rows}, {cols}]; expected [batch, N, 6]")

    names_raw = meta.get("names") or getattr(model, "names", {}) or {}
    if isinstance(names_raw, dict):
        n = max(int(k) for k in names_raw) + 1
        names_list = common.dense_names({int(k): str(v) for k, v in names_raw.items()}, n)
    else:
        names_list = [str(v) for v in names_raw]

    stride_raw = meta.get("stride", 32)
    stride = int(max(stride_raw)) if isinstance(stride_raw, (list, tuple)) else int(stride_raw)

    metadata = {
        "family": "yolo",
        "task": "detect",
        "names": names_list,
        "imgsz": int(args.imgsz),
        "stride": stride,
        "end2end": True,
        "input": info["input"]["name"],
        "outputs": [oname],
        "queries": int(rows),
    }
    cfg = common.build_config_pbtxt(
        model_name=name,
        metadata=metadata,
        input_spec={"name": info["input"]["name"], "data_type": "TYPE_FP32",
                    "dims": [3, int(args.imgsz), int(args.imgsz)]},
        output_specs=[{"name": oname, "data_type": common.triton_dtype(info["outputs"][0]["elem"]),
                       "dims": [int(rows), int(cols)]}],
        max_batch_size=args.max_batch,
        queue_delay_us=args.queue_delay_us,
    )
    repo = common.write_repo(args.repo, name, onnx_path, cfg)
    print(f"OK: {repo}/config.pbtxt + 1/model.onnx — '{name}' e2e {rows}x{cols}, "
          f"{len(names_list)} classes, imgsz={args.imgsz}, stride={stride}")


if __name__ == "__main__":
    main()
