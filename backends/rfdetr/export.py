#!/usr/bin/env python3
"""Export an RF-DETR detection model to ONNX + Triton repo.

Supports three checkpoint flavors:

  1. Official pretrained tier (downloads via the rfdetr package):
       uv run --extra rfdetr python backends/rfdetr/export.py --variant base
       (variants: nano|small|base|medium|large, plus xlarge|2xlarge — see LICENSE)

  2. A Hugging Face Transformers RF-DETR repo — a directory (or .safetensors)
     with config.json + model.safetensors (model_type "rf_detr"). Exported via
     `transformers`; native resolution / class names / background slot are read
     from the repo's config (id2label, image_size):
       uv run --extra rfdetr python backends/rfdetr/export.py \
           --checkpoint /path/to/rf-detr-small            # directory
       uv run --extra rfdetr python backends/rfdetr/export.py \
           --checkpoint /path/to/rf-detr-small/model.safetensors

  3. A native rfdetr checkpoint (.pth written by rfdetr training; auto-detects
     variant/classes/resolution via RFDETR.from_checkpoint):
       uv run --extra rfdetr python backends/rfdetr/export.py \
           --checkpoint /path/to/my_rfdetr.pth

The gateway serves the fixed native square resolution, ONNX outputs `dets`
[batch, N, 4] normalized cxcywh + `labels` [batch, N, num_slots] raw logits;
the Rust gateway applies per-class sigmoid, drops the checkpoint's background
slot (--bg-slot, default auto-detected) and maps boxes to original pixels.
See https://rfdetr.roboflow.com/latest/exports/onnx/.

LICENSE: the rfdetr package and the N/S/M/B/L weights are Apache-2.0. The
XLarge / 2XLarge weights are NOT open source (Roboflow Platform Model License);
serving them commercially — including staking a Pocket Network supplier —
requires a license from Roboflow. You are responsible for that, and for the
license of any custom checkpoint. See backends/rfdetr/README.md.
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import common  # noqa: E402
from coco import COCO80, COCO91  # noqa: E402

VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    # Not open source (Roboflow Platform Model License): bring your own license.
    "xlarge": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
}
RESTRICTED_VARIANTS = {"xlarge", "2xlarge"}

_BG_LABELS = {"n/a", "no_object", "no object", "background", "_blank_", "null"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=sorted(VARIANTS), default=None,
                   help="official pretrained tier (with no --checkpoint), or architecture hint for a .pth")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="HF repo dir containing config.json+model.safetensors, a .safetensors file, or a native .pth")
    p.add_argument("--class-names", nargs="*", default=None,
                   help="explicit space-separated class names (index==slot)")
    p.add_argument("--name", default=None, help="Triton model name (default: rfdetr-<variant|checkpoint stem>)")
    p.add_argument("--repo", type=Path, default=ROOT / "triton" / "model_repository")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--queue-delay-us", type=int, default=4000)
    p.add_argument("--bg-slot", default="auto",
                   help="background logit slot: auto | none (sparse COCO) | -1 (contiguous custom) | <int>")
    p.add_argument("--output-dir", type=Path, default=ROOT / "triton" / "exports")
    return p.parse_args()


# --- format detection -------------------------------------------------------

def _as_hf_dir(checkpoint: Path) -> Path | None:
    """Return the directory if the checkpoint is HF-format, else None."""
    d = checkpoint if checkpoint.is_dir() else checkpoint.parent
    cfg = d / "config.json"
    if not cfg.exists():
        return None
    try:
        j = json.loads(cfg.read_text())
    except Exception:
        return None
    arch = j.get("architectures") or []
    if j.get("model_type") == "rf_detr" or any("RfDetr" in a for a in arch):
        if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
            return d
    return None


def _variant_class(rfdetr, variant: str):
    cls = getattr(rfdetr, VARIANTS[variant], None)
    if cls is None:
        raise SystemExit(
            f"this rfdetr version has no {VARIANTS[variant]} (variant '{variant}'); "
            "upgrade rfdetr or install the package that ships it"
        )
    return cls


def load_native(args):
    """Path 1/3: rfdetr package (pretrained tier or native .pth checkpoint)."""
    import rfdetr

    if args.checkpoint:
        if args.variant:
            model = _variant_class(rfdetr, args.variant)(pretrain_weights=str(args.checkpoint))
        else:
            model = rfdetr.RFDETR.from_checkpoint(str(args.checkpoint))
    else:
        model = _variant_class(rfdetr, args.variant)()
    return model


def load_hf(checkpoint_dir: Path):
    """Path 2: HF transformers RfDetrForObjectDetection."""
    from transformers import RfDetrForObjectDetection

    model = RfDetrForObjectDetection.from_pretrained(str(checkpoint_dir))
    model.eval()
    return model


def export_native_onnx(model, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for kw in (
        {"output_dir": str(out_dir), "simplify": True, "dynamic_batch": True},
        {"output_dir": str(out_dir), "simplify": True},
        {"output_dir": str(out_dir)},
    ):
        try:
            res = model.export(**kw)
        except TypeError:
            continue
        p = Path(res) if res else None
        if p and p.exists():
            return p
        cands = sorted(out_dir.glob("**/*.onnx"))
        if cands:
            return cands[0]
    raise SystemExit("rfdetr model.export() produced no ONNX file")


def export_hf_onnx(model, imgsz: int, out_path: Path):
    """Wrap HF forward to (pred_boxes, logits) and export ONNX at native res.

    The HF port only interpolates positional embeddings through
    `aten::_upsample_bicubic2d_aa`, which ONNX export (any opset) cannot map.
    Served at the model's trained square resolution the interpolation is a
    mathematical no-op (num_patches == num_positions), so we pin the eager
    constant before tracing — bit-exact, no unsupported op in the graph.
    """
    import torch
    from transformers.models.rf_detr.modeling_rf_detr import RfDetrDinov2Embeddings

    emb_mods = [m for m in model.modules() if isinstance(m, RfDetrDinov2Embeddings)]
    for emb in emb_mods:
        num_positions = emb.position_embeddings.shape[1] - 1
        num_patches = int(getattr(emb.patch_embeddings, "num_patches", 0) or 0)
        if num_patches and num_positions != num_patches:
            raise SystemExit(
                f"checkpoint position embeddings cover {num_positions} patches but the "
                f"{imgsz}x{imgsz} input yields {num_patches}; export needs the model's "
                "trained resolution. Re-convert the checkpoint with rfdetr instead."
            )
        emb.interpolate_pos_encoding = lambda embeddings, height, width, _pe=emb.position_embeddings: _pe

    class DetOut(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values)
            return o.pred_boxes, o.logits

    wrapped = DetOut(model).eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    # batch is specialized to 1 by the modeling code; Triton serves this repo
    # with max_batch_size 0 + instance_group concurrency (see build_config).
    torch.onnx.export(
        wrapped, (dummy,), str(out_path),
        input_names=["pixel_values"], output_names=["dets", "labels"],
        opset_version=17, dynamo=False,
    )
    return out_path


def hf_metadata(cfg_dir: Path) -> dict:
    cfg = json.loads((cfg_dir / "config.json").read_text())
    id2label = cfg.get("id2label") or {}
    n = len(id2label) or int(cfg.get("num_labels") or 0)
    names = [str(id2label.get(str(i), f"class_{i}")) for i in range(n)]
    imgsz = int(cfg.get("image_size") or (cfg.get("backbone_config") or {}).get("image_size") or 560)
    bg_slot: int | None = None
    if n and names[0].strip().lower() in _BG_LABELS:
        bg_slot = 0
    return {"names": names, "imgsz": imgsz, "bg_slot": bg_slot,
            "queries": int(cfg.get("num_queries") or 300)}


def _resolve_names(args, model, num_slots: int) -> list[str]:
    """Class names indexed by OUTPUT SLOT.

    Official COCO checkpoints have 91 slots where slot == sparse COCO category
    id (1 = person, 15 = bench, ...), while `model.class_names` returns the
    dense 80-name list — using that directly shifts every label. Detect the
    stock COCO names and map them by category id instead.
    """
    if args.class_names is not None:
        return common.dense_names(args.class_names, num_slots)
    names: list[str] | dict[int, str] | None = None
    for attr in ("names", "class_names", "label_names"):
        val = getattr(model, attr, None)
        if isinstance(val, dict):
            names = {int(k): str(v) for k, v in val.items()}
            break
        if isinstance(val, (list, tuple)):
            names = [str(v) for v in val]
            break
    if names is None or (num_slots == 91 and list(names)[:80] == COCO80):
        return list(COCO91)
    return common.dense_names(names, num_slots)


def parse_bg_slot(spec: str, num_slots: int, default_native: int | None) -> int | None:
    if spec == "auto":
        return default_native
    if spec.lower() in ("none", "null"):
        return None
    v = int(spec)
    if not -num_slots <= v < num_slots:
        raise SystemExit(f"--bg-slot {v} outside {num_slots} slots")
    return v


def main() -> None:
    args = parse_args()
    if not args.variant and not args.checkpoint:
        sys.exit("provide --variant (pretrained tier) and/or --checkpoint (path)")
    if args.variant in RESTRICTED_VARIANTS:
        print(f"WARNING: RF-DETR {args.variant} weights are NOT open source (Roboflow Platform "
              "Model License). Commercial serving, including a Pocket Network supplier, "
              "requires a Roboflow license.", file=sys.stderr)

    if args.name:
        name = args.name
    elif args.checkpoint is None:
        name = f"rfdetr-{args.variant}"
    else:
        ck = args.checkpoint
        stem = ck.name if ck.is_dir() else (ck.parent.name if (ck.parent / "config.json").exists() else ck.stem)
        name = f"rfdetr-{stem.removeprefix('rf-detr-')}" if stem.startswith("rf-detr-") else f"rfdetr-{stem}"

    hf_dir = _as_hf_dir(args.checkpoint) if args.checkpoint else None

    if hf_dir is not None:
        print(f"Exporting RF-DETR from HF checkpoint: {hf_dir}")
        meta = hf_metadata(hf_dir)
        onnx_path = export_hf_onnx(load_hf(hf_dir), meta["imgsz"], args.output_dir / f"{name}.onnx")
        info = common.inspect_onnx(onnx_path)
        input_name = info["input"]["name"]
        idims = info["input"]["dims"]
        imgsz = int(idims[2]) if isinstance(idims[2], int) else meta["imgsz"]
        outs = {o["name"]: o for o in info["outputs"]}
        d_name = next((n for n in outs if "dets" in n.lower()), info["outputs"][0]["name"])
        l_name = next((n for n in outs if "labels" in n.lower()), [n for n in outs if n != d_name][0])
        queries = int(outs[d_name]["dims"][-2])
        num_slots = int(outs[l_name]["dims"][-1])
        names_list = args.class_names if args.class_names is not None else meta["names"]
        names_list = common.dense_names(names_list, num_slots)
        bg_slot = meta["bg_slot"] if args.bg_slot == "auto" else parse_bg_slot(args.bg_slot, num_slots, meta["bg_slot"])
        output_names = [d_name, l_name]
        in_elem = info["input"]["elem"]
        out_elems = [outs[d_name]["elem"], outs[l_name]["elem"]]
        # HF export bakes batch=1 (`.shape` unpack in modeling) -> serve no-batch.
        serving_batch = 0
        input_dims = [int(x) for x in idims]
        output_dims = [[1, queries, 4], [1, queries, num_slots]]  # full shape (no implicit batch)
    else:
        import rfdetr  # noqa: F401  (validates the extra is installed)
        model = load_native(args)
        out_dir = args.output_dir / name
        onnx_path = export_native_onnx(model, out_dir)
        info = common.inspect_onnx(onnx_path)
        input_name = info["input"]["name"]
        idims = info["input"]["dims"]
        imgsz = idims[2] if isinstance(idims[2], int) else None
        if not imgsz:
            sys.exit(f"input dims {idims} expose no static square size; re-export without dynamic shapes")
        outs = {o["name"]: o for o in info["outputs"]}
        dets_name = next((n for n in outs if "dets" in n.lower() or "box" in n.lower()), info["outputs"][0]["name"])
        labels_name = next((n for n in outs if ("labels" in n.lower() or "logit" in n.lower()) and n != dets_name),
                           [n for n in outs if n != dets_name][0] if len(info["outputs"]) > 1 else dets_name)
        queries = int(outs[dets_name]["dims"][-2])
        num_slots = int(outs[labels_name]["dims"][-1])
        names_list = _resolve_names(args, model, num_slots)
        # rfdetr-native conventions: official sparse-ID COCO checkpoints keep
        # all 91 slots (a real foreground class occupies the last slot => bg
        # None); contiguous-ID fine-tunes treat the last slot as background (-1).
        default_native = None if num_slots == 91 and args.class_names is None else -1
        bg_slot = parse_bg_slot(args.bg_slot, num_slots, default_native)
        in_elem = info["input"]["elem"]
        out_elems = [outs[dets_name]["elem"], outs[labels_name]["elem"]]
        output_names = [dets_name, labels_name]
        if common.has_dynamic_batch(idims):
            serving_batch = args.max_batch
            input_dims = [3, int(imgsz), int(imgsz)]
            output_dims = [[queries, 4], [queries, num_slots]]
        else:
            # Static batch-1 graph: Triton cannot batch it; serve with
            # explicit full dims + parallel instances instead.
            print(f"note: ONNX input {idims} has a static batch; serving with "
                  "max_batch_size 0 + 4 instances", file=sys.stderr)
            serving_batch = 0
            input_dims = [int(x) for x in idims]
            output_dims = [[1, queries, 4], [1, queries, num_slots]]

    metadata = {
        "family": "rfdetr",
        "task": "detect",
        "names": names_list,
        "imgsz": int(imgsz),
        "bg_slot": bg_slot,
        "input": input_name,
        "outputs": output_names,
        "queries": int(queries),
    }
    cfg = common.build_config_pbtxt(
        model_name=name,
        metadata=metadata,
        input_spec={"name": input_name, "data_type": common.triton_dtype(in_elem), "dims": input_dims},
        output_specs=[
            {"name": output_names[0], "data_type": common.triton_dtype(out_elems[0]), "dims": output_dims[0]},
            {"name": output_names[1], "data_type": common.triton_dtype(out_elems[1]), "dims": output_dims[1]},
        ],
        max_batch_size=serving_batch,
        queue_delay_us=args.queue_delay_us,
        instance_group_count=4 if serving_batch == 0 else None,
    )
    repo = common.write_repo(args.repo, name, onnx_path, cfg)
    print(f"OK: {repo}/config.pbtxt + 1/model.onnx — '{name}' imgsz={imgsz}, "
          f"{queries} queries, {num_slots} slots, names={len(names_list)}, bg_slot={bg_slot}")


if __name__ == "__main__":
    main()
