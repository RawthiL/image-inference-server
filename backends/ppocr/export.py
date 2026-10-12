#!/usr/bin/env python3
"""Put a PaddleOCR (PP-OCR) detector + recognizer ONNX pair into the Triton repo.

Writes two Triton models, `<name>-det` and `<name>-rec`. The detector lists the
recognizer in its `companions` metadata, so the gateway config only names the
detector:

    endpoints:
      ocr: { model: <name>-det }

Needs only the `onnx` extra (no paddle):

    uv run --extra onnx python backends/ppocr/export.py \
        --det det.onnx --rec rec.onnx --dict ppocr_keys_v1.txt --name ppocr-v4

Getting ONNX files: PP-OCRv6 has official ONNX exports on Hugging Face
(PaddlePaddle/PP-OCRv6_<tier>_{det,rec}_onnx, recommended; see README). Older
versions: convert with paddle2onnx, or use RapidOCR's converted files. The
character dictionary comes from `--dict` (one character per line, or the
recognizer's `inference.yml`), or from the recognizer ONNX `character`
metadata when present (RapidOCR files embed it).

LICENSE: PP-OCR weights are Apache-2.0 (PaddleOCR). See backends/ppocr/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import triton_repo  # noqa: E402


def read_dict(path: Path | None, rec_onnx: Path) -> list[str]:
    """Characters in class order (class k+1 = chars[k]), without the space
    that PaddleOCR appends when `use_space_char` is on."""
    if path is None:
        import onnx

        meta = {p.key: p.value for p in onnx.load(str(rec_onnx)).metadata_props}
        if "character" not in meta:
            sys.exit("--dict is required: the recognizer ONNX has no 'character' metadata")
        return meta["character"].splitlines()
    if path.suffix in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError:
            sys.exit("reading inference.yml needs PyYAML: add `--with pyyaml` to uv run")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        chars = (cfg.get("PostProcess") or {}).get("character_dict")
        if not chars:
            sys.exit(f"{path} has no PostProcess.character_dict")
        return [str(c) for c in chars]
    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det", type=Path, required=True, help="text detection ONNX (DB)")
    p.add_argument("--rec", type=Path, required=True, help="text recognition ONNX (CTC)")
    p.add_argument("--dict", type=Path, default=None,
                   help="character dict (.txt, one per line) or recognizer inference.yml")
    p.add_argument("--no-space-char", action="store_true",
                   help="the recognizer was trained without use_space_char")
    p.add_argument("--name", default="ppocr", help="Triton model name prefix (-det / -rec)")
    p.add_argument("--repo", type=Path, default=ROOT / "triton" / "model_repository")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--queue-delay-us", type=int, default=2000)
    # Detector pre/post-processing (PaddleOCR DetResizeForTest + DBPostProcess).
    p.add_argument("--limit-side-len", type=int, default=960)
    p.add_argument("--limit-type", choices=["max", "min"], default="max")
    p.add_argument("--max-side-limit", type=int, default=4000)
    p.add_argument("--thresh", type=float, default=0.3, help="probability map binarization")
    p.add_argument("--box-thresh", type=float, default=0.6, help="default minimum region score")
    p.add_argument("--unclip-ratio", type=float, default=1.5)
    # Recognizer input.
    p.add_argument("--rec-imgh", type=int, default=48)
    p.add_argument("--rec-imgw", type=int, default=320)
    p.add_argument("--rec-max-width", type=int, default=3200)
    p.add_argument("--rec-batch", type=int, default=6, help="crops per recognizer request")
    args = p.parse_args()

    det, rec = common.inspect_onnx(args.det), common.inspect_onnx(args.rec)
    for label, info in (("detector", det), ("recognizer", rec)):
        if len(info["outputs"]) != 1:
            sys.exit(f"{label} must have exactly one output, got {[o['name'] for o in info['outputs']]}")
        if args.max_batch > 0 and not common.has_dynamic_batch(info["input"]["dims"]):
            sys.exit(f"{label} input {info['input']['dims']} has a static batch dim; "
                     "use --max-batch 0 or re-export with a dynamic batch")

    charset = read_dict(args.dict, args.rec)
    if not args.no_space_char:
        charset.append(" ")
    classes = rec["outputs"][0]["dims"][-1]
    if isinstance(classes, int) and classes != len(charset) + 1:
        sys.exit(f"recognizer has {classes} classes but the dict gives {len(charset)} "
                 f"characters (+1 blank); wrong dict, or toggle --no-space-char")
    classes = classes if isinstance(classes, int) else len(charset) + 1

    det_name, rec_name = triton_repo.write_pair(
        args.repo, args.name, args.det, args.rec,
        det_meta={
            "input": det["input"]["name"], "outputs": [det["outputs"][0]["name"]],
            "limit_side_len": args.limit_side_len, "limit_type": args.limit_type,
            "max_side_limit": args.max_side_limit,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "channel_order": "bgr",
            "thresh": args.thresh, "box_thresh": args.box_thresh,
            "unclip_ratio": args.unclip_ratio, "max_candidates": 1000,
        },
        rec_meta={
            "input": rec["input"]["name"], "outputs": [rec["outputs"][0]["name"]],
            "imgh": args.rec_imgh, "imgw": args.rec_imgw, "max_width": args.rec_max_width,
            "channel_order": "bgr", "batch": args.rec_batch, "charset": charset,
        },
        classes=classes, max_batch=args.max_batch, queue_delay_us=args.queue_delay_us,
    )
    print(f"OK: {args.repo / det_name} + {args.repo / rec_name} "
          f"({len(charset)} characters). Gateway config:\n"
          f"  endpoints:\n    ocr: {{ model: {det_name} }}")


if __name__ == "__main__":
    main()
