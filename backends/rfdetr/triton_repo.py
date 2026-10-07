"""RF-DETR backend hooks for the shared tooling (`tools/make_fixtures.py`,
`tools/verify_repo.py`). Needs only `onnx` + `numpy` — no weights, no torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import common  # noqa: E402
from coco import COCO80, COCO91  # noqa: E402

FAMILY = "rfdetr"
FIXTURE_NAME = "rfdetr-base"


def check(conf, meta: dict) -> None:
    """Family-specific config.pbtxt checks (raise AssertionError on failure)."""
    b = [1] if conf.max_batch_size == 0 else []
    assert len(conf.output) == 2, "rfdetr needs dets + labels"
    assert list(conf.output[0].dims) == [*b, meta["queries"], 4], list(conf.output[0].dims)
    assert list(conf.output[1].dims) == [*b, meta["queries"], len(meta["names"])], \
        list(conf.output[1].dims)
    # 91-slot COCO heads index by sparse category id (slot 1 = person); a dense
    # 80-name list here shifts every label (person -> "bicycle", ...).
    if len(meta["names"]) == 91:
        assert meta["names"][:80] != COCO80, (
            "names are the dense 80-class COCO list but the head has 91 sparse-id slots; "
            "re-export with backends/rfdetr/export.py (slot 1 must be 'person')"
        )


def make_fixture(repo: Path, workdir: Path, imgsz: int = 560, queries: int = 300,
                 slots: int = 91) -> str:
    """Constant-output ONNX (dynamic batch): query 0 is a confident 'dog'
    (slot 17 == COCO id)."""
    import numpy as np

    dets = np.array([[[0.5, 0.5, 0.25, 0.125]] * queries], dtype=np.float32)
    labels = np.full((1, queries, slots), -20.0, dtype=np.float32)
    labels[0, 0, 17] = 8.0
    path = workdir / f"{FIXTURE_NAME}.onnx"
    common.batched_constant_onnx(path, "input", imgsz, {"dets": dets, "labels": labels})

    common.write_repo(repo, FIXTURE_NAME, path, common.build_config_pbtxt(
        model_name=FIXTURE_NAME,
        metadata={"family": FAMILY, "task": "detect", "names": COCO91, "imgsz": imgsz,
                  "bg_slot": None, "input": "input", "outputs": ["dets", "labels"],
                  "queries": queries},
        input_spec={"name": "input", "data_type": "TYPE_FP32", "dims": [3, imgsz, imgsz]},
        output_specs=[
            {"name": "dets", "data_type": "TYPE_FP32", "dims": [queries, 4]},
            {"name": "labels", "data_type": "TYPE_FP32", "dims": [queries, slots]},
        ],
    ))
    return FIXTURE_NAME
