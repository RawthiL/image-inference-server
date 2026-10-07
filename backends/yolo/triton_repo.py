"""YOLO backend hooks for the shared tooling (`tools/make_fixtures.py`,
`tools/verify_repo.py`). Needs only `onnx` + `numpy` — no weights, no torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import common  # noqa: E402
from coco import COCO80  # noqa: E402

FAMILY = "yolo"
FIXTURE_NAME = "yolo26n"


def check(conf, meta: dict) -> None:
    """Family-specific config.pbtxt checks (raise AssertionError on failure)."""
    b = [1] if conf.max_batch_size == 0 else []
    assert meta.get("end2end") is True, "only end-to-end (NMS-free) heads are served"
    assert len(conf.output) == 1, "yolo has exactly one output"
    assert list(conf.output[0].dims) == [*b, meta["queries"], 6], list(conf.output[0].dims)


def make_fixture(repo: Path, workdir: Path, imgsz: int = 640, queries: int = 300) -> str:
    """Constant-output ONNX (dynamic batch): one 'dog' row in letterbox
    space, rest padding."""
    import numpy as np

    rows = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]] * queries], dtype=np.float32)
    rows[0, 0] = [100.0, 140.0, 320.0, 500.0, 0.87, 17.0]
    path = workdir / f"{FIXTURE_NAME}.onnx"
    common.batched_constant_onnx(path, "images", imgsz, {"output0": rows})

    common.write_repo(repo, FIXTURE_NAME, path, common.build_config_pbtxt(
        model_name=FIXTURE_NAME,
        metadata={"family": FAMILY, "task": "detect", "names": COCO80, "imgsz": imgsz,
                  "stride": 32, "end2end": True, "input": "images", "outputs": ["output0"],
                  "queries": queries},
        input_spec={"name": "images", "data_type": "TYPE_FP32", "dims": [3, imgsz, imgsz]},
        output_specs=[{"name": "output0", "data_type": "TYPE_FP32", "dims": [queries, 6]}],
    ))
    return FIXTURE_NAME
