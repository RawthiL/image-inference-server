"""PP-OCR backend hooks for the shared tooling (`tools/make_fixtures.py`,
`tools/verify_repo.py`). Needs only `onnx` + `numpy` — no weights, no paddle.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import common  # noqa: E402

FAMILY = "ppocr"
FIXTURE_NAME = "ppocr-fixture"
FIXTURE_CHARSET = ["H", "I", " ", "Y", "O", "U"]


def check(conf, meta: dict) -> None:
    """Family-specific config.pbtxt checks (raise AssertionError on failure)."""
    # Without Triton batching the dims carry the (dynamic) batch dim too.
    b = [] if conf.max_batch_size > 0 else list(conf.input[0].dims[:1])
    stage = meta.get("stage")
    assert meta.get("task") == "ocr", "ppocr models are task 'ocr'"
    assert len(conf.output) == 1, "ppocr stages have exactly one output"
    if stage == "det":
        assert meta.get("companions"), "the detector must list its recognizer in 'companions'"
        assert list(conf.input[0].dims) == [*b, 3, -1, -1], list(conf.input[0].dims)
        assert list(conf.output[0].dims)[-3:] == [1, -1, -1], list(conf.output[0].dims)
    elif stage == "rec":
        charset = meta.get("charset") or []
        assert charset, "recognizer needs a non-empty 'charset'"
        assert list(conf.input[0].dims) == [*b, 3, meta["imgh"], -1], list(conf.input[0].dims)
        classes = list(conf.output[0].dims)[-1]
        assert classes in (-1, len(charset) + 1), \
            f"recognizer has {classes} classes, charset {len(charset)} (+1 blank)"
    else:
        raise AssertionError(f"stage must be 'det' or 'rec', got {stage!r}")


def write_pair(repo: Path, name: str, det_onnx: Path, rec_onnx: Path, *, det_meta: dict,
               rec_meta: dict, classes: int, max_batch: int = 8,
               queue_delay_us: int = 2000) -> tuple[str, str]:
    """Write `<name>-det` + `<name>-rec` into the repository. Returns their names."""
    det_name, rec_name = f"{name}-det", f"{name}-rec"
    b = [] if max_batch > 0 else [-1]
    det_meta = {"family": FAMILY, "task": "ocr", "stage": "det", "companions": [rec_name],
                **det_meta}
    rec_meta = {"family": FAMILY, "task": "ocr", "stage": "rec", **rec_meta}
    common.write_repo(repo, det_name, det_onnx, common.build_config_pbtxt(
        model_name=det_name,
        metadata=det_meta,
        input_spec={"name": det_meta["input"], "data_type": "TYPE_FP32", "dims": [*b, 3, -1, -1]},
        output_specs=[{"name": det_meta["outputs"][0], "data_type": "TYPE_FP32",
                       "dims": [*b, 1, -1, -1]}],
        max_batch_size=max_batch, queue_delay_us=queue_delay_us,
    ))
    common.write_repo(repo, rec_name, rec_onnx, common.build_config_pbtxt(
        model_name=rec_name,
        metadata=rec_meta,
        input_spec={"name": rec_meta["input"], "data_type": "TYPE_FP32",
                    "dims": [*b, 3, rec_meta["imgh"], -1]},
        output_specs=[{"name": rec_meta["outputs"][0], "data_type": "TYPE_FP32",
                       "dims": [*b, -1, classes]}],
        max_batch_size=max_batch, queue_delay_us=queue_delay_us,
    ))
    return det_name, rec_name


def make_fixture(repo: Path, workdir: Path) -> str:
    """Detector: "text" wherever the image is bright (sigmoid of the first
    normalized channel). Recognizer: every crop reads "HI YOU"."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    det_path = workdir / f"{FIXTURE_NAME}-det.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["x", "_s", "_e", "_a"], ["_c0"]),
            helper.make_node("Mul", ["_c0", "_k"], ["_z"]),
            helper.make_node("Sigmoid", ["_z"], ["sigmoid_0.tmp_0"]),
        ],
        "ppocr_det_fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, "h", "w"])],
        [helper.make_tensor_value_info("sigmoid_0.tmp_0", TensorProto.FLOAT, ["batch", 1, "h", "w"])],
        [
            numpy_helper.from_array(np.array([0], dtype=np.int64), "_s"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "_e"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "_a"),
            numpy_helper.from_array(np.array([4.0], dtype=np.float32), "_k"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(det_path))

    classes = len(FIXTURE_CHARSET) + 1
    steps = 40
    probs = np.zeros((1, steps, classes), dtype=np.float32)
    probs[0, :, 0] = 0.95
    for i in range(len(FIXTURE_CHARSET)):
        probs[0, 3 * i + 1, 0] = 0.0
        probs[0, 3 * i + 1, i + 1] = 0.95
    rec_path = workdir / f"{FIXTURE_NAME}-rec.onnx"
    common.batched_constant_onnx(rec_path, "x", 48, {"softmax_0.tmp_0": probs},
                                 input_shape=["batch", 3, 48, "w"])

    det_name, _ = write_pair(
        repo, FIXTURE_NAME, det_path, rec_path,
        det_meta={"input": "x", "outputs": ["sigmoid_0.tmp_0"]},
        rec_meta={"input": "x", "outputs": ["softmax_0.tmp_0"], "imgh": 48, "imgw": 320,
                  "charset": FIXTURE_CHARSET},
        classes=classes,
    )
    return det_name
