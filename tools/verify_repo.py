#!/usr/bin/env python3
"""Validate generated config.pbtxt files with the real protobuf text-format
parser (the same mechanism Triton uses), against the official
`model_config.proto`. No Triton install required.

Generic checks run here; family-specific checks are delegated to
`backends/<family>/triton_repo.py:check(conf, meta)`.

    uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def compile_pb2(proto_dir: Path) -> Path:
    out = Path(tempfile.mkdtemp(prefix="modelcfg_pb2_"))
    subprocess.run(
        [sys.executable, "-m", "grpc_tools.protoc", f"-I{proto_dir}",
         "--python_out", str(out), "model_config.proto"],
        check=True, cwd=proto_dir,
    )
    return out


def load_pb2(out_dir: Path):
    spec = importlib.util.spec_from_file_location("model_config_pb2", out_dir / "model_config_pb2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=common.ROOT / "triton" / "model_repository")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    proto_dir = here / "_proto"
    if not (proto_dir / "model_config.proto").exists():
        sys.exit("missing tools/_proto/model_config.proto (download from triton-inference-server/common)")

    mc = load_pb2(compile_pb2(proto_dir))
    backends = common.load_backend_modules()
    from google.protobuf import text_format

    failures = 0
    for cfg_path in sorted(args.repo.glob("*/config.pbtxt")):
        name = cfg_path.parent.name
        try:
            conf = mc.ModelConfig()
            text_format.Parse(cfg_path.read_text(), conf)
            assert conf.backend == "onnxruntime", f"backend={conf.backend}"
            assert conf.max_batch_size >= 0, "max_batch_size"
            assert len(conf.input) >= 1 and len(conf.output) >= 1
            meta_str = conf.parameters["metadata"].string_value
            meta = json.loads(meta_str)
            assert meta["family"] in backends, f"unknown family {meta['family']!r} (have {sorted(backends)})"
            assert meta["task"] == "detect"
            assert meta["names"], "empty names"
            b = [1] if conf.max_batch_size == 0 else []  # no-batch repos carry explicit batch
            expect = [*b, 3, meta["imgsz"], meta["imgsz"]]
            assert list(conf.input[0].dims) == expect, list(conf.input[0].dims)
            assert conf.input[0].name == meta["input"], (conf.input[0].name, meta["input"])
            assert [o.name for o in conf.output] == meta["outputs"], [o.name for o in conf.output]
            backends[meta["family"]].check(conf, meta)
            onnx_file = cfg_path.parent / "1" / "model.onnx"
            if onnx_file.exists():
                common.check_onnx_matches_batching(onnx_file, conf.max_batch_size)
            print(f"OK   {name}: family={meta['family']} imgsz={meta['imgsz']} "
                  f"classes={len(meta['names'])} batch={conf.max_batch_size}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
