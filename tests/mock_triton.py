#!/usr/bin/env python3
"""A faithful in-process Triton HTTP server (v2 + binary tensor extension)
backed by ONNX Runtime. Lets you run the full gateway contract WITHOUT pulling
the multi-GB Triton image — it speaks the exact wire protocol from
docs/protocol/extension_binary_data.md and executes the real ONNX graphs in
triton/model_repository.

Usage:
    python tests/mock_triton.py --repo triton/model_repository --port 8000

Endpoints:
    GET  /v2                            server metadata
    GET  /v2/health/ready
    GET  /v2/models/<name>/config       renders config.pbtxt metadata as JSON
    POST /v2/models/<name>/infer        binary tensor inference (ORT)
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort

_T_DS = re.compile(r"data_type:\s*(TYPE_\w+)")

_DTYPE = {
    "TYPE_FP32": np.float32,
    "TYPE_FP16": np.float16,
    "TYPE_UINT8": np.uint8,
    "TYPE_INT64": np.int64,
}


def parse_config(pbtxt: str) -> dict:
    """Minimal config.pbtxt reader: backend/max_batch/params/ios."""
    mb = re.search(r"max_batch_size:\s*(\d+)", pbtxt)
    meta = None
    mm = re.search(r'key: "metadata"\s*\n\s*value \{\s*string_value: "((?:[^"\\]|\\.)*)"', pbtxt)
    if mm:
        s = mm.group(1).encode().decode("unicode_escape")
        meta = json.loads(s)
    # input/output blocks: name/data_type/dims
    def io_blocks(kind):
        out = []
        for m in re.finditer(kind + r" \[(.*?)^\]", pbtxt, re.S | re.M):
            for blk in re.finditer(r"\{\s*(.*?)\s*\}", m.group(1), re.S):
                t = blk.group(1)
                name = re.search(r'name:\s*"((?:[^"\\]|\\.)*)"', t)
                dt = _T_DS.search(t)
                dims = re.search(r"dims:\s*\[([^\]]*)\]", t)
                out.append({
                    "name": name.group(1),
                    "data_type": dt.group(1) if dt else "TYPE_FP32",
                    "dims": [int(x) for x in (dims.group(1).replace(" ", "").split(",") if dims and dims.group(1).strip() else [])],
                })
        return out

    return {
        "max_batch_size": int(mb.group(1)) if mb else 0,
        "metadata": meta,
        "input": io_blocks("input"),
        "output": io_blocks("output"),
    }


class Model:
    def __init__(self, path: Path):
        self.name = path.name
        self.conf = parse_config((path / "config.pbtxt").read_text())
        self.session = ort.InferenceSession(
            str(path / "1" / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        # Same load-time rule as real Triton: a batching model needs a dynamic
        # first dim on its ONNX inputs.
        if self.conf["max_batch_size"] > 0:
            for i in self.session.get_inputs():
                if not i.shape or isinstance(i.shape[0], int):
                    raise SystemExit(
                        f"model '{self.name}', tensor '{i.name}': for the model to support "
                        f"batching the first dimension must be -1; but shape is {i.shape}"
                    )
        self.dtypes = {o["name"]: _DTYPE.get(o["data_type"], np.float32) for o in self.conf["output"]}

    def config_json(self) -> dict:
        j = {
            "name": self.name,
            "backend": "onnxruntime",
            "max_batch_size": self.conf["max_batch_size"],
            "input": self.conf["input"],
            "output": self.conf["output"],
        }
        if self.conf["metadata"] is not None:
            j["parameters"] = {"metadata": {"string_value": json.dumps(self.conf["metadata"])}}
        return j

    def infer(self, inputs: list[dict], blobs: dict[str, bytes]) -> tuple[list[dict], bytes]:
        feeds = {}
        for i in inputs:
            arr = np.frombuffer(blobs[i["name"]], dtype=np.float32).reshape(i["shape"])
            feeds[i["name"]] = arr
        outs = self.session.run(None, feeds)
        resp_outs, binary = [], b""
        for o, data in zip(self.session.get_outputs(), outs):
            arr = np.asarray(data, dtype=np.float32)
            raw = arr.reshape(-1).tobytes()
            resp_outs.append({
                "name": o.name,
                "shape": [d for d in arr.shape],
                "datatype": "FP32",
                "parameters": {"binary_data_size": len(raw)},
            })
            binary += raw
        return resp_outs, binary


def make_handler(models: dict[str, Model]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            if self.headers.get("X-DEBUG"):
                super().log_message(*a)

        def _send(self, code: int, payload: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/v2", "/v2/"):
                return self._send(200, json.dumps({"name": "triton-mock-onnxruntime", "version": "2.99-mock", "extensions": ["binary_tensor_data"]}).encode())
            if self.path.startswith("/v2/health"):
                return self._send(200, b"")
            m = re.fullmatch(r"/v2/models/([^/]+)/config", self.path)
            if m and m.group(1) in models:
                return self._send(200, json.dumps(models[m.group(1)].config_json()).encode())
            self._send(404, json.dumps({"error": "not found"}).encode())

        def do_POST(self):
            m = re.fullmatch(r"/v2/models/([^/]+)/infer", self.path)
            if not m or m.group(1) not in models:
                return self._send(404, json.dumps({"error": "model not found"}).encode())
            model = models[m.group(1)]
            total = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(total)
            hcl = self.headers.get("Inference-Header-Content-Length")
            if hcl is None:
                return self._send(400, json.dumps({"error": "Inference-Header-Content-Length required"}).encode())
            n = int(hcl)
            header = json.loads(body[:n].rstrip())
            blobs, off = {}, n
            for i in header.get("inputs", []):
                sz = i.get("parameters", {}).get("binary_data_size")
                if sz is None:
                    return self._send(400, json.dumps({"error": "input missing binary_data_size"}).encode())
                blobs[i["name"]] = body[off:off + sz]
                off += sz
            outs, binary = model.infer(header.get("inputs", []), blobs)
            resp = json.dumps({"model_name": model.name, "outputs": outs})
            while len(resp) % 4:
                resp += " "
            payload = resp.encode() + binary
            return self._send(
                200, payload,
                ctype="application/octet-stream",
                extra={"Inference-Header-Content-Length": len(resp),
                       "Content-Type": "application/octet-stream"},
            )

    return Handler


def serve(repo: Path, port: int) -> ThreadingHTTPServer:
    models = {}
    for d in sorted(repo.iterdir()):
        if (d / "config.pbtxt").exists() and (d / "1" / "model.onnx").exists():
            models[d.name] = Model(d)
    if not models:
        raise SystemExit(f"no models found in {repo} (run tools/make_fixtures.py)")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(models))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("triton/model_repository"))
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    httpd = serve(args.repo, args.port)
    print(f"mock Triton (ONNX Runtime) on http://127.0.0.1:{args.port}")
    httpd_thread = threading.Event()
    try:
        httpd_thread.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
