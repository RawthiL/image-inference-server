#!/usr/bin/env python3
"""Contract integration test.

Boots the *real* compiled Rust gateway binary against a *real* ONNX Runtime
mock Triton that speaks the exact Triton HTTP/binary protocol, then posts the
canonical Ultralytics snippet (`requests.post(url, data=args, files={"file": f})`)
for BOTH model families (YOLO e2e and RF-DETR) and validates each response
against the vendored Ultralytics `PredictResponse` OpenAPI schema.

Run:
    # build the gateway first: cargo build --release -p gateway
    uv run --extra test python tests/test_predict_integration.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GATEWAY_BIN = Path(os.environ.get("GATEWAY_BIN", ROOT / "target" / "release" / "gateway"))
SCHEMA = json.loads((HERE / "openapi_predict_schema.json").read_text())
API_KEY = "integration-test-key"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_image(path: Path, w=640, h=480):
    from PIL import Image
    im = Image.new("RGB", (w, h))
    for x in range(0, w, 8):
        for y in range(0, h, 8):
            im.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, 100))
    im.save(path, "JPEG")


def validate(resp_json, model_name):
    assert resp_json["metadata"]["model"] == model_name
    # structural validation against the real Ultralytics PredictResponse schema
    _validate_against(resp_json, SCHEMA)
    # and semantic shape invariants
    img = resp_json["images"][0]
    h, w = img["shape"]
    assert h > 0 and w > 0
    for r in img["results"]:
        b = r["box"]
        assert set(b) >= {"x1", "y1", "x2", "y2"}
        assert 0 <= r["confidence"] <= 1, r["confidence"]
        assert 0 <= b["x1"] <= b["x2"] <= w, (b, w)
        assert 0 <= b["y1"] <= b["y2"] <= h, (b, h)


def _expect_family_specific(model, body):
    """Both families decode their own head format but produce identical wire
    shape and correctly-mapped boxes. Class *ids* differ by family (YOLO dense
    0..79 vs RF-DETR slot==sparse COCO id), so we assert name==classNames[class]
    consistency plus exact box geometry rather than a hardcoded name."""
    names = body["metadata"]["classNames"]
    r = body["images"][0]["results"]
    assert len(r) == 1, r
    assert r[0]["name"] == names[r[0]["class"]], r[0]
    if model == "yolo26n":
        # fixture row [100,140,320,500,.87,17]; 640x480 letterbox -> r=1, dh=80
        # -> inverse maps to (100,60,320,380).
        assert r[0]["box"] == {"x1": 100.0, "y1": 60.0, "x2": 320.0, "y2": 420.0}, r[0]["box"]
        assert abs(r[0]["confidence"] - 0.87) < 1e-6
    else:
        # fixture query0 cxcywh (0.5,0.5,0.25,0.125) on 640x480 ->
        # (0.375*640,0.4375*480,0.625*640,0.5625*480)
        assert r[0]["box"] == {"x1": 240.0, "y1": 210.0, "x2": 400.0, "y2": 270.0}, r[0]["box"]


def _validate_against(instance, schema, _root=None):
    """Tiny JSON-schema checker (subset used by the vendored fragment)."""
    _root = _root or schema
    if "$ref" in schema:
        ref = schema["$ref"].lstrip("#").split("/")
        node = _root
        for p in ref[1:]:
            node = node[p]
        schema = node
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        assert isinstance(instance, dict), f"expected object, got {type(instance)}"
        for req in schema.get("required", []):
            assert req in instance, f"missing required key '{req}'"
        for k, v in instance.items():
            sub = schema.get("properties", {}).get(k)
            if sub is None and schema.get("additionalProperties") is False:
                assert k in schema.get("properties", {}), f"unexpected key '{k}'"
            elif sub is not None:
                _validate_against(v, sub, _root)
            ap = schema.get("additionalProperties")
            if isinstance(ap, dict):
                _validate_against(v, ap, _root)
    elif t == "array" or "items" in schema:
        assert isinstance(instance, list), "expected array"
        items = schema.get("items")
        if isinstance(items, dict):
            for it in instance:
                _validate_against(it, items, _root)
    elif t == "string":
        assert isinstance(instance, str)
    elif t == "integer":
        assert isinstance(instance, int)
    elif t == "number":
        assert isinstance(instance, (int, float))
    elif t == "boolean":
        assert isinstance(instance, bool)


def call_predict(url, image_path: Path, args):
    with open(image_path, "rb") as f:
        return requests.post(
            url,
            headers={"Authorization": f"Bearer {API_KEY}"},
            data=args,
            files={"file": f},
        )


def run_case(model: str, mock_port: int, gw_proc, gw_port, image: Path):
    url = f"http://127.0.0.1:{gw_port}/predict"
    args = {"conf": 0.25, "iou": 0.7, "imgsz": 640}
    r = call_predict(url, image, args)
    assert r.status_code == 200, f"{model}: HTTP {r.status_code} {r.text}"
    body = r.json()
    validate(body, model)
    _expect_family_specific(model, body)
    print(f"  {model}: {len(body['images'][0]['results'])} detections, "
          f"contract OK (classNames={len(body['metadata'].get('classNames', []))})")
    return body


def main():
    if not GATEWAY_BIN.exists():
        sys.exit(f"gateway binary not found at {GATEWAY_BIN}; run: cargo build --release -p gateway")

    # ensure fixtures exist
    repo = ROOT / "triton" / "model_repository"
    if not (repo / "yolo26n" / "config.pbtxt").exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_fixtures.py"),
                        "--repo", str(repo)], check=True)

    tmp = Path(tempfile.mkdtemp(prefix="gw-integration-"))
    image = tmp / "predict-image.jpg"
    make_image(image)

    mock_port = free_port()
    gw_port = free_port()

    # gateway config (written per model below)
    cfg = tmp / "gw-integration.yaml"

    # start mock triton as a subprocess (real ORT)
    mock_proc = subprocess.Popen(
        [sys.executable, str(HERE / "mock_triton.py"), "--repo", str(repo), "--port", str(mock_port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # wait for the mock to load the ONNX models
        deadline = time.time() + 60
        while True:
            if mock_proc.poll() is not None:
                sys.exit("mock Triton exited during startup")
            try:
                requests.get(f"http://127.0.0.1:{mock_port}/v2/health/ready", timeout=1).raise_for_status()
                break
            except requests.RequestException:
                if time.time() > deadline:
                    sys.exit("mock Triton did not become ready")
                time.sleep(0.2)

        results = {}
        for model in ("yolo26n", "rfdetr-base"):
            cfg.write_text(f"""
server: {{ listen: "127.0.0.1:{gw_port}" }}
triton: {{ url: "http://127.0.0.1:{mock_port}", timeout_ms: 10000 }}
default_model: "{model}"
limits: {{ max_upload_mb: 20, allow_private_urls: true }}
defaults: {{ conf: 0.25, iou: 0.7 }}
api_keys: ["{API_KEY}"]
""")
            gw = subprocess.Popen(
                [str(GATEWAY_BIN), "--config", str(cfg)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            # wait for gateway health
            ok = False
            for _ in range(50):
                try:
                    if requests.get(f"http://127.0.0.1:{gw_port}/health", timeout=2).status_code == 200:
                        ok = True
                        break
                except Exception:
                    pass
                if gw.poll() is not None:
                    sys.exit(f"gateway exited for {model}: {gw.stderr.read()}")
                time.sleep(0.1)
            assert ok, f"gateway never became healthy for {model}"
            results[model] = run_case(model, mock_port, gw, gw_port, image)
            gw.terminate(); gw.wait(timeout=5)

        # identical top-level structure across both families
        assert list(results["yolo26n"].keys()) == list(results["rfdetr-base"].keys())
        assert list(results["yolo26n"]["images"][0].keys()) == list(results["rfdetr-base"]["images"][0].keys())
        print("OK: identical PredictResponse structure for YOLO and RF-DETR, schema-valid.")
    finally:
        mock_proc.terminate()


if __name__ == "__main__":
    main()
