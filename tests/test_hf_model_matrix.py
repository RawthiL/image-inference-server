#!/usr/bin/env python3
"""Live matrix against the real exported HF RF-DETR model (rfdetr-small).

Skips gracefully if triton/model_repository/rfdetr-small is absent (it is
generated from an HF checkpoint via backends/rfdetr/export.py). Boots the mock Triton +
the release gateway on ephemeral ports and verifies:

  1. multipart file upload            -> 200 + schema + real detections
  2. urlencoded oversized base64      -> 400 'source too long' (spec cap)
  3. urlencoded small base64 (data-uri) + normalize=true -> 200, coords <=1
  4. urlencoded file field            -> 400 'multipart'
  5. wrong content-type               -> 400

Run: cargo build --release -p gateway
     uv run --extra test python tests/test_hf_model_matrix.py
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GATEWAY = ROOT / "target" / "release" / "gateway"
REPO = ROOT / "triton" / "model_repository"
MODEL = "rfdetr-small"
KEY = "matrix-test-key"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def contract(body: dict, model: str):
    assert set(body) == {"images", "metadata"}, body.keys()
    img = body["images"][0]
    assert set(img) == {"shape", "speed", "results"}, img.keys()
    assert set(img["speed"]) == {"preprocess", "inference", "postprocess"}
    md = body["metadata"]
    assert {"imageCount", "functionTimeAlive", "functionTimeCall", "version"} <= set(md)
    assert md["model"] == model and md["task"] == "detect"
    for r in img["results"]:
        assert set(r) == {"name", "class", "confidence", "box"}, r
        assert set(r["box"]) == {"x1", "y1", "x2", "y2"}


def main() -> None:
    if not (REPO / MODEL / "config.pbtxt").exists():
        print(f"SKIP: {REPO / MODEL} not present (export with backends/rfdetr/export.py first)")
        return
    if not GATEWAY.exists():
        sys.exit(f"gateway binary missing: {GATEWAY}")

    import io
    from PIL import Image

    # a real-ish photo with content (gradients with shapes); also a tiny copy
    tmp = Path(tempfile.mkdtemp(prefix="hf-matrix-"))
    img_path = tmp / "hf-matrix.jpg"
    im = Image.new("RGB", (400, 300), "white")
    for x in range(0, 400, 2):
        for y in range(100, 250, 2):
            im.putpixel((x, y), (90, 140, 70))
    im.save(img_path, "JPEG")
    tiny = io.BytesIO()
    im.resize((48, 36)).save(tiny, "JPEG", quality=50)
    tiny_bytes = tiny.getvalue()

    mock_port, gw_port = free_port(), free_port()
    cfg = tmp / f"hf-matrix-{gw_port}.yaml"
    cfg.write_text(f"""
server: {{ listen: "127.0.0.1:{gw_port}" }}
triton: {{ url: "http://127.0.0.1:{mock_port}", timeout_ms: 30000 }}
default_model: "{MODEL}"
api_keys: ["{KEY}"]
limits: {{ max_upload_mb: 20, allow_private_urls: true }}
defaults: {{ conf: 0.25, iou: 0.7 }}
""")

    mock = subprocess.Popen([sys.executable, str(HERE / "mock_triton.py"),
                             "--repo", str(REPO), "--port", str(mock_port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gw = subprocess.Popen([str(GATEWAY), "--config", str(cfg)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if requests.get(f"http://127.0.0.1:{mock_port}/v2/health/ready", timeout=1).ok \
                   and requests.get(f"http://127.0.0.1:{gw_port}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            sys.exit("servers did not come up")

        url = f"http://127.0.0.1:{gw_port}/predict"
        H = {"Authorization": f"Bearer {KEY}"}

        r = requests.post(url, headers=H, files={"file": open(img_path, "rb")},
                          data={"conf": 0.25, "iou": 0.7, "imgsz": 640}, timeout=60)
        assert r.status_code == 200, r.text
        body = r.json()
        contract(body, MODEL)
        shape = body["images"][0]["shape"]
        assert shape == [300, 400], shape
        print(f"1 multipart file: 200, {len(body['images'][0]['results'])} detections, shape {shape} OK")

        big_b64 = base64.b64encode(img_path.read_bytes()).decode()
        assert len(big_b64) > 4096
        r = requests.post(url, headers=H, data={"source": big_b64}, timeout=30)
        assert r.status_code == 400 and "source too long" in r.json()["error"], r.text
        print("2 urlencoded big base64: 400 'source too long' (OpenAPI 4096 cap) OK")

        small_b64 = base64.b64encode(tiny_bytes).decode()
        assert len(small_b64) <= 4096
        r = requests.post(url, headers=H,
                          data={"conf": 0.25, "normalize": "true",
                                "source": f"data:image/jpeg;base64,{small_b64}"}, timeout=30)
        assert r.status_code == 200, r.text
        body = r.json()
        contract(body, MODEL)
        for d in body["images"][0]["results"]:
            assert all(0.0 <= v <= 1.0 for v in d["box"].values()), d
        print(f"3 urlencoded base64 + normalize: 200, {len(body['images'][0]['results'])} detections OK")

        r = requests.post(url, headers=H, data={"file": "not-multipart"}, timeout=30)
        assert r.status_code == 400 and "multipart" in r.json()["error"], r.text
        print("4 urlencoded 'file' field: 400 'must be sent as multipart' OK")

        r = requests.post(url, headers=H, json={"source": "x"}, timeout=30)
        assert r.status_code == 400, r.text
        print("5 application/json body: 400 OK")

        print("HF MODEL MATRIX PASSED")
    finally:
        gw.terminate()
        mock.terminate()


if __name__ == "__main__":
    main()
