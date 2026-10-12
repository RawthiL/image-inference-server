#!/usr/bin/env python3
"""OCR contract integration test (Cloud Vision `images:annotate`).

Boots the *real* compiled gateway against the ONNX Runtime mock Triton with
the PP-OCR fixture pair (detector: "text" = bright pixels; recognizer: always
"HI YOU") and checks the Cloud Vision response shape, the region-only
extension, and that a detect-only model can't be served on the OCR endpoint.

If a real PP-OCR export named `ppocr` is in the repository
(`backends/ppocr/export.py --name ppocr`), it also OCRs a rendered text image
with it; otherwise that part is skipped.

Run:
    # build the gateway first: cargo build --release -p gateway
    uv run --extra test python tests/test_vision_integration.py
"""

from __future__ import annotations

import base64
import io
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
REPO = ROOT / "triton" / "model_repository"
API_KEY = "integration-test-key"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def png(draw) -> bytes:
    from PIL import Image

    im = Image.new("RGB", (640, 240), "black")
    draw(im)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def annotate(port: int, image: bytes, *features: str) -> requests.Response:
    body = {"requests": [{
        "image": {"content": base64.b64encode(image).decode()},
        "features": [{"type": f} for f in features],
    }]}
    return requests.post(f"http://127.0.0.1:{port}/v1/images:annotate",
                         headers={"x-goog-api-key": API_KEY}, json=body, timeout=60)


class Gateway:
    def __init__(self, mock_port: int, tmp: Path, model: str):
        self.port = free_port()
        cfg = tmp / f"gw-{self.port}.yaml"
        cfg.write_text(f"""
server: {{ listen: "127.0.0.1:{self.port}" }}
triton: {{ url: "http://127.0.0.1:{mock_port}", timeout_ms: 30000 }}
endpoints:
  ocr: {{ model: "{model}", api_keys: ["{API_KEY}"] }}
""")
        self.proc = subprocess.Popen([str(GATEWAY_BIN), "--config", str(cfg)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    def wait(self, want: int = 200) -> dict:
        for _ in range(100):
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                if r.status_code == want and (want == 200 or "error" in r.json()["endpoints"]["ocr"]):
                    return r.json()
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                sys.exit(f"gateway exited: {self.proc.stderr.read()}")
            time.sleep(0.1)
        raise AssertionError(f"gateway /health never returned {want}")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)


def fixture_case(mock_port: int, tmp: Path) -> None:
    gw = Gateway(mock_port, tmp, "ppocr-fixture-det")
    try:
        gw.wait()
        line = png(lambda im: im.paste((255, 255, 255), (60, 100, 400, 140)))

        r = annotate(gw.port, line, "TEXT_DETECTION")
        assert r.status_code == 200, r.text
        resp = r.json()["responses"][0]
        ann = resp["textAnnotations"]
        assert [a["description"] for a in ann] == ["HI YOU\n", "HI", "YOU"], ann
        full = resp["fullTextAnnotation"]
        assert full["text"] == "HI YOU\n"
        page = full["pages"][0]
        assert (page["width"], page["height"]) == (640, 240)
        for a in ann:
            assert len(a["boundingPoly"]["vertices"]) == 4
            for v in a["boundingPoly"]["vertices"]:
                assert 0 <= v["x"] <= 640 and 0 <= v["y"] <= 240, v

        r = annotate(gw.port, line, "TEXT_REGION_DETECTION")
        resp = r.json()["responses"][0]
        assert "textAnnotations" not in resp, resp
        assert len(resp["textRegionAnnotations"]) == 1, resp

        r = annotate(gw.port, line, "FACE_DETECTION")
        assert r.status_code == 200 and r.json()["responses"][0]["error"]["code"] == 3
        print("  ppocr-fixture: Cloud Vision contract OK (text, regions, errors)")
    finally:
        gw.stop()

    # A detect model on the OCR endpoint never becomes ready.
    gw = Gateway(mock_port, tmp, "yolo26n")
    try:
        health = gw.wait(want=503)
        assert "is a 'detect' model" in health["endpoints"]["ocr"]["error"], health
        print("  task mismatch refused: OK")
    finally:
        gw.stop()


def real_model_case(mock_port: int, tmp: Path) -> None:
    if not (REPO / "ppocr-det" / "config.pbtxt").exists():
        print("  real PP-OCR model: skipped (export one with backends/ppocr/export.py --name ppocr)")
        return
    from PIL import ImageDraw, ImageFont

    def draw(im):
        im.paste((255, 255, 255), (0, 0, 640, 240))
        d = ImageDraw.Draw(im)
        d.text((30, 40), "Total: $50.00", fill="black", font=ImageFont.load_default(size=40))
        d.text((30, 140), "2026-10-12", fill="black", font=ImageFont.load_default(size=40))

    gw = Gateway(mock_port, tmp, "ppocr-det")
    try:
        gw.wait()
        r = annotate(gw.port, png(draw), "DOCUMENT_TEXT_DETECTION")
        assert r.status_code == 200, r.text
        text = r.json()["responses"][0]["textAnnotations"][0]["description"]
        assert "50.00" in text and "2026-10-12" in text, text
        print(f"  real PP-OCR model: read {text!r}")
    finally:
        gw.stop()


def main():
    if not GATEWAY_BIN.exists():
        sys.exit(f"gateway binary not found at {GATEWAY_BIN}; run: cargo build --release -p gateway")
    if not (REPO / "ppocr-fixture-det" / "config.pbtxt").exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_fixtures.py"),
                        "--repo", str(REPO)], check=True)

    tmp = Path(tempfile.mkdtemp(prefix="gw-vision-"))
    mock_port = free_port()
    mock = subprocess.Popen(
        [sys.executable, str(HERE / "mock_triton.py"), "--repo", str(REPO), "--port", str(mock_port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 60
        while True:
            if mock.poll() is not None:
                sys.exit("mock Triton exited during startup")
            try:
                requests.get(f"http://127.0.0.1:{mock_port}/v2/health/ready", timeout=1).raise_for_status()
                break
            except requests.RequestException:
                if time.time() > deadline:
                    sys.exit("mock Triton did not become ready")
                time.sleep(0.2)
        fixture_case(mock_port, tmp)
        real_model_case(mock_port, tmp)
        print("OK: Cloud Vision images:annotate contract.")
    finally:
        mock.terminate()


if __name__ == "__main__":
    main()
