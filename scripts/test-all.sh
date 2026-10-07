#!/usr/bin/env bash
# Full verification for image-inference-server. No GPU/Triton image required:
# the contract tests use a protocol-accurate ONNX-Runtime mock Triton.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.cargo/bin:${PATH}"

echo "==> Backend fixtures + Triton config schema validation (real model_config.proto)"
uv run --with onnx --with numpy python tools/make_fixtures.py
uv run --with grpcio-tools --with protobuf --with onnx python tools/verify_repo.py

echo "==> Rust: fmt + clippy + tests (contract crate, every backend, gateway e2e)"
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

echo "==> Rust: each backend builds on its own (feature isolation)"
cargo build -q -p gateway --no-default-features --features yolo
cargo build -q -p gateway --no-default-features --features rfdetr

echo "==> Rust release binary + Python OpenAPI contract test (real ONNX, all fixtures)"
cargo build --release -p gateway
uv run --extra test python tests/test_predict_integration.py
uv run --extra test python tests/test_hf_model_matrix.py   # skips if rfdetr-small not exported

echo "All checks passed."
