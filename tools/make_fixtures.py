#!/usr/bin/env python3
"""Generate constant-tensor ONNX + Triton repo fixtures for every backend
(no torch / ultralytics / rfdetr needed). Each `backends/<family>/triton_repo.py`
provides `make_fixture(repo, workdir)`. Real weights go through
`backends/<family>/export.py`.

    uv run --extra onnx python tools/make_fixtures.py [--repo triton/model_repository]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=common.ROOT / "triton" / "model_repository")
    p.add_argument("--only", nargs="*", default=None, help="limit to these families")
    args = p.parse_args()

    backends = common.load_backend_modules()
    with tempfile.TemporaryDirectory(prefix="fixtures-") as tmp:
        for family, mod in backends.items():
            if args.only and family not in args.only:
                continue
            name = mod.make_fixture(args.repo, Path(tmp))
            print(f"fixture [{family}] -> {args.repo / name}")


if __name__ == "__main__":
    main()
