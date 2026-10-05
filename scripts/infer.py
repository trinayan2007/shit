"""Phase 6 CLI: score windows with the trained model.

Usage: python scripts/infer.py [--split val|train|all]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ml.predict import run_inference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    args = parser.parse_args()
    splits = ["train", "val"] if args.split == "all" else [args.split]
    out = {s: run_inference(split=s, collect_step_errors=True) for s in splits}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
