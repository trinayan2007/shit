"""Phase 3+4 CLI: features then behavioral windows.

Usage: python scripts/build_features.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.feature_engineering import run_feature_pipeline  # noqa: E402
from app.services.sequence_builder import build_windows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    feat = run_feature_pipeline()
    seq = build_windows()
    print(json.dumps({"features": feat, "sequences": seq}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
