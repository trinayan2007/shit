"""Phase 2 CLI: raw CloudTrail CSV -> artifacts/clean_events.csv + stats.

Usage: python scripts/preprocess.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.preprocessing import run_preprocess  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    stats = run_preprocess()
    print(json.dumps({k: stats[k] for k in (
        "rows_in", "duplicate_event_ids_dropped", "invalid_timestamps_dropped",
        "rows_out", "unique_principals", "time_min", "time_max",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
