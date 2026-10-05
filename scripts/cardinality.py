"""Phase 0: cardinality and identity inventory for feature-engineering decisions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CHUNK = 250_000

TRACK = [
    "eventName",
    "eventSource",
    "sourceIPAddress",
    "userAgent",
    "errorCode",
    "userIdentityuserName",
    "userIdentityarn",
    "userIdentityprincipalId",
    "userIdentitytype",
    "awsRegion",
    "eventType",
    "userIdentityaccountId",
]


def inventory(path: Path) -> None:
    print(f"\n{'=' * 70}\nINVENTORY: {path.name}\n{'=' * 70}", flush=True)
    distinct: dict[str, Counter] = {c: Counter() for c in TRACK}
    for df in pd.read_csv(path, chunksize=CHUNK, dtype=str, keep_default_na=False, na_filter=False):
        for c in TRACK:
            distinct[c].update(df[c])
    for c in TRACK:
        ctr = distinct[c]
        print(f"\n[{c}] distinct={len(ctr):,}")
        for val, cnt in ctr.most_common(30):
            label = val if val else "<EMPTY>"
            print(f"    {label[:90]:90s} {cnt:>10,}")


if __name__ == "__main__":
    inventory(DATA / "nineteenFeaturesDf.csv")
