"""Phase 0: profile both CloudTrail CSVs without loading them fully into memory."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNK = 200_000


def profile(path: Path) -> dict:
    print(f"\n{'=' * 70}\nPROFILE: {path.name}\n{'=' * 70}", flush=True)
    n = 0
    empty: Counter[str] = Counter()
    distinct: dict[str, set] = {}
    track = [
        "userIdentitytype",
        "eventType",
        "awsRegion",
        "eventVersion",
        "errorCode",
        "userIdentityaccountId",
    ]
    top: dict[str, Counter] = {
        c: Counter()
        for c in [
            "eventSource",
            "eventName",
            "sourceIPAddress",
            "userAgent",
            "userIdentityuserName",
            "userIdentityarn",
            "userIdentityprincipalId",
            "userIdentityaccessKeyId",
            "requestParametersinstanceType",
            "eventType",
            "userIdentitytype",
            "errorCode",
        ]
    }
    tmin, tmax = None, None
    id_set: set[str] = set()

    for df in pd.read_csv(path, chunksize=CHUNK, dtype=str, keep_default_na=False, na_filter=False):
        n += len(df)
        for c in df.columns:
            empty[c] += (df[c].str.strip() == "").sum()
        for c in track:
            if c not in distinct:
                distinct[c] = set()
            if len(distinct[c]) < 10_000:
                distinct[c].update(df[c].unique())
        for c in top:
            top[c].update(df[c])
        times = df["eventTime"]
        t = times[times != ""]
        if len(t):
            lo, hi = t.min(), t.max()
            tmin = lo if tmin is None else min(tmin, lo)
            tmax = hi if tmax is None else max(tmax, hi)
        id_set.update(df["eventID"])

    print(f"rows: {n:,}")
    print(f"eventTime range: {tmin} .. {tmax}")
    print("\n-- empty/blank counts per column --")
    for c, v in empty.items():
        print(f"  {c:45s} {v:>10,} ({100 * v / n:5.2f}%)")
    print("\n-- distinct values (capped 10k) --")
    for c, s in distinct.items():
        print(f"  {c:45s} {len(s):>10,}")
    print("\n-- top values --")
    for c, counter in top.items():
        print(f"  [{c}]")
        for val, cnt in counter.most_common(8):
            shown = val if val else "<EMPTY>"
            print(f"      {shown[:80]:80s} {cnt:>10,}")
    return {"rows": n, "ids": id_set, "tmin": tmin, "tmax": tmax}


def main() -> None:
    files = [DATA_DIR / "dec12_18features.csv", DATA_DIR / "nineteenFeaturesDf.csv"]
    results = {}
    for f in files:
        results[f.name] = profile(f)

    a = results["dec12_18features.csv"]
    b = results["nineteenFeaturesDf.csv"]
    print(f"\n{'=' * 70}\nCOMPARISON\n{'=' * 70}")
    print(f"rows            dec12={a['rows']:,}  nineteen={b['rows']:,}")
    print(f"time range      dec12={a['tmin']}..{a['tmax']}   nineteen={b['tmin']}..{b['tmax']}")
    only_a = len(a["ids"] - b["ids"])
    only_b = len(b["ids"] - a["ids"])
    both = len(a["ids"] & b["ids"])
    print(f"eventID overlap shared={both:,}  only_dec12={only_a:,}  only_nineteen={only_b:,}")


if __name__ == "__main__":
    sys.exit(main())
