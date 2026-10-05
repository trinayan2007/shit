"""Phase 0 verification: empties, duplicates, timestamp validity, cardinalities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CHUNK = 250_000


def verify(path: Path) -> None:
    print(f"\n{'=' * 70}\nVERIFY: {path.name}\n{'=' * 70}", flush=True)
    rows = 0
    bad_time = 0
    empty_time = 0
    dup_ids = 0
    parsed_ok = 0
    id_counts: dict[str, int] = {}
    cols_empty: dict[str, int] = {}
    cols_seen: set[str] = set()

    for df in pd.read_csv(path, chunksize=CHUNK, dtype=str, keep_default_na=False, na_filter=False):
        rows += len(df)
        for c in df.columns:
            cols_seen.add(c)
            cols_empty[c] = cols_empty.get(c, 0) + int((df[c].str.strip() == "").sum())
        t = df["eventTime"]
        empty_time += int((t.str.strip() == "").sum())
        non_empty = t[t.str.strip() != ""]
        # strict ISO like 2017-02-12T19:57:06Z
        ok = non_empty.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        parsed_ok += int(ok.sum())
        bad_time += int((~ok).sum())
        if (~ok).any():
            for v in non_empty[~ok].head(5):
                print(f"  BAD TIME SAMPLE: {v!r}")
        for i in df["eventID"]:
            if i in id_counts:
                id_counts[i] += 1
                dup_ids += 1
            else:
                id_counts[i] = 1

    print(f"rows: {rows:,}")
    print(f"eventTime: empty={empty_time:,}  iso_ok={parsed_ok:,}  bad={bad_time:,}")
    print(f"eventID: unique={len(id_counts):,}  duplicate_rows={dup_ids:,}")
    print("empty per column:")
    for c, v in sorted(cols_empty.items(), key=lambda x: -x[1]):
        print(f"  {c:45s} {v:>10,} ({100 * v / rows:5.2f}%)")
    # top duplicate ids
    top_dups = sorted(id_counts.items(), key=lambda x: -x[1])[:5]
    print("most repeated eventIDs:", [(k[:13] + "...", v) for k, v in top_dups])


if __name__ == "__main__":
    verify(DATA / "dec12_18features.csv")
    verify(DATA / "nineteenFeaturesDf.csv")
