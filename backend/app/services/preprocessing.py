"""Preprocessing: dedup, timestamp parsing, identity resolution.

Turns the raw CloudTrail CSV into `artifacts/clean_events.csv` with:
- eventID deduplication (keep first occurrence)
- strict ISO-8601 eventTime parsing (+ epoch seconds)
- a stable `principal_key` built from CloudTrail identity fields with
  documented fallbacks (arn -> principalId -> accountId -> sourceIP)
- a human-readable `display_name`
- normalized accountId (strips the '.0' float artifact)

All processing is chunked; the raw file is never fully loaded.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from app.config import CLEAN_EVENTS_CSV, DATASET_FILE, PREPROCESS_STATS, SEQUENCE_LENGTH
from app.services.data_loader import iter_raw_chunks

logger = logging.getLogger(__name__)

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

CLEAN_COLUMNS = [
    "eventID",
    "eventTime",
    "epoch",
    "principal_key",
    "display_name",
    "eventName",
    "eventSource",
    "awsRegion",
    "userIdentitytype",
    "eventType",
    "sourceIPAddress",
    "userAgent",
    "errorCode",
    "requestParametersinstanceType",
    "userIdentityaccountId",
    "userIdentityarn",
    "userIdentityprincipalId",
    "userIdentityuserName",
]


def normalize_account_id(value: str) -> str:
    """Strip pandas float artifacts: '811596193553.0' -> '811596193553'."""
    v = (value or "").strip()
    if v.endswith(".0") and v[:-2].isdigit():
        return v[:-2]
    return v


def resolve_principal(row: dict[str, str]) -> str:
    """Most defensible identity key from actual CloudTrail identity fields.

    Priority: arn > principalId > normalized accountId > sourceIPAddress.
    userName alone is NOT used - it is empty for root/assumed-role/service events.
    """
    arn = (row.get("userIdentityarn") or "").strip()
    if arn:
        return f"arn:{arn}"
    pid = (row.get("userIdentityprincipalId") or "").strip()
    if pid:
        return f"pid:{pid}"
    acct = normalize_account_id(row.get("userIdentityaccountId", ""))
    if acct:
        return f"acct:{acct}"
    return f"ip:{(row.get('sourceIPAddress') or '').strip()}"


def derive_display_name(row: dict[str, str], principal_key: str) -> str:
    """Human-readable label: userName, else last ARN segment, else key fragment."""
    user = (row.get("userIdentityuserName") or "").strip()
    if user and user != "HIDDEN_DUE_TO_SECURITY_REASONS":
        return user
    arn = (row.get("userIdentityarn") or "").strip()
    if arn:
        tail = arn.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if tail == "root":
            acct = normalize_account_id(row.get("userIdentityaccountId", ""))
            return f"root@{acct}" if acct else "root"
        if tail:
            return tail
    if principal_key.startswith("acct:"):
        return f"account {principal_key[5:]}"
    if principal_key.startswith("ip:"):
        return principal_key
    return principal_key


def _process_chunk(chunk: pd.DataFrame, seen: set[str]) -> tuple[pd.DataFrame, dict]:
    """Dedup + parse one raw chunk. Returns clean frame + chunk stats."""
    stats = {
        "rows_in": len(chunk),
        "dup_dropped": 0,
        "bad_time_dropped": 0,
    }
    # dedup on eventID: within-chunk duplicates AND against the global seen-set
    # (keep first occurrence everywhere)
    dup_in_chunk = chunk["eventID"].duplicated()
    mask = ~dup_in_chunk & ~chunk["eventID"].isin(seen)
    stats["dup_dropped"] = int((~mask).sum())
    chunk = chunk[mask]
    for eid in chunk["eventID"]:
        seen.add(eid)

    # strict timestamp parse
    times = chunk["eventTime"]
    valid = times.str.match(ISO_RE)
    stats["bad_time_dropped"] = int((~valid).sum())
    chunk = chunk[valid]
    if chunk.empty:
        return pd.DataFrame(columns=CLEAN_COLUMNS), stats

    # unit-independent seconds-since-epoch (pandas 2.x may parse to us, not ns)
    dt = pd.to_datetime(chunk["eventTime"], format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    epoch = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")

    rows = chunk.to_dict(orient="records")
    principal_keys = [resolve_principal(r) for r in rows]
    display_names = [derive_display_name(r, pk) for r, pk in zip(rows, principal_keys)]

    out = pd.DataFrame(
        {
            "eventID": chunk["eventID"].values,
            "eventTime": chunk["eventTime"].values,
            "epoch": epoch.values,
            "principal_key": principal_keys,
            "display_name": display_names,
            "eventName": chunk["eventName"].values,
            "eventSource": chunk["eventSource"].values,
            "awsRegion": chunk["awsRegion"].values,
            "userIdentitytype": chunk["userIdentitytype"].values,
            "eventType": chunk["eventType"].values,
            "sourceIPAddress": chunk["sourceIPAddress"].values,
            "userAgent": chunk["userAgent"].values,
            "errorCode": chunk["errorCode"].values,
            "requestParametersinstanceType": chunk["requestParametersinstanceType"].values,
            "userIdentityaccountId": chunk["userIdentityaccountId"].values,
            "userIdentityarn": chunk["userIdentityarn"].values,
            "userIdentityprincipalId": chunk["userIdentityprincipalId"].values,
            "userIdentityuserName": chunk["userIdentityuserName"].values,
        }
    )
    return out, stats


def run_preprocess(
    raw_path: Path | None = None,
    out_path: Path | None = None,
    stats_path: Path | None = None,
) -> dict:
    """Stream the raw CSV to a clean deduplicated CSV + statistics JSON."""
    out_path = out_path or CLEAN_EVENTS_CSV
    stats_path = stats_path or PREPROCESS_STATS
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    totals = defaultdict(int)
    principals: Counter[str] = Counter()
    display_names: dict[str, str] = {}
    event_names: Counter[str] = Counter()
    event_sources: Counter[str] = Counter()
    identity_types: Counter[str] = Counter()
    error_codes: Counter[str] = Counter()
    daily_events: Counter[str] = Counter()
    t_min, t_max = None, None
    first = True

    for chunk in iter_raw_chunks(raw_path):
        clean, cstats = _process_chunk(chunk, seen)
        totals["rows_in"] += cstats["rows_in"]
        totals["dup_dropped"] += cstats["dup_dropped"]
        totals["bad_time_dropped"] += cstats["bad_time_dropped"]
        totals["rows_out"] += len(clean)
        if clean.empty:
            continue

        clean.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False

        principals.update(clean["principal_key"])
        for pk, dn in zip(clean["principal_key"], clean["display_name"]):
            display_names.setdefault(pk, dn)
        event_names.update(clean["eventName"])
        event_sources.update(clean["eventSource"])
        identity_types.update(clean["userIdentitytype"])
        error_codes.update(clean.loc[clean["errorCode"] != "", "errorCode"])
        daily_events.update(clean["eventTime"].str[:10])

        times = clean["eventTime"]
        t_min = times.min() if t_min is None else min(t_min, times.min())
        t_max = times.max() if t_max is None else max(t_max, times.max())

        logger.info("processed chunk -> total rows_out=%d", totals["rows_out"])

    stats = {
        "source_file": str(raw_path or DATASET_FILE),
        "rows_in": totals["rows_in"],
        "duplicate_event_ids_dropped": totals["dup_dropped"],
        "invalid_timestamps_dropped": totals["bad_time_dropped"],
        "rows_out": totals["rows_out"],
        "unique_principals": len(principals),
        "time_min": t_min,
        "time_max": t_max,
        "distinct_event_names": len(event_names),
        "distinct_event_sources": len(event_sources),
        "distinct_identity_types": len(identity_types),
        "distinct_error_codes": len(error_codes),
        "top_principals": principals.most_common(20),
        "top_event_names": event_names.most_common(30),
        "top_event_sources": event_sources.most_common(20),
        "identity_types": dict(identity_types),
        "daily_event_counts": dict(sorted(daily_events.items())),
        "sequence_length": SEQUENCE_LENGTH,
        "principal_display_names": display_names,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("preprocess done: %s", {k: stats[k] for k in ("rows_in", "rows_out", "unique_principals")})
    return stats


def load_clean_events(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned events CSV (small column subset, string dtypes)."""
    if not CLEAN_EVENTS_CSV.exists():
        raise FileNotFoundError(
            f"{CLEAN_EVENTS_CSV} missing - run `python scripts/preprocess.py` first."
        )
    usecols = columns or CLEAN_COLUMNS
    return pd.read_csv(CLEAN_EVENTS_CSV, usecols=usecols, dtype=str, keep_default_na=False)
