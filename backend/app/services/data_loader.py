"""Chunked loading of the raw CloudTrail CSV.

The raw dataset is ~1 GB, so nothing here ever calls read_csv() on the whole
file. Every consumer iterates over chunks with an explicit column subset.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from app.config import DATASET_FILE

CHUNK_ROWS = 250_000

# Columns actually used downstream (verified against the real header).
RAW_COLUMNS = [
    "eventID",
    "eventTime",
    "sourceIPAddress",
    "userAgent",
    "eventName",
    "eventSource",
    "awsRegion",
    "eventVersion",
    "userIdentitytype",
    "eventType",
    "requestID",
    "userIdentityaccountId",
    "userIdentityprincipalId",
    "userIdentityarn",
    "userIdentityaccessKeyId",
    "userIdentityuserName",
    "errorCode",
    "errorMessage",
    "requestParametersinstanceType",
]


def iter_raw_chunks(
    path: Path | None = None,
    columns: list[str] | None = None,
    chunksize: int = CHUNK_ROWS,
) -> Iterator[pd.DataFrame]:
    """Yield raw dataset chunks as string frames (no type inference)."""
    path = path or DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Extract archive.zip into data/ first."
        )
    cols = columns or RAW_COLUMNS
    yield from pd.read_csv(
        path,
        usecols=cols,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def count_rows(path: Path | None = None) -> int:
    """Count data rows without loading the file into memory."""
    path = path or DATASET_FILE
    total = 0
    for chunk in iter_raw_chunks(path, columns=["eventID"]):
        total += len(chunk)
    return total
