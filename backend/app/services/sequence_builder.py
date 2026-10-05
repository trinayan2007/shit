"""Behavioral sequence (window) construction.

After feature engineering, rows are sorted by (split, principal, epoch).
Each contiguous run of one principal inside one split becomes sliding windows:
  - length SEQUENCE_LENGTH (32), stride SEQUENCE_STRIDE (16)
  - a final tail window is added so the last events are covered
  - runs shorter than SEQUENCE_LENGTH yield ONE zero-padded window; padding
    points at the dedicated zero row in features.npy and is masked out of
    the loss and from the anomaly score.

Train/val windows are built strictly within their own temporal split, so no
window crosses the temporal boundary (leakage-safe). Events are never shuffled.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from app.config import (
    ARTIFACTS_DIR,
    FEATURES_META,
    FEATURES_NPY,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    TRAIN_WINDOWS_NPY,
    VAL_WINDOWS_NPY,
)

logger = logging.getLogger(__name__)


def find_runs(split: np.ndarray, principal: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (start, end, split_flag) of contiguous (split, principal) runs."""
    runs: list[tuple[int, int, int]] = []
    start = 0
    n = len(split)
    for i in range(1, n + 1):
        if i == n or split[i] != split[start] or principal[i] != principal[start]:
            runs.append((start, i, int(split[start])))
            start = i
    return runs


def windows_for_run(
    start: int,
    end: int,
    pad_row: int,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = SEQUENCE_STRIDE,
) -> list[np.ndarray]:
    """Build (possibly padded) index windows for one run of rows."""
    length = end - start
    if length == 0:
        return []
    if length < seq_len:
        w = np.full(seq_len, pad_row, dtype=np.int32)
        w[:length] = np.arange(start, end, dtype=np.int32)
        return [w]
    starts = list(range(0, length - seq_len + 1, stride))
    tail = length - seq_len
    if starts[-1] != tail:
        starts.append(tail)
    return [
        np.arange(start + s, start + s + seq_len, dtype=np.int32) for s in starts
    ]


def build_windows() -> dict:
    """Build train/val window index arrays. Returns summary stats."""
    meta = np.load(FEATURES_META, allow_pickle=True)
    split = meta["split"]
    principal = meta["principal_codes"]
    pad_row = int(meta["pad_row_index"])
    n = int(meta["n_rows"])

    # sanity: features file must contain the pad row
    feat = np.load(FEATURES_NPY, mmap_mode="r")
    if feat.shape[0] != n + 1:
        raise ValueError(f"features shape {feat.shape} inconsistent with meta n={n}")

    train_windows: list[np.ndarray] = []
    val_windows: list[np.ndarray] = []
    short_runs = 0

    for start, end, split_flag in find_runs(split, principal):
        ws = windows_for_run(start, end, pad_row)
        if end - start < SEQUENCE_LENGTH:
            short_runs += 1
        (train_windows if split_flag == 0 else val_windows).extend(ws)

    train_arr = np.vstack(train_windows) if train_windows else np.zeros(
        (0, SEQUENCE_LENGTH), dtype=np.int32
    )
    val_arr = np.vstack(val_windows) if val_windows else np.zeros(
        (0, SEQUENCE_LENGTH), dtype=np.int32
    )
    np.save(TRAIN_WINDOWS_NPY, train_arr)
    np.save(VAL_WINDOWS_NPY, val_arr)

    stats = {
        "train_windows": int(len(train_arr)),
        "val_windows": int(len(val_arr)),
        "total_windows": int(len(train_arr) + len(val_arr)),
        "sequence_length": SEQUENCE_LENGTH,
        "stride": SEQUENCE_STRIDE,
        "pad_row_index": pad_row,
        "padded_short_runs": short_runs,
        "coverage_note": "windows built within temporal splits; events never shuffled",
    }
    (ARTIFACTS_DIR / "sequence_report.json").write_text(json.dumps(stats, indent=2))
    logger.info("windows built: %s", stats)
    return stats
