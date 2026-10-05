"""Phase 7-10 orchestrator: scores -> risk -> MITRE -> response -> SQLite.

Usage: python scripts/detect.py [--split val|train|all]

Reads REAL artifacts produced by earlier phases:
  artifacts/features_meta.npz    window row indices + epochs + principals
  artifacts/window_scores_*.npy  actual model anomaly scores
  artifacts/step_errors_*.npy    actual per-step reconstruction errors
  models/anomaly_threshold.json  calibrated thresholds
  config/mitre_mapping.json      evidence rules (verified technique IDs)

Writes every scored window as a detection row (LOW/MEDIUM/HIGH) plus the
automatic response state from the risk policy. Nothing here fabricates data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import (  # noqa: E402
    ANOMALY_THRESHOLD,
    ARTIFACTS_DIR,
    FEATURES_META,
    PREPROCESS_STATS,
    STEP_ERRORS_NPY,
    TRAIN_WINDOWS_NPY,
    VAL_WINDOWS_NPY,
)
from app.database.database import clear_detections, get_conn, set_meta  # noqa: E402
from app.database.models import insert_detection, upsert_principal  # noqa: E402
from app.services.classification import Evidence, Thresholds, classify_risk  # noqa: E402
from app.services.mitre_mapping import get_mitre_mapper  # noqa: E402
from app.services.preprocessing import load_clean_events  # noqa: E402
from app.services.response_engine import automatic_action_for  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("detect")


def epoch_to_iso(e: int | float) -> str:
    return datetime.fromtimestamp(int(e), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap windows per split")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]

    meta = np.load(FEATURES_META, allow_pickle=True)
    epochs_sorted = meta["epochs"]            # sorted order == features row order
    orig_index = meta["orig_index_of_sorted"]  # sorted row -> clean CSV row
    pad_row = int(meta["pad_row_index"])
    principal_list = [str(p) for p in meta["principal_list"].tolist()]
    principal_codes = meta["principal_codes"]

    th = Thresholds.load()
    mapper = get_mitre_mapper()

    # clean events indexed by ORIGINAL csv row order
    clean = load_clean_events(
        columns=[
            "eventTime", "epoch", "principal_key", "display_name", "eventName",
            "eventSource", "awsRegion", "sourceIPAddress", "errorCode",
            "userIdentitytype",
        ]
    )
    # remap original rows -> sorted row order for direct window indexing
    order = orig_index
    col_event_time = clean["eventTime"].to_numpy()[order]
    col_name = clean["eventName"].to_numpy()[order]
    col_source = clean["eventSource"].to_numpy()[order]
    col_ip = clean["sourceIPAddress"].to_numpy()[order]
    col_err = clean["errorCode"].to_numpy()[order]
    col_region = clean["awsRegion"].to_numpy()[order]
    del clean

    # principal aggregates from sorted rows (event counts, first/last seen)
    uniq, inv = np.unique(principal_codes, return_inverse=True)
    counts = np.bincount(inv)
    first_seen = {}
    last_seen = {}
    for code in uniq:
        mask = principal_codes == code
        evs = epochs_sorted[mask]
        first_seen[int(code)] = epoch_to_iso(evs.min())
        last_seen[int(code)] = epoch_to_iso(evs.max())

    principal_ids: dict[str, int] = {}
    for code in uniq:
        code = int(code)
        pkey = principal_list[code]
        pid = upsert_principal(
            principal_key=pkey,
            display_name=pkey,
            identity_type="",
            event_count=int(counts[code]),
            first_seen=first_seen[code],
            last_seen=last_seen[code],
        )
        principal_ids[pkey] = pid
    logger.info("upserted %d principals", len(principal_ids))

    # enrich display names + identity types from clean events
    clean2 = load_clean_events(
        columns=["principal_key", "display_name", "userIdentitytype"]
    )
    conn = get_conn()
    disp = (
        clean2.groupby("principal_key")
        .agg(
            display_name=("display_name", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
            identity_type=("userIdentitytype", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
            n=("display_name", "size"),
        )
        .reset_index()
    )
    for row in disp.itertuples(index=False):
        conn.execute(
            "UPDATE principals SET display_name=?, identity_type=? WHERE principal_key=?",
            (row.display_name, row.identity_type, row.principal_key),
        )
    conn.commit()
    del clean2

    stats = json.loads(PREPROCESS_STATS.read_text())
    set_meta("dataset", {
        "rows_in": stats["rows_in"],
        "rows_out": stats["rows_out"],
        "duplicate_event_ids_dropped": stats["duplicate_event_ids_dropped"],
        "unique_principals_db": len(principal_ids),
        "time_min": stats["time_min"],
        "time_max": stats["time_max"],
        "distinct_event_names": stats["distinct_event_names"],
        "source_file": stats["source_file"],
    })
    thr_json = json.loads(ANOMALY_THRESHOLD.read_text())
    set_meta("model", {
        "name": "GRU-Autoencoder (hybrid)",
        "weights": "models/gru_autoencoder.pt",
        "threshold": thr_json["threshold"],
        "threshold_method": thr_json["method"],
        "quantile": thr_json["quantile"],
        "epochs": thr_json["epochs"],
        "train_score_stats": thr_json["train_score_stats"],
        "val_score_stats": thr_json["val_score_stats"],
    })
    set_meta("daily_event_counts", stats["daily_event_counts"])

    clear_detections()

    total_written = 0
    for split in splits:
        windows_path = TRAIN_WINDOWS_NPY if split == "train" else VAL_WINDOWS_NPY
        scores_path = ARTIFACTS_DIR / f"window_scores_{split}.npy"
        steps_path = ARTIFACTS_DIR / f"step_errors_{split}.npy"
        if not scores_path.exists():
            raise FileNotFoundError(f"{scores_path} missing - run scripts/infer.py")
        windows = np.load(windows_path)
        scores = np.load(scores_path)
        steps = np.load(steps_path) if steps_path.exists() else None
        assert len(windows) == len(scores), "windows/scores length mismatch"
        if steps is not None:
            assert steps.shape == windows.shape, "step errors shape mismatch"

        n = len(windows) if args.limit == 0 else min(args.limit, len(windows))
        risk_counter = Counter()
        mitre_counter = Counter()

        for wi in range(n):
            rows = windows[wi]
            valid = rows[rows != pad_row]
            if valid.size == 0:
                continue
            names = [col_name[r] for r in valid]
            errs = [col_err[r] for r in valid]

            evidence = Evidence.from_window(names, errs)
            score = float(scores[wi])
            decision = classify_risk(score, evidence, th)
            mitre = mapper.map_window(names, errs)

            # peak deviation step -> representative event for the dashboard row
            if steps is not None:
                step_valid = steps[wi][rows != pad_row]
                peak_local = int(np.argmax(step_valid))
            else:
                peak_local = len(valid) - 1
            peak_row = int(valid[peak_local])

            window_start = epoch_to_iso(epochs_sorted[int(valid[0])])
            window_end = epoch_to_iso(epochs_sorted[int(valid[-1])])
            action, status = automatic_action_for(decision.risk_level)

            top_events = [f"{k} x{v}" for k, v in Counter(names).most_common(3)]
            pkey = principal_list[int(principal_codes[peak_row])]

            insert_detection({
                "window_index": wi,
                "split": split,
                "principal_id": principal_ids[pkey],
                "window_start": window_start,
                "window_end": window_end,
                "anomaly_score": score,
                "threshold": th.threshold,
                "risk_level": decision.risk_level,
                "reasons": decision.reasons,
                "evidence_summary": decision.evidence_summary,
                "top_events": top_events,
                "mitre_status": mitre.status,
                "mitre_technique_id": mitre.technique_id,
                "mitre_technique_name": mitre.technique_name,
                "mitre_tactic": mitre.tactic,
                "mitre_rationale": mitre.rationale,
                "mitre_support": mitre.supporting_behavior,
                "response_action": action,
                "response_status": status,
                "peak_event": {
                    "eventTime": str(col_event_time[peak_row]),
                    "eventName": str(col_name[peak_row]),
                    "eventSource": str(col_source[peak_row]),
                    "sourceIPAddress": str(col_ip[peak_row]),
                    "awsRegion": str(col_region[peak_row]),
                    "errorCode": str(col_err[peak_row]),
                    "peak_step_error": float(steps[wi][rows != pad_row][peak_local])
                    if steps is not None else None,
                },
            })
            risk_counter[decision.risk_level] += 1
            mitre_counter[mitre.technique_id or "no_mapping"] += 1

        set_meta(f"detection_stats_{split}", {
            "windows_scored": int(n),
            "risk_distribution": dict(risk_counter),
            "mitre_top": mitre_counter.most_common(15),
        })
        logger.info("[%s] wrote %d detections: %s", split, n, dict(risk_counter))
        total_written += n

    set_meta("detections_total", total_written)
    logger.info("detection pipeline complete: %d rows", total_written)
    print(json.dumps({
        "detections_written": total_written,
        "train": __import__("app.database.database", fromlist=["get_meta"]).get_meta("detection_stats_train"),
        "val": __import__("app.database.database", fromlist=["get_meta"]).get_meta("detection_stats_val"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
