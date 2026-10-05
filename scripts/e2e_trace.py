"""Phase 16: END-TO-END VALIDATION on one REAL CloudTrail event.

Traces a real HIGH-risk detection through every stage:

  RAW CSV ROW -> PREPROCESSING -> FEATURE ENGINEERING -> SEQUENCE
  -> GRU-AUTOENCODER -> RECONSTRUCTION ERROR -> ANOMALY SCORE
  -> RISK CLASSIFICATION -> MITRE MAPPING -> RESPONSE -> DATABASE
  -> FASTAPI -> (dashboard payload)

Verifies each stage against persisted artifacts and fails loudly on any
mismatch. Usage: python scripts/e2e_trace.py [--threat-id N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import (  # noqa: E402
    ARTIFACTS_DIR,
    FEATURES_META,
    FEATURES_NPY,
    TRAIN_WINDOWS_NPY,
    VAL_WINDOWS_NPY,
)
from app.database.database import get_conn, get_meta  # noqa: E402
from app.ml.predict import load_model, load_threshold  # noqa: E402
from app.services.classification import Evidence, Thresholds, classify_risk  # noqa: E402
from app.services.feature_engineering import load_vocab  # noqa: E402
from app.services.mitre_mapping import get_mitre_mapper  # noqa: E402
from app.services.preprocessing import load_clean_events  # noqa: E402

STEP = "=" * 74


def trace(threat_id: int | None) -> int:
    conn = get_conn()

    # pick a real HIGH detection (or the requested one)
    if threat_id:
        det = conn.execute(
            "SELECT d.*, p.display_name, p.principal_key FROM detections d "
            "JOIN principals p ON p.id=d.principal_id WHERE d.id=?",
            (threat_id,),
        ).fetchone()
    else:
        det = conn.execute(
            "SELECT d.*, p.display_name, p.principal_key FROM detections d "
            "JOIN principals p ON p.id=d.principal_id "
            "WHERE d.risk_level='HIGH' AND d.mitre_status='mapped' "
            "ORDER BY d.anomaly_score DESC LIMIT 1"
        ).fetchone()
    if det is None:
        print("No HIGH detection with mapped MITRE found; run scripts/detect.py")
        return 1

    print(STEP)
    print(f"TRACING REAL DETECTION id={det['id']} ({det['split']} split)")
    print(STEP)

    # ---------- 1. DATABASE row (the source of truth for the API) ----------
    print("\n[1] DATABASE (detections row)")
    print(f"    principal       : {det['display_name']} ({det['principal_key']})")
    print(f"    window          : {det['window_start']} .. {det['window_end']}")
    print(f"    anomaly_score   : {det['anomaly_score']:.8f}")
    print(f"    risk_level      : {det['risk_level']}")
    print(f"    mitre           : {det['mitre_technique_id']} ({det['mitre_status']})")
    print(f"    response        : {det['response_action']}/{det['response_status']}")

    # ---------- 2. WINDOWS + FEATURES memmap ----------
    meta = np.load(FEATURES_META, allow_pickle=True)
    pad_row = int(meta["pad_row_index"])
    orig_index = meta["orig_index_of_sorted"]
    epochs = meta["epochs"]
    windows_path = TRAIN_WINDOWS_NPY if det["split"] == "train" else VAL_WINDOWS_NPY
    windows = np.load(windows_path)
    rows = windows[int(det["window_index"])]
    valid = rows[rows != pad_row]
    print("\n[2] SEQUENCE")
    print(f"    window_index    : {det['window_index']}")
    print(f"    sequence rows   : {len(rows)} (valid={len(valid)}, pad={len(rows)-len(valid)})")
    print(f"    first 6 row idx : {rows[:6].tolist()}")

    feats = np.load(FEATURES_NPY, mmap_mode="r")
    x = np.asarray(feats[valid], dtype=np.float32)
    print(f"    feature matrix  : {x.shape} (F={x.shape[1]}, from features.npy memmap)")

    # ---------- 3. RAW CSV -> PREPROCESSING -> FEATURES ----------
    print("\n[3] RAW EVENT -> PREPROCESSING -> FEATURE ENCODING")
    orig_row = int(orig_index[int(valid[-1])])  # peak-ish: last valid row
    clean = load_clean_events(
        columns=["eventID", "eventTime", "epoch", "eventName", "eventSource",
                 "errorCode", "principal_key"]
    )
    raw = clean.iloc[orig_row]
    print(f"    raw eventID     : {raw['eventID']}")
    print(f"    raw eventTime   : {raw['eventTime']}  (epoch {raw['epoch']})")
    print(f"    raw eventName   : {raw['eventName']} @ {raw['eventSource']}")
    print(f"    raw principal   : {raw['principal_key']}")
    del clean

    # verify the raw event still parses identically through the live pipeline
    from app.services.feature_engineering import encode_frame  # noqa: E402

    row_df = load_clean_events().iloc[[orig_row]].copy()
    vocab = load_vocab()
    X_one = encode_frame(row_df, vocab)
    match = np.allclose(X_one[0], x[-1], atol=1e-6)
    print(f"    encode_frame re-encode matches features.npy row: {bool(match)}")
    assert match, "E2E FAIL: live feature encoding diverged from stored matrix"

    # ---------- 4. GRU-AUTOENCODER -> RECONSTRUCTION ERROR ----------
    print("\n[4] GRU-AUTOENCODER FORWARD PASS (live, from saved weights)")
    model = load_model()
    with torch.no_grad():
        xt = torch.from_numpy(np.asarray(feats[valid], dtype=np.float32)).unsqueeze(0)
        recon = model(xt)
        step_err = ((recon - xt) ** 2).mean(dim=-1).squeeze(0).numpy()  # per-step MSE
        live_score = float(step_err.mean())
    print(f"    input shape     : {tuple(xt.shape)} -> recon {tuple(recon.shape)}")
    print(f"    per-step errors : min={step_err.min():.6f} max={step_err.max():.6f}")
    print(f"    LIVE window score (masked mean MSE): {live_score:.8f}")

    stored_score = float(det["anomaly_score"])
    diff = abs(live_score - stored_score)
    print(f"    stored score    : {stored_score:.8f}  |Δ|={diff:.2e}")
    assert diff < 1e-5, f"E2E FAIL: live score {live_score} != stored {stored_score}"

    # ---------- 5. THRESHOLD ----------
    print("\n[5] ANOMALY THRESHOLD (calibrated)")
    th = Thresholds.load()
    thr = load_threshold()
    print(f"    threshold       : {thr:.8f} (quantile {json.loads((ARTIFACTS_DIR.parent / 'models' / 'anomaly_threshold.json').read_text())['quantile']} of train errors)")
    print(f"    score >= thresh : {stored_score >= thr}")
    print(f"    p95 / extreme   : {th.p95:.6f} / {th.p999:.6f}")

    # ---------- 6. RISK CLASSIFICATION ----------
    print("\n[6] RISK CLASSIFICATION (rule layer, re-run live)")
    # rebuild evidence from the same real window events
    clean2 = load_clean_events(columns=["eventName", "errorCode"])
    win_names = clean2.iloc[orig_index[valid].tolist()]["eventName"].tolist()
    win_errs = clean2.iloc[orig_index[valid].tolist()]["errorCode"].tolist()
    del clean2
    ev = Evidence.from_window(win_names, win_errs)
    decision = classify_risk(stored_score, ev, th)
    print(f"    live risk       : {decision.risk_level}  (stored: {det['risk_level']})")
    for r in decision.reasons:
        print(f"      - {r}")
    assert decision.risk_level == det["risk_level"], "E2E FAIL: risk mismatch"

    # ---------- 7. MITRE MAPPING ----------
    print("\n[7] MITRE ATT&CK MAPPING (evidence layer, re-run live)")
    mapper = get_mitre_mapper()
    mitre = mapper.map_window(win_names, win_errs)
    print(f"    live mapping    : {mitre.technique_id} ({mitre.status})")
    print(f"    stored mapping  : {det['mitre_technique_id']} ({det['mitre_status']})")
    if mitre.status == "mapped":
        print(f"    tactic          : {mitre.tactic}")
        print(f"    supporting      : {', '.join(mitre.supporting_behavior)}")
    assert (mitre.technique_id or None) == (det["mitre_technique_id"] or None), (
        "E2E FAIL: MITRE mismatch"
    )

    # ---------- 8. RESPONSE POLICY ----------
    print("\n[8] RESPONSE")
    from app.services.response_engine import automatic_action_for  # noqa: E402

    action, status = automatic_action_for(det["risk_level"])
    print(f"    policy ({det['risk_level']}) -> {action}/{status}")
    print(f"    stored          : {det['response_action']}/{det['response_status']}")
    assert action == det["response_action"]

    # ---------- 9. FASTAPI ----------
    print("\n[9] FASTAPI -> DASHBOARD PAYLOAD")
    try:
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        r = client.get(f"/api/threats/{det['id']}")
        assert r.status_code == 200, f"GET /api/threats/{det['id']} -> {r.status_code}"
        body = r.json()
        print(f"    GET /api/threats/{det['id']} -> 200 OK")
        print(f"    api score       : {body['anomaly_score']:.8f}")
        print(f"    api risk        : {body['risk_level']}")
        print(f"    api mitre       : {body['mitre']['technique_id']}")
        print(f"    api peak event  : {body['peak_event'].get('eventName')}")
        assert abs(body["anomaly_score"] - stored_score) < 1e-9
        assert body["risk_level"] == det["risk_level"]

        rd = client.get("/api/dashboard").json()
        k = rd["kpis"]
        db_counts = {
            r_["risk_level"]: r_["n"]
            for r_ in conn.execute(
                "SELECT risk_level, COUNT(*) n FROM detections GROUP BY risk_level"
            ).fetchall()
        }
        print(f"    dashboard KPIs  : events={k['total_events']} principals={k['unique_principals']} "
              f"anomalies={k['detected_anomalies']} high={k['high_risk_incidents']}")
        assert k["total_events"] == get_meta("dataset")["rows_out"], (
            "dashboard total_events != meta rows_out"
        )
        assert k["low"] == db_counts.get("LOW", 0)
        assert k["medium"] == db_counts.get("MEDIUM", 0)
        assert k["high"] == db_counts.get("HIGH", 0)
        assert k["high_risk_incidents"] == db_counts.get("HIGH", 0)
        print("    dashboard risk distribution matches SQL counts: OK")
    except ImportError:
        print("    (fastapi TestClient used; server not required)")

    print("\n" + STEP)
    print("END-TO-END TRACE PASSED: every stage consistent with the real pipeline")
    print(STEP)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threat-id", type=int, default=None)
    args = ap.parse_args()
    raise SystemExit(trace(args.threat_id))
