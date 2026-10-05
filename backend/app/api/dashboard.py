"""Dashboard aggregate endpoint - all values computed from SQLite."""

from __future__ import annotations

import json

from fastapi import APIRouter

from app.api.deps import model_status
from app.database.database import get_conn, get_meta

router = APIRouter(tags=["dashboard"])


@router.get("/api/dashboard")
def dashboard() -> dict:
    conn = get_conn()
    dataset = get_meta("dataset") or {}
    model = get_meta("model") or {}

    total_events = int(dataset.get("rows_out", 0))
    unique_principals = conn.execute(
        "SELECT COUNT(*) n FROM principals"
    ).fetchone()["n"]
    risk = {
        r["risk_level"]: r["n"]
        for r in conn.execute(
            "SELECT risk_level, COUNT(*) n FROM detections GROUP BY risk_level"
        ).fetchall()
    }
    low = risk.get("LOW", 0)
    medium = risk.get("MEDIUM", 0)
    high = risk.get("HIGH", 0)
    # windows at/above the calibrated anomaly threshold (exact, from scores)
    flagged = conn.execute(
        "SELECT COUNT(*) n FROM detections WHERE anomaly_score >= threshold"
    ).fetchone()["n"]

    # threat timeline: detections per week (real rows)
    timeline = [
        {"period": r["period"], "count": r["n"], "high": r["high"]}
        for r in conn.execute(
            """
            SELECT substr(window_start, 1, 7) AS period,
                   COUNT(*) n,
                   SUM(risk_level = 'HIGH') high
            FROM detections
            GROUP BY period
            ORDER BY period
            """
        ).fetchall()
    ]

    # recent threats: highest-scoring non-LOW detections
    recent = []
    for r in conn.execute(
        """
        SELECT d.id, d.window_start, d.anomaly_score, d.risk_level,
               d.mitre_technique_id, d.mitre_technique_name,
               d.response_action, d.response_status, d.top_events,
               p.display_name
        FROM detections d
        JOIN principals p ON p.id = d.principal_id
        WHERE d.risk_level != 'LOW'
        ORDER BY d.anomaly_score DESC
        LIMIT 8
        """
    ).fetchall():
        item = dict(r)
        item["top_events"] = json.loads(item["top_events"] or "[]")
        recent.append(item)

    # MITRE distribution over detected (non-LOW) threats
    mitre = [
        {"technique_id": r["mitre_technique_id"], "count": r["n"]}
        for r in conn.execute(
            """
            SELECT COALESCE(mitre_technique_id, 'none') AS mitre_technique_id,
                   COUNT(*) n
            FROM detections
            WHERE risk_level != 'LOW'
            GROUP BY mitre_technique_id
            ORDER BY n DESC
            LIMIT 10
            """
        ).fetchall()
    ]

    return {
        "status": {
            "system": "ONLINE",
            "backend": "ONLINE",
            "model": "LOADED" if model_status()["loaded"] else "NOT_LOADED",
            "response_mode": "SIMULATED",
        },
        "kpis": {
            "total_events": total_events,
            "unique_principals": unique_principals,
            "detected_anomalies": flagged,
            "high_risk_incidents": high,
            "low": low,
            "medium": medium,
            "high": high,
            "total_detections": low + medium + high,
        },
        "threshold": model.get("threshold"),
        "threshold_method": model.get("threshold_method"),
        "timeline": timeline,
        "risk_distribution": {"LOW": low, "MEDIUM": medium, "HIGH": high},
        "recent_threats": recent,
        "mitre_distribution": mitre,
    }
