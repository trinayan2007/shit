"""Threats endpoints: detected anomalies (MEDIUM/HIGH) + block action."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import clamp_limit, clamp_offset
from app.database.database import get_conn, utcnow
from app.database.models import insert_response
from app.services.response_engine import get_adapter

router = APIRouter(tags=["threats"])

THREAT_WHERE = "d.risk_level != 'LOW'"

BASE_SELECT = """
SELECT d.*, p.display_name, p.principal_key, p.status AS principal_status
FROM detections d
JOIN principals p ON p.id = d.principal_id
"""


def _threat(r, full: bool = False) -> dict:
    d = {
        "id": r["id"],
        "split": r["split"],
        "window_start": r["window_start"],
        "window_end": r["window_end"],
        "anomaly_score": r["anomaly_score"],
        "threshold": r["threshold"],
        "risk_level": r["risk_level"],
        "top_events": json.loads(r["top_events"]),
        "mitre_technique_id": r["mitre_technique_id"],
        "mitre_technique_name": r["mitre_technique_name"],
        "response_action": r["response_action"],
        "response_status": r["response_status"],
        "principal": {
            "id": r["principal_id"],
            "display_name": r["display_name"],
            "principal_key": r["principal_key"],
            "status": r["principal_status"],
        },
    }
    if full:
        peak = json.loads(r["peak_event"] or "{}")
        d.update(
            {
                "peak_event": peak,
                "reasons": json.loads(r["reasons"]),
                "evidence_summary": json.loads(r["evidence_summary"]),
                "mitre": {
                    "status": r["mitre_status"],
                    "technique_id": r["mitre_technique_id"],
                    "technique_name": r["mitre_technique_name"],
                    "tactic": r["mitre_tactic"],
                    "rationale": r["mitre_rationale"],
                    "supporting_behavior": json.loads(r["mitre_support"] or "[]"),
                },
                "created_at": r["created_at"],
            }
        )
    return d


@router.get("/api/threats")
def list_threats(
    limit: int | None = Query(None, ge=1, le=500),
    offset: int | None = Query(None, ge=0),
    risk: str | None = Query(None, pattern="^(MEDIUM|HIGH)$"),
    sort: str = Query("score", pattern="^(score|time)$"),
) -> dict:
    conn = get_conn()
    where = [THREAT_WHERE]
    params: list = []
    if risk:
        where.append("d.risk_level = ?")
        params.append(risk)
    wsql = " WHERE " + " AND ".join(where)
    total = conn.execute(
        f"SELECT COUNT(*) n FROM detections d{wsql}", params
    ).fetchone()["n"]
    lim, off = clamp_limit(limit), clamp_offset(offset)
    order = (
        "d.anomaly_score DESC, d.id DESC"
        if sort == "score"
        else "d.window_start DESC, d.id DESC"
    )
    rows = conn.execute(
        BASE_SELECT + wsql + f" ORDER BY {order} LIMIT ? OFFSET ?",
        [*params, lim, off],
    ).fetchall()
    return {
        "total": total,
        "limit": lim,
        "offset": off,
        "items": [_threat(r) for r in rows],
    }


@router.get("/api/threats/{threat_id}")
def get_threat(threat_id: int) -> dict:
    conn = get_conn()
    r = conn.execute(BASE_SELECT + " WHERE d.id = ?", (threat_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Threat not found")
    responses = [
        dict(x)
        for x in conn.execute(
            "SELECT id, action, mode, status, detail, executed_at "
            "FROM responses WHERE detection_id = ? ORDER BY id DESC",
            (threat_id,),
        ).fetchall()
    ]
    out = _threat(r, full=True)
    out["responses"] = responses
    return out


@router.post("/api/threats/{threat_id}/block")
def block_threat(threat_id: int) -> dict:
    """Execute the response action for a HIGH threat (SIMULATED by default)."""
    conn = get_conn()
    r = conn.execute(BASE_SELECT + " WHERE d.id = ?", (threat_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Threat not found")
    if r["risk_level"] != "HIGH":
        raise HTTPException(
            status_code=400,
            detail="Blocking is only allowed for HIGH risk threats",
        )
    if r["response_status"] == "EXECUTED":
        raise HTTPException(status_code=409, detail="Response already executed")

    adapter = get_adapter()
    result = adapter.execute_block(
        r["principal_key"],
        {"threat_id": threat_id, "score": r["anomaly_score"]},
    )

    # persist: mark detection executed, flip principal status, log response
    conn.execute(
        "UPDATE detections SET response_status = ?, response_action = ? WHERE id = ?",
        (result.status, result.action, threat_id),
    )
    conn.execute(
        "UPDATE principals SET status = 'BLOCKED' WHERE id = ?", (r["principal_id"],)
    )
    conn.commit()
    insert_response(threat_id, result.action, result.mode, result.status, result.detail)

    return {
        "threat_id": threat_id,
        "response_executed": True,
        "mode": result.mode,
        "simulated": result.mode == "SIMULATED",
        "label": "SIMULATED RESPONSE" if result.mode == "SIMULATED" else "REAL RESPONSE",
        "action": result.action,
        "status": result.status,
        "detail": result.detail,
        "executed_at": result.executed_at or utcnow(),
        "principal_status": "BLOCKED",
    }
