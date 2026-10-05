"""Events endpoint: paginated detection rows (the dashboard activity table).

Each row is a real scored behavioral window joined with its principal and the
peak-deviation event's CloudTrail fields.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import clamp_limit, clamp_offset
from app.database.database import get_conn

router = APIRouter(tags=["events"])


def _row(r, with_peak: bool = False) -> dict:
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
            "status": r["status"],
        },
    }
    if with_peak:
        d.update(
            {
                "peak_event": json.loads(r["peak_event"] or "{}"),
                "reasons": json.loads(r["reasons"]),
                "evidence_summary": json.loads(r["evidence_summary"]),
                "mitre_tactic": r["mitre_tactic"],
                "mitre_rationale": r["mitre_rationale"],
                "mitre_support": json.loads(r["mitre_support"] or "[]"),
                "mitre_status": r["mitre_status"],
            }
        )
    return d


BASE_SELECT = """
SELECT d.*, p.display_name, p.principal_key, p.status
FROM detections d
JOIN principals p ON p.id = d.principal_id
"""


@router.get("/api/events")
def list_events(
    limit: int | None = Query(None, ge=1, le=500),
    offset: int | None = Query(None, ge=0),
    risk: str | None = Query(None, pattern="^(LOW|MEDIUM|HIGH)$"),
    principal_id: int | None = Query(None, ge=1),
    q: str | None = Query(None, max_length=100),
) -> dict:
    conn = get_conn()
    where, params = [], []
    if risk:
        where.append("d.risk_level = ?")
        params.append(risk)
    if principal_id:
        where.append("d.principal_id = ?")
        params.append(principal_id)
    if q:
        where.append(
            "(p.display_name LIKE ? OR p.principal_key LIKE ? OR d.top_events LIKE ?)"
        )
        like = f"%{q}%"
        params += [like, like, like]
    wsql = (" WHERE " + " AND ".join(where)) if where else ""

    total = conn.execute(
        f"SELECT COUNT(*) n FROM detections d JOIN principals p ON p.id=d.principal_id{wsql}",
        params,
    ).fetchone()["n"]

    lim, off = clamp_limit(limit), clamp_offset(offset)
    rows = conn.execute(
        BASE_SELECT + wsql + " ORDER BY d.window_start DESC LIMIT ? OFFSET ?",
        [*params, lim, off],
    ).fetchall()
    return {
        "total": total,
        "limit": lim,
        "offset": off,
        "items": [_row(r) for r in rows],
    }


@router.get("/api/events/{event_id}")
def get_event(event_id: int) -> dict:
    conn = get_conn()
    r = conn.execute(BASE_SELECT + " WHERE d.id = ?", (event_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return _row(r, with_peak=True)
