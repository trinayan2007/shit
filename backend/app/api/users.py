"""Users/principals endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import clamp_limit, clamp_offset
from app.database.database import get_conn

router = APIRouter(tags=["users"])


@router.get("/api/users")
def list_users(
    limit: int | None = Query(None, ge=1, le=500),
    offset: int | None = Query(None, ge=0),
    q: str | None = Query(None, max_length=100),
) -> dict:
    conn = get_conn()
    where, params = [], []
    if q:
        where.append("(display_name LIKE ? OR principal_key LIKE ?)")
        like = f"%{q}%"
        params += [like, like]
    wsql = (" WHERE " + " AND ".join(where)) if where else ""
    total = conn.execute(
        f"SELECT COUNT(*) n FROM principals{wsql}", params
    ).fetchone()["n"]
    lim, off = clamp_limit(limit), clamp_offset(offset)
    rows = conn.execute(
        f"""
        SELECT p.*,
               (SELECT COUNT(*) FROM detections d
                WHERE d.principal_id = p.id) AS detections,
               (SELECT COUNT(*) FROM detections d
                WHERE d.principal_id = p.id
                  AND d.risk_level = 'HIGH') AS high_risk
        FROM principals p
        {wsql}
        ORDER BY p.event_count DESC
        LIMIT ? OFFSET ?
        """,
        [*params, lim, off],
    ).fetchall()
    items = [
        {
            "id": r["id"],
            "principal_key": r["principal_key"],
            "display_name": r["display_name"],
            "identity_type": r["identity_type"],
            "status": r["status"],
            "event_count": r["event_count"],
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "detections": r["detections"],
            "high_risk": r["high_risk"],
        }
        for r in rows
    ]
    return {"total": total, "limit": lim, "offset": off, "items": items}


@router.get("/api/users/{user_id}")
def get_user(user_id: int) -> dict:
    conn = get_conn()
    r = conn.execute("SELECT * FROM principals WHERE id = ?", (user_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="User not found")
    dist = {
        row["risk_level"]: row["n"]
        for row in conn.execute(
            "SELECT risk_level, COUNT(*) n FROM detections "
            "WHERE principal_id = ? GROUP BY risk_level",
            (user_id,),
        ).fetchall()
    }
    mitre = [
        {"technique_id": row["mitre_technique_id"], "count": row["n"]}
        for row in conn.execute(
            "SELECT COALESCE(mitre_technique_id,'none') mitre_technique_id, COUNT(*) n "
            "FROM detections WHERE principal_id = ? AND risk_level != 'LOW' "
            "GROUP BY mitre_technique_id ORDER BY n DESC LIMIT 8",
            (user_id,),
        ).fetchall()
    ]
    top = [
        {
            "id": row["id"],
            "window_start": row["window_start"],
            "anomaly_score": row["anomaly_score"],
            "risk_level": row["risk_level"],
            "mitre_technique_id": row["mitre_technique_id"],
            "response_status": row["response_status"],
            "top_events": json.loads(row["top_events"]),
        }
        for row in conn.execute(
            "SELECT * FROM detections WHERE principal_id = ? AND risk_level != 'LOW' "
            "ORDER BY anomaly_score DESC LIMIT 10",
            (user_id,),
        ).fetchall()
    ]
    return {
        "id": r["id"],
        "principal_key": r["principal_key"],
        "display_name": r["display_name"],
        "identity_type": r["identity_type"],
        "status": r["status"],
        "event_count": r["event_count"],
        "first_seen": r["first_seen"],
        "last_seen": r["last_seen"],
        "risk_distribution": dist,
        "mitre_top": mitre,
        "top_threats": top,
    }
