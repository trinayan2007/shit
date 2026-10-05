"""ORM-lite row helpers for the SQLite tables (kept in one place)."""

from __future__ import annotations

import json
import sqlite3

from app.database.database import get_conn, utcnow


def upsert_principal(
    principal_key: str,
    display_name: str,
    identity_type: str,
    event_count: int,
    first_seen: str,
    last_seen: str,
) -> int:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO principals(principal_key, display_name, identity_type,
                               event_count, status, first_seen, last_seen)
        VALUES(?,?,?,?,'ACTIVE',?,?)
        ON CONFLICT(principal_key) DO UPDATE SET
            display_name=excluded.display_name,
            identity_type=excluded.identity_type,
            event_count=excluded.event_count,
            first_seen=MIN(principals.first_seen, excluded.first_seen),
            last_seen=MAX(principals.last_seen, excluded.last_seen)
        """,
        (principal_key, display_name, identity_type, event_count, first_seen, last_seen),
    )
    row = conn.execute(
        "SELECT id FROM principals WHERE principal_key=?", (principal_key,)
    ).fetchone()
    conn.commit()
    return int(row["id"])


def insert_detection(d: dict) -> int:
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO detections(
            window_index, split, principal_id, window_start, window_end,
            anomaly_score, threshold, risk_level, reasons, evidence_summary,
            top_events, mitre_status, mitre_technique_id, mitre_technique_name,
            mitre_tactic, mitre_rationale, mitre_support, peak_event,
            response_action, response_status, created_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(split, window_index) DO UPDATE SET
            anomaly_score=excluded.anomaly_score,
            risk_level=excluded.risk_level,
            mitre_technique_id=excluded.mitre_technique_id
        """,
        (
            d["window_index"], d["split"], d["principal_id"],
            d["window_start"], d["window_end"],
            d["anomaly_score"], d["threshold"], d["risk_level"],
            json.dumps(d["reasons"]), json.dumps(d["evidence_summary"]),
            json.dumps(d["top_events"]),
            d["mitre_status"], d.get("mitre_technique_id"),
            d.get("mitre_technique_name"), d.get("mitre_tactic"),
            d.get("mitre_rationale"), json.dumps(d.get("mitre_support", [])),
            json.dumps(d.get("peak_event", {})),
            d["response_action"], d["response_status"], utcnow(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_response(detection_id: int, action: str, mode: str, status: str, detail: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO responses(detection_id, action, mode, status, detail, executed_at) "
        "VALUES(?,?,?,?,?,?)",
        (detection_id, action, mode, status, detail, utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}
