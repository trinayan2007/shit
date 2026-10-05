"""SQLite persistence (Phase 10).

Tables (kept deliberately small):
  principals   - one row per principal seen in the dataset (+ block status)
  detections   - one row per scored behavioral window (anomaly, risk, MITRE)
  responses    - response actions taken on detections (simulated by default)
  meta         - key/value run metadata (dataset stats, model, threshold)

The rows are written by the detection pipeline (scripts/detect.py), never
fabricated by the API.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import DATABASE_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS principals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    principal_key TEXT UNIQUE NOT NULL,
    display_name  TEXT NOT NULL,
    identity_type TEXT,
    event_count   INTEGER DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ACTIVE',
    first_seen    TEXT,
    last_seen     TEXT
);
CREATE TABLE IF NOT EXISTS detections (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    window_index       INTEGER NOT NULL,
    split              TEXT NOT NULL,
    principal_id       INTEGER NOT NULL REFERENCES principals(id),
    window_start       TEXT NOT NULL,
    window_end         TEXT NOT NULL,
    anomaly_score      REAL NOT NULL,
    threshold          REAL NOT NULL,
    risk_level         TEXT NOT NULL,
    reasons            TEXT NOT NULL,
    evidence_summary   TEXT NOT NULL,
    top_events         TEXT NOT NULL,
    mitre_status       TEXT NOT NULL,
    mitre_technique_id TEXT,
    mitre_technique_name TEXT,
    mitre_tactic       TEXT,
    mitre_rationale    TEXT,    mitre_support    TEXT,
    peak_event       TEXT,
    response_action    TEXT NOT NULL,
    response_status    TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    UNIQUE (split, window_index)
);
CREATE INDEX IF NOT EXISTS idx_detections_risk ON detections(risk_level);
CREATE INDEX IF NOT EXISTS idx_detections_score ON detections(anomaly_score DESC);
CREATE TABLE IF NOT EXISTS responses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL REFERENCES detections(id),
    action       TEXT NOT NULL,
    mode         TEXT NOT NULL,
    status       TEXT NOT NULL,
    detail       TEXT NOT NULL,
    executed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_detection ON responses(detection_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = db_path or DATABASE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(path), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


def close_conn() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def reset_conn(db_path: Path | None = None) -> None:
    """Testing helper: drop the cached connection (next get_conn uses db_path)."""
    close_conn()
    get_conn(db_path)


def set_meta(key: str, value) -> None:
    conn = get_conn()
    payload = json.dumps(value) if not isinstance(value, str) else value
    with _lock:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, payload),
        )
        conn.commit()


def get_meta(key: str, default=None):
    conn = get_conn()
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    raw = row["value"]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def clear_detections() -> None:
    """Reset detection/response tables for a fresh pipeline run."""
    conn = get_conn()
    with _lock:
        conn.execute("DELETE FROM responses")
        conn.execute("DELETE FROM detections")
        conn.commit()
