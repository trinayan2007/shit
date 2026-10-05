"""System health + model status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.deps import model_status
from app.database.database import get_conn, get_meta

router = APIRouter(tags=["system"])


@router.get("/api/health")
def health() -> dict:
    conn = get_conn()
    n_det = conn.execute("SELECT COUNT(*) n FROM detections").fetchone()["n"]
    n_prin = conn.execute("SELECT COUNT(*) n FROM principals").fetchone()["n"]
    model = model_status()
    dataset = get_meta("dataset") or {}
    return {
        "status": "online",
        "version": __version__,
        "backend_status": "ONLINE",
        "model_status": "LOADED" if model["loaded"] else "NOT_LOADED",
        "model": model,
        "database": {
            "detections": n_det,
            "principals": n_prin,
        },
        "dataset": dataset,
        "mode": "SIMULATED_RESPONSE (real AWS blocking disabled)",
    }
