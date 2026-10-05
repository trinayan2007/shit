"""FastAPI application entrypoint.

Run:  cd backend && uvicorn app.main:app --port 8000
The API only READS artifacts/database produced by the pipeline scripts;
it never retrains the model on request.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import dashboard, events, system, threats, users
from app.config import CORS_ORIGINS
from app.database.database import get_conn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(
    title="Insider Threat Classification and Detection in Cloud-Based Environments",
    description=(
        "Hybrid GRU-Autoencoder anomaly detection over AWS CloudTrail with "
        "MITRE ATT&CK evidence mapping and simulated risk-based response."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(dashboard.router)
app.include_router(events.router)
app.include_router(threats.router)
app.include_router(users.router)


@app.on_event("startup")
def _startup() -> None:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) n FROM detections").fetchone()["n"]
    logger.info("startup: detections in database = %d", n)
    if n == 0:
        logger.warning(
            "Database is empty - run: python scripts/detect.py --split all"
        )
