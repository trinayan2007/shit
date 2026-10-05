"""Pytest bootstrap: force a temp database + backend import path BEFORE app imports."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
sys.path.insert(0, str(ROOT / "backend"))

# Isolate tests from the real app.db (must happen before app.config import)
_tmp = tempfile.mkdtemp(prefix="itd_test_")
os.environ["DATABASE_PATH"] = str(Path(_tmp) / "test_app.db")
os.environ["AWS_REAL_BLOCKING_ENABLED"] = "false"
os.environ.setdefault(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
)
