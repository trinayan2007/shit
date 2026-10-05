"""Central configuration: paths, env vars, model/feature constants.

All paths resolve relative to the project root so the app runs from any cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # .env is optional; the app must run without it
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except Exception:  # pragma: no cover - dotenv optional
    pass

# backend/app/config.py -> project root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
CONFIG_DIR = PROJECT_ROOT / "config"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", PROJECT_ROOT / "backend" / "data" / "app.db"))

# The selected dataset (see artifacts/dataset_assessment.md for why)
DATASET_FILE = DATA_DIR / "nineteenFeaturesDf.csv"
DATASET_FILE_LEGACY = DATA_DIR / "dec12_18features.csv"

# Derived artifacts
CLEAN_EVENTS_CSV = ARTIFACTS_DIR / "clean_events.csv"
FEATURES_NPY = ARTIFACTS_DIR / "features.npy"
FEATURES_META = ARTIFACTS_DIR / "features_meta.npz"
PREPROCESS_STATS = ARTIFACTS_DIR / "preprocess_stats.json"
TRAIN_WINDOWS_NPY = ARTIFACTS_DIR / "train_windows.npy"
VAL_WINDOWS_NPY = ARTIFACTS_DIR / "val_windows.npy"
STEP_ERRORS_NPY = ARTIFACTS_DIR / "step_errors.npy"

# Model artifacts
MODEL_WEIGHTS = MODELS_DIR / "gru_autoencoder.pt"
MODEL_CONFIG = MODELS_DIR / "model_config.json"
PREPROCESSING_ARTIFACTS = MODELS_DIR / "preprocessing_artifacts.json"
ANOMALY_THRESHOLD = MODELS_DIR / "anomaly_threshold.json"

MITRE_MAPPING_FILE = CONFIG_DIR / "mitre_mapping.json"

# Sequence / feature defaults (must match preprocessing_artifacts.json at inference)
SEQUENCE_LENGTH = 32
SEQUENCE_STRIDE = 16
TRAIN_FRACTION = 0.8  # temporal split quantile

# Response engine safety switch: real AWS calls are OFF unless explicitly enabled.
AWS_REAL_BLOCKING_ENABLED = os.getenv("AWS_REAL_BLOCKING_ENABLED", "false").lower() == "true"

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
