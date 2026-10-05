"""Shared API dependencies: model status, pagination helpers."""

from __future__ import annotations

from app.config import ANOMALY_THRESHOLD, MODEL_CONFIG, MODEL_WEIGHTS


def model_status() -> dict:
    """Report real artifact presence (used by /api/health + dashboard)."""
    weights_ok = MODEL_WEIGHTS.exists()
    config_ok = MODEL_CONFIG.exists()
    threshold_ok = ANOMALY_THRESHOLD.exists()
    return {
        "weights": weights_ok,
        "config": config_ok,
        "threshold": threshold_ok,
        "loaded": weights_ok and config_ok and threshold_ok,
    }


def clamp_limit(limit: int | None, default: int = 50, maximum: int = 500) -> int:
    if limit is None or limit <= 0:
        return default
    return min(limit, maximum)


def clamp_offset(offset: int | None) -> int:
    if offset is None or offset < 0:
        return 0
    return offset
