"""Inference: score all windows with a trained model, persist results.

Saves artifacts/window_scores_{split}.npy (window anomaly scores) and
artifacts/step_errors_{split}.npy (per-step reconstruction error used for
evidence/peak-event selection downstream).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from app.config import (
    ANOMALY_THRESHOLD,
    ARTIFACTS_DIR,
    MODEL_CONFIG,
    MODEL_WEIGHTS,
)
from app.ml.dataset import make_loader
from app.ml.gru_autoencoder import ModelConfig, build_model, masked_mse_per_step

logger = logging.getLogger(__name__)


def load_model(device: str = "cpu") -> torch.nn.Module:
    """Load trained weights + config. Raises FileNotFoundError if never trained."""
    if not MODEL_WEIGHTS.exists() or not MODEL_CONFIG.exists():
        raise FileNotFoundError(
            f"Model artifacts missing ({MODEL_WEIGHTS}). Run `python scripts/train.py` first."
        )
    cfg = ModelConfig(**json.loads(MODEL_CONFIG.read_text()))
    model = build_model(cfg)
    state = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_threshold() -> float:
    if not ANOMALY_THRESHOLD.exists():
        raise FileNotFoundError(
            f"{ANOMALY_THRESHOLD} missing. Run `python scripts/train.py` first."
        )
    return float(json.loads(ANOMALY_THRESHOLD.read_text())["threshold"])


def score_windows(
    model: torch.nn.Module,
    windows_path: Path,
    batch_size: int = 256,
    collect_step_errors: bool = False,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Score every window. Returns (window_scores, optional step_errors).

    window_scores: (W,) masked mean squared reconstruction error.
    step_errors:   (W, T) per-step error (kept for evidence display).
    """
    loader = make_loader(windows_path, batch_size, shuffle=False)
    all_scores: list[np.ndarray] = []
    all_steps: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, mask, _ in loader:
            x = x.to(device)
            mask = mask.to(device)
            recon = model(x)
            per_step = masked_mse_per_step(recon, x, mask)  # (B, T)
            denom = mask.sum(dim=1).clamp(min=1.0)
            scores = per_step.sum(dim=1) / denom
            all_scores.append(scores.cpu().numpy())
            if collect_step_errors:
                all_steps.append(per_step.cpu().numpy())
    scores_arr = np.concatenate(all_scores) if all_scores else np.zeros(0, np.float32)
    steps_arr = (
        np.concatenate(all_steps) if collect_step_errors and all_steps else None
    )
    return scores_arr, steps_arr


def run_inference(split: str = "val", collect_step_errors: bool = True) -> dict:
    """Score train or val windows and save artifacts/window_scores_{split}.npy."""
    from app.config import TRAIN_WINDOWS_NPY, VAL_WINDOWS_NPY

    windows_path = TRAIN_WINDOWS_NPY if split == "train" else VAL_WINDOWS_NPY
    model = load_model()
    scores, steps = score_windows(
        model, windows_path, collect_step_errors=collect_step_errors
    )
    out = ARTIFACTS_DIR / f"window_scores_{split}.npy"
    np.save(out, scores)
    step_out = ARTIFACTS_DIR / f"step_errors_{split}.npy"
    threshold = load_threshold()
    if steps is not None:
        np.save(step_out, steps.astype(np.float32))
    summary = {
        "split": split,
        "windows_scored": int(scores.size),
        "threshold": threshold,
        "flagged": int((scores > threshold).sum()),
        "flagged_pct": round(100.0 * float((scores > threshold).mean()), 3),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "step_errors_saved": bool(steps is not None),
        "output": str(out),
        "step_output": str(step_out) if steps is not None else None,
    }
    (ARTIFACTS_DIR / f"inference_{split}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("inference[%s]: %s", split, summary)
    return summary
