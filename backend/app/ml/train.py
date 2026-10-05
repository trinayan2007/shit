"""Training loop: fit GRU-Autoencoder, validate, calibrate threshold, persist."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.config import (
    ANOMALY_THRESHOLD,
    MODEL_CONFIG,
    MODEL_WEIGHTS,
    PREPROCESSING_ARTIFACTS,
)
from app.ml.dataset import make_loader
from app.ml.gru_autoencoder import (
    ModelConfig,
    build_model,
    masked_mse_per_step,
    window_scores,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    threshold_quantile: float = 0.99  # calibrated from train error distribution
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainResult:
    train_loss_history: list[float]
    val_loss_history: list[float]
    train_score_stats: dict
    val_score_stats: dict
    threshold: float
    epochs_run: int
    seconds: float


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, np.ndarray]:
    """One pass over the loader. Returns (mean window loss, all window scores)."""
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    scores: list[np.ndarray] = []
    with torch.set_grad_enabled(training):
        for x, mask, _ in loader:
            x = x.to(device)
            mask = mask.to(device)
            recon = model(x)
            per_step = masked_mse_per_step(recon, x, mask)     # (B, T)
            denom = mask.sum(dim=1).clamp(min=1.0)
            loss = (per_step.sum(dim=1) / denom).mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.item()))
            scores.append(window_scores(recon.detach(), x, mask).cpu().numpy())
    all_scores = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, all_scores


def _score_stats(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {}
    qs = np.quantile(scores, [0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "p50": float(qs[0]),
        "p90": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
        "max": float(qs[4]),
    }


def train_model(
    feature_dim: int,
    train_windows_path,
    val_windows_path,
    train_cfg: TrainConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> TrainResult:
    """Train, validate, calibrate the anomaly threshold, save artifacts."""
    cfg = train_cfg or TrainConfig()
    _set_seed(cfg.seed)

    model_cfg = model_cfg or ModelConfig(feature_dim=feature_dim)
    model = build_model(model_cfg).to(cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loader = make_loader(train_windows_path, cfg.batch_size, shuffle=True)
    val_loader = make_loader(val_windows_path, cfg.batch_size, shuffle=False)

    train_history: list[float] = []
    val_history: list[float] = []
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_scores = run_epoch(model, train_loader, optimizer, cfg.device)
        va_loss, va_scores = run_epoch(model, val_loader, None, cfg.device)
        train_history.append(tr_loss)
        val_history.append(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        logger.info(
            "epoch %2d/%d  train_loss=%.6f  val_loss=%.6f  (%.1fs)",
            epoch, cfg.epochs, tr_loss, va_loss, time.time() - t0,
        )

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # final scoring pass with the best model (reproducible threshold calibration)
    _, tr_scores = run_epoch(model, train_loader, None, cfg.device)
    _, va_scores = run_epoch(model, val_loader, None, cfg.device)

    # Threshold calibrated from the TRAINING reconstruction-error distribution.
    # Quantile 0.99 -> the top 1% of reconstruction errors are flagged; this is
    # derived from actual model output, not an arbitrary 0.5/0.7/0.8 constant.
    threshold = float(np.quantile(tr_scores, cfg.threshold_quantile))

    train_stats = _score_stats(tr_scores)
    val_stats = _score_stats(va_scores)
    result = TrainResult(
        train_loss_history=train_history,
        val_loss_history=val_history,
        train_score_stats=train_stats,
        val_score_stats=val_stats,
        threshold=threshold,
        epochs_run=len(train_history),
        seconds=round(time.time() - t0, 1),
    )

    # persist artifacts
    MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS)
    MODEL_CONFIG.write_text(json.dumps(model_cfg.to_dict(), indent=2))
    ANOMALY_THRESHOLD.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "method": (
                    "quantile of TRAINING window reconstruction-error distribution "
                    "(masked mean squared error) produced by the trained model"
                ),
                "quantile": cfg.threshold_quantile,
                "train_score_stats": train_stats,
                "val_score_stats": val_stats,
                "epochs": result.epochs_run,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.lr,
            },
            indent=2,
        )
    )
    PREPROCESSING_ARTIFACTS.write_text(
        json.dumps(
            {
                "model": model_cfg.to_dict(),
                "train_config": {
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "lr": cfg.lr,
                    "seed": cfg.seed,
                },
                "train_loss_history": train_history,
                "val_loss_history": val_history,
                "train_windows": len(train_loader.dataset),  # type: ignore[arg-type]
                "val_windows": len(val_loader.dataset),      # type: ignore[arg-type]
                "train_seconds": result.seconds,
            },
            indent=2,
        )
    )
    logger.info("threshold=%.6f  artifacts saved to %s", threshold, MODEL_WEIGHTS)
    return result
