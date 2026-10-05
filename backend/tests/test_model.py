"""Phase 15: model forward pass, loading, anomaly score, threshold tests."""

from __future__ import annotations

import json

import numpy as np
import torch

from app.config import ANOMALY_THRESHOLD, MODEL_CONFIG, MODEL_WEIGHTS
from app.ml.gru_autoencoder import (
    ModelConfig,
    build_model,
    masked_mse_per_step,
    window_scores,
)
from app.ml.predict import load_model, load_threshold


def test_forward_pass_shape():
    cfg = ModelConfig(feature_dim=138, dense_hidden=16, gru_hidden=16)
    model = build_model(cfg)
    x = torch.randn(4, 32, 138)
    out = model(x)
    assert out.shape == (4, 32, 138)
    assert torch.isfinite(out).all()


def test_model_is_genuinely_recurrent_and_trainable():
    """Overfit check on COMPRESSIBLE structured sequences: loss must drop.

    Random per-step noise is information-theoretically hard for any bottleneck
    autoencoder, so we use smooth sequences (offset + slow trend + noise) that a
    GRU-AE must compress through its latent - a proper test that gradients flow
    through both the recurrent and autoencoder parts.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(feature_dim=8, dense_hidden=16, gru_hidden=32)
    model = build_model(cfg)
    t = torch.linspace(0, 6.28, 12).unsqueeze(0).unsqueeze(-1)  # (1,12,1)
    base = torch.sin(t) + 0.3 * torch.cos(2 * t)                 # (1,12,1)
    scales = torch.linspace(0.5, 3.0, 16).unsqueeze(1).unsqueeze(2)
    offsets = torch.linspace(-2.0, 2.0, 16).unsqueeze(1).unsqueeze(2)
    x = (base * scales + offsets).expand(16, 12, 8) + 0.05 * torch.randn(16, 12, 8)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    mask = torch.ones(16, 12)
    first = None
    for _ in range(80):
        recon = model(x)
        loss = masked_mse_per_step(recon, x, mask).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.item())
    final = float(loss.detach().item())
    assert final < first * 0.5, (
        f"reconstruction loss did not drop ({first} -> {final})"
    )


def test_masked_mse_ignores_padded_steps():
    recon = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    target[0, 3] = 100.0  # huge error on a padded step
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    per_step = masked_mse_per_step(recon, target, mask)
    assert per_step[0, 3] == 0.0  # masked out
    scores = window_scores(recon, target, mask)
    assert scores.item() == 0.0


def test_window_scores_nonzero_when_recon_differs():
    recon = torch.zeros(2, 8, 4)
    target = torch.ones(2, 8, 4)
    mask = torch.ones(2, 8)
    scores = window_scores(recon, target, mask)
    assert scores.shape == (2,)
    assert torch.allclose(scores, torch.ones(2), atol=1e-6)


def test_saved_model_artifacts_exist_and_load():
    assert MODEL_WEIGHTS.exists(), "run scripts/train.py first"
    assert MODEL_CONFIG.exists()
    assert ANOMALY_THRESHOLD.exists()
    cfg = json.loads(MODEL_CONFIG.read_text())
    assert cfg["feature_dim"] == 138
    model = load_model()
    x = torch.randn(2, 32, cfg["feature_dim"])
    out = model(x)
    assert out.shape == x.shape


def test_threshold_is_calibrated_not_arbitrary():
    """Threshold must come from the train error distribution, not 0.5/0.7/0.8."""
    th = load_threshold()
    data = json.loads(ANOMALY_THRESHOLD.read_text())
    assert th == data["threshold"]
    assert th not in (0.5, 0.7, 0.8), "threshold looks arbitrary"
    assert data["quantile"] == 0.99
    stats = data["train_score_stats"]
    assert stats["p95"] < th <= stats["max"]
    assert data["train_score_stats"]["count"] > 1000


def test_inference_scores_match_saved_artifacts():
    from app.config import ARTIFACTS_DIR, VAL_WINDOWS_NPY
    from app.ml.predict import score_windows

    saved = ARTIFACTS_DIR / "window_scores_val.npy"
    if not saved.exists():
        return
    model = load_model()
    scores, _ = score_windows(model, VAL_WINDOWS_NPY, collect_step_errors=False)
    expected = np.load(saved)
    assert np.allclose(scores, expected, atol=1e-6), (
        "live scoring diverged from persisted scores"
    )
