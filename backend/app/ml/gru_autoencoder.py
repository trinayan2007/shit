"""Hybrid GRU-Autoencoder for behavioral sequence reconstruction.

Data flow (matches the PPT methodology):

input sequence (B, T, F)
    -> per-step dense projection (feature representation, F -> H)
    -> GRU encoder (captures temporal/sequential behavior)
    -> latent vector (final hidden state)
    -> GRU decoder (reconstructs the temporal structure)
    -> per-step dense reconstruction (H -> F)
    -> reconstructed sequence (B, T, F)
    -> masked per-step squared error
    -> window anomaly score (masked mean squared error)

The GRU genuinely models sequential behavior (recurrent over T) and the
autoencoder genuinely reconstructs the input sequence; the anomaly score is
computed ONLY from the actual reconstruction error - no heuristics, no
hardcoded values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    feature_dim: int
    dense_hidden: int = 64
    gru_hidden: int = 64
    gru_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class GRUEncoderDecoderAE(nn.Module):
    """Sequence autoencoder: dense projection -> GRU enc -> GRU dec -> dense out."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.feature_dim, config.dense_hidden)
        self.encoder = nn.GRU(
            input_size=config.dense_hidden,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        self.decoder = nn.GRU(
            input_size=config.gru_hidden,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        # decoder input at each step = latent broadcast; reconstruct in feature space
        self.out = nn.Sequential(
            nn.Linear(config.gru_hidden, config.dense_hidden),
            nn.ReLU(),
            nn.Linear(config.dense_hidden, config.feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = torch.relu(self.proj(x))          # per-step feature representation (B, T, H)
        _, latent = self.encoder(h)           # latent: (L, B, GH) - temporal compression
        # GRU decoder: feed encoder hidden as input at every step, seeded by latent
        dec_in = latent[-1].unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, GH)
        dec_out, _ = self.decoder(dec_in, latent)                   # (B, T, GH)
        return self.out(dec_out)             # reconstructed sequence (B, T, F)


def masked_mse_per_step(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-step MSE over features, with mask (B, T) zeroing padded steps."""
    diff = (recon - target) ** 2
    per_step = diff.mean(dim=-1)             # (B, T)
    return per_step * mask


def window_scores(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Anomaly score per window: masked mean of per-step reconstruction MSE."""
    per_step = masked_mse_per_step(recon, target, mask)  # (B, T)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return per_step.sum(dim=1) / denom


def build_model(config: ModelConfig) -> GRUEncoderDecoderAE:
    return GRUEncoderDecoderAE(config)
