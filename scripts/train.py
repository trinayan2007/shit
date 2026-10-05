"""Phase 5+6 CLI: train the hybrid GRU-Autoencoder and calibrate the threshold.

Usage: python scripts/train.py [--epochs N] [--batch-size N] [--lr X]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import FEATURES_META, TRAIN_WINDOWS_NPY, VAL_WINDOWS_NPY  # noqa: E402
from app.ml.train import TrainConfig, train_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    meta = __import__("numpy").load(FEATURES_META, allow_pickle=True)
    feature_dim = int(meta["feature_dim"])

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    result = train_model(feature_dim, TRAIN_WINDOWS_NPY, VAL_WINDOWS_NPY, cfg)
    print(json.dumps(dataclasses.asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
