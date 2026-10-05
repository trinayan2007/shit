"""Window dataset + dataloaders over the features memmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.config import FEATURES_META, FEATURES_NPY


class WindowDataset(Dataset):
    """Lazy window reader: gathers (T, F) rows from the memmap on access.

    mask[t] = 0 for padded steps (pad row index), 1 otherwise.
    """

    def __init__(self, windows_path: Path, device: str = "cpu"):
        self.features = np.load(FEATURES_NPY, mmap_mode="r")
        self.windows = np.load(windows_path)
        meta = np.load(FEATURES_META, allow_pickle=True)
        self.pad_row = int(meta["pad_row_index"])
        self.device = device

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self.windows[idx]                       # (T,)
        x = np.asarray(self.features[rows], dtype=np.float32)  # (T, F)
        mask = (rows != self.pad_row).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.tensor(idx, dtype=torch.int64),
        )


def make_loader(windows_path: Path, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    ds = WindowDataset(windows_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
