from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ModularAdditionConfig:
    p: int = 97
    train_frac: float = 0.3
    seed: int = 0
    one_hot: bool = True


class ModularAdditionDataset(Dataset):
    """
    Dataset of all pairs (a, b) over {0, ..., p-1}, labeled by (a + b) mod p.

    If one_hot=True:
        x is a float tensor of shape [2p], formed by concatenating one-hot(a) and one-hot(b)

    If one_hot=False:
        x is a long tensor token sequence of shape [3]:
            [a, b, sep_token]
        where sep_token = p
    """

    def __init__(
        self,
        a_vals: np.ndarray,
        b_vals: np.ndarray,
        y_vals: np.ndarray,
        p: int,
        one_hot: bool = True,
    ) -> None:
        assert len(a_vals) == len(b_vals) == len(y_vals)
        self.a_vals = a_vals.astype(np.int64)
        self.b_vals = b_vals.astype(np.int64)
        self.y_vals = y_vals.astype(np.int64)
        self.p = int(p)
        self.one_hot = bool(one_hot)
        self.sep_token = self.p

    def __len__(self) -> int:
        return len(self.y_vals)

    def __getitem__(self, idx: int):
        a = int(self.a_vals[idx])
        b = int(self.b_vals[idx])
        y = int(self.y_vals[idx])

        if self.one_hot:
            x = torch.zeros(2 * self.p, dtype=torch.float32)
            x[a] = 1.0
            x[self.p + b] = 1.0
        else:
            x = torch.tensor([a, b, self.sep_token], dtype=torch.long)

        y_tensor = torch.tensor(y, dtype=torch.long)

        meta = {
            "a": a,
            "b": b,
            "y": y,
        }
        return x, y_tensor, meta


def build_all_examples(p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_grid, b_grid = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a_vals = a_grid.reshape(-1)
    b_vals = b_grid.reshape(-1)
    y_vals = (a_vals + b_vals) % p
    return a_vals, b_vals, y_vals


def make_train_test_split(
    n_examples: int,
    train_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_examples)

    n_train = int(train_frac * n_examples)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])

    return train_idx, test_idx


def build_modular_addition_datasets(
    cfg: ModularAdditionConfig,
) -> Dict[str, object]:
    a_vals, b_vals, y_vals = build_all_examples(cfg.p)
    train_idx, test_idx = make_train_test_split(
        n_examples=len(y_vals),
        train_frac=cfg.train_frac,
        seed=cfg.seed,
    )

    train_dataset = ModularAdditionDataset(
        a_vals=a_vals[train_idx],
        b_vals=b_vals[train_idx],
        y_vals=y_vals[train_idx],
        p=cfg.p,
        one_hot=cfg.one_hot,
    )

    test_dataset = ModularAdditionDataset(
        a_vals=a_vals[test_idx],
        b_vals=b_vals[test_idx],
        y_vals=y_vals[test_idx],
        p=cfg.p,
        one_hot=cfg.one_hot,
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "full_a": a_vals,
        "full_b": b_vals,
        "full_y": y_vals,
    }


def dataset_summary(cfg: ModularAdditionConfig) -> str:
    obj = build_modular_addition_datasets(cfg)
    train_dataset = obj["train_dataset"]
    test_dataset = obj["test_dataset"]
    full_y = obj["full_y"]

    counts = np.bincount(full_y, minlength=cfg.p)

    lines = [
        "Modular Addition Dataset Summary",
        f"p = {cfg.p}",
        f"total examples = {cfg.p * cfg.p}",
        f"train examples = {len(train_dataset)}",
        f"test examples = {len(test_dataset)}",
        f"min class count = {counts.min()}",
        f"max class count = {counts.max()}",
        f"one_hot = {cfg.one_hot}",
        f"seed = {cfg.seed}",
    ]
    return "\n".join(lines)