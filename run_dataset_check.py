from __future__ import annotations

import torch

from src.data.modular_addition import (
    ModularAdditionConfig,
    build_modular_addition_datasets,
    dataset_summary,
)


def main() -> None:
    cfg = ModularAdditionConfig(
        p=97,
        train_frac=0.3,
        seed=0,
        one_hot=True,
    )

    print(dataset_summary(cfg))
    print("-" * 60)

    obj = build_modular_addition_datasets(cfg)
    train_dataset = obj["train_dataset"]
    test_dataset = obj["test_dataset"]

    print(f"len(train_dataset) = {len(train_dataset)}")
    print(f"len(test_dataset)  = {len(test_dataset)}")

    x, y, meta = train_dataset[0]

    print("\nFirst training example:")
    print(f"x.shape = {tuple(x.shape)}")
    print(f"y = {y.item()}")
    print(f"meta = {meta}")

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 1
    assert y.ndim == 0
    assert x.shape[0] == 2 * cfg.p

    expected_total = cfg.p * cfg.p
    actual_total = len(train_dataset) + len(test_dataset)
    assert expected_total == actual_total, "Train/test sizes do not sum to full dataset."

    train_idx = set(obj["train_idx"].tolist())
    test_idx = set(obj["test_idx"].tolist())
    assert train_idx.isdisjoint(test_idx), "Train/test split overlaps."

    print("\nDataset checks passed.")


if __name__ == "__main__":
    main()