from __future__ import annotations

import torch

from src.data.modular_addition import (
    ModularAdditionConfig,
    build_modular_addition_datasets,
)
from src.models.mlp import MLPConfig, ModularMLP


def main() -> None:
    # -----------------------------
    # Build dataset
    # -----------------------------
    data_cfg = ModularAdditionConfig(
        p=97,
        train_frac=0.3,
        seed=0,
        one_hot=True,
    )

    data_obj = build_modular_addition_datasets(data_cfg)
    train_dataset = data_obj["train_dataset"]

    # -----------------------------
    # Build model
    # -----------------------------
    model_cfg = MLPConfig(
        input_dim=2 * data_cfg.p,
        hidden_dim=256,
        output_dim=data_cfg.p,
        activation="relu",
        use_bias=True,
    )

    model = ModularMLP(model_cfg)
    print(model)
    print("-" * 60)

    # -----------------------------
    # Single example check
    # -----------------------------
    x, y, meta = train_dataset[0]

    # Add batch dimension: [input_dim] -> [1, input_dim]
    x_batch = x.unsqueeze(0)

    out = model(x_batch)

    print("Single-example forward pass:")
    print(f"input shape         = {tuple(x_batch.shape)}")
    print(f"hidden_pre shape    = {tuple(out['hidden_pre'].shape)}")
    print(f"hidden_post shape   = {tuple(out['hidden_post'].shape)}")
    print(f"logits shape        = {tuple(out['logits'].shape)}")
    print(f"target y            = {y.item()}")
    print(f"meta                = {meta}")

    assert out["hidden_pre"].shape == (1, model_cfg.hidden_dim)
    assert out["hidden_post"].shape == (1, model_cfg.hidden_dim)
    assert out["logits"].shape == (1, model_cfg.output_dim)

    pred = out["logits"].argmax(dim=-1)
    print(f"pred shape          = {tuple(pred.shape)}")
    print(f"predicted class     = {pred.item()}")

    # -----------------------------
    # Small batch check
    # -----------------------------
    batch_x = []
    batch_y = []

    for i in range(8):
        x_i, y_i, _ = train_dataset[i]
        batch_x.append(x_i)
        batch_y.append(y_i)

    batch_x = torch.stack(batch_x, dim=0)  # [8, input_dim]
    batch_y = torch.stack(batch_y, dim=0)  # [8]

    out_batch = model(batch_x)

    print("\nBatch forward pass:")
    print(f"batch_x shape       = {tuple(batch_x.shape)}")
    print(f"batch_y shape       = {tuple(batch_y.shape)}")
    print(f"batch logits shape  = {tuple(out_batch['logits'].shape)}")

    assert batch_x.shape == (8, model_cfg.input_dim)
    assert batch_y.shape == (8,)
    assert out_batch["hidden_pre"].shape == (8, model_cfg.hidden_dim)
    assert out_batch["hidden_post"].shape == (8, model_cfg.hidden_dim)
    assert out_batch["logits"].shape == (8, model_cfg.output_dim)

    # -----------------------------
    # Loss smoke test
    # -----------------------------
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out_batch["logits"], batch_y)

    print("\nLoss smoke test:")
    print(f"cross-entropy loss  = {loss.item():.6f}")

    assert torch.isfinite(loss), "Loss is not finite."

    print("\nModel checks passed.")


if __name__ == "__main__":
    main()