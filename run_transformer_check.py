from __future__ import annotations

import torch

from src.data.modular_addition import (
    ModularAdditionConfig,
    build_modular_addition_datasets,
)
from src.models.transformer import TransformerConfig, ModularTransformer


def main() -> None:
    data_cfg = ModularAdditionConfig(
        p=97,
        train_frac=0.3,
        seed=0,
        one_hot=False,
    )

    data_obj = build_modular_addition_datasets(data_cfg)
    train_dataset = data_obj["train_dataset"]

    model_cfg = TransformerConfig(
        vocab_size=data_cfg.p + 1,   # numbers 0..p-1 plus separator token p
        seq_len=3,
        d_model=128,
        n_heads=4,
        d_mlp=512,
        n_layers=2,
        output_dim=data_cfg.p,
        use_bias=True,
    )

    model = ModularTransformer(model_cfg)
    print(model)
    print("-" * 80)

    x, y, meta = train_dataset[0]
    x_batch = x.unsqueeze(0)

    print("Single-example input:")
    print(f"x = {x.tolist()}")
    print(f"x.shape = {tuple(x.shape)}")
    print(f"y = {y.item()}")
    print(f"meta = {meta}")
    print("-" * 80)

    out = model(x_batch)

    print("Forward pass shapes:")
    print(f"token_embeddings shape = {tuple(out['token_embeddings'].shape)}")
    print(f"x_final shape          = {tuple(out['x_final'].shape)}")
    print(f"final_token_rep shape  = {tuple(out['final_token_rep'].shape)}")
    print(f"logits shape           = {tuple(out['logits'].shape)}")

    assert x.dtype == torch.long
    assert x.shape == (3,)
    assert out["token_embeddings"].shape == (1, 3, model_cfg.d_model)
    assert out["x_final"].shape == (1, 3, model_cfg.d_model)
    assert out["final_token_rep"].shape == (1, model_cfg.d_model)
    assert out["logits"].shape == (1, data_cfg.p)

    pred = out["logits"].argmax(dim=-1)
    print(f"predicted class = {pred.item()}")

    batch_x = []
    batch_y = []
    for i in range(8):
        x_i, y_i, _ = train_dataset[i]
        batch_x.append(x_i)
        batch_y.append(y_i)

    batch_x = torch.stack(batch_x, dim=0)
    batch_y = torch.stack(batch_y, dim=0)

    out_batch = model(batch_x)
    loss = torch.nn.CrossEntropyLoss()(out_batch["logits"], batch_y)

    print("-" * 80)
    print("Batch check:")
    print(f"batch_x shape = {tuple(batch_x.shape)}")
    print(f"batch_y shape = {tuple(batch_y.shape)}")
    print(f"batch logits shape = {tuple(out_batch['logits'].shape)}")
    print(f"loss = {loss.item():.6f}")

    assert batch_x.shape == (8, 3)
    assert batch_y.shape == (8,)
    assert out_batch["logits"].shape == (8, data_cfg.p)
    assert torch.isfinite(loss)

    print("\nTransformer checks passed.")


if __name__ == "__main__":
    main()