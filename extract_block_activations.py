"""
extract_block_activations.py — extract per-layer (block-level) activations from transformer.

Saves activations from four locations in the residual stream:
    embedding  : token_embed + pos_embed (before any block)
    after_block0 : residual stream after block 0
    after_block1 : residual stream after block 1 (= after final block, before final_ln)
    final_token  : post-final-LN, last token position (same as main activations)

Only extracts the LAST token position (index -1), which is the readout position.

Output:
    runs/<run>/activations/<ckpt_stem>/block_activations.npz
        keys: embedding, after_block0, after_block1, final_token
        each shape [N, d_model]
    (metadata.npz already saved by extract_activations.py)

Usage:
    python extract_block_activations.py --run-dir runs/<run> --checkpoints epoch_0050 epoch_0150 epoch_1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.modular_addition import ModularAdditionConfig, build_modular_addition_datasets
from src.models.transformer import ModularTransformer, TransformerConfig


def load_model(ckpt_path, cfg, device):
    model = ModularTransformer(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_all_layers(model: ModularTransformer, dataset, device, batch_size=512):
    """Extract final-token representations from each stage of the residual stream."""
    buffers = {
        "embedding": [],
        "after_block0": [],
        "after_block1": [],
        "final_token": [],
    }
    n = len(dataset)
    for start in range(0, n, batch_size):
        xs = []
        for i in range(start, min(start + batch_size, n)):
            x, _, _ = dataset[i]
            xs.append(x)
        tokens = torch.stack(xs).to(device)
        bsz, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        tok_emb = model.token_embed(tokens)
        pos_emb = model.pos_embed(positions)
        x = tok_emb + pos_emb
        buffers["embedding"].append(x[:, -1, :].cpu().numpy())

        for i, block in enumerate(model.blocks):
            out = block(x)
            x = out["x"]
            key = f"after_block{i}"
            if key in buffers:
                buffers[key].append(x[:, -1, :].cpu().numpy())

        x_final = model.final_ln(x)
        buffers["final_token"].append(x_final[:, -1, :].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in buffers.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    cfg_dir = run_dir / "configs"
    device = torch.device(args.device)

    with open(cfg_dir / "model_config.json") as f:
        model_cfg = TransformerConfig(**json.load(f))
    with open(cfg_dir / "data_config.json") as f:
        data_cfg = ModularAdditionConfig(**json.load(f))

    data = build_modular_addition_datasets(data_cfg)
    train_ds = data["train_dataset"]
    test_ds = data["test_dataset"]

    # Build combined dataset order matching existing metadata.npz (train first, then test)
    class _Combined:
        def __init__(self, a, b):
            self._a = a
            self._b = b
        def __len__(self):
            return len(self._a) + len(self._b)
        def __getitem__(self, i):
            if i < len(self._a):
                return self._a[i]
            return self._b[i - len(self._a)]
    combined = _Combined(train_ds, test_ds)

    acts_base = run_dir / "activations"
    for stem in args.checkpoints:
        ckpt_path = ckpt_dir / f"{stem}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping")
            continue
        out_dir = acts_base / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting block activations: {stem}...")
        model = load_model(ckpt_path, model_cfg, device)
        layer_acts = extract_all_layers(model, combined, device)

        np.savez(out_dir / "block_activations.npz", **layer_acts)
        for k, v in layer_acts.items():
            print(f"  {k}: {v.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
