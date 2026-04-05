"""
extract_activations.py — extract hidden representations from transformer checkpoints.

Saves activations (final_token_rep = post-final-LN residual at last token) for
the full dataset, with metadata (a, b, y, split).

Output layout:
    runs/<run>/activations/<checkpoint_stem>/
        activations.npy   shape [N, d_model]
        metadata.npz      keys: a, b, y, split (0=train, 1=test)

Usage:
    python extract_activations.py --run-dir runs/<run_name> --checkpoints epoch_0050 epoch_0075 epoch_0150 epoch_1000
    python extract_activations.py --run-dir runs/<run_name> --all-checkpoints
    python extract_activations.py --run-dir runs/<run_name> --from-selection  # uses checkpoint_selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.modular_addition import ModularAdditionConfig, build_modular_addition_datasets
from src.models.transformer import ModularTransformer, TransformerConfig


def load_model(ckpt_path: Path, cfg: TransformerConfig, device: torch.device) -> ModularTransformer:
    model = ModularTransformer(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract(model: ModularTransformer, dataset, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Returns activations of shape [N, d_model] (final_token_rep)."""
    all_acts = []
    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = []
        for i in range(start, end):
            x, _, _ = dataset[i]
            xs.append(x)
        batch = torch.stack(xs).to(device)
        out = model(batch)
        all_acts.append(out["final_token_rep"].cpu().numpy())
    return np.concatenate(all_acts, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", nargs="*", help="checkpoint stems, e.g. epoch_0050 epoch_0150")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--from-selection", action="store_true", help="use runs/<run>/analysis/checkpoint_selection.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    cfg_dir = run_dir / "configs"
    device = torch.device(args.device)

    # Load model config
    with open(cfg_dir / "model_config.json") as f:
        mcfg_dict = json.load(f)
    with open(cfg_dir / "data_config.json") as f:
        dcfg_dict = json.load(f)

    model_cfg = TransformerConfig(**mcfg_dict)
    data_cfg = ModularAdditionConfig(**dcfg_dict)

    # Build full dataset (train + test separately, then combine)
    data = build_modular_addition_datasets(data_cfg)
    train_ds = data["train_dataset"]
    test_ds = data["test_dataset"]

    # Determine which checkpoints to extract
    if args.from_selection:
        sel_path = run_dir / "analysis" / "checkpoint_selection.json"
        if not sel_path.exists():
            print(f"ERROR: {sel_path} not found. Run select_checkpoints.py first.")
            return
        with open(sel_path) as f:
            sel = json.load(f)
        ckpt_stems = []
        for phase, info in sel.items():
            if info and info.get("checkpoint_file"):
                stem = Path(info["checkpoint_file"]).stem
                if stem not in ckpt_stems:
                    ckpt_stems.append(stem)
    elif args.all_checkpoints:
        ckpt_stems = [p.stem for p in sorted(ckpt_dir.glob("epoch_*.pt"))]
    elif args.checkpoints:
        ckpt_stems = args.checkpoints
    else:
        print("ERROR: specify --checkpoints, --all-checkpoints, or --from-selection")
        return

    acts_base = run_dir / "activations"
    acts_base.mkdir(exist_ok=True)

    for stem in ckpt_stems:
        ckpt_path = ckpt_dir / f"{stem}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping")
            continue

        out_dir = acts_base / stem
        out_dir.mkdir(exist_ok=True)

        print(f"Extracting {stem}...")
        model = load_model(ckpt_path, model_cfg, device)

        train_acts = extract(model, train_ds, device, args.batch_size)
        test_acts = extract(model, test_ds, device, args.batch_size)

        all_acts = np.concatenate([train_acts, test_acts], axis=0)
        all_a = np.concatenate([train_ds.a_vals, test_ds.a_vals])
        all_b = np.concatenate([train_ds.b_vals, test_ds.b_vals])
        all_y = np.concatenate([train_ds.y_vals, test_ds.y_vals])
        all_split = np.array([0] * len(train_ds) + [1] * len(test_ds), dtype=np.int64)

        np.save(out_dir / "activations.npy", all_acts)
        np.savez(out_dir / "metadata.npz", a=all_a, b=all_b, y=all_y, split=all_split)

        print(f"  Saved to {out_dir}  shape={all_acts.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
