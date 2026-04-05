from __future__ import annotations

import itertools
import subprocess
import sys


def main() -> None:
    train_fracs = [0.2, 0.3]
    hidden_dims = [256, 512]
    weight_decays = [1e-2, 1e-1, 1.0]
    seeds = [0]

    jobs = list(itertools.product(train_fracs, hidden_dims, weight_decays, seeds))

    print(f"Launching {len(jobs)} runs...")

    for i, (train_frac, hidden_dim, weight_decay, seed) in enumerate(jobs, start=1):
        print("=" * 100)
        print(
            f"[{i}/{len(jobs)}] "
            f"train_frac={train_frac}, hidden_dim={hidden_dim}, "
            f"weight_decay={weight_decay}, seed={seed}"
        )
        print("=" * 100)

        cmd = [
            sys.executable,
            "train_modadd.py",
            "--arch", "mlp",
            "--train-frac", str(train_frac),
            "--hidden-dim", str(hidden_dim),
            "--weight-decay", str(weight_decay),
            "--seed", str(seed),
            "--num-epochs", str(800),
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()