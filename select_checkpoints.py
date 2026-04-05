"""
select_checkpoints.py — identify key training-phase checkpoints for a run.

Reads metrics.csv, applies threshold rules, and writes
runs/<run>/analysis/checkpoint_selection.json

Usage:
    python select_checkpoints.py --run-dir runs/<run_name>
    python select_checkpoints.py --run-dir runs/<run_name> --train-thresh 0.99 --test-thresh 0.95
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: _try_num(v) for k, v in row.items()})
    return rows


def _try_num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def nearest_checkpoint(epoch: int, ckpt_dir: Path, every: int = 25) -> str | None:
    """Return the checkpoint filename whose epoch is closest to `epoch`."""
    candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not candidates:
        return None
    # Parse epochs from filenames
    parsed = []
    for c in candidates:
        try:
            e = int(c.stem.split("_")[1])
            parsed.append((e, c.name))
        except (IndexError, ValueError):
            pass
    if not parsed:
        return None
    best = min(parsed, key=lambda x: abs(x[0] - epoch))
    return best[1]


def select(rows: list[dict], train_thresh: float, test_thresh: float) -> dict:
    """
    Returns a dict with keys:
        early, memorization, plateau, grokking, late
    Each value is {epoch, train_acc, test_acc, checkpoint_file}.
    """
    result = {}

    # Early: ~20% through training, before train convergence
    total_epochs = int(rows[-1]["epoch"])
    early_target = max(1, total_epochs // 5)
    early_row = min(rows, key=lambda r: abs(r["epoch"] - early_target))
    result["early"] = early_row

    # Memorization: first epoch with train_acc >= train_thresh
    mem_row = None
    for r in rows:
        if r.get("train_acc", 0) >= train_thresh:
            mem_row = r
            break
    result["memorization"] = mem_row

    # Plateau: epoch halfway between memorization and grokking
    grok_row = None
    for r in rows:
        if r.get("test_acc", 0) >= test_thresh:
            grok_row = r
            break
    result["grokking"] = grok_row

    if mem_row and grok_row:
        plateau_target = (int(mem_row["epoch"]) + int(grok_row["epoch"])) // 2
        plateau_row = min(rows, key=lambda r: abs(r["epoch"] - plateau_target))
        result["plateau"] = plateau_row
    else:
        result["plateau"] = None

    # Late: last 10% of training
    late_target = int(0.9 * total_epochs)
    late_row = min(rows, key=lambda r: abs(r["epoch"] - late_target))
    result["late"] = late_row

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-thresh", type=float, default=0.99)
    parser.add_argument("--test-thresh", type=float, default=0.95)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    ckpt_dir = run_dir / "checkpoints"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    out_path = analysis_dir / "checkpoint_selection.json"

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found")
        return

    rows = load_metrics(metrics_path)
    selected = select(rows, args.train_thresh, args.test_thresh)

    output = {}
    for phase, row in selected.items():
        if row is None:
            output[phase] = None
            continue
        epoch = int(row["epoch"])
        ckpt_file = nearest_checkpoint(epoch, ckpt_dir)
        output[phase] = {
            "epoch": epoch,
            "train_acc": round(float(row.get("train_acc", 0)), 4),
            "test_acc": round(float(row.get("test_acc", 0)), 4),
            "checkpoint_file": ckpt_file,
            "checkpoint_path": str(ckpt_dir / ckpt_file) if ckpt_file else None,
        }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Checkpoint selection written to {out_path}")
    print()
    for phase, info in output.items():
        if info:
            print(f"  {phase:15s}: epoch={info['epoch']:4d}  train={info['train_acc']:.3f}  test={info['test_acc']:.3f}  ckpt={info['checkpoint_file']}")
        else:
            print(f"  {phase:15s}: NOT FOUND")


if __name__ == "__main__":
    main()
