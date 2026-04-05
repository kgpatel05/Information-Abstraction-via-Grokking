"""
summarize_runs.py — scan all runs/, extract key metrics, write results/run_summary.csv

Usage:
    python summarize_runs.py
    python summarize_runs.py --runs-dir runs --out results/run_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metrics(metrics_path: Path) -> list[dict]:
    rows = []
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _try_float(v) for k, v in row.items()})
    return rows


def _try_float(v: str):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def summarize_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    rows = load_metrics(metrics_path)
    if not rows:
        return None

    # Load config
    cfg_dir = run_dir / "configs"
    cli_args = {}
    if (cfg_dir / "cli_args.json").exists():
        with open(cfg_dir / "cli_args.json") as f:
            cli_args = json.load(f)

    final = rows[-1]
    best_test_row = max(rows, key=lambda r: r.get("test_acc", 0.0))
    best_train_row = max(rows, key=lambda r: r.get("train_acc", 0.0))

    # Find epoch where train_acc first reaches 0.99
    memorization_epoch = None
    for r in rows:
        if r.get("train_acc", 0) >= 0.99:
            memorization_epoch = int(r["epoch"])
            break

    # Find epoch where test_acc first exceeds 0.5
    grokking_epoch = None
    for r in rows:
        if r.get("test_acc", 0) >= 0.5:
            grokking_epoch = int(r["epoch"])
            break

    arch = cli_args.get("arch", final.get("arch", "unknown"))

    summary = {
        "run_dir": run_dir.name,
        "arch": arch,
        "p": cli_args.get("p", ""),
        "train_frac": cli_args.get("train_frac", final.get("train_frac", "")),
        "learning_rate": cli_args.get("learning_rate", final.get("learning_rate", "")),
        "weight_decay": cli_args.get("weight_decay", final.get("weight_decay", "")),
        "num_epochs": cli_args.get("num_epochs", int(final.get("epoch", 0))),
        "seed": cli_args.get("seed", final.get("seed", "")),
        # arch-specific size
        "hidden_dim": cli_args.get("hidden_dim", ""),
        "d_model": cli_args.get("d_model", final.get("d_model", "")),
        "n_heads": cli_args.get("n_heads", final.get("n_heads", "")),
        "d_mlp": cli_args.get("d_mlp", final.get("d_mlp", "")),
        "n_layers": cli_args.get("n_layers", final.get("n_layers", "")),
        # training outcomes
        "final_train_acc": final.get("train_acc", ""),
        "final_test_acc": final.get("test_acc", ""),
        "best_test_acc": best_test_row.get("test_acc", ""),
        "best_test_epoch": int(best_test_row.get("epoch", 0)),
        "memorization_epoch": memorization_epoch if memorization_epoch else "",
        "grokking_epoch_test50": grokking_epoch if grokking_epoch else "",
        "total_epochs": int(final.get("epoch", 0)),
        "final_param_norm": final.get("param_l2_norm", ""),
        # tag
        "grokked": "yes" if float(final.get("test_acc", 0)) >= 0.9 else "no",
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="results/run_summary.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        s = summarize_run(run_dir)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("No runs found.")
        return

    fieldnames = list(summaries[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} run summaries to {out_path}")

    # Print quick table to stdout
    print(f"\n{'Run':<60} {'arch':<12} {'wd':<6} {'epochs':<7} {'train':<7} {'test':<7} {'grokked'}")
    print("-" * 115)
    for s in summaries:
        print(
            f"{s['run_dir']:<60} {s['arch']:<12} {s['weight_decay']:<6} "
            f"{s['total_epochs']:<7} {s['final_train_acc']:<7.3f} {s['final_test_acc']:<7.3f} {s['grokked']}"
        )


if __name__ == "__main__":
    main()
