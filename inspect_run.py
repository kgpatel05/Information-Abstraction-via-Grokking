from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()

    metrics_path = Path(args.run_dir) / "metrics.csv"
    df = pd.read_csv(metrics_path)

    best_test_idx = df["test_acc"].idxmax()
    best_row = df.loc[best_test_idx]

    print("=" * 80)
    print(f"Run: {args.run_dir}")
    print("=" * 80)
    print("Best test accuracy row:")
    print(best_row.to_string())
    print("=" * 80)

    print("Final row:")
    print(df.iloc[-1].to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()