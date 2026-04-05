from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _finalize_plot(path: str | Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def plot_train_test_accuracy(metrics_csv: str | Path, output_path: str | Path, x_col: str) -> None:
    df = pd.read_csv(metrics_csv)

    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df["train_acc"], label="train_acc")
    plt.plot(df[x_col], df["test_acc"], label="test_acc")
    plt.legend()
    _finalize_plot(
        path=output_path,
        title=f"Train/Test Accuracy vs {x_col}",
        xlabel=x_col,
        ylabel="Accuracy",
    )


def plot_train_test_loss(metrics_csv: str | Path, output_path: str | Path, x_col: str) -> None:
    df = pd.read_csv(metrics_csv)

    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df["train_loss"], label="train_loss")
    plt.plot(df[x_col], df["test_loss"], label="test_loss")
    plt.legend()
    _finalize_plot(
        path=output_path,
        title=f"Train/Test Loss vs {x_col}",
        xlabel=x_col,
        ylabel="Loss",
    )


def plot_param_norm(metrics_csv: str | Path, output_path: str | Path, x_col: str) -> None:
    df = pd.read_csv(metrics_csv)

    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df["param_l2_norm"], label="param_l2_norm")
    plt.legend()
    _finalize_plot(
        path=output_path,
        title=f"Parameter L2 Norm vs {x_col}",
        xlabel=x_col,
        ylabel="L2 norm",
    )


def plot_both(metrics_csv: str | Path, plot_dir: str | Path) -> None:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for x_col in ["epoch", "global_step"]:
        plot_train_test_accuracy(
            metrics_csv=metrics_csv,
            output_path=plot_dir / f"accuracy_vs_{x_col}.png",
            x_col=x_col,
        )
        plot_train_test_loss(
            metrics_csv=metrics_csv,
            output_path=plot_dir / f"loss_vs_{x_col}.png",
            x_col=x_col,
        )
        plot_param_norm(
            metrics_csv=metrics_csv,
            output_path=plot_dir / f"param_norm_vs_{x_col}.png",
            x_col=x_col,
        )