"""
plot_analysis.py — generate all analysis figures for a grokking run.

Figures produced in runs/<run>/plots/:
    fig1_training_curves.png         train/test accuracy + loss (2 panels)
    fig2_metrics_over_training.png   probe test acc, eff rank, fourier conc, entropy (4 panels)
    fig3_pca_spectrum.png            cumulative explained variance at 3 checkpoints
    fig4_fourier_spectrum.png        Fourier power spectrum of class means at 3 checkpoints
    fig5_block_probe.png             per-layer probe test accuracy at key checkpoints

Usage:
    python plot_analysis.py --run-dir runs/<run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

PHASE_COLORS = {
    "epoch_0050": "#e74c3c",   # red — memorization
    "epoch_0100": "#e67e22",   # orange — plateau
    "epoch_0150": "#27ae60",   # green — grokking
    "epoch_1000": "#2980b9",   # blue — late
}
PHASE_LABELS = {
    "epoch_0050": "Memorization (ep 50)",
    "epoch_0100": "Plateau (ep 100)",
    "epoch_0150": "Post-grokking (ep 150)",
    "epoch_1000": "Late (ep 1000)",
}


def savefig(path: Path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── fig 1: training curves ────────────────────────────────────────────────────
def fig1_training_curves(metrics_csv: Path, plot_dir: Path, mem_epoch=50, grok_epoch=150):
    df = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_acc"], label="Train accuracy", color="#2c3e50", linewidth=1.5)
    ax.plot(df["epoch"], df["test_acc"],  label="Test accuracy",  color="#e74c3c", linewidth=1.5)
    ax.axvline(mem_epoch,  color="#e74c3c", linestyle="--", alpha=0.6, label=f"Memorization (ep {mem_epoch})")
    ax.axvline(grok_epoch, color="#27ae60", linestyle="--", alpha=0.6, label=f"Grokking (ep {grok_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train / Test Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(df["epoch"], df["train_loss"], label="Train loss", color="#2c3e50", linewidth=1.5)
    ax.plot(df["epoch"], df["test_loss"],  label="Test loss",  color="#e74c3c", linewidth=1.5)
    ax.axvline(mem_epoch,  color="#e74c3c", linestyle="--", alpha=0.6)
    ax.axvline(grok_epoch, color="#27ae60", linestyle="--", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Train / Test Loss")
    ax.legend(fontsize=8)

    savefig(plot_dir / "fig1_training_curves.png")


# ── fig 2: representation metrics over training ───────────────────────────────
def fig2_metrics_over_training(analysis_csv: Path, metrics_csv: Path, plot_dir: Path,
                                mem_epoch=50, grok_epoch=150):
    df_a = pd.read_csv(analysis_csv)
    df_m = pd.read_csv(metrics_csv)

    # Parse epoch from checkpoint name
    df_a["epoch"] = df_a["checkpoint"].str.extract(r"epoch_(\d+)").astype(int)
    df_a = df_a.sort_values("epoch")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    vline_kw = dict(linestyle="--", alpha=0.5, linewidth=1.2)

    def add_vlines(ax):
        ax.axvline(mem_epoch,  color="#e74c3c", label=f"Memorization (ep {mem_epoch})", **vline_kw)
        ax.axvline(grok_epoch, color="#27ae60", label=f"Grokking (ep {grok_epoch})", **vline_kw)

    # Panel 1: probe test accuracy + model test accuracy
    ax = axes[0, 0]
    ax.plot(df_m["epoch"], df_m["test_acc"], color="gray", linewidth=1, alpha=0.6, label="Model test acc")
    ax.plot(df_a["epoch"], df_a["probe_test_acc"], color="#8e44ad", linewidth=2, marker="o",
            markersize=4, label="Probe test acc (linear)")
    add_vlines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probe Test Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: effective rank
    ax = axes[0, 1]
    ax.plot(df_a["epoch"], df_a["effective_rank_participation"],
            color="#16a085", linewidth=2, marker="o", markersize=4)
    add_vlines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Representation Effective Rank")
    ax.legend(fontsize=8)

    # Panel 3: Fourier top-5 concentration
    ax = axes[1, 0]
    ax.plot(df_a["epoch"], df_a["fourier_top5_power_frac"],
            color="#d35400", linewidth=2, marker="o", markersize=4)
    add_vlines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of power in top-5 modes")
    ax.set_title("Fourier Structure of Class Means")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)

    # Panel 4: spectral entropy
    ax = axes[1, 1]
    ax.plot(df_a["epoch"], df_a["fourier_spectral_entropy"],
            color="#2980b9", linewidth=2, marker="o", markersize=4)
    add_vlines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spectral entropy (nats)")
    ax.set_title("Fourier Spectral Entropy of Class Means")
    ax.legend(fontsize=8)

    savefig(plot_dir / "fig2_metrics_over_training.png")


# ── fig 3: PCA explained variance spectrum ────────────────────────────────────
def fig3_pca_spectrum(acts_base: Path, plot_dir: Path,
                      checkpoints=("epoch_0050", "epoch_0150", "epoch_1000")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for stem in checkpoints:
        acts_dir = acts_base / stem
        if not (acts_dir / "activations.npy").exists():
            continue
        acts = np.load(acts_dir / "activations.npy")
        meta = np.load(acts_dir / "metadata.npz")
        X_train = acts[meta["split"] == 0]

        n_comp = min(64, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X_train)
        ev = pca.explained_variance_ratio_

        color = PHASE_COLORS.get(stem, "gray")
        label = PHASE_LABELS.get(stem, stem)

        # Panel 1: individual component variance
        axes[0].plot(range(1, len(ev) + 1), ev, marker="o", markersize=3,
                     linewidth=1.5, color=color, label=label)
        # Panel 2: cumulative variance
        axes[1].plot(range(1, len(ev) + 1), np.cumsum(ev), marker="o", markersize=3,
                     linewidth=1.5, color=color, label=label)

    for ax in axes:
        ax.set_xlabel("Principal component")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA Scree Plot")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative PCA Variance")
    axes[1].axhline(0.9, color="gray", linestyle=":", alpha=0.6, label="90% threshold")
    axes[1].legend(fontsize=8)

    savefig(plot_dir / "fig3_pca_spectrum.png")


# ── fig 4: Fourier power spectrum of class means ─────────────────────────────
def fig4_fourier_spectrum(acts_base: Path, plot_dir: Path, p: int = 97,
                          checkpoints=("epoch_0050", "epoch_0150", "epoch_1000")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for stem in checkpoints:
        acts_dir = acts_base / stem
        if not (acts_dir / "activations.npy").exists():
            continue
        acts = np.load(acts_dir / "activations.npy")
        meta = np.load(acts_dir / "metadata.npz")

        # Class means over full dataset
        class_means = np.zeros((p, acts.shape[1]))
        for c in range(p):
            mask = meta["y"] == c
            if mask.sum() > 0:
                class_means[c] = acts[mask].mean(axis=0)

        # DFT along class axis
        F = np.fft.rfft(class_means, axis=0)           # [p//2+1, d]
        power = np.abs(F) ** 2                          # [p//2+1, d]
        total = power.sum(axis=0, keepdims=True)
        frac = power / np.where(total > 1e-20, total, 1.0)  # normalized per dim

        # Mean over dimensions
        mean_frac = frac.mean(axis=1)  # [p//2+1]
        cumulative = np.cumsum(mean_frac) / mean_frac.sum()

        color = PHASE_COLORS.get(stem, "gray")
        label = PHASE_LABELS.get(stem, stem)
        freqs = np.arange(len(mean_frac))

        axes[0].plot(freqs[1:], mean_frac[1:], linewidth=1.5, color=color, label=label)
        axes[1].plot(freqs[1:], cumulative[1:], linewidth=1.5, color=color, label=label)

    axes[0].set_xlabel("Fourier frequency k")
    axes[0].set_ylabel("Mean power fraction")
    axes[0].set_title("Fourier Power Spectrum of Class-Mean Activations")
    axes[0].set_xlim(0, 25)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Fourier frequency k (cumulative up to)")
    axes[1].set_ylabel("Cumulative power fraction")
    axes[1].set_title("Cumulative Fourier Power")
    axes[1].set_xlim(0, 25)
    axes[1].axhline(0.9, color="gray", linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8)

    savefig(plot_dir / "fig4_fourier_spectrum.png")


# ── fig 5: block-level probe test accuracy ────────────────────────────────────
def _run_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=0)
    clf.fit(X_tr, y_train)
    return float(clf.score(X_te, y_test))


def fig5_block_probe(acts_base: Path, plot_dir: Path,
                     checkpoints=("epoch_0050", "epoch_0100", "epoch_0150", "epoch_0200",
                                  "epoch_0500", "epoch_1000")):
    layers = ["embedding", "after_block0", "after_block1", "final_token"]
    layer_labels = ["Embedding", "After Block 0", "After Block 1", "Final (post-LN)"]

    results = {}  # stem -> list of test_acc per layer
    for stem in checkpoints:
        acts_dir = acts_base / stem
        bp = acts_dir / "block_activations.npz"
        mp = acts_dir / "metadata.npz"
        if not bp.exists() or not mp.exists():
            continue
        blk = np.load(bp)
        meta = np.load(mp)
        mask_tr = meta["split"] == 0
        mask_te = meta["split"] == 1
        y_tr = meta["y"][mask_tr]
        y_te = meta["y"][mask_te]

        accs = []
        for layer in layers:
            if layer not in blk:
                accs.append(np.nan)
                continue
            X = blk[layer]
            acc = _run_probe(X[mask_tr], y_tr, X[mask_te], y_te)
            accs.append(acc)
        results[stem] = accs

    if not results:
        print("  No block activations found, skipping fig5")
        return

    x = np.arange(len(layers))
    n_ckpts = len(results)
    width = 0.8 / n_ckpts

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (stem, accs) in enumerate(results.items()):
        offset = (i - n_ckpts / 2 + 0.5) * width
        epoch = int(stem.split("_")[1])
        color = PHASE_COLORS.get(stem, plt.cm.viridis(i / n_ckpts))
        label = PHASE_LABELS.get(stem, f"ep {epoch}")
        bars = ax.bar(x + offset, accs, width * 0.9, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Linear probe test accuracy")
    ax.set_title("Layer-wise Linear Probe Accuracy (97-class output label)")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="upper left")

    savefig(plot_dir / "fig5_block_probe.png")


# ── fig 6: param norm + test acc (grokking signature) ─────────────────────────
def fig6_grokking_signature(metrics_csv: Path, plot_dir: Path):
    df = pd.read_csv(metrics_csv)
    fig, ax1 = plt.subplots(figsize=(10, 4))

    color1, color2, color3 = "#2c3e50", "#e74c3c", "#8e44ad"
    ax2 = ax1.twinx()

    ax1.plot(df["epoch"], df["train_acc"], color=color1, linewidth=1.5, label="Train acc")
    ax1.plot(df["epoch"], df["test_acc"],  color=color2, linewidth=1.5, label="Test acc")
    ax1.set_ylabel("Accuracy", color=color1)
    ax1.set_ylim(-0.05, 1.1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2.plot(df["epoch"], df["param_l2_norm"], color=color3, linewidth=1.5,
             linestyle="--", label="Param L2 norm")
    ax2.set_ylabel("Parameter L2 norm", color=color3)
    ax2.tick_params(axis="y", labelcolor=color3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    ax1.set_xlabel("Epoch")
    ax1.set_title("Grokking Signature: Test Accuracy Rise + Parameter Norm Decay")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "fig6_grokking_signature.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: fig6_grokking_signature.png")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_csv = run_dir / "metrics.csv"
    analysis_csv = run_dir / "analysis" / "representation_analysis.csv"
    acts_base = run_dir / "activations"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Load p from config
    p = 97
    dcfg = run_dir / "configs" / "data_config.json"
    if dcfg.exists():
        with open(dcfg) as f:
            p = json.load(f).get("p", 97)

    # Find grokking epoch from metrics
    df = pd.read_csv(metrics_csv)
    mem_rows = df[df["train_acc"] >= 0.99]
    mem_epoch = int(mem_rows["epoch"].iloc[0]) if len(mem_rows) else 57
    grok_rows = df[df["test_acc"] >= 0.9]
    grok_epoch = int(grok_rows["epoch"].iloc[0]) if len(grok_rows) else 150

    print(f"Memorization epoch: {mem_epoch}, Grokking epoch: {grok_epoch}")

    print("Generating figures...")

    print("  Fig 1: training curves")
    fig1_training_curves(metrics_csv, plot_dir, mem_epoch=mem_epoch, grok_epoch=grok_epoch)

    print("  Fig 2: representation metrics over training")
    if analysis_csv.exists():
        fig2_metrics_over_training(analysis_csv, metrics_csv, plot_dir,
                                    mem_epoch=mem_epoch, grok_epoch=grok_epoch)

    print("  Fig 3: PCA spectrum")
    fig3_pca_spectrum(acts_base, plot_dir,
                      checkpoints=("epoch_0050", "epoch_0150", "epoch_1000"))

    print("  Fig 4: Fourier spectrum")
    fig4_fourier_spectrum(acts_base, plot_dir, p=p,
                          checkpoints=("epoch_0050", "epoch_0150", "epoch_1000"))

    print("  Fig 5: block-level probe")
    fig5_block_probe(acts_base, plot_dir,
                     checkpoints=("epoch_0050", "epoch_0100", "epoch_0150",
                                  "epoch_0200", "epoch_0500", "epoch_1000"))

    print("  Fig 6: grokking signature")
    fig6_grokking_signature(metrics_csv, plot_dir)

    print(f"\nAll figures saved to {plot_dir}/")


if __name__ == "__main__":
    main()
