"""
analyze_representations.py — linear probes, PCA, and Fourier alignment on extracted activations.

Reads activations from runs/<run>/activations/<ckpt_stem>/
Writes results to runs/<run>/analysis/

Usage:
    # Analyze specific checkpoints:
    python analyze_representations.py --run-dir runs/<run> --checkpoints epoch_0050 epoch_0075 epoch_0150 epoch_1000

    # Analyze all extracted checkpoints:
    python analyze_representations.py --run-dir runs/<run> --all

    # Use checkpoint_selection.json:
    python analyze_representations.py --run-dir runs/<run> --from-selection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_acts(acts_dir: Path):
    acts = np.load(acts_dir / "activations.npy")
    meta = np.load(acts_dir / "metadata.npz")
    return acts, meta


def split_acts(acts, meta):
    mask_train = meta["split"] == 0
    mask_test = meta["split"] == 1
    return (
        acts[mask_train], meta["y"][mask_train],
        acts[mask_test], meta["y"][mask_test],
        meta["a"], meta["b"], meta["y"],
    )


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_linear_probe(X_train, y_train, X_test, y_test, max_iter: int = 1000) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=max_iter,
        C=1.0,
        random_state=0,
    )
    clf.fit(X_tr, y_train)
    train_acc = float(clf.score(X_tr, y_train))
    test_acc = float(clf.score(X_te, y_test))
    return {"probe_train_acc": train_acc, "probe_test_acc": test_acc}


# ---------------------------------------------------------------------------
# PCA / effective rank
# ---------------------------------------------------------------------------

def run_pca(acts: np.ndarray, n_components: int | None = None) -> dict:
    if n_components is None:
        n_components = min(acts.shape[0], acts.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(acts)
    ev = pca.explained_variance_ratio_

    # Participation ratio (effective rank): (sum ev)^2 / sum ev^2
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())

    # Top-k cumulative variance
    top1 = float(ev[0])
    top5 = float(ev[:5].sum())
    top10 = float(ev[:10].sum())
    top20 = float(ev[:20].sum())

    return {
        "effective_rank_participation": pr,
        "top1_var": top1,
        "top5_var": top5,
        "top10_var": top10,
        "top20_var": top20,
        "n_components": n_components,
        "explained_variance_ratio": ev.tolist(),
    }


# ---------------------------------------------------------------------------
# Fourier alignment
# ---------------------------------------------------------------------------

def build_fourier_basis(p: int, n_modes: int | None = None) -> np.ndarray:
    """
    Returns Fourier basis vectors of shape [n_modes*2, p].
    For k=1..n_modes: cos(2pi*k*j/p), sin(2pi*k*j/p) over j=0..p-1.
    """
    if n_modes is None:
        n_modes = p // 2
    j = np.arange(p)
    rows = []
    for k in range(1, n_modes + 1):
        rows.append(np.cos(2 * np.pi * k * j / p))
        rows.append(np.sin(2 * np.pi * k * j / p))
    return np.stack(rows)  # [2*n_modes, p]


def run_fourier_alignment(acts: np.ndarray, y: np.ndarray, p: int, n_modes: int = 10) -> dict:
    """
    Measure Fourier structure of class-conditional mean representations.

    For each hidden dimension d:
      1. Compute class means m_d[c] = mean(acts[y==c, d]) for c in 0..p-1
      2. Compute DFT of m_d
      3. Compute fraction of power in top-K Fourier modes

    A representation aligned with modular arithmetic structure should have
    class means that vary sinusoidally across classes, concentrating power
    in a few modes (especially k=1 for mod-p addition).
    """
    # Class means: shape [p, d_model]
    class_means = np.zeros((p, acts.shape[1]))
    for c in range(p):
        mask = y == c
        if mask.sum() > 0:
            class_means[c] = acts[mask].mean(axis=0)

    # DFT of each dimension along the class axis
    # class_means: [p, d_model]  -> fft along axis=0 -> [p, d_model]
    F = np.fft.rfft(class_means, axis=0)  # [p//2+1, d_model]
    power = np.abs(F) ** 2  # [p//2+1, d_model]

    total_power = power.sum(axis=0)  # [d_model]
    # Avoid dividing by zero for dead dimensions
    safe_total = np.where(total_power > 1e-20, total_power, 1.0)

    # Fraction of power in top-K modes per dimension, then averaged
    sorted_power = np.sort(power, axis=0)[::-1]  # descending by frequency

    top1_frac = float((sorted_power[0] / safe_total).mean())
    top3_frac = float((sorted_power[:3].sum(axis=0) / safe_total).mean())
    top5_frac = float((sorted_power[:5].sum(axis=0) / safe_total).mean())
    top10_frac = float((sorted_power[:n_modes].sum(axis=0) / safe_total).mean())

    # Spectral entropy (lower = more concentrated = more structured)
    p_norm = power / safe_total[None, :]  # normalize each dim
    p_norm = np.clip(p_norm, 1e-30, 1.0)
    entropy = float(-np.sum(p_norm * np.log(p_norm), axis=0).mean())

    # Max spectral entropy for p//2+1 modes (uniform distribution)
    n_freq = power.shape[0]
    max_entropy = float(np.log(n_freq))

    return {
        "fourier_top1_power_frac": top1_frac,
        "fourier_top3_power_frac": top3_frac,
        "fourier_top5_power_frac": top5_frac,
        "fourier_top10_power_frac": top10_frac,
        "fourier_spectral_entropy": entropy,
        "fourier_spectral_entropy_max": max_entropy,
        "fourier_concentration": float(top5_frac),  # primary metric
        "n_modes": n_modes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_checkpoint(stem: str, acts_dir: Path, p: int) -> dict:
    if not (acts_dir / "activations.npy").exists():
        return {}

    acts, meta = load_acts(acts_dir)
    X_train, y_train, X_test, y_test, a_all, b_all, y_all = split_acts(acts, meta)

    results = {"checkpoint": stem}

    # Linear probe
    probe = run_linear_probe(X_train, y_train, X_test, y_test)
    results.update(probe)

    # PCA on train activations
    pca = run_pca(X_train, n_components=min(64, X_train.shape[0], X_train.shape[1]))
    results.update({k: v for k, v in pca.items() if k != "explained_variance_ratio"})
    # Store full ev separately
    results["explained_variance_ratio"] = pca["explained_variance_ratio"]

    # Fourier alignment (use all examples for class means)
    fourier = run_fourier_alignment(acts, meta["y"], p=p, n_modes=10)
    results.update(fourier)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--all", action="store_true", dest="all_ckpts")
    parser.add_argument("--from-selection", action="store_true")
    parser.add_argument("--p", type=int, default=97)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    acts_base = run_dir / "activations"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Load p from data_config if available
    dcfg_path = run_dir / "configs" / "data_config.json"
    if dcfg_path.exists():
        with open(dcfg_path) as f:
            p = json.load(f).get("p", args.p)
    else:
        p = args.p

    # Determine checkpoints to analyze
    if args.from_selection:
        sel_path = analysis_dir / "checkpoint_selection.json"
        if not sel_path.exists():
            print(f"ERROR: {sel_path} not found.")
            return
        with open(sel_path) as f:
            sel = json.load(f)
        stems = []
        for info in sel.values():
            if info and info.get("checkpoint_file"):
                s = Path(info["checkpoint_file"]).stem
                if s not in stems:
                    stems.append(s)
    elif args.all_ckpts:
        stems = [d.name for d in sorted(acts_base.iterdir()) if d.is_dir()]
    elif args.checkpoints:
        stems = args.checkpoints
    else:
        print("ERROR: specify --checkpoints, --all, or --from-selection")
        return

    all_results = []
    for stem in stems:
        acts_dir = acts_base / stem
        if not acts_dir.exists():
            print(f"  WARNING: activations for {stem} not found at {acts_dir}, skipping")
            continue
        print(f"Analyzing {stem}...")
        r = analyze_checkpoint(stem, acts_dir, p=p)
        if r:
            all_results.append(r)
            print(f"  probe_train={r.get('probe_train_acc', 0):.3f}  probe_test={r.get('probe_test_acc', 0):.3f}  "
                  f"eff_rank={r.get('effective_rank_participation', 0):.1f}  "
                  f"fourier_conc={r.get('fourier_concentration', 0):.3f}  "
            f"spectral_entropy={r.get('fourier_spectral_entropy', 0):.2f}")

    if not all_results:
        print("No results to write.")
        return

    # Write full results (without explained_variance_ratio list) to CSV
    import csv
    skip_keys = {"explained_variance_ratio"}
    scalar_keys = [k for k in all_results[0] if k not in skip_keys]
    summary_path = analysis_dir / "representation_analysis.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in scalar_keys})

    # Write full results to JSON (includes ev arrays)
    full_path = analysis_dir / "representation_analysis_full.json"
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Write summary markdown
    md_lines = ["# Representation Analysis Summary\n"]
    md_lines.append(f"Run: {run_dir.name}\n\n")
    md_lines.append("| Checkpoint | Probe Train | Probe Test | Eff Rank | Fourier Conc (top5) | Spectral Entropy |\n")
    md_lines.append("|---|---|---|---|---|---|\n")
    for r in all_results:
        md_lines.append(
            f"| {r['checkpoint']} | {r.get('probe_train_acc', 0):.3f} | "
            f"{r.get('probe_test_acc', 0):.3f} | "
            f"{r.get('effective_rank_participation', 0):.1f} | "
            f"{r.get('fourier_concentration', 0):.3f} | "
            f"{r.get('fourier_spectral_entropy', 0):.2f} |\n"
        )
    md_path = analysis_dir / "summary.md"
    with open(md_path, "w") as f:
        f.writelines(md_lines)

    print(f"\nResults written to:")
    print(f"  {summary_path}")
    print(f"  {full_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
