# Grokking and Representational Reorganization in Transformers

We train a small transformer on modular addition (mod 97) with strong weight decay and analyze hidden states across training. The model **grokks**: it memorizes first, then generalizes sharply after a long plateau. We ask whether that behavioral jump lines up with **measurable changes** in representations—linear decodability, effective dimensionality, and Fourier structure—and **where** in the network those changes appear.

**Takeaway:** Grokking coincides with a clear **representational reorganization**, not just a better readout: metrics move together after generalization, and **task information emerges in the second block**, not the embedding or first block.

**Key figures (overview)**

| Training and loss | Representation metrics | Layer-wise linear probes |
|:---:|:---:|:---:|
| ![Training curves](figures/fig1_training_curves.png) | ![Metrics over training](figures/fig2_metrics_over_training.png) | ![Block-level probe accuracy](figures/fig5_block_probe.png) |

## Main findings (at a glance)

- **Grokking trajectory:** Train accuracy hits 1.0 around epoch 57; test accuracy stays low until a rapid jump (~epochs 123–147), alongside falling weight norm (weight decay).
- **Linear probes:** Probe test accuracy on the final token goes from ~6% at memorization to **100%** after grokking; mid-plateau, the probe can **beat** the model’s own accuracy—structure appears before the head fully uses it.
- **Dimensionality:** Effective rank and PC counts for 90% variance **drop sharply** after grokking (high-dimensional memorization → compact solution).
- **Fourier structure:** Class-mean spectra **concentrate** into a few modes (notably frequency **k = 3** late in training), consistent with modular arithmetic on the cyclic group Z/pZ.
- **Layers:** Embedding stays at chance; block 0 improves slowly; **block 1** shows a **phase transition** to full probe accuracy at grokking.

## Full technical write-up

Motivation, experimental setup, methods (probes, PCA, Fourier), **all tables and figure captions**, summary matrix, discussion, and limitations:

**[documents/grokking_analysis_details.md](documents/grokking_analysis_details.md)**

## Reproduction

### Requirements

```bash
pip install torch numpy matplotlib pandas scikit-learn
```

### Train (transformer, grokking setting)

```bash
python train_modadd.py \
  --arch transformer \
  --train-frac 0.3 \
  --d-model 128 --n-heads 4 --d-mlp 512 --n-layers 2 \
  --weight-decay 1.0 \
  --learning-rate 1e-3 \
  --num-epochs 1000 \
  --seed 0
```

### Analysis pipeline (after a run exists under `runs/`)

```bash
python summarize_runs.py
python select_checkpoints.py --run-dir runs/<run_name>
python extract_activations.py --run-dir runs/<run_name> --all-checkpoints
python extract_block_activations.py --run-dir runs/<run_name> \
  --checkpoints epoch_0050 epoch_0100 epoch_0150 epoch_0200 epoch_0500 epoch_1000
python analyze_representations.py --run-dir runs/<run_name> --all
python plot_analysis.py --run-dir runs/<run_name>
python inspect_run.py --run-dir runs/<run_name>
```

Artifacts land under `runs/<run_name>/` (metrics, analysis, plots). Copy or compare plots to `figures/` for documentation if you like.

### Repository layout

```
grokking/
├── README.md
├── documents/
│   └── grokking_analysis_details.md   ← full analysis write-up
├── requirements.txt
├── train_modadd.py
├── summarize_runs.py
├── select_checkpoints.py
├── extract_activations.py
├── extract_block_activations.py
├── analyze_representations.py
├── plot_analysis.py
├── inspect_run.py
├── src/
├── runs/                 ← created by training (not in git by default)
├── figures/              ← key figures for the README
└── results/              ← e.g. run_summary.csv
```

## References

- Power et al. (2022). [Grokking: Generalization beyond overfitting on small algorithmic datasets](https://arxiv.org/abs/2201.02177).
- Nanda et al. (2023). [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217).
- Liu et al. (2022). [Towards understanding grokking](https://arxiv.org/abs/2205.10343).
