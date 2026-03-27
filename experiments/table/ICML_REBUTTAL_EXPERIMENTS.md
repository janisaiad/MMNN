# ICML rebuttal experiments (multi-seed frozen RF)

Script: `icml_rebuttal_experiments.py`

## What it does

Trains low-rank **MMNN** with **`fixWb=True`** (frozen random features) for **several independent seeds**. Each seed reinitializes the frozen layers, so reported **mean ± std** answers reviewer questions about variance over RF draws.

## Run (from repo root)

Use the project environment (e.g. `uv run`):

```bash
mkdir -p experiments/table/icml_rebuttal_runs
uv run python experiments/table/icml_rebuttal_experiments.py \
  --mnist --seeds 0 1 2 3 4 --ranks 15 25 --epochs 30 \
  --out-dir experiments/table/icml_rebuttal_runs
```

CIFAR-10 (optional, long on CPU):

```bash
uv run python experiments/table/icml_rebuttal_experiments.py \
  --cifar10 --seeds 0 1 2 3 4 --ranks 15 25 --epochs 30 \
  --out-dir experiments/table/icml_rebuttal_runs_cifar
```

Quick test:

```bash
uv run python experiments/table/icml_rebuttal_experiments.py --mnist --quick --seeds 0 1 2 --ranks 15
```

## Outputs

- `icml_rebuttal_seed_sweep.json` — one entry per run (dataset, rank, seed, test acc/loss, param counts).
- `icml_rebuttal_summary.md` — aggregated mean ± std per (dataset, rank).

## Latest MNIST run (example)

See `icml_rebuttal_runs/icml_rebuttal_summary.md` after a full run.
