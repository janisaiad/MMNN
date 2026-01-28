# Log-ratio tracking during training

Track partial functions \(f_k\) and log-ratios \(R_{i,j} = \log|f_i| - \log|f_j|\) at \(x=0\) **during** training for the factor=4, rank=15 config (15 channels → 225 pairs).

## Usage

```bash
# From project root:
uv run python experiments/table/logratio/track_logratio_during_training.py [OPTIONS]
```

**Options:**

- `--config PATH` – Use `config.json` from an existing run (e.g. `.../factor4_rank15_SGD_mom0.3_lr0.01_AdaptiveStagnation/config.json`). Default: built-in factor4/rank15 SGD + AdaptiveStagnation.
- `--out-dir PATH` – Output directory. Default: `logratio/runs/factor4_rank15_SGD_mom0.3_...`.
- `--checkpoint-every N` – Compute \(f_k\), \(R\) every N epochs (default: 50).
- `--x X` – Input location for partials (default: 0.0).
- `--eps EPS` – Epsilon in \(\log(|f|+\varepsilon)\) (default: 1e-6).
- `--num-epochs N` – Override `num_epochs` (e.g. 100 for quick tests).
- `--seed N` – Random seed (default: 42).

## Outputs

For each run, the script writes into `--out-dir`:

| File | Description |
|------|-------------|
| `epochs.npy` | Checkpoint epochs (int64). |
| `times.npy` | Wall-clock time in seconds at each checkpoint. |
| `fk_x0.npy` | Partial functions at \(x=0\); shape `(n_checkpoints, 15)`. |
| `R_x0.npy` | Log-ratio matrices; shape `(n_checkpoints, 15, 15)`. |
| `trajectories_225.npy` | All 225 \(R_{i,j}\) trajectories; shape `(n_checkpoints, 225)` (row-major \(i,j\)). |
| `trajectory_max_ratio.png` | Max over \((i,j)\) of \(R_{i,j}\) vs **epoch** and vs **time** (two subplots). |
| `summary.json` | Short summary (max ratio, etc.). |
| `config.json` | Config used for the run. |

## Example

```bash
# Quick test (30 epochs, checkpoint every 10):
uv run python experiments/table/logratio/track_logratio_during_training.py \
  --config experiments/table/experiments/table/results_tune_lr_decay_L2/factor4_rank15_SGD_mom0.3_lr0.01_AdaptiveStagnation/config.json \
  --num-epochs 30 --checkpoint-every 10 --out-dir experiments/table/logratio/runs/my_run

# Full 10k-epoch run (default config, checkpoint every 50):
uv run python experiments/table/logratio/track_logratio_during_training.py
```

## Layer

Partial functions are taken from **layer 2** (second low-rank block):  
`fcs[0]→ReLU→fcs[1]→fcs[2]→ReLU→fcs[3]` → output `(batch, 15)`.

## Figure 5 configs and loss decay

The paper’s Fig. 5 uses **SGD lr 0.01, batch 160, cos(8πx), 20 epochs** (caption) or **10k epochs** (Table 3).  
**Plain SGD, no scheduler, momentum=0, 20 epochs** on cos(8πx) usually does **not** show loss decay.

**Configs that make loss decay:**

| Config | Target | Optimizer | Scheduler | Epochs | Use when |
|--------|--------|-----------|-----------|--------|----------|
| `config_fig5_adam.json` | cos(8πx) | **Adam** 0.001 | — | 100 | Fast decay; can switch to SGD when loss &lt; 1e-3 |
| `config_fig5_sgd_scheduler.json` | cos(8πx) | SGD 0.01, mom 0.3 | **AdaptiveStagnation** | 100 | Closer to paper; matches `paper_50epochs`-style |
| `config_fig5_simple.json` | **cos(2πx)** | SGD 0.01, mom 0 | **AdaptiveStagnation** | 100 | Easiest: 1 oscillation, batch 4 |

**Run examples:**

```bash
# Adam (loss decays quickly):
uv run python experiments/table/logratio/track_logratio_during_training.py \
  --config experiments/table/logratio/config_fig5_adam.json \
  --out-dir experiments/table/logratio/runs/fig5_adam

# SGD + AdaptiveStagnation + momentum (cos(8πx), like paper_50epochs):
uv run python experiments/table/logratio/track_logratio_during_training.py \
  --config experiments/table/logratio/config_fig5_sgd_scheduler.json \
  --out-dir experiments/table/logratio/runs/fig5_sgd_sched

# Simplest: cos(2πx), SGD + AdaptiveStagnation:
uv run python experiments/table/logratio/track_logratio_during_training.py \
  --config experiments/table/logratio/config_fig5_simple.json \
  --out-dir experiments/table/logratio/runs/fig5_simple
```
