# Baseline sweep: setups and ranks

All baseline-related commands in `run_scaling_law_depth_width.py`, with hidden rank and output directory.

| Command | Rank | Output directory | Target / notes |
|--------|------|------------------|----------------|
| `--baseline` | 10 | `results_stable_baseline/` | Single run: $\cos(2\pi x)$, N=width=1024, L=2 |
| `--baseline-sweep` | **10** | `results_baseline_sweep_sumcos/` | Sumcos sweep: $\sum_{k=1}^{\mathrm{factor}} \cos(2\pi k x)$, factor 1..5, N = base×factor, bs ∈ {1,2,4,8,16}, L ∈ 1..2×factor |
| `--baseline-sweep-rank5` | **5** | `results_baseline_sweep_sumcos_rank5/` | Same sumcos sweep, hidden_rank=5 |
| `--baseline-sweep-rank 20` | **20** | `results_baseline_sweep_sumcos_rank20/` | Same sumcos sweep, hidden_rank=20 |
| `--baseline-sweep-rank N` | **N** | `results_baseline_sweep_sumcos_rank{N}/` | Same sumcos sweep, hidden_rank=N (any integer) |
| `--baseline-sweep-expcos` | 10 | `results_baseline_sweep_expcos/` | Expcos sweep: $\sum_{k=0}^{\mathrm{factor}} \cos(2^k \pi x)$, factor 3 and 4 only; N = mult×2^factor |

## Summary by rank

| Rank | Output directory | Command |
|------|------------------|--------|
| 5 | `results_baseline_sweep_sumcos_rank5/` | `--baseline-sweep-rank5` |
| 10 | `results_baseline_sweep_sumcos/` | `--baseline-sweep` |
| 10 | `results_baseline_sweep_expcos/` | `--baseline-sweep-expcos` |
| 10 | `results_stable_baseline/` | `--baseline` |
| N (e.g. 20) | `results_baseline_sweep_sumcos_rank{N}/` | `--baseline-sweep-rank N` |

## Run name format (sumcos sweep)

`f{factor}_N{n_train}_bs{batch_size}_L{num_layers}`  
Example: `f3_N768_bs4_L3` = factor 3, N=768, batch_size 4, 3 layers.
