# Resume: baseline sweep, ranks, and results (discussion summary)

Summary of the table experiments, scripts, and findings from this discussion.

---

## 1. Setups and ranks reference

| Command | Rank | Output directory |
|--------|------|------------------|
| `--baseline-sweep` | 10 | `results_baseline_sweep_sumcos/` |
| `--baseline-sweep-rank5` | 5 | `results_baseline_sweep_sumcos_rank5/` |
| `--baseline-sweep-rank N` (e.g. 20) | N | `results_baseline_sweep_sumcos_rank{N}/` |
| `--baseline-sweep-expcos` | 10 | `results_baseline_sweep_expcos/` |

Full table: **`experiments/table/BASELINE_SWEEP_SETUPS_AND_RANKS.md`**

---

## 2. Scripts and fixes

### `plot_baseline_sweep_run.py`
- Plots for one run: (1) target function, (2) target vs prediction, (3) loss curve.
- **Single run:** `python plot_baseline_sweep_run.py <run_name> [--results-dir <dir>]`
- **All runs with a given rank:** `python plot_baseline_sweep_run.py --rank 5 [--results-dir results_baseline_sweep_sumcos_rank5]`
- Rank is read from each run’s `config.json`; `--rank N` filters which runs to plot.

### `run_scaling_law_depth_width.py` — bugs fixed
- **Done check:** Was `(out_dir / "losses.json").exists()` → skipped all runs if any file existed at root. **Fixed:** `(run_out_dir / "losses.json").exists()` per run.
- **Output directory:** Was `train_baseline_sweep_one(cfg, out_dir)` → wrote to root. **Fixed:** `train_baseline_sweep_one(cfg, run_out_dir)` so each run has its own subdir.

### `analyze_baseline_sweep.py`
- **Rank 20 support:** `--sumcos-rank20` → histogram and summary for `results_baseline_sweep_sumcos_rank20/`.
- Histogram title includes “— rank N” when using `--sumcos-rank5` or `--sumcos-rank20`.
- Graceful exit when the results directory is missing.

---

## 3. Main result tables

### 3.1 Best configs (rank 5, min_loss < 2e-2)

**Factor 5, rank 5 (2 configs):**

| Config | min_loss | final_test_err | N | bs | L |
|--------|----------|----------------|---|---|-----|---|
| f5_N1280_bs4_L3 | 2.80e-03 | 3.26e-03 | 1280 | 4 | 3 |
| f5_N1280_bs2_L2 | 9.48e-03 | 1.11e-02 | 1280 | 2 | 2 |

**Factor 4, rank 5 (4 configs):**

| Config | min_loss | final_test_err | N | bs | L |
|--------|----------|----------------|---|-----|---|
| f4_N1024_bs4_L3 | 1.42e-03 | 1.59e-03 | 1024 | 4 | 3 |
| f4_N1024_bs2_L3 | 5.50e-03 | 5.94e-03 | 1024 | 2 | 3 |
| f4_N1024_bs1_L2 | 9.37e-03 | 1.08e-02 | 1024 | 1 | 2 |
| f4_N1024_bs2_L2 | 1.06e-02 | 1.22e-02 | 1024 | 2 | 2 |

### 3.2 Parameter counts (rank 5 configs above)

| Config | total_parameters |
|--------|------------------|
| L=3 (f5_N1280_bs4_L3, f4_N1024_bs4_L3, f4_N1024_bs2_L3) | **36,880** |
| L=2 (f5_N1280_bs2_L2, f4_N1024_bs1_L2, f4_N1024_bs2_L2) | **25,611** |

Formula (W=width, r=rank, L=num_layers):  
`2*W + L*(2*W*r + W + r) + (W+1)`.

### 3.3 Batch size vs performance (rank 5)

| bs | count | worked (test<0.01) | mean(test_err) | best test_err |
|----|-------|--------------------|----------------|---------------|
| 1  | 150   | 15 (10%)           | 1.72e+00       | 9.37e-05      |
| 2  | 150   | 20 (13%)           | 1.62e+00       | 1.88e-05      |
| 4  | 150   | 19 (13%)           | 1.46e+00       | 5.18e-05      |
| 8  | 150   | 13 (9%)            | 1.36e+00       | 2.17e-04      |
| 16 | 150   | 9 (6%)             | 1.40e+00       | 3.93e-04      |

**Rank 10 (sumcos):** same trend — larger batch (8, 16) → fewer worked runs and worse best test error than bs=1,2,4.

**Conclusion:** Large batch (bs=8, 16) trained worse than small batch (1–4) in this setup. For factor 5 rank 5, only **bs=4** had a worked config (f5_N1280_bs4_L3).

---

## 4. Where things live

### Loss curves and plots (rank 5)
- **Data:** `results_baseline_sweep_sumcos_rank5/<run_name>/losses.json` (key `all_losses`).
- **Figure:** `results_baseline_sweep_sumcos_rank5/<run_name>/plot_target_prediction_loss.png` (target, prediction, loss curve).

Regenerate plots for the six best configs:
```bash
for run in f5_N1280_bs4_L3 f5_N1280_bs2_L2 f4_N1024_bs4_L3 f4_N1024_bs2_L3 f4_N1024_bs1_L2 f4_N1024_bs2_L2; do
  python experiments/table/plot_baseline_sweep_run.py "$run" --results-dir results_baseline_sweep_sumcos_rank5
done
```

### Histograms and CSVs
- **Rank 5:** `baseline_sweep_sumcos_rank5_loss_histogram.png`, `baseline_sweep_sumcos_rank5_results.csv`, `baseline_sweep_sumcos_rank5_summary.md`
- **Rank 20:** `baseline_sweep_sumcos_rank20_loss_histogram.png` (generate where `results_baseline_sweep_sumcos_rank20/` exists):
  ```bash
  python experiments/table/analyze_baseline_sweep.py --sumcos-rank20
  ```

---

## 5. Preconditioned training

- **Params:** ~25k–37k; low-rank (r=5) + width 1024 → different scales per direction.
- **Preconditioning** (natural gradient, K-FAC, Shampoo, or diagonal) is a promising direction: can help with ill-conditioning, plateau escape, and alignment with curvature; cost is affordable at this size.

---

## 6. Quick command reference

| Goal | Command |
|------|--------|
| Train rank 5 sweep | `python run_scaling_law_depth_width.py --baseline-sweep-rank5` |
| Train rank 20 sweep | `python run_scaling_law_depth_width.py --baseline-sweep-rank 20` |
| Plot all rank 5 runs | `python plot_baseline_sweep_run.py --rank 5 --results-dir results_baseline_sweep_sumcos_rank5` |
| Histogram + summary rank 5 | `python analyze_baseline_sweep.py --sumcos-rank5` |
| Histogram + summary rank 20 | `python analyze_baseline_sweep.py --sumcos-rank20` |

---

*Generated as a resume of the baseline sweep and rank discussion.*
