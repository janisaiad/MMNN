# Exhaustive Resume: Tables and Results from the Scaling-Law / Table Discussion

This document summarizes **all** tables, result directories, and findings from the discussion on scaling laws for depth and width, stable baselines, baseline sweeps (sumcos / expcos), mean-field experiments, and the SinQuad scaling script.

---

## 1. Discussion Context and Goals

- **Goal**: Find scaling laws relating training data \(N\), width \(W\), depth \(L\), and frequency (factor); understand training stability for large depth and width.
- **Observation**: More layers can lead to unstable training (e.g. L=56 huge errors, L=80 NaN in frequency benchmark). Width should scale with training data; depth with frequency.
- **Approach**: (1) Establish a stable baseline (cos \(2\pi x\)), (2) Sweep over factor, \(N\), batch size, and layers with target \(\sum_{k=1}^{\mathrm{factor}} \cos(2\pi k x)\), (3) Use same saving/plotting strategy (losses.json, min_loss checkpoints, loss curves, config.json), (4) Add SinQuad scaling experiment for large depth/width.

---

## 2. Result Directories Index

| Directory | Description |
|-----------|-------------|
| `results_stable_baseline/` | cos(2πx), N=width=1024, rank=10, L=2, SGD + AdaptiveStagnation, 4 momenta |
| `results_baseline_sweep_sumcos/` | Target \(\sum_{k=1}^f \cos(2\pi k x)\), factor 1..5, N = base×factor, rank=10 |
| `results_baseline_sweep_sumcos_rank5/` | Same sumcos target, **rank=5** |
| `results_baseline_sweep_expcos/` | Target \(\sum_{k=0}^f \cos(2^k \pi x)\), factor 3 and 4 only |
| `results_baseline_sweep/` | Earlier baseline sweep (cos(2π factor x) style) |
| `results_frequency_layer_scaling/` | Multi-frequency target, Adam, StepLR (from run_scaling_law_depth_width.py --train) |
| `results_sinquad_depth_width_scaling/` | SinQuad target, width 1024, ranks 5/10/20, depths 8/10/12/15 (script: run_depth_width_scaling_sinquad.py) |
| `meanfield_two_step_results/` | Mean-field 2-step function, coupling & channel specialization |
| `meanfield_cosine_results/` | Mean-field cosine (cos 12πx), channel shares and log-ratios |
| `results_1d_comprehensive/` | 1D flowbench/pinnacle, fixWb × rank × seeds (see RESULTS_SUMMARY.md) |
| `results_tune_lr_decay_L2/` | LR decay tuning, L=2, factor=1, SGD + AdaptiveStagnation |
| `results_sumcos_lowlr_table_below_1e2/` | Sumcos with low LR table (SGD) |
| `results_sumcos_lowlr_table_below_1e2_adam/` | Same with Adam |
| `results_sumcos_selected_rerun/` | Selected sumcos configs rerun |

---

## 3. Stable Baseline: cos(2πx), N = Width = 1024

**Config**: Target cos(2πx) on \([-1,1]\), N_train=1024, width=1024, rank=10, depth=2, batch_size=4, 250 epochs, SGD lr=0.01, AdaptiveStagnation (lr sequence [0.01, 0.005, 0.001, 0.0005, 0.0001], window=10, min_epochs_before_reduce=20).

**Output**: `results_stable_baseline/`

### 3.1 Results Table (all 4 momenta)

| Config name | Momentum | Final train error | Final test error | Epochs | Time (s) |
|-------------|----------|-------------------|------------------|--------|-----------|
| cos2pi_N1024_W1024_rank10_L2_SGD_mom0.0_AdaptiveStagnation | 0.0 | (in results.json) | (in results.json) | 250 | — |
| cos2pi_N1024_W1024_rank10_L2_SGD_mom0.3_AdaptiveStagnation | 0.3 | — | — | 250 | — |
| cos2pi_N1024_W1024_rank10_L2_SGD_mom0.6_AdaptiveStagnation | 0.6 | — | — | 250 | — |
| cos2pi_N1024_W1024_rank10_L2_SGD_mom0.7_AdaptiveStagnation | 0.7 | 1.71e-06 | 1.91e-06 | 250 | 27.9 |

**Example (mom=0.7)**: final_train_error=1.707e-06, final_test_error=1.912e-06, total_parameters=46101, lr_reduction_epochs=[62, 129].

---

## 4. Baseline Sweep: Sumcos (rank=10)

**Target**: \(\sum_{k=1}^{\mathrm{factor}} \cos(2\pi k x)\).  
**Sweep**: factor ∈ {1,2,3,4,5}; N = [16,32,64,128,256] × factor; batch_size ∈ {1,2,4,8,16}; L ∈ {1..2×factor}; max 10K epochs; lr 1e-2, divide by 2 on stagnation; stop when lr < 1e-6.  
**Worked** = final test error < 0.01; **Failed** = test error ≥ 0.5 or NaN/Inf.

**Output**: `results_baseline_sweep_sumcos/`  
**Summary file**: `baseline_sweep_sumcos_summary.md`

### 4.1 Summary counts

- **Worked** (test err < 0.01): **73** configs  
- **Failed**: **284** configs  
- **Total completed**: 429  

### 4.2 Best configs (by final test error)

| config | factor | N | bs | L | final_test_err | final_train_err | epochs |
|--------|--------|---|-----|---|----------------|-----------------|--------|
| f1_N256_bs1_L2 | 1 | 256 | 1 | 2 | 2.6630e-06 | 2.9867e-06 | 10000 |
| f1_N256_bs2_L2 | 1 | 256 | 2 | 2 | 1.7445e-05 | 1.9327e-05 | 1062 |
| f2_N512_bs2_L3 | 2 | 512 | 2 | 3 | 1.9382e-05 | 1.9263e-05 | 1042 |
| f1_N256_bs4_L2 | 1 | 256 | 4 | 2 | 3.4366e-05 | 3.7932e-05 | 1062 |
| f1_N128_bs1_L2 | 1 | 128 | 1 | 2 | 7.7556e-05 | 1.0740e-04 | 589 |
| f2_N512_bs4_L4 | 2 | 512 | 4 | 4 | 7.8730e-05 | 7.8352e-05 | 861 |
| f1_N128_bs2_L2 | 1 | 128 | 2 | 2 | 8.0275e-05 | 1.0634e-04 | 669 |
| f1_N256_bs4_L1 | 1 | 256 | 4 | 1 | 1.7293e-04 | 1.8626e-04 | 1350 |
| f1_N256_bs1_L1 | 1 | 256 | 1 | 1 | 1.7793e-04 | 1.9738e-04 | 543 |
| f1_N256_bs2_L1 | 1 | 256 | 2 | 1 | 2.0222e-04 | 2.2042e-04 | 605 |

### 4.3 Table by factor (sumcos rank=10)

| factor | N range | total | worked | failed | best test err |
|--------|---------|-------|--------|--------|---------------|
| 1 | 16–256 | 50 | 36 | 14 | 2.6630e-06 |
| 2 | 32–512 | 100 | 27 | 73 | 1.9382e-05 |
| 3 | 48–768 | 150 | 10 | 140 | 1.3396e-03 |
| 4 | 64–512 | 129 | 0 | 129 | 2.0452e-02 |

**CSV**: `baseline_sweep_results.csv`. **Histogram**: `baseline_sweep_loss_histogram.png`.

---

## 5. Baseline Sweep: Sumcos Rank 5

**Same target and sweep logic as sumcos, but hidden_rank=5.**

**Output**: `results_baseline_sweep_sumcos_rank5/`  
**Summary**: `baseline_sweep_sumcos_rank5_summary.md`

### 5.1 Summary counts

- **Worked**: **76** configs  
- **Failed**: **566** configs  
- **Total completed**: 750  

### 5.2 Best configs (rank=5)

| config | factor | N | bs | L | final_test_err | final_train_err | epochs |
|--------|--------|---|-----|---|----------------|-----------------|--------|
| f1_N256_bs2_L2 | 1 | 256 | 2 | 2 | 1.8807e-05 | 2.0153e-05 | 1433 |
| f1_N256_bs2_L1 | 1 | 256 | 2 | 1 | 4.4141e-05 | 4.7591e-05 | 1202 |
| f1_N256_bs4_L2 | 1 | 256 | 4 | 2 | 5.1790e-05 | 5.4960e-05 | 1062 |
| f1_N256_bs1_L2 | 1 | 256 | 1 | 2 | 9.3687e-05 | 1.0587e-04 | 537 |
| f2_N512_bs2_L2 | 2 | 512 | 2 | 2 | 1.1339e-04 | 1.1298e-04 | 591 |
| f2_N512_bs2_L3 | 2 | 512 | 2 | 3 | 1.2469e-04 | 1.2463e-04 | 581 |
| f1_N128_bs2_L2 | 1 | 128 | 2 | 2 | 2.0254e-04 | 2.3739e-04 | 601 |
| f1_N256_bs8_L2 | 1 | 256 | 8 | 2 | 2.1739e-04 | 2.3117e-04 | 1123 |
| f1_N256_bs1_L1 | 1 | 256 | 1 | 1 | 2.3299e-04 | 2.5383e-04 | 576 |
| f1_N16_bs16_L2 | 1 | 16 | 16 | 2 | 3.9344e-04 | 1.1768e-04 | 10000 |

### 5.3 Table by factor (sumcos rank=5)

| factor | N range | total | worked | failed | best test err |
|--------|---------|-------|--------|--------|---------------|
| 1 | 16–256 | 50 | 34 | 16 | 1.8807e-05 |
| 2 | 32–512 | 100 | 27 | 73 | 1.1339e-04 |
| 3 | 48–768 | 150 | 12 | 138 | 5.2597e-04 |
| 4 | 64–1024 | 200 | 2 | 198 | 1.5903e-03 |
| 5 | 80–1280 | 250 | 1 | 249 | 3.2620e-03 |

**CSV**: `baseline_sweep_sumcos_rank5_results.csv`. **Histogram**: `baseline_sweep_sumcos_rank5_loss_histogram.png`.

---

## 6. Baseline Sweep: Expcos (factors 3 and 4)

**Target**: \(\sum_{k=0}^{\mathrm{factor}} \cos(2^k \pi x)\). N = mult × 2^factor (mult ∈ {4,8,16}). Same lr/epoch/checkpoint strategy.

**Output**: `results_baseline_sweep_expcos/`  
**Summary**: `baseline_sweep_expcos_summary.md`

### 6.1 Summary

- **Worked**: **0** configs  
- **Failed**: **196** configs  
- **Total completed**: 196  

### 6.2 Table by factor (expcos)

| factor | N range | total | worked | failed | best test err |
|--------|---------|-------|--------|--------|---------------|
| 3 | 32–128 | 90 | 0 | 90 | 5.0374e-01 |
| 4 | 64–256 | 106 | 0 | 106 | 7.7963e-01 |

Best configs listed in summary are still above 0.5 test error (e.g. f3_N128_bs8_L3 ≈ 0.50).

---

## 7. Original Baseline Sweep (cos(2π factor x))

**Output**: `results_baseline_sweep/`  
**Summary**: `baseline_sweep_summary.md`

- **Worked**: 52 configs  
- **Failed**: 632 configs  
- **Total completed**: 761  
- Best: f1_N256_bs1_L2 test_err=2.6630e-06, L=2, N=256, bs=1.

---

## 8. Mean-Field Two-Step Experiment

**Setup**: Two-sided step function; mean-field ODE vs finite-width MMNN; coupling distance and channel specialization over time.

**Output**: `meanfield_two_step_results/`  
**File**: `results.json`

### 8.1 Coupling distance (example)

- distance: 4.029  
- w1_max_diff: 4.029, w2_max_diff: 3.756  
- w1_mean_diff: 0.485, w2_mean_diff: 0.963  

### 8.2 Channel specialization (spike locations -0.5 and 0.5)

- **time 0**: spike_0 shares ≈ [0.916, 0.084], spike_1 ≈ [0.569, 0.431].  
- **time 750–3000**: spike_0 shares ≈ [0.972, 0.028], spike_1 ≈ [0.943, 0.057]; log-ratios increase (channel specialization).  

**Parameters**: n1=n2=1000, r=2, t_span=[0,3000], dt=0.1.  
**Plots**: `meanfield_channel_specialization.png`, `weight_distributions_through_time.png`.

---

## 9. Mean-Field Cosine Results

**Target**: cos(12πx). Multiple ranks {2,5,10,15,20,25}, depth 6, width 0.12.  
**Output**: `meanfield_cosine_results/` — results.json contains channel shares and log-ratios at several locations (0, 0.5, -0.5) and layers.

---

## 10. 1D Comprehensive Benchmark (RESULTS_SUMMARY.md)

**Source**: `RESULTS_SUMMARY.md` (1D flowbench / pinnacle).

- **Total**: 84 networks (fixWb × rank × seeds).  
- **Best test error**: 8.24e-01 (flowbench, fixWb=False, rank=15, seed=123).  
- **Ranks**: 3, 6, 10, 15, 25, 50, 1024.  
- **Finding**: Ranks 15–25 best for fixWb=False; fixWb=True with low ranks (6,10) problematic.

---

## 11. Frequency / Layer Scaling (run_scaling_law_depth_width.py)

- **Dir**: `results_frequency_layer_scaling/`  
- **Configs**: Multi-frequency target, Adam, StepLR; varying freq_multiplier, rank, num_layers.  
- **Summary**: `frequency_layer_scaling_summary.txt` (if generated by --analyze).  
- **Script**: `python run_scaling_law_depth_width.py --baseline` (stable baseline), `--baseline-sweep` (sumcos), `--baseline-sweep-expcos` (expcos), `--train` / `--analyze`.

---

## 12. SinQuad Depth/Width Scaling (Planned)

**Script**: `run_depth_width_scaling_sinquad.py`  
**Target**: \(\cos(36\pi x^2) - 0.8\cos(17\pi (x+0.5)^2)\) on \([-2,2]\).  
**Constants**: width=1024, ranks=[5,10,20], depths=[8,10,12,15], lr=1e-2, AdaptiveStagnation, N_train=2000, batch_size=64, max 10K epochs.

**Output dir**: `results_sinquad_depth_width_scaling/` (created when you run the script).  
**Naming**: `rank{R}_L{L}_W1024` (e.g. rank5_L8_W1024).  
**Saved per run**: losses.json, config.json, model_parameters.pth, params_at_div_1.2_k/, loss_curve.png, final_prediction.png; after all runs: summary.json, summary_heatmaps.png.

**Usage**:
```bash
python run_depth_width_scaling_sinquad.py           # run all 12 configs
python run_depth_width_scaling_sinquad.py --plot   # regenerate plots only
```

*(No result table here yet; run the script to populate.)*

---

## 13. Saving and Plotting Strategy (Unified)

Across the discussed experiments:

- **Per run**:  
  - `config.json`: full config  
  - `losses.json`: all_losses, all_lrs, lr_reduction_epochs, min_loss_checkpoints, final_train_error, final_test_error, epochs_run, etc.  
  - `model_parameters.pth`: final state dict  
  - Checkpoints when min_loss < init_loss/1.2^k: `params_at_div_1.2_{k}/model_parameters.pth` (no per-checkpoint epoch_info.json)  
- **Plots**: loss_curve.png (with lr reduction markers), final_prediction.png (learned vs true).  
- **Analysis**: Summaries in .md; CSVs and histograms (e.g. baseline_sweep_sumcos, rank5) with "worked" = test err < 0.01, "failed" = test err ≥ 0.5 or NaN/Inf.

---

## 14. Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analyze_baseline_sweep.py` | Load losses.json from a sweep dir; build table, worked/failed counts, write .md and optional CSV/histogram. Use --sumcos, --sumcos-rank5, --sumcos-rank20, --expcos for each variant. |
| `run_scaling_law_depth_width.py` | Main runner: --baseline, --baseline-sweep, --baseline-sweep-expcos, --train, --analyze. |
| `run_depth_width_scaling_sinquad.py` | SinQuad scaling: 12 configs (3 ranks × 4 depths), same saving/plotting as above. |
| `plot_table_below_lowlr_losses.py` | Plot loss curves for runs in a given results dir (e.g. results_sumcos_lowlr_table_below_1e2). |
| `plot_selected_sumcos_configs.py` | Plot selected sumcos configs from results_sumcos_selected_rerun. |

---

## 15. Quick Reference: Key Numbers

| Experiment | Worked | Failed | Total | Best test err (example) |
|------------|--------|--------|-------|-------------------------|
| Stable baseline (mom 0.7) | 1 run | 0 | 1 | 1.91e-06 |
| Sumcos rank=10 | 73 | 284 | 429 | 2.66e-06 (f1_N256_bs1_L2) |
| Sumcos rank=5 | 76 | 566 | 750 | 1.88e-05 (f1_N256_bs2_L2) |
| Expcos f3,f4 | 0 | 196 | 196 | 5.04e-01 |
| Original baseline sweep | 52 | 632 | 761 | 2.66e-06 |

---

*Document generated to capture all tables and results from the scaling-law / table discussion. Update this file when new sweeps or analyses are run.*
