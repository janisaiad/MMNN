# Exhaustive results: baseline sweep and ranks

Complete reference of setups, scripts, result tables, and file locations from the baseline-sweep and rank-5/10/20 discussion.  
All paths are relative to `experiments/table/` unless noted.

---

## 1. Setups and ranks (full reference)

### 1.1 Commands and output directories

| Command | Rank | Output directory | Target / notes |
|--------|------|------------------|----------------|
| `--baseline` | 10 | `results_stable_baseline/` | Single run: $\cos(2\pi x)$, N=width=1024, L=2 |
| `--baseline-sweep` | **10** | `results_baseline_sweep_sumcos/` | Sumcos: $\sum_{k=1}^{\mathrm{factor}} \cos(2\pi k x)$, factor 1..5 |
| `--baseline-sweep-rank5` | **5** | `results_baseline_sweep_sumcos_rank5/` | Same sumcos sweep, hidden_rank=5 |
| `--baseline-sweep-rank 20` | **20** | `results_baseline_sweep_sumcos_rank20/` | Same sumcos, hidden_rank=20 |
| `--baseline-sweep-rank N` | **N** | `results_baseline_sweep_sumcos_rank{N}/` | Same sumcos, any integer N |
| `--baseline-sweep-expcos` | 10 | `results_baseline_sweep_expcos/` | Expcos: $\sum_{k=0}^{\mathrm{factor}} \cos(2^k \pi x)$, factor 3,4 only |

### 1.2 Sweep definition (sumcos)

- **Target:** $\sum_{k=1}^{\mathrm{factor}} \cos(2\pi k x)$ on $[-1,1]$.
- **Factors:** 1, 2, 3, 4, 5.
- **N (train size):** `SWEEP_BASE_N * factor` with `SWEEP_BASE_N = [16, 32, 64, 128, 256]`.
- **Batch sizes:** 1, 2, 4, 8, 16 (only configs with `bs ≤ N`).
- **Layers:** 1 to 2×factor.
- **Epochs:** up to 10000; stop when lr < 1e-6 (AdaptiveStagnation, lr halved on plateau).
- **Run name:** `f{factor}_N{n_train}_bs{batch_size}_L{num_layers}`.

### 1.3 Summary by rank

| Rank | Output directory | Command |
|------|------------------|--------|
| 5 | `results_baseline_sweep_sumcos_rank5/` | `--baseline-sweep-rank5` |
| 10 | `results_baseline_sweep_sumcos/` | `--baseline-sweep` |
| 10 | `results_baseline_sweep_expcos/` | `--baseline-sweep-expcos` |
| 10 | `results_stable_baseline/` | `--baseline` |
| N | `results_baseline_sweep_sumcos_rank{N}/` | `--baseline-sweep-rank N` |

---

## 2. Scripts: changes and usage

### 2.1 `plot_baseline_sweep_run.py`

- **Purpose:** For each run: (1) target function, (2) target vs prediction, (3) loss curve.
- **Single run:**  
  `python plot_baseline_sweep_run.py <run_name> [--results-dir <dir>]`  
  Example: `python plot_baseline_sweep_run.py f3_N768_bs4_L3 --results-dir results_baseline_sweep_sumcos`
- **All runs with a given rank:**  
  `python plot_baseline_sweep_run.py --rank 5 [--results-dir results_baseline_sweep_sumcos_rank5]`
- **Behaviour:** Rank is read from each run’s `config.json`; `--rank N` restricts to runs with `hidden_rank == N`. Output: `<run_dir>/plot_target_prediction_loss.png`.

### 2.2 `run_scaling_law_depth_width.py` — bugs fixed

- **Bug 1 — “done” check:**  
  **Before:** `(out_dir / "losses.json").exists()` → one file at root caused all runs to be skipped.  
  **After:** `(run_out_dir / "losses.json").exists()` so each run is skipped only if its own subdir has `losses.json`.
- **Bug 2 — output directory:**  
  **Before:** `train_baseline_sweep_one(cfg, out_dir)` → all runs wrote to the same root directory.  
  **After:** `train_baseline_sweep_one(cfg, run_out_dir)` so each run writes to `results_.../f{f}_N{n}_bs{bs}_L{L}/`.
- **Rerun list:** `baseline_sweep_rerun.txt` — configs listed there are always re-run (one name per line; `#` comments). Current content: `f3_N768_bs4_L3`.

### 2.3 `analyze_baseline_sweep.py`

- **Added:** `--sumcos-rank20` for `results_baseline_sweep_sumcos_rank20/` (summary, CSV, histogram).
- **Added:** Optional `rank_label` so histogram title can include “— rank N” for rank 5 and rank 20.
- **Behaviour:** If the chosen results directory does not exist, script exits with a clear message instead of raising.

---

## 3. Rank 5 — full result tables

### 3.1 Global counts (rank 5)

- **Total runs:** 750.
- **Worked** (final test error < 0.01): **76**.
- **Failed** (test error ≥ 0.5 or NaN/Inf): **674**.

### 3.2 By factor (rank 5)

| factor | N range | total | worked | failed | best test err |
|--------|---------|-------|--------|--------|---------------|
| 1 | 16–256 | 50 | 34 | 16 | 1.8807e-05 |
| 2 | 32–512 | 100 | 27 | 73 | 1.1339e-04 |
| 3 | 48–768 | 150 | 12 | 138 | 5.2597e-04 |
| 4 | 64–1024 | 200 | 2 | 198 | 1.5903e-03 |
| 5 | 80–1280 | 250 | 1 | 249 | 3.2620e-03 |

### 3.3 By factor and batch size (rank 5) — worked / total

| factor | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 |
|--------|------|------|------|------|-------|
| 1 | 6/10 | 8/10 | 7/10 | 7/10 | 6/10 |
| 2 | 6/20 | 7/20 | 7/20 | 4/20 | 3/20 |
| 3 | 3/30 | 4/30 | 3/30 | 2/30 | 0/30 |
| 4 | 0/40 | 1/40 | 1/40 | 0/40 | 0/40 |
| 5 | 0/50 | 0/50 | **1/50** | 0/50 | 0/50 |

For factor 5, rank 5, only **bs=4** has a worked config: **f5_N1280_bs4_L3**.

### 3.4 By batch size only (rank 5) — aggregate over all factors

| bs | count | worked (test<0.01) | mean(test_err) | mean(min_loss) | best test err |
|----|-------|--------------------|----------------|----------------|---------------|
| 1 | 150 | 15 (10%) | 1.7229e+00 | 1.7419e+00 | 9.37e-05 |
| 2 | 150 | 20 (13%) | 1.6192e+00 | 1.6365e+00 | 1.88e-05 |
| 4 | 150 | 19 (13%) | 1.4560e+00 | 1.4912e+00 | 5.18e-05 |
| 8 | 150 | 13 (9%) | 1.3643e+00 | 1.4038e+00 | 2.17e-04 |
| 16 | 150 | 9 (6%) | 1.4011e+00 | 1.4404e+00 | 3.93e-04 |

### 3.5 Configs with min_loss < 2e-2 (rank 5)

**Factor 5 (2 configs):**

| config | min_loss | final_test_err | N | bs | L |
|--------|----------|----------------|-----|---|-----|
| f5_N1280_bs4_L3 | 2.8049e-03 | 3.2620e-03 | 1280 | 4 | 3 |
| f5_N1280_bs2_L2 | 9.4821e-03 | 1.1089e-02 | 1280 | 2 | 2 |

**Factor 4 (4 configs):**

| config | min_loss | final_test_err | N | bs | L |
|--------|----------|----------------|-----|---|-----|
| f4_N1024_bs4_L3 | 1.4153e-03 | 1.5903e-03 | 1024 | 4 | 3 |
| f4_N1024_bs2_L3 | 5.5035e-03 | 5.9413e-03 | 1024 | 2 | 3 |
| f4_N1024_bs1_L2 | 9.3721e-03 | 1.0789e-02 | 1024 | 1 | 2 |
| f4_N1024_bs2_L2 | 1.0618e-02 | 1.2203e-02 | 1024 | 2 | 2 |

### 3.6 Parameter counts (rank 5, these configs)

| config | total_parameters |
|--------|------------------|
| f5_N1280_bs4_L3 | 36,880 |
| f5_N1280_bs2_L2 | 25,611 |
| f4_N1024_bs4_L3 | 36,880 |
| f4_N1024_bs2_L3 | 36,880 |
| f4_N1024_bs1_L2 | 25,611 |
| f4_N1024_bs2_L2 | 25,611 |

**Formula (W=hidden_width, r=hidden_rank, L=num_layers):**  
`2*W + L*(2*W*r + W + r) + (W+1)`  
With W=1024, r=5: L=2 → 25,611; L=3 → 36,880.

### 3.7 All 76 worked configs (rank 5), sorted by final_test_error

| # | config | factor | N | bs | L | final_test_err | final_train_err | min_loss | epochs |
|---|--------|--------|---|-----|---|----------------|-----------------|----------|--------|
| 1 | f1_N256_bs2_L2 | 1 | 256 | 2 | 2 | 1.8807e-05 | 2.0153e-05 | 2.0152e-05 | 1433 |
| 2 | f1_N256_bs2_L1 | 1 | 256 | 2 | 1 | 4.4141e-05 | 4.7591e-05 | 4.7588e-05 | 1202 |
| 3 | f1_N256_bs4_L2 | 1 | 256 | 4 | 2 | 5.1790e-05 | 5.4960e-05 | 5.4958e-05 | 1062 |
| 4 | f1_N256_bs1_L2 | 1 | 256 | 1 | 2 | 9.3687e-05 | 1.0587e-04 | 1.0524e-04 | 537 |
| 5 | f2_N512_bs2_L2 | 2 | 512 | 2 | 2 | 1.1339e-04 | 1.1298e-04 | 1.1278e-04 | 591 |
| 6 | f2_N512_bs2_L3 | 2 | 512 | 2 | 3 | 1.2469e-04 | 1.2463e-04 | 1.2307e-04 | 581 |
| 7 | f1_N128_bs2_L2 | 1 | 128 | 2 | 2 | 2.0254e-04 | 2.3739e-04 | 2.3726e-04 | 601 |
| 8 | f1_N256_bs8_L2 | 1 | 256 | 8 | 2 | 2.1739e-04 | 2.3117e-04 | 2.3116e-04 | 1123 |
| 9 | f1_N256_bs1_L1 | 1 | 256 | 1 | 1 | 2.3299e-04 | 2.5383e-04 | 2.5318e-04 | 576 |
| 10 | f1_N16_bs16_L2 | 1 | 16 | 16 | 2 | 3.9344e-04 | 1.1768e-04 | 1.1768e-04 | 10000 |
| 11 | f1_N128_bs4_L2 | 1 | 128 | 4 | 2 | 3.9356e-04 | 4.8056e-04 | 4.8016e-04 | 562 |
| 12 | f1_N128_bs8_L2 | 1 | 128 | 8 | 2 | 4.9362e-04 | 5.9057e-04 | 5.9041e-04 | 668 |
| 13 | f3_N768_bs4_L4 | 3 | 768 | 4 | 4 | 5.2597e-04 | 4.8491e-04 | 4.7947e-04 | 779 |
| 14 | f2_N512_bs8_L3 | 2 | 512 | 8 | 3 | 5.2927e-04 | 5.2834e-04 | 5.2804e-04 | 734 |
| 15 | f1_N256_bs4_L1 | 1 | 256 | 4 | 1 | 5.3459e-04 | 5.6770e-04 | 5.6747e-04 | 685 |
| 16 | f1_N128_bs1_L2 | 1 | 128 | 1 | 2 | 6.4565e-04 | 8.2650e-04 | 7.5627e-04 | 435 |
| 17 | f2_N512_bs4_L3 | 2 | 512 | 4 | 3 | 7.0865e-04 | 7.1092e-04 | 7.0365e-04 | 631 |
| 18 | f2_N256_bs4_L3 | 2 | 256 | 4 | 3 | 8.1173e-04 | 8.9953e-04 | 8.9708e-04 | 655 |
| 19 | f2_N512_bs2_L1 | 2 | 512 | 2 | 1 | 9.2568e-04 | 9.2437e-04 | 9.1709e-04 | 728 |
| 20 | f1_N64_bs2_L2 | 1 | 64 | 2 | 2 | 9.7215e-04 | 1.5446e-03 | 1.5268e-03 | 475 |
| 21 | f2_N512_bs1_L3 | 2 | 512 | 1 | 3 | 9.7929e-04 | 9.7683e-04 | 9.4616e-04 | 544 |
| 22 | f3_N768_bs4_L3 | 3 | 768 | 4 | 3 | 1.0722e-03 | 9.9142e-04 | 9.7441e-04 | 583 |
| 23 | f2_N512_bs4_L2 | 2 | 512 | 4 | 2 | 1.0792e-03 | 1.0753e-03 | 1.0484e-03 | 585 |
| 24 | f1_N128_bs1_L1 | 1 | 128 | 1 | 1 | 1.3124e-03 | 1.6219e-03 | 1.5279e-03 | 423 |
| 25 | f1_N256_bs8_L1 | 1 | 256 | 8 | 1 | 1.3363e-03 | 1.4083e-03 | 1.4080e-03 | 669 |
| 26 | f1_N128_bs2_L1 | 1 | 128 | 2 | 1 | 1.3828e-03 | 1.6775e-03 | 1.6561e-03 | 467 |
| 27 | f2_N512_bs4_L4 | 2 | 512 | 4 | 4 | 1.3934e-03 | 1.3896e-03 | 1.3863e-03 | 877 |
| 28 | f2_N512_bs8_L2 | 2 | 512 | 8 | 2 | 1.4156e-03 | 1.4134e-03 | 1.4121e-03 | 693 |
| 29 | f1_N256_bs16_L2 | 1 | 256 | 16 | 2 | 1.5005e-03 | 1.5758e-03 | 1.5758e-03 | 696 |
| 30 | f2_N512_bs1_L1 | 2 | 512 | 1 | 1 | 1.5187e-03 | 1.5161e-03 | 1.4982e-03 | 631 |
| 31 | f4_N1024_bs4_L3 | 4 | 1024 | 4 | 3 | 1.5903e-03 | 1.4192e-03 | 1.4153e-03 | 739 |
| 32 | f3_N768_bs8_L3 | 3 | 768 | 8 | 3 | 1.5992e-03 | 1.5270e-03 | 1.5230e-03 | 706 |
| 33 | f3_N384_bs2_L2 | 3 | 384 | 2 | 2 | 1.9548e-03 | 2.0603e-03 | 2.0092e-03 | 701 |
| 34 | f1_N128_bs4_L1 | 1 | 128 | 4 | 1 | 2.0366e-03 | 2.4010e-03 | 2.3932e-03 | 518 |
| 35 | f1_N64_bs1_L2 | 1 | 64 | 1 | 2 | 2.1116e-03 | 3.1459e-03 | 3.0126e-03 | 496 |
| 36 | f2_N256_bs2_L3 | 2 | 256 | 2 | 3 | 2.2555e-03 | 2.3929e-03 | 2.2971e-03 | 572 |
| 37 | f2_N512_bs1_L2 | 2 | 512 | 1 | 2 | 2.3430e-03 | 2.3380e-03 | 2.2504e-03 | 399 |
| 38 | f2_N256_bs1_L2 | 2 | 256 | 1 | 2 | 2.4634e-03 | 2.7636e-03 | 2.7024e-03 | 518 |
| 39 | f1_N256_bs16_L1 | 1 | 256 | 16 | 1 | 2.5564e-03 | 2.6749e-03 | 2.6743e-03 | 733 |
| 40 | f2_N256_bs2_L2 | 2 | 256 | 2 | 2 | 2.7476e-03 | 3.0907e-03 | 2.8225e-03 | 478 |
| 41 | f2_N512_bs16_L3 | 2 | 512 | 16 | 3 | 2.7868e-03 | 2.7846e-03 | 2.7837e-03 | 1132 |
| 42 | f3_N768_bs1_L3 | 3 | 768 | 1 | 3 | 2.7908e-03 | 2.6731e-03 | 2.6691e-03 | 802 |
| 43 | f2_N256_bs4_L2 | 2 | 256 | 4 | 2 | 2.8482e-03 | 3.1233e-03 | 3.0805e-03 | 649 |
| 44 | f3_N768_bs2_L2 | 3 | 768 | 2 | 2 | 2.8559e-03 | 2.6798e-03 | 2.5853e-03 | 458 |
| 45 | f2_N512_bs16_L2 | 2 | 512 | 16 | 2 | 2.9039e-03 | 2.8995e-03 | 2.8975e-03 | 1014 |
| 46 | f3_N768_bs2_L1 | 3 | 768 | 2 | 1 | 3.2084e-03 | 3.1023e-03 | 3.0703e-03 | 708 |
| 47 | f2_N512_bs4_L1 | 2 | 512 | 4 | 1 | 3.2460e-03 | 3.2417e-03 | 3.2263e-03 | 755 |
| 48 | **f5_N1280_bs4_L3** | 5 | 1280 | 4 | 3 | **3.2620e-03** | 2.8449e-03 | 2.8049e-03 | 713 |
| 49 | f3_N768_bs8_L2 | 3 | 768 | 8 | 2 | 3.3853e-03 | 3.2067e-03 | 3.1873e-03 | 732 |
| 50 | f1_N64_bs2_L1 | 1 | 64 | 2 | 1 | 3.5499e-03 | 5.0795e-03 | 5.0001e-03 | 454 |
| 51 | f1_N64_bs4_L2 | 1 | 64 | 4 | 2 | 3.5851e-03 | 4.8100e-03 | 4.6867e-03 | 527 |
| 52 | f3_N768_bs4_L2 | 3 | 768 | 4 | 2 | 3.5975e-03 | 3.3868e-03 | 3.2763e-03 | 541 |
| 53 | f1_N128_bs16_L2 | 1 | 128 | 16 | 2 | 3.6905e-03 | 4.0997e-03 | 4.0985e-03 | 850 |
| 54 | f2_N128_bs1_L2 | 2 | 128 | 1 | 2 | 3.7658e-03 | 4.3229e-03 | 4.3215e-03 | 858 |
| 55 | f3_N768_bs2_L3 | 3 | 768 | 2 | 3 | 3.7815e-03 | 3.6060e-03 | 3.5583e-03 | 562 |
| 56 | f2_N256_bs1_L1 | 2 | 256 | 1 | 1 | 3.8468e-03 | 4.0939e-03 | 4.0495e-03 | 603 |
| 57 | f2_N512_bs8_L4 | 2 | 512 | 8 | 4 | 4.0657e-03 | 4.0687e-03 | 4.0680e-03 | 1083 |
| 58 | f1_N128_bs8_L1 | 1 | 128 | 8 | 1 | 4.2231e-03 | 4.7777e-03 | 4.7708e-03 | 521 |
| 59 | f1_N32_bs2_L2 | 1 | 32 | 2 | 2 | 4.2253e-03 | 6.9310e-03 | 6.9159e-03 | 683 |
| 60 | f1_N64_bs4_L1 | 1 | 64 | 4 | 1 | 4.8457e-03 | 6.3836e-03 | 6.3804e-03 | 499 |
| 61 | f3_N768_bs1_L2 | 3 | 768 | 1 | 2 | 4.8508e-03 | 4.6000e-03 | 4.4803e-03 | 519 |
| 62 | f2_N256_bs2_L1 | 2 | 256 | 2 | 1 | 4.8892e-03 | 5.1964e-03 | 5.1793e-03 | 629 |
| 63 | f1_N64_bs1_L1 | 1 | 64 | 1 | 1 | 5.4118e-03 | 7.2490e-03 | 6.6851e-03 | 492 |
| 64 | f4_N1024_bs2_L3 | 4 | 1024 | 2 | 3 | 5.9413e-03 | 5.5957e-03 | 5.5035e-03 | 612 |
| 65 | f2_N256_bs4_L1 | 2 | 256 | 4 | 1 | 6.1031e-03 | 6.4290e-03 | 6.4191e-03 | 676 |
| 66 | f1_N64_bs8_L2 | 1 | 64 | 8 | 2 | 6.4832e-03 | 7.8191e-03 | 7.8190e-03 | 615 |
| 67 | f2_N512_bs16_L4 | 2 | 512 | 16 | 4 | 6.7909e-03 | 6.7945e-03 | 6.7925e-03 | 1676 |
| 68 | f2_N128_bs2_L1 | 2 | 128 | 2 | 1 | 7.1519e-03 | 9.0542e-03 | 8.9884e-03 | 725 |
| 69 | f1_N64_bs8_L1 | 1 | 64 | 8 | 1 | 7.5179e-03 | 8.9395e-03 | 8.9385e-03 | 749 |
| 70 | f2_N512_bs8_L1 | 2 | 512 | 8 | 1 | 7.7727e-03 | 7.7650e-03 | 7.7535e-03 | 978 |
| 71 | f1_N128_bs16_L1 | 1 | 128 | 16 | 1 | 7.7859e-03 | 8.3756e-03 | 8.3749e-03 | 835 |
| 72 | f1_N32_bs4_L2 | 1 | 32 | 4 | 2 | 8.1179e-03 | 1.0998e-02 | 1.0991e-02 | 870 |
| 73 | f3_N768_bs1_L1 | 3 | 768 | 1 | 1 | 8.5673e-03 | 8.3608e-03 | 8.2727e-03 | 681 |
| 74 | f1_N32_bs8_L2 | 1 | 32 | 8 | 2 | 8.9746e-03 | 1.1270e-02 | 1.1268e-02 | 1116 |
| 75 | f1_N64_bs16_L2 | 1 | 64 | 16 | 2 | 9.5702e-03 | 1.0814e-02 | 1.0813e-02 | 930 |
| 76 | f1_N32_bs2_L1 | 1 | 32 | 2 | 1 | 9.9887e-03 | 1.3293e-02 | 1.3279e-02 | 667 |

---

## 4. Rank 10 (sumcos) — batch size summary

| bs | count | worked (test<0.01) | mean(test_err) | best test err |
|----|-------|--------------------|----------------|---------------|
| 1 | 92 | 17 (18%) | 1.26e+00 | 2.66e-06 |
| 2 | 85 | 18 (21%) | 1.11e+00 | 1.74e-05 |
| 4 | 84 | 16 (19%) | 1.01e+00 | 3.44e-05 |
| 8 | 84 | 13 (15%) | 9.93e-01 | 2.51e-04 |
| 16 | 84 | 9 (11%) | 1.02e+00 | 5.18e-04 |

Same trend as rank 5: larger batch (8, 16) → fewer worked runs and worse best test error.

---

## 5. File and directory locations (exhaustive)

### 5.1 Results directories (per rank)

- `results_baseline_sweep_sumcos/` — rank 10 sumcos (one subdir per run: `f{f}_N{n}_bs{bs}_L{L}/`).
- `results_baseline_sweep_sumcos_rank5/` — rank 5 sumcos (same structure).
- `results_baseline_sweep_sumcos_rank20/` — rank 20 sumcos (same structure; create by running `--baseline-sweep-rank 20`).
- `results_baseline_sweep_expcos/` — rank 10 expcos.
- `results_stable_baseline/` — single baseline run (rank 10).

### 5.2 Per-run files (inside each run subdir)

- `config.json` — full training config (factor, n_train, batch_size, hidden_width, hidden_rank, num_layers, lr_sequence, etc.).
- `losses.json` — `all_losses`, `lr_reduction_epochs`, `final_test_error`, `final_train_error`, `min_loss`, `total_parameters`, `config`, etc.
- `model_parameters.pth` — final model state (if saved).
- `params_at_div_1.2_<k>/model_parameters.pth` — checkpoints when loss crossed threshold (optional).
- `plot_target_prediction_loss.png` — target, prediction, loss curve (after running `plot_baseline_sweep_run.py`).

### 5.3 Analysis outputs (in `experiments/table/`)

| File | Description |
|------|-------------|
| `baseline_sweep_sumcos_rank5_results.csv` | All 750 rank-5 runs (name, factor, N, bs, L, final_test_error, final_train_error, min_loss, epochs_run, worked, failed). |
| `baseline_sweep_sumcos_rank5_summary.md` | Rank-5 summary: best configs, worked/failed counts, table by factor, representative lists. |
| `baseline_sweep_sumcos_rank5_loss_histogram.png` | Histogram of min loss per factor (rank 5). |
| `baseline_sweep_sumcos_results.csv` | Rank-10 sumcos runs. |
| `baseline_sweep_sumcos_summary.md` | Rank-10 sumcos summary. |
| `baseline_sweep_sumcos_loss_histogram.png` | Histogram of min loss per factor (rank 10). |
| `baseline_sweep_sumcos_rank20_results.csv` | Rank-20 runs (after `--sumcos-rank20`). |
| `baseline_sweep_sumcos_rank20_summary.md` | Rank-20 summary. |
| `baseline_sweep_sumcos_rank20_loss_histogram.png` | Rank-20 histogram (generate where rank-20 results exist). |

### 5.4 Config and rerun

- `baseline_sweep_rerun.txt` — list of run names always re-run; current content: `f3_N768_bs4_L3`.
- `BASELINE_SWEEP_SETUPS_AND_RANKS.md` — short reference of commands and ranks.

---

## 6. Commands (exhaustive)

From repo root (`MMNN/`):

| Goal | Command |
|------|--------|
| Train rank 5 sweep | `python experiments/table/run_scaling_law_depth_width.py --baseline-sweep-rank5` |
| Train rank 20 sweep | `python experiments/table/run_scaling_law_depth_width.py --baseline-sweep-rank 20` |
| Train rank 10 sumcos | `python experiments/table/run_scaling_law_depth_width.py --baseline-sweep` |
| Plot one run (e.g. rank 5) | `python experiments/table/plot_baseline_sweep_run.py f5_N1280_bs4_L3 --results-dir results_baseline_sweep_sumcos_rank5` |
| Plot all runs with rank 5 | `python experiments/table/plot_baseline_sweep_run.py --rank 5 --results-dir results_baseline_sweep_sumcos_rank5` |
| Regenerate 6 best rank-5 plots | `for run in f5_N1280_bs4_L3 f5_N1280_bs2_L2 f4_N1024_bs4_L3 f4_N1024_bs2_L3 f4_N1024_bs1_L2 f4_N1024_bs2_L2; do python experiments/table/plot_baseline_sweep_run.py "$run" --results-dir results_baseline_sweep_sumcos_rank5; done` |
| Analyze rank 5 (summary + CSV + histogram) | `python experiments/table/analyze_baseline_sweep.py --sumcos-rank5` |
| Analyze rank 20 | `python experiments/table/analyze_baseline_sweep.py --sumcos-rank20` |
| Analyze rank 10 sumcos | `python experiments/table/analyze_baseline_sweep.py --sumcos` |

---

## 7. Preconditioned training (short note)

- **Parameter scale:** ~25.6k (L=2) to ~36.9k (L=3) for the configs above; low-rank (r=5) and width 1024.
- **Preconditioning** (e.g. natural gradient, K-FAC, Shampoo, or diagonal) is a promising direction: can help with ill-conditioning and plateau escape; cost is manageable at this size.

---

## 8. Conclusions (from discussion)

1. **Large batch (bs=8, 16)** trained worse than small batch (1–4) for both rank 5 and rank 10: fewer worked runs, worse best test error.
2. **Factor 5 rank 5:** only one worked config, **f5_N1280_bs4_L3** (bs=4).
3. **Best factor-4 and factor-5 configs** with min_loss < 2e-2 are listed in §3.5; their loss-curve plots live in the corresponding run dirs (§5.2).
4. **Bugs in `run_scaling_law_depth_width.py`** (done check and output dir) were fixed so each run uses its own subdir and is skipped only when that subdir has `losses.json`.

---

*Exhaustive results document for baseline sweep and ranks. Generated from the discussion and existing summary/CSV outputs.*
