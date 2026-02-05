# Scaling law for depth and width — extracted working code

This describes the **former working Python code** that produced the frequency/layer scaling summary (`frequency_layer_scaling_summary.txt`).

## Extracted entrypoint (single script)

**File:** `run_scaling_law_depth_width.py`

This script contains the full pipeline for depth/width scaling:

- **Training:** same configs as the original run (freq multipliers 0.3–10, ranks 10/15/25, layer counts per freq, MMNN, Adam + StepLR, loss thresholds, checkpoints).
- **Analysis:** loads `results_frequency_layer_scaling`, builds summaries by frequency, rank (width), and layers (depth), writes CSV + summary.

## Original split (two files)

The same logic was originally in:

1. **`train_frequency_layer_scaling.py`** — training only. Writes to `experiments/table/results_frequency_layer_scaling/`.
2. **`analyze_frequency_layer_scaling_results.py`** — analysis only. Reads that directory, writes `frequency_layer_scaling_analysis.csv` and `frequency_layer_scaling_summary.txt`.

## How to run (from repo root)

```bash
cd /Data/janis.aiad/MMNN

# 1) Run scaling-law training only
python experiments/table/run_scaling_law_depth_width.py --train

# 2) Run analysis only (after some runs are done)
python experiments/table/run_scaling_law_depth_width.py --analyze

# 3) Run both (default)
python experiments/table/run_scaling_law_depth_width.py
```

Training outputs go to:  
`experiments/table/results_frequency_layer_scaling/`  
(config dirs like `freq0.3_rank10_L3`, etc.).

Analysis outputs:

- `experiments/table/frequency_layer_scaling_analysis.csv`
- `experiments/table/frequency_layer_scaling_summary.txt`

## Config summary (from the code)

- **Target:** multi-frequency cosine with phase shifts, scaled by `freq_multiplier` (base freqs 12, 24, 36, 72).
- **Freq multipliers:** 0.3, 0.6, 1.5, 2, 3, 5, 7, 10.
- **Ranks (width):** 10, 15, 25.
- **Layers (depth):** per-frequency sets (e.g. for 0.3: 3, 5; for 0.6: 3, 5, 8; etc.).
- **Epochs:** `2 * freq_mult * 10000`.
- **Other:** hidden_width 777, N_train 1000, batch 100, Adam lr 0.001, StepLR gamma 0.9 step 100, interval [-1,1].

Use this extracted runner to launch training and analysis for scaling-law experiments in depth and width.

---

## CSV and results analysis

### Where to find the CSV

- **Canonical path:** `experiments/table/frequency_layer_scaling_analysis.csv`
- **Comprehensive text report:** `experiments/table/frequency_layer_scaling_full_analysis.txt` (includes tables by frequency, rank, layers, cross-tabs, and conclusions).

A copy of the CSV was also written under `experiments/table/experiments/table/` when the analysis was run from `experiments/table/`; the canonical path above is the one to use.

### How to (re-)run analysis

Analysis requires the **training results** to exist:

```bash
# From repo root: run training first (writes to results_frequency_layer_scaling/)
python experiments/table/run_scaling_law_depth_width.py --train

# Then run analysis (reads results_frequency_layer_scaling/, writes CSV + summary)
python experiments/table/run_scaling_law_depth_width.py --analyze
```

If `results_frequency_layer_scaling/` is missing, `--analyze` exits with "No completed results found". The CSV and full_analysis.txt in the repo correspond to a **previous run** with 58 completed configs.

**Alternative (standalone analyzer with richer tables):**

```bash
# From repo root; uses experiments/table/results_frequency_layer_scaling
python experiments/table/analyze_frequency_layer_scaling_results.py
```

That script writes the comprehensive report to `frequency_layer_scaling_full_analysis.txt` and the CSV (path depends on cwd; run from repo root to avoid nested paths).

### Summary of the 58 configurations

| Metric | Value |
|--------|--------|
| Total completed configs | 58 |
| Freq multipliers | 0.3, 0.6, 1.5, 2, 3, 5, 7 |
| Ranks | 10, 15, 25 |
| Layer counts | 3, 5, 8, 12, 16, 24, 40, 56, 80 |

### Results by frequency multiplier

| Freq × | Configs | Mean test error | Min test error | Note |
|--------|--------|------------------|----------------|------|
| 0.3 | 6 | ~2e-6 | ~1.4e-6 | All good; best regime |
| 0.6 | 9 | ~3e-3 | ~9.5e-6 | Mixed; L5/L8 best |
| 1.5 | 9 | ~1.19 | ~0.63 | All poor; no thresholds |
| 2.0 | 9 | ~1.61 | ~1.00 | All poor |
| 3.0 | 9 | ~1.79 | ~1.58 | All poor |
| 5.0 | 9 | Blown (L56) | ~1.92 | **L=56 explodes** (rank 10/15); L24/L40 plateau |
| 7.0 | 7 | ~1.93 | ~1.93 | **L=80 NaN** (rank 10); others plateau |

### Instability and scaling

- **Depth:** At **freq × 5**, **L=56** gives huge test errors (e.g. 1.78e31, 2.78e32) for rank 10 and 15; rank 25 L=56 stays at ~1.91 (plateau). So very deep nets can **blow up** in this setting.
- **Very deep:** At **freq × 7**, **L=80** for rank 10 yields **NaN** (incomplete/diverged); L=80 for rank 15/25 completes but with high error (~1.93).
- **Rank:** On average rank 25 is best (mean test err ~1.27 excluding blow-ups); rank 10/15 are dragged by L=56 explosions.
- **Layer scaling:** Best layer count by freq in the data: 0.3→5, 0.6→5, 1.5→8, 2→12, 3→24, 5→24, 7→40 (with 5/7 not learning well).

### Best and worst configs (from full_analysis)

- **Best test error:** `freq0.3_rank25_L5` — test_error ≈ 1.39e-6, max_error ≈ 0.011.
- **Worst (blow-up):** `freq5_rank15_L56` — test_error ≈ 2.78e32.
- **Best max error:** `freq0.3_rank25_L3` — max_error ≈ 0.0065.

### Takeaways

1. **Low freq (0.3, 0.6)** with moderate depth (L=3–8) and rank 10–25 works well; error can reach ~1e-5–1e-6.
2. **High freq (1.5 and above)** with current Adam/StepLR and epoch budget does not converge; test error stays ~1 or higher.
3. **Very deep nets (L=56, L=80)** are unstable: L=56 at freq×5 explodes for some ranks; L=80 can produce NaN.
4. To **reproduce or extend** the CSV/summary, run training to populate `results_frequency_layer_scaling/`, then run `--analyze` or `analyze_frequency_layer_scaling_results.py`.
