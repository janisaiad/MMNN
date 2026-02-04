# Baseline sweep histograms — resume and conclusions

## What the histograms are

- **Script:** `experiments/table/analyze_baseline_sweep.py`
- **Content:** For each **factor**, one panel shows the **distribution of min loss (train)** over all configs that finished with a finite min loss. X-axis: min loss (log scale). Y-axis: count. So we see how many runs reached which loss level per factor.
- **Variants:** Baseline (cos(2π·factor·x)); sumcos (∑ cos(2πkx)); sumcos rank 5 / rank 20; expcos (∑ cos(2^k π x), factors 3–4). Each variant has its own results dir and histogram.

## Where they are

All in **`experiments/table/`**:

| File | Variant |
|------|--------|
| `baseline_sweep_loss_histogram.png` | Baseline (rank 10) |
| `baseline_sweep_sumcos_loss_histogram.png` | Sumcos, rank 10 |
| `baseline_sweep_sumcos_rank5_loss_histogram.png` | Sumcos, rank 5 |
| `baseline_sweep_sumcos_rank20_loss_histogram.png` | Sumcos, rank 20 (if generated) |
| `baseline_sweep_expcos_loss_histogram.png` | Expcos, rank 10 |

Regenerate with:
```bash
python experiments/table/analyze_baseline_sweep.py              # baseline
python experiments/table/analyze_baseline_sweep.py --sumcos    # sumcos r=10
python experiments/table/analyze_baseline_sweep.py --sumcos-rank5
python experiments/table/analyze_baseline_sweep.py --sumcos-rank20
python experiments/table/analyze_baseline_sweep.py --expcos
```

## Results (from summaries)

- **Baseline (cos(2π·factor·x)):** 52 worked (test err &lt; 0.01), 632 failed; best configs at factor 1 (e.g. f1_N256_bs1_L2, test err ~2.7e-6). Histogram shows a long tail of high min losses and a cluster of low ones for factor 1.
- **Sumcos rank 5:** Best configs at factor 4–5 (e.g. f4_N1024_bs4_L3, f5_N1280_bs4_L3); ~25k–37k params. Worked rate per batch size: bs=1–4 ~10–13%, bs=8–16 ~6–9%.
- **Rank 10 (sumcos):** Same trend as rank 5: **larger batch (8, 16) → fewer worked runs and worse best test error** than bs=1,2,4.

## Conclusions we made

1. **Batch size:** In these MMNN sweeps (1D cos/sumcos targets), **small batch (1–4) trains better** than large batch (8, 16): more “worked” configs and better best test error. The histograms show the spread of min loss per factor; high-loss bars dominate for harder factors and for large-batch configs.
2. **Rank:** Lower rank (5) can still reach good loss with suitable N, bs, L (e.g. factor 4–5, bs=2–4, L=2–3). Rank 5 and 20 sweeps are analyzed the same way; histogram title includes “— rank N” when using `--sumcos-rank5` or `--sumcos-rank20`.
3. **Use of the histogram:** Quick visual check of (a) how many runs per factor get to low vs high min loss, and (b) that most mass is often at high loss (many failures) with a small number of good runs — consistent with sensitivity to hyperparameters and possible preconditioning/optimization improvements (e.g. natural gradient, K-FAC) for these sizes (~25k–37k params).

## Reference

- Full discussion and commands: **`DISCUSSION_RESUME_BASELINE_SWEEP_AND_RANKS.md`**
- Setups and ranks: **`BASELINE_SWEEP_SETUPS_AND_RANKS.md`**
- Per-variant tables: `baseline_sweep_summary.md`, `baseline_sweep_sumcos_summary.md`, `baseline_sweep_sumcos_rank5_summary.md`, etc.
