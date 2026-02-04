# Index: log-ratio runs, scaling law (batch size / ranks), and epoch escape time

This index points to where the discussion and experiments on **log-ratio tracking**, **scaling law for batch size and ranks**, and **epoch escape time** live in the repo. They are not in a single “discussion” doc; they are spread across idea notes, plateau experiments, and log-ratio scripts.

---

## 1. Log-ratio runs

### 1.1 Plateau escape runs (log ratios during escape)

- **Script:** `experiments/table/plateau/run_plateau_escape.py`
  - Trains until loss &lt; threshold (plateau escape); records **epochs to escape**, param norm at escape, and **log-ratio distribution** at \(x=0\) over time.
  - Log ratios: \(R_{i,j} = \log|f_i| - \log|f_j|\) (pairwise over low-rank channels), saved every N epochs.
- **Output dir:** `experiments/table/plateau/`
  - Per run (e.g. `lr1e-03_bs4/`): `logratio_epochs.npy`, `logratio_values.npy`, `logratio_trajectories.png`, `logratio_distribution.gif`, `results.json` (includes `epochs_to_escape`).
- **Doc (figure generation):** `refs/ICLR/FIGURE_GENERATION_CODE.md` and `refs/icml_sgdadamlandscapedynamical/FIGURE_GENERATION_CODE.md` — describe log-ratio tracking and plots.

### 1.2 Standalone log-ratio tracking (full training)

- **Script:** `experiments/table/logratio/track_logratio_during_training.py`
  - Tracks \(R_{i,j} = \log|f_i| - \log|f_j|\) at \(x=0\) during full training (e.g. factor=4, rank=15).
- **README:** `experiments/table/logratio/README.md` — usage and options.
- **Output:** `experiments/table/logratio/runs/` (config-named subdirs).

### 1.3 Mean-field and layer-wise log-ratios (results / descriptions)

- **Mean-field cosine:** `experiments/table/meanfield_cosine_results/` — channel shares and log-ratios (mentioned in `DISCUSSION_RESULTS_RESUME.md` §8–9).
- **Mean-field multifreq:** `experiments/table/meanfield_cosine_multifreq_results/` — `EXPLICATION_PRECISE.md`, `PLOTS_DESCRIPTION.md` (how log ratio is computed and plotted).
- **Layer 1 / Layer 2 log-ratio plots:** `experiments/table/experiments/table/results_tune_lr_decay_L2/LAYER1_LAYER2_LOGRATIOS_PLOTS_DESCRIPTION.md` — factor=4, rank=15, log-ratio histograms at different \(x\).

---

## 2. Scaling law for batch size and ranks

### 2.1 Conceptual notes (hypotheses)

- **refs/ICLR/idea.md** (around lines 79–82, 110–129):
  - Scaling law: **epoch to escape \(\propto 1/\text{lr}\)**; **\(\propto \sqrt{\text{batch size}}\)** (lower variance with larger bs).
  - “There is an optimal batch size”; “small batch sizes escape very well for large factor and low rank 5”; “critical batch size”; “batch size also”.
  - “We got 0.75 because there is a bigger dependency somewhere? between 1/N and 1/sqrt(N) bias–variance compromise?”

### 2.2 Empirical sweep (batch size and rank)

- **Baseline sweep by batch size and rank:** `experiments/table/EXHAUSTIVE_RESULTS_BASELINE_SWEEP_RANKS.md` and `experiments/table/DISCUSSION_RESUME_BASELINE_SWEEP_AND_RANKS.md`
  - Sumcos target; batch sizes 1, 2, 4, 8, 16; ranks 5, 10, 20.
  - “Worked” vs “failed” by factor, N, batch size, rank; **small batch (1–4) trains better** than large batch in these MMNN sweeps.
  - **Rank 5:** e.g. §3.3 “By factor and batch size”, §3.4 “By batch size only”; §4 “Rank 10”; rank 20 summary.
- **Histogram resume:** `experiments/table/BASELINE_SWEEP_HISTOGRAM_RESUME.md` — batch size vs performance (e.g. bs=1–4 ~10–13% worked for sumcos rank 5).

### 2.3 Scaling law for depth/width (L, freq)

- **Depth/width vs frequency (not batch/rank):** `experiments/table/SCALING_LAW_DEPTH_WIDTH.md`, `experiments/table/SCALING_LAW_CONCLUSIONS.md`, `experiments/table/VERIFICATION_SUMMARY.md` — loss vs L/freq, etc.

---

## 3. Epoch escape time

### 3.1 Plateau escape experiment (main source)

- **Script:** `experiments/table/plateau/run_plateau_escape.py`
  - Trains until loss &lt; threshold; records **epochs_to_escape**, param norm at escape, and (optionally) log-ratio trajectories.
  - Usage: `python run_plateau_escape.py [--config PATH] [--threshold 1.2e-2] [--max-epochs N]`.
- **Output dir:** `experiments/table/plateau/`
  - **Plots:**
    - `epochs_to_escape_vs_lr.png` — time to escape vs learning rate (log–log), one curve per batch size.
    - `epochs_to_escape_vs_bs.png` — time to escape vs batch size, one curve per lr (script comment: “scaling law for lr=1e-3 and bs&gt;10”).
  - **Per-run:** e.g. `lr1e-03_bs4/results.json` — `escaped`, `epochs_to_escape`, `norm_diff`; configs in `lr*_bs*/config.json` (e.g. `lr1e-04_bs4`, `lr1e-02_bs4`, `lr1e-03_bs1` … `lr1e-03_bs128`).

### 3.2 Idea notes (hypotheses)

- **refs/ICLR/idea.md** (lines 110–120, 124–129):
  - “Plateau escape time (already have a good picture f3N384bs1L2 verify this plateau hypothesis) 2k epoch e-4, 20 e-2.”
  - “1st experiment scaling law in epoch to escape = 1/lr; this is coherent and sqrt(bs) because lower variance still.”
  - “Small batch sizes escape very well for large factor and low rank 5.”

---

## 4. Quick pointer summary

| Topic | Where it lives |
|-------|----------------|
| **Log-ratio runs (during plateau escape)** | `experiments/table/plateau/run_plateau_escape.py` + `plateau/lr*_bs*/` (logratio_*.npy, logratio_trajectories.png, *.gif) |
| **Log-ratio runs (full training)** | `experiments/table/logratio/track_logratio_during_training.py` + `logratio/README.md` |
| **Scaling law (batch size & ranks)** | Ideas: `refs/ICLR/idea.md`. Empirical: `EXHAUSTIVE_RESULTS_BASELINE_SWEEP_RANKS.md`, `DISCUSSION_RESUME_BASELINE_SWEEP_AND_RANKS.md`, `BASELINE_SWEEP_HISTOGRAM_RESUME.md` |
| **Epoch escape time** | Experiment: `experiments/table/plateau/run_plateau_escape.py`; plots `epochs_to_escape_vs_lr.png`, `epochs_to_escape_vs_bs.png`; per-run `results.json`. Hypotheses: `refs/ICLR/idea.md` |

If you want a **single narrative doc** that summarizes the discussion (log-ratio runs, scaling law for batch size/ranks, epoch escape), it can be added as a new section in `DISCUSSION_RESULTS_RESUME.md` or as a separate “Plateau escape and log-ratio scaling” note that links to this index.
