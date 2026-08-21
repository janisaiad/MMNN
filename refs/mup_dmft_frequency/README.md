# Dynamic Fourier saddles under masked and full-training muP

This directory contains the paper source. The empirical source of record is
`experiments/mup_dmft_frequency/run_study.py`; it generates the masked-model
CSV files and vector figures.  The full-training depth and optimizer source
of record is `experiments/mup_dmft_frequency/run_full_training_muon.py`.

Build from this directory with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The model is a signed low-rank factorization with a frozen left factor and a
trainable right factor. It is not a theorem about strict nonnegative matrix
factorization. The DMFT claim uses an extensive bottleneck, `r / m -> rho`;
fixed rank remains a random finite-channel limit.

The companion campaign removes that mask.  It trains every block of dense
and signed factorized networks at affine depths 3, 5, and 7, and compares
maximal-update gradient descent with direct memoryless spectral-power descent.
Calibration and confirmation use disjoint seeds:

```bash
uv run python experiments/mup_dmft_frequency/run_full_training_muon.py \
  --mode calibrate
uv run python experiments/mup_dmft_frequency/run_full_training_muon.py \
  --mode campaign
uv run python experiments/mup_dmft_frequency/run_full_training_muon.py \
  --mode discretization
uv run python experiments/mup_dmft_frequency/run_full_training_muon.py \
  --mode discretization-quarter
uv run python experiments/mup_dmft_frequency/run_full_training_muon.py \
  --mode discretization-eighth
uv run python experiments/mup_dmft_frequency/audit_spectral_backend.py
uv run python experiments/mup_dmft_frequency/sync_full_training_figures.py
```

Rebuild the PDF after figure synchronization, then run the fail-closed audit:

```bash
uv run python experiments/mup_dmft_frequency/audit_full_training_results.py \
  --require-discretization
```

Here “Muon” means a direct compact-SVD direction and “Muon^p” means
`U diag(s**p) V.T`, RMS-normalized per matrix block. Singular values below
`1e-7` of the largest are treated as numerical zeros; CUDA runs request the
accurate `gesvd` driver. The study intentionally
omits momentum, weight decay, and Newton–Schulz approximation. It is a
mechanism test, not a production-optimizer benchmark.

The audit verifies figure hashes and PDF freshness and writes a SHA-256
manifest for the raw data, source, figures, and compiled paper into the
full-training audit JSON.

The original leap complexity of Abbe, Boix-Adserà, and Misiakiewicz is a
coordinate-support complexity.  It does not order later Fourier modes on an
already exposed one-dimensional coordinate.  The paper’s Fourier hierarchy
is instead a within-coordinate tangent-regularity result.
