# Exact depth scaling and low-rank Feynman calculus

This directory contains the manuscript and the repository contains its full
computational companion. The central distinction is:

| Object | Status |
|---|---|
| Exponents printed in the plots of arXiv:2508.11522v4 | finite-width Monte Carlo regression slopes |
| Critical ReLU powers for `V,D,F,A,B` | proved exactly: `1,2,2,3,3` |
| Componentwise leading coefficients | proved in closed rational form |
| Gaussian MMNN powers for the analogues of `V,D,F,A,B` | proved in the joint perturbative regime: `1,2,2,3,3`, with amplitudes `1/n+1/r` |
| Stiefel-whitened MMNN powers | same depth powers, with amplitudes `1/n+gamma(n,r)` and exact rank-sector cancellation at `r=n` |
| Gaussian MMNN Gram moments/cumulants | exact at finite rank and every fixed valence |
| Haar--Stiefel orientation moments/cumulants | exact finite-`(n,r)` Weingarten formulas |
| MMNN NTK recursion | exact pathwise at finite width and rank |
| Fixed-rank, arbitrarily deep MMNN | open nonperturbative product-Jacobian problem; not claimed by the paper |
| Benigni--Paquette/MMNN concatenation | exact tangent Gram identity; conditional MP bulk law under the stated quadratic-form and block-decoupling hypotheses |
| Homogeneous concatenated BP spectrum | closed law `MP(gamma_1/L) boxtimes pushforward_L(mu_BP)` |
| NQF symmetry reduction | rigorous local normal form under `C^3 + S_d + ZGZ`; not a global network identity |
| NQF order-parameter dynamics and NTK | exact at finite size for the quadratic model; closes on `(mu,M)` |
| NQF output/NTK initialization cumulants | exact all-valence Gaussian trace words or finite Stiefel Weingarten contractions |
| NQF isotropic Riccati resummation | exact matrix solution under the sample four-tensor isotropy identity |
| NQF orthogonal-feature NTK spectrum | exact finite-sample eigenpairs and logistic mode ignition |
| Gaussian/Stiefel ignition clocks | exact chi-square/beta seed laws; displayed clock moments are small-initialization asymptotics |
| One-block right-factor MMNN as NQF | generally the linear `g` sector, not the pure `M=WW^T` sector |
| Two-block MMNN quadratic form | local bilinear normal form for smooth activation with `phi(0)=0` |
| Deep NQF | exact closure for the defined deep NQF; layerwise local surrogate for a generic deep MMNN |
| Dynamic NQF/BP spectrum | conditional MP law at each time; exact finite-sample formula only in the aligned orthogonal sector |
| Dyson eigenvalue density and overlaps | conditional large-data limit after the covariance sector becomes Gaussian |
| Power-law early-stopping deformation | deterministic equivalent for the stated PSD random-matrix model |

The paper is [exact_mmnn_feynman_theory.tex](exact_mmnn_feynman_theory.tex),
with a compiled [PDF](exact_mmnn_feynman_theory.pdf).

## Quick verification

Run from the repository root:

```bash
uv run pytest -q \
  tests/test_deformed_wigner_dyson.py \
  tests/test_dyson_powerlaw_early_stopping.py \
  tests/test_exact_mmnn_ntk_recursion.py \
  tests/test_exact_mmnn_gram_wick.py \
  tests/test_exact_orthogonal_weingarten.py \
  tests/test_exact_relu_tensor_recursions.py \
  tests/test_powerlaw_weingarten.py \
  tests/test_relu_tensor_asymptotics.py \
  tests/test_mmnn_ntk_variance_scaling.py \
  tests/test_benigni_paquette_mmnn_spectrum.py \
  tests/test_nqf_mmnn_mode_ignition.py
```

The expected result is 37 passing tests.

Recreate the lightweight symbolic and publication outputs with:

```bash
uv run python experiments/feynman/exact_mmnn_gram_wick.py
uv run python experiments/feynman/exact_orthogonal_weingarten.py
uv run python experiments/feynman/benigni_paquette_mmnn_spectrum.py
uv run python experiments/feynman/nqf_mmnn_mode_ignition.py
uv run python experiments/feynman/compare_paper_depth_exponents.py
uv run python experiments/feynman/relu_tensor_asymptotics.py
uv run python experiments/feynman/certify_relu_collision_sources.py
uv run python experiments/feynman/deformed_wigner_dyson.py
uv run python experiments/feynman/run_dyson_powerlaw_early_stopping.py \
  --dimension 4096 \
  --output-dir data/feynman/dyson_powerlaw_early_stopping_n4096
```

The long deterministic recursion archives are already included under
`data/feynman/`. To regenerate the principal archives:

```bash
uv run python experiments/feynman/run_exact_relu_tensor_recursions.py \
  --depth 30 \
  --output-dir data/feynman/exact_relu_tensor_recursions
uv run python experiments/feynman/run_exact_relu_tensor_recursions.py \
  --depth 2000 \
  --output-dir data/feynman/exact_relu_tensor_depth2000
uv run python experiments/feynman/run_mmnn_ntk_variance_scaling.py \
  --samples 300 \
  --output-dir data/feynman/mmnn_ntk_variance_scaling
```

Compile the paper with two LaTeX passes:

```bash
cd refs/feynman
pdflatex -interaction=nonstopmode -halt-on-error exact_mmnn_feynman_theory.tex
pdflatex -interaction=nonstopmode -halt-on-error exact_mmnn_feynman_theory.tex
```

## Principal artifacts

- `data/feynman/paper_depth_inputs.json`: exact four inputs used in the
  published depth-stability experiment.
- `data/feynman/paper_exponent_comparison/`: published slopes versus the same
  finite-window slopes of the deterministic recursion.
- `data/feynman/relu_tensor_asymptotics/`: exact coefficient convergence and
  the termwise `A`-collision certificate.
- `data/feynman/exact_mmnn_gram_wick/` and
  `data/feynman/exact_orthogonal_weingarten/`: exact rational vertex tables.
- `data/feynman/deformed_wigner_dyson/`: density, local spectral measures,
  overlaps, and moments through order eight.
- `data/feynman/benigni_paquette_mmnn_spectrum/`: conditional exact
  homogeneous concatenation spectrum, MP stability factor, and verification
  of the spectral linear-response formula.
- `data/feynman/nqf_mmnn_mode_ignition/`: exact orthogonal-mode NTK
  eigenvalues, finite-rank half-ignition clocks, chi-square/beta log-moment
  comparisons, and the full-rank Stiefel cancellation.
- `data/feynman/dyson_powerlaw_early_stopping_n4096/`: PSD power-law learning
  curves and optimal stopping times.
- `data/feynman/mmnn_ntk_variance_scaling/`: faithful Gaussian-versus-Stiefel
  finite-MMNN sweep and perturbative fit diagnostics.

The sources being connected are arXiv:2508.11522v4, arXiv:2407.00765,
arXiv:2508.20036, arXiv:1909.11304, arXiv:2602.23039v2, and the full text of
arXiv:2608.13335. The manuscript
states explicitly where a result is an exact identity, an asymptotic theorem,
or a conditional random-matrix reduction.

## NQF integration: what is now closed and what remains

Closed in the present companion:

- the notation-safe dictionary between the NQF structure matrix, its dynamic
  Gram order parameter, the quenched MMNN Gram vertex, and the four-index
  Feynman `A/B` tensors;
- the exact all-orders resummation of two-leg NQF vertices on `M`;
- the exact dynamic NTK spectrum in the commuting orthogonal-feature sector;
- the Wishart-versus-Weingarten correction to mode-ignition times;
- the exact/local/conditional status table in the manuscript.

Highest-value remaining calculations:

1. Classify the first quartic MMNN/NQF vertices by the integer partitions of
   four and derive the minimal augmented moment closure beyond `M`.
2. Bound the accumulated layerwise Taylor remainder for a deep MMNN over the
   NQF ignition time, which grows like `log(1/epsilon)`.
3. Prove quadratic-form concentration uniformly along the dynamic NQF tangent
   rows, upgrading the time-dependent Benigni--Paquette law from conditional
   to a trajectory-level theorem.
4. Solve the coupled projector-mode flow when transported multilayer
   structure matrices do not commute; scalar logistics are then unavailable.
5. Treat ReLU at the origin and active-cone/NMF constraints separately; the
   smooth ZGZ theorem and Euclidean tangent calculus do not cover them as
   written.
