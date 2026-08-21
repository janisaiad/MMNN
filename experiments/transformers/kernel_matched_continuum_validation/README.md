# Kernel-matched continuum validation campaign

This directory is an independent, failure-preserving validation suite for
`kernel_matched_continuum_looped_gp.tex`.  It does not overwrite the earlier
Transformer/PDE audits.

The campaign is deliberately hierarchical:

1. **Algebra and quadrature.** Verify the exact RBF-softmax identity including
   quadrature weights, reversibility, positive symmetrization, continuum
   convergence, Ritz positivity, the spectral certificate, and fixed-polynomial
   trace risk.
2. **RMT and reduced DMFT.** Use a linear-Wishart control for the exact
   Marchenko--Pastur limit.  For normalized nonlinear RBF attention, estimate
   its own one- and two-resolvent laws by finite-size convergence; MP is only a
   mismatch diagnostic, never imported as a theory.  Validate FS/RRS flows,
   parameterization, and depth/width/context/time exponents.
3. **Discretization transfer.** Test weighted continuum attention and Ritz
   features on nested and nonuniform meshes; compare the commutator and
   effective condition number with an unweighted ablation.
4. **Elliptic inverse problem.** Construct a variable-coefficient 2-D elliptic
   source-inversion posterior with a nontrivial low-rank-plus-floor latent
   covariance.  Compare kernel-conditioned Ritz HB/Chebyshev/PCG with CG,
   Jacobi, randomized Ritz, oracle Ritz, Woodbury, dense Cholesky where feasible,
   and sparse direct/AMG inner PDE solves.  Report setup, cached solve, memory,
   equal-accuracy iteration counts, multi-query amortization, and measured
   crossovers.

## Acceptance policy

- Float64 algebraic identities: relative residual below `1e-10` (or a stated
  conditioning-adjusted threshold).
- Spectral bounds: no eigenvalue outside a nonvacuous certified interval after
  a `1e-10` numerical cushion.
- Scaling exponents: the measured slope lies within
  `max(2 * regression standard error, declared finite-window tolerance)` of
  theory and `R^2 >= 0.95`; otherwise the check is retained as failed.  The
  tolerance is recorded per law because deterministic truncation bias is not
  sampling noise.
- Mesh transfer: weighted commutator converges with positive rate and the
  finest-grid effective condition numbers remain bounded; the unweighted
  ablation is expected to expose sampling-measure bias.
- Solver comparison: same precision, same right-hand sides, explicit residual
  tolerance, all setup costs, common-context costs separated, and at least five
  timing repetitions after warmup.  A speed claim is emitted only on the
  actually measured side of a crossover.

Quick runs are smoke tests only.  Publication results use the full profile and
multiple seeds.

```bash
.venv/bin/python -m experiments.transformers.kernel_matched_continuum_validation.run_theory_validation \
  --profile full --outdir experiments/transformers/kernel_matched_continuum_validation/results/theory

.venv/bin/python -m experiments.transformers.kernel_matched_continuum_validation.run_discretization_validation \
  --profile full --outdir experiments/transformers/kernel_matched_continuum_validation/results/discretization

.venv/bin/python -m experiments.transformers.kernel_matched_continuum_validation.run_pde_validation \
  --profile full --outdir experiments/transformers/kernel_matched_continuum_validation/results/pde

.venv/bin/python -m experiments.transformers.kernel_matched_continuum_validation.run_crossover_validation \
  --profile full --outdir experiments/transformers/kernel_matched_continuum_validation/results/crossover

.venv/bin/python -m experiments.transformers.kernel_matched_continuum_validation.aggregate_report \
  --root experiments/transformers/kernel_matched_continuum_validation/results
```
