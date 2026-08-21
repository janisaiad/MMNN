# Complete validation campaign

Passed numerical checks: **45/46**.

This report distinguishes algebraic verification, asymptotic finite-size evidence, and wall-clock claims. A failed check remains visible and narrows the admissible claim.

## Claim audit

| Claim | Status | Evidence | Scope |
|---|---:|---|---|
| Quadrature-aware RBF softmax exactly realizes the prescribed normalized kernel. | VERIFIED | maximum relative identity residual 7.859e-16 | fixed model-chosen RBF length scale and quadrature rule |
| The normalized nonlinear kernel may use the linear-Wishart MP law. | REJECTED | MP passed only on the separate linear control; nonlinear one- and two-resolvents were estimated directly. | no closed nonlinear-kernel deterministic equivalent is claimed |
| The kernel-specific one- and two-resolvent statistics stabilize with size. | VERIFIED | penultimate/coarsest discrepancy ratio 0.1494 | sampled sizes and negative-real resolvent probes only |
| FS/RRS reduced DMFT and resource exponents match the derived laws. | VERIFIED | fitted slopes, standard errors, finite-window tolerances, and R2 are recorded in theory/summary.json | reduced commuting/random-rotation model, not a closed DMFT for arbitrary softmax kernels |
| The Ritz spectral certificate and fixed-polynomial trace risk hold numerically. | VERIFIED | spectral violations 0; trace-risk max relative error 0.007222 | all tested ranks, perturbations, seeds, and fixed HB/Chebyshev depths |
| Quadrature-aware features and Ritz metrics transfer covariantly across meshes. | VERIFIED | metric slope 2; unweighted bias ratio 4.037e+04 | nonuniform periodic-grid lift experiment |
| The contextual kernel-Ritz metric materially reduces elliptic posterior conditioning. | VERIFIED | largest-grid condition reduction 885.7x; contextual condition 1.017 | variable-coefficient 2-D elliptic inverse problem through N=16384 |
| A global stored eigenspace is interchangeable with prompt-conditioned geometry. | REJECTED | global/contextual residual ratio at 8 HVP 2.222e+13 | task-rotated latent covariance family |
| Kernel-HB universally outperforms exact Woodbury. | REJECTED | at m=512, one-query Woodbury/kernel total ratio 0.2137 | Woodbury remains the preferred baseline below the measured long-context crossover |
| Kernel-HB beats Woodbury in a statistically separated long-context, few-query regime. | VERIFIED | nonoverlapping bootstrap intervals at m=2048,4096 | fixed low effective rank, one context, one query; setup included |
| Kernel-HB is faster than dense posterior Cholesky in the largest tested context. | VERIFIED | dense/kernel median total ratio 12.82 | N=16384, m=4096, one query, setup included |

## Suite totals

| Suite | Passed | Total |
|---|---:|---:|
| theory | 26 | 26 |
| discretization | 5 | 5 |
| pde | 10 | 11 |
| crossover | 4 | 4 |

## Reproducible artifacts

- `theory/`: exact identities, RMT controls, nonlinear resolvents, DMFT and scaling fits.
- `discretization/`: projector, Ritz metric, commutator, and unweighted ablation.
- `pde/`: elliptic assembly, spectra, equal-HVP accuracy, setup/solve and inner PDE baselines.
- `crossover/`: raw timing samples and bootstrap intervals for dense/Woodbury crossovers.

The raw CSV and JSON files, not the plots, are the source of every number above.
