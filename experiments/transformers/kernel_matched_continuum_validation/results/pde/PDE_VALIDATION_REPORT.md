# Elliptic Bayesian inverse-problem validation

Profile: `full`. Passed: **10/11**.

The RBF score scale is prescribed by the spatial covariance model and is never trained. All posterior solvers use the same assembled sensitivities and right-hand sides. Common PDE/context assembly is reported separately from solver-specific setup.

| Check | Status | Value | Criterion |
|---|---:|---:|---|
| elliptic_uniform_positivity | PASS | 0.406479 | minimum diffusion coefficient > 0.25 |
| latent_covariance_nontrivial_effective_rank | PASS | 4.15494 | posterior data update effective rank >= 3 on every mesh |
| kernel_ritz_condition_reduction | PASS | 885.655 | identity condition / kernel-Ritz condition >= 5 on largest mesh |
| mesh_uniform_kernel_ritz_condition | PASS | 1.09721 | max/min kernel-Ritz condition across meshes < 2 |
| kernel_loop_equal_accuracy | PASS | 1.60885e-07 | all selected kernel loops achieve <= 1.25 times target residual |
| amg_inner_pde_baseline | PASS | 7.77755e-11 | AMG-PCG inner elliptic residual < 2e-9 |
| model_matched_rbf_feature_advantage | PASS | 3.86315 | too-short-kernel / model-matched zero-refinement condition >= 3 |
| one_refinement_reaches_oracle_spectrum | PASS | 1.00011 | matched one-refinement condition / oracle condition <= 1.2 |
| contextual_vs_global_rotated_geometry | PASS | 2.2215e+13 | global / contextual PCG residual at 8 HVP > 5 |
| kernel_vs_woodbury_one_query_total_time | FAIL | 0.213726 | measured Woodbury total / kernel-HB total > 1 on largest mesh |
| dense_system_memory_avoidance | PASS | 32 | dense Hessian memory / matrix-free sensitivity memory > 8 |

A failed speed check is not hidden: it means that the classical Woodbury or dense baseline is faster in that measured regime.  Any paper claim must be restricted to the measured crossover where setup plus solve is actually lower.
