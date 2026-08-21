# Discretization validation report

Passed: **5/5**.

| Check | Status | Value | Criterion |
|---|---:|---:|---|
| weighted_feature_projector_convergence | PASS | 1.99997 | projector error slope >= 1.5 and R2 >= 0.95 |
| weighted_ritz_metric_convergence | PASS | 1.99997 | metric error slope >= 1.5 and R2 >= 0.95 |
| ritz_transfer_commutator_convergence | PASS | 2.43734 | common-lift commutator slope >= 1.3 and R2 >= 0.9 |
| mesh_uniform_effective_condition | PASS | 1 | max/min effective condition over meshes < 1.2 |
| unweighted_mesh_bias_ablation | PASS | 40367 | finest unweighted / weighted metric error > 10 |
