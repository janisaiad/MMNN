# Theory validation report

Profile: `full`. Passed: **26/26**.

The Marchenko--Pastur comparison is restricted to the linear-Wishart control. The normalized RBF kernel is assessed through its own empirical one- and two-resolvent convergence.

| Check | Status | Value | Criterion |
|---|---:|---:|---|
| quadrature_weighted_softmax_identity | PASS | 7.85862e-16 | maximum relative row error < 1e-12 |
| kernel_attention_reversibility | PASS | 4.65294e-16 | detailed-balance relative error < 1e-12 |
| kernel_attention_positive_symmetrization | PASS | -3.14262e-16 | minimum eigenvalue > -1e-11 |
| weighted_continuum_convergence | PASS | 2.00054 | midpoint quadrature slope >= 1.75 and R2 >= 0.98 |
| quadrature_weights_remove_sampling_bias | PASS | 9.62402e-05 | finest weighted error / unweighted error < 0.1 |
| ritz_inverse_operator_bound | PASS | 0 | zero violations of ||B-H^-1|| <= r_S + C_M delta |
| ritz_certified_spectral_enclosure | PASS | 0 | at least one nonvacuous certificate and zero spectral violations |
| fixed_polynomial_trace_risk | PASS | 0.00722243 | maximum Monte Carlo relative error below profile threshold |
| heavy_ball_finite_depth_bound | PASS | 1 | max |p_L| / ([1+L(1+q)]q^L) <= 1 |
| chebyshev_finite_depth_bound | PASS | 1 | max |p_L| / (2q^L) <= 1 |
| linear_wishart_mp_control | PASS | 0.000561459 | finest-size bootstrap upper CI below profile threshold |
| nonlinear_kernel_resolvent_finite_size_convergence | PASS | 0.149369 | penultimate-to-limit discrepancy / coarsest-to-limit discrepancy < 0.8 |
| nonlinear_kernel_spectrum_psd | PASS | -1.65897e-16 | minimum symmetrized normalized-kernel eigenvalue > -1e-10 |
| fixed_spectrum_time_exponent | PASS | -0.315824 | |slope - -0.315789| <= max(2 SE, 0.05) and R2 >= 0.95 |
| rrs_loss_vs_gamma_exponent | PASS | -1.50611 | |slope - -1.5| <= max(2 SE, 0.06) and R2 >= 0.95 |
| rrs_parameterized_time_exponent_r1 | PASS | -0.430174 | |slope - -0.428571| <= max(2 SE, 0.055) and R2 >= 0.95 |
| rrs_parameterized_time_exponent_r5 | PASS | -0.792815 | |slope - -0.789474| <= max(2 SE, 0.055) and R2 >= 0.95 |
| rrs_width_context_exponent | PASS | -1.1977 | |slope - -1.2| <= max(2 SE, 0.035) and R2 >= 0.95 |
| rrs_depth_exponent | PASS | -1.55823 | |slope - -1.5| <= max(2 SE, 0.08) and R2 >= 0.95 |
| finite_task_dmft_isotropy_rate | PASS | -0.496654 | |slope - -0.5| <= max(2 SE, 0.12) and R2 >= 0.95 |
| finite_task_dmft_scalar_closure | PASS | 4.13714e-16 | maximum finest-batch scalar drift error below profile threshold |
| preconditioned_logarithmic_depth_width_shape | PASS | 1.53887 | slope of depth versus log width within 12% of theory |
| unpreconditioned_polynomial_depth_width_shape | PASS | 0.797314 | log-log depth-width slope agrees with nu |
| master_risk_monte_carlo_identity | PASS | 0.589565 | Monte Carlo discrepancy < 4 standard errors |
| bayesian_context_exponent | PASS | -0.545354 | |slope - -0.545455| <= max(2 SE, 0.035) and R2 >= 0.95 |
| gp_width_tail_exponent | PASS | -1.19879 | |slope - -1.2| <= max(2 SE, 0.025) and R2 >= 0.95 |

Failed checks are intentionally retained; they delimit which asymptotic claims are numerically resolved by the selected finite-size profile.
