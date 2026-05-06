True finite-network NTK cumulant validation outputs.
This is a heavy autodiff experiment on actual low-rank finite networks.
K_NNGP = final hidden feature Gram h_L h_L^T / n.
Theta_NTK = empirical NTK with respect to trainable right factors B and readout a.
Main CSV files:
  true_finite_ntk_cumulant_summary.csv
  true_finite_ntk_cumulant_slopes.csv
Expected scaling, not exact finite-network constants:
  V = O(epsilon L), C = O(epsilon L^2), A = O(epsilon L^3).
Use large reps and n for smoother ratios.
