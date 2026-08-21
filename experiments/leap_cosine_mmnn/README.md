# Cosine-frequency saddles in a right-factor MMNN

This experiment studies a three-affine-map ReLU network whose inner matrix is
factorized as

\[
W_2=UV^\top,\qquad \operatorname{rank}(W_2)\le r.
\]

The left factor `U`, the first-layer random features, the biases, and the scalar
readout are frozen. Only the right factor `V` is trained. This is the precise
fixed-left/right-trainable interpretation of the MMNN used here; it does **not**
impose entrywise nonnegativity.

The target is either

\[
y_k(x)=\cos x+\tfrac12\cos(kx)
\]

for a frequency-gap sweep, or a four-mode cosine sum for a direct visualization
of hierarchical recovery.

Run the full experiment from the repository root:

```bash
uv run python experiments/leap_cosine_mmnn/run_experiment.py
```

A short smoke run is available with `--quick`. Results are written to
`experiments/leap_cosine_mmnn/results/` by default.

The key outputs are:

- `hierarchy_trajectory.png`: sequential recovery of the target modes;
- `escape_time_vs_frequency_leap.png`: plateau duration against the next
  frequency gap;
- `tangent_spectrum_and_saddle_index.png`: the right-factor tangent-kernel
  eigenvalues and the inverse-curvature saddle index;
- CSV files containing every plotted value and `summary.json` containing fitted
  exponents and the complete configuration.

The analytic interpretation and its limits are in
[`refs/leap/cosine_mmnn_saddle_theory.md`](../../refs/leap/cosine_mmnn_saddle_theory.md).

