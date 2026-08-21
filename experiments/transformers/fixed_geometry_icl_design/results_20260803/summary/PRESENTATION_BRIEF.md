# Fixed-geometry kernel ICL and experiment design: proof-of-concept brief

## Protocol

- New GP function in every episode on a fixed 1D geometry (64 locations).
- RBF softmax lengthscale fixed at 0.18; it is never trainable.
- The loop controller is trained end-to-end from prediction error only; exact KRR is evaluation-only.
- Experiment design is trained through the frozen looped predictor; weighted variance-greedy is evaluation-only.
- Reported uncertainty is variation across independent training seeds.

## Headline results

- Matched fixed-kernel loop, 12 observations: MSE 0.1205 ± 0.0025 across seeds.
- Exact KRR evaluation reference, 12 observations: MSE 0.1136 ± 0.0025.
- Mismatched fixed-kernel loop, 12 observations: MSE 0.209 ± 0.0015.
- Kernel mismatch multiplies 12-observation error by 1.73×.
- Vanilla Transformer, 12 observations: MSE 0.137 ± 0.0028.
- Learned design, budget 8: weighted MSE 0.06049 ± 0.0047.
- Random design, budget 8: weighted MSE 0.2342 ± 0.0046.
- Learned design changes error versus random by -74.2%.
- Uniform/maximin design, budget 8: weighted MSE 0.1493 ± 0.0025.
- Learned design changes error versus uniform/maximin by -59.5%.

## Claims this PoC can and cannot support

It tests whether a prescribed nonlinear feature geometry plus a trained tied loop can perform ICL across fresh GP draws, and whether a downstream-trained design policy improves measurement allocation. It is not a universal scaling law, a PDE benchmark, or evidence about learned kernel identification.

## Figures

- `icl_scaling.png`: context-length scaling and controls.
- `depth_scaling.png`: computation-time scaling from additional tied iterations.
- `design_scaling.png`: measurement-budget scaling for learned and reference designs.
- `selected_locations.png`: learned versus reference measurement locations.
- `qualitative_reconstruction.png`: one held-out function reconstructed from each design.
