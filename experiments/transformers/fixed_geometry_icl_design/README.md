# Fixed-geometry kernel ICL and learned experiment design

> **Scope correction.** This directory is a one-dimensional GP/kernel-regression
> control, not an implementation of the Linear Sampling Method (LSM) for inverse
> scattering. It contains no Helmholtz scattering matrix and no two-dimensional
> LSM indicator. The corrected problem statement and proposed Bayesian looped
> LSM architecture are documented in the summary PDF. These results must not be
> cited as empirical evidence for LSM.

This directory contains two deliberately small training-only proofs of concept.

1. **Fixed-geometry ICL.** Every episode contains a fresh draw from the same RBF
   Gaussian process. The RBF lengthscale is a fixed modelling choice. A tied
   softmax-kernel loop learns only its recurrent solver controller from query
   prediction error. A deliberately mismatched kernel and a vanilla Transformer
   are trained as controls.
2. **Experiment design.** A sequential policy chooses measurement locations and
   is trained through the frozen looped predictor. Random and nested space-filling
   designs are controls; weighted posterior-variance greedy is evaluation-only.
   The final policy uses a fixed cumulative-coverage gate in the prescribed
   kernel geometry so that training cannot waste late-budget measurements on
   a region that is already covered.

Exact KRR is never used as a target or inside a training loss. It is computed
only during held-out evaluation to show the remaining solver gap.

Quick checks:

```bash
uv run pytest -q experiments/transformers/fixed_geometry_icl_design/test_poc.py
uv run python experiments/transformers/fixed_geometry_icl_design/poc.py --mode smoke --device cpu
```

Full three-seed suite:

```bash
uv run python experiments/transformers/fixed_geometry_icl_design/poc.py \
  --mode suite --seeds 0,1,2 --icl-steps 3000 --design-steps 3000
```

Command used for the reported five-seed run:

```bash
uv run python experiments/transformers/fixed_geometry_icl_design/poc.py \
  --mode suite --device cuda \
  --outdir experiments/transformers/fixed_geometry_icl_design/results_20260803 \
  --seeds 0,1,2,3,4 --icl-steps 5000 --design-steps 5000 \
  --design-name design_separated --design-policy separated \
  --eval-batches 20 --eval-batch-size 256
```

To retrain only the design policy from completed matched-ICL checkpoints:

```bash
uv run python experiments/transformers/fixed_geometry_icl_design/poc.py \
  --mode design --outdir experiments/transformers/fixed_geometry_icl_design/results \
  --design-name design_separated --design-policy separated --seeds 0,1,2
```

Runs resume from `last.pt`, skip completed models, and write raw training logs,
held-out evaluations, multi-seed CSV summaries, figures, and
`results/summary/PRESENTATION_BRIEF.md`. The companion
`architecture_audit.json` lists every trainable tensor and verifies that the
kernel lengthscale never entered the optimizer.
