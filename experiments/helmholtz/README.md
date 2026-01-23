## Helmholtz operator benchmark (inhomogeneous + boundary forcing)

This folder contains a full benchmark pipeline:

- **data generation**: random domains (multiple shape families), random inhomogeneous coefficient $k(x)^2$, random forcing $f(x)$, random Dirichlet boundary forcing $g(x)$, and a sparse finite-difference solve of

$$
(-\Delta - k(x)^2)u(x) = f(x), \qquad u|_{\partial\Omega} = g.
$$

- **models**:
  - **FNO2d** baseline operator learner (grid-to-grid)
  - **DeepONet (grid)** with **MMNN trunk** as a high-frequency basis, and branch ablations:
    - usual MLP branch
    - MMNN branch with RF / LR / RF-LR settings

- **logging**:
  - `stdout.log`, `metrics.jsonl`, `curves.png`, TensorBoard logs, checkpoints, configs per run

### Run the full ablation suite

From repo root:

```bash
uv run python experiments/helmholtz/helmholtz.py --n_grid 64 --n_train 256 --n_test 64 --epochs 50 --batch 8 --lr 1e-3 --device cuda --seed 0
```

This creates a run folder:

- `experiments/helmholtz/runs/bench_YYYYMMDD_HHMMSS/`

and a summary plot + JSON under:

- `experiments/helmholtz/runs/bench_.../_summary/`

### Run a single model (filter)

```bash
uv run python experiments/helmholtz/helmholtz.py --only deeponet_trunkMMNN_branchMMNN_RFLR --device cpu --epochs 5
```

### Notes

- The DeepONet trunk is an **MMNNTest2D** module mirroring `experiments/former/SinQuad/test2d.py` (including the normalization by the product term).
- The dataset is stored as a single `.npz` file per setting in `experiments/helmholtz/data/`.

