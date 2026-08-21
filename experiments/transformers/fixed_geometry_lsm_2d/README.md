# Fixed-geometry ICL for the 2D Linear Sampling Method

This directory contains the actual two-dimensional LSM proofs of concept. It
does not use the earlier one-dimensional GP-regression control as evidence for
inverse scattering.

## Physical scope

- 2D active deterministic multistatic acquisition.
- 32 plane-wave incident directions and 32 receiver directions on a circle.
- One random obstacle per task; the sources themselves are not random.
- Complex Born/Foldy far-field matrix `F_D` and a 32 x 32 probing grid.
- Training shapes: disks and ellipses.
- Held-out shapes: kites and pairs of disks.

The random-source matrices `C_D` and `C_tilde_D` from the 2022/2024 papers are
future experiments, not claims of this PoC.

## Original near-field extension

`near_field_lsm.py` is a separate physical model, not a relabeling of the
far-field proof of concept. Deterministic point sources illuminate one to six
sound-soft obstacles, a distinct receiver curve records the complex multistatic
near field, and an MFS boundary solve enforces the Dirichlet condition.

Following the posterior-predictive construction of Kang, Lee and Cheng
(arXiv:2605.26713), one tied recurrent cell solves two right-hand-side blocks in
parallel:

```text
H_D Q_mean = Phi,             H_D Q_cov = N_D K_Gamma,
M_D = K_Gamma N_D* Q_mean,    Sigma_D = K_Gamma - K_Gamma N_D* Q_cov.
```

The RHS-weighted Krylov matrices used by the endpoint controller are spectral
statistics only; they are not confused with these posterior moments. The same
encoder and Bayesian decoder support Richardson, heavy-ball, Chebyshev, and
PCG. The softmax/von-Mises kernel remains a fixed modelling choice.

## Model

The angular von Mises/softmax kernel is fixed as a modelling choice. The model
does not learn its temperature, an image, an LSM indicator, a noise estimator,
or a spatial regularization map.

The recurrent cell is complex PCG. Its exact Krylov coefficients are computed
in context. The only learned solver component is an SPD preconditioner

```text
P_D = I + a_D L_Gamma + b_D L_Gamma^2,  a_D,b_D >= 0,
```

where `L_Gamma` is the graph Laplacian of the fixed angular softmax features.
The experiment-design stage freezes this loop and trains six continuous shared
source/receiver angles with a hard 28-degree minimum separation.

## Reproduce

From `/root/MMNN`:

```bash
PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_experiments \
  --output-dir experiments/transformers/fixed_geometry_lsm_2d/results_final_20260804

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.analyze_final \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_final_20260804

cd experiments/transformers/fixed_geometry_lsm_2d/results_final_20260804
pdflatex -interaction=nonstopmode -halt-on-error results_note.tex
pdflatex -interaction=nonstopmode -halt-on-error results_note.tex
```

Fast structural checks:

```bash
/root/MMNN/.venv/bin/pytest -q \
  experiments/transformers/fixed_geometry_lsm_2d/test_lsm_core.py
/root/MMNN/.venv/bin/ruff check \
  experiments/transformers/fixed_geometry_lsm_2d \
  --exclude results_20260804 \
  --exclude results_faithful_20260804 \
  --exclude results_final_20260804
```

Run the original near-field posterior-moment audit and build its English PDF:

```bash
PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_near_field_moments \
  --output-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_final_20260804 \
  --steps 1200 --eval-tasks 48 --seeds 17,29,43 --depth 64

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.analyze_near_field_moments \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_final_20260804
```

The equal-matvec near-field control compares identity-CG with population-PCG.
At depth 64 the learned geometry/population factor reduces the macro mean
residual from `7.03e-3` to `2.84e-5` (248x) and the covariance residual from
`2.12e-4` to `4.26e-6` (50x). Localization AP is already saturated for both,
so this is evidence for faster conditioning rather than a sharper physical
point-spread function.

## Multi-axis scaling laws

`run_near_field_scaling.py` performs a resumable, matched-budget sweep over
training sample size, controller/preconditioner width, and the number of
source/receiver tokens. The same learned model accepts several context lengths
through a deterministic harmonic covariance sketch; the physical near-field
matrix is neither padded nor replaced by a learned surrogate. Population-PCG
is trained directly rather than being derived from an HB checkpoint.

The full protocol uses dataset sizes 128--32768, widths 32--512, contexts
8--48, three seeds, and six learned solvers. In addition to population-PCG,
`context-PCG` reads task-specific Hessian statistics and predicts positive
modal gains once per prompt; its SPD factor is fixed throughout the solve, so
the outer loop remains standard PCG rather than flexible CG. Its identifying
loss penalizes the normalized Frobenius error of the transformed operator from
the identity. `hybrid-PCG` starts from the training-free Hessian diagonal in the
prescribed GP angular basis and learns only a positive multiplicative residual
correction. Identity-CG and exact dense solves are training-free controls.
At evaluation, CG, PCG and all looped solvers use 32 operator applications for
both posterior right-hand-side blocks.

```bash
PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_near_field_scaling \
  --output-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --resume

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.analyze_near_field_scaling \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805
```

Post-training depth and spectral-conditioning audits use the same final
checkpoints and held-out tasks:

```bash
PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_near_field_depth_scaling \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --resume

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.audit_near_field_conditioning \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --resume

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_near_field_geometry_generalization \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --resume

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_near_field_cg_stress \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --geometry-draws 64 --tasks-per-geometry 4 --resume

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_near_field_joint_conditioning \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --geometry-draws 64 --tasks-per-geometry 4 --resume

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.plot_near_field_joint_reconstructions \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --seed 17 --geometry-draw 0 --task 0

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.run_near_field_classical_preconditioners \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805 \
  --resume

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.refresh_near_field_pcg_runtime \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805

PYTHONPATH=/root/MMNN /root/MMNN/.venv/bin/python \
  -m experiments.transformers.fixed_geometry_lsm_2d.audit_near_field_scaling_results \
  --results-dir experiments/transformers/fixed_geometry_lsm_2d/results_near_field_scaling_20260805
```

The analyzer writes task-level aggregates, fitted empirical scaling exponents,
wall-clock learning curves, inference-time scaling, and a simultaneous 95%
held-out Hoeffding bound for localization risk. The bound uses the number of
independent held-out tasks, not the training-set size; the latter indexes the
learned hypothesis. This distinction avoids presenting a post-training
confidence interval as a training-sample generalization theorem.

The completed scaling directory contains 90 trained model runs, 810 dataset
checkpoints, and 362,880 learned-model/task evaluations. At width 128,
dataset size 32768, context 24, and 32 matched operator applications,
the original training-grid metric gives an 8.03x transformed-coordinate gain.
The coordinate-invariant post-training audit is the reported CG comparison:
at extrapolative context 48, context-PCG reduces the physical residual from
1.067 to 0.521 (2.05x), the relative LSM-score error from 0.0292 to 0.00454
(6.44x), and raises numerical 95% posterior-band coverage from 0.243 to 0.873.
The condition-number audit records a 23.22x median reduction. Both transformed
and physical residuals are retained explicitly in the released CSV files.
Here `r_tr = ||C*(b-Hq)||/||C*b||` is the optimization coordinate used by the
original training grid, whereas `r_phys = ||b-Hq||/||b||` is the common
physical coordinate used for direct solver comparisons. Since `C` depends on
the method, `r_tr` is never used to claim a physical CG speedup; the two-page
CG note and every post-training robustness claim use `r_phys`.

After removing the endpoint controller that PCG does not use, context-PCG costs
1.14x the CG batch time at context 24 and 1.27x at context 48; dense
factorization is slower on these batches. At context 48, context-PCG reaches
mean residual 0.01 in 96 iterations and 23.13 ms, versus 128 iterations and
24.91 ms for the shared-architecture CG. A stripped optimized CG reaches the
same target in 22.86 ms, so the learned method is not the fastest strict
classical implementation. Width 64 gives the smallest context-48 residual;
width 512 uses 22.45x more parameters and is 3.5% worse, identifying a
fixed-feature-basis bottleneck rather than a foundation-model-size bottleneck.

The training-free controls are essential to the interpretation. Physical
Jacobi gives essentially no condition-number reduction and angular block
Jacobi gives about 2.05x, but Jacobi in the fixed GP angular basis gives 24.28x,
slightly above context-PCG's 23.22x. At depth 32 its physical residual is 0.629
versus 0.521 for context-PCG at comparable runtime; at depth 96 the analytical
control is more accurate. Learning is therefore only modestly useful for the
finite-horizon objective in the current feature basis, not intrinsically
necessary to expose the available diagonal angular conditioning signal.
At residual 0.01, angular-Jacobi-PCG is the fastest audited method at 19.24 ms.

The conservative analytic--learned hybrid is initialized exactly at
angular-Jacobi, constrains every learned log-gain correction to `[-0.5, 0.5]`,
and regularizes the correction rather than the analytic base. At context 48 and
depth 32 its physical residual is 0.630 versus 0.629 for angular-Jacobi and
0.521 for context-PCG. The learned correction therefore
does not add a robust signal in the prescribed fixed angular basis. Looped HB
reduces the physical CG residual only from 1.067 to 0.983 at the same budget,
but its average
precision is only 0.478, numerical-UQ coverage 0.095, and batch time 83.62 ms;
it is not a competitive reconstruction/UQ method here.

On the independent-acquisition audit, context-PCG has physical residual 0.501,
versus 0.559 for angular-Jacobi and 0.566 for the hybrid. Pairing all 768
context-48 batches gives a CG-to-context gain of 2.241 (95% interval
2.169--2.316), with context-PCG winning 96.4% of batches. The paired
angular-to-context gain is 1.103; context-PCG wins 70.1% of those pairs. The
advantage over angular-Jacobi is not scenario-uniform:
angular-Jacobi remains slightly better for six-obstacle and frequency-OOD
scenarios, while context-PCG gains most on half-aperture geometries.

The stress audit adds 16 marginal levels over obstacle count, relative noise,
aperture and wavenumber plus four joint-shift levels, with 192 independent
acquisition batches per level. Context-PCG wins at least 62.5% of paired
batches at every level and every 95% physical-residual gain interval stays
above one; its gain ranges from 1.07x to 4.32x. At the most extreme joint
shift, context-PCG retains 1.96x and angular-Jacobi 1.34x, whereas
population-PCG and looped HB fall below CG at 0.42x and 0.12x. Context-PCG's
numerical-coverage gain nevertheless collapses to zero there, so this is
residual robustness rather than universal UQ calibration.
The joint spectral audit shows why a scalar condition number is insufficient:
at the extreme shift, context-PCG and angular-Jacobi leave nearly identical
median condition numbers (590 and 588), yet their physical-residual gains are
1.96x and 1.34x. Finite-horizon clustering and right-hand-side alignment still
matter.

The time-to-tolerance frontier makes the distinction explicit. Across 18
context/target cells, angular-Jacobi is fastest in 11, stripped optimized CG
in 5, block-Jacobi in 2, and context-PCG in 0.
Context-PCG is iteration-efficient and robust to unknown task conditioning,
but it does not beat a correctly specified analytic GP-basis preconditioner in
strict wall-clock time on this small dense problem.

Use `results_near_field_scaling_20260805/cg_comparison_note.pdf` for the
two-page CG comparison and `results_note_english.pdf` for the full English
manuscript (`near_field_scaling_note.pdf` is the byte-identical canonical PDF).

## Final evidence

The final directory contains task-level results over three seeds and 576
held-out tasks per condition.

- At 15% noise, trained PCG at depth 20 matches direct Tikhonov average
  precision to four decimals on ellipses, unseen kites, and unseen two-obstacle
  tasks.
- The mean relative residual on kites is 0.0102 for trained PCG at depth 20,
  versus 0.0141 for identity-preconditioned CG.
- A fine-grid forward (64 x 64) with coarse probing (32 x 32) retains average
  precision 0.9896 / 0.9858 / 0.8926.
- Six learned angles attain average precision 0.9582 / 0.9628 / 0.8790,
  compared with 0.2604 / 0.2626 / 0.2439 for six uniform angles.
- Learned angles reduce the maximum point-spread sidelobe from 0.999 to 0.562.

Use `results_final_20260804/results_note.pdf` for presentation. The folders
`results_20260804` and `results_faithful_20260804` are development audits and
are not the final reported experiment.
