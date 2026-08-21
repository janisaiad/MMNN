# Adversarial review and claim audit

## Recommendation before revision: weak reject

The original thesis—“muP explains Fourier leaps because feature learning
escapes the small NTK eigenvalues”—was not supported. In the exact
right-factor-only model, the rich trajectory often suppresses high-frequency
curvature and learns those modes more slowly than the lazy limit. A paper
claiming universal rich-regime acceleration would be falsified by its own
experiments.

## Repaired central claim

The defensible result is stronger in one direction and narrower in another:

- At every point of the nonlinear trajectory, not only at initialization,
  bounded-variation tangent features imply
  `Lambda_qq(t) <= B_t / q^2`.
- A residual concentrated at frequency `q` is therefore an approximate
  saddle with inverse-gradient index at least `q^2 / B_t`.
- A quadratic residence-time lower bound additionally requires a measured
  Fourier-leakage condition. It is not unconditional.
- DMFT governs the time-dependent prefactor through the back-propagated gate
  kernel. It can amplify or suppress a mode.

## Major objections tested

### 1. “This is an NTK proof in disguise.”

The proof never replaces `K_t` by `K_0`. The rich runs move the latent state
by 3.35 initial norms, flip 35.7% of gates, and change tangent curvatures
frequency-selectively. The frozen-kernel interpretation is empirically
false, while the instantaneous Fourier bound remains valid.

### 2. “Fourier modes need not diagonalize a learned kernel.”

Correct. The paper uses the full Fourier tangent matrix and states a leakage
condition for the residence-time result. Only the instantaneous diagonal
curvature bound is unconditional. The measured post-plateau off-diagonal
Frobenius ratio is about 0.136 in the rich regime, so exact diagonalization is
not claimed.

### 2a. “Finite-width BV automatically proves a uniform μP-limit bound.”

It does not. The finite-network theorem has the explicit budget `B_t`, which
may in principle grow with width. Passing the same bound to DMFT requires
`sup_m B_t^(m) < infinity` on the time interval. The paper now states this as
a wide-limit qualification. Width transfer and measured tangent curvatures
support, but do not prove, that stronger regularity statement.

### 3. “A larger frequency leap means a larger difference `q-p`.”

False in general. The proof depends on absolute `q`. The experiment holds
`q-p=3` fixed while translating both frequencies and still observes rapidly
increasing recovery time. The manuscript explicitly narrows the phrase
“bigger leap” to absolute next frequency, or to raw gap when `p` is fixed.

### 4. “Adam creates the plateaux.”

The main study uses deterministic full-batch gradient descent with no
momentum or adaptive preconditioner. Adam is excluded.

### 5. “Low rank means fixed rank, but the DMFT assumes self-averaging.”

The deterministic DMFT statement uses `r/m -> rho in (0,1)`. Fixed rank is
explicitly excluded from that proposition. The exact finite-width kernel
factorization and Fourier bound still hold at fixed rank.

### 6. “This is strict nonnegative matrix factorization.”

It is not. The factors are signed. Strict NMF requires projected or
tangent-cone dynamics and different boundary order parameters. The title,
abstract, model section, and README now say this explicitly.

### 7. “HCIZ should close the dynamics.”

The Barbier et al. HCIZ calculation is a static Bayes-optimal
teacher–student equilibrium theory. Here the left factor is quenched and the
object needed is a time-dependent gradient kernel. Importing the static free
energy would not prove the training dynamics. The paper cites HCIZ only as a
complementary specialization/metastability theory.

### 8. “The hierarchy is a threshold artifact.”

Recovery curves are supplemented by the threshold-free tangent spectrum and
inverse-curvature saddle index. The fitted exponent is approximately -2.13
over frequencies at a trained, non-kernel state.

### 9. “Rank creates exactly r stages.”

The data do not support this. Rank changes time-scale constants and
finite-channel fluctuations, but no exact `r`-plateau claim is made.

### 10. “The Euler discretization is unstable.”

The step is 5 while the largest measured initial full kernel-operator
eigenvalue is about 0.13 (`dt * lambda_max` about 0.64), leaving a margin
relative to the linear stability threshold 2. Width-transfer curves remain consistent. A
smaller-step convergence ablation would nevertheless strengthen a final
conference submission.

## Remaining limitations

1. The paper specializes established tensor-program/DMFT convergence rather
   than proving a new master theorem for the masked architecture.
2. It does not directly solve the full two-time DMFT response equations and
   overlay that numerical solution with finite-width trajectories; width
   collapse is the current validation.
3. The residence-time theorem is conditional on uniform Fourier leakage and
   variation-budget bounds. The experiments measure related diagnostics but
   do not certify the condition continuously for every run.
4. ReLU is handled at differentiability times. A smooth-activation theorem
   followed by a fully quantified ReLU limit would improve rigor.
5. The study is one-dimensional and synthetic. This is appropriate for the
   theorem but limits claims about natural data.

## Expanded full-training/Muon revision

The first revision did not answer whether the phenomenon survived several
fully trained dense layers, whether training both MMNN factors changed the
mechanism, or whether a matrix optimizer could move the clocks.  The expanded
revision adds those missing controls rather than extrapolating from the
right-factor mask.

### 11. “Abbe et al.’s leap complexity already proves the Fourier result.”

It does not. Their complexity counts newly exposed coordinates (with a
Gaussian multiplicity on the first appearance). For a one-coordinate Hermite
sum, the leap is the minimum included degree; later degrees introduce no new
coordinate. The continuous circle problem is outside their formal basis in
any case. The manuscript now proves this mismatch as a proposition and calls
the Fourier obstruction a within-coordinate regularity cost.

### 12. “Several layers and full training were never checked.”

The new campaign trains every block at affine depths 3, 5, and 7 for both a
dense network and an MMNN in which both signed factors move. The generalized
finite-depth BV theorem covers all of these finite models; the empirical
campaign tests sector clocks, dynamic curvature, feature displacement, and
width transfer.

### 12a. “Saying that DMFT ‘supplies a mode’ is only a metaphor.”

The revised manuscript makes the supply algebra exact. For a dense hidden
weight, the metric-normalized tangent feature is the product of its forward
activation and backward preactivation derivative; the Fourier coefficient is
therefore their convolution. The two MMNN factor derivatives give the same
product after projection through the right or left rank channel. DMFT is
needed for the time evolution of these forward/backward spectra and
responses, not for the finite-width product identity itself.

### 13. “Muon acceleration follows from total descent.”

False. The exact singular-value power map is a strict total descent direction,
but an individual Fourier coefficient may initially move in the wrong
direction. The mode-clock theorem therefore declares a strong biorthogonal
singular-sector closure. Signed velocity diagnostics expose violations rather
than hiding them.

### 14. “This is a standard production-Muon comparison.”

It is an exact memoryless mechanism experiment. Momentum, weight decay,
Newton–Schulz approximation, stochastic batches, and wall-clock matching are
excluded. Learning rates are selected on seeds 101–102 and evaluated on
disjoint paired confirmation seeds. The result can establish spectral
reshaping of sector clocks, not production superiority.

### 15. “Established gradient-flow DMFT proves the reused-data Muon law.”

It does not. The spectral map is nonseparable; its Fréchet derivative is a
Loewner divided-difference operator plus the derivative of RMS normalization.
A same-data DMFT must close this matrix response and retarded Onsager memory.
The manuscript follows the proof-status boundary in Janis Aiad’s three-layer
program: the finite response interface is exact, while the joint reused-data
closure remains conditional.

### 16. “A Gram eigendecomposition is numerically equivalent to a direct SVD.”

Not for the polar update in float32. Squaring the gradient to form `G.T @ G`
squares its condition number and produced substantially different directions
in an ill-conditioned audit. Those outputs are classified as pilots and are
excluded from the paper. The confirmation implementation uses
`torch.linalg.svd` directly (requesting CUDA `gesvd`), declares a relative
numerical-rank threshold of `1e-7`, and includes an ill-conditioned regression test. The final metadata
must identify this backend before the empirical gates can pass.
Across nine representative square gradient blocks, the rejected polar
shortcut had median cosine `0.537` and median relative error `0.963` against
the direct-SVD direction; the fractional errors fell rapidly toward `p=1`.

### 17. “A polished plot can still be based on stale derived data.”

This occurred during the audit: an archived analysis JSON predated the raw
trajectory files and supported an outdated claim that the full-training
curvature slope barely moved. Recomputing from the trajectories changed that
conclusion, most visibly for the MMNN. The paper now reports dynamic slope
reshaping rather than a universal q-to-the-minus-two fitted exponent. The
final audit requires the analysis and every figure to be newer than their raw
sources, requires paper-figure hashes to match generated figures, and requires
the PDF to be newer than every included figure and bibliography source.

### 18. “The calibrated base step is already a converged ODE discretization.”

It is not. In the power-law control, halving the step at matched optimizer
time materially changes endpoint loss for both dense gradient descent and
Muon^1/3. The manuscript therefore treats the confirmation runs as calibrated
discrete update rules and does not use base-to-half agreement as a continuum
claim. Quarter- and eighth-step refinements test the approach to the
continuous-time endpoint, while a doubled-grid control separates time
discretization from population quadrature.
The clock audit is estimand-specific: the core frequency ordering must persist,
and dense Muon^1/3 must retain at least one earlier weak sector. The small MMNN
Muon^1/3 power-law advantage disappears under step halving, so the paper
reports it as architecture- and discretization-sensitive rather than pooling
it into a universal speedup.
All 60 control runs are stable.  The maximum group-median quarter-to-eighth
$L^2$ residual-norm ratio is `1.61`, the doubled-grid maximum is `1.33`, and
every control retains the ordered `q=4,8,16` clock. Dense Muon^1/3 advances a
weak sector in all five controls; the MMNN fails that comparison only at the
half step and is reported accordingly.

### Acceptance gates for the expanded revision

- all pre-specified confirmation runs must finish stably or instability must
  be reported as an outcome;
- no event time may be imputed for a censored run;
- optimizer claims must use paired seeds and report endpoint risk separately
  from sector-clock speed;
- width controls must reuse the width-128 learning rate without retuning;
- full-training curvature must demonstrably change, otherwise the non-kernel
  empirical claim fails;
- the paper must retain the distinction between Muon^p and muP;
- spectral confirmation metadata must report the direct-SVD backend and
  numerical-rank threshold; Gram-eigendecomposition pilots are inadmissible;
- the PDF, code tests, raw-data finiteness audit, and plot-source consistency
  must all pass after the campaign, not only on the smoke data.
- the derived analysis must postdate the raw trajectories; archived or stale
  JSON summaries are inadmissible evidence.
- the eighth-step refinement must agree with the quarter-step $L^2$ residual
  norm within a factor of two in every architecture--optimizer group, and the
  doubled-grid control must satisfy the same groupwise gate; the larger drift
  in squared population loss is reported, not hidden;
- the power-law control must retain the ordered `q=4,8,16` clock, while only
  the dense Muon clock acceleration is claimed robust to every numerical
  control.

## Final recommendation after the expanded revision: preprint-ready, workshop accept

The manuscript now has an exact finite-width identity, a genuinely
non-kernel theorem, direct-SVD full-training experiments, falsification
controls, numerical refinement, and clear negative results. It is suitable as
a rigorous preprint and focused theory-workshop submission. For a strong
main-conference theorem claim, the highest-value next addition remains a
direct numerical solver for the full two-time DMFT and nonseparable Muon
response, together with a finite-width convergence theorem and a
smooth-activation ablation. Those are correctly labeled as future work rather
than hidden prerequisites for the finite-width results proved here.
