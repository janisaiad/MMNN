# PoC result

Default reproducible run: seed `20260911`, 1,024 train tasks, 4,096 IID
test tasks, 4,096 expanded-box shift tasks, eight RK2 steps, 16 right-hand-side
evaluations, and four learned parameters per contextual policy.  CPU runtime
was about 15 s on the current host.  There is one training seed; this is a
mechanism check, not a statistical robustness study.

Here "terminal normalized MSE" means
`mean(((prediction-target)/(1+abs(target)))^2)`; it is a pointwise relative-like
normalization, not the conventional global variance-normalized MSE.

## Main IID result

| Method | Terminal normalized MSE | Terminal MAE | Path normalized RMSE (common times) | NFE |
|---|---:|---:|---:|---:|
| Midpoint | 1.241e-5 | 4.574e-3 | 1.677e-3 | 16 |
| Ralston | 8.809e-6 | 3.670e-3 | 1.399e-3 | 16 |
| Heun | 3.681e-6 | 2.317e-3 | 8.983e-4 | 16 |
| Best learned constant (`alpha=1.4`) | 1.271e-6 | 1.568e-3 | 6.567e-4 | 16 |
| Analytic local-error cancellation | 5.154e-7 | 8.248e-4 | 3.342e-4 | 16 |
| Learned open-loop sequence | 7.235e-7 | 1.201e-3 | 4.736e-4 | 16 |
| Static task-context policy | 7.415e-9 | 1.129e-4 | 2.108e-4 | 16 |
| Dynamic state-context policy | 5.910e-9 | 9.885e-5 | 3.565e-4 | 16 |
| Per-task constant oracle | 7.056e-12 | 3.513e-6 | 2.084e-4 | 16 |
| Classical RK4, compute matched | 2.941e-9 | 4.442e-5 | 2.431e-5 | 16 |

Relative to the best population-wide constant, the static and dynamic
contextual policies improve the trained terminal objective by `171x` and
`215x`.  The dynamic policy also beats a learned time-indexed/open-loop
sequence by `122x`.  This supports the mechanism-level claim: different
members of one ODE family do prefer different feasible coefficients, and a
small amortized policy can predict useful choices in this controlled setting.

The `215x` figure is a squared-error ratio (`14.7x` in normalized RMSE), and
the dynamic policy has smaller absolute error than the best constant on 97.9%
of IID tasks.  It is nevertheless beaten by RK4 on 94.9% of tasks; RK4's
median absolute error is about `31x` smaller.  Thus the mean-square headline
must not obscure the paired comparison.

This is not an overall win over classical integration.  At equal NFE, RK4 has
about half the normalized MSE and 45% of the MAE of the dynamic policy, and its
full-path error is more than an order of magnitude lower.  Policy-evaluation
overhead was not charged, so this comparison is already generous to the
learned method.

## Checks against a misleading positive result

- The per-task oracle coefficient is genuinely dispersed: its IID quantiles
  from minimum through maximum are `0.6775, 0.7175, 0.7475, 0.9125, 1.2781,
  1.5625, 1.86`.  The context is therefore predicting real heterogeneity,
  rather than merely relearning one coefficient.
- This is also an unusually easy amortization problem: one scalar control is
  fitted to one scalar terminal residual, and the near-zero oracle error shows
  that a task-specific coefficient can almost interpolate a root of that
  residual.  The contextual gain must not be extrapolated to vector systems.
- More strongly, the engineered feature `q=y^2/(a+y^2)` gives the analytic
  Riccati choice `alpha_local=2/3+4q/3`, which cancels the third-order local
  error term.  That analytic policy has MAE `8.25e-4` at `N=8`, then converges
  with slope `2.99`; it essentially catches the learned dynamic policy by
  `N=16`.  The learned policy's exceptional `N=8` result is therefore a
  global terminal-cancellation fit on top of highly aligned features.
- Shuffling the static context increases its IID terminal MSE by `720x`.  The
  policy is not succeeding without the task context.
- On autonomous linear equations, random per-step coefficients and midpoint
  agree to `1.67e-16`, as required by the common RK2 stability polynomial.
- The hard order residuals are at most `5.56e-17`, and all output weights are
  nonnegative.
- A hand-coded backward discrete adjoint for independent/open-loop controls
  agrees with PyTorch reverse mode to `3.04e-22` maximum absolute error; a
  centered finite difference agrees to `1.37e-14`.
- A separate closed-loop check includes both `Phi_alpha*pi_y` in the state
  Jacobian and `Phi_alpha*pi_eta` for the four shared policy weights.  It
  agrees with reverse mode to `3.18e-22` and with finite differences to
  `4.74e-14`; the nonzero feedback term is therefore actually exercised.
- The asymptotic empirical convergence slopes are `1.98` for both contextual
  policies, versus `2.00` for the best constant.  The dynamic policy is
  nonmonotone from `N=8` to `N=16`, however: terminal-error cancellation was
  optimized specifically at `N=8` before ordinary second-order convergence
  reappears.

## Constraint and distribution-shift stress tests

The large headline gain uses `alpha in [0.5,2]`.  This preserves the two exact
order conditions and nonnegative weights, but allows `c2=alpha>1`, so the
second stage can lie outside the nominal step.  When the more conventional
constraint `alpha in [0.5,1]` is imposed, the contextual gain over the best
strict constant is only `1.02x`.  On this toy, most of the spectacular gain is
therefore bought by relaxing stage location, not by context alone.

The dynamic policy uses `alpha>1` on 48.1% of IID stages.  For a nonautonomous
problem this can request a final-stage evaluation beyond `T`; it can also leave
the state domain.  Positive weights and exact order two do not by themselves
give SSP, contractivity, internal-stability, or nonlinear-stability guarantees.
This is an order-constrained-only experiment, not a production-constrained
solver.

The enlarged parameter box is a mixed-support distribution shift, not a pure
extrapolation test: 2,459 of 4,096 points (60.0%) remain inside the training
rectangle.  On the 1,637 genuinely outside-support points, the dynamic policy
improves over the best global constant by `25.7x` in normalized MSE, but RK4 is
`3.68x` better.  This is still the same Riccati vector-field family, not
transfer to an unseen field.  The dynamic policy's worst absolute error rises
from `3.77e-3` IID to `3.44e-2` under the expanded-box shift.

The oracle row reports only the selected method's 16 rollout evaluations.  Its
selection itself searches 601 coefficients using the exact target and is not
available online; it is an upper-bound diagnostic, not a deployable baseline.

## Verdict

The mechanism is real: a terminal discrete adjoint can train a tiny contextual
controller on an exact RK constraint manifold, and the controller captures
problem-specific coefficient variation.  The practical proposition is not yet
validated.  The result is fragile to a natural stage constraint, does not beat
compute-matched RK4, and visibly exploits horizon/step-specific cancellation.
It is scalar root interpolation in a deliberately favorable setting.  The
hand-designed context also includes the known Riccati parameter `a` and a
feature aligned with the nonlinearity; this is an identifiability test, not a
black-box feature set that would transfer automatically to an arbitrary Neural
ODE.

A larger experiment is justified only for regimes where each right-hand-side
evaluation is expensive and higher-stage methods are unattractive (for
example, a frozen Neural ODE at very low NFE).  The next decisive test should
train across several step counts under `c_i in [0,1]` plus a stability
constraint, then evaluate on unseen vector fields and compare wall time as well
as NFE.
