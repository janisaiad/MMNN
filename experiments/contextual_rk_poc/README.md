# Contextual RK under hard order constraints: literature check and PoC

## Bottom line

The broad idea is valid, but it is not new: learning Runge--Kutta coefficients
for a problem or a distribution of problems has existed for more than two
decades.  A narrower question remains interesting:

> Can a cheap context-conditioned policy select a tableau on an exact RK order
> manifold, using a fully rolled-out terminal objective and its discrete
> adjoint, while retaining useful distribution-shift and step-size behavior?

The experiment here gives a **positive mechanism check**, not a publication
claim.  Context is genuinely informative on a nonlinear toy family, but the
gain nearly vanishes under the stricter `c2 <= 1` constraint and a compute-
matched RK4 remains better overall.  That is evidence to continue cautiously,
not evidence for a universal new solver.

## Targeted primary-literature review

This was a targeted search, not an exhaustive patent/citation-graph audit.
No claim of "no prior art" can be made from it.

| Work | What was already done | Remaining difference |
|---|---|---|
| [Tsitouras (2002)](https://doi.org/10.1109/72.977309) | RK stages represented as NN layers and coefficients trained on test IVPs for broader performance. | Static coefficients; no contextual terminal-control formulation. |
| [Anastassi (2011/2014)](https://arxiv.org/abs/1106.1194) | Learned a constrained two-stage RK method specialized to the two-body problem. | Static coefficients and essentially local supervised loss. |
| [Gusak et al. (2021)](https://arxiv.org/abs/2103.08561) | Exact parameterizations of RK2/RK4 families for Neural ODE solver selection and smoothing. | No terminal numerical-error policy. |
| [Ouala et al. (2021)](https://arxiv.org/abs/2105.04999) | Differentiable `A,b,c`, stability/problem adaptation, and joint learning with an ODE model. | A global tableau and mainly one-step/teacher-forced trajectory losses. |
| [Guo et al. (2022)](https://arxiv.org/abs/2105.01303) | Population-risk learning of RK-like integrators for targeted ODE families. | Shared learned algorithm, rather than a context-to-tableau policy optimized through a terminal rollout. |
| [Cao & Petzold (2004)](https://doi.org/10.1137/S1064827503420969) | Adjoint-based global/final ODE error estimation and control. | The adjoint insight itself is prior art; it was not used to learn contextual RK coefficients. |
| [Sanz-Serna (2016)](https://doi.org/10.1137/151002769) | Connects RK discrete adjoints, reverse accumulation/automatic differentiation, and optimal control. | Establishes the gradient machinery as classical, not a novelty claim. |
| [Dellnitz et al. (2023)](https://doi.org/10.1137/21M1412682) | A learned/RL contextual controller tailored to a class of ODEs. | The action is the time step, not an order-constrained tableau. |
| [Kim, Tang & Yu (2023)](https://arxiv.org/abs/2309.16421) | A small adjustment to an existing diffusion ODE solver is distilled from a more accurate, higher-step teacher over batches of samples. | Model-wide diffusion specialization, not a context-to-hard-order-tableau policy. |
| [Nüßlein, Ranocha & Ketcheson (2021)](https://doi.org/10.2140/camcos.2021.16.155) | RK weights adapted at each step by a constrained optimization that preserves order and positivity. | Direct optimization for a structural property, rather than a learned task/state policy for terminal error. |
| [Ranocha et al. (2020)](https://arxiv.org/abs/1905.09129) | A per-step relaxation parameter chosen to enforce conservation or entropy properties. | Shows that state-dependent RK-like coefficients are established; the objective is structure preservation, not learned terminal error. |
| [BNS (Shaul et al., ICML 2024)](https://proceedings.mlr.press/v235/shaul24a.html) | Fewer than 200 nonstationary solver parameters optimized on endpoint error for diffusion/flow models; the solver family subsumes RK and multistep methods. | Very strong terminal/global-loss overlap, but model-wide rather than instance-conditioned and without hard classical RK order constraints. |
| [ARKM (Zhao et al., 2024)](https://arxiv.org/abs/2405.14504) | A state/input-conditioned gate mixes RK-like increments in a spatiotemporal prediction architecture. | It is not an integrator for a known ODE, and the learned gate does not in general enforce classical RK2 order conditions. |
| [S4S (Frankel et al., ICML 2025)](https://proceedings.mlr.press/v267/frankel25a.html) | Time-indexed solver coefficients optimized directly against a teacher's terminal output over random diffusion samples. | Closest global-loss overlap; diffusion-specific, coefficients are not a state/task-conditioned hard-order RK policy. |
| [DSS (Wang et al., ICML 2025)](https://proceedings.mlr.press/v267/wang25bh.html) | Differentiable search over time steps and solver coefficients for few-step diffusion/flow sampling. | A globally searched, Adams-like diffusion solver rather than an instance-conditioned classical RK tableau. |
| [Anastassi (2025)](https://doi.org/10.3390/math13172796) | Evolutionary optimization of four degrees of freedom in a parameterized order-six RK family against global rollout error on oscillatory problems. | Hard-order and global-error overlap, but one static method per problem class, without a contextual policy or adjoint training. |
| [Ouala et al. (TPAMI 2025/2026)](https://doi.org/10.1109/TPAMI.2025.3599629) | Trainable numerical schemes jointly adapted to Neural ODE models with fixed NFE. | Again a trained scheme, not the precise contextual/goal-oriented constrained controller studied here. |
| [Izgin, Ketcheson & Meister (2025)](https://arxiv.org/abs/2305.14297) | General order theory for RK-like methods with solution-dependent coefficients. | A warning: classical Butcher conditions cannot be assumed sufficient for arbitrary higher-order state-dependent policies. |
| [INDIS (Yuan et al., 2026)](https://arxiv.org/abs/2603.17671) | An instance-conditioned network, trained on terminal teacher error, generates sampler times, evaluation-time shifts, and denoiser scale factors. | Very close contextual-control pattern, but it does not generate a classical RK tableau with a hard order guarantee. |

The targeted search did not identify a work combining all four of: an
instance/state-conditioned policy, generated RK-tableau degrees of freedom, an
exact order guarantee in the stated setting, and end-to-end terminal rollout
training through the discrete adjoint.  Every ingredient and several partial
combinations have close precedents, so this is an apparently underexplored
intersection, not proof of novelty.  A defensible working title would be
**goal-oriented contextual RK policies with hard order constraints**, not
"learned RK" or "adjoint RK optimization" by themselves.

## Optimal-control formulation

Let `xi` identify an ODE, initial value, horizon, and any random context.  With
the numerical state as the controlled state and tableau parameters as controls,

```text
Y[n+1] = Phi_h(Y[n], alpha[n]; xi)
alpha[n] = pi_eta(context[n]) in M_p
min_eta E_xi [ 1/2 ||D_xi^-1 (Y[N] - y_xi(T))||^2 + regularization ] .
```

`M_p` is the feasible set defined by order equalities and desired stability,
positivity, or stage-location inequalities.  This is stochastic optimization
when one shared/static tableau is trained under an expectation.  It becomes a
sequential stochastic-control problem when a policy observes context and
chooses a new action at every step.

For the closed-loop step `G[n,eta](Y)`, the exact discrete adjoint is

```text
lambda[N] = D^-T D^-1 (Y[N] - y(T))
lambda[n] = (dG[n,eta]/dY)^T lambda[n+1]
grad_eta J = E_xi sum_n (dG[n,eta]/deta)^T lambda[n+1].
```

Reverse-mode differentiation through the finite rollout computes this discrete
adjoint.  A continuous Neural ODE adjoint is not, in general, the exact gradient
of the finite-step discretization error one is trying to optimize.

## Minimal experiment

We use the one-parameter, explicit two-stage order-two family

```text
k1 = f(Yn)
k2 = f(Yn + alpha h k1)
Y[n+1] = Yn + h [(1 - 1/(2 alpha)) k1 + k2/(2 alpha)].
```

Thus `a21=c2=alpha`, `b2=1/(2 alpha)`, and `b1=1-b2`.  The two order
conditions `b1+b2=1` and `b2*c2=1/2` hold to machine precision for every
policy output.  For this specific family, a direct Taylor expansion proves
second order uniformly when `alpha` stays in a compact interval.  We do not
extrapolate this claim to arbitrary higher-order solution-dependent RK
families.

The task distribution is the scalar Riccati family

```text
y' = a + y^2,  a in [0.1, 1],  y0 in [-0.5, 0.5],  T=0.8,
```

whose exact solution is available.  Eight RK2 steps cost 16 right-hand-side
evaluations.  The learned policy is only an affine sigmoid with four scalars;
its features encode the initial/current state, the nonlinear fraction
`y^2/(a+y^2)`, and optionally time.  No large network is warranted for this
four-dimensional policy class.  Because this feature exposes the known task
parameter `a`, the setup is a favorable identifiability/mechanism test rather
than a black-box context extractor for arbitrary Neural ODEs.

The test includes:

- midpoint, Ralston, Heun, and the best population-wide constant coefficient;
- the problem-specific analytic coefficient that cancels Riccati's leading
  local-error term;
- one coefficient per time step shared across tasks (open-loop/S4S-like);
- one context-conditioned coefficient held over a trajectory;
- a context-conditioned coefficient chosen at every step;
- an oracle constant coefficient per task and shuffled-context ablation;
- RK4 at exactly the same 16 right-hand-side evaluations;
- a stricter `alpha in [0.5,1]` ablation, which keeps the second stage inside
  the current step;
- an autonomous linear negative control, where all RK2 members must agree;
- hand-derived open- and closed-loop discrete-adjoint gradient checks and a
  convergence study;
- an expanded-box shift, with the genuinely outside-training-support subset
  reported separately.

The main interval `alpha in [0.5,2]` guarantees exact order two and nonnegative
weights, but `alpha>1` places the second stage beyond the nominal step.  This is
harmless for this autonomous toy and is explicitly reported as an **order-only
relaxation**, not as a generally safe production constraint.

## Reproduce

From the repository root:

```bash
.venv/bin/python experiments/contextual_rk_poc/poc.py
```

Outputs are written to `results/metrics.csv`, `results/summary.json`, and
`results/poc_summary.png`.  The default run uses CPU only and takes tens of
seconds.  See `results/REPORT.md` for the measured result and interpretation.

## Decision rule for a larger study

Continue only if the policy beats the best learned static tableau under hard
production constraints, at equal NFE and wall time, on held-out ODE families
and multiple step sizes.  For Neural ODEs, first freeze the vector field and
learn only the solver; then evaluate that same field with an independent,
accurate solver.  Otherwise the model and solver can collude, and improved task
loss need not mean a better continuous-time model or integrator.
