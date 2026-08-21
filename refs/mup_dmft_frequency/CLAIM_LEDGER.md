# Claim ledger

This ledger assigns every central statement in the manuscript to a proof
status. It is intentionally stricter than the narrative summary.

## Proved in the manuscript

- **P1 — finite masked kernel factorization.** For the frozen-left,
  right-factor-only architecture, the instantaneous kernel is exactly
  `K_t = Phi_m * G_{r,t}` at every finite width and ReLU differentiability
  time.
- **P2 — architecture-general Fourier curvature.** For any finite parameter
  vector with a positive diagonal gradient metric, bounded-variation tangent
  features imply `Lambda_qq(t) <= B_t / q^2` without freezing the kernel.
- **P3 — finite-depth applicability.** Generic finite-width dense ReLU
  networks and signed factorized MMNNs at every fixed finite depth have BV
  tangent features at differentiability times. This does not assert a
  width-uniform bound on the total variation budget.
- **P4 — conditional residence time.** Quadratic-in-frequency residence time
  follows under the stated uniform BV-budget and Fourier-leakage conditions.
  The conditions, not just the conclusion, are part of the theorem.
- **P5 — one-coordinate leap collapse.** In the Abbe–Boix-Adserà–Misiakiewicz
  definition, a one-coordinate Hermite sum has leap equal to its smallest
  included degree. Later degrees introduce no new coordinate.
- **P6 — spectral-power descent.** Every nonzero exact compact-SVD
  `U diag(s**p) V.T` update, with the declared RMS normalization, is a strict
  first-order descent direction for `0 <= p <= 1`. The implementation uses a
  direct SVD and an explicitly declared numerical-rank threshold.
- **P7 — conditional Muon sector clock.** Under biorthogonal singular sectors
  and a uniformly controlled global normalization, a power-law Fourier target
  has the half-error exponent derived in the paper.
- **P8 — exact forward/backward Fourier supply.** For every dense hidden
  matrix, the metric-normalized tangent feature is exactly a forward
  activation times a backward field, so its Fourier coefficient is their
  convolution. For a signed factorized map, the corresponding products pass
  through the left and right rank channels. These are finite-width
  chain-rule identities; the DMFT claim concerns how their joint law evolves.

## Imported results

- **I1 — rich infinite-width feature-learning limit.** Tensor-program and
  gradient-flow DMFT convergence are imported from Yang–Hu and
  Bordelon–Pehlevan for compatible smooth computation graphs and finite time.
- **I2 — finite-width order-parameter fluctuations.** The inverse-square-root
  fluctuation expectation is imported away from critical amplification.
- **I3 — Muon^p Schatten geometry.** The constrained steepest-descent
  interpretation is imported from Dong–Sawin.

## Conditional physics closures

- **C1 — masked extensive-rank DMFT.** The masked process is a specialization
  of the imported wide-limit machinery; the paper does not reprove the master
  theorem or directly solve the complete two-time equations.
- **C2 — full-training deep DMFT.** Every layer requires its own activation,
  backward, correlation, and response kernels. The experiment tests width
  consistency but does not numerically solve that closure or prove a
  width-uniform BV budget.
- **C3 — reused-data Muon DMFT.** The nonseparable spectral channel has the
  exact Fréchet derivative written in the paper, but its matrix-valued Onsager
  memory and same-data self-averaging are open. Ordinary Muon is not called
  AMP.
- **C4 — no HCIZ dynamics claim.** Static HCIZ/replica equilibrium theory is
  treated only as a complementary metastability analogy.

## Empirical statements

- **E1 — masked feature learning.** The original 93-run campaign establishes
  order-one latent/gate motion, a dynamic `q^-2.13` tangent spectrum, fixed-gap
  controls, and high-frequency suppression in the masked rich regime.
- **E2 — full-training depth hierarchy.** Evidence is the paired dense/MMNN
  campaign at affine depths 3, 5, and 7, with all parameter blocks trainable.
- **E3 — Muon sector-clock reshaping.** Evidence is the disjoint-calibration,
  paired-confirmation comparison of maximal-update gradient descent, polar
  Muon, and powers `1/3` and `2/3` on sparse and power-law Fourier targets.
- **E4 — no universal optimizer dominance.** Sector half-error clocks,
  right-censored event coverage, and paired endpoint loss are reported as
  distinct estimands.
- **E5 — numerical controls.** Width transfer does not retune the learning
  rate. A separate control halves, quarters, and eighths the step while
  increasing the horizon at fixed optimizer time, and doubles the periodic
  grid resolution.
  Base-to-half endpoint sensitivity is reported rather than called continuum
  convergence; quarter-to-eighth and grid agreement of the $L^2$ residual norm
  are audited groupwise, while squared-loss drift is retained. The
  dense Muon^1/3 clock advantage survives these controls, whereas its small
  MMNN power-law advantage is discretization-sensitive. A separate
  ill-conditioned regression excludes the Gram-eigendecomposition shortcut
  for the polar update.

The empirical entries are accepted only after the full raw-data finiteness,
stability, paired-seed, plot-source, and PDF audits pass. Smoke runs are not
evidence for final numerical values.
