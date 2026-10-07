# Prompt-conditioned PCG decoder

## Problem formulation

The next project should optimize the decoder itself.  The current Richardson
analysis remains the control experiment and the encoder--decoder transfer
template, but PCG is the main candidate solver.

The first PCG decoder should solve the **primal SPD ridge system**

\[
H_{\mathcal C} z=c_{\mathcal C},\qquad
H_{\mathcal C}=\widehat G_{\mathcal C}^{\top}
\widehat G_{\mathcal C}+\lambda I_K,
\qquad
c_{\mathcal C}=\widehat G_{\mathcal C}^{\top}\widehat b_{\mathcal C}.
\]

When \(M>K\), the primal system is smaller than the overdetermined dual system
and is SPD for every \(\lambda>0\). The dual matrix also has \(M-K\) directions
that are invisible to the coefficient readout; these directions can make a
full dual residual bound weak despite small coefficient error.

## Exact recurrent PCG target

Let \(P_{\theta}(\mathcal C)\) be an inverse preconditioner that is fixed during
one forward pass.  Require

\[
P_{\theta}(\mathcal C)=L_{\theta}(\mathcal C)
L_{\theta}(\mathcal C)^{\top}+\varepsilon I\succ0.
\]

Starting from \(z_0=0\), the loop is

\[
\begin{aligned}
r_0&=c-Hz_0,& y_0&=P_\theta r_0,& p_0&=y_0,\\
\alpha_\ell&=\frac{r_\ell^\top y_\ell}
{p_\ell^\top H p_\ell},&
z_{\ell+1}&=z_\ell+\alpha_\ell p_\ell,&
r_{\ell+1}&=r_\ell-\alpha_\ell Hp_\ell,\\
y_{\ell+1}&=P_\theta r_{\ell+1},&
\beta_\ell&=\frac{r_{\ell+1}^\top y_{\ell+1}}
{r_\ell^\top y_\ell},&
p_{\ell+1}&=y_{\ell+1}+\beta_\ell p_\ell.
\end{aligned}
\]

One loop therefore needs persistent states \((z,r,y,p)\), one application of
\(H\), one application of \(P_\theta\), two global inner products, and scalar
division.  For weak-form tokens,

\[
Hp=\sum_{i=1}^{M} g_i(g_i^\top p)+\lambda p,
\]

so attention can implement the matrix--vector product without materializing
\(H\).  A state token carries \((z,r,y,p)\); equation tokens compute
\(g_i^\top p\) and return \(g_i(g_i^\top p)\); global pooling computes the two
dot products. Comparisons must report matrix--vector products and FLOPs because
a PCG step contains more arithmetic than a Richardson step. Nominal
Transformer depth alone is not a matched computational budget.

## Encoder--PCG transfer bound

Let \(A,\widehat A\succeq a_0I\). Let
\(f\sim\mathcal N(0,\Sigma)\) be independent of
\((A,\widehat A,P)\), and let PCG with a prompt-dependent,
depth-independent SPD inverse preconditioner \(P\) solve
\(\widehat A u=f\) from \(u_0=0\). Define

\[
\mathcal R_A=\frac1D\mathbb E\|\widehat A-A\|_F^2,
\qquad
\mathcal R_{u,L}^{\rm PCG}
=\mathbb E\|u_L-A^{-1}f\|_2^2.
\]

Assume

\[
\kappa\!\left(P^{1/2}\widehat A P^{1/2}\right)\le \bar\kappa,
\qquad
\rho_{\rm CG}=\frac{\sqrt{\bar\kappa}-1}
{\sqrt{\bar\kappa}+1}.
\]

The standard PCG energy estimate and the resolvent identity give, pointwise,

\[
\|u_L-A^{-1}f\|_2
\le
\left(
\frac{2\rho_{\rm CG}^{L}}{a_0}
+\frac{\|\widehat A-A\|_{\rm op}}{a_0^2}
\right)\|f\|_2.
\]

Consequently,

\[
\mathcal R_{u,L}^{\rm PCG}
\le
\frac{8\operatorname{tr}\Sigma}{a_0^2}
\rho_{\rm CG}^{2L}
+\frac{2D\operatorname{tr}\Sigma}{a_0^4}\mathcal R_A.
\]

Indeed, PCG gives the first term in the pointwise estimate in the
\(\widehat A\)-energy norm. Coercivity converts it to the Euclidean norm.
After squaring the pointwise estimate, independence of \(f\) and
\((A,\widehat A)\), together with
\(\|\widehat A-A\|_{\rm op}\leq\|\widehat A-A\|_F\), gives the displayed
risk bound.

This is the direct PCG replacement for the current Richardson transfer bound.
It changes the solver depth from

\[
L_{\rm Rich}=O\!\left(\kappa\log(1/\epsilon)\right)
\quad\text{to}\quad
L_{\rm PCG}=O\!\left(\sqrt\kappa\log(1/\epsilon)\right).
\]

The same proof applies to the ridge coefficient system with
\(\widehat A=H_{\mathcal C}\). PCG is nonlinear in the right-hand side because
its polynomial depends on \(\alpha_\ell,\beta_\ell\). Consequently, the
conditional Gaussian trace identity for a fixed Richardson polynomial does
not apply. The deterministic norm bound remains valid.

## Solver comparisons

Use identical sampled tasks for every method and report error against exact
ridge/Cholesky as a function of:

- number of \(H\)-vector products;
- wall-clock time and attention FLOPs;
- recurrent state memory;
- condition number and spectral shape;
- prompt length, coefficient dimension, and ridge level;
- float64, float32, bfloat16, and perturbed dot products;
- in-distribution and out-of-distribution spectra.

The minimum solver table is:

1. optimal scalar Richardson;
2. Jacobi/Richardson;
3. Chebyshev semi-iteration;
4. unpreconditioned CG;
5. Jacobi-PCG;
6. low-rank spectral/deflated PCG;
7. learned-SPD PCG;
8. flexible or restarted CG when the learned preconditioner varies by layer.

PCG should be called the best decoder only if it wins on error versus
matrix--vector products and retains the advantage after accounting for its
extra reductions and recurrent state.

## Training target

Do not train the preconditioner only to approximate \(H^{-1}\) in Frobenius
norm.  Train it for the actual finite-depth objective

\[
\min_\theta\;\mathbb E_{\mathcal C,c}
\|z_L^{\rm PCG}(H_{\mathcal C},c;P_\theta)-H_{\mathcal C}^{-1}c\|_{H_{\mathcal C}}^2,
\]

with an SPD parameterization and explicit monitoring of
\(\kappa(P_\theta^{1/2}H P_\theta^{1/2})\).  A secondary spectral loss can
penalize the spread of stochastic Rayleigh quotients.  If \(P_{\theta,\ell}\)
depends on the current residual or changes with depth, standard PCG theory no
longer applies; that experiment must be labeled flexible CG (FCG) or restarted
PCG.

## Finite-dimensional solver isolation

The constructive lab now supports paired Richardson/PCG sweeps:

```bash
uv run --no-sync python \
  experiments/transformers/constructive_weakform_richardson_transformer.py \
  --mode sweep_solver --solver-grid richardson,pcg \
  --K 16 --prompt-len 128 --design correlated --cond 1000 \
  --precond jacobi --depth-grid 1,2,4,8,16,32 \
  --device cuda --dtype float64 \
  --outdir data/transformers/runs_pcg_vs_richardson
```

The constructive PCG trace and the learned-SPD experiment are now implemented
in `structured_one_head_heavyball.py`.  Despite its historical filename, the
script accepts `--solver-cell pcg`.  In that mode the model has one attention
head, no MLP, and no learned solver coefficients.  The fixed prompt-dependent
preconditioner is trained through the exact finite-depth PCG objective.

For the initial \(K=8,m=32,\kappa_{\rm design}=100,L=4\) isolation run, the
learned-PCG MSE is \(1.35\times10^{-3}\), compared with
\(2.47\times10^{-3}\) for PCG--Jacobi.  The same frozen model beats
PCG--Jacobi at all 12 tested combinations of prompt length
\(m\in\{16,32,64,128\}\) and condition number
\(\{10,100,1000\}\).

The next milestone is to scale the learned preconditioner to larger \(K\),
compare it with incomplete-Cholesky and deflation baselines at matched
wall-clock cost, and connect the learned encoder error to the PCG transfer
bound.

## Learned preconditioners for flexible CG

Rudikov et al., *Neural operators meet conjugate gradients: The FCG-NO method
for efficient PDE solving* (2024), uses learned neural operators as
preconditioners for flexible CG.  It is a useful baseline and a warning: a
nonlinear, iteration-dependent learned preconditioner belongs to FCG, not
standard PCG.  The present opportunity is different: implement the Krylov
solver itself in-context and jointly condition its SPD preconditioner on the
prompt.

## NQF initialization regimes

The NQF normal forms sharpen the initialization policy but do not modify PCG.
For full multi-head attention with \(Q,K,V,O\) all initialized at scale
\(\varepsilon\), value/readout corrections enter the output at quadratic
order while query-key routing enters at quartic order. At initialization
scale \(\varepsilon\), the Q/K loss gradient is \(O(\varepsilon^3)\), compared
with \(O(\varepsilon)\) for V/O. This parameterization suppresses the local
learning signal for prompt-dependent routing.

The decoder head is query-key-only: its values are fixed weak rows. In this
case Q/K routing is already quadratic and has an order-\(\varepsilon\) loss
gradient. Two initialization regimes retain a leading residual signal:

1. train Q/K with fixed nonzero values, as in the current head; or
2. fix nonzero kernel-matched Q/K features and train only value/readout
   residual channels.

An all-blocks-small initialization does not learn Q/K preconditioner
directions at leading NQF order. Rank-deficient QR at zero also violates the
smoothness assumption used by the NQF expansion.

NQF modes are parameter order parameters indexed by training time. PCG modes
are eigencomponents of \(P_\theta^{1/2}HP_\theta^{1/2}\) indexed by inference
iteration. Relating them requires an additional alignment result. The
standard PCG theorem only requires that the trained SPD preconditioner remain
fixed during a solve.

The initialization audit in
`audit_nqf_attention_residual_corrections.py`: Taylor slopes, blockwise
gradient slopes, the pre-QR QK normal-form remainder, and separate labels for
full MHA and QK-only heads. Subsequent PDE runs should also record
order-parameter eigenvalues during training and their overlaps with slow
eigenvectors of the effective preconditioned operator.
