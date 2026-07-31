# PCG as the next decoder, not as a DMFT proxy

## Decision

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

It should not start from the current overdetermined dual system.  When the
prompt has \(M>K\), the dual matrix has \(M-K\) directions that are invisible to
the coefficient readout.  Those directions made the Richardson convergence
certificate nearly marginal even when coefficient error was good.  The primal
system removes this mismatch and is SPD for every \(\lambda>0\).

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
dot products.  Comparisons must be made per matrix--vector product and per
FLOP, not only per nominal Transformer layer, because a PCG step contains more
arithmetic than a Richardson step.

## First theorem to prove

Let \(A,\widehat A\succeq a_0I\), and let PCG with a prompt-dependent but
depth-independent SPD inverse preconditioner \(P\) solve
\(\widehat A u=f\), from \(u_0=0\).  Assume

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

Consequently, for \(f\sim\mathcal N(0,\Sigma)\),

\[
\boxed{
\mathcal R_{u,L}^{\rm PCG}
\le
\frac{8\operatorname{tr}\Sigma}{a_0^2}
\rho_{\rm CG}^{2L}
+\frac{2D\operatorname{tr}\Sigma}{a_0^4}\mathcal R_A
}.
\]

This is the direct PCG replacement for the current Richardson transfer bound.
It changes the solver depth from

\[
L_{\rm Rich}=O\!\left(\kappa\log(1/\epsilon)\right)
\quad\text{to}\quad
L_{\rm PCG}=O\!\left(\sqrt\kappa\log(1/\epsilon)\right).
\]

The same proof applies to the ridge coefficient system by replacing
\(\widehat A\) with \(H_{\mathcal C}\).  Unlike Richardson, PCG is not a fixed
linear map of the right-hand side: its polynomial depends on the right-hand
side through \(\alpha_\ell,\beta_\ell\).  The exact conditional Gaussian trace
identity from the Richardson paper therefore does **not** transfer verbatim;
the deterministic norm bound above does.

## Benchmark required before training a Transformer

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
3. Chebyshev semi-iteration (important fixed-coefficient competitor);
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

## Implemented experiment

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

The next milestone is no longer to emulate PCG inside generic Transformer
blocks.  It is to scale the learned preconditioner to larger \(K\), compare
against incomplete Cholesky/deflation baselines at matched wall-clock cost,
and then connect the learned encoder error to the PCG transfer theorem.

## Closest external reference

Rudikov et al., *Neural operators meet conjugate gradients: The FCG-NO method
for efficient PDE solving* (2024), uses learned neural operators as
preconditioners for flexible CG.  It is a useful baseline and a warning: a
nonlinear, iteration-dependent learned preconditioner belongs to FCG, not
standard PCG.  The present opportunity is different: implement the Krylov
solver itself in-context and jointly condition its SPD preconditioner on the
prompt.
