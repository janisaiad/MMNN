# HeavyBall as the first pure looped-Transformer acceleration

## Architectural decision

The first accelerated decoder should be **preconditioned HeavyBall**, not a
Transformer imitation of PCG.  PCG remains the numerical upper baseline and a
useful external decoder, but its adaptive dot products and divisions are not
the cleanest object for a vanilla recurrent Transformer theorem.

Let the existing Richardson block implement

\[
\mathcal R_{\mathcal C}(z)
=z+B_{\mathcal C}(c_{\mathcal C}-H_{\mathcal C}z).
\]

The HeavyBall update is exactly

\[
\begin{aligned}
z_{\ell+1}
&=z_\ell+\alpha B_{\mathcal C}(c_{\mathcal C}-H_{\mathcal C}z_\ell)
  +\beta(z_\ell-z_{\ell-1})\\
&=\alpha\mathcal R_{\mathcal C}(z_\ell)
 +(1-\alpha+\beta)z_\ell-\beta z_{\ell-1}.
\end{aligned}
\]

This identity is the key implementation result.  Starting from any attention
construction of a Richardson block, HeavyBall only requires:

1. two iterate channels, carrying \((z_\ell,z_{\ell-1})\);
2. the same attention path already used by Richardson for the prompt operator;
3. a fixed linear output map with coefficients
   \((\alpha,1-\alpha+\beta,-\beta)\);
4. the channel shift \((z_{\ell+1},z_\ell)\) before the next tied loop.

It introduces no reciprocal, adaptive inner product, or new scalar
multiplication to approximate with an MLP.  If the Richardson operator is
implemented up to error \(\varepsilon_{\rm att}\), the HeavyBall block inherits
that same approximation, scaled by \(|\alpha|\); momentum adds no new operator
approximation.  This is a relative statement: it does not claim that an
arbitrary signed matrix-vector product is exactly represented by ordinary
softmax attention.  It says that acceleration is exact relative to the
Richardson attention primitive already established in the model.

With tied weights, one loop represents one HeavyBall iteration.  With
\(z_{-1}=z_0=0\), the first iteration has zero momentum automatically.  The two
scalars \((\alpha,\beta)\) may be learned globally.  The preconditioner
\(B_{\mathcal C}\) may be prompt-conditioned, but it must remain fixed across
the solve if the stationary spectral theorem below is invoked.

## One head is the reference architecture

The main solver should use one attention head, not a multi-head mixture.  In
the GP/KRR construction of Kang, Lee, and Cheng (2026), a single kernel
attention operator advances both posterior mean and variance recursions in
parallel value channels.  Multi-head attention is described as an optional
extension for distinct components of a hierarchical GP mixture, rather than
as a requirement for one linear solve:

<https://arxiv.org/html/2605.26713v1>

For a positive RBF kernel, normalized attention is exactly the row-stochastic
operator \(D^{-1}G\).  The context and query state update can therefore be
written, for each carried right-hand-side channel, as

\[
u_{\ell+1}(x_j)=u_\ell(x_j)
+\alpha\left[
\sum_i\frac{G_{ji}}{s_j}\bigl(v_i-u_\ell(x_i)\bigr)
-\frac{\lambda}{s_j}u_\ell(x_j)
\right]
+\beta\bigl(u_\ell(x_j)-u_{\ell-1}(x_j)\bigr).
\]

One head supplies the normalized kernel correction simultaneously to all
value channels.  A linear skip supplies the ridge drift and the HeavyBall
memory.  This preserves one stationary operator and therefore one meaningful
spectrum.  Extra heads are retained only as an ablation or for genuinely
different kernel components; they should not be used merely to increase
capacity in the reference theorem.

This argument applies to strictly positive kernels.  It does not rescue a
softmax head applied to a signed weak-form Gram matrix: a row-stochastic
softmax output cannot in general equal \(GG^\top\alpha\).  The signed primal
weak-form branch must use an exact linear/bilinear attention primitive or an
alternative positive-kernel representation.

## Spectral dynamics and stability

Assume \(H\succ0\), \(B\succ0\), and let

\[
S=B^{1/2}HB^{1/2},\qquad
\sigma(S)\subseteq[m,M].
\]

In an eigenmode \(\lambda\in[m,M]\), the transformed error satisfies

\[
\xi_{\ell+1}
=(1+\beta-\alpha\lambda)\xi_\ell-\beta\xi_{\ell-1}.
\]

The exact roots are those of

\[
r^2-(1+\beta-\alpha\lambda)r+\beta=0.
\]

A convenient strict stability region is

\[
0\leq\beta<1,
\qquad
0<\alpha M<2(1+\beta).
\]

When exact bounds \(m,M\) are known, the classical quadratic optimum is

\[
\alpha_\star=\frac{4}{(\sqrt M+\sqrt m)^2},\qquad
\beta_\star=
\left(\frac{\sqrt M-\sqrt m}{\sqrt M+\sqrt m}\right)^2,
\qquad
q=\sqrt{\beta_\star}
=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
\]

At the spectral endpoints the characteristic roots coalesce.  Therefore the
finite-depth theorem must retain the Jordan/transient factor; it must not state
the false bound \(\|e_L\|\le q^L\|e_0\|\) with constant one.  For the
initialization \(e_{-1}=e_0\), a safe uniform polynomial bound is

\[
\|e_L\|_{B^{-1}}
\le C_L(q)q^L\|e_0\|_{B^{-1}},
\qquad
C_L(q)\le 1+L(1+q).
\]

The exact scalar polynomial, rather than this loose envelope, should be used
for plots and risk calculations.

## Encoder--decoder transfer

Let the encoder produce \(\widehat A\succeq a_0I\), while the target is
\(A\succeq a_0I\).  Let \(p_L\) be the HeavyBall residual polynomial for
\(B^{1/2}\widehat A B^{1/2}\).  Conditional on the encoded operator, the
decoder is linear in the right-hand side:

\[
\widehat u_L
=\widehat A^{-1}(I-p_L(B\widehat A))f.
\]

Unlike PCG, its coefficients do not depend on \(f\).  Consequently the exact
conditional Gaussian trace calculation used for Richardson remains available:

\[
\mathbb E_f\|\widehat u_L-\widehat A^{-1}f\|_2^2
=\operatorname{tr}
\left(\Sigma_f E_L^\top E_L\right),
\quad
E_L=\widehat A^{-1}p_L(B\widehat A).
\]

Combining this with the resolvent identity gives the deterministic transfer
template

\[
\|\widehat u_L-A^{-1}f\|_2
\le
\left[
\frac{C_L(q)q^L}{a_0}
+\frac{\|\widehat A-A\|_{\rm op}}{a_0^2}
\right]\|f\|_2.
\]

Thus, if
\(\mathbb E\|\widehat A-A\|_{\rm op}^2\le D\mathcal R_A\),

\[
\mathcal R_{u,L}^{\rm HB}
\le
\frac{2\operatorname{tr}(\Sigma_f)}{a_0^2}
C_L(q)^2q^{2L}
+\frac{2D\operatorname{tr}(\Sigma_f)}{a_0^4}\mathcal R_A.
\]

This has the same encoder term as Richardson and the accelerated spectral
factor of HeavyBall.  The sharper trace expression should replace the norm
envelope whenever the eigenvector alignment with \(\Sigma_f\) is modeled.

For replica/RMT calculations, introduce the augmented first-order state

\[
\begin{bmatrix}e_{\ell+1}\\e_\ell\end{bmatrix}
=
\begin{bmatrix}
(1+\beta)I-\alpha B\widehat A&-\beta I\\
I&0
\end{bmatrix}
\begin{bmatrix}e_\ell\\e_{\ell-1}\end{bmatrix}.
\]

Equivalently, the decoder risk is a spectral integral weighted by
\(p_L(\lambda)^2\).  This fixed-polynomial structure is materially simpler
than PCG, whose random Krylov coefficients depend on the right-hand side.

## HeavyBall versus Chebyshev

Chebyshev semi-iteration is the stronger fixed-polynomial comparison when
reliable spectral bounds are supplied.  It also avoids divisions and adaptive
dot products, but its coefficients change with depth.  Therefore:

- a tied recurrent block with no iteration controller naturally implements
  constant-coefficient HeavyBall;
- an untied depth-\(L\) Transformer can store the Chebyshev coefficients in its
  layer weights, with no arithmetic MLP;
- a tied loop can implement Chebyshev only after adding a depth state or a
  coefficient controller, which weakens the architectural minimality claim.

The experimental order should consequently be:

1. learn and analyze tied HeavyBall;
2. compare against PCG and oracle Chebyshev per matrix-vector product;
3. add a depth-conditioned Chebyshev loop only if its empirical gain justifies
   the extra state;
4. reserve external PCG with a Transformer-learned SPD preconditioner for the
   best practical hybrid decoder.

## Current paired evidence

For \(K=16\), prompt length 128, correlated design condition 1000, Jacobi
preconditioning, batch 64, and float64, the effective condition number is
approximately 3.16.  At depth 8, the coefficient MSE against the exact
posterior is:

| decoder | MSE |
|---|---:|
| Richardson | \(4.25\times10^{-3}\) |
| oracle HeavyBall | \(6.85\times10^{-7}\) |
| PCG | \(1.60\times10^{-8}\) |

Learning one pair shared by all eight loops produced
\(\alpha=1.5044\), \(\beta=0.1331\).  On a fresh paired evaluation its mean
coefficient error was \(9.70\times10^{-8}\), versus
\(2.48\times10^{-4}\) for Richardson, \(5.87\times10^{-8}\) for prompt-wise
oracle HeavyBall, and \(1.63\times10^{-9}\) for PCG.

These results support the architectural choice, but not yet the final claim.
The required next evidence is a trained attention model whose hidden-state
trajectory matches the constructive HeavyBall recurrence, followed by
out-of-distribution spectral tests and comparison by matrix-vector products,
wall time, and parameter budget.

### First one-head attention ablation

A tied one-head normalized-RBF model was trained with no MLP and depth 8.  Its
only learned solver parameters were the shared step and momentum; the kernel
was fixed to the data-generating lengthscale.  On the same seeded evaluation
batch, the learned decoders gave:

| decoder | context-state MSE | query-mean MSE |
|---|---:|---:|
| one-head Richardson | \(8.67\times10^{-3}\) | \(4.15\times10^{-2}\) |
| one-head HeavyBall | \(3.86\times10^{-3}\) | \(3.68\times10^{-2}\) |

HeavyBall learned \(\alpha=2.140\), \(\beta=0.272\) inside the imposed stable
region.  In a separate joint geometry/solver run, the single head moved its
RBF lengthscale from 0.40 to approximately 0.275 when the true value was 0.25.
This is encouraging but preliminary: the query metric has higher Monte Carlo
variance than the full context-state metric, and larger paired evaluations are
still required.
