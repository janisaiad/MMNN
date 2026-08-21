# Structured one-head decoder: learn geometry, hard-code algebra

## Design rule

The decoder is not asked to rediscover a known iterative algorithm.  Every
deterministic linear relation is frozen.  The attention head learns only the
prompt-dependent spectral information that is unavailable a priori.

| Component | Status | Reason |
|---|---|---|
| weak system \(H=G^\top G+\lambda I\), \(c=G^\top b\) | frozen | definition of the encoded inverse problem |
| residual \(r_\ell=c-Hz_\ell\) | frozen | exact solver algebra |
| ridge drift and attention masks | frozen | known problem structure |
| state shift \((z,r,y,p)\) | frozen | exact PCG memory |
| PCG inner products and scalar divisions | frozen | known Krylov algebra |
| spectral slots/directions | learned | task-dependent geometry |
| strength of each spectral correction | learned, positive | finite-depth preconditioner design |
| PCG \(\alpha_\ell,\beta_\ell\) | computed exactly, not learned | prompt- and iteration-dependent Krylov coefficients |
| MLP/FFN | absent | no unknown nonlinear map is needed |

## NQF initialization audit: full MHA is not QK-only attention

The local normal form depends on which blocks are small and trainable. For a
full head

\[
o^\top V X\operatorname{softmax}(X^\top K^\top Qx/\sqrt{d_k}),
\]

scaling all of \(Q,K,V,o\) by \(\varepsilon\) gives an order-
\(\varepsilon^2\) uniform-value term and an order-\(\varepsilon^4\) routed
term. Consequently the loss gradients of \(V/O\) and \(Q/K\) are respectively
orders \(\varepsilon\) and \(\varepsilon^3\) for an order-one residual.

Our head is different: values are fixed to normalized weak rows and only the
keys and slot queries are learned. Before QR,

\[
u_s=\bar g+\frac{1}{\sqrt{d_h}}\Sigma_gW_K^\top q_s
+O(\|(W_K,q_s)\|^4),
\]

so the prompt-covariance correction is quadratic and its Q/K loss gradient is
order \(\varepsilon\). This is exactly the query-key-only NQF normal form of
Liu Ziyin, Xu, Poggio, and Chuang (2026), not the full-MHA normal form.

The statement stops before orthogonalization. QR is not smooth at a
rank-deficient zero matrix, and NQF does not turn QR, Ritz, Cholesky, scalar
reductions, or PCG divisions into learned operations. The production head
also starts from a nonzero identity-like key map and nonzero queries, rather
than the all-zero asymptotic point.

The focused float64 audit in
`audit_nqf_attention_residual_corrections.py` recovered slopes
\(2.0000,4.0001\) for the total/routed full-MHA output,
\(3.0001,1.0001\) for its QK/VO gradients, and \(2.0000,1.0000\) for the
QK-only routed output/gradient. The decoder's pre-QR Taylor remainder has
slope \(4.0000\). These are trainability diagnostics near initialization,
not final PDE-error claims.

| Claim | Status |
|---|---|
| full-MHA and QK-only Taylor orders | theorem/direct expansion, numerically checked |
| pre-QR decoder covariance normal form | proved and checked |
| exact QR--Ritz--PCG algebra | unchanged and exact |
| complete decoder is an NQF at zero | not proved; QR degeneracy violates the smooth setup |
| NQF modes equal PCG Krylov modes | not claimed; requires a new alignment theorem |
| GLV switch times on generic PDE prompts | open unless commutativity/isotropy is established |

## One head, several spectral slots

The number of heads and the number of learned directions are different
quantities.  Use one head with \(S\) learned query tokens (spectral slots):

\[
q_s\in\mathbb R^{d_h},\qquad
k_i=W_K\widetilde g_i,\qquad
v_i=W_V\widetilde g_i,
\]

\[
a_{si}=\operatorname{softmax}_i(q_s^\top k_i/\sqrt{d_h}),
\qquad
u_s=\sum_i a_{si}v_i.
\]

Thus there is one attention geometry, but it may return several directions
\(u_1,\ldots,u_S\).  Multiple value channels (posterior mean, variance, or
several right-hand sides) also share this same attention matrix.

## Hard-coded SPD preconditioner

Let \(D=\operatorname{diag}(H)\) and let the normalized slot directions form
\(U=[u_1,\ldots,u_S]\).  The inverse preconditioner is constructed outside an
MLP as

\[
P_\theta(\mathcal C)
=D^{-1/2}
\frac{I+U\operatorname{diag}(w)U^\top}{1+\sum_s w_s}
D^{-1/2},
\qquad w_s\geq0.
\]

This parameterization is SPD for every prompt.  Jacobi is recovered at
\(w=0\).  The normalization bounds the learned normalized inverse from above;
the low-rank term changes the relative weight of the selected directions.
Applying
\(P_\theta r\) is exact linear algebra; the head is responsible only for
choosing \(U\) and \(w\).

For the HeavyBall ablation, the tied decoder is

\[
\boxed{
\begin{aligned}
r_\ell&=c-Hz_\ell,\\
s_\ell&=P_\theta(\mathcal C)r_\ell,\\
z_{\ell+1}&=z_\ell+\alpha s_\ell
+\beta(z_\ell-z_{\ell-1}).
\end{aligned}}
\]

No division by a learned quantity and no MLP approximation occurs in the
loop.  The diagonal square root is a deterministic tokenizer/preconditioner
operation.  If an absolutely vanilla Transformer interface is required, its
result can be supplied as a fixed token feature.

The main decoder now gives the same fixed \(P_\theta(\mathcal C)\) to an
explicit PCG cell.  Its dot products, divisions, residual update, and search
direction recurrence are exact tensor operations.  In particular, the PCG
model contains no trainable step-size or momentum parameter: all trainable
parameters belong to the single preconditioner head.

## What must be demonstrated

1. **Solver isolation:** with exact \(G,b\), train only
   \((W_K,W_V,q_s,w,\alpha,\beta)\) and compare to Jacobi Richardson,
   Jacobi HeavyBall, Chebyshev, and PCG.
2. **Mechanism:** measure principal-angle overlap between the learned slot
   span and the slow eigenspace of
   \(D^{-1/2}HD^{-1/2}\).
3. **Ablations:** one head versus several heads at fixed total slot count;
   learned slots versus random slots; frozen versus learned \(\alpha,\beta\).
4. **Stability:** certify the roots of
   \(r^2-(1+\beta-\alpha\lambda)r+\beta\) on every evaluation prompt.
5. **End to end:** unfreeze the encoder only after the decoder beats the
   stationary baselines in solver isolation.
6. **Initialization blocks:** report Q/K and V/O gradient norms versus
   initialization scale, and keep full MHA separate from fixed-value QK-only
   attention.
7. **Two spectra:** track NQF order-parameter modes over training and the
   preconditioned PDE spectrum over inference without identifying them.

The key scientific claim is therefore not that a generic FFN happens to mimic
an inverse.  It is that one attention head identifies useful prompt-dependent
geometry, and that a fixed SPD construction converts it into a preconditioner
for an exact PCG loop.

## First solver-isolation result

For \(K=8\), prompt length 32, correlated design condition 100, two spectral
slots, depth 4, and float64, a 2000-step decoder-only run gave:

| method/control | coefficient MSE |
|---|---:|
| Richardson--Jacobi | \(1.323\times10^{-1}\) |
| HeavyBall--Jacobi, prompt-wise oracle coefficients | \(1.680\times10^{-2}\) |
| initial one-head geometry + final learned coefficients | \(1.558\times10^{-2}\) |
| trained one-head geometry + final learned coefficients | \(1.470\times10^{-2}\) |
| Chebyshev--Jacobi | \(9.91\times10^{-3}\) |
| PCG--Jacobi | \(2.47\times10^{-3}\) |

The learned slot span overlap with the slow normalized eigenspace increased
from 0.171 to 0.386; its overlap with the solution-weighted target directions
increased from 0.295 to 0.428.  Comparing the trained and initial heads with
the same final \((\alpha,\beta)\) isolates an approximately 5.6% gain from the
learned attention geometry itself.  The full structured decoder improves on
oracle-coefficient Jacobi HeavyBall by approximately 12.5%.

This is evidence that the head learns a useful space, not evidence that the
current preconditioner is optimal: Chebyshev and PCG remain stronger at the
same matrix--vector-product depth.

## Direct learned-PCG result

Keeping the same task and architecture but training the head through the
exact depth-4 PCG objective changes the conclusion:

| method | coefficient MSE |
|---|---:|
| Richardson--Jacobi | \(1.323\times10^{-1}\) |
| Chebyshev--Jacobi | \(9.91\times10^{-3}\) |
| PCG--Jacobi | \(2.47\times10^{-3}\) |
| one-head learned-preconditioner PCG | \(1.35\times10^{-3}\) |

This is a 45.1% reduction relative to PCG--Jacobi.  Slow-space overlap rises
from 0.170 at initialization to 0.463, while the preconditioner strength rises
from 0.050 to 0.284.

The frozen model also beats PCG--Jacobi on every point of a 12-point OOD grid
with prompt lengths \(m\in\{16,32,64,128\}\) and design condition numbers
\(\{10,100,1000\}\).  Representative comparisons are:

| \((m,\kappa_{\rm design})\) | learned PCG | PCG--Jacobi | reduction |
|---|---:|---:|---:|
| \((16,100)\) | \(2.90\times10^{-2}\) | \(4.64\times10^{-2}\) | 37.5% |
| \((32,1000)\) | \(1.84\times10^{-3}\) | \(4.83\times10^{-3}\) | 62.0% |
| \((64,100)\) | \(8.29\times10^{-5}\) | \(1.61\times10^{-4}\) | 48.4% |
| \((128,1000)\) | \(2.14\times10^{-5}\) | \(4.10\times10^{-5}\) | 47.8% |

The precise claim supported by this experiment is therefore a
**Transformer-conditioned PCG decoder**: attention learns only the fixed SPD
preconditioner, and PCG supplies all solver algebra.

## Stronger hard-coded base: block-Jacobi plus learned global correction

Jacobi is not the strongest local preconditioner.  The architecture therefore
also supports

\[
B_0=\operatorname{blockdiag}(H_{11}^{-1},\ldots,H_{JJ}^{-1}),
\qquad B_0=L_0L_0^\top,
\]

followed by the same one-head correction in normalized coordinates,

\[
P_\theta=L_0
\frac{I+U_\theta\operatorname{diag}(w)U_\theta^\top}
{1+\sum_s w_s}
L_0^\top.
\]

All block inversions are hard-coded.  Attention learns only the global
directions missing from the local block solves.

At \(K=8,m=32,\kappa_{\rm design}=100,L=4\), exact block-Jacobi PCG gives
\(4.92\times10^{-4}\), whereas its learned one-head correction gives
\(1.91\times10^{-4}\), a 61.2% reduction.  On the 12-point OOD grid, the
corrected solver beats block-Jacobi in 11 cases, with a 42.5% average relative
reduction.  The exception is \((m,\kappa)=(128,1000)\), where the learned
solver is 26.4% worse; this failure must remain part of the reported OOD
result.

A strength sweep diagnoses this failure without changing the learned
directions: the best correction strength is approximately 0.8 at \(m=16\),
0.4 at \(m=32\), and 0.1 at \(m=128\).  The resulting deterministic
normalization

\[
w_s(m)=\min\!\left\{w_{\max},\frac{m_{\rm ref}}{m}w_s(m_{\rm ref})\right\}
\]

is therefore hard-coded rather than delegated to a network.  With
\(m_{\rm ref}=32\), the same frozen attention head then beats block-Jacobi PCG
on all 12 OOD points.  The mean relative reduction is 49.2%, ranging from
30.2% to 72.4%.  This result isolates the desired division of labor: prompt
normalization is known algebra, while prompt-dependent directions remain
learned.

The first dimension-scaling experiment uses \(K=16,m=64\), four coordinate
blocks, four spectral slots, and depth 4.  After 1000 training steps:

| method | coefficient MSE |
|---|---:|
| PCG--Jacobi | \(6.60\times10^{-3}\) |
| PCG--block-Jacobi | \(3.57\times10^{-3}\) |
| one-head correction + PCG--block-Jacobi | \(2.77\times10^{-3}\) |

Thus the learned correction retains a 22.3% gain over the stronger hard-coded
base at twice the coefficient dimension, although the gain is smaller than at
\(K=8\).

## Slot collapse and the (K=32) correction

Naively increasing the number of queries does not produce a higher-rank
preconditioner.  With independent normalization, four slots at \(K=16\) and
eight slots at \(K=32\) both collapsed to effective rank approximately 1;
the mean absolute off-diagonal slot cosine at \(K=32\) was 0.9996.  Longer
training reduced the learned strength toward zero rather than fixing the
collapse.

The decoder now optionally hard-codes

\[
U_\theta=\operatorname{qf}(\widetilde U_\theta)
\]

using the reduced QR factor of the raw attention outputs.  This introduces no
MLP and no additional head.  Attention still chooses the span; QR merely
enforces the known geometric requirement that the spectral slots form a
non-redundant orthonormal basis.

For \(K=32,m=128,L=8\), eight blocks and eight QR slots, this raises the
slow-space overlap from approximately 0.13 to 0.36.  The coefficient MSE is
\(6.12\times10^{-6}\), compared with \(6.79\times10^{-6}\) for block-Jacobi
PCG.  The learned preconditioner also wins at each tested depth
\(L\in\{2,4,6,8,10,12\}\).

The wall-clock conclusion is deliberately weaker.  In single-threaded CPU
tests with batched dense systems, construction plus solve costs approximately
4.81 ms versus 2.35 ms at \(K=8\), 14.24 ms versus 6.28 ms at \(K=16\), and
31.17 ms versus 21.31 ms at \(K=32\), for learned versus block-Jacobi PCG.
Dense direct solves remain substantially faster at these small dimensions.
The current evidence therefore establishes improved error per matrix--vector
product, not a universal wall-clock victory.
