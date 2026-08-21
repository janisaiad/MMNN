# A spectral-leap theory for cosine learning in right-factor MMNNs

## Executive statement

For one-dimensional cosine regression, the `leap` of Abbe, Boix-Adserà, and
Misiakiewicz cannot be identified with a numerical jump in Fourier frequency.
Their leap counts newly discovered latent coordinates (with Hermite
multiplicity for Gaussian inputs); every cosine on a one-dimensional torus
depends on the same coordinate. A separate, architecture-dependent notion is
needed.

For the fixed-left, right-factor-trained MMNN considered here, that notion is
the inverse Fourier eigenvalue of the right-factor tangent kernel:

\[
S_k:=\lambda_k^{-1}.
\]

It has three simultaneous meanings:

1. it is the relaxation time of mode $k$ under kernel gradient flow;
2. it is twice the residual loss gap divided by the squared parameter-gradient
   norm at the state which has fitted all lower modes;
3. in a fixed ReLU gate cell, it is the inverse Gauss--Newton curvature of that
   Fourier direction.

For a broad stationary ReLU right-factor kernel, a cusp calculation gives

\[
\lambda_k\sim C k^{-2},\qquad S_k\sim C^{-1}k^2.
\]

Consequently, if the already learned reference frequency is fixed and the next
target frequency is $k_{\rm next}=k_{\rm old}+\Delta k$, then the next
approximate saddle grows quadratically in the spectral leap:

\[
S_{k_{\rm next}}\asymp (k_{\rm old}+\Delta k)^2.
\]

This is a rigorous statement for stationary tangent-kernel gradient flow and a
conditional local statement for the finite MMNN. It is **not** a proof that the
finite nonlinear trajectory visits exact Morse saddles. The correct objects are
frequency-indexed approximate saddles or metastable plateaux. In fact, inside a
fixed ReLU gate cell the loss is locally quadratic and positive semidefinite in
the trained right factor, so calling these points exact saddles would generally
be false.

The accompanying finite-network experiment finds:

- right-factor tangent spectrum $\lambda_k\propto k^{-2.07}$ for $k\ge8$
  ($R^2=0.966$, five seeds), close to the predicted $k^{-2}$;
- median nonlinear escape time proportional to
  $\Delta k^{2.45}$ ($R^2=0.998$) when sweeping
  $\cos x+\tfrac12\cos((1+\Delta k)x)$;
- sequential recovery of modes $1,4,8,16$ in a single target.

The difference between the predicted exponent $2$ and the measured escape
exponent $2.45$ is not hidden: the prediction concerns a fixed tangent metric,
whereas the experiment uses Adam and crosses ReLU gate boundaries.

## 1. Precise architecture and interpretation of “NMF”

The experiment uses a two-hidden-layer network, i.e. three affine maps if the
scalar readout is counted:

\[
\begin{aligned}
\psi(x)&=(\cos x,\sin x),\\
h(x)&=\rho(B\psi(x)+b_1)\in\mathbb R^m,\\
g_V(x)&=UV^\top h(x)+b_2\in\mathbb R^p,\\
f_V(x)&=c^\top\rho(g_V(x)).
\end{aligned}
\]

Here $\rho(z)=\max(z,0)$, $U\in\mathbb R^{p\times r}$ is frozen,
$V\in\mathbb R^{m\times r}$ is trained, and $r\ll m,p$. Thus the inner
matrix

\[
W_2=UV^\top
\]

has rank at most $r$, and only its right factor is trained. All of
$B,b_1,U,b_2,c$ are fixed random draws.

This note interprets “NMF factorized” as the neural low-rank matrix
factorization used in the MMNN repository. It does not impose entrywise
nonnegativity on $U,V$. If literal nonnegative matrix factorization is
intended, the signed cosine target requires at least a signed readout, and
projected/softplus dynamics add boundary KKT saddles; the theorem below would
need to be restated for that constrained metric.

## 2. What transfers from leap complexity, and what does not

The source paper [SGD learning on neural networks: leap complexity and
saddle-to-saddle dynamics](https://arxiv.org/abs/2302.11055) expands a
low-latent-dimensional target in a Hermite basis (Gaussian inputs) or
Fourier--Walsh basis (Boolean inputs). If its nonzero multi-indices are
$S_1,\ldots,S_M$, its leap is

\[
\operatorname{Leap}(h_*)
=\min_{\pi}\max_i
\left\|S_{\pi(i)}\setminus\bigcup_{j<i}S_{\pi(j)}\right\|_1.
\]

It measures how many new latent coordinates, with multiplicity, must be aligned
at one stage. Their proved setting is a high-dimensional isotropic Gaussian
problem, a smooth two-layer network, layer-wise projected online SGD, and a
representative target class. Their general $d^{\operatorname{Leap}-1}$
online-step law is conjectural outside that setting.

On $\mathbb T^1$, the modes $\cos(kx)$ all use the same input coordinate.
Therefore:

- a raw frequency jump is not their leap;
- their theorem does not prove that learning time grows with $k$;
- the transferable mechanism is sequential motion through states whose
  unlearned directions have progressively smaller gradient signal.

For cosine data, define instead three related quantities:

\[
\Delta k_j=k_j-k_{j-1},\qquad
\mathfrak L_j=\log\frac{\lambda_{k_{j-1}}}{\lambda_{k_j}},\qquad
S_j=\lambda_{k_j}^{-1}.
\]

$\Delta k_j$ is the data-space leap, $\mathfrak L_j$ is the
architecture-aware spectral leap, and $S_j$ is the saddle/relaxation index.
The monotone theorem is about $S_j$, not about $\Delta k_j$ in isolation.
For example, the absolute next frequency generally matters more than the raw
gap: a jump from 100 to 110 can be harder than a jump from 1 to 16 even though
its gap is smaller.

## 3. Exact right-factor tangent kernel

Let

\[
d_V(x):=U^\top\big(c\odot \rho'(g_V(x))\big)\in\mathbb R^r.
\]

Away from ReLU gate boundaries, differentiation gives the rank-one matrix

\[
\nabla_V f_V(x)=h(x)d_V(x)^\top.
\]

Therefore the tangent kernel for training only $V$ is exactly

\[
\boxed{
K_V(x,x')
=\langle h(x),h(x')\rangle
 \langle d_V(x),d_V(x')\rangle .
}
\]

This identity is finite-width and does not require a Gaussian approximation.
It explains why the fixed-left factorization is analytically convenient: the
kernel is a product of a frozen-feature covariance and an $r$-dimensional
outer-gate covariance.

The normalization $U_{ij}=O(r^{-1/2})$ makes the mean scale of this kernel
order one. Rank then principally controls finite-$r$ fluctuations and the
number of independent gate channels. Under independent isotropic channels one
expects relative fluctuations of order $r^{-1/2}$, while very small $r$ can
make individual empirical Fourier directions nearly inaccessible.

## 4. Fourier dynamics theorem

Let $\mu(dx)=dx/(2\pi)$ on $\mathbb T=[0,2\pi)$. Write the orthonormal real
Fourier basis as

\[
e_{k,c}(x)=\sqrt2\cos(kx),\qquad
e_{k,s}(x)=\sqrt2\sin(kx).
\]

Assume the tangent kernel is stationary,
$K(x,x')=\kappa(x-x')$, continuous, positive semidefinite, and frozen during
the time interval under consideration. Its integral operator

\[
(T_Kq)(x)=\int K(x,x')q(x')\,d\mu(x')
\]

diagonalizes in Fourier space:

\[
T_Ke_{k,c}=\lambda_ke_{k,c},\qquad
T_Ke_{k,s}=\lambda_ke_{k,s},\qquad \lambda_k\ge0.
\]

### Theorem 1 (mode-wise relaxation and plateau hierarchy)

For population square loss

\[
\mathcal L(f)=\frac12\|f-y\|_{L^2(\mu)}^2
\]

and kernel gradient flow

\[
\partial_t f_t=-T_K(f_t-y),
\]

each Fourier coefficient error $z_k(t)=\langle f_t-y,e_k\rangle$ obeys

\[
z_k(t)=z_k(0)e^{-\lambda_kt}.
\]

Hence the time to reduce its magnitude from $a$ to $\varepsilon$ is

\[
\tau_k(\varepsilon)=\lambda_k^{-1}\log(a/\varepsilon).
\]

If $\lambda_k$ decreases with $k$ and target amplitudes are comparable, the
modes are learned in increasing frequency order. After modes below $k_j$ have
been fitted, any residual supported on modes $k\ge k_j$ satisfies

\[
-\frac{d}{dt}\mathcal L(t)
=\sum_{k\ge k_j}\lambda_kz_k(t)^2
\le 2\lambda_{k_j}\mathcal L(t).
\]

Thus a fixed-factor reduction of the remaining loss takes at least order
$\lambda_{k_j}^{-1}$. Spectral gaps between successive $\lambda_{k_j}$'s
produce separated plateaux.

#### Proof

Expand $f_t-y$ in the orthonormal eigenbasis of $T_K$. Projecting the flow
onto an eigenfunction gives the scalar ODE
$\dot z_k=-\lambda_kz_k$, which yields the exponential solution. The loss
identity follows from

\[
\dot{\mathcal L}
=\langle f_t-y,\partial_tf_t\rangle
=-\langle f_t-y,T_K(f_t-y)\rangle
=-\sum_k\lambda_kz_k^2.
\]

Monotonicity of $\lambda_k$ on the residual support gives the last bound.
$\square$

## 5. Approximate-saddle theorem

Suppose a parameter $V_{<k}$ represents all already learned modes exactly and
the only omitted target term is $a\cos(kx)=b e_{k,c}(x)$, where
$b=a/\sqrt2$. Let $\lambda_k$ be the tangent eigenvalue at this state.
Then

\[
\Delta\mathcal L_k=\frac12b^2=\frac{a^2}{4}
\]

and

\[
\|\nabla_V\mathcal L(V_{<k})\|_F^2=b^2\lambda_k.
\]

### Theorem 2 (saddle index identity)

Define

\[
S_k:=\frac{2\Delta\mathcal L_k}
{\|\nabla_V\mathcal L(V_{<k})\|_F^2}.
\]

Then exactly

\[
\boxed{S_k=\lambda_k^{-1}.}
\]

Thus, if $\lambda_k\to0$ while $a$ stays fixed, the lower-frequency solution
has a nonvanishing loss gap but a vanishing gradient. It is a hierarchy of
increasingly stationary approximate saddles.

#### Proof

Let $\Phi_V(x)=\nabla_Vf_V(x)$. The gradient of the omitted-mode loss is

\[
\nabla_V\mathcal L
=-b\int e_{k,c}(x)\Phi_V(x)\,d\mu(x).
\]

Its squared norm is

\[
b^2\iint e_{k,c}(x)K_V(x,x')e_{k,c}(x')
\,d\mu(x)d\mu(x')
=b^2\lambda_k.
\]

Substitution into the definition of $S_k$ proves the claim. $\square$

Inside a fixed ReLU gate cell, $f_V$ is affine in $V$. The local Hessian of
square loss is then the Gauss--Newton matrix, positive semidefinite, with
curvature $\lambda_k$ in the normalized Fourier direction. The plateau is a
flat, slowly descending channel. Gate changes can open new directions and make
the global geometry saddle-like, but negative curvature is not needed for the
waiting-time result.

## 6. Why a ReLU right-factor kernel gives $k^{-2}$

In an isotropic infinite-width approximation, let

\[
C(\delta)=\mathbb E[h_i(x)h_i(x+\delta)],\qquad
q(\delta)=C(\delta)/C(0).
\]

For the outer ReLU gates, two centered unit Gaussian preactivations with
correlation $q$ satisfy

\[
D(q):=\mathbb E[\rho'(G)\rho'(G')]
=\frac{\pi-\arccos q}{2\pi}.
\]

Under the standard channel-decoupling limit, the expected right-factor kernel
has the form

\[
\kappa(\delta)=\gamma C(\delta)D(q(\delta))
\]

for a positive normalization constant $\gamma$.

Assume

\[
q(\delta)=1-\alpha\delta^2+O(|\delta|^3),\qquad \alpha>0.
\]

Since

\[
\arccos(1-u)=\sqrt{2u}+O(u^{3/2}),
\]

the outer derivative kernel has a cusp:

\[
D(q(\delta))
=\frac12-\frac{\sqrt{2\alpha}}{2\pi}|\delta|+O(\delta^2).
\]

Consequently

\[
\kappa(\delta)=\kappa(0)-A|\delta|+O(\delta^2),
\qquad
A=\frac{\gamma C(0)\sqrt{2\alpha}}{2\pi}>0.
\]

### Proposition 3 (ReLU cusp law)

If $\kappa$ is periodic, piecewise $C^2$ away from zero, the displayed cusp
is its only singularity contributing at order $k^{-2}$, and the smooth
remainder has $o(k^{-2})$ Fourier coefficients, then

\[
\boxed{\lambda_k=\frac{A}{\pi k^2}+o(k^{-2}).}
\]

#### Proof

The distributional second derivative of $-A|\delta|$ contains
$-2A\delta_0$. Twice integrating the Fourier coefficient by parts gives

\[
\widehat\kappa(k)
=-\frac{1}{2\pi k^2}\widehat{\kappa''}_{\rm unnormalized}(k)
=\frac{A}{\pi k^2}+o(k^{-2}).
\]

$\square$

Random biases are useful here: they remove antipodal parity degeneracies that
can cancel a leading coefficient for selected even or odd modes in special
bias-free circle embeddings.

Combining Theorem 2 and Proposition 3 gives

\[
\|\nabla_V\mathcal L(V_{<k})\|_F=\Theta(k^{-1}),
\qquad
S_k=\Theta(k^2),
\qquad
\tau_k(\varepsilon)=\Theta(k^2\log(1/\varepsilon)).
\]

This is the promised rigorous hierarchy in the frozen-kernel regime.

## 7. A single multi-cosine target

Let

\[
y(x)=\sum_{j=1}^J a_j\cos(k_jx),
\qquad 1\le k_1<\cdots<k_J.
\]

In exact kernel flow,

\[
\mathcal L(t)=\frac14\sum_{j=1}^J
a_j^2e^{-2\lambda_{k_j}t}
\]

when initialized at zero. A visible staircase requires separation, not merely
ordered eigenvalues. One sufficient condition at tolerance $\varepsilon$ is

\[
\frac{1}{\lambda_{k_j}}
\log\frac{|a_j|}{\varepsilon}
\ll
\frac{1}{\lambda_{k_{j+1}}}
\log\frac{|a_{j+1}|}{\varepsilon}.
\]

With $\lambda_k\asymp k^{-2}$ and comparable amplitudes, geometric frequency
spacing gives the cleanest separated stages. This is why the diagnostic target
uses $k=(1,4,8,16)$.

For a finite feature-learning MMNN, write the exact function-space dynamics as

\[
\partial_t f_t=-T_{K_t}(f_t-y).
\]

If, during phase $j$, the empirical kernel remains approximately stationary
and its high-mode eigenvalues satisfy

\[
c_- k^{-2}\le\lambda_k(K_t)\le c_+ k^{-2},
\]

the same plateau bounds hold with constants $c_-,c_+$. Proving this spectral
stability through gate changes is the main missing step for a full nonlinear
theorem.

## 8. Nonlinear physical picture beyond the proof

The tangent result explains the entrance to each plateau. A complementary
finite-ReLU picture explains why escape can be slower than $k^2$:

1. after low modes are fitted, the residual $\cos(kx)$ oscillates on scale
   $1/k$, so its correlation with broad existing features largely cancels;
2. fitting it requires coordinated gate changes that create or reposition
   $O(k)$ alternating regions;
3. the right-factor bottleneck provides only $r$ independent outer gate
   channels, so many changes cannot be made independently;
4. once enough gates align, the empirical $\lambda_k(K_t)$ increases and the
   trajectory exits the plateau.

This suggests a nonlinear escape law

\[
\tau_k\approx \lambda_k(K_{\rm plateau})^{-1}
\times \mathcal B(k,r),
\]

where $\mathcal B\ge1$ is a gate-reorganization factor. The measured
$2.45$ exponent, compared with the tangent prediction $2$, is consistent
with a slowly growing $\mathcal B$, but the present experiment does not prove
that interpretation.

An algebraic refinement for compositional networks is the harmonic mixing
length

\[
d_{\mathcal A}(k)
=\min\left\{\|n\|_1:
k=\sum_{q\in\mathcal A}n_qq,\ n_q\in\mathbb Z\right\},
\]

where $\mathcal A$ is the set of already active harmonics. Products of
cosines generate sums and differences, so depth can reduce this mixing length.
This may be the correct analogue of “how many new features must be discovered
at once” outside the lazy regime. It is a conjectural extension, not used in
the proof above.

## 9. Relation to Barbier et al. and HCIZ

The paper [Statistical physics of deep learning: Optimal learning of a
multi-layer perceptron near interpolation](https://arxiv.org/abs/2510.24616)
studies a matched Bayesian teacher--student MLP with widths proportional to the
input dimension and $n\asymp d^2$ samples. Its replica potentials describe
static universal and specialized branches through matrix overlap order
parameters. It uses HCIZ/spherical integrals to integrate rotational matrix
degrees of freedom in regimes where quadratic or matrix-product components
matter. The paper explicitly separates these static predictions from a theory
of the training trajectory; its optimization evidence shows that practical
algorithms can remain near suboptimal universal solutions before
specialization.

The conceptual mapping is useful:

- the low-frequency MMNN plateau is analogous to an unspecialized branch;
- activating a new Fourier band is analogous to developing a nonzero
  mode-specific overlap;
- a narrow transition channel resembles a statistical--computational gap.

HCIZ is not needed for the theorem in this note. We condition on the frozen
left factor $U$, train only $V$, and diagonalize a one-dimensional
stationary kernel directly in Fourier space. An HCIZ calculation would become
relevant for a proportional-width thermodynamic theory that averages over the
relative singular-vector orientations of extensive matrices and seeks the
free-entropy barrier between frequency-specialized overlap branches. That is a
substantially different, static calculation and would not by itself prove the
escape-time law.

## 10. Numerical experiment and results

The implementation is
[`experiments/leap_cosine_mmnn/run_experiment.py`](../../experiments/leap_cosine_mmnn/run_experiment.py).
The full configuration is stored in
[`summary.json`](../../experiments/leap_cosine_mmnn/results/summary.json).

The main sweep uses $m=p=192$, rank $r=8$, 512 uniform points on the
torus, Adam with learning rate $2\times10^{-3}$, 20,000 steps, and five
frozen-feature seeds. Only $V$ is trainable.

For the targets

\[
y_k(x)=\cos x+\frac12\cos(kx),
\qquad k\in\{2,4,6,8,12,16\},
\]

define the plateau duration as the number of steps between first reaching 90%
relative accuracy on mode 1 and first reaching 90% on mode $k$. The observed
medians are:

| next frequency $k$ | gap $\Delta k=k-1$ | median plateau steps | seed IQR |
|---:|---:|---:|---:|
| 2 | 1 | 11 | 7--11 |
| 4 | 3 | 147 | 145--152 |
| 6 | 5 | 511 | 447--534 |
| 8 | 7 | 1,127 | 1,039--1,141 |
| 12 | 11 | 3,397 | 3,007--3,403 |
| 16 | 15 | 9,355 | 9,116--9,411 |

All 30 runs crossed the threshold. A log--log fit gives

\[
\tau_{\rm escape}\propto (\Delta k)^{2.454},
\qquad R^2=0.9983.
\]

After separately fitting $\cos x$, the empirical right-factor tangent
spectrum has tail

\[
\lambda_k\propto k^{-2.069},
\qquad R^2=0.9658,
\qquad k\ge8,
\]

which supports the ReLU cusp law. The measured $S_k=1/\lambda_k$ therefore
forms the predicted hierarchy of approximate saddles.

## 11. What is proved, supported, and still open

### Proved under stated assumptions

- exact finite-width factorization of the right-factor tangent kernel;
- mode-wise exponential dynamics for a frozen stationary kernel;
- exact saddle-index identity $S_k=1/\lambda_k$;
- $k^{-2}$ Fourier tail from a unique ReLU gate cusp;
- quadratic growth of plateau time and saddle index in that kernel regime.

### Numerically supported

- the $k^{-2}$ right-factor tangent spectrum after the first mode is fitted;
- sequential recovery of $1,4,8,16$ in one finite MMNN;
- strongly increasing nonlinear plateau duration with the next frequency;
- a finite-network escape exponent slightly larger than the tangent prediction.

### Not yet proved

- that the finite Adam/SGD trajectory stays in a spectrally controlled kernel
  family through every gate transition;
- existence of exact Morse saddles indexed by target frequency;
- a universal exponent $2.45$ (it will depend on optimizer, rank, width,
  activation, biases, and threshold);
- an HCIZ free-energy barrier whose height equals the observed escape time;
- monotonicity in raw gap $\Delta k$ when the previous frequency also varies.

## 12. Falsifiable next tests

1. Replace Adam with small-step full-batch gradient descent and measure time in
   continuous units $t=\eta\times\text{steps}$. The exponent should move
   toward the tangent value $2$ in the lazy regime.
2. Sweep rank. The mean exponent should remain near the ReLU cusp exponent once
   $r$ is large enough, while seed variance and censored modes should grow at
   very small rank.
3. Replace ReLU by a smooth analytic activation. If the stationary kernel is
   analytic, its Fourier spectrum should decay exponentially, producing much
   larger high-frequency plateaux.
4. Remove biases. Parity-dependent cancellations should appear between even
   and odd Fourier modes.
5. Track $K_t$ during each transition. The physical gate-reorganization
   picture predicts a rise in $\lambda_{k_j}(K_t)$ immediately before the
   $j$-th loss drop.
6. For a literal nonnegative factorization, compare projected-gradient KKT
   residuals with the unconstrained saddle index derived here.

These tests distinguish a genuine spectral-saddle mechanism from an optimizer
artifact or a generic low-frequency bias description.
