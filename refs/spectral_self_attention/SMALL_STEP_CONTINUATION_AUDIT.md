# Small-step continuation audit

## Question

The finite transformer block has the serial form

\[
y_i=\operatorname{norm}(x_i+hA_i(X)),\qquad
x_i^+=\operatorname{norm}(y_i+hM(y_i)).
\]

The experiment asks which stable period-3, period-4, and apparently chaotic
attractors are properties of this finite layer map, and which survive when the
layer step tends to zero while the elapsed physical time is held fixed.

The four structural types are:

1. tied symmetric attention and potential MLP;
2. tied symmetric attention and general MLP;
3. untied attention and potential MLP;
4. untied attention and general MLP.

All experiments in this audit use tokens on the circle, one attention head, and
the serial Attention-then-MLP block. The main census uses one to four tokens;
the scaling census adds eight and sixteen tokens.

## Continuous limit

Write \(P_x=I-xx^\top\),

\[
A_i(X)=\sum_j a_{ij}(X)Vx_j,
\qquad
a_{ij}=\frac{\exp(\beta x_i^\top Sx_j)}
{\sum_k\exp(\beta x_i^\top Sx_k)}.
\]

The first-order continuous equation is

\[
\dot x_i=P_{x_i}\{A_i(X)+M(x_i)\}.
\]

The serial and parallel blocks have this same first-order equation. Their
difference starts at order \(h^2\).

Indeed, for a unit vector \(x\), normalization has the elementary expansion

\[
\operatorname{norm}(x+hz)=x+h(I-xx^\top)z+O(h^2).
\]

After attention, \(y=x+hP_xA+O(h^2)\). Smoothness gives
\(M(y)=M(x)+O(h)\) and \(P_y=P_x+O(h)\), so the second normalization adds
\(hP_xM(x)+O(h^2)\). The fact that the MLP is after attention therefore enters
the finite-step correction, but not the first-order continuous vector field.

On the circle the exact scalar velocity makes the observable issue explicit:

\[
\dot\theta_i=\eta\left[
\sum_j \operatorname{softmax}_j(\beta K^S_{ij})R^V_{ij}+m_i
\right],
\]

where \(\eta\) is the unshrunk block scale,
\(t_i=(-\sin\theta_i,\cos\theta_i)\), \(K^S_{ij}=x_i^\top Sx_j\),
\(R^V_{ij}=t_i^\top Vx_j\), and \(m_i=t_i^\top M(x_i)\). The score kernel
\(K^S\) determines routing, but it is not by itself a closed state for a
general value matrix and anisotropic MLP. An exact feature description must
also retain the tangent value responses and the MLP features (for the
quadratic layer, the projections \(u_r^\top x_i\)). The Gram matrix
\(x_i^\top x_j\) is the natural rotation-independent geometry. The measure
tests below therefore report Gram, score kernel, attention weights, and their
joint distribution rather than treating the weights alone as the state.

Because the token state space is compact and all functions used here are
smooth, the remainder and its first derivative are uniformly bounded. A
standard discrete Gronwall estimate therefore gives, for every fixed physical
time \(T\),

\[
\max_{0\le k\le T/h}\|F_h^k(x)-\varphi_{kh}(x)\|=O(h),
\]

where \(\varphi_t\) is the flow of the equation above. This is the precise
reason for holding burn, observation, recurrence, and Lyapunov windows fixed in
normalized time while increasing the number of layers as \(1/h\). The measured
one-step order, the whole-population plateau, the cycle-period law, and the
direct RK4 replay test four different consequences of this same limit.

This statement deliberately fixes \(T\). It does not allow the limits
\(h\to0\) and \(T\to\infty\) to be exchanged automatically. Near a boundary
between two basins, an \(O(h)\) displacement can eventually send two runs to
different attractors even though they agree over every fixed finite window.
That is why the audit separately measures finite-horizon error, long-time
persistence, and sensitivity of basin labels; none of those three can replace
the other two.

Equivalently, the layer map is the time-\(h\) map of a modified vector field
\(f_h=f+h g+O(h^2)\) to the same order. Away from basin boundaries this small
change cannot alter the selected attractor. Near the stable manifold of a
saddle, however, the boundary itself moves by order \(h\), so the final basin
label can jump at a sharply defined step even while the vector fields and all
fixed-time trajectories converge. This is the mechanism tested explicitly
below.

There is also a population prediction. If the basin boundaries are regular
codimension-one surfaces and the initial density is bounded, only an
\(O(h)\)-thick tube around them can change label. Hence

\[
\Pr_{X\sim\rho}\{B_h(X)\ne B_0(X)\}=O(h).
\]

The large random-start partition experiment below tests this statement while
retaining the identities of the tokens in every final cluster.

For a Lipschitz feature map \(g\), the same coupling immediately gives a
Wasserstein statement at fixed time:

\[
W_1\!\left((g\circ F_h^{T/h})_\#\rho,
(g\circ\varphi_T)_\#\rho\right)=O(h)
\]

for any common initial distribution \(\rho\). This does not automatically
imply convergence of one infinitely long empirical trajectory. That stronger
claim needs a unique statistically stable invariant measure; coexisting basins
or several chaotic components can make the limits \(h\to0\) and
\(T\to\infty\) disagree. The experiments therefore perform both tests: shared
random starts at fixed time, and long occupational measures first on one orbit
and then on an ensemble of starts.

For a potential quadratic MLP,

\[
M(x)=b+Lx+\sum_r c_r u_r(u_r^\top x+a_r)^2,
\]

with symmetric \(L\), the tokenwise potential is

\[
\phi(x)=b^\top x+\tfrac12x^\top Lx+
\sum_r\frac{c_r}{3}(u_r^\top x+a_r)^3.
\]

When \(\beta=0\) and \(V=V^\top\), the full global potential is

\[
\Phi(X)=\frac1{2n}\sum_{ij}x_i^\top Vx_j+\sum_i\phi(x_i),
\]

and

\[
\frac{d\Phi}{dt}=\sum_i\|P_{x_i}\nabla_{x_i}\Phi\|^2\ge0.
\]

Consequently this case cannot have an attracting nonconstant cycle or chaos in
continuous time. The analytic potential and its directional derivative are
covered by a regression test.

### Why a fixed three- or four-layer cycle cannot remain the same object

If \(F_h=I+hf+O(h^2)\), then for fixed integer \(p\),

\[
F_h^p(x)=x+phf(x)+O(h^2).
\]

Therefore a bounded sequence of fixed-\(p\) cycles approaching \(h=0\) can only
accumulate where \(f(x)=0\). A genuine continuous periodic orbit can still
survive, but its period measured in layers must grow like \(T/h\), rather than
stay equal to three or four.

More precisely, a stable continuous cycle becomes an attracting closed loop
for a sufficiently small layer step. One layer advances only a fraction
\(h/T\) around that loop. Depending on whether this fraction is nearly rational,
the finite map may close after many layers or only return approximately; an
exact short map period is not expected to survive. The invariant object to
compare is therefore the loop, its physical return time, and its transverse
contraction, not the original label p3 or p4.

Likewise, if \(\lambda\) is a continuous growth or contraction rate, the
corresponding multiplier of one layer satisfies
\(\log|\mu_h|=h\lambda+O(h^2)\). Thus growth measured per layer must vanish,
whereas growth divided by elapsed physical time must converge to \(\lambda\).
This distinguishes a continuous chaotic flow from large-step map chaos without
requiring their trajectories to stay pointwise close for long times.
For chaos, finite-horizon convergence is the rigorous numerical-limit test;
agreement of individual paths forever is neither expected nor required.
Long positive spectra and basin surveys are additional evidence about the
limiting attractor, but they are not a theorem of structural stability for an
arbitrary nonuniformly chaotic set.

### Curl and the topology of the token torus

At \(\beta=0\), the cross-token curl of attention is controlled exactly by
the antisymmetric part of \(V\):

\[
\partial_{\theta_j}f_i-\partial_{\theta_i}f_j
=\frac1n t_i^\top(V-V^\top)t_j.
\]

A curl-free field on a torus need not be a single-valued gradient. Its mean
winding force at \(\beta=0\) is

\[
\frac{V_{21}-V_{12}}{2n}
+\frac{L_{21}-L_{12}}2
+\sum_r a_r\,o_r^\top(-u_{r,2},u_{r,1}).
\]

This term vanishes identically for type 1. It is generically nonzero for a
general MLP even when the local curl is zero.

Row-normalized softmax introduces curl even when \(S=V=V^\top\). One explicit
two-token counterexample uses

\[
V=S=\begin{pmatrix}2&0.4\\0.4&-1\end{pmatrix},\quad
\beta=1.7,\quad(\theta_1,\theta_2)=(0.2,1.4),
\]

for which the two cross derivatives are approximately 0.1746 and 0.0118. At
\(\beta=0\) they agree. This counterexample is also a regression test.

### The exact statistical-physics model at uniform attention

The \(\beta=0\) case is not a vacuous attention limit. Every routing weight is
\(1/n\), but the value matrix still turns the empirical token mean into a force.
Writing

\[
V=\begin{pmatrix}a&b\\c&d\end{pmatrix},
\]

the force exerted by phase \(\theta_j\) on phase \(\theta_i\) is exactly

\[
\begin{aligned}
t_i^\top Vx_j={}&\frac{a+d}{2}\sin(\theta_j-\theta_i)
+\frac{c-b}{2}\cos(\theta_i-\theta_j)\\
&+\frac{d-a}{2}\sin(\theta_i+\theta_j)
+\frac{b+c}{2}\cos(\theta_i+\theta_j).
\end{aligned}
\]

Thus the system is exactly a finite population of globally coupled phase
oscillators with first-harmonic mean-field coupling. The potential quadratic
MLP adds an individual pinning force with Fourier harmonics only up to order
three. The antisymmetric value coefficient \(c-b\) is the nonreciprocal phase
lag. In statistical-physics language, the selected \(\beta=0\), type-3 model is
therefore a nonreciprocally coupled active-rotator model with low-order pinning,
not merely an analogy to one.

This reduction explains why four tokens can already be rich. Symmetric value
transport plus the potential pinning is a gradient system. Antisymmetric value
transport supplies circulation; the pinning supplies wells and separatrices.
Their interaction can produce coexistence between fixed points, cycles, and
chaotic motion even though the attention weights themselves never change.

## Scale and sampling

- Main random cohort: 131,072 independently drawn models and 2,097,152 initial
  trajectories.
- Exact-potential negative control: 16,384 additional type-1, \(\beta=0\)
  models and 262,144 trajectories.
- Total harvest: 147,456 models and 2,359,296 trajectories.
- Each harvested period-3 and period-4 candidate was refined by nonlinear root
  solving and checked through its full cycle Jacobian.
- Main certified candidates: 1,308 screened cycles, of which 1,288 were both
  primitive and stable after refinement.
- Negative-control certified candidates: 108/108 primitive and stable.
- Continued main attractors: 1,908. Continued negative controls: 146.

Two supplemental audits raise the final computational scale to 188,416 models
and 2,818,048 trajectories. The first adds 24,576 models with 8 or 16 tokens;
it finds 581 complex candidates and certifies 284/289 refined short cycles as
primitive and stable. The second adds 16,384 type-3, \(\beta=0\) models for the
value-symmetry intervention described below.

The six-stage fresh unscreened finite-horizon validation adds another 26,112
model-state pairs: 12,288 through \(h/1024\), followed by two independent
3,072-model batches through \(h/4096\) at times 2 and 10, and a final
pair of 3,072-model time-10 batches on independent non-dyadic step grids. A
final 1,536-model batch extends the unscreened horizon to time 20. Counting
those draws, the audit covers 214,528 independently generated models and
2,844,160
model-wide initial trajectories
before the separate multi-start convergence and basin experiments.

Beyond the attractor harvest, fixed-time convergence is checked on 2,802
selected records (1,908 main, 581 high-token, and 313 uniform-attention), plus
40,960 uniform random states across five representative models at two
independent depth ranges. Completely unscreened controls add 26,112 freshly
drawn model-state pairs: 512 pairs in every cell of the four types crossed with
1, 2, 3, 4, 8, and 16 tokens through \(h/1024\), plus 128 new pairs per cell
through \(h/4096\) at each of times 2 and 10, and 128 more per cell on the
each of two independent non-dyadic grids at time 10. A last 64 pairs per cell
reach time 20 at the three deepest dyadic ratios. The
identity-aware basin law uses another 98,304 random four-token starts across
two independent potential landscapes.

The fresh-model finite-horizon runs alone perform 1,071,439,872 discrete
model-layer updates, or 6,071,492,608 updates after weighting by token count.
Their independent RK4 references add 307,200,000 model-field evaluations, or
1,740,800,000 token-field evaluations. Combining their discrete work with the
deep attractor continuations gives 23,789,394,642 model-layer and
102,108,101,290 token-layer updates; the RK4 work remains listed separately
because each RK4 step evaluates the field four times.

The continuation used 17 ratios from \(h/h_0=1\) down to \(1/256\). Every
window is specified in normalized physical time, never in a fixed number of
layers: the burn is 600 and the saved observation window is unchanged, while
the exponent window is 300 in the first pass, 240 at \(1/512\)--\(1/1024\),
and 180 at \(1/2048\)--\(1/4096\). The number of layers therefore grows as
\(1/h\) in every phase. Direct RK4 integration of the limiting equation starts
again from the original finite-map attractor and uses an infinitesimal
perturbation to avoid artificial locking inside an unstable symmetry subspace.

The measured one-step discrepancy between the discrete normalized velocity and
the continuous vector field scales as \(h^{1.00}\) in every structural group,
as required by the first-order derivation.

A separate whole-cohort finite-horizon test starts all 1,908 candidates from
the same state, applies the layer map for two normalized time units, and
compares with an RK4 reference at step 0.001. The median final-state error
scales as \(h^{1.0056}\); the family-specific orders are 1.0115, 0.9894, 1.0182,
and 1.0024 for types 1--4. At \(h/1024\), the median angular RMS error is
\(2.63\times10^{-4}\) and the 90th percentile is \(1.99\times10^{-3}\).
This verifies first-order trajectory convergence over the entire selected
population, not only at one point or on the four exemplars.

A Richardson cancellation gives a more stringent test of the expansion itself.
For every one of the 1,908 records, let \(e_h\) and \(e_{h/2}\) be the signed,
wrapped endpoint-error vectors against the same continuous reference. If
\(e_h=c_1h+c_2h^2+O(h^3)\), then \(2e_{h/2}-e_h=O(h^2)\). Across six nested
step pairs the ordinary median error has order 1.0001, while the cancelled
error has order 2.0179. The type-specific cancelled orders are 1.998, 2.009,
2.016, and 2.027. On the finest pair, the median falls from
\(2.63\times10^{-4}\) to \(3.39\times10^{-7}\), a factor 776. This directly
verifies the regular \(h f+h^2g+\cdots\) expansion rather than only observing a
first-order slope.

The same cancellation is repeated outside the main population. The
Richardson orders are 2.004 for all 581 eight- and sixteen-token records, 1.995
for the 146 uniform-attention potential controls, and 2.035 for the 167
uniform-attention type-3 records. At the finest pair, their median improvement
factors are 787, 691, and 710. Thus every sampled structural and token-count
population exhibits the same regular small-step expansion.

The main cancellation is also repeated at time 10, where sensitive and chaotic
paths have had five times longer to separate. The ordinary and cancelled
orders remain 0.9996 and 2.0040; the four cancelled type orders are 1.990,
2.000, 1.997, and 1.996. At \(h/1024\), the median falls from
\(2.40\times10^{-4}\) to \(2.51\times10^{-7}\), while the 90th percentile falls
from 0.00306 to \(2.04\times10^{-5}\). A few maxima remain large because
Richardson extrapolation cannot cancel trajectories that already selected
different chaotic phases or basins, but the population bulk retains the exact
second-order signature.

The same all-record comparison was then repeated for ten normalized time
units, using the five deepest ratios from \(1/64\) to \(1/1024\). The aggregate
order is 1.0014, and the type-specific orders are 1.0030, 1.0060, 1.0004, and
0.9985. At \(h/1024\), the median error is \(2.40\times10^{-4}\) and the 90th
percentile is \(3.06\times10^{-3}\). The maximum remains much larger because a
small set of sensitive trajectories separates in phase over the longer
window; its error still falls systematically as the step is reduced. At
\(h/1024\), 125/1,908 errors exceed 0.01 radian, only 11 exceed 0.1, and one
exceeds 1.0. At \(h/64\) those counts were 564, 154, and 31, respectively, so
the whole tail contracts along with the median rather than merely hiding a
fixed exceptional population.

The same time-10 experiment covers all 581 selected eight- and sixteen-token
trajectories at \(h/64\), \(h/256\), and \(h/1024\). Its aggregate median order
is 0.9905; the four type-specific orders are 0.9983, 1.0025, 1.0014, and
0.9995. At the deepest step the median error is \(2.37\times10^{-4}\), the 90th
percentile is 0.00723, the maximum is 0.451, and no record remains above one
radian. First-order convergence is therefore just as clear in the
high-dimensional selected cohort.

Uniform attention passes the same whole-cohort test. Across all 146 exact
type-1 potential controls, the time-10 median order is 0.9994; at \(h/1024\)
the median error is \(3.35\times10^{-4}\), the 90th percentile is 0.00131, and
the maximum is 0.00445. Across all 167 \(\beta=0\) type-3 records, the order is
0.9812 and the corresponding deepest values are \(2.99\times10^{-4}\),
0.00220, and 0.126. Thus constant attention weights neither invalidate the
small-step limit nor make the weights a sufficient observable of the token
geometry.

Their asymptotic continuation cleanly separates reciprocity. In the exact
type-1 potential control, the moving count falls from 145/146 at the original
large step to zero by ratio 0.18, then remains exactly zero at all 16 subsequent
ratios through \(h/4096\). In the \(\beta=0\) type-3 intervention it instead
settles to 10 or 11 movers from ratio 0.125 through \(h/4096\). The one-count
oscillation is the known intermittent threshold case. Constant attention
therefore does not determine the fate: symmetric value transport plus a
potential MLP collapses, while nonreciprocal value transport preserves a
small continuous moving population.

At \(h/4096\), all 167 type-3 continuation decisions agree exactly with the
independently restarted anti-lock ODE replay: 157 are fixed in both and 10 move
in both. The separately stability-certified intermittent lock supplies the
eleventh strict mover discussed below.

The finite-horizon law is not an artifact of screening for complex attractors.
A separate test draws 12,288 completely fresh models and one fresh state per
model, with 512 models in each of the 24 type-by-token-count cells. Nothing is
burned in or selected. At time 2, using ratios \(1/64\), \(1/256\), and
\(1/1024\), the angular-error orders for types 1--4 are respectively 1.0019,
0.9999, 1.0012, and 0.9982. The corresponding Gram orders are 1.0057, 1.0012,
1.0087, and 1.0007, while the score-kernel orders are 1.0013, 1.0017, 1.0009,
and 0.9998. Across all 24 cells, including 8 and 16 tokens, every angular order
lies between 0.9958 and 1.0058. At \(h/1024\), the four family median angular
errors are only \(7.50\times10^{-5}\), \(9.44\times10^{-5}\),
\(7.73\times10^{-5}\), and \(9.43\times10^{-5}\). This removes both the
complex-attractor selection bias and the low-token-count bias from the
finite-time convergence claim.

An independent unscreened deep batch then draws 3,072 more models, with 128 in
each of the same 24 cells, and uses ratios \(1/1024\), \(1/2048\), and
\(1/4096\). The type-1 through type-4 angular orders are 0.99989, 0.99998,
0.99998, and 1.00013. Across all 24 individual cells they lie between 0.99920
and 1.00283; the corresponding Gram and score-kernel orders are equally close
to one. At \(h/4096\), the family median angular errors are
\(2.05\times10^{-5}\), \(2.63\times10^{-5}\), \(1.84\times10^{-5}\), and
\(1.94\times10^{-5}\), and every family 99th percentile is below 0.0022.
Selection-independent first-order convergence therefore remains exact at the
deepest tested layer scale, not only at \(h/1024\).

The same deep unscreened design is finally repeated on 3,072 new models at time
10, five times longer than the preceding batch. The four angular orders are
0.99991, 0.99982, 0.99999, and 0.99998, and all 24 cell-specific orders lie
between 0.99939 and 1.00127. At \(h/4096\), the family median angular errors are
between \(1.05\times10^{-5}\) and \(1.69\times10^{-5}\); all 99th percentiles
are below 0.0043. One type-2 trajectory has a 2.07-radian error after selecting
a different sensitive path, but the other 99% and every median shrink at the
predicted rate. Thus neither longer amplification time nor unscreened random
parameters reveal a breakdown of the continuous limit.

To ensure that this result is not helped by repeatedly halving the step, a
fourth independent 3,072-model batch uses the non-dyadic step ratios
\(10/15317\), \(10/30793\), and \(10/50021\) at time 10. The aggregate orders
are 0.99998 for token angles, 0.99999 for the Gram geometry, and 0.99912 for
the score kernel. All 24 cell-specific angular orders lie between 0.99780 and
1.00105; across every nonzero angle, Gram, and kernel median the full range is
0.99766 to 1.00145, and every median decreases at every step. At the deepest
step the four family median angular errors range from \(8.39\times10^{-6}\) to
\(1.44\times10^{-5}\), with every family 99th percentile below 0.0029. The
largest isolated error is 0.136 radians in a sensitive type-3 trajectory, but
its cell median and the whole tail still shrink with first-order slope. This
independent grid rules out dyadic step resonance in the unscreened population,
not only in the four certified cycles.

A fifth, fully independent 3,072-model batch changes both the random seed and
the grid to \(10/12011\), \(10/27179\), and \(10/45007\). Its aggregate angle,
Gram, and score-kernel orders are 0.99994, 0.99956, and 0.99995. The 24
cell-specific angular orders lie between 0.99986 and 1.00036; all 68 nonzero
cell-feature medians decrease at every refinement. Every one of its 136
nontrivial 90th- and 99th-percentile sequences also decreases monotonically.
The deepest aggregate angular median is \(1.34\times10^{-5}\), its 99th
percentile is 0.00182, and its maximum is 0.0653. The result therefore
replicates on new models, new initial states, and a new non-dyadic grid.

The first non-dyadic batch's tail audit gives the same answer. Ignoring the
exactly zero one-token Gram errors, 135 of 136 cell-by-feature 90th- and
99th-percentile sequences decrease
at every refinement. The sole nonmonotone sequence is the type-3, 16-token
score-kernel 99th percentile (0.751, 0.188, 0.202); its fitted order is still
1.17, while the angular 99th percentile in that same cell decreases
monotonically from 0.148 to 0.077 to 0.049. This is the expected amplification
of a few sensitive paths, not a persistent finite-step floor.

Finally, a sixth unscreened batch pushes the physical horizon from 10 to 20 on
1,536 new models, with 64 models in each of the 24 cells and steps
\(1/1024\), \(1/2048\), and \(1/4096\). Its aggregate angle, Gram, and
score-kernel orders are 1.00014, 1.00074, and 0.99951. The 24 cell-specific
angular orders lie between 0.99949 and 1.00037, and all 68 nonzero
cell-feature medians decrease at every refinement. At \(h/4096\), the aggregate
angular median is \(1.17\times10^{-5}\), the 99th percentile is 0.00338, and
the maximum is 0.123. Only two of 136 nontrivial cell-feature tail sequences
are nonmonotone, both 99th percentiles in the same sensitive type-3,
16-token cell; its bulk medians and all angular cell slopes still converge.
Thus doubling the already long random-model horizon does not reveal a
finite-step error floor.

### Uniform random states, not only harvested attractors

The fixed-time check was also run from 4,096 uniform random configurations for
each of five representative models: a reciprocal/potential control, a stable
type-2 cycle, the strong three-token chaotic model, the eight-token
hyperchaotic model, and the intermittent \(\beta=0\) type-3 model. Each state
was compared at time 10 for \(h/64\), \(h/256\), and \(h/1024\). The fitted
median orders are:

| Representative | Token angles | Pairwise Gram | Score kernel |
|---|---:|---:|---:|
| type-1, 4 tokens | 1.001 | 1.032 | 1.015 |
| type-2 stable cycle, 3 tokens | 0.961 | 0.979 | 0.955 |
| type-3, \(\beta=0\), 4 tokens | 0.999 | 0.999 | 0.998 |
| type-4 strong chaos, 3 tokens | 1.022 | 1.018 | 1.016 |
| type-4 hyperchaos, 8 tokens | 1.001 | 0.999 | 0.998 |

An independent second batch of 4,096 starts per model then probes the three
deeper ratios \(1/1024\), \(1/2048\), and \(1/4096\):

| Representative | Angle order | Gram order | Median angle error at \(1/4096\) | 99th percentile |
|---|---:|---:|---:|---:|
| type-1, 4 tokens | 1.000 | 1.003 | 0.000285 | 0.000339 |
| type-2 stable cycle, 3 tokens | 0.994 | 0.996 | 0.000826 | 0.0240 |
| type-3, \(\beta=0\), 4 tokens | 1.000 | 1.000 | 0.0000223 | 0.000305 |
| type-4 strong chaos, 3 tokens | 1.003 | 1.003 | 0.00220 | 0.0326 |
| type-4 hyperchaos, 8 tokens | 1.000 | 1.000 | 0.0000376 | 0.000197 |

The five score-kernel orders in this deepest independent batch lie between
0.993 and 1.002. The type-1 batch still contains one extreme start that chooses
the other valid cluster basin and has 1.52-radian error, but its 99th percentile
is only 0.000339. The isolated basin switch therefore remains visible without
changing the first-order bulk law.

Thus the first-order limit holds both on and far away from the originally
harvested invariant sets, in the raw token coordinates and in the two natural
feature observables. For the eight-token example at \(h/1024\), the 99th
percentile Gram error is only \(5.68\times10^{-4}\). The strong three-token
chaotic model has a larger 99th percentile, 0.108, but its median and tail still
shrink with first-order slope, as expected when finite-time expansion magnifies
the constant multiplying \(h\).

The reciprocal/potential model exposes the complementary basin-boundary
effect. The deepest-step 99th-percentile geometry error is tiny, yet one start
selects a \(3+1\) token cluster while the continuous flow
selects \(2+2\). A dedicated 20-step scan locates the switch sharply: the
finite map still chooses \(3+1\) at ratio \(1/1056\), but agrees with the
continuous \(2+2\) basin at \(1/1088\) and every smaller tested step down to
\(1/8192\). Once across the switch, its angular errors are 0.001071,
0.000569, 0.000285, and 0.000142 at denominators 1088, 2048, 4096, and 8192.
Repeating the decisive denominators out to time 100 leaves the endpoints and
errors unchanged to the reported precision: these are different finite-map
equilibria, not a slow transient at time 10.
From denominator 2048 onward, every halving of the step halves the error. This
is the concrete signature of an order-\(h\) basin
boundary displacement, and explains how two extremely close token states can
ultimately form different clusters without contradicting finite-time
convergence.

The corresponding population test draws 65,536 uniform four-token states and
retains the token identities in the final partition. At denominators 32, 64,
128, 256, 512, and 1024, the numbers whose partition differs from the
continuous flow are 1,156, 579, 289, 137, 85, and 43. The fitted mismatch law
is \(h^{0.946}\), close to the predicted linear tube law. At \(h/1024\), only
0.0656% of starts change partition, and the total-variation distance between
the two full partition distributions is 0.0168%. Meanwhile the median angular
error halves at every step. The dramatic single-start basin switch is therefore
real but occupies a vanishing-probability boundary band; it does not spoil
convergence of the token-allocation distribution.

An independent replication uses a second potential model and 32,768 new
uniform starts. Its mismatch counts are 1,003, 510, 244, 138, 68, and 36 over
the same six denominators, giving exponent 0.958. Its full-distribution
total-variation exponent is also 0.958. The two independent landscapes
therefore give 0.946 and 0.958 for the identity-aware token-partition law,
which rules out a slope produced by the single extreme start or a single set of
weights.

Both landscapes were then extended on the same 32,768 starts to denominators
512, 1024, 2048, and 4096. Their identity-aware mismatch counts are respectively
\((46,24,12,8)\) and \((68,36,14,5)\). At \(h/4096\), only 0.0244% and
0.0153% of starts change partition, while the total-variation distances between
the complete partition distributions are 0.00916% and 0.00610%. Fitting all
eight depths from 32 through 4096 gives mismatch exponents 0.887 and 1.057,
which straddle the predicted linear exponent. The first tail halves exactly
through denominator 2048 and then observes eight cases rather than the six
expected from exact halving; that two-count difference is below one binomial
standard error. The second tail decays slightly faster than linear. Meanwhile
the median angular error halves at every one of the four deep steps in both
landscapes. Thus the boundary tube continues to shrink at the predicted scale
down to a handful of allocations rather than reaching a nonzero floor.

The population count itself also reaches a plateau. Across the 1,908 selected
main candidates, the number still moving is 360 at \(h/4\), 232 at \(h/16\),
214 at \(h/64\), 208 at \(h/256\), 206 at \(h/512\), and 205 at each of
\(h/1024\), \(h/2048\), and \(h/4096\). The deepest finite-step split by type is
0, 66, 35, and 104 movers. Split by the original screening label it is 56 p3,
55 p4, and 94 chaos records: the original short period does not determine the
continuous fate.

The \(h/4096\) endpoint and the independently restarted direct ODE agree on
1,899/1,908 records (99.53%): 202 are moving in both, 1,697 are fixed in both,
and only nine switch class. Thus the 10.7% finite-step plateau is also visible
without relying on the separate ODE code path. The complete main continuation
performed 16,129,424,082 model-layer updates, or 43,890,612,906 token-layer
updates after weighting by token count.

The nine endpoint disagreements are basin choices, not failures of the local
equation. Of the six deepest finite endpoints labelled fixed while the
independent replay moved, five have continuous-field RMS between
\(2.1\times10^{-5}\) and \(4.6\times10^{-4}\) and strictly negative local
growth; the sixth is the type-3 transient already shown to decay under the
20,000-time-unit audit. They are valid continuous equilibria reached from a
different continuation path. Conversely, the three finite endpoints labelled
moving while the independent replay fixed have motion 0.189, 0.664, and 1.812
with continuous fields of comparable size, so they are not residual numerical
drift. This is direct evidence for coexistence of continuous basins and explains
why the agreement plateaus below 100% even after the discretization error has
vanished.

### Numerical symmetry locking

Permutation symmetry creates a second, subtler numerical hazard. If several
tokens become equal to machine precision, deterministic floating-point
integration keeps them exactly equal forever because their subsequent
operations are bitwise identical. The equality subspace may nevertheless be
unstable to a real transverse perturbation. A direct ODE trajectory can
therefore shadow a nonphysical invariant branch even after the finite layer
step has been removed.

The final census uses two independent perturb-and-relax stages. For candidates
with intermittent synchronization, the stricter audit also applies identical
transverse kicks of amplitude \(10^{-12}\) every 50 normalized time units while
leaving the tangent calculation unchanged. This amplitude is many orders below
all reported geometric scales; its sole purpose is to prevent bitwise equality
from becoming an absorbing numerical state. Divergence along saved trajectories
provides an independent check on the sum of the Lyapunov spectrum.

This also fixes the necessary observation time. If a coincident-token branch
has transverse growth rate \(\lambda>0\), a splitting of size \(\delta\) needs

\[
t_{\rm escape}\simeq \lambda^{-1}\log(\varepsilon/\delta)
\]

to reach visible size \(\varepsilon\). With \(\delta=10^{-12}\),
\(\varepsilon=10^{-3}\), the two weak locks found below require about 2,530 and
2,030 normalized time units from their measured rates 0.0082 and 0.0102. A
600-time-unit replay cannot reveal them even though its integration is exact.
The Jacobian census followed by 5,000-time-unit targeted replays is therefore
not optional bookkeeping; it closes a predictable blind spot of finite-window
simulation.

## Stability signatures in the small-step limit

For an equilibrium with continuous Jacobian eigenvalue \(\lambda\), the layer
multiplier satisfies \(\mu(h)=1+h\lambda+O(h^2)\). This organizes all relevant
patterns:

| Continuous object | Exponent pattern | Small-step layer behavior |
|---|---|---|
| attracting equilibrium | all real parts negative | all multipliers approach 1 from inside the unit disk |
| saddle or repeller | at least one positive real part | at least one multiplier approaches 1 from outside |
| stable periodic orbit | one zero, all transverse exponents negative | period in layers grows like \(T/h\) |
| invariant \(k\)-torus | \(k\) zero exponents, remaining transverse exponents nonpositive | quasiperiodic long-layer motion |
| chaotic flow | at least one positive, one zero along the flow, and contraction elsewhere | positive exponent per layer shrinks like \(h\), but per physical time stays positive |

For equilibria, complex conjugate pairs split the first two rows into stable
spirals (negative real part), neutral centers (zero real part), and unstable
spirals (positive real part); repeated zero modes signal a continuous symmetry
or a non-isolated equilibrium manifold and must be quotiented before stability
is inferred. For recurrent motion, two or more positive exponents give
hyperchaos rather than ordinary chaos. These cases exhaust the local
eigenvalue-sign patterns relevant to the sampled smooth finite-dimensional
flows; nonnormal transient growth can amplify disturbances for a while, but it
does not change the asymptotic sign classification.

A potential flow has a symmetric linearization in the appropriate tangent
metric at an equilibrium, hence real local eigenvalues; complex conjugate pairs,
spirals, Hopf cycles, and positive-exponent chaos require the non-potential part.
The experiment certifies the first, third, and fifth rows. It does not yet claim
a robust quasiperiodic torus.

Among the 1,699 stability-audited equilibria reached in the main replay, the
numerical Jacobian found 28 stable spirals: one in type 1, nineteen in type 3,
and eight in type 4.
None occurred in the exact type-1, \(\beta=0\) potential control. The isolated
type-1 spiral is direct evidence that row-softmax destroys the global-gradient
structure even when attention matrices and the MLP are individually symmetric.

The state dimension is the number of circle tokens. A smooth autonomous flow
with one or two such coordinates cannot support a strange chaotic attractor;
three tokens are the first possible dimension. Both fully certified chaotic
examples indeed use exactly three tokens.

## Main direct-ODE census

The counts below are conditional on having first found a finite-map period-3,
period-4, or chaotic candidate. "Rigid" means that all pairwise token angles
stay fixed while the whole configuration turns. "Internal" means that the
relative token geometry changes.

| Type | Replayed attractors | Fixed | Still moving | Rigid | Internal |
|---:|---:|---:|---:|---:|---:|
| 1 | 286 | 286 | 0 | 0 | 0 |
| 2 | 378 | 313 | 65 | 38 | 27 |
| 3 | 768 | 732 | 36 | 21 | 15 |
| 4 | 476 | 368 | 108 | 55 | 53 |
| **Total** | **1,908** | **1,699** | **209** | **114** | **95** |

Thus 89.0% of the selected finite-map complex attractors end at an equilibrium
of the continuous equation. The moving fraction is 11.0%. The rates by
structure are 0% for type 1, 17.2% for type 2, 4.7% for type 3, and 22.7% for
type 4. Structure predicts survival much more
strongly than whether the original screen called the trajectory p3, p4, or
chaotic: the corresponding moving rates are 9.3%, 9.7%, and 13.2%.

As a conservative lower bound on the unscreened random populations, the audit
detects continuous motion in 0/32,768 type-1 models, 63/16,384 type-2,
36/65,536 type-3, and 103/16,384 type-4 after the long stability correction:
202 distinct models among 131,072 (0.154%). These are detection lower bounds,
not estimates of the full prevalence, because only basins first flagged as
finite-map p3, p4, or chaotic were replayed.
The 5,000-time-unit strict replay initially removes two distinct type-3 models.
A targeted continuation to 20,000 time units shows that one of them is
intermittent rather than dead, tightening the final lower bound only to
201/131,072 (0.153%) without changing its interpretation.

The headline count is insensitive to the symmetry-breaking protocol. Raising
the perturbation amplitude from \(10^{-5}\) to \(10^{-3}\), then allowing two
full 200-time-unit relaxations, still gives 208/1,908 movers. Record by record,
1,906/1,908 classifications agree: one slow type-3 case is revealed and one
type-4 case decays. Varying the motion threshold from \(10^{-4}\) to \(10^{-2}\)
changes the raw finite-window count only from 209 to 206. A targeted
5,000-time-unit replay of the closest one-token sub-threshold case reduces its
motion to \(1.88\times10^{-4}\), with a near-zero growth estimate; it does not
enter the headline moving class.

Those finite-window counts contain one machine-locked type-4 equilibrium. Its
Jacobian has a positive rate 0.0082, so a \(10^{-12}\) splitting needs roughly
2,500 time units merely to become macroscopic. A targeted 5,000-time-unit
paired replay releases it into recurrent internal motion (motion 0.0174). The
table applies this stability-audited correction, raising the raw 208 movers to
209. No other nominally fixed main state has a positive Jacobian rate; one has
a very weak but negative rate, and all remaining 1,698 are clearly attracting.
At the original 600-time-unit burn, the paired micro-kick census agrees on all
1,908 classifications; the local spectrum is what identifies the one escape
whose amplification time is longer than that window.

A separate metastability replay then starts from all 208 raw movers and adds
5,000 normalized time units. The provisional strict count becomes 206: all 65
type-2 and all 107 type-4 records persist, while type 3 goes from 36 to 34.
One excluded record is genuinely stationary at \(3.35\times10^{-6}\). The other
falls to \(7.47\times10^{-4}\) at 5,000 but, when continued to time 20,000,
returns to clear internal motion of 0.0129 with a small positive finite-time
growth rate \(7.6\times10^{-4}\). It is therefore an intermittent mover with an
exceptionally long quiet episode, not a decaying equilibrium. Restoring it and
the independently certified unstable lock gives the preferred very-long-time
count 208/1,908, split into 114 rigid motions and 94 internal motions. Thus the
short table reports 209 and the deepest table 208; the single loss is the
genuinely stationary type-3 transient.

Repeating the full 5,000-time-unit replay with no anti-lock perturbation gives
the same moving/fixed decision on all 208 starting movers. Thus the small-token
long-time count is not created by the micro-kick protocol.

Halving the RK4 step from 0.02 to 0.01 over the entire 1,908-record census,
while drawing an independent perturbation replica, gives 209 movers. The two
integrations agree on 1,907/1,908 classifications (99.95%); the only change is
one additional slow type-3 mover. This makes the population result insensitive
to both the integrator step and the perturbation replica.

In the stricter paired comparison, the step is again halved but every initial
condition and every symmetry-breaking perturbation is held identical. All
1,908/1,908 classifications agree exactly: the family counts remain
0, 65, 36, and 107 movers. This isolates the integration step from basin
selection and rules out an RK4-step explanation for the census.

Reducing the RK4 step fourfold again, from 0.02 to 0.005, also preserves all
1,908/1,908 classifications and the same family counts. Among the 1,700 raw
fixed endpoints the median and 90th-percentile motion differences are exactly
zero, and even the maximum is \(5.7\times10^{-6}\). Chaotic trajectories need
not remain phase-aligned, so their pointwise motion estimates differ more, but
none crosses the fixed/moving boundary. The result has therefore converged at
the population level over a factor-four integrator refinement.

The exact-potential negative control gives 146/146 equilibria in direct
continuous integration.

### Eight and sixteen tokens

The exact finite-step continuation of all 581 selected candidates reaches four
new ratios beyond the original audit. Its moving count is 120 at \(h/256\),
120 at \(h/512\), 120 at \(h/1024\), 119 at \(h/2048\), and 118 at
\(h/4096\). Thus the high-token population is already on the same roughly
120-record plateau seen by the independent continuous replay, rather than
continuing to collapse with the step.

Four nominally fixed \(h/4096\) endpoints have positive measured growth rates:
two type-3 locks at 0.0264 and 0.0630, and two type-4 locks at 0.0134 and 5.03.
They have motion below the cutoff only because their coincident-token branches
have not yet escaped during the finite observation window. Applying exactly the
same stability rule used in the continuous census promotes those four locks:

| Type | Continued | Raw moving at \(h/4096\) | Unstable locks | Stability-adjusted | Strict ODE |
|---:|---:|---:|---:|---:|---:|
| 1 | 61 | 0 | 0 | 0 | 0 |
| 2 | 143 | 38 | 0 | 38 | 39 |
| 3 | 152 | 11 | 2 | 13 | 14 |
| 4 | 225 | 69 | 2 | 71 | 69 |
| **Total** | **581** | **118** | **4** | **122** | **122** |

After also applying the already documented two long-time ODE promotions and
one ODE decay, the stability-adjusted finite and continuous decisions agree on
575/581 records (98.97%): 456 are fixed in both, 119 move in both, and only
three differ in each direction. The three deepest finite fixed endpoints that
reach moving ODE basins have small continuous fields and negative local growth;
the three deepest finite movers whose independent ODE restart reaches a fixed
point have motion 0.0189, 0.0919, and 0.252 and the ODE equilibria are locally
stable. These six disagreements are therefore coexisting basin choices, just
like the nine small-token cases, not missing terms in the limiting equation.

The final four-ratio extension alone performs 6,588,530,688 model-layer and
52,145,995,776 token-layer updates across the high-token and uniform-attention
cohorts. Together with the main continuation, the deepest audit performs
22,717,954,770 model-layer or 96,036,608,682 token-layer updates.

The strict supplemental high-token replay contains the same 581 candidates:

| Type | Replayed | Still moving | Internal |
|---:|---:|---:|---:|
| 1 | 61 | 0 | 0 |
| 2 | 143 | 39 | 26 |
| 3 | 152 | 14 | 11 |
| 4 | 225 | 69 | 55 |
| **Total** | **581** | **122** | **92** |

The moving rates are almost identical at 8 tokens (83/397, 20.9%) and 16
tokens (39/184, 21.2%). The rise relative to the one-to-four-token selected
sample is therefore not a one-off at a single token count. It is also not just
collective rotation: 92/122 survivors change their pairwise geometry.

These final counts use a paired anti-lock replay: the ordinary run and the
micro-kick run share exactly the same initial and relaxation perturbations.
They agree on 580/581 candidates. The sole change is a 16-token type-4 state
that looked fixed at machine precision but becomes strongly internally chaotic
(motion 0.125, largest finite-time exponent 0.355) after \(10^{-12}\) transverse
kicks. The equilibrium-spectrum audit finds one further weakly unstable
eight-token lock (positive rate 0.0102). A 5,000-time-unit paired replay
releases it into recurrent internal motion. The table includes both
stability-audited corrections; all other nominally fixed high-token endpoints
have negative largest real part.

One additional 16-token record lay just below the fixed motion cutoff after the
short replay (0.00094). After 5,000 more time units it remains recurrent and
rises to 0.00159, with changing pairwise geometry and a near-zero largest
growth rate. Conversely, the 5,000-time-unit replay of all 121 original movers
finds one different 16-token type-4 record collapsing completely to a fixed
point: its motion falls from 0.0732 to \(1.23\times10^{-14}\). The two changes
cancel in the total. The table is the final strict count: 120 long-lived raw
movers plus the weakly unstable eight-token lock and the slow sixteen-token
recurrent state.

Both weak survivors were then continued to time 20,000. The eight-token lock
remains internally recurrent with motion 0.00947 and Gram variation 0.753. The
sixteen-token threshold case remains above the cutoff at 0.00181, with Gram
variation 0.00938. Hence neither correction is a 5,000-time-unit transient and
the strict 122/581 count is unchanged.

The corresponding no-kick replay reports only 117 raw movers, but the three
extra zero-motion endpoints are not attracting equilibria: their largest
growth rates are respectively 0.0285, 0.0479, and 4.77. They are exact
token-equality locks in floating-point arithmetic. With the transverse
micro-kicks, those same type-3 eight-token, type-3 sixteen-token, and type-4
sixteen-token records show internal motion 0.00868, 0.0113, and 0.112. This
paired result makes the role of the perturbation diagnostic rather than
creative: it only releases states whose own linearization says that an
arbitrarily small unequal-token perturbation must grow.

These are conditional rates after screening for a complex finite-map fate. As
a deliberately conservative population lower bound, the harvest directly
detected continuous motion in 115 distinct models among 24,576 random models
(0.47%). The paired and long-time corrections raise this to 117 models, without
changing the rounded fraction. The search was not designed to estimate all basins of every model, so
this lower bound must not be read as a complete prevalence estimate.

There is a simple exact explanation for repeated exponents in clustered
high-token states. With shared parameters and no token labels in the equations,
permuting tokens leaves the vector field unchanged. If \(m\) tokens coincide,
they remain coincident; the equality subspace is invariant. Its linearization
has one collective cluster direction and \(m-1\) equivalent ways to split the
cluster. Those transverse exponents must therefore occur with multiplicity
\(m-1\) for a persistent exact cluster. Intermittent joining and leaving
broadens this ideal degeneracy, but repeated exponent groups still diagnose
the cluster structure.

### Type-3 causal split

Type 3 keeps the MLP potential while varying the untied score and value
matrices. For the p3 and p4 cohorts combined, no moving continuous limit was
found among 238 records with symmetric \(V\), whereas 20 of 274 records with
general \(V\) kept moving. The one-sided Fisher exact probability is
\(2.7\times10^{-6}\). Including the chaos-screened cohort and using the fully
relaxed perturbations gives two movers among 367 symmetric-\(V\) records versus
34 among 401 general-\(V\) records (one-sided Fisher probability
\(2.1\times10^{-8}\)). In the same records, symmetric versus general score
matrix gives 16/386 versus 20/382 movers (two-sided probability 0.50). Thus
value symmetry, not score symmetry, is the
empirically decisive split in this controlled type-3 family.

This is not a proof that symmetric \(V\) forbids motion at positive \(\beta\):
softmax still has curl. It is a strong empirical separation in this sampled
family.

An additional \(\beta=0\) type-3 control removes that caveat. It harvested 167
finite-map complex candidates. Stability-audited ODE replay found 0 movers
among 79 records with symmetric \(V\), versus 11 among 88 with general \(V\)
(one-sided Fisher probability \(6.3\times10^{-4}\)). The eleventh is the
intermittently synchronized attractor below: its unperturbed trajectory can
lock on a transversely unstable equality branch, but three anti-lock amplitudes
recover positive-exponent motion. At uniform attention the score matrix is
irrelevant, so this is a direct intervention on value reciprocity: symmetric
value transport plus the potential MLP is a global gradient; its antisymmetric
part is the only source of cross-token circulation.

## Certified continuous attractors

### Stable internal cycles

Four representative internal cycles were rerun from every phase of their
original finite cycle and four independent perturbations per phase:

- type-2 p3: 12/12 runs returned to internal periodic motion;
- type-2 p4: 16/16;
- type-4 p3: 12/12;
- type-4 p4: 16/16.

Their largest exponent is numerically zero, while transverse exponents are
negative. Independent 3,000-time-unit spectra for the other three examples are

\[
\begin{array}{ll}
\text{type-2 p3}:&(-0.00031,-0.01837,-0.29066),\\
\text{type-2 p4}:&(-0.00032,-0.01693,-0.04067,-0.04099),\\
\text{type-4 p4}:&(-0.00009,-0.00249,-0.00332,-0.19532).
\end{array}
\]

Each has exactly one neutral direction rather than two, ruling out a
quasiperiodic two-frequency torus for these representatives. The type-2 p3
example illustrates why the full-state check matters: its relative geometry
repeats every 16.67 time units while the whole configuration rotates at about
0.377 radians per unit time; the angles return only after two shape cycles,
near 33.3. It is a rotating periodic orbit, not an unclosed drift.

For the selected type-4 p3 cycle the full spectrum is approximately

\[
(-0.0002,-0.0952,-1.0452),
\]

where the first number is the neutral direction along the orbit. Its observed
period is about 9.4 normalized time units in the deeply continued basin.

The finite-layer period supplies a direct scaling check. After an explicit
\(10^{-3}\) initial perturbation removes the exact equality lock, it grows from
292 layers at \(h/32\), to 592 at \(h/64\), 2,394 at \(h/256\), 9,618 at
\(h/1024\), and 38,458 at \(h/4096\). The deepest two points have identical
physical return time, 9.389, so their layer-period exponent is exactly one to
the displayed precision; the Gram-period fit over all nine steps is 1.009.
An RK4 trajectory restarted from that basin gives 9.400, a 0.12% difference.
At the coarsest checked step, \(h/16\), the full angles return only after two
relative-geometry cycles, illustrating finite-step phase locking rather than a
failure of the limit. A fixed three- or four-layer cycle would instead have a
physical period that vanished with the step. This observed \(1/h\) growth is
therefore a direct certificate of a genuine continuous cycle.

The same test on the other three certified cycles gives the same asymptotic
law. At the deepest checked step, their physical return times versus RK4 are
16.632 versus 16.660 for type-2 p3, 9.331 versus 9.340 for type-2 p4, and
6.4395 versus 6.4400 for type-4 p4. The relative discrepancies are 0.17%,
0.096%, and 0.0085%. Some coarser steps select a doubled or tripled return;
the deepest pair in every case settles on the continuous branch while its
layer count doubles when the step halves.

As a separate resonance control, all four cycles were rerun at the non-dyadic
steps (h/1531), (h/3079), and (h/5003). Their fitted layer-period
exponents are 1.0016, 1.0029, 0.9992, and 0.9950 for type-2 p3, type-2 p4,
type-4 p3, and type-4 p4. The deepest physical periods differ from RK4 by
0.030%, 0.031%, 0.134%, and 0.312%, respectively, using the
rotation-independent Gram return for type-2 p4. The latter has a longer
labeled-angle return because the same relative shape can recur after a token
permutation or global rotation. Thus the (1/h) law is not an accidental
commensurability with powers of two.

For that fully untied type-4 example, attention alone still produces an
internal stable cycle, whereas the MLP alone converges to a fixed point. Thus a
general untied attention head can sustain a continuous cycle without help from
the MLP; the full block changes, but does not create, that example's basic
periodic mechanism. Its attention-only spectrum is approximately
\((-0.0027,-0.4568,-0.4594)\).

Two stable examples have \(\beta=0\), so their attention weights are identically
uniform while the score kernel and token geometry continue to oscillate. This
is a direct counterexample to treating the attention weights alone as a closed
observable for the full Attention-MLP block.

### Uniform-attention intermittent chaos and coexistence

One type-3, four-token, \(\beta=0\) model combines a potential MLP with a
general value matrix. Its attention weights are exactly \(1/4\) at every time,
yet the anti-lock spectrum has a positive largest exponent. With transverse
kicks of \(10^{-12}\) every 50 time units, the 10,000-time-unit spectrum is

\[
(0.0167,-0.0274,-0.0578,-0.1367),
\]

and kicks of \(10^{-10}\) give
\((0.0116,-0.0265,-0.0606,-0.1413)\). The kicked pseudo-flow is weakly
nonautonomous, so it need not retain an exact zero flow exponent; the robust
claim is the positive first exponent and negative total sum at both amplitudes.
Reducing the kick by another factor of 100 to \(10^{-14}\), and applying it only
every 100 time units, still gives a positive leading exponent, 0.0188, with
negative total sum, -0.2382.
Halving the RK4 step from 0.02 to 0.01 gives the same leading value (0.0169).
The direct mean divergence on 25,000 matching saved states is -0.2188 versus an
exponent sum of -0.2053.

The motion switches between clustered and unclustered phases. At tolerance
\(10^{-6}\), 50.7% of saved states have a \(3+1\) token partition, 43.3% have
four distinct tokens, and 6.0% have \(2+1+1\). All three relative-angle
covariance directions have substantial variance, so this is not a one-curve
cycle masquerading as chaos.

The source of circulation is causal. Replacing \(V\) by its symmetric part,
removing attention, or removing the MLP makes this selected system converge to
a fixed point. The isolated antisymmetric part of \(V\) still produces
positive-exponent internal motion. Thus nonreciprocal value transport supplies
the circulation; the potential MLP and symmetric transport reshape its basins.

A 512-start global survey shows coexistence for the full model: 100 starts end
fixed, 206 pass the strict positive-exponent criterion, 88 reach recurrent
internal motion, 103 remain internally moving but unresolved on the finite
window, and 15 are slow or unresolved. The same weights therefore support
qualitatively different long-run states selected by initialization.

The paired anti-lock survey keeps all initial and finite relaxation
perturbations identical and adds only the \(10^{-12}\) periodic splitting. It
still shows coexistence, but reallocates the 512 starts to 147 fixed, 110 strict
positive-exponent, 97 recurrent, 135 internally unresolved, and 23 slow. Only
54 starts retain the strict-chaos label in both protocols. This sensitivity is
not evidence that the certified attractor disappears: its three kick
amplitudes and long spectra remain positive. It shows instead that the basins
around the intermittent equality manifolds are extremely interwoven, so a
single finite-time basin percentage is not an intrinsic constant of this
model.

The ten ordinary \(\beta=0\), type-3 movers were also continued for 5,000
additional time units. All ten remain moving; three retain internal shape
motion. Repeating the continuation with no periodic micro-kick gives exactly
the same 10/10 and 3/10 counts, and the individual motion amplitudes agree to
the displayed numerical precision. Their persistence is therefore not being
maintained by the anti-lock intervention.

### A robust continuous chaotic attractor

One type-4, three-token model was originally a stable, synchronized period-3
map orbit. As the step is reduced, transverse symmetry breaks near
\(h/h_0\approx0.35\); a regular internal oscillation appears and becomes chaotic
near \(h/h_0\approx0.09\).

A 61-point downward/upward sweep refines this route. On the downward path the
internal cycle first appears at \(h/h_0=0.428\), and its largest exponent turns
strongly positive at 0.0896. On the upward path regular motion is recovered
between 0.102 and 0.109, and the synchronized/fixed branch returns at 0.457.
Thus the cycle-to-chaos change has a narrow hysteretic or coexistence window of
roughly 0.09--0.11; it is not a numerical jump caused by the coarse 17-point
grid.

Deep in the small-step regime, its finite-map exponent per normalized time is
0.157 at \(h/256\), 0.154 at \(h/512\), and 0.134 at \(h/1024\), while the
exponent contributed by one layer falls from \(6.14\times10^{-4}\) to
\(3.01\times10^{-4}\) and \(1.31\times10^{-4}\). Thus individual layers become
arbitrarily close to the identity, but expansion over fixed physical time
remains nonzero, exactly as a genuine continuous chaotic flow requires.

The direct continuous equation gives:

- 24/24 positive-exponent outcomes across all three original cycle phases and
  eight perturbations per phase;
- largest exponent range 0.126--0.175 in that robustness test;
- independent runs at RK4 steps 0.02, 0.01, and 0.005 agree;
- at step 0.005 the largest exponent is 0.1584;
- the fully relaxed 3,000-time-unit spectrum is
  \((0.1489,0.0007,-0.3923)\), with the middle value converging to the neutral
  flow direction.

The corresponding trajectory fills a two-dimensional region in relative-angle
space rather than a closed curve. Its pairwise geometry and attention weights
both vary strongly and fail the recurrence tests.

An independent volume-contraction check agrees with the exponent calculation.
The mean divergence measured directly along 20,000 saved states is -0.24289,
while the sum of the three independently computed exponents is -0.24265. Their
0.10% discrepancy is a stringent Liouville consistency check. Although the
trajectory locally expands volume in 51.9% of sampled states, the long-run mean
contracts, as a bounded attracting chaotic set requires.

The model has symmetric score matrix, general value matrix, general MLP, and
positive softmax temperature. Ablation identifies a genuine interaction:

- attention alone converges to a fixed aligned state;
- MLP alone produces only synchronized regular rotation;
- attention plus MLP produces the chaotic attractor.

The basin is not a small remnant of the original period-3 orbit. In a separate
global survey, all 512 initial configurations drawn uniformly over the full
three-token torus converged to internal positive-exponent motion. Their largest
finite-time exponents range from 0.105 to 0.199.

A second, weaker type-2 chaotic candidate has positive largest-exponent
estimates in all eight perturbation runs (0.028--0.043; seven exceed the
pre-registered 0.03 threshold). Its independent, fully relaxed 3,000-time-unit
spectrum is \((0.0333,-0.0011,-0.7253)\), so it also passes the
full-spectrum test, albeit with substantially weaker expansion. In this tied,
symmetric-attention example, attention alone and MLP alone each converge to a
fixed point; chaos exists only in their serial interaction.

Its basin is also broad: all 512 uniform random starts remained in internal
motion; 449/512 exceeded the strict 0.03 chaos threshold and the remaining 63
still had positive estimates between 0.022 and 0.03. The finite-time threshold,
not loss of the moving attractor, accounts for the split.

### An eight-token hyperchaotic chimera

A type-4 eight-token survivor has three positive directions rather than one.
Its 10,000-time-unit spectrum at RK4 step 0.02 is

\[
(0.1458,0.1455,0.1435,0.0031,-0.0409,-0.1871,-0.1872,-0.1896).
\]

The near-zero fourth exponent is the flow direction, and the total sum is
-0.1669. An independent step-0.01, 5,000-time-unit replica also has three
positive exponents (0.1114, 0.1095, 0.1056) and a negative total sum (-0.1753).
This is therefore hyperchaos in the literal dynamical-systems sense: more than
one independent perturbation direction grows.

The anti-lock control preserves the conclusion. With \(10^{-12}\) kicks every
50 time units, another 10,000-time-unit run gives three positive exponents
(0.0622, 0.0549, 0.0479) and total sum -0.1699. The three expanding directions
are therefore not an artifact of bitwise synchronized tokens.

The direct mean divergence over a matching 25,000-state trace is -0.1635, only
0.0035 away from the exponent sum. The trajectory is a cluster chimera. At
tolerance \(10^{-4}\), every saved state contains a synchronized group of four;
92.3% have a \(4+4\) partition and 7.7% a \(4+3+1\) partition. At the stricter
\(10^{-6}\) tolerance, the corresponding fractions are 90.4%, 8.7%, with 0.9%
in a \(3+3+1+1\) transition. Thus tokens repeatedly condense into synchronized
groups while the cluster phases themselves evolve chaotically.

The attention weights vary only weakly (maximum temporal standard deviation
0.0034) despite substantial pairwise-geometry motion (Gram standard deviation
0.481). This is another concrete failure of attention weights alone as a state
observable.

All 256 uniformly random initial configurations in a global basin survey
converged to internal moving dynamics. On the shorter 600-time-unit exponent
window, 67 already exceeded 0.03 and the remaining 189 had positive estimates
between 0.003 and 0.03. The long spectra above show that short windows
substantially understate the eventual expansion rate in this intermittent
cluster regime.

The paired anti-lock survey again leaves all 256 starts internally moving: 63
cross the strict 0.03 cutoff, one is classified recurrent, and 192 remain
internally unresolved on the finite exponent window. Hence the broad basin of
motion is stable even though the short-window chaos label is not.

### Long-run observable measures

For long chaotic runs, pointwise token agreement is neither expected nor the
right target. After a 1,000-time-unit burn, the audit saved 4,096 samples and
compared the empirical distributions of the Gram entries, score-kernel
entries, attention weights, coherence, and a 64-projection sliced-Wasserstein
distance on their joint standardized feature vector. A split between the first
and second half of the continuous run supplies the finite-sample noise floor.

For the type-2 stable cycle, the joint distance falls from 0.0878 at \(h/64\)
to 0.00475 at \(h/1024\), below the continuous split-half baseline 0.00680. For
the strong three-token chaotic example, the deepest joint distance is 0.0994
against a continuous baseline 0.0974; its Gram and score-kernel marginals are
also at or below that baseline. For the intermittent \(\beta=0\) example, all
steps from \(h/128\) onward are well inside the continuous sampling floor. Its
attention-weight distance is exactly zero at every step even when its Gram
distance is nonzero, because every row remains identically uniform. This is a
direct numerical demonstration that attention weights alone discard dynamical
information.

The eight-token hyperchaotic example exposes the infinite-time caveat instead
of giving a monotone one-orbit curve. Its individual marginal distances are of
the same order as the continuous sampling noise, but the joint standardized
distance grows at the deepest two steps, showing that the single trajectory
visits a different correlated component. This does not conflict with the clean
fixed-time order-one convergence above. It means that one occupational orbit
cannot distinguish a discretization error from a step-dependent basin choice;
the ensemble audit in the next paragraph is the appropriate comparison.

That ensemble audit uses 128 shared uniform starts, a 1,000-time-unit burn, and
512 saved states per start. It removes the one-orbit ambiguity in all three
hard examples. At \(h/1024\), the joint distance versus the split-ensemble
continuous baseline is 0.00582 versus 0.01005 for the strong three-token
chaos, 0.00822 versus 0.08416 for the intermittent uniform-attention model,
and 0.02392 versus 0.05381 for the eight-token hyperchaos. Thus every deepest
joint distance lies below its own continuous finite-sample noise floor. In the
eight-token case this remains true at all three tested steps, even though the
single-orbit joint curve was nonmonotone. The map and the flow therefore sample
the same ensemble-level observable law; the apparent discrepancy was selection
of a correlated component by one chaotic orbit, not failure of the small-step
limit.

## What is and is not observable from the attention kernel

The score matrix \(K_{ij}=x_i^\top Sx_j\) and the row-softmax weights are useful
because they directly expose routing and token geometry. They cleanly separate
rigid collective rotation from internal rearrangement in many examples.

They are not a closed state description of the serial block in general. The MLP
also sees

\[
z_{ir}=u_r^\top x_i+a_r,
\]

as well as linear projections of each token. Uniform attention can coexist with
a nontrivial MLP-driven cycle. A sufficient feature-level observable therefore
has to include, at minimum, the attention score/weight matrices and the MLP
hidden features (plus the value and linear projected features needed by the
outputs). The attention kernel alone is sufficient only under additional
symmetry or injectivity assumptions.

## Interpretation relative to the continuous consensus literature

The large-step stable polygons and short cycles in the tied/potential type do
not contradict continuous-time instability results: in this audit all selected
type-1 p3, p4, and chaotic candidates collapse to equilibria. They are stable
orbits of the finite layer map, not nonconstant attractors of its ODE limit.

What lies beyond that setting is the serial MLP and the breaking of reciprocal
value interactions. A general MLP can sustain rotations and stable internal
cycles even with uniform, symmetric attention. General value transport plus a
general MLP can create a robust continuous chaotic attractor even when the
score matrix itself is symmetric.

## Conclusion in plain language

Making each transformer layer weaker does not simply erase everything that
looked complicated. It separates two very different phenomena.

First, the spectacular three- and four-layer loops found in the fully
reciprocal, energy-like model are large-step tricks. When one uses more and
weaker layers to represent the same elapsed time, every one of them settles
down. Their small polygons are not lasting continuous motions.

Second, once either the MLP or the transport between tokens can push in a
nonreciprocal way, some motions remain. Their loops take more layers as each
layer becomes weaker, so the time needed to go around stays constant. Some
systems keep changing shape forever; a few are stably periodic and others are
genuinely unpredictable. This persists with 3, 4, 8, and 16 tokens and even
when all attention weights are uniform.

This is not a consequence of first choosing unusually favorable cycles. In
26,112 fresh, unscreened models across six independent depth/time/step grids,
every one of the 24 type-by-token-count groups approaches the same continuous
motion: dividing the strength of a layer divides the typical error by the same
factor. This includes independent time-2 and time-10 batches down to
\(h/4096\), two separate non-dyadic time-10 replications, and a time-20
replication. In the selected eight- and sixteen-token population, the
stability-adjusted deepest layer map
and the strict continuous replay both retain 122 movers; they agree record by
record in 98.97% of cases, with the remaining six selecting different valid
basins.

Two almost identical starts can still end in different token groups. The
reason is not that the weak-layer limit fails: they lie on opposite sides of a
very thin frontier between destinations. The width and probability of this
frontier shrink almost exactly in proportion to the layer strength. Most
starts converge normally, while a vanishing fraction chooses another valid
destination.

Finally, attention weights alone do not tell us what the tokens are doing. In
the uniform-attention examples the weights never change at all, while token
geometry can cycle or become chaotic. To observe the state, one must at least
keep the token-to-token geometry, the value-transformed directions, and what
the MLP sees.

This is a large numerical audit, not a proof covering every possible matrix or
every number of tokens. What it rules out is the simple numerical explanation
for the sampled phenomena: the surviving cycles and chaos do not disappear
when layers are made thousands of times weaker, the physical time is held
fixed, the integrator is refined, or the depth grid and random seed are
changed.

## Reproducibility map

- Cohort harvest: `small_step_cohort_harvest.py`
- Cycle certification: `certify_small_step_cohorts.py`
- Small-step continuation: `small_step_continuation.py`
- Deep extension to 1/4096: `extend_small_step_continuation.py`
- Whole-cohort finite-horizon convergence: `finite_horizon_convergence.py`
- Leading-error cancellation: `richardson_finite_horizon.py`
- Fresh unscreened random-model convergence: `random_model_finite_horizon.py`
- Uniform random-state convergence: `random_state_finite_horizon.py`
- Basin-boundary step scan: `basin_boundary_step_scan.py`
- Random-start basin partition scaling: `basin_partition_mismatch_scaling.py`
- Direct continuous integration: `continuous_ode_audit.py`
- Cycle period-versus-step scaling: `cycle_period_step_scaling.py`
- Long-time metastability replay: `continuous_long_time_replay.py`
- Observable-measure convergence: `observable_measure_convergence.py`
- Ensemble observable-measure convergence: `ensemble_observable_measure_convergence.py`
- Curl and torus-winding census: `continuous_curl_census.py`
- Phase/noise robustness: `continuous_ode_robustness.py`
- Full exponent spectrum: `continuous_lyapunov_spectrum.py`
- Saved continuous trajectories: `continuous_attractor_trace.py`
- Consolidated machine-readable summaries: `data/spectral_self_attention/`

The focused regression suite checks the map classifier, serial/batched
agreement, the continuous limit, RK4 consistency, the exact potential,
the torus harmonic formula, and the explicit softmax-curl counterexample.
