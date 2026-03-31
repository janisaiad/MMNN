We thank the reviewer for the careful reading and thoughtful questions.

**Q1 (Technical novelty).** We apologize for not being precise enough in the coupling argument presentation: one intermediate step was missing in the draft wording.

More precisely, relative to Nguyen et al., we do not claim technical novelty for the existence/uniqueness line of argument; we include those steps for completeness and to check that the $r$ coupled mean-field equations are well-posed. That block is ancillary to the paper’s main message.

The substantive point is conceptual. Under convergence-type assumptions, the $r$ channel trajectories could *a priori* lack spatial expressivity—independent paths might learn maps that poorly target localized input structure. Empirically, such effects can appear without mechanisms that diversify features across channels (see Fig. 8, fourth panel). We argue this is less of a limitation in our RF-LR setting because the **frozen** random first-layer features $W^0$ provide the spatial diversity needed for channel-resolved adaptation.

**Technical novelty (coupling).** Our modification is to adapt their coupling template to the **channel-wise sum** (their three-layer assumption / coupling display). After introducing the coupling $\pi\_t$ and using bounded first-layer activations, one obtains a bound of the form

$$
\mathbb{E}\_{\pi\_t}\left[\left|\left\langle \mathbb{E}\_{Z}\left[ \Delta\_{2}^{H*}(t,\cdot)-\Delta\_{2}^{H*}(\cdot;\bar w)\mid X=x \right],\varphi\_{1}(\langle u,x\rangle) \right\rangle\right|\right]
\le K \mathbb{E}\_{\pi\_t}\left[\left|\Delta\_{2}^{H*}(t,\cdot)-\Delta\_{2}^{H*}(\cdot;\bar w)\right|\right],
$$

and one then controls the right-hand side by weighted coupled gaps (same logic as Nguyen & Pham), with low-rank-specific channel aggregation.

Our low-rank translation of this missing step is that the main coupling term is controlled through a channel sum under $\pi\_t$:

$$
\mathbb{E}\_{\pi\_t}\Bigl[(1+|\bar w\_{2}|)\ |\bar w\_{2}|\ \sum\_{k=1}^{r} |\bar w\_{1,k}|\ |w\_{1,k}-\bar w\_{1,k}|\Bigr]\to 0,
$$

with notation matched to our neuronal embedding variables in Eq. (1). This is exactly the place where the RF-LR structure replaces the fully connected contraction chain by channelwise control plus mixing bounds.

We will make this explicit in the appendix by adding: (i) the post-translates-to display; (ii) the time-regularity condition for each channel,

$$
\operatorname{ess\,sup}\_{t}\left|\partial\_t w\_1(t,U\_k)\right|\to 0,\qquad k=1,\dots,r;
$$

(iii) the bi-Lipschitz/bounded-mixing constants used to close the backward estimate.

For ReLU, the closure step uses the sharp inequality

$$
\left|\operatorname{ReLU}\left(\sum\_{i=1}^{r}x\_i\right)-\operatorname{ReLU}\left(\sum\_{i=1}^{r}y\_i\right)\right|
\le \left|\sum\_{i=1}^{r}(x\_i-y\_i)\right|
\le \sum\_{i=1}^{r}|x\_i-y\_i|
$$

rather than any extra slack of order $r$. We will state this explicitly (with compact-support / boundedness assumptions for pre-activations) to clarify why coupling across $r$ channels closes.











**Q2 (Spectral bias and Section 4).** On universality and training regimes: we will revise Section 4 to separate statements cleanly. In the NTK / lazy training regime, the effective bias is governed by the kernel spectrum and one expects the classical low-frequency-first behavior emphasized in that line of work. The empirical tilt toward comparatively higher-frequency structure that we highlight is observed in a richer feature-learning setting for our low-rank RF architecture; we do not claim that this behavior is universal across all scalings (including finite width outside mean-field parameterizations or lazy limits).

Regarding Rahaman et al., "On the Spectral Bias of Neural Networks" (2019), arXiv:1806.08734, [https://arxiv.org/abs/1806.08734](https://arxiv.org/abs/1806.08734), is an apt reference for the standard Fourier / low-frequency-first picture in sufficiently wide, kernel-like regimes.

On why low-rank maps might carry more high-frequency content (strictly heuristic, under review): in ReLU networks of depth $L$, one can view NN outputs through "sum over paths" (per-region linear parametrizations); shrinking the effective rank reduces how many independent hinge directions can co-activate across depth, which loosely relaxes constraints on how linear regions can tile input space---informally, a less constrained CPWL can allow relatively more high-dimensional faces versus low-dimensional ones and thus alter how energy is distributed across spatial frequencies in a Fourier lens. This complements the shorter "fewer effectively independent hinge directions / boundary geometry" phrasing we will keep in the main text and will be marked explicitly as exploratory reasoning and ongoing review, not as a formal implication.

On exposition and scope: we agree Section 4 is currently qualitative and that the two-point analytic anchor is minimal. We will tone down claims, acknowledge the simplicity of the two-point construction, and add citations for the CPWL / compositional picture and Fourier heuristics beyond Rahaman et al.
