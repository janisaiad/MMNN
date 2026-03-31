We thank the reviewer. We will sharpen scope/novelty in the introduction and discussion and trim minor repetition.

**Q1 (Novelty; freezing; applicability; Theorem 4.1).** RF-LR / channelwise technical novelty vs. Nguyen & Pham (2023): see reviewer Z9GA **Q1 (technical novelty)**; not repeated here. The revision will state more explicitly that our proof builds on their template.

Frozen first-layer features: same phrasing as **Sec. 3.1** (reviewer VCqo); not repeated here.

For Theorem 4.1 we keep the two-point/two-channel proof as anchor; the revision adds a brief remark on positive feedback (**details in Q4 below**). Higher-dimensional/overlapping features stay empirical/open, not in the theorem’s scope.


Applicability: extensions/benchmarks are in reviewer ByWV **Q1 (Additional experiments)**



**Q2 (Symmetry sensitivity; exponential factor).** We agree this is important. The Gronwall exponential in finite-width bounds is a worst-case stability artifact; it should not be read as a causal explanation of any particular symmetry diagnostic. To avoid anecdotal conclusions, we ran a focused post-review sweep (width 1024, 5 seeds) and observed that the even-function symmetry diagnostic can improve when increasing rank. In this protocol, with $D\_{\mathrm{sym}}=\mathbb{E}[(f(x)-f(-x))^2]$ evaluated on a positive grid in $x$, values decreased from $1.89 \pm 1.73 \times 10^{-4}$ at $r=10$ to $4.18 \pm 0.58 \times 10^{-5}$ at $r=20$, with comparable test MSE; the relative version dividing by output energy behaved similarly. In the revision we will report mean and std and tone down any universal claim that a specific rank is "necessary" for symmetry.

For transparency, we will also describe the exact protocol (target, evaluation grid, seed list) and, if space is tight, move the full table to the supplement while keeping a concise summary in the main text.



**Q3 (Scaling to higher intrinsic dimension; phase transition).** We agree this is one of the most important "beyond the theorem" questions. Our current mean-field guarantees are formulated for fixed rank $r$ in the infinite-width limit; they are not, by themselves, a phase diagram in $(d,r,N)$. We will state that limitation plainly in the discussion.

That said, we can still articulate a research roadmap. On the NTK / kernel side, how rank interacts with input dimension depends on controlling the induced kernel ensemble; related random-matrix analyses become delicate for low-rank structured weights, and we are building on recent frameworks for structured kernels (e.g. [arxiv:2508.20036](https://arxiv.org/abs/2508.20036)) to connect scaling of $r$ with high-dimensional concentration. On the mean-field / Chizat side, recent scaling-law discussions suggest a "sweet spot" for faithful ODE descriptions when rank grows sublinearly with width (e.g. regimes such as $r\sim\sqrt{\text{width}}$ are often discussed as practically relevant; see also [this scaling note](https://arxiv.org/pdf/2509.10167)). Separately, in LoRA-style NTK analyses, rank thresholds of the form $r\gtrsim M^{\alpha}$ with $\alpha\in[1/4,1/2]$ are sometimes discussed as regimes where kernel descriptions remain predictive; we treat such statements as indicators, not as theorems for our exact architecture.



**Q4 (Mechanism behind the channel spike; beyond the two-point toy).** The explanatory content is not the simplified two-point, two-channel statement itself, but the positive feedback it isolates. Fix a second-layer neuron index $c\_2$ and recall the low-rank pre-activation

$$
H\_2(c\_2;x,W)=\sum\_{j=1}^r W\_{c\_2,j}\, f\_j(x;W).
$$

For ReLU, gating enters mean-field gradients through $\varphi'\_2(H\_2)=\mathbf{1}\{H\_2>0\}$. The backward signal for channel $k$ (cf. the ODE for $\partial\_t w\_1(\cdot,k)$) is built from expectations over $C\_2$ of terms that couple (i) the mixing weight $W\_{C\_2,k}$, (ii) the gate $\mathbf{1}\{H\_2(C\_2;x,W)>0\}$, and (iii) upstream factors (output error and later-layer weights). When channel $k$ dominates, neurons $c\_2$ that contribute to the drift along channel $k$ are precisely those with the gate ``on.'' Across $c\_2$, the effective weights in the expectation are then positively aligned with $W\_{c\_2,k}$ and with $\mathbf{1}\{H\_2(c\_2)>0\}$, producing a large contribution to the drift of the $k$th channel, which in turn pushes $f\_k$ further in the same direction: a reinforcing loop. The ODE-level mechanism is dimension-agnostic; the two-point theorem is a minimal tractable instance. We will add this explanation in the main text.

**Q5 (Limitations: frozen mixing).** Frozen mixing is not a universal model of all deep nets, but RF/factorized weights are standard in SciML/structured learning—and our results show global-optimality-type statements can still align with strong low rank under mean-field scaling (pushing back on “low rank always hurts”). We will rewrite limitations as a clear **scope statement** (assumptions $\to$ claims).