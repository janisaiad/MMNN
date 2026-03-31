We thank the reviewer for these additional points. They help us sharpen what is proved, what is empirical, and what remains open. Several of these points overlap with other reviews, and we will make the overall paper wording consistent across all responses.

**Q1 (Convergence guarantees).** Our main guarantee is conditional on convergence of the mean-field dynamics: given convergence, we can conclude global optimality within the RF-LR class under the stated assumptions. We will revise the abstract/introduction/theorem statements so this reads as "conditional global optimality" rather than as an unconditional convergence-rate claim.

We do not currently prove an unconditional convergence rate for deep RF-LR, and we agree this remains largely open. In the revision we will be explicit about which assumptions are used for the global-optimality implication (non-degeneracy of the initialization relative to the trivial predictor, and the regularity condition used to pass from a vanishing conditional gradient to a vanishing residual), versus which statements are empirical/heuristic.

**Q2 (High-dimensional spike mechanism).** The ODE-level feedback behind channel specialization is dimension-agnostic at the algebraic level: it is driven by the same identity

$$
H_2(c_2;x,W)=\sum_{j=1}^r W_{c_2,j}\, f_j(x;W)
$$

together with gating/backprop terms. To make the mechanism more transparent, we will add a short explanation in the main text. For ReLU-type gates, the derivative term is an indicator

$$
\mathbf{1}\{H_2>0\},
$$

so the drift and backprop signal for channel $k$ is built from expectations over $C_2$ of terms that couple (i) the mixing weight $W_{C_2,k}$, (ii) the gate $\mathbf{1}\{H_2(C_2;x,W)>0\}$, and (iii) upstream factors (output error and later-layer weights). When one channel becomes slightly dominant on part of the input space, it increases $H_2$ there, turns on more gates for neurons aligned with that channel, and thereby increases the drift along channel $k$. This in turn pushes $f_k$ further in the same direction: a reinforcing feedback loop. The two-point theorem is a minimal tractable instance where this loop can be proved cleanly.

What we currently prove is a minimal two-point anchor; for general $x\in\mathbb{R}^d$, overlapping channel contributions complicate the rigorous isolation of "well-separated spikes". We will state this distinction clearly, keep discussion in dimension $d$ honest as primarily empirical, and add higher-$d$ experiments/diagnostics as space permits. We will also note that taking $d=1$ is mainly for visualization clarity; the algebraic structure above does not rely on $d=1$.

For ReLU specifically, we will also clarify the main degenerate failure mode: convergence to a stationary point where gates are off on a large set (so $\phi_2'(H_2)=0$ despite nonzero residual). In RF-LR,

$$
H_2(c_2;x,W)=\sum_{k=1}^r W_{c_2,k}\, f_k(x;W)
$$

is a sum over $r$ channels, so the event that all channels are simultaneously dead is a joint sign-alignment event across channels. Under symmetric random mixing and non-degenerate channel features, one can heuristically view this as exponentially unlikely in $r$ (e.g. on the order of $2^{-r}$ pointwise), which motivates the "high probability in $r$" intuition. We will keep this strictly as intuition (not as a proof step) and will be careful about how it is stated.

**Q3 (Sensitivity to the frozen draw).** The analytical conditions are not tied to one sampling recipe: once mixing matrices are frozen, the arguments use bounded mixing and a richness/diversity assumption for the frozen first-layer features, not Gaussianity. Concretely, a convenient sufficient form is: for all $c$ and $k$,

$$
|W^{(\ell)}_{c,k}|\le K,
$$

and hence

$$
\lVert W^{(\ell)}\rVert_{\infty,1}:=\sup_c\sum_{k=1}^r |W^{(\ell)}_{c,k}|\le rK
$$

almost surely. This can be satisfied by any bounded-support mixing distribution (and is also compatible with nonnegative matrix factorized mixing); there is no Gaussian-specific requirement in the theory.

At finite width, performance can vary with the frozen draw. In our current experiments we often use Gaussian/Xavier initializations for convenience, even though those are unbounded; in the camera-ready we will clearly separate the theorem assumptions from this engineering choice, and where feasible we will include multi-draw variability (multiple frozen draws, multiple seeds) in addition to the usual seed variability. We will also clarify that any ReLU dead-gate-across-all-channels probability discussion is heuristic: in a symmetric mixing picture it can be exponentially small in $r$, but we treat this as intuition, not as a proof step.

We have not yet run a systematic post-review sweep over independent frozen draws with optimization held fixed; in the revision we will either add a compact version (if space permits) or state this explicitly as a limitation/future-work item.
