# Review: "Low rank is enough for the neural tangent kernel: depth–rank parameter tradeoff in kernel optimization"

## Summary

The paper studies the NTK of a low-rank random-feature architecture (RF-LR): wide layers with frozen random feature directions and trainable readouts into a bottleneck of dimension \(r \ll N\). Under the sequential infinite-width limit and a deterministic proxy for the random correlation chain, the authors derive an explicit NTK recursion with a visible \(1/r\) factor at each bottleneck layer, closed-form depth scaling, condition-number bounds for the proxy Gram matrix, and—for three layers only—RKHS equivalence with the shallow ReLU kernel. Numerical experiments are in the appendix.

## Strengths

1. **Clear problem and setting.** RF-LR is well defined (frozen features, trainable readouts, EOC scaling), and the distinction between base/derivative kernels and the classical NNGP/NTK is stated.
2. **Explicit recursion.** The infinite-width NTK recursion (Theorem 1) and the \(2L\)-term closed form (Corollary 1) are concrete and usable; the \(1/r\) bottleneck factor is derived, not assumed.
3. **Depth scaling.** Theorem 2 gives sharp rates (\(1-\rho_k = \Theta(k^{-2})\), saturation, gap \(\asymp 1/(rk)\)) and connects cleanly to the MLP-at-EOC literature.
4. **Conditioning.** The lower bound \(\kappa \ge \Omega(r \cdot L)\) for general data and exact \(\kappa_\perp = 1\) (or \(1+o(1)\)) for equicorrelated / high-dimensional spherical data are clearly stated; the tradeoff under a fixed budget \(O(NLr)\) is a useful takeaway.
5. **RKHS (three layers).** Fisher–Kibble decoupling and the Puiseux analysis are nontrivial; the conclusion that the three-layer mean kernel has the same RKHS as the shallow ReLU kernel is a clear theoretical plus.
6. **Proxy–empirical link.** Theorem 4 gives a rigorous bound \(\|(\hat{K}-K_{\mathrm{proxy}})|_{\mathbf{1}^\perp}\|_{\mathrm{op}} = O_P(L/r + 1/\sqrt{r})\) for equicorrelated data, which is important for connecting theory to the actual Gram matrix.

## Weaknesses and Concerns

1. **Heavy reliance on the proxy.** All condition-number results are for the *proxy* kernel. The general proxy–empirical bound is only a proof sketch (Appendix: "sketch-based"; full proof only for equicorrelated). For COLT, a rigorous non-equicorrelated concentration bound (or a clear "open problem" statement) would strengthen the contribution. As is, the message "\(\kappa \ge \Omega(r \cdot L)\)" is proven for the proxy; its validity for the empirical kernel in the general case is not fully established.

2. **RKHS only for \(L=3\).** The title and message "low rank is enough" are partly driven by RKHS equivalence, which is proved only for three layers. The paper honestly states that \(L \ge 4\) is open, but for a reader expecting a general "depth + low rank" story, the scope is limited. The extension to \(L \ge 4\) is mentioned as future work without a clear technical roadmap.

3. **Limited optimization consequences.** There is no theorem linking condition number to convergence rate or sample complexity of gradient descent / kernel regression. The discussion stays at the level of "conditioning is relevant for convergence"; for a theory venue, at least one concrete implication (e.g., convergence rate or sample complexity in terms of \(\kappa\), \(r\), \(L\)) would significantly increase impact.

4. **Experiments in appendix.** Experiments are relegated to the appendix and not tightly integrated with the main text. A short main-text subsection with one or two key plots (e.g., depth scaling and proxy–empirical agreement) would make the empirical validation more visible and the paper more self-contained.

5. **Notation and terminology.** "MMNN" appears in the appendix (e.g., "RF-LR/MMNN") but is not defined in the main text; the recursion section says "NTK recursion for MMNNs" while the rest of the paper uses "RF-LR." Unifying terminology would avoid confusion. The base/derivative kernel definition is careful but dense; a single clarifying sentence (e.g., "conditional on previous layer") right after Definition 1 would help.

6. **Bias term "+1".** The constant "+1" from biases is stated but not fully tied to "centered" kernel regression. A one-sentence remark that for mean-zero targets the constant mode is irrelevant and that \(\kappa_\perp\) is the relevant quantity would make the narrative cleaner.

## Questions for the Authors

- **Q1.** For general (non-equicorrelated) data, can you give a full proof of \(\|\hat{K} - K_{\mathrm{proxy}}\|_{\mathrm{op}}\) (or \(\|\cdot\|_{\mathbf{1}^\perp}\)) with an explicit rate in \(r,L,n\), or would you be willing to state this as an open problem?
- **Q2.** For \(L \ge 4\), what is the main obstruction to RKHS equivalence—only the Laplace-type expansion for the Fisher chain, or also endpoint/Puiseux control at \(\rho = \pm 1\)?
- **Q3.** Can you add a short remark (or one result) on how \(\kappa \ge \Omega(r \cdot L)\) (or \(\kappa_\perp\) in the "good" cases) translates into a convergence or sample-complexity guarantee for kernel regression with this kernel?

## Verdict

**Borderline weak accept / weak reject.**

The paper gives a clear NTK analysis of an interesting low-rank lazy architecture, with an explicit recursion, sharp depth scaling, and condition-number bounds. The three-layer RKHS result and the equicorrelated proxy–empirical bound are solid contributions. However, (i) the main conditioning story is largely about the proxy, with the general proxy–empirical link only sketched; (ii) RKHS equivalence is limited to \(L=3\); and (iii) there is no direct link to optimization (convergence rate or sample complexity). For COLT, I would lean toward **weak accept** if the authors add a short discussion (or open-problem statement) on the general proxy–empirical concentration and clarify the scope of the RKHS result in the abstract/intro; I would lean toward **weak reject** if the proxy-centric nature of the conditioning results and the lack of optimization implications are judged as too limiting for the venue.
