# ICML Review: Mean-Field Global Convergence for Low-Rank Neural Networks without Neural Collapse

**Overall recommendation: Weak Reject (5)**

---

## Summary

The paper studies low-rank random-feature (RF-LR) neural networks in the mean-field regime. The main claims are: (1) when the mean-field dynamics converges, the limit is a global minimizer of the population loss, for any depth $L \ge 2$ and under standard i.i.d. initialization; (2) freezing mixing matrices $W^{(\ell)}$ as random features avoids neural collapse (unlike full-rank networks, which require ad-hoc initialization per Nguyen et al.); (3) the mean-field ODE system is well-posed; (4) a finite-width approximation bound is given; (5) a conditional theorem on log-ratio growth in a two-point ("two-sided step") toy model characterizes channel specialization (spatial localization and dominance). Experiments on 1D oscillatory targets and MNIST are reported.

---

## Strengths

1. **Relevant question.** Whether global convergence persists under low-rank constraints in mean-field networks is a natural and timely question. The paper clearly positions itself relative to Nguyen et al. (2023) and Chizat–Bach.

2. **Clear architectural choice.** The RF-LR design (train only $A_\ell$, freeze $W^{(\ell)}$) is stated explicitly, and the link to avoiding neural collapse is motivated by prior work.

3. **Structured appendix.** The mapping table (Appendix) from the paper’s lemmas/theorems to the corresponding results in Nguyen et al. is helpful for assessing novelty and for reproducibility of the adaptations.

4. **Comparison table.** Table comparing Chizat–Bach, Nguyen–Pham, and this work (depth, initialization, architecture, loss, convergence) is useful.

5. **Well-posedness.** Existence and uniqueness of the mean-field ODE solution are stated and a proof sketch is given, with the low-rank adaptations (bi-Lipschitz in $W$, $\|W^{(\ell)}\|_{\infty,1} \le rK$, $\max_k$ over channels) indicated.

6. **Feature-learning narrative.** The discussion of channel-wise spike learning and the two-sided step toy model gives intuition for how dominance and log-ratio growth can arise.

---

## Weaknesses

1. **Conditional main result.** The central theorem is *conditional*: *if* the dynamics converges, *then* the limit is a global minimizer. The paper does not establish that the dynamics actually converges in the low-rank setting; it defers to prior work (convex loss or Morse–Sard conditions). Thus the main guarantee is "no bad limit points" rather than "convergence to a global minimizer." For a convergence paper, the lack of a convergence result in the low-rank case is a significant gap.

2. **Theorem 2 (log-ratio) is heavily conditional.** Theorem (two-sided step, log-ratio growth) is explicitly conditional on (i) $-d_0(t) \ge 0$, (ii) sign coherence of $B_{1,1}$ with $f_1$, and (iii) a dominance bound $|B_{2,1}| \le \rho_0 \frac{|f_2|}{|f_1|}|B_{1,1}|$. The authors do not prove that (i)–(iii) hold from the model or initialization; they argue heuristically and say these hold "in practice." So the theorem only says: *if* these conditions hold on an interval, *then* the log-ratio is non-decreasing. Without verification of the hypothesis, the result is a conditional structural statement rather than a predictive guarantee.

3. **ReLU and main assumptions.** Main results assume $\varphi_2'$ bounded away from zero, which excludes ReLU. The ReLU case is relegated to "high probability in $r$" in the appendix. Given that several experiments use ReLU (e.g. log-ratio setup, "ReLU($H_8$)" in figures), the mismatch between theory (Leaky ReLU / sigmoid) and experiments (ReLU) should be clearly stated in the main text and its implications discussed.

4. **Proofs are sketches.** Key steps are omitted: e.g. Lemma (bounds-a-priori) states "We do not provide this proof here because of conciseness." Well-posedness and global convergence are largely adaptations of Nguyen et al.; the review table helps, but a moderately strict reviewer will note that several arguments are only outlined. Full proofs or a clear "proof of X is identical to Nguyen et al. except for Y" would strengthen the paper.

5. **Quantitative bound is pessimistic.** The finite-width bound includes a factor $e^{K_T(1+rK)}$, which is acknowledged as extremely large; the text then appeals to "channel specialization" in practice without a theoretical justification for better scaling. The bound is therefore of limited practical use as stated.

6. **Experiments.**  
   - Evaluations are mostly 1D function approximation; MNIST appears in a single table.  
   - There is no direct empirical comparison showing "collapse vs no collapse" (e.g. same data/width, full-rank i.i.d. vs RF-LR, or trainable vs frozen $W$) to support the narrative that freezing avoids collapse.  
   - Figure captions are dense and sometimes inconsistent (e.g. "(a) (b) Asym. Layer 7. (c) Asym. Layer 16" with only two panels; multiple \label's on one figure).  
   - Table 1 (MNIST): column header "rModel" is a typo and should be "Model."

7. **Language and notation.**  
   - "Fastly decaying" is non-standard; "rapidly decaying" or "fast-decaying" is standard.  
   - Minor: "proveide" in a commented line; double \label's on figures can cause ambiguous references.

8. **Depth $L \ge 2$.** The main theorem is stated for any $L \ge 2$, but the toy model and much of the analysis are for $L=2$ or $L=3$. A short remark on how the argument extends to general $L$ (e.g. via the same backprop structure and dense span) would clarify scope.

---

## Questions for the Authors

1. Can you add a concise subsection or proposition stating *under what conditions* the mean-field dynamics is known to converge (e.g. from Chizat–Bach / Nguyen et al.), and whether those conditions apply to your loss/architecture? This would make the conditional nature of the main theorem and the open problem (convergence in the low-rank case) explicit.

2. For Theorem (log-ratio), can you either (a) prove that (i)–(iii) hold on a non-trivial interval under stated assumptions (e.g. two-sided step, specific init), or (b) clearly label the result as "structural: under these hypotheses, dominance is preserved," and avoid claiming it as a predictive guarantee for training?

3. Experiments: could you include one controlled experiment (same data, width, and target) comparing: (i) full-rank i.i.d. init, (ii) RF-LR (frozen $W$), and optionally (iii) low-rank with trainable $W$, with a clear metric or visualization indicating collapse vs non-collapse (e.g. effective rank of activations or variance across neurons)? This would support the claim that freezing avoids collapse.

4. Please replace "fastly decaying" with "rapidly decaying" (or "fast-decaying") throughout, and fix the table header "rModel" → "Model" and the duplicate \label's on the two figure* environments (lines 877–880 and 889–892).

5. How does the constant in the well-posedness and finite-width bounds depend on depth $L$? The appendix emphasizes $r$ and $\|W^{(\ell)}\|_{\infty,1}$; a sentence on $L$-dependence would help.

---

## Detailed Feedback (optional)

- **Appendix Table (lemma mapping):** Useful. Consider adding one column: "Proof given in this paper? (Y / sketch / reference only)."
- **Assumption (convergence to limit point):** The coupling formulation is heavy. A one-sentence summary in the main text (e.g. "weights converge in a Wasserstein-like sense to a limit $(\bar{A}_1,\ldots,\bar{A}_{L-1})$") would improve accessibility.
- **Figure (four-setups-loss):** The caption mentions "red-bar decay" and momentum values; ensure the caption is self-contained so that the comparison (low-rank vs full-rank, role of red bars) is clear without reading the body.

---

## Recommendation and justification

**Weak Reject (5).**

The paper addresses an important question and provides a clear adaptation of the Nguyen et al. framework to low-rank + frozen random features, with a well-posedness result and a conditional global-optimality result. The appendix mapping and the comparison table are strengths. However, the main theorem is conditional on convergence, which is not established here; the feature-learning theorem is conditional on unverified hypotheses; several proofs are sketches; and the experiments do not directly demonstrate collapse vs no-collapse or the benefit of the proposed architecture in a controlled way. The combination of these issues makes the contribution feel incomplete for a top-tier venue in its current form. A revision that (i) makes the conditional nature of both main theorems explicit and either proves or clearly labels the log-ratio hypotheses, (ii) adds a controlled experiment supporting the collapse narrative, and (iii) tightens presentation (notation, figure labels, full proofs or clear pointers) could support an accept.
