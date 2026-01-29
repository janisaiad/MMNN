# ICML 2026 Review: "Low-Rank Neural Network Is Sufficient for Global Convergence: A Mean-Field Perspective"

## Overall Assessment: **REJECT**

**Summary:** This paper studies mean-field convergence of low-rank neural networks with frozen random features. While the theoretical framework is well-structured, the results are severely limited by conditional convergence guarantees, strong architectural restrictions, and insufficient experimental validation. The main theorem only guarantees that *if* convergence occurs, it is to a global minimizer—but provides no conditions under which convergence actually happens. This fundamentally undermines the paper's contribution.

---

## Major Concerns

### 1. **Conditional Convergence: The Elephant in the Room**

**Severity: CRITICAL**

The main result (Theorem 2) is fundamentally conditional: "if the mean-field dynamics converges, then it is to a global minimizer." This is a significant limitation that the paper does not adequately address.

**Issues:**
- Section 2.5 ("When does convergence occur?") is essentially empty—it merely cites prior work without providing sufficient conditions for convergence in the low-rank setting.
- The paper provides no characterization of when convergence actually occurs, making the result of limited practical value.
- The assumption of convergence (Assumption 6) is stated but never verified or characterized—it's simply assumed to hold.

**What's needed:** The paper should either:
1. Provide sufficient conditions under which convergence is guaranteed (e.g., under convex loss, or with specific initialization/architecture constraints), OR
2. Explicitly acknowledge this as a major limitation and discuss when convergence might fail.

**Impact:** This reduces the main theorem to: "If you're already at a global minimizer, you're at a global minimizer"—which is trivial. The non-trivial part (guaranteeing convergence) is left unaddressed.

### 2. **Frozen Random Features: A Severe Architectural Restriction**

**Severity: CRITICAL**

The paper requires *all* mixing matrices $W^{(\ell)}$ to be frozen as random features. This is presented as a "solution, not a limitation" (line 276), but this is misleading.

**Issues:**
- This is a very strong architectural restriction that fundamentally changes the learning problem.
- The justification (avoiding neural collapse) is valid, but the cost is high: the network cannot learn task-specific features in the mixing layers.
- The paper claims this "enables global convergence" but doesn't show that training the mixing matrices would prevent convergence—only that it might cause neural collapse.
- Most practical low-rank methods (e.g., LoRA) train both factors, making this result less applicable.

**What's needed:**
- A clear discussion of the trade-offs: what expressivity is lost by freezing?
- Comparison with partially trainable mixing matrices (e.g., training with small learning rates).
- Acknowledgment that this is indeed a limitation, not just a design choice.

### 3. **ReLU Exclusion and Activation Restrictions**

**Severity: MAJOR**

Assumption 1 explicitly excludes ReLU, requiring $\inf_u |\varphi_2'(u)| \ge 1/K > 0$. This excludes the most commonly used activation function in practice.

**Issues:**
- The paper mentions a "high-probability relaxation" for ReLU (Appendix A.3, line 1526) but this is vague: "exponentially rare in $r$" is not quantified.
- The relaxation is not incorporated into the main theorems—it's only a remark.
- Most practical networks use ReLU, making the results less applicable.

**What's needed:**
- Either prove the result for ReLU (with quantified probability bounds), or clearly state this as a limitation.
- The high-probability argument should be made rigorous with explicit bounds.

### 4. **Incomplete Proofs and Proof Sketches**

**Severity: MAJOR**

Many critical proofs are incomplete or deferred.

**Issues:**
- Theorem 2 (global convergence): The proof in Appendix A.4 is a high-level sketch that relies heavily on prior work. Key steps are missing:
  - The argument that $B_k^{(2)}(x;\bar W) \neq 0$ on a set of positive measure (line 1522) is hand-wavy.
  - The connection between dense span and the conclusion needs more rigor.
- Well-posedness proof: While more complete, it relies on adapting prior work without clearly showing where the low-rank structure requires new arguments.
- Theorem 3 (feature learning): The proof is conditional on hypotheses (i)-(iii) that are never verified—only discussed heuristically in Appendix A.5.

**What's needed:**
- Complete, self-contained proofs for all main theorems.
- Clear identification of what is new vs. what is adapted from prior work.

### 5. **Insufficient Experimental Validation**

**Severity: MAJOR**

The experimental section is limited and does not adequately validate the theoretical claims.

**Issues:**
- Experiments are primarily on 1D synthetic functions—this is too narrow to claim general applicability.
- MNIST results (Table 1) show the architecture works, but don't validate the convergence guarantees.
- No experiments showing that convergence actually occurs (or fails) in practice.
- No comparison with full-rank networks on the same tasks to validate the claimed advantages.
- The "faster convergence" claim (line 359) is not supported by convergence plots or quantitative comparisons.

**What's needed:**
- Experiments on higher-dimensional, real-world datasets.
- Convergence analysis: plots showing loss evolution and whether convergence to global minimizers occurs.
- Direct comparison with full-rank networks on identical tasks.
- Ablation studies on the effect of freezing vs. training mixing matrices.

### 6. **Exponential Bound in Quantitative Theorem**

**Severity: MODERATE**

Theorem 3 provides a finite-width approximation bound with an exponential factor $e^{K_T(1+rK)}$.

**Issues:**
- For typical values ($r=50$, $K \approx 1$, $T=10$), this gives $e^{K_T(1+rK)} \approx e^{1000}$, which is astronomically large and renders the bound essentially meaningless.
- The paper acknowledges this (line 1933) but dismisses it as "worst-case" without providing better bounds or showing that the worst-case doesn't occur in practice.

**What's needed:**
- Either improve the bound (e.g., show it's actually polynomial in practice), or provide empirical validation that the bound is loose.
- Discussion of when the exponential factor might be tight.

---

## Minor Concerns

### 7. **Notation and Clarity**

- The notation is dense and sometimes inconsistent (e.g., $W$ vs. $W^{(\ell)}$ vs. mixing matrix $L$).
- The neuronal embedding framework is introduced without sufficient motivation.
- The forward/backward equations (Section 2.2-2.3) are hard to follow—more intuition would help.

### 8. **Related Work**

- The comparison with prior work (especially Nguyen et al. 2023) is good, but the discussion of when the frozen features approach is preferable is insufficient.
- Missing discussion of recent work on low-rank training (e.g., Kim et al. 2025, mentioned but not compared).

### 9. **Figure Quality**

- Figure 1 (architecture) is helpful but could be clearer.
- Many experimental figures are referenced but not shown (e.g., "Figure X shows..." but figure is missing or unclear).

### 10. **Writing and Presentation**

- Some sentences are overly long and hard to parse (e.g., line 276).
- The abstract claims "faster convergence" but this isn't clearly demonstrated.
- The paper would benefit from a clearer roadmap of contributions vs. limitations.

---

## Strengths

1. **Theoretical Framework:** The mean-field framework is well-structured and the adaptation to low-rank networks is technically sound.

2. **Well-Posedness:** The well-posedness result (Theorem 1) appears complete and rigorous.

3. **Feature Learning Mechanism:** The channel specialization mechanism (Theorem 4) is interesting and provides insight into how low-rank networks learn.

4. **Connection to Prior Work:** Good positioning relative to Nguyen et al. 2023 and other mean-field results.

---

## Specific Technical Issues

### Line 1522: The $B_k^{(2)} \neq 0$ Argument

The proof states: "Hence for $\mathcal{P}_X$-a.e. $x$ and almost every $c_2$, the factor in $B_k^{(2)}$ involving $\varphi_2'$ is non-zero, so $B_k^{(2)}(x;\bar W)$ is non-zero on a set of positive $\mathcal{P}_X$-measure."

This reasoning is flawed:
- Non-zero $\varphi_2'$ for a.e. $(x,c_2)$ does not guarantee that the *expectation* $B_k^{(2)}$ is non-zero.
- The mixing matrix $W_{C_2,k}$ could cancel out contributions.
- This needs a more careful argument using the "better than null" assumption and the structure of the mixing matrix.

### Line 1933: Exponential Bound Dismissal

The paper states: "However, in practice, channel specialization enables a more favorable learning regime." This is speculation—there's no proof that channel specialization improves the bound, and no experiments validating this claim.

---

## Recommendations for Revision

If the authors wish to revise, they must address:

1. **Provide sufficient conditions for convergence** (or explicitly state this as a limitation).
2. **Rigorously handle ReLU** (with quantified bounds) or clearly exclude it.
3. **Complete all proofs** with clear identification of novel vs. adapted arguments.
4. **Expand experiments** to validate convergence claims and compare with baselines.
5. **Acknowledge limitations** of frozen random features more honestly.
6. **Fix the $B_k^{(2)} \neq 0$ argument** in the convergence proof.

---

## Final Verdict

**Recommendation: REJECT**

While the paper tackles an important problem and uses sophisticated techniques, the fundamental limitation of conditional convergence, combined with strong architectural restrictions and insufficient validation, makes it unsuitable for ICML in its current form. The theoretical framework is sound, but the results are too limited to be of significant practical or theoretical value.

The paper would be more suitable for a workshop or a more specialized venue after addressing the major concerns above.

---

## Detailed Comments by Section

### Abstract
- Line 162: "When the mean-field dynamics converges" — this conditional should be more prominent.
- Line 165: "faster convergence" — not clearly demonstrated in experiments.

### Introduction
- Line 276: The claim that frozen features are "a solution, not a limitation" is misleading.
- Line 280: The informal theorem should emphasize the conditional nature.

### Section 2 (Main Results)
- Section 2.5 is essentially empty—this is a critical gap.
- The assumptions are reasonable but some (e.g., ReLU exclusion) need better justification.

### Section 3 (Feature Learning)
- Theorem 4 is interesting but conditional on unverified hypotheses.
- The experimental validation is limited to synthetic 1D functions.

### Section 4 (Experiments)
- Too narrow: only 1D synthetic functions.
- Missing: convergence analysis, comparison with full-rank, ablation studies.

### Appendix
- Proofs are incomplete or sketchy.
- The ReLU relaxation (A.3) needs quantification.

---

**Reviewer Confidence: High**

I am confident in this assessment. The conditional convergence issue is fundamental and cannot be overlooked. The paper needs substantial revision before it can be considered for publication.
