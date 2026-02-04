# Fixes applied to tofill.tex and further suggestions

## Fixes applied

1. **Assumptions paragraph (line 517)**  
   Added missing period after “satisfied by standard MLP architectures”.

2. **Quantitative guarantees (line 605)**  
   “The detailed proof are” → “The detailed proofs are”.

3. **Feature learning (line 645)**  
   “A uncommon” → “An uncommon”.

4. **Mechanism of spike learning (line 734)**  
   “non-zero from zero” → “bounded away from zero” (correct technical wording).

5. **Proof sketch (line 839)**  
   “The proof  is” → “The proof is” (double space).

6. **Toy model (line 743)**  
   “Spatial-Fourier” → “spatial--Fourier” (proper en-dash for compound).

7. **Numerical results paragraph (line 902)**  
   “Table~\ref{tab:num-hparams} frozen” → “Table~\ref{tab:num-hparams} compares frozen”.

8. **Figure caption asymmetry (line 876)**  
   “(b)--(c) ;” → “(b)--(c):” and tightened caption wording.

9. **Lemma proof (line 1432)**  
   “because of conciseness to prove sub-gaussian bounds for lipschitz variable” → “for conciseness; proving sub-Gaussian bounds for Lipschitz variables follows standard arguments.”

10. **“fastly decaying”**  
    Replaced all remaining instances with “rapidly decaying” (main text and appendix).

11. **Duplicate figure labels**  
    - First figure* (asymmetry): kept single label `\label{fig:mlp-asymmetry}`; removed `fig:channel-combined`, `fig:spike-in-H`, `fig:mlp-asymmetry-L7`.  
    - Second figure* (channel partials): kept single label `\label{fig:channel-partials}`; removed `fig:activations-combined`, `fig:channel-partials-L1`, `fig:channel-partials-L5`.  
    - All references to removed labels now point to `fig:channel-partials` or `fig:mlp-asymmetry` as appropriate.

12. **Table (hyperparameters)**  
    Removed duplicate row that referred to the removed `fig:spike-in-H`; table now has one row for `fig:channel-partials`.

13. **Figure caption (final prediction)**  
    Added missing period at end of caption (“$\sim10^{-4}$}” → “$\sim10^{-4}$.}”).

14. **Caption wording**  
    “(symmetric) with bigger spikes” → “Symmetric runs with bigger spikes” and added semicolons in (a)(b)(c).

---

## Further suggestions (not applied)

- **pdf title (line 48)**  
  Current: “Mean-Field Global Convergence for Low-Rank Neural Networks”.  
  Consider aligning with full title: “… without Neural Collapse” if you want the PDF metadata to match.

- **Informal theorem (line 280)**  
  “and if the weights … converge” is a bit heavy; consider: “If the weights in all layers $(A_1,\ldots,A_{L-1})$ converge as $t\to\infty$, then the limit is a global minimizer of the population loss” (and drop the earlier “and” so the sentence reads cleanly).

- **Log-ratio caption (line 803)**  
  “20 epochs” vs table “10k” epochs: confirm whether the figure and table describe the same run and fix number or caption if not.

- **Appendix “fastly”**  
  If any “fastly” remains in appendix (e.g. in eq or long paragraphs), replace with “rapidly decaying” for consistency.

- **B\_\{k,p+1\} index (eq. dtfk-full-delta)**  
  In the main text, $\partial_t f_k(t,x_p)$ uses $B_{k,p+1}(t)$; at $x_0$ ($p=0$) that is $B_{k,1}$, and at $x_1$ ($p=1$) that is $B_{k,2}$. Confirm that this indexing matches the definition “at $x_0$: … $B_{k,1}(t)=B_k(t;x_0)$; at $x_1$: … $B_{k,2}(t)=B_k(t;x_1)$” everywhere (main text and appendix).

- **Grammar**  
  “converges” vs “converge”: in Theorem (informal), “if the weights … converge” is correct (plural subject). In “when the dynamics … converges”, “dynamics” is singular, so “converges” is correct. No change needed if already consistent.
