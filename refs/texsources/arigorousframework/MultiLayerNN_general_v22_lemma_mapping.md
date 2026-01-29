# Appendix lemma-to-lemma mapping (lines 4140–7225)

## MultiLayerNN_general_v22.tex

Table columns: **Label** | **Type** | **Section** | **Lines** | **One-line statement** | **Proves** | **Used in proof of** | **Uses**

No rows are included for `lem:bounds MF a priori` or `lem:difference MF` (main-text statements; only their proofs appear in the appendix). Citations *to* them in appendix proofs are reflected in the **Used in proof of** column of the citing blocks.

---

| Label | Type | Section | Lines | One-line statement | Proves | Used in proof of | Uses |
|-------|------|---------|-------|--------------------|--------|------------------|------|
| `thm:azuma-hilbert` | thm | Useful tools | 4147–4170 | Martingale concentration in separable Hilbert spaces: $\mathbb{P}(\max_{k\le n}|Z_k|\ge t)\le 2\exp(-t^2/(16nR^2))$ for $t<nR$ when increments are bounded by $R$. | — | `thm:iid-hilbert-concentration` | (pinelis cite only) |
| `thm:iid-hilbert-concentration` | thm | Useful tools | 4173–4201 | For $n$ $\eta$-independent r.v.s in a Hilbert space with $|X_i-\mathbb{E}[X_i]|\le R$, $\mathbb{P}(n^{-1}|\sum_i X_i-\mathbb{E}[X_i]|\ge\delta)\le 2\exp(-n\delta^2/(64R^2))$ when $\delta>2\eta R$. | — | `lem:square hoeffding`; Proof of `cor:gradient descent quality` | `thm:azuma-hilbert` |
| `thm:iid-hilbert-higher-moment` | thm | Useful tools | 4203–4257 | Moments of $\eta$-independent heavy-tailed sums: $m$-th root of the $m$-th moment of the centered average is $\le Km^{1+k/2}\max(\eta n^{0.01},n^{-1/2})$ under a $k$-moment growth condition. | — | `lem:initialization_compare` | (pinelis cite only) |
| `lem:Lipschitz forward MF` | lem | sec:Remaining-proofs-existence-MF | 4333–4341, 4465–4472 | Under Assumption forward, $H_i$ and $\hat{y}$ are Lipschitz in $W$: $L^2$- and ess-sup bounds with constant $K^L K_0^L(T)\|W'-W''\|_t$. | — | Proof of `lem:difference MF` | `enu:Assump_forward`; `lem:Lipschitz forward MF - general` (in proof) |
| `lem:Lipschitz backward MF` | lem | sec:Remaining-proofs-existence-MF | 4343–4356, 4509–4514 | Under forward/backward and high-probability bounds on $\mathsf{max}_T^w$, $\Delta_i^w$, $\Delta_i^b$, $\Delta_1^w$ are Lipschitz in $W$ with $D(t,W',W'')$ control. | — | Proof of `lem:difference MF` | `enu:Assump_forward`, `enu:Assump_backward`; `lem:Lipschitz backward MF - general` (in proof) |
| `lem:Lipschitz forward MF - general` | lem | sec:Remaining-proofs-existence-MF | 4391–4428, 4474–4506 | With $\tilde{C}_i$, $K_*(T)$ and $\tilde{d}_t$ as in the statement, $H_i(X,\tilde{C}_i;W')-H_i(X,\tilde{C}_i;W'')$ is bounded in $L^2$ by $K^L K_*^L(T)\tilde{d}_t(W',W'')$ under Assumption forward. | — | `lem:Lipschitz forward MF`; `lem:Lipschitz backward MF - general` | `enu:Assump_forward` |
| `lem:Lipschitz backward MF - general` | lem | sec:Remaining-proofs-existence-MF | 4431–4460, 4516–4690 | With $\tilde{C}_i$, $K_*$, $\tilde{d}_t$, $\Xi$ as in the statement, $\Delta_i^w$, $\Delta_i^b$, $\Delta_1^w$ (in $\tilde{C}$ and $C$ variants) are $L^2$-Lipschitz with $\tilde{D}(t,W',W'')$ under forward/backward. | — | `lem:Lipschitz backward MF` | `lem:Lipschitz forward MF - general`; `enu:Assump_forward`, `enu:Assump_backward` |
| `lem:square hoeffding` | lem | sec:Remaining-proofs-main-MF | 4702–4750 | For $\eta$-independent $(c_i)$, independent $x$, and $f_i$ with $|f_i(c_i,x)-f_i(x)|\le R$, $\mathbb{P}(\mathbb{E}_x[|n^{-1}\sum_i f_i(c_i,x)-f_i(x)|]\ge\delta)\le (4R/\delta)\exp(-n\delta^2/(512R^2))$ when $\delta>2\eta R$. | — | `prop:particle coupling - bounded`; `prop:gradient descent - bounded` | `thm:iid-hilbert-concentration` |
| `lem:initialization_compare` | lem | sec:Remaining-proofs-main-MF | 4753–4871 | Under `assump:init`, after coupling: with high probability, moment and excess bounds hold for $\interleave\tilde{W}\interleave_0$, sampled $|w_i^0|$, and tail fractions vs $\mathbb{P}(|w_i^0|\ge B)$, $\mathbb{P}(|b_i^0|\ge B)$. | — | `prop:particle coupling - bounded`; `prop:gradient descent - bounded`; Proof of `cor:gradient descent quality` | `assump:init`; `assump:neuronal-embedding`; `thm:iid-hilbert-higher-moment` |
| `lem:bounds NN a priori` | lem | sec:Remaining-proofs-main-MF | 4874–4931 | Under lr-schedule and backward, $\interleave\tilde{W}\interleave_t$, $\interleave\mathbf{W}\interleave_{\lfloor t/\epsilon\rfloor}$, $\interleave W\interleave_{\mathrm{samp},t}$ and related $\Delta_i^{\mathbf{H}},\Delta_i^H$ are at most $K^{\kappa_L}(1+t^{\kappa_L})(1+\interleave\cdot\interleave_0^{\kappa_L})$. | — | `lem:a priori MF - time difference`; Proof of `cor:gradient descent quality`; `prop:particle coupling - bounded` | `enu:Assump_lrSchedule`, `enu:Assump_backward`; `lem:bounds MF a priori` |
| `lem:a priori MF - time difference` | lem | sec:Remaining-proofs-main-MF | 5889–5909 | For the MF trajectory $W(t)$ under lr-schedule, backward and `assump:init`, $\|W-W_\zeta\|_T\le K_{T+\zeta}\zeta$ with $W_\zeta(t)=W(t+\zeta)$ and $K_{T+\zeta}$ depending on init and $\zeta$. | — | Proof of `cor:gradient descent quality` | `enu:Assump_lrSchedule`, `enu:Assump_backward`, `assump:init`; `prop:particle coupling - bounded`; `lem:bounds NN a priori` |
| `thm:iid dynamics-full` | thm | subsec:Infinite-M-limit-full | 6331–6655 | Complete statement of `thm:iid dynamics`: $\sup_{t\le T}\langle W^M-W^\infty\rangle_t\le K_{T,L}/M^{0.499}$ and, for $L\ge4$ and $2\le i\le L-2$, analogous $H_i$–$H_i^*$ bound. | — | `thm:iid dynamics` | `thm:iid dynamics`; `subsec:Canonical-neuronal-embeddings`, `subsec:Infinite-M-canonical`; `enu:Assump_lrSchedule`, `enu:Assump_backward`, `assump:init`; in proof: `lem:bounds MF a priori`, `thm:existence ODE` |
| `lem:full-support-2` | lem | sec:Remaining-proofs-global-conv-iid (subsec:Proof-two-layers) | 6730–6841 | For the 2-layer MF ODE ($\mathbb{W}_1=\mathbb{R}^d$, no $b_2$), if the support of $\mathrm{Law}(w_1(0,C_1),w_2(0,C_1,1))$ contains the graph of a bounded continuous $F:\mathbb{W}_1\to\mathbb{W}_2$, then the support of $\mathrm{Law}(w_1(t,C_1))$ is $\mathbb{W}_1$ for all $t$. | — | `thm:global-optimum-2`; `thm:global-optimum-2-ms` | `subsec:MF`; `enu:Assump_lrSchedule`, `enu:Assump_backward`, `assump:init` |

---

## Compiled numbers (pdflatex, MultiLayerNN\_general\_v22.tex)

thm/lem/prop/cor share one counter. **Main text:** Lemma 8 (lem:bounds MF a priori), Lemma 10 (lem:difference MF); Corollary 17 (cor:gradient descent quality); Propositions 22, 23 (prop:particle coupling - bounded; prop:gradient descent - bounded). **Appendix:** Theorem 43 (thm:azuma-hilbert), 44 (thm:iid-hilbert-concentration), 45 (thm:iid-hilbert-higher-moment); Lemmas 46–49 (Lipschitz forward/backward MF, general); Lemmas 50–53 (lem:square hoeffding; lem:initialization\_compare; lem:bounds NN a priori; lem:a priori MF - time difference); Theorem 54 (thm:iid dynamics-full); Lemma 55 (lem:full-support-2).

---

## Notes

- **Proves**: All appendix theorem-like blocks have Proves = "—" by design.
- **Section**: Smallest enclosing `\section` or `\subsection`; “Useful tools” has no `\label`.
- **Uses**: Refs of the form `\ref{thm:...}`, `\ref{lem:...}`, `\ref{prop:...}`, `\ref{cor:...}`, `\ref{assump:...}`, `\ref{enu:Assump_...}`, and `\ref{sec:...}` / `\ref{subsec:...}` when they are the primary structural reference. External citations (e.g. pinelis) are noted in prose only.
- **Used in proof of**: Blocks Y (including main-text results whose proofs are in the appendix) that cite this block in `\begin{proof}[Proof of ... \ref{Y}]` or in the body of Y. Citations to `lem:bounds MF a priori` and `lem:difference MF` inside the appendix are only reflected in the **Used in proof of** of the citing blocks; those two lemmas do not appear as rows.
- **Lines**: Statement and proof; for `lem:Lipschitz forward MF` and `lem:Lipschitz backward MF` the statement and proof are non-contiguous, so both ranges are given.
