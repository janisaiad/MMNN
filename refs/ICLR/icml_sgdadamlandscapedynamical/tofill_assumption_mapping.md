# Assumptions mapping — tofill.tex

Table columns: **Label** | **Name** | **Section** | **Lines** | **One-line statement** | **Used in**

---

| Label | Name | Section | Lines | One-line statement | Used in |
|-------|------|---------|-------|--------------------|---------|
| `assump:regularity` | Bounded Activations and Mixing | app:assumptions | 1130–1137 | $\|\varphi_1\|_\infty,\|\varphi_2\|_\infty,\|\varphi_2'\|_\infty\le K$; $\varphi_1,\varphi_2$ $K$-Lipschitz; $\varphi_2'$ non-vanishing; $|L_{c_2,k}|\le K$ and $\|L\|_{\infty,1}\le rK$. | sec:assumptions; `thm:wellposed`; `thm:convergence`; `thm:quantitative-main`; `lem:bounds-a-priori`; `lem:invariance-F`; `lem:difference-F`; proof of `thm:convergence` (1502, 1508); `thm:quantitative-lowrank`; proofs in app:quantitative (1762, 1825, 1848, 1857, 1929, 1944, 1967) |
| `assump:init` | Sub-Gaussian Initialization | app:assumptions | 1139–1148 | $\psi_2$ (or $m^{-1/2}$-moment) bounds on $w_1^0$ and $w_2^0$: $\llbracket w_1(0)\rrbracket_\psi,\llbracket w_2(0)\rrbracket_\psi<\infty$. | sec:assumptions; `thm:wellposed`; `thm:convergence`; (implicitly in well-posedness a priori bounds) |
| `assump:data` | Data Distribution and Loss Regularity | app:assumptions | 1150–1157 | $|X|\le K$ and $\|L^0(c_1)\|\le K$; $\partial_2\mathcal{L}(y,\cdot)$ $K$-bounded and $K$-Lipschitz, $\partial_2\mathcal{L}(y,u)=0$ only when $u=0$; no convexity required. | sec:assumptions; `thm:wellposed`; `thm:convergence`; `lem:bounds-a-priori`; `lem:invariance-F`; `lem:difference-F` |
| `assump:diversity` | Diversity of Random Features | app:assumptions | 1159–1162 | $\operatorname{supp}(\rho^1)=\mathbb{R}^d$ (or dense), so $\{\varphi_1(\langle L^0(c_1),\cdot\rangle):c_1\in\Omega_1\}$ has dense span in $L^2(\mathcal{P}_X)$. | sec:assumptions; `thm:wellposed`; `thm:convergence`; proof of `thm:convergence` (1484, with `thm:uap-automatic`) |
| `assump:lr` | Non-Degeneracy | app:assumptions | 1164–1172 | $\mathscr{L}(w_1^0,\ldots,w_L^0)<\mathbb{E}_Z[\mathcal{L}(Y,\varphi_L(0))]$ so the limit satisfies $\max_{1\le k\le r}\mathbb{P}(\bar{w}_1(C_1,k)\ne 0)>0$ and $\mathbb{P}(\bar{w}_\ell(C_\ell)\ne 0)>0$ for $\ell=2,\ldots,L$. | sec:assumptions; `thm:convergence`; proof of `thm:convergence` (1502) |
| `assump:convergence` | Convergence to Limit Point | app:assumptions | 1180–1186 | Coupling $\pi_t$ and Wasserstein-like integrals (eq. conv-w1--conv-w2) and $L$-layer analogues tend to 0 as $t\to\infty$. | sec:assumptions; `thm:convergence`; proof of `thm:convergence` (1487, 1508) |
| `assump:quant-init` | Initialization for Low-Rank Networks | app:quantitative | 1617–1620 | $\operatorname{ess-sup}\max_{1\le k\le r}|w_1^0(C_1,k)|\le K$ and $\operatorname{ess-sup}|w_2^0(C_2)|\le K$. | `thm:quantitative-main`; `thm:quantitative-lowrank` |

---

## Notes

- **assump:regularity**–**assump:convergence** live in `app:assumptions` (Section “Assumptions”); **assump:quant-init** is in `app:quantitative` and is used only for finite-width / quantitative approximation results.
- **assump:init** (sub-Gaussian) is weaker than **assump:quant-init** (ess-sup bounded); the former is for well-posedness and global convergence, the latter for explicit finite-width bounds (`thm:quantitative-main`, `thm:quantitative-lowrank`).
- **Used in**: theorems, lemmas, and proof steps that explicitly assume or cite the assumption; “sec:assumptions” is the main-text summary that lists all six core assumptions.
