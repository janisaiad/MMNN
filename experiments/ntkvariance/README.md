# NTK variance & depth-scaling experiments (`experiments/ntkvariance/`)

This folder contains small, self-contained scripts that **numerically illustrate** several deterministic/proxy statements and finite-width/finite-rank phenomena discussed in the COLT paper.

The main paper compile target is:

- `refs/colt2026/TOCOMPILE_colt2026-sample.tex`

which includes (in order):

- `\input{recursion}` → `refs/colt2026/recursion.tex`
- `\input{depth_scaling}` → `refs/colt2026/depth_scaling.tex`
- `\input{rkhs}` → `refs/colt2026/rkhs.tex`
- `\input{appendix}` → `refs/colt2026/appendix.tex`
- `\input{spectra}` → `refs/colt2026/spectra.tex`

All scripts here are **headless** (matplotlib `"Agg"` backend) and save figures next to the script.

## Experimental setups (depth, width, ranks, etc.)

| Script | Depth \(L\) | Width | \(n\) | \(d\) | Ranks \(r\) | Trials / notes |
|--------|-------------|--------|-------|-------|-------------|----------------|
| `confirm_depth_scaling.py` | \(k_{\max}=4000\) (proxy recursion) | — | 64 (Prop 17) | — | [5, 10, 20, 50] | deterministic |
| `scalingdepth.py` | \(L=200\) | — | — | — | [5, 10, 20, 50] | deterministic, \(c_0=1/2\) |
| `minimalvariance.py` | 2 (3-layer, 2 ReLU) | 16000 | 2 pts \(x=-1,x'=1\) | 1 (scalar) | [5, 10, 20, 50] | 100 |
| `min_eig_distribution.py` | 2 | 16000 | 64 | 16 | [5, 10, 20, 50] | 50 |
| `condition_number_proxy_empirical.py` | 2 | 20000 | 32 | 64 | [10,…,1e4] | 40, equicorrelated \(\rho_0=0\) |
| `condition_number_highdim_spherical.py` | 2 | 16000 | 32 | 256, 128 | [10,…,200] | 30 per \(d\) |
| `condition_number_non_equicorrelated.py` | 2 | 20000 | 48 | 64 | [10,…,1e4,1e5] | 30, 4 clusters (slower conv.) |
| `kernel_regression_risk.py` | 2 | 12000 | 64 train, 256 test | 32 | [5,…,100] | 20 |
| `fisher_kibble_and_decay_constants.py` | — | — | — | — | \(r\in[3,200]\) (grid) | analytic (no MC) |

- **Depth:** All finite-width NTK scripts use a **3-layer (2 ReLU)** RF-LR architecture, i.e. \(L=2\) hidden ReLU layers and one output layer; the proxy/deterministic scripts iterate the recursion to large \(k\) (e.g. \(k_{\max}=4000\) or \(L=200\)).
- **Width:** Hidden layer width is 12000 or 16000 in Monte Carlo scripts so that finite-width variance is small and the proxy limit is visible.
- **Ranks:** Bottleneck rank \(r\) is swept as listed; equicorrelated proxy–empirical goes to \(10^4\), non-equicorrelated to \(10^4\) and \(10^5\) (with width \(2\times 10^4\)) to show slower convergence toward \(\kappa_\perp=1\).
- **Data:** \(n\) = number of inputs, \(d\) = input dimension; equicorrelated/high-dim spherical use unit-sphere or equicorrelated design; kernel regression uses train/test split on the sphere.

## 1) Deterministic mean-field / proxy depth scaling (Theorem 15, Props 16–17)

### `confirm_depth_scaling.py`
**What it computes**

- Deterministic ReLU EOC correlation recursion \( \rho_k=\varrho(\rho_{k-1}) \)
- Deterministic proxy NTK recursion (Eq. (11) in the paper’s depth section):
  \[
  \Theta^{(1)}(\rho_1)=1+s(\rho_1),\qquad
  \Theta^{(k)}(\rho_1)=1+\frac{1}{r}\Theta^{(k-1)}(\rho_1)\dot s(\rho_k)+\frac{1}{r}s(\rho_k)
  \]
- Theorem 15 diagnostics: \(w_k/k\), \((1-\rho_k)k^2\), \(k|\Theta^{(k)}-\Theta_\star(r)|\), and \(rk(\Theta_{\rm diag}^{(k)}-\Theta_{\rm off}^{(k)})\)
- Proposition 17 (equicorrelated model) scalings via explicit eigenvalues:
  \[
  \lambda_1=\Theta^{(L)}(1)+(n-1)\Theta^{(L)}(\rho_0),\quad
  \lambda_\perp=\Theta^{(L)}(1)-\Theta^{(L)}(\rho_0)
  \]

**Where it maps in the paper**

- Theorem 15: `refs/colt2026/depth_scaling.tex` (`thm:rflr-depth-scaling-mean`)
- Proposition 16: `refs/colt2026/depth_scaling.tex` (`prop:equicorrelated-gap`)
- Proposition 17: `refs/colt2026/depth_scaling.tex` (`prop:equicorrelated-spectrum`)
- Remarks clarifying centering/conditioning: `refs/colt2026/depth_scaling.tex` (`rmk:conditioning-centered`, `rmk:lambda-min-negative`)

**Outputs**

- **`confirm_thm15.png`** — Four panels (Theorem 15): (1) \(w_k/k\) vs \(k\) (should stabilize); (2) \(1-\rho_k\) vs \(k\) on log–log with \(k^{-2}\) reference; (3) \(k|\Theta^{(k)}(\rho_1)-\Theta_\star(r)|\) vs \(k\) (bounded); (4) \(rk(\Theta_{\rm diag}^{(k)}-\Theta_{\rm off}^{(k)})\) vs \(k\) (stabilizes). One curve per rank \(r\in\{5,10,20,50\}\).
- **`confirm_prop17.png`** — Two panels (Proposition 17, equicorrelated \(n=64\), \(\rho_0=0\)): (1) spike saturation \(L|\lambda_1/n-\Theta_\star(r)|/n\) vs \(L\) (log–log); (2) gap scaling \(rL\lambda_\perp\) vs \(L\) (log–log). One curve per \(r\).

## 2) Exponential depth suppression factor (closed form product bound)

### `scalingdepth.py`
**What it computes**

This script plots the **multiplicative attenuation factor** that appears in the closed form expansion of the RF-LR NTK recursion:

- It uses the deterministic mean-field correlation path \( \rho_k=\varrho(\rho_{k-1}) \) (ReLU EOC),
- then plots, as a function of \(j=L-\ell\),
  \[
  \prod_{k=\ell+1}^{L}\dot\Sigma^{(k)}
  \;=\;
  \prod_{k=\ell+1}^{L}\frac{\bar{\dot\Sigma}(\rho_k)}{r},
  \qquad
  \bar{\dot\Sigma}(\rho)=\frac12-\frac{\arccos(\rho)}{2\pi}\le c_0=\frac12,
  \]
  and compares it to the bound \((c_0/r)^j\).

It also plots correlation alignment:

- \( \rho_k \) vs \(k\)
- \( 1-\rho_k \) vs \(k\) on log–log scale with a \(k^{-2}\) reference line (Theorem 15’s \(1-\rho_k=O(k^{-2})\)).

**Where it maps in the paper**

- Closed form recursion expansion (the product term): `refs/colt2026/recursion.tex` (Eq. `eq:ntk_explicit_form`)
- Exponential suppression discussion: `refs/colt2026/recursion.tex` (“Large-depth regime: probabilistic kernel and exponential decay”)
- Correlation alignment rate: `refs/colt2026/depth_scaling.tex` (Theorem 15, “Correlation alignment”)

**Outputs**

- `decay_vs_depth.png`
- `rho_vs_depth.png`
- `one_minus_rho_vs_depth_loglog.png`

## 3) Finite-width Monte Carlo: variance of a single NTK entry

### `minimalvariance.py`
**What it computes**

Monte Carlo over random initializations of a 3-layer (2-ReLU) low-rank RF-LR network, estimating:

- \(\mathrm{Var}[K(-1,1)]\) vs \(r\) (log–log)

This is an **empirical** (finite-width) check that fluctuations shrink as \(r\) grows.

**Where it maps in the paper**

- Rank-driven concentration discussion: `refs/colt2026/rkhs.tex` / `refs/colt2026/appendix.tex` (concentration statements around Fisher/Kibble corrections)

**Outputs**

- `variance_vs_r.png`

## 4) Finite-width Monte Carlo: distribution of smallest centered eigenvalue

### `min_eig_distribution.py`
**What it computes**

Monte Carlo over random initializations, for a fixed random dataset \(x_1,\dots,x_n\) on the unit sphere:

- Builds the empirical Gram matrix \(K\), centers it \(K_c=HKH\),
- records \( \lambda_{\min}^+(K_c) \) (the smallest **strictly positive** eigenvalue; centering creates a forced 0 eigenvalue).

This illustrates the “**driven close to 0 by noise**” phenomenon discussed in the paper’s remark (even though \(K_c\succeq 0\) always).

**Where it maps in the paper**

- Remark “Can the smallest eigenvalue become negative?”: `refs/colt2026/depth_scaling.tex` (`rmk:lambda-min-negative`)

**Outputs**

- `min_eig_distribution.png`
- `min_eig_cdf.png`

## 5) Condition number: proxy vs empirical (equicorrelated)

### `condition_number_proxy_empirical.py`
**What it computes**

- **Proxy:** for equicorrelated data the proxy Gram has \(\kappa_\perp=1\) on \(\mathbf{1}^\perp\) (Corollary 13). The script confirms this (proxy curve at 1).
- **Empirical:** 3-layer RF-LR NTK Gram over equicorrelated inputs; condition number of the *centered* Gram (restriction to \(\mathbf{1}^\perp\)) vs bottleneck rank \(r\), with mean and std over random initializations.
- Validates that the empirical condition number concentrates around the proxy value 1 as \(r\) grows (Theorem equicorrelated-op-bound in the appendix).

**Where it maps in the paper**

- Corollary 13 (exact conditioning equicorrelated): `refs/colt2026/depth_scaling.tex` (`cor:conditioning-equicorrelated-highdim`)
- Proxy–empirical concentration on \(\mathbf{1}^\perp\): `refs/colt2026/appendix.tex` (`thm:equicorrelated-op-bound`)

**Outputs**

- `condition_number_proxy_vs_empirical.png`

**Run**

- `python3 condition_number_proxy_empirical.py` (requires `numpy`, `matplotlib`).

## 6) High-dimensional spherical data: condition number (Corollary 13, second item)

### `condition_number_highdim_spherical.py`
**What it computes**

- Same condition-number plot for **i.i.d. uniform** points on \(S^{d-1}\) with large \(d\) (e.g. \(d=256\)).
- Validates \(\kappa_\perp = 1 + o(1)\) and concentration as \(r\) grows (Corollary 13, high-dimensional random data).
- With high probability \(\max_{i\neq j} |\rho_{ij}| = O(1/\sqrt{d})\), so data are approximately equicorrelated.

**Where it maps in the paper**

- Corollary 13 (second item): `refs/colt2026/depth_scaling.tex` (high-dimensional spherical data).

**Outputs**

- `condition_number_highdim_spherical.png`

**Run**

- `python3 condition_number_highdim_spherical.py` (requires `numpy`, `matplotlib`).

## 7) Kernel regression risk vs \(r\)

### `kernel_regression_risk.py`
**What it computes**

- Fix a target (linear or low-degree polynomial on the sphere).
- Fit kernel ridge regression with the empirical RF-LR NTK Gram.
- Plots **test MSE** vs bottleneck rank \(r\) (and optionally vs \(L\)).
- Validates that as \(r\) grows and the Gram concentrates, kernel regression improves.

**Where it maps in the paper**

- RKHS, concentration, and optimization implications: `refs/colt2026/rkhs.tex`, `refs/colt2026/depth_scaling.tex`.

**Outputs**

- `kernel_regression_risk_vs_r.png`

**Run**

- `python3 kernel_regression_risk.py` (requires `numpy`, `matplotlib`).

## 8) Non-equicorrelated data: condition number (Proposition 6)

### `condition_number_non_equicorrelated.py`
**What it computes**

- Same condition-number vs \(r\) for a **clustered** design with varying \(\rho_{ij}\) (e.g. a few clusters).
- Illustrates that the proxy lower bound \(\kappa \geq \Omega(r \cdot L)\) can be large and that empirical \(\kappa\) need not approach 1 (contrast with equicorrelated data where \(\kappa_\perp = 1\)).

**Where it maps in the paper**

- Proposition 6 (condition number lower bound): `refs/colt2026/depth_scaling.tex`.

**Outputs**

- `condition_number_non_equicorrelated.png`

**Run**

- `python3 condition_number_non_equicorrelated.py` (requires `numpy`, `matplotlib`).

## 9) Analytic Fisher/Kibble densities + \(I(r)\) and spectral-decay prefactor vs \(r\)

### `fisher_kibble_and_decay_constants.py`
**What it computes**

Plots **closed-form curves** (not empirical sampling):

- Fisher density \(p(\hat\rho\mid \rho,r)\) with the hypergeometric \({}_2F_1\) term
- Kibble joint density \(f(u,v)\) with modified Bessel \(I_\nu\)

Then plots **\(r\)-dependent constants** appearing in the Puiseux coefficient and spectral-decay amplitude:

- \(I(r)\) and checks \(\sqrt r\,I(r)\) stabilizes
- \(|2c_1(r)|\) (even-parity spectral-decay amplitude up to the dimension constant \(C(d)\))

**Where it maps in the paper**

- Fisher/Kibble closed forms: `refs/colt2026/appendix.tex` (`rmk:fisher-kibble-law`)
- \(I(r)\sim C/\sqrt r\): `refs/colt2026/appendix.tex` (`appendix:I-r-scaling`)
- Puiseux coefficient \(c_1(r)\) and RKHS decay: `refs/colt2026/rkhs.tex` and appendix Puiseux derivations

**Outputs**

- `fisher_density_curves.png`
- `kibble_density_contours.png`
- `I_of_r_scaling.png`
- `sqrt_r_I_of_r.png`
- `spectral_decay_prefactor_vs_r.png`

**Dependency**

- Requires `scipy` (`scipy.special.hyp2f1` and `scipy.special.iv`).

## Notes and cautions (COLT-style)

- Scripts in Sections 1–2 are **deterministic proxy / mean-field** computations (they illustrate Theorem 15–17 as stated in `depth_scaling.tex`).
- Scripts in Sections 3–5, 7–8 are **finite-width / finite-rank Monte Carlo** and are meant to illustrate remarks about fluctuation scales (they do not constitute proofs).
- Section 6 (high-dim spherical) and Section 9 (Fisher/Kibble) mix proxy and analytic computations.
- Centering removes the rank-one spike aligned with \(\mathbf{1}\), but does not “improve” the remaining spectral shape; see `refs/colt2026/spectra.tex` (“Centered vs uncentered”) and `rmk:bulk-vs-extremes`.

## Quick run (all experiments)

```bash
cd experiments/ntkvariance
python3 confirm_depth_scaling.py
python3 scalingdepth.py
python3 minimalvariance.py
python3 min_eig_distribution.py
python3 condition_number_proxy_empirical.py
python3 condition_number_highdim_spherical.py
python3 kernel_regression_risk.py
python3 condition_number_non_equicorrelated.py
python3 fisher_kibble_and_decay_constants.py   # requires scipy
```

