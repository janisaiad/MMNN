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

- `confirm_thm15.png`
- `confirm_prop17.png`

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

## 5) Analytic Fisher/Kibble densities + \(I(r)\) and spectral-decay prefactor vs \(r\)

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
- Scripts in Sections 3–4 are **finite-width / finite-rank Monte Carlo** and are meant to illustrate remarks about fluctuation scales (they do not constitute proofs).
- Centering removes the rank-one spike aligned with \(\mathbf{1}\), but does not “improve” the remaining spectral shape; see `refs/colt2026/spectra.tex` (“Centered vs uncentered”) and `rmk:bulk-vs-extremes`.

