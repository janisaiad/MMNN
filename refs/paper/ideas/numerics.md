# Visualization Strategy: Strong, Astonishing Plots for Low-Rank NTK Paper

## Design Principles
1. **Visual Proof**: Plots should directly demonstrate theoretical claims
2. **Clear Contrast**: Show RF-LR advantages vs MLPs side-by-side
3. **Exponential Scaling**: Emphasize exponential concentration (not just polynomial)
4. **Publication Quality**: Clean, professional, with proper mathematical notation
5. **Multi-scale**: Show both microscopic (rank-level) and macroscopic (spectral) phenomena

---

## Plot 1: Rank-Driven Concentration (Exponential Decay) ⭐⭐⭐
**Purpose**: Visually prove Theorem 3.2 (exponential concentration in rank)

### Design
- **Main plot**: Log-log or semi-log showing concentration rate vs rank
- **X-axis**: Rank $r \in [5, 200]$
- **Y-axis**: $\mathbb{P}(|W-1| \geq \epsilon)$ on log scale
- **Multiple curves**: Different $\epsilon$ values (0.1, 0.2, 0.5, 1.0)
- **Theoretical bound**: Overlay $4\exp(-r\epsilon^2/8)$ curves
- **Empirical**: Monte Carlo estimates with error bars

### Key Visual Elements
- **Exponential decay**: Clear straight line on semi-log plot demonstrating exponential concentration
- **Contrast with $1/\sqrt{r}$**: Overlay classical $O(1/\sqrt{r})$ rate to show dramatic improvement
- **Multiple $\epsilon$ curves**: Show concentration holds across different deviation thresholds
- **Theoretical vs empirical**: Overlay theoretical bound $4\exp(-r\epsilon^2/8)$ with Monte Carlo estimates
- **Annotation**: "Exponential concentration: $O(\exp(-r\epsilon^2))$ vs classical $O(1/\sqrt{r})$"

---

## Plot 2: Three-Layer NTK Concentration (Fisher-Kibble Decoupling) ⭐⭐⭐
**Purpose**: Show how empirical NTK concentrates around deterministic limit

### Design
- **Main plot**: Empirical NTK vs deterministic limit as function of correlation $\rho$
- **X-axis**: $\rho \in [-1, 1]$
- **Y-axis**: NTK value
- **Multiple panels**: Different ranks $r \in \{10, 30, 100\}$
- **Shaded regions**: $\pm 2$ standard deviations from mean
- **Overlay**: Deterministic limit $K_\infty(\rho)$

### Key Visual Elements
- **Convergence**: Shaded $\pm 2\sigma$ bands shrink dramatically as rank increases, demonstrating $O(1/r)$ variance decay
- **Fisher-Kibble structure**: Visualize how angular (Fisher) and radial (Kibble) components combine
- **Deterministic limit overlay**: Show $K_\infty(\rho)$ as reference, with empirical mean converging
- **Multi-rank comparison**: Three panels showing $r \in \{10, 30, 100\}$ to demonstrate rank-driven concentration
- **Annotation**: "Variance decays as $O(1/r)$ via Fisher-Kibble decoupling"

---

## Plot 3: Spectral Decay (RKHS Equivalence) ⭐⭐⭐
**Purpose**: Prove Corollary 3.5 - same RKHS despite depth

### Design
- **Main plot**: Eigenvalue decay on log-log scale
- **X-axis**: Spherical harmonic index $k$ (log scale)
- **Y-axis**: Eigenvalue $\mu_k$ (log scale)
- **Multiple curves**: Shallow (1-layer), 3-layer, 5-layer, 10-layer
- **Reference line**: $k^{-d}$ decay (theoretical prediction)
- **Highlight**: All curves parallel to $k^{-d}$ line

### Key Visual Elements
- **Perfect overlap**: All depth curves ($L \in \{1, 2, 3, 5, 10\}$) should visually coincide, proving RKHS equivalence
- **Power law reference**: Overlay $k^{-d}$ reference line to show all curves follow same scaling
- **Log-log scale**: Essential to visualize power-law decay over multiple orders of magnitude
- **Surprising result**: Depth provides no RKHS advantage—counterintuitive but theoretically proven
- **Annotation**: "Same spectral decay $\mu_k \sim k^{-d}$ for all depths—RKHS equivalence (Corollary 3.5)"

---

## Plot 4: Marchenko-Pastur Spectrum (Spike-Bulk Structure) ⭐⭐⭐
**Purpose**: Visualize Theorem 4.2 - deformed MP law with spike

### Design
- **Main plot**: Histogram of eigenvalues with theoretical MP density overlay
- **X-axis**: Eigenvalue $\lambda$
- **Y-axis**: Density
- **Multiple panels**: Different $\gamma_{\text{ratio}} = n/r$ values
- **Spike**: Vertical line showing outlier eigenvalue
- **Bulk**: Smooth MP density curve

### Key Visual Elements
- **Spike prominence**: Vertical line at $\lambda_{\text{spike}} \approx n \cdot K_\infty(0)$ clearly separated from bulk (orders of magnitude larger)
- **Bulk support**: Histogram of bulk eigenvalues with theoretical Marchenko-Pastur density overlay showing perfect match
- **Scaling contrast**: Visual separation between $O(n)$ spike and $O(1)$ bulk demonstrates different scaling regimes
- **Multi-ratio comparison**: Three panels with different $\gamma_{\text{ratio}} = n/r$ values show how rank affects spectrum
- **Theoretical validation**: MP density curve overlays empirical histogram, validating Theorem 4.2
- **Annotation**: "Spike-bulk structure: $O(n)$ outlier + deformed MP bulk (Theorem 4.2)"

---

## Plot 5: Computational Efficiency (O(rN) vs O(N²)) ⭐⭐
**Purpose**: Show parameter budget advantage

### Design
- **Main plot**: Parameter count vs width $N$ (log-log)
- **X-axis**: Width $N$
- **Y-axis**: Parameter count (log scale)
- **Two curves**: RF-LR ($O(rN)$) vs MLP ($O(N^2)$)
- **Multiple panels**: Different ranks $r \in \{10, 30, 100\}$
- **Annotation**: "Entry cost to kernel regime: $O(rN)$ vs $O(N^2)$"

### Key Visual Elements
- **Quadratic gap**: Log-log plot showing dramatic separation between $O(rN)$ and $O(N^2)$ curves
- **Rank scaling**: Three panels with $r \in \{10, 30, 100\}$ demonstrate how rank affects efficiency
- **Kernel regime threshold**: Vertical line marking $N \sim n^2$ requirement for MLPs to enter kernel regime
- **Practical impact**: Show parameter savings at realistic widths (e.g., $N=1000$)
- **Annotation**: "Entry cost to kernel regime: $O(rN)$ vs $O(N^2)$—quadratic parameter savings"

---

## Plot 6: Fisher-Kibble Decoupling Visualization ⭐⭐
**Purpose**: Show independence of angular and radial components

### Design
- **Scatter plot**: Angular correlation $\hat{\rho}_r$ vs radial product $w_r$
- **X-axis**: $\hat{\rho}_r$ (sample correlation)
- **Y-axis**: $w_r = \|x_1\|\|y_1\|/r$
- **Multiple panels**: Different true correlations $\rho \in \{0, 0.5, 0.9\}$
- **Overlay**: Marginal distributions on axes
- **Test**: Show independence via conditional distributions

### Key Visual Elements
- **Independence visualization**: Scatter plot showing no correlation structure between angular and radial components
- **Marginal distributions**: Histograms on axes showing Fisher distribution (angular) and Kibble distribution (radial)
- **Multiple correlation values**: Three panels with $\rho \in \{0, 0.5, 0.9\}$ demonstrate independence holds across all correlations
- **Concentration line**: Horizontal line at $\mathbb{E}[w_r] = 1$ showing radial component concentrates
- **Statistical independence**: Circular scatter pattern (no correlation) proves Lemma 2.1 visually
- **Annotation**: "Angular (Fisher) and radial (Kibble) are independent—fundamental decoupling (Lemma 2.1)"

---

## Plot 7: Mean NTK vs Deterministic Limit (Puiseux Expansion) ⭐⭐
**Purpose**: Show how mean NTK differs but preserves RKHS

### Design
- **Main plot**: Mean NTK and deterministic limit vs correlation $\rho$
- **X-axis**: $\rho \in [0.9, 1.0]$ (zoom near boundary)
- **Y-axis**: NTK value
- **Two curves**: Mean NTK $\tilde{\Theta}^{(2)}$ vs $K_\infty$
- **Difference plot**: Subplot showing $t^{1/2}$ scaling
- **Annotation**: "Same $t^{1/2}$ leading term → same RKHS"

### Key Visual Elements
- **Near-boundary zoom**: Focus on $\rho \in [0.9, 1.0]$ to reveal Puiseux expansion structure near singularity
- **Two-panel layout**: Upper panel shows both kernels, lower panel shows difference on log-log scale
- **Leading term match**: Both curves exhibit $t^{1/2}$ behavior (parallel to reference line in log-log), proving same RKHS
- **Higher-order divergence**: Difference plot shows where kernels diverge (higher-order terms), but leading term determines RKHS
- **Critical insight**: Same $t^{1/2}$ leading term → same spectral decay → same RKHS (Corollary 3.4)
- **Annotation**: "Same $t^{1/2}$ leading term → same RKHS despite different higher-order corrections"

---

## Plot 8: NTK Recursion Visualization (Depth Expansion) ⭐
**Purpose**: Show how NTK grows with depth via recursion

### Design
- **Main plot**: NTK value vs depth $L$
- **X-axis**: Depth $L \in [1, 10]$
- **Y-axis**: NTK value at fixed $\rho$
- **Multiple curves**: Different $\rho$ values to show correlation dependence
- **Term breakdown**: Stacked area or bar chart showing bias vs fresh basis contributions

### Key Visual Elements
- **Linear growth**: Demonstrate how $2L$ terms accumulate with depth (Corollary 2.3)
- **Term structure**: Visualize the two types of contributions: bias terms ($+1$ at each layer) and fresh basis terms ($\Sigma^{(\ell)}$)
- **Recursion pattern**: Show how each layer adds new terms while multiplying previous ones by derivative kernels

---

## Plot 9: Concentration Rate Comparison (1/r² vs 1/N) ⭐⭐⭐
**Purpose**: Highlight the key advantage: $O(1/r^2)$ vs $O(1/N)$ concentration rates

### Design
- **Main plot**: Standard deviation vs parameter count (log-log scale)
- **X-axis**: Parameter count (log scale)
- **Y-axis**: NTK standard deviation (log scale)
- **Two curves**: RF-LR ($O(1/r^2)$) vs MLP ($O(1/N)$)
- **Reference lines**: Overlay theoretical scaling laws

### Key Visual Elements
- **Steep slope contrast**: RF-LR curve has much steeper negative slope, demonstrating faster concentration
- **Crossover point**: Mark where RF-LR becomes superior (at relatively small parameter counts)
- **Exponential advantage**: Visual demonstration that $O(1/r^2)$ is exponentially faster than $O(1/N)$
- **Practical impact**: Show parameter savings needed to achieve same concentration level
- **Annotation**: "$O(1/r^2)$ exponentially faster than $O(1/N)$—key theoretical advantage"

---

## Plot 10: Training Dynamics (Stepwise Loss) ⭐⭐
**Purpose**: Show practical advantage in training with stepwise loss behavior

### Design
- **Main plot**: Training loss vs iteration (semi-log scale)
- **X-axis**: Training iteration
- **Y-axis**: Loss (log scale)
- **Multiple curves**: RF-LR with different ranks $r \in \{10, 30, 100\}$ vs MLP baseline
- **Highlight**: Staircase/stepwise behavior unique to RF-LR

### Key Visual Elements
- **Stepwise drops**: Clear discrete drops of size $O(r)$ corresponding to learning different frequency modes
- **Faster convergence**: RF-LR reaches lower final loss faster than MLP
- **Multi-index learning connection**: Each step corresponds to learning a new frequency component
- **Rank dependence**: Show how larger $r$ leads to more steps but faster overall convergence
- **Theoretical validation**: Connect staircase pattern to multi-index learning theory

---

## Implementation Checklist

### High Priority (Must Have)
- [ ] Plot 1: Rank-driven concentration (Theorem 3.2)
- [ ] Plot 3: Spectral decay RKHS equivalence (Corollary 3.5)
- [ ] Plot 4: Marchenko-Pastur spectrum (Theorem 4.2)
- [ ] Plot 9: Concentration rate comparison ($1/r^2$ vs $1/N$)

### Medium Priority (Should Have)
- [ ] Plot 2: Three-layer NTK concentration
- [ ] Plot 5: Computational efficiency
- [ ] Plot 6: Fisher-Kibble decoupling
- [ ] Plot 7: Mean NTK Puiseux expansion

### Lower Priority (Nice to Have)
- [ ] Plot 8: NTK recursion visualization
- [ ] Plot 10: Training dynamics

---

## Design Guidelines

### Color Scheme
- **Primary**: Blue for RF-LR, Red for MLP/classical
- **Theoretical**: Black dashed lines
- **Empirical**: Colored solid lines with transparency
- **Use**: `viridis`, `plasma`, or `cividis` for multi-curve plots

### Typography
- **Math**: Use LaTeX notation in labels: `$\\rho$`, `$r$`, etc.
- **Font size**: 12pt for labels, 14pt for titles, 10pt for legends
- **Consistency**: Same notation as paper (e.g., `$\\Kop$` for NTK)

### Layout
- **Aspect ratio**: 4:3 or 16:9 for main figures
- **Subplots**: Use `tight_layout()` or `constrained_layout=True`
- **Margins**: Leave space for axis labels

### File Formats
- **Vector**: PDF for LaTeX (best quality)
- **Raster**: PNG at 300 DPI for presentations
- **Naming**: `fig_rank_concentration.pdf`, `fig_spectral_decay.pdf`, etc.

---

## Precomputation Requirements

This section specifies all computations that must be completed before plotting. Each computation should save results to disk in a standardized format (e.g., NumPy `.npz`, HDF5, or JSON) so that plotting scripts only load data and visualize.

---

## Precise Computation Summary

Exact specifications for all computations. Each outputs a `.npz` file with arrays ready for plotting.

**Note**: All computations include FLOPs estimates and scaling law analysis where applicable.

### Plot 1: Rank-Driven Concentration → `data/plot1_rank_concentration.npz`
**Inputs**: $r \in [5, 200]$ (50 log-spaced), $\epsilon \in \{0.1, 0.2, 0.5, 1.0\}$, $N_{\text{MC}} = 10^5$  
**Compute**: 
- For each $(r, \epsilon)$: Sample $(x,y) \sim \mathcal{N}(0,I_r)$, compute $W = \|x\|\|y\|/r$, count $\mathbb{P}(|W-1| \geq \epsilon)$
- Theoretical: $4\exp(-r\epsilon^2/8)$  
**Output**: `r_vals`, `epsilons`, `prob_empirical[50,4]`, `prob_std[50,4]`, `prob_theoretical[50,4]`  
**FLOPs**: $\sim 50 \times 4 \times 10^5 \times (2r + 5) \approx 10^{10}$ FLOPs (scales as $O(N_{\text{MC}} \cdot n_r \cdot n_\epsilon \cdot r)$)  
**Scaling law**: Concentration rate $\sim \exp(-r\epsilon^2)$ (exponential in rank)

### Plot 2: Three-Layer NTK Concentration → `data/plot2_ntk_concentration.npz`
**Inputs**: $r \in \{10, 30, 100\}$, $\rho \in [-1, 1]$ (200 points), $N_{\text{MC}} = 10^4$  
**Compute**: 
- For each $(r, \rho)$: Sample $\hat{\rho}_r \sim \text{Fisher}(\rho,r)$, $w_r \sim \text{Kibble}(r,\rho)$, compute $\hat{\Theta}^{(2)}_r = \Psi(\hat{\rho}_r, w_r)$
- Deterministic: $K_\infty(\rho) = \Theta^{(1)}(\rho)(1-\arccos(\rho)/\pi) + \Sigma^{(1)}(\rho) + 1$  
**Output**: `r_vals[3]`, `rho_vals[200]`, `ntk_mean[3,200]`, `ntk_std[3,200]`, `ntk_deterministic[200]`  
**FLOPs**: $\sim 3 \times 200 \times 10^4 \times (C_{\text{Fisher}} + C_{\text{Kibble}} + C_{\text{NTK}}) \approx 10^{10}$ FLOPs (scales as $O(n_r \cdot n_\rho \cdot N_{\text{MC}})$)  
**Scaling law**: Variance $\sim O(1/r)$ (Fisher-Kibble decoupling)

### Plot 3: Spectral Decay (RKHS Equivalence) → `data/plot3_spectral_decay.npz`
**Inputs**: $L \in \{1, 2, 3, 5, 10\}$, $k \in [1, 1000]$, $d = 10$  
**Compute**: 
- For each $L$: Expand $\tilde{\Theta}^{(L)}(\rho)$ in spherical harmonics on $\mathbb{S}^{d-1}$, extract eigenvalues $\mu_k^{(L)}$
- Reference: $\mu_k = k^{-d}$  
**Output**: `depths[5]`, `k_vals[1000]`, `eigenvalues[5,1000]`, `eigenvalues_reference[1000]`  
**Time**: ~1-2 days (requires Gegenbauer expansion or numerical integration)

### Plot 4: Marchenko-Pastur Spectrum → `data/plot4_marchenko_pastur.npz`
**Inputs**: $n = 1000$, $\gamma_{\text{ratio}} \in \{0.5, 1.0, 2.0\}$ → $r \in \{2000, 1000, 500\}$, $N_{\text{init}} = 10$  
**Compute**: 
- For each $(n, r)$: Generate data $X \in \mathbb{R}^{n \times d}$, compute NTK Gram matrix $\mathbf{M}_{ij} = \Theta^{(2)}(X_i, X_j)$, eigendecompose
- Spike: $\lambda_{\text{spike}} = \lambda_1$ (largest eigenvalue)  
**Output**: `n`, `gamma_ratios[3]`, `r_vals[3]`, `eigenvalues_all[3,1000]`, `lambda_spike[3]`, `mp_density_params`  
**FLOPs**: $\sim 3 \times 10 \times (n^2 \cdot C_{\text{NTK}} + n^3) \approx 3 \times 10^{12}$ FLOPs (scales as $O(n_{\text{ratios}} \cdot N_{\text{init}} \cdot (n^2 \cdot C_{\text{NTK}} + n^3))$)  
**Scaling law**: Spike $\sim O(n)$, bulk $\sim O(1)$; eigenvalue computation $\sim O(n^3)$

### Plot 5: Computational Efficiency → `data/plot5_efficiency.npz`
**Inputs**: $N \in [10, 1000]$ (100 log-spaced), $r \in \{10, 30, 100\}$  
**Compute**: 
- RF-LR: `params_rflr = r * N` (3 curves)
- MLP: `params_mlp = N^2`  
**Output**: `N_vals[100]`, `r_vals[3]`, `params_rflr[3,100]`, `params_mlp[100]`  
**Time**: < 1 minute

### Plot 6: Fisher-Kibble Decoupling → `data/plot6_fisher_kibble.npz`
**Inputs**: $r = 30$, $\rho \in \{0, 0.5, 0.9\}$, $N_{\text{MC}} = 10^5$  
**Compute**: 
- For each $\rho$: Sample $(\hat{\rho}_r, w_r)$ pairs (Fisher + Kibble, independent)
- Compute marginals: histograms of $\hat{\rho}_r$ and $w_r$
- Test: $\text{corr}(\hat{\rho}_r, w_r) \approx 0$  
**Output**: `r`, `rho_true_vals[3]`, `rho_samples[3,100000]`, `w_samples[3,100000]`, `rho_hist[3,30]`, `w_hist[3,30]`, `correlation_coef[3]`  
**Time**: ~1-2 hours

### Plot 7: Mean NTK Puiseux Expansion → `data/plot7_puiseux_expansion.npz`
**Inputs**: $r = 30$, $\rho \in [0.9, 1.0]$ (1000 points), integration method  
**Compute**: 
- For each $\rho$: $\tilde{\Theta}^{(2)}(\rho) = \int\int \Psi(\hat{\rho}_r, w_r) p_{\text{Fisher}}(\hat{\rho}_r|\rho,r) p_{\text{Kibble}}(w_r|r,\rho) d\hat{\rho}_r dw_r$
- Deterministic: $K_\infty(\rho)$
- Difference: $\text{diff} = \tilde{\Theta}^{(2)} - K_\infty$, fit $c_1(r) t^{1/2}$ term  
**Output**: `r`, `rho_vals[1000]`, `mean_ntk[1000]`, `deterministic_ntk[1000]`, `difference[1000]`, `t_vals[1000]`, `puiseux_coefficient`  
**Time**: ~1 day (high-dimensional numerical integration)

### Plot 8: NTK Recursion Visualization → `data/plot8_ntk_recursion.npz`
**Inputs**: $L \in [1, 10]$, $\rho \in \{-1, -0.8, \ldots, 1\}$ (or specific values)  
**Compute**: 
- For each $(L, \rho)$: $\Theta^{(L)}(\rho)$ using recursion from Theorem 2.1
- Optional: Decompose into $2L$ terms (bias + fresh basis)  
**Output**: `depths[10]`, `rho_vals`, `ntk_values[10, n_rho]`, `term_breakdown[10, n_rho, 2]` (optional)  
**Time**: ~10 minutes

### Plot 9: Concentration Rate Comparison → `data/plot9_concentration_comparison.npz`
**Inputs**: $P \in [10^2, 10^6]$ (50 log-spaced), scaling assumptions  
**Compute**: 
- RF-LR: $\sigma_{\text{RF-LR}}(P) = O(1/r^2)$ where $P = r \cdot N$ (fix $N$ or $r$)
- MLP: $\sigma_{\text{MLP}}(P) = O(1/N) = O(1/\sqrt{P})$ where $P = N^2$
- Optional: Empirical validation via Monte Carlo  
**Output**: `param_counts[50]`, `std_rflr[50]`, `std_mlp[50]`, `std_rflr_empirical[50]` (optional), `std_mlp_empirical[50]` (optional)  
**FLOPs**: Theoretical: $\sim 10^3$ FLOPs; Empirical: $\sim 10^{12}$ FLOPs (scales as $O(n_P \cdot N_{\text{MC}} \cdot C_{\text{NTK}})$)  
**Scaling law**: RF-LR $\sigma \sim P^{-2}$ (if $r \propto \sqrt{P}$), MLP $\sigma \sim P^{-1/2}$; **exponential advantage** for RF-LR

### Plot 10: Training Dynamics (FLOPs-based) → `data/plot10_training_dynamics.npz`
**Inputs**: RF-LR $r \in \{10, 30, 100\}$, MLP baseline, dataset, $N_{\text{runs}} = 5$  
**Compute**: 
- For each architecture: Train network, record loss $L(\text{FLOPs})$ at each FLOP count
- Compute FLOPs per iteration: 
  - RF-LR forward: $O(r \cdot N \cdot d)$ FLOPs
  - RF-LR backward: $O(r \cdot N \cdot d)$ FLOPs  
  - MLP forward: $O(N^2 \cdot d)$ FLOPs
  - MLP backward: $O(N^2 \cdot d)$ FLOPs
- Track cumulative FLOPs: $\text{FLOPs}(t) = t \cdot \text{FLOPs}_{\text{per\_iter}}$
- Average over $N_{\text{runs}}$ initializations
- Optional: Detect stepwise drops, analyze scaling laws  
**Output**: `r_vals[4]` (includes MLP), `flops[4, n_points]`, `loss_mean[4, n_points]`, `loss_std[4, n_points]`, `flops_per_iter[4]`, `drop_indices[4, n_drops]` (optional), `scaling_exponents[4]` (optional)  
**FLOPs**: $\sim 4 \times 5 \times T \times (C_{\text{forward}} + C_{\text{backward}}) \approx 10^{15}$ FLOPs total (scales as $O(n_{\text{arch}} \cdot N_{\text{runs}} \cdot T \cdot \text{FLOPs}_{\text{per\_iter}})$)  
**Scaling law**: Loss vs FLOPs: $L(\text{FLOPs}) \sim \text{FLOPs}^{-\alpha}$; RF-LR should show $\alpha_{\text{RF-LR}} > \alpha_{\text{MLP}}$ (better scaling)

---

### Execution Order (by Cost)

**Immediate** (< 1 hour): Plot 5, Plot 8  
**Short** (1-6 hours): Plot 1, Plot 6  
**Medium** (1 day): Plot 2, Plot 4, Plot 7, Plot 9 (theoretical)  
**Long** (2-3 days): Plot 3, Plot 9 (empirical), Plot 10

**Total**: ~1-2 weeks (parallelizable) | **Total FLOPs**: $\sim 10^{15}$ FLOPs (dominated by Plot 10 training)

---

## FLOPs Computation Guide

### General FLOPs Counting Rules

1. **Matrix-vector multiplication**: $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$ → $m \cdot n$ FLOPs
2. **Matrix-matrix multiplication**: $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$ → $m \cdot n \cdot p$ FLOPs
3. **Element-wise operations**: $O(n)$ FLOPs for $n$ elements
4. **Transcendental functions** (exp, log, sin, arccos): Count as 10-50 FLOPs each (implementation-dependent)
5. **Eigendecomposition**: $O(n^3)$ FLOPs for $n \times n$ matrix

### FLOPs for Key Operations

**NTK computation** (3-layer, rank $r$):
- Kernel evaluation: $O(r)$ FLOPs (dominated by arccos, sqrt operations)
- Full Gram matrix ($n \times n$): $O(n^2 \cdot r)$ FLOPs

**Fisher distribution sampling**:
- Inverse transform or rejection sampling: $O(r)$ FLOPs per sample
- Hypergeometric function evaluation: $O(r)$ FLOPs

**Kibble distribution sampling**:
- Chi-square sampling: $O(r)$ FLOPs per sample
- Bessel function evaluation: $O(r)$ FLOPs

**Spherical harmonic expansion**:
- Gegenbauer polynomial evaluation: $O(k \cdot d)$ FLOPs per $k$
- Full expansion to $k_{\max}$: $O(k_{\max}^2 \cdot d)$ FLOPs

### Tracking FLOPs in Practice

- Use profilers (e.g., `torch.profiler`, `line_profiler`) for empirical counts
- For theoretical estimates, count operations manually
- Store FLOPs counts alongside all computation results
- Include FLOPs in metadata for reproducibility

---

## Scaling Laws Analysis

For plots involving computational scaling, compute and store scaling law exponents:

### Power Law Fits
For any quantity $Q$ that scales with parameter $P$ as $Q \sim P^{\alpha}$, compute:
- **Exponent $\alpha$**: Via log-log linear regression: $\log Q = \alpha \log P + \text{const}$
- **Confidence intervals**: Bootstrap or analytical error bars on $\alpha$
- **Goodness of fit**: $R^2$ or residual analysis

### Key Scaling Laws to Extract

**Plot 1**: Concentration probability vs rank
- $\mathbb{P}(|W-1| \geq \epsilon) \sim \exp(-r\epsilon^2)$ (exponential, not power law)
- Extract: Decay rate $\beta = \epsilon^2/8$ from fit

**Plot 2**: NTK variance vs rank  
- $\text{Var}(\hat{\Theta}^{(2)}_r) \sim r^{-\alpha}$ where $\alpha = 1$ (theoretical: $O(1/r)$)
- Extract: $\alpha$ from log-log fit of variance vs $r$

**Plot 4**: Eigenvalue scaling
- Spike: $\lambda_{\text{spike}} \sim n^{\alpha}$ where $\alpha = 1$ (theoretical: $O(n)$)
- Bulk width: $\Delta\lambda \sim n^{\alpha}$ where $\alpha = 0$ (theoretical: $O(1)$)
- Extract: Exponents from fits

**Plot 9**: Concentration vs parameters
- RF-LR: $\sigma \sim P^{-\alpha}$ where $\alpha = 2$ (if $r \propto \sqrt{P}$)
- MLP: $\sigma \sim P^{-\alpha}$ where $\alpha = 1/2$
- Extract: Both exponents, show exponential advantage

**Plot 10**: Loss vs FLOPs (neural scaling laws)
- $L(\text{FLOPs}) \sim \text{FLOPs}^{-\alpha}$ (power law decay)
- Extract: $\alpha_{\text{RF-LR}}$ vs $\alpha_{\text{MLP}}$, should show $\alpha_{\text{RF-LR}} > \alpha_{\text{MLP}}$
- Also fit: $L(N) \sim N^{-\beta}$ (loss vs model size) and $L(D) \sim D^{-\gamma}$ (loss vs data)

### Output Format for Scaling Laws

Each relevant plot should include in its `.npz` file:
- `scaling_exponents`: dict with fitted exponents (e.g., `{'alpha': 1.0, 'alpha_std': 0.05}`)
- `scaling_fits`: dict with fit parameters (slope, intercept, $R^2$)
- `scaling_confidence`: confidence intervals or bootstrap samples

---

### Plot 1: Rank-Driven Concentration

**Computation Task**: Monte Carlo estimation of $\mathbb{P}(|W-1| \geq \epsilon)$ for different ranks and thresholds

**Parameters to vary**:
- Rank $r \in \{5, 6, 7, \ldots, 200\}$ (or log-spaced: `np.logspace(0.7, 2.3, 50)`)
- Thresholds $\epsilon \in \{0.1, 0.2, 0.5, 1.0\}$
- Number of Monte Carlo samples: $N_{\text{MC}} = 10^5$ (or higher for better accuracy)

**What to compute**:
1. For each $(r, \epsilon)$ pair:
   - Sample $N_{\text{MC}}$ pairs of vectors $(x, y) \sim \mathcal{N}(0, I_r)$
   - Compute $W = \|x\|\|y\|/r$ for each pair
   - Count fraction where $|W-1| \geq \epsilon$
   - Store: `prob_empirical[r, eps] = fraction`
   - Store: `prob_std[r, eps] = std_error` (for error bars)

2. Theoretical bounds:
   - For each $(r, \epsilon)$: `prob_theoretical[r, eps] = 4 * exp(-r * eps^2 / 8)`
   - For large deviations: `prob_theoretical_large[r, eps] = c1 * exp(-c2 * r * eps)`

**Output format**: 
- `data/plot1_rank_concentration.npz` containing:
  - `r_vals`: array of rank values
  - `epsilons`: array of epsilon values
  - `prob_empirical`: 2D array [n_ranks, n_epsilons]
  - `prob_std`: 2D array [n_ranks, n_epsilons] (standard errors)
  - `prob_theoretical`: 2D array [n_ranks, n_epsilons]
  - `metadata`: dict with `n_mc_samples`, `random_seed`, etc.

**Computational cost**: Moderate—$O(N_{\text{MC}} \times n_r \times n_\epsilon)$ where $n_r \approx 50$, $n_\epsilon = 4$

---

### Plot 2: Three-Layer NTK Concentration

**Computation Task**: Sample empirical NTK values from Fisher-Kibble distributions and compute statistics

**Parameters to vary**:
- Rank $r \in \{10, 30, 100\}$
- Correlation $\rho \in [-1, 1]$ (dense grid: 100-200 points)
- Number of samples per $(\rho, r)$: $N_{\text{MC}} = 10^4$

**What to compute**:
1. For each $(r, \rho)$ pair:
   - Sample $N_{\text{MC}}$ empirical correlations $\hat{\rho}_r \sim \text{Fisher}(\rho, r)$
   - Sample $N_{\text{MC}}$ norm products $w_r$ from Kibble distribution
   - Compute empirical NTK: $\hat{\Theta}^{(2)}_r = \Psi(\hat{\rho}_r, w_r)$ where
     \[
     \Psi(u, w) = \Theta^{(1)}(u)\left(1-\frac{\arccos(u)}{\pi}\right) + w \cdot \Sigma^{(1)}(u) + 1
     \]
   - Store: `ntk_mean[r, rho_idx] = mean(empirical_NTK)`
   - Store: `ntk_std[r, rho_idx] = std(empirical_NTK)`

2. Deterministic limit:
   - For each $\rho$: `ntk_deterministic[rho_idx] = K_infty(rho)`
   - Where $K_\infty(\rho) = \Theta^{(1)}(\rho)(1-\arccos(\rho)/\pi) + \Sigma^{(1)}(\rho) + 1$

**Output format**:
- `data/plot2_ntk_concentration.npz` containing:
  - `r_vals`: array [10, 30, 100]
  - `rho_vals`: array of correlation values
  - `ntk_mean`: 3D array [n_ranks, n_rho] 
  - `ntk_std`: 3D array [n_ranks, n_rho]
  - `ntk_deterministic`: 1D array [n_rho]
  - `metadata`: dict with formulas, random seeds, etc.

**Computational cost**: High—requires sampling from Fisher and Kibble distributions, computing NTK for each sample

**Key functions needed**:
- `sample_fisher_correlation(rho_true, r, n_samples)`: Sample from Fisher distribution
- `sample_kibble_norms(r, rho, n_samples)`: Sample norm products from Kibble distribution
- `compute_ntk_3layer(rho_hat, w_r)`: Compute empirical NTK from sampled values

---

### Plot 3: Spectral Decay (RKHS Equivalence)

**Computation Task**: Compute eigenvalues of kernel's spherical harmonic expansion

**Parameters to vary**:
- Depths $L \in \{1, 2, 3, 5, 10\}$
- Spherical harmonic index $k \in \{1, 2, \ldots, k_{\max}\}$ where $k_{\max} = 1000$
- Input dimension $d$ (e.g., $d = 10$)

**What to compute**:
1. For each depth $L$:
   - Compute kernel function $\tilde{\Theta}^{(L)}(\rho)$ (mean NTK or deterministic limit)
   - Expand in spherical harmonics on $\mathbb{S}^{d-1}$:
     \[
     \tilde{\Theta}^{(L)}(\rho) = \sum_{k=0}^\infty \mu_k^{(L)} P_k^{(d)}(\rho)
     \]
   - Extract eigenvalues $\mu_k^{(L)}$ for $k = 1, \ldots, k_{\max}$
   - Store: `eigenvalues[L_idx, k] = mu_k^{(L)}`

2. Theoretical reference:
   - `eigenvalues_reference[k] = k^{-d}` (dominant term)

**Output format**:
- `data/plot3_spectral_decay.npz` containing:
  - `depths`: array [1, 2, 3, 5, 10]
  - `k_vals`: array [1, 2, ..., k_max]
  - `eigenvalues`: 2D array [n_depths, k_max]
  - `eigenvalues_reference`: 1D array [k_max]
  - `d`: input dimension
  - `metadata`: dict with expansion method, Gegenbauer polynomial details

**Computational cost**: Very high—requires computing Gegenbauer polynomial expansions or numerical integration

**Key functions needed**:
- `compute_kernel_function(L, rho)`: Compute $\tilde{\Theta}^{(L)}(\rho)$
- `spherical_harmonic_expansion(kernel_func, d, k_max)`: Expand kernel in spherical harmonics
- `gegenbauer_coefficients(k, d)`: Compute Gegenbauer polynomial coefficients

**Note**: This is the most computationally intensive plot. May require:
- Numerical integration over sphere
- Gegenbauer polynomial evaluation
- FFT-based methods for zonal kernels
- Or use known closed forms if available

---

### Plot 4: Marchenko-Pastur Spectrum

**Computation Task**: Compute eigenvalues of NTK Gram matrix for different data/rank ratios

**Parameters to vary**:
- Number of data points $n \in \{500, 1000, 2000\}$ (or fixed $n=1000$)
- Rank ratios $\gamma_{\text{ratio}} = n/r \in \{0.5, 1.0, 2.0\}$ → ranks $r \in \{2000, 1000, 500\}$
- Input dimension $d$ (scales with $r$: $d \asymp r$)
- Number of random initializations: $N_{\text{init}} = 10$ (to average over randomness)

**What to compute**:
1. For each $(n, r, \gamma_{\text{ratio}})$:
   - Generate data: $X \in \mathbb{R}^{n \times d}$ (e.g., Gaussian with covariance $\Sigma_d$)
   - For each initialization:
     - Initialize RF-LR network with rank $r$
     - Compute NTK Gram matrix: $\mathbf{M}_{ij} = \Theta^{(2)}(X_i, X_j)$
     - Compute eigenvalues: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$
     - Store eigenvalues
   - Average over initializations: `eigenvalues_mean = mean(eigenvalues, axis=0)`
   - Store: `eigenvalues_all[r_idx, gamma_idx, :] = eigenvalues_mean`

2. Identify spike:
   - `lambda_spike[r_idx, gamma_idx] = eigenvalues_mean[0]` (largest eigenvalue)

3. Theoretical MP density:
   - For each $\gamma_{\text{ratio}}$: compute MP density parameters
   - Store: `mp_density_params[gamma_idx] = {beta, gamma, support}`

**Output format**:
- `data/plot4_marchenko_pastur.npz` containing:
  - `n`: number of data points
  - `gamma_ratios`: array [0.5, 1.0, 2.0]
  - `r_vals`: corresponding ranks
  - `eigenvalues_all`: 3D array [n_ratios, n_init, n] (or averaged: [n_ratios, n])
  - `lambda_spike`: 2D array [n_ratios, n_init] (or 1D if averaged)
  - `mp_density_params`: dict with theoretical parameters
  - `metadata`: dict with data generation details, random seeds

**Computational cost**: Very high—requires computing full NTK Gram matrix ($O(n^2)$ kernel evaluations) for multiple initializations

**Key functions needed**:
- `generate_data(n, d, covariance)`: Generate input data
- `compute_ntk_gram_matrix(X, r, network_params)`: Compute full NTK matrix
- `compute_mp_density(gamma_ratio, beta, gamma)`: Compute theoretical MP density

**Optimization**: Can use random matrix theory to directly sample eigenvalues without full matrix computation (if applicable)

---

### Plot 5: Computational Efficiency

**Computation Task**: None—this is purely theoretical

**What to compute**: Nothing. Just evaluate formulas:
- RF-LR: `params_rflr = r * N` for $N \in [10, 1000]$ (log-spaced)
- MLP: `params_mlp = N^2`

**Output format**: Can be computed on-the-fly or precomputed for consistency
- `data/plot5_efficiency.npz` containing:
  - `N_vals`: array of width values
  - `r_vals`: array [10, 30, 100]
  - `params_rflr`: 2D array [n_ranks, n_N]
  - `params_mlp`: 1D array [n_N]

---

### Plot 6: Fisher-Kibble Decoupling

**Computation Task**: Sample from Fisher and Kibble distributions to show independence

**Parameters to vary**:
- Rank $r = 30$ (or multiple values)
- True correlations $\rho \in \{0, 0.5, 0.9\}$
- Number of samples: $N_{\text{MC}} = 10^5$

**What to compute**:
1. For each $\rho$:
   - Sample $N_{\text{MC}}$ pairs $(\hat{\rho}_r, w_r)$:
     - $\hat{\rho}_r \sim \text{Fisher}(\rho, r)$
     - $w_r$ from Kibble distribution (independent)
   - Store: `rho_samples[rho_idx, :] = rho_hat_samples`
   - Store: `w_samples[rho_idx, :] = w_samples`

2. Compute marginals (for histograms):
   - `rho_hist[rho_idx, bins]`: histogram of $\hat{\rho}_r$
   - `w_hist[rho_idx, bins]`: histogram of $w_r$

3. Test independence:
   - `correlation_coef[rho_idx] = corr(rho_samples, w_samples)` (should be ~0)

**Output format**:
- `data/plot6_fisher_kibble.npz` containing:
  - `r`: rank value
  - `rho_true_vals`: array [0, 0.5, 0.9]
  - `rho_samples`: 2D array [n_rho, n_samples]
  - `w_samples`: 2D array [n_rho, n_samples]
  - `rho_hist`: 2D array [n_rho, n_bins] with bin edges
  - `w_hist`: 2D array [n_rho, n_bins] with bin edges
  - `correlation_coef`: 1D array [n_rho] (should be near zero)
  - `metadata`: dict with sampling details

**Computational cost**: Moderate—sampling from distributions

**Key functions needed**:
- `sample_fisher_distribution(rho_true, r, n_samples)`: Sample from Fisher
- `sample_kibble_distribution(r, rho, n_samples)`: Sample from Kibble

---

### Plot 7: Mean NTK Puiseux Expansion

**Computation Task**: Compute mean NTK by integrating over Fisher-Kibble distributions

**Parameters to vary**:
- Rank $r = 30$ (or multiple values)
- Correlation $\rho \in [0.9, 1.0]$ (dense grid near boundary: 1000 points)
- Integration method: numerical integration or Monte Carlo

**What to compute**:
1. For each $\rho$:
   - Compute mean NTK: $\tilde{\Theta}^{(2)}(\rho) = \mathbb{E}[\hat{\Theta}^{(2)}_r]$ over Fisher-Kibble
   - This requires integrating:
     \[
     \tilde{\Theta}^{(2)}(\rho) = \int \int \Psi(\hat{\rho}_r, w_r) \, p_{\text{Fisher}}(\hat{\rho}_r | \rho, r) \, p_{\text{Kibble}}(w_r | r, \rho) \, d\hat{\rho}_r \, dw_r
     \]
   - Store: `mean_ntk[rho_idx] = integrated_value`

2. Deterministic limit:
   - `deterministic_ntk[rho_idx] = K_infty(rho)`

3. Difference and Puiseux analysis:
   - `difference[rho_idx] = mean_ntk[rho_idx] - deterministic_ntk[rho_idx]`
   - `t_vals = 1 - rho_vals`
   - Fit: `difference ~ c * t^{1/2} + higher_order` to verify leading term

**Output format**:
- `data/plot7_puiseux_expansion.npz` containing:
  - `r`: rank value
  - `rho_vals`: array (dense near 1)
  - `mean_ntk`: 1D array
  - `deterministic_ntk`: 1D array
  - `difference`: 1D array
  - `t_vals`: 1D array (1 - rho_vals)
  - `puiseux_coefficient`: fitted $c_1(r)$ coefficient
  - `metadata`: dict with integration method, accuracy

**Computational cost**: Very high—requires high-dimensional numerical integration

**Key functions needed**:
- `integrate_fisher_kibble_expectation(rho, r, kernel_func)`: Numerical integration
- `compute_puiseux_coefficient(difference, t_vals)`: Fit $t^{1/2}$ term

**Alternative**: Use Monte Carlo integration with many samples ($N_{\text{MC}} = 10^6$ or more)

---

### Plot 8: NTK Recursion Visualization

**Computation Task**: Compute NTK values at different depths using recursion

**Parameters to vary**:
- Depths $L \in \{1, 2, \ldots, 10\}$
- Correlations $\rho \in \{-1, -0.8, \ldots, 1\}$ (or specific values like $\{0, 0.5, 0.9\}$)

**What to compute**:
1. For each $(L, \rho)$:
   - Compute NTK using recursion: $\Theta^{(L)}(\rho)$
   - Decompose into terms (bias vs fresh basis) if needed
   - Store: `ntk_values[L_idx, rho_idx] = Theta^{(L)}(rho)`

2. Term breakdown (optional):
   - For each $L$: decompose into $2L$ terms from Corollary 2.3
   - Store: `bias_terms[L_idx, rho_idx]` and `fresh_basis_terms[L_idx, rho_idx]`

**Output format**:
- `data/plot8_ntk_recursion.npz` containing:
  - `depths`: array [1, 2, ..., 10]
  - `rho_vals`: array of correlation values
  - `ntk_values`: 2D array [n_depths, n_rho]
  - `term_breakdown`: optional 3D array [n_depths, n_rho, 2] (bias, fresh_basis)
  - `metadata`: dict with recursion formulas

**Computational cost**: Low—just evaluating closed-form or recursive formulas

**Key functions needed**:
- `compute_ntk_recursion(L, rho)`: Compute NTK using Theorem 2.1 recursion
- `decompose_ntk_terms(L, rho)`: Break down into $2L$ terms

---

### Plot 9: Concentration Rate Comparison

**Computation Task**: Compute NTK standard deviation for RF-LR vs MLP at different parameter counts

**Parameters to vary**:
- Parameter counts: $P \in [10^2, 10^6]$ (log-spaced, 50 points)
- For RF-LR: vary rank $r$ such that $P = r \cdot N$ (fix $N$ or vary both)
- For MLP: vary width $N$ such that $P = N^2$ (assuming square layers)

**What to compute**:
1. For RF-LR:
   - For each parameter count $P$:
     - Choose $(r, N)$ such that $P = r \cdot N$ (e.g., fix $r=30$, vary $N$)
     - Compute NTK standard deviation: $\sigma_{\text{RF-LR}}(P) = O(1/r^2)$
     - Since $r \propto \sqrt{P}$ (if $N$ fixed) or $r \propto P$ (if $N$ fixed), get scaling
     - Store: `std_rflr[P_idx] = theoretical_std(P)`

2. For MLP:
   - For each parameter count $P$:
     - Width $N = \sqrt{P}$
     - Compute NTK standard deviation: $\sigma_{\text{MLP}}(P) = O(1/N) = O(1/\sqrt{P})$
     - Store: `std_mlp[P_idx] = theoretical_std(P)`

3. Empirical validation (optional):
   - For selected $P$ values, run Monte Carlo to verify theoretical scaling

**Output format**:
- `data/plot9_concentration_comparison.npz` containing:
  - `param_counts`: array of parameter counts (log-spaced)
  - `std_rflr`: 1D array [n_P] (theoretical or empirical)
  - `std_mlp`: 1D array [n_P] (theoretical or empirical)
  - `std_rflr_empirical`: optional 1D array (if Monte Carlo done)
  - `std_mlp_empirical`: optional 1D array
  - `metadata`: dict with scaling assumptions, formulas

**Computational cost**: Low if theoretical, high if empirical validation included

**Key functions needed**:
- `compute_ntk_std_rflr(param_count, r, N)`: Compute RF-LR std (theoretical: $1/r^2$)
- `compute_ntk_std_mlp(param_count, N)`: Compute MLP std (theoretical: $1/N$)

---

### Plot 10: Training Dynamics (FLOPs-based)

**Computation Task**: Run actual training experiments and record loss curves vs FLOPs (not iterations)

**Parameters to vary**:
- Architecture: RF-LR with ranks $r \in \{10, 30, 100\}$ vs MLP baseline
- Dataset: Multi-index target function (or synthetic regression task)
- Training: Until convergence or fixed FLOP budget (e.g., $10^{12}$ FLOPs)
- Number of runs: $N_{\text{runs}} = 5$ (to average over initializations)
- Batch size: $B$ (affects FLOPs per iteration)

**FLOPs per iteration calculation**:

For **RF-LR** (3-layer, rank $r$, width $N$, input dim $d$, batch size $B$):
- Forward pass: $B \cdot (r \cdot d + r \cdot N + N) \approx B \cdot r \cdot N$ FLOPs
- Backward pass: $B \cdot (r \cdot N + r \cdot d) \approx B \cdot r \cdot N$ FLOPs
- **Total per iteration**: $\text{FLOPs}_{\text{RF-LR}} \approx 2B \cdot r \cdot N$ FLOPs

For **MLP** (2-layer, width $N$, input dim $d$, batch size $B$):
- Forward pass: $B \cdot (N \cdot d + N^2) \approx B \cdot N^2$ FLOPs
- Backward pass: $B \cdot (N^2 + N \cdot d) \approx B \cdot N^2$ FLOPs
- **Total per iteration**: $\text{FLOPs}_{\text{MLP}} \approx 2B \cdot N^2$ FLOPs

**What to compute**:
1. For each architecture and rank:
   - Initialize network
   - Train with gradient descent (or specified optimizer)
   - At each iteration $t$:
     - Compute FLOPs used: $\text{FLOPs}(t) = t \cdot \text{FLOPs}_{\text{per\_iter}}$
     - Record loss: `loss[t]`
     - Record cumulative FLOPs: `flops[t]`
   - Repeat for $N_{\text{runs}}$ different initializations
   - Average: `loss_mean[r_idx, :] = mean(loss_runs, axis=0)`
   - Average: `flops_mean[r_idx, :] = mean(flops_runs, axis=0)`
   - Store: `loss_std[r_idx, :] = std(loss_runs, axis=0)` (for error bars)

2. Interpolate to common FLOP grid:
   - Create log-spaced FLOP grid: `flops_grid = np.logspace(log10(min_flops), log10(max_flops), 1000)`
   - Interpolate loss curves: `loss_interp[r_idx, :] = interp1d(flops_mean[r_idx], loss_mean[r_idx])(flops_grid)`

3. Scaling law analysis:
   - Fit power law: $\log L = -\alpha \log(\text{FLOPs}) + \text{const}$
   - Extract exponent: `scaling_exponent[r_idx] = alpha`
   - Compute confidence intervals via bootstrap

4. Identify stepwise drops (optional):
   - Detect discrete drops in loss (if applicable)
   - Store: `drop_flops[r_idx, :]` and `drop_sizes[r_idx, :]`

**Output format**:
- `data/plot10_training_dynamics.npz` containing:
  - `r_vals`: array [10, 30, 100] (and MLP)
  - `flops_grid`: 1D array [n_points] (common FLOP grid for all architectures)
  - `loss_mean`: 2D array [n_architectures, n_points] (interpolated to common grid)
  - `loss_std`: 2D array [n_architectures, n_points]
  - `flops_per_iter`: 1D array [n_architectures] (FLOPs per iteration for each)
  - `scaling_exponents`: 1D array [n_architectures] (power law exponents $\alpha$)
  - `scaling_exponent_std`: 1D array [n_architectures] (std errors on exponents)
  - `scaling_R2`: 1D array [n_architectures] (goodness of fit)
  - `drop_flops`: optional 2D array (step locations in FLOPs)
  - `drop_sizes`: optional 2D array (step magnitudes)
  - `metadata`: dict with training hyperparameters, dataset details, random seeds, batch size

**Computational cost**: Very high—requires full training runs

**Key functions needed**:
- `train_rflr_network(r, dataset, max_flops)`: Train RF-LR network, return (loss, flops) arrays
- `train_mlp_network(width, dataset, max_flops)`: Train MLP baseline, return (loss, flops) arrays
- `compute_flops_per_iter(architecture, r, N, d, B)`: Compute FLOPs per iteration
- `fit_scaling_law(flops, loss)`: Fit $L \sim \text{FLOPs}^{-\alpha}$, return $\alpha$, std, $R^2$
- `detect_stepwise_drops(loss_curve, flops_curve)`: Identify staircase pattern

---

## Summary: Computation Priorities

### Quick Wins (Low computational cost, high impact):
1. **Plot 1**: Rank concentration (Monte Carlo sampling)
2. **Plot 5**: Efficiency (theoretical formulas)
3. **Plot 8**: NTK recursion (formula evaluation)

### Moderate Effort:
4. **Plot 6**: Fisher-Kibble decoupling (distribution sampling)
5. **Plot 9**: Concentration comparison (theoretical + optional validation)

### High Computational Cost (Require careful planning):
6. **Plot 2**: Three-layer NTK concentration (Fisher-Kibble sampling + NTK computation)
7. **Plot 4**: Marchenko-Pastur (full NTK Gram matrix computation)
8. **Plot 7**: Puiseux expansion (high-dimensional integration)
9. **Plot 10**: Training dynamics (full training runs)

### Very High Computational Cost (May need approximations):
10. **Plot 3**: Spectral decay (spherical harmonic expansion—most complex)

---

## Data Storage Recommendations

### File Organization:
```
refs/paper/
├── data/
│   ├── plot1_rank_concentration.npz
│   ├── plot2_ntk_concentration.npz
│   ├── plot3_spectral_decay.npz
│   ├── plot4_marchenko_pastur.npz
│   ├── plot5_efficiency.npz
│   ├── plot6_fisher_kibble.npz
│   ├── plot7_puiseux_expansion.npz
│   ├── plot8_ntk_recursion.npz
│   ├── plot9_concentration_comparison.npz
│   └── plot10_training_dynamics.npz
├── scripts/
│   ├── compute_plot1.py
│   ├── compute_plot2.py
│   └── ... (computation scripts)
├── plots/
│   ├── plot_plot1.py
│   ├── plot_plot2.py
│   └── ... (plotting scripts)
└── figures/
    └── (output PDF files)
```

### Metadata Standards:
Each `.npz` file should include a `metadata` dictionary with:
- `computation_date`: timestamp
- `random_seed`: for reproducibility
- `parameters`: dict of all parameters used
- `formulas`: relevant theoretical formulas
- `version`: data format version
- `notes`: any important implementation details

---

## Key Messages to Emphasize

1. **Exponential concentration**: Not just polynomial improvement
2. **RKHS preservation**: Depth doesn't help (surprising result)
3. **Computational efficiency**: Quadratic parameter savings
4. **Spectral structure**: Spike-bulk Marchenko-Pastur law
5. **Fisher-Kibble decoupling**: Fundamental statistical structure

---

## Implementation Roadmap

### Phase 1: Core Theoretical Validation (High Priority)
1. **Plot 1**: Rank-driven concentration—validate Theorem 3.2 with exponential decay
2. **Plot 3**: Spectral decay RKHS equivalence—prove Corollary 3.5 visually
3. **Plot 4**: Marchenko-Pastur spectrum—demonstrate Theorem 4.2 spike-bulk structure
4. **Plot 9**: Concentration rate comparison—highlight $O(1/r^2)$ vs $O(1/N)$ advantage

### Phase 2: Supporting Visualizations (Medium Priority)
5. **Plot 2**: Three-layer NTK concentration—show Fisher-Kibble decoupling
6. **Plot 5**: Computational efficiency—demonstrate parameter budget savings
7. **Plot 6**: Fisher-Kibble decoupling—visualize independence structure
8. **Plot 7**: Mean NTK Puiseux expansion—show RKHS preservation mechanism

### Phase 3: Additional Insights (Lower Priority)
9. **Plot 8**: NTK recursion visualization—illustrate depth expansion
10. **Plot 10**: Training dynamics—connect theory to practice

### Quality Assurance
- **Theoretical validation**: All numerical constants must match paper derivations
- **Visual clarity**: Each plot should be self-explanatory with proper annotations
- **Publication standards**: Professional typography, consistent color scheme, proper scaling
- **Caption writing**: Each figure needs a caption explaining what it demonstrates and how it relates to theory
- **LaTeX integration**: Figures should be properly referenced and integrated into paper narrative

---

## Generated Figures (paths)

- Plot 1 (Rank-driven concentration): `refs/paper/figures/fig_plot1_rank_concentration.{png,pdf}`
- Plot 2 (Three-layer NTK concentration, approx.): `refs/paper/figures/fig_plot2_ntk_concentration.{png,pdf}`
- Plot 3 (Spectral decay, conceptual): `refs/paper/figures/fig_plot3_spectral_decay.{png,pdf}`
- Plot 4 (MP spectrum, from grid data): `refs/paper/figures/fig_plot4_mp_spectrum.{png,pdf}`
- Plot 5 (Efficiency): `refs/paper/figures/fig_plot5_efficiency.{png,pdf}`
- Plot 6 (Fisher–Kibble decoupling, approx.): `refs/paper/figures/fig_plot6_fisher_kibble.{png,pdf}`
- Plot 7 (Mean NTK vs deterministic, approx. near ρ=1): `refs/paper/figures/fig_plot7_puiseux.{png,pdf}`
- Plot 8 (NTK recursion visualization): `refs/paper/figures/fig_plot8_recursion.{png,pdf}`
- Plot 9 (Concentration vs parameters, theory): `refs/paper/figures/fig_plot9_concentration.{png,pdf}`
- Plot 10 (Training dynamics vs FLOPs, placeholder): `refs/paper/figures/fig_plot10_training.{png,pdf}`

Figure index JSON: `refs/paper/data/figures_index.json` (maps plot names to file paths).

