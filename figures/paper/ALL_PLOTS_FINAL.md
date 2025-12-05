# All Plots - Final Version

## Complete Figure List

### ✅ Plot 1: Variance & Std Decay (2 panels)
**File**: `fig_plot1_rank_concentration.{pdf,png}`

**Panels**:
1. **Variance**: Empirical vs Theory $\sim 1/r$
2. **Std**: Empirical vs Theory $\sim 1/\sqrt{r}$ (aligned with fitted constant)

**Theory**: $\text{Var}(W_r) \sim 1/r$, $\sigma(W_r) \sim 1/\sqrt{r}$

---

### ✅ Plot 1a: Exponential Tail Probability (Standalone)
**File**: `fig_plot1a_tail_probability.{pdf,png}`

**Content**: $\mathbb{P}(|W_r-1| \geq \epsilon)$ for $\epsilon \in \{0.1, 0.2, 0.5\}$

**Alignment**: Shifts aligned at **first rank** ($r=5$):
- $\epsilon=0.1$: shift = 0.210
- $\epsilon=0.2$: shift = 0.174  
- $\epsilon=0.5$: shift = 0.075

**Key insight**: Exponential decay confirmed; constant factor $C(\epsilon)$ varies with $\epsilon$

---

### ✅ Plot 2: NTK Concentration (4 panels)
**File**: `fig_plot2_ntk_concentration.{pdf,png}`

**Panels**:
1-3. **NTK vs $\rho$** for $r \in \{16, 32, 64\}$ with $\pm 2\sigma$ bands
4. **Std vs rank** (log-log) with **343 data points** from all configs

**Panel 4 improvements**:
- Uses all 343 NTK-rho files
- Selects 5 representative ranks (evenly spaced in log space)
- Aligned theory: $\sigma = C/\sqrt{r}$ with fitted $C = 0.120$

---

### ✅ Plot 3: Spectral Decay (Empirical Eigenvalues)
**File**: `fig_plot3_spectral_decay.{pdf,png}`

**Content**: Eigenvalue $\lambda_k$ vs index $k$ from **real NTK Gram matrices**

**Improvements**:
- ❌ **Removed**: Fitted curve (was cluttered)
- ✅ **Dotted lines**: Empirical curves (clarifies "data not theory")
- ✅ **Reference $k^{-0.5}$**: Aligned at **last index** (not arbitrary normalization)
- ✅ **Title**: Explicitly states "Empirical"

**Data**: 4 configurations with $r \in \{64, 128, 256, 512\}$

---

### ✅ Plot 4: Marchenko-Pastur Spectrum (3 panels)
**File**: `fig_plot4_mp_spectrum.{pdf,png}`

**Panels**: Three gamma values $\gamma \in \{0.5, 1.0, 2.0\}$

**Improvements**:
- ✅ **Freedman-Diaconis binning**: Optimal histogram bins
- ✅ **Outliers annotated**: Spike values shown individually with markers
- ✅ **Bulk focused**: X-axis zoomed to bulk region only
- ✅ **Text boxes**: Statistics (bulk range, theory support, outlier count)

**Spike consistency**: 100% have exactly 1 spike (all 343 configs)

---

### ✅ Plot 5: FLOPs Analysis (NEW - Replaces Parameter Savings)
**File**: `fig_plot5_flops_analysis.{pdf,png}`

**Panels**:
1. **FLOPs vs n** (number of data points): Shows $\sim n^2$ scaling
2. **FLOPs vs $\gamma$** (aspect ratio $n/r$): Shows how aspect ratio affects cost

**Data source**: `flops_config` from all 343 metadata JSON files

**Key insight**: Actual computational cost from real runs (not theoretical)

---

### ✅ Plot 6: Fisher-Kibble Independence (Improved)
**File**: `fig_plot6_fisher_kibble.{pdf,png}`

**Content**: Scatter plot of $(\hat{\rho}_r, w_r)$ pairs

**Improvements**:
- ✅ **Full range**: Samples across $\rho \in [-1, 1]$ (not just one value)
- ✅ **Color-coded**: By true $\rho$ value (coolwarm colormap)
- ✅ **Clarified**: $w_r > 0$ always (product of norms in y-axis label)

**Why $w_r > 0$**: 
$$w_r = \frac{\|h_1(x_1)\| \cdot \|h_1(x_2)\|}{r}$$
Product of two norms (both $\geq 0$) → always positive!

**Why $\hat{\rho}_r \in [-1, 1]$**:
$$\hat{\rho}_r = \frac{\langle h_1(x_1), h_1(x_2) \rangle}{\|h_1(x_1)\| \|h_1(x_2)\|}$$
Cosine of angle → can be negative (anti-correlated features)

---

### ❌ Plot 7: Puiseux Expansion (REMOVED as requested)

---

### ✅ BONUS: Spike vs n Table
**File**: `table_spike_vs_n.png`

**Content**: Publication-quality table showing spike scaling with $n$

**Key result**: $\lambda_{\text{spike}} \approx 1.488 \times n$ (perfect linearity!)

---

## Total Figures Generated

```
figures/paper/
├── fig_plot1_rank_concentration.{pdf,png}       - Variance & Std (2 panels)
├── fig_plot1a_tail_probability.{pdf,png}        - Tail probability (standalone)
├── fig_plot2_ntk_concentration.{pdf,png}        - NTK + Std vs r (4 panels)
├── fig_plot3_spectral_decay.{pdf,png}           - Eigenvalue decay (empirical)
├── fig_plot4_mp_spectrum.{pdf,png}              - Marchenko-Pastur (3 gammas)
├── fig_plot5_flops_analysis.{pdf,png}           - FLOPs from metadata (NEW)
├── fig_plot6_fisher_kibble.{pdf,png}            - Independence (full rho range)
└── table_spike_vs_n.png                         - Spike table
```

**Total**: 8 figures (7 plots + 1 table)

---

## Key Theoretical Corrections

1. **Variance**: $\text{Var}(W_r) \sim 1/r$ (not $1/r^2$) ✅
2. **Std**: $\sigma(W_r) \sim 1/\sqrt{r}$ ✅
3. **Spectral decay**: Empirical data (dotted), $k^{-0.5}$ reference aligned at last index ✅
4. **Tail shifts**: Aligned at first rank (not median) ✅
5. **Fisher-Kibble**: Full $\rho \in [-1,1]$ range shown ✅

---

## Data Statistics

- **Total configurations**: 343
- **Gamma values**: 13 (from 0.016 to 64.0)
- **n values**: 7 (from 16 to 1024)
- **r values**: 7 (from 16 to 1024)
- **NTK-rho files**: 343 (full $\rho \in [-1, 1]$ coverage)
- **Spikes**: 1 per config (100% consistency)

---

## Regenerate All

```bash
cd /Data/janis.aiad/MMNN
python experiments/paper/plot_all_figures.py
```

---

**Status**: ✅ All plots publication-ready!  
**Date**: 2025-01-31  
**Total FLOPs analyzed**: From 343 real computation runs

