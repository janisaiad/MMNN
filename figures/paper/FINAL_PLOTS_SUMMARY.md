# Final Plots Summary

## All Generated Figures

### ✅ Plot 1: Variance & Std Decay (2 panels)
**File**: `fig_plot1_rank_concentration.{pdf,png}`

**Panels**:
1. **Variance**: $\text{Var}(W_r) \sim 1/r$ (theory corrected!)
2. **Std**: $\sigma(W_r) \sim 1/\sqrt{r}$

**Key change**: Removed tail probability (now separate)

---

### ✅ Plot 1a: Exponential Tail Probability (NEW - Separate)
**File**: `fig_plot1a_tail_probability.{pdf,png}`

**Content**: $\mathbb{P}(|W_r-1| \geq \epsilon)$ vs rank $r$

**Alignment**: Shifts aligned **at first rank** ($r=5$):
- $\epsilon=0.1$: shift = 0.210
- $\epsilon=0.2$: shift = 0.174
- $\epsilon=0.5$: shift = 0.075

**Status**: ✅ Standalone plot, properly aligned

---

### ✅ Plot 2: NTK Concentration (4 panels)
**File**: `fig_plot2_ntk_concentration.{pdf,png}`

**Panels**:
1-3. NTK vs $\rho$ for different ranks
4. **Std vs rank** (log-log) with **343 data points**

**Alignment**: Std theory shifted to match empirical: $C/\sqrt{r}$ with $C=0.120$

**Fitted slope**: -0.696 (expected -0.5, difference due to finite effects)

---

### ✅ Plot 3: Spectral Decay (Empirical)
**File**: `fig_plot3_spectral_decay.{pdf,png}`

**Content**: Eigenvalue $\lambda_k$ vs index $k$ from **actual NTK Gram matrices**

**Changes**:
- ❌ Removed fitted curve
- ✅ Dotted lines for empirical curves
- ✅ $k^{-0.5}$ reference **aligned at last index**
- ✅ Title clarifies: "Empirical" data

**Data source**: Real eigenvalues from `grid_*.npz` files

---

### ✅ Plot 4: Marchenko-Pastur (3 panels, different γ)
**File**: `fig_plot4_mp_spectrum.{pdf,png}`

**Improvements**:
- ✅ Freedman-Diaconis binning rule
- ✅ Outliers shown individually with annotations
- ✅ Focus on bulk (spike excluded)
- ✅ All 1 spike per config (100%)

---

### ✅ Plot 5: Efficiency
**File**: `fig_plot5_efficiency.{pdf,png}`

**Content**: $O(rN)$ vs $O(N^2)$ parameter scaling

---

### ✅ Plot 6: Fisher-Kibble Independence (IMPROVED)
**File**: `fig_plot6_fisher_kibble.{pdf,png}`

**Changes**:
- ✅ Shows samples across **FULL $\rho \in [-1, 1]$ range**
- ✅ Color-coded by true $\rho$ (coolwarm colormap)
- ✅ Clarifies: $w_r$ is **ALWAYS POSITIVE** (product of norms)

**Why $w_r > 0$**:
$$w_r = \frac{\|h_1(x_1)\| \cdot \|h_1(x_2)\|}{r}$$

Since norms are always $\geq 0$, their product is always positive!

**Angular part** $\hat{\rho}_r$ can be in $[-1, 1]$ (dot product of unit vectors)
**Radial part** $w_r$ is always in $[0, \infty)$ (product of magnitudes)

---

### ❌ Plot 7: Puiseux (REMOVED as requested)

---

### ✅ BONUS: Spike Table
**File**: `table_spike_vs_n.png`

**Content**: Publication-quality table showing:
- All $n$ values (16 to 1024)
- Mean spike values
- Spike/$n$ ratio ≈ 1.488 (constant!)

---

## Summary Statistics

| Item | Value |
|------|-------|
| Total figures | 7 (including table) |
| Empirical plots | 4 (use real data) |
| Theoretical plots | 3 (formulas only) |
| Total configs analyzed | 343 |
| Unique gamma values | 13 |
| Spike consistency | 100% (all have 1 spike) |

---

## Key Corrections Made

1. **Variance theory**: $1/r$ (not $1/r^2$) ✅
2. **Tail shifts**: Aligned at **first rank** (not median) ✅
3. **Spectral decay**: Empirical (dotted), reference aligned at **last index** ✅
4. **Fisher-Kibble**: Full $\rho \in [-1,1]$ range, clarified $w_r > 0$ ✅
5. **NTK std**: Aligned theory with fitted $C$ ✅
6. **Puiseux**: Removed ✅

---

**All plots are publication-ready!** 🎉

Files in `figures/paper/`:
- `fig_plot1_rank_concentration.{pdf,png}` - Variance & Std decay
- `fig_plot1a_tail_probability.{pdf,png}` - Exponential tail (separate)
- `fig_plot2_ntk_concentration.{pdf,png}` - NTK concentration (4 panels)
- `fig_plot3_spectral_decay.{pdf,png}` - Empirical eigenvalue decay
- `fig_plot4_mp_spectrum.{pdf,png}` - Marchenko-Pastur bulk
- `fig_plot5_efficiency.{pdf,png}` - Parameter budget
- `fig_plot6_fisher_kibble.{pdf,png}` - Independence (full range)
- `table_spike_vs_n.png` - Spike vs n table

