# Width and Hyperparameters Explanation

## Key Hyperparameters (Only 2!)

You're absolutely correct! The important parameters are **ratios**, not absolute values:

### 1. **Aspect Ratio**: $\gamma = n/r$
- Ratio of data points to rank
- Controls Marchenko-Pastur spectrum (spike-bulk separation)
- Values in data: $\gamma \in \{0.016, 0.031, ..., 32, 64\}$ (13 unique values)

### 2. **Dimension-Rank Ratio**: $d/r$
- In our data: **$d = r$** (dimension equals rank for all configs)
- This is set by `d_policy="equal_r"` in `largescale.py`

---

## Width $N$ (Network Parameter)

The width $N$ is the **number of neurons** in hidden layers:
- **Not a key hyperparameter** for theoretical analysis
- Should not affect NTK in infinite-width limit ($N \to \infty$)
- In practice: finite $N$ introduces corrections

### Widths Available in Data

```
N ∈ {16, 32, 64, 128, 256, 512, 1024}  (7 values, powers of 2)
```

Total grid: $7 \times 7 \times 7 = 343$ configurations  
- 7 values of $n$
- 7 values of $r$ (determines $\gamma = n/r$)
- 7 values of $N$

---

## Which Width is Used in Each Plot?

### Plot 1: Variance & Std Decay
**Width**: None (theoretical Monte Carlo, no network)

### Plot 1a: Tail Probability
**Width**: None (theoretical Monte Carlo, no network)

### Plot 2: NTK Concentration
**Width**: $N = 64$ (default config)
- Config 1: n=64, N=64, r=16, d=16
- Config 2: n=128, N=64, r=32, d=32
- Config 3: n=256, N=64, r=64, d=64

**Panel 4 (Std vs r)**: Uses all 343 configs (all widths mixed)

### Plot 3: Spectral Decay
**Width**: $N = 64$ (fixed, 4 configs)
- Config 1: n=256, **N=64**, r=64, d=64
- Config 2: n=256, **N=64**, r=128, d=128
- Config 3: n=256, **N=64**, r=256, d=256
- Config 4: n=256, **N=64**, r=512, d=512

### Plot 4: Marchenko-Pastur
**Width**: $N = 64$ (default)
- Uses configs with N=64 and different gamma values

### Plot 5: FLOPs Analysis
**Width**: All widths (aggregated by $n$ and $\gamma$)

### Plot 6: Fisher-Kibble
**Width**: $N = 64$ (default config n=128, N=64, r=32, d=32)

---

## Summary of Choices

Most plots use **$N = 64$** as the default width because:
1. Large enough to be in "wide" regime ($N > d$ for most configs)
2. Not too large (computational cost manageable)
3. Consistent across comparisons

For plots that aggregate data (like Plot 2 panel 4, Plot 5), **all widths** are used, showing that results hold across different $N$ values.

---

## True Hyperparameters for Theory

From a theoretical perspective, in the limit $N \to \infty$:

### Key ratios:
1. **$\gamma = n/r$**: Determines Marchenko-Pastur structure
2. **$d/r$**: Determines feature dimension scaling (we use $d = r$)

### Fixed by data:
- $N$ (width): Should not matter in limit
- $n$ (data size): Sets spike scale ($\lambda_{\text{spike}} \sim n$)
- $r$ (rank): Sets concentration rate ($\sigma \sim 1/\sqrt{r}$)
- $d$ (input dim): Sets spectral decay (if $d/r$ matters)

In our case with $d = r$, we effectively have:
- **1 ratio**: $\gamma = n/r$
- **1 scale**: $r$ (which determines $n$ via $\gamma$)

---

## Recommendation

For clearer presentation, plots could specify:
- "Fixed $N = 64$" in caption
- "All configs have $d = r$" in methods

This clarifies that we're varying $\gamma = n/r$ while keeping $d/r = 1$ constant.

---

**Width used in most plots**: $N = 64$ (fixed)  
**Key hyperparameter**: $\gamma = n/r$ (aspect ratio)  
**Secondary constraint**: $d = r$ (dimension equals rank)

