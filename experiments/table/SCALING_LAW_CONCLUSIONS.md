# Scaling Law Analysis: Layers (L) vs Frequency Multiplier

## Summary

We analyzed the relationship between the optimal number of layers (L) and the frequency multiplier (freq) across 55 completed training configurations.

## Key Findings

### 1. Best Fitting Model: **Linear Scaling**

**Formula:** `L = 5.161 × freq + 2.553`

**R² = 0.9377** (excellent fit)

This linear relationship explains **93.8%** of the variance in optimal layer counts.

### 2. Alternative Models Tested

| Model | Formula | R² | Interpretation |
|-------|---------|-----|----------------|
| **Linear** | `L = 5.161 × freq + 2.553` | **0.9377** | **BEST FIT** |
| Power Law | `L = 7.410 × freq^0.842` | 0.9344 | Nearly as good |
| Logarithmic | `L = 10.217 × log(freq) + 10.964` | 0.7782 | Moderate fit |
| Toeplitz (simple) | `L = round(freq × 8)` | 0.4583 | Poor fit |

### 3. Toeplitz Structure Analysis

The simple Toeplitz model `L = round(freq × 8)` (assuming baseline of 8 layers at freq×1.0) **only matches 2/7 cases (28.6%)**:

| Frequency | Optimal L | Toeplitz L (freq×8) | Match |
|-----------|-----------|---------------------|-------|
| 0.3 | 5 | 2 | ✗ |
| 0.6 | 5 | 5 | ✓ |
| 1.5 | 8 | 12 | ✗ |
| 2.0 | 12 | 16 | ✗ |
| 3.0 | 24 | 24 | ✓ |
| 5.0 | 24 | 40 | ✗ |
| 7.0 | 40 | 56 | ✗ |

**Observation:** The Toeplitz model consistently **overestimates** the optimal layer count, especially for:
- Low frequencies (0.3, 0.6): predicts too few layers
- High frequencies (5.0, 7.0): predicts too many layers

### 4. Optimal Layer Counts by Frequency

| Frequency | Optimal Layers | Test Error | Tested Layers |
|-----------|----------------|------------|---------------|
| 0.3 | **5** | 1.39×10⁻⁶ | 3, 5 |
| 0.6 | **5** | 9.53×10⁻⁶ | 3, 5, 8 |
| 1.5 | **8** | 6.28×10⁻¹ | 8, 12, 24 |
| 2.0 | **12** | 1.00 | 12, 16, 24 |
| 3.0 | **24** | 1.58 | 16, 24, 40 |
| 5.0 | **24** | 1.92 | 24, 40, 56 |
| 7.0 | **40** | 1.93 | 40, 56, 80 |

## Proposed Scaling Law

### Primary Recommendation: **Linear Scaling**

```
L_optimal ≈ 5.16 × freq + 2.55
```

**Rationale:**
- Highest R² (0.9377)
- Simple and interpretable
- Works well across the entire frequency range

### Alternative: **Modified Toeplitz**

If you want to maintain the Toeplitz structure, consider:

```
L_optimal ≈ round(5.16 × freq + 2.55)
```

This gives:
- freq×0.3 → L ≈ 4 (we found 5)
- freq×0.6 → L ≈ 6 (we found 5)
- freq×1.5 → L ≈ 10 (we found 8)
- freq×2.0 → L ≈ 13 (we found 12)
- freq×3.0 → L ≈ 18 (we found 24)
- freq×5.0 → L ≈ 28 (we found 24)
- freq×7.0 → L ≈ 39 (we found 40)

### Power Law Alternative

For theoretical considerations, the power law `L = 7.41 × freq^0.84` is nearly as good (R² = 0.9344) and suggests a **sub-linear scaling** (exponent < 1).

## Insights

1. **No simple Toeplitz structure**: The simple `L = round(freq × 8)` model doesn't hold. The relationship is more complex.

2. **Linear scaling dominates**: Despite testing multiple models, linear scaling provides the best fit, suggesting a **proportional relationship** between frequency and required depth.

3. **Saturation at high frequencies**: For freq ≥ 3.0, the optimal layer count plateaus around 24-40 layers, suggesting diminishing returns from adding more layers.

4. **Low frequency behavior**: For freq ≤ 0.6, optimal layers are lower (5) than the linear model would predict, suggesting a **minimum depth requirement**.

## Recommendations

1. **Use linear scaling** for predicting optimal layers: `L ≈ 5.16 × freq + 2.55`

2. **Round to nearest integer** for practical implementation

3. **Consider minimum depth**: For very low frequencies (freq < 0.5), use at least L = 5 layers

4. **Consider maximum depth**: For very high frequencies (freq > 5), L = 40-56 layers may be sufficient

5. **For Toeplitz structure**: If you need a Toeplitz-like pattern, use the modified formula `L = round(5.16 × freq + 2.55)` instead of `L = round(freq × 8)`

## Next Steps

1. Test the linear scaling law on new frequency values to validate
2. Investigate why the simple Toeplitz structure doesn't hold (architecture-specific reasons?)
3. Analyze if the scaling law changes with different ranks or other hyperparameters
4. Consider theoretical justification for the linear relationship
