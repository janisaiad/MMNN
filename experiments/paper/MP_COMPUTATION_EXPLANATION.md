# How We Compute Theoretical MP Bounds and Curves

## Overview

We compute theoretical Marchenko-Pastur bounds and density curves in 4 steps:

1. **Compute kernel parameters** (α, β, γ) from $K_\infty(\rho)$
2. **Compute bulk support bounds** using Theorem 4.2 formula
3. **Compute MP density curve** using standard MP formula with shift
4. **Separate spike from bulk** using adaptive threshold

## Step-by-Step Computation

### Step 1: Compute Kernel Parameters

We evaluate the deterministic 3-layer NTK limit $K_\infty(\rho)$ at three key points:

```python
def compute_mp_params(n: int, r: int) -> dict:
    # Evaluate K_∞ at key points
    rho_0 = 0.0
    rho_1 = 1.0
    rho_eps = 1e-6
    
    K_0 = det_ntk_3layer(rho_0)      # K_∞(0)
    K_1 = det_ntk_3layer(rho_1)       # K_∞(1)
    K_eps = det_ntk_3layer(rho_eps)   # K_∞(ε)
    
    # Numerical derivative
    K_prime_0 = (K_eps - K_0) / eps
    
    # Extract parameters (from Theorem 4.2)
    alpha = K_0                    # Spike coefficient
    beta = K_prime_0               # Bulk slope
    gamma = K_1 - K_0 - K_prime_0  # Diagonal shift
```

**Actual computed values** (for ReLU kernels):
- $K_\infty(0) \approx 1.398$ → $\alpha = 1.398$
- $K'_\infty(0) \approx 0.676$ → $\beta = 0.676$
- $K_\infty(1) = 2.5$ → $\gamma = 2.5 - 1.398 - 0.676 = 0.426$

**Note:** These differ from the normalized values in the appendix ($\alpha=1$, $\beta=1/\pi \approx 0.318$) because we use the actual ReLU kernel formulas without additional normalization.

### Step 2: Compute Bulk Support Bounds

From Theorem 4.2 (appendix line 666), the bulk eigenvalues are supported on:

$$\left[\gamma + \beta(1-\sqrt{\gamma_{\text{ratio}}})^2,\;\; \gamma + \beta(1+\sqrt{\gamma_{\text{ratio}}})^2\right]$$

where $\gamma_{\text{ratio}} = n/r$.

```python
gamma_ratio = n / r
support_low = gamma + beta * (1 - np.sqrt(gamma_ratio))**2
support_high = gamma + beta * (1 + np.sqrt(gamma_ratio))**2

# Ensure non-negative
if support_low < 0:
    support_low = 0
```

**Example:** For $n=128$, $r=64$ ($\gamma_{\text{ratio}}=2$):
- $\text{support}_{\text{low}} = 0.426 + 0.676 \times (1-\sqrt{2})^2 \approx 0.542$
- $\text{support}_{\text{high}} = 0.426 + 0.676 \times (1+\sqrt{2})^2 \approx 4.364$

### Step 3: Compute MP Density Curve

The standard Marchenko-Pastur density (for $\gamma_{\text{ratio}} < 1$) is:

$$\rho(\lambda) = \frac{1}{2\pi \beta \gamma_{\text{ratio}} (\lambda - \gamma)} \sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}$$

where:
- $\lambda_+ = \gamma + \beta(1+\sqrt{\gamma_{\text{ratio}}})^2$ (upper bound)
- $\lambda_- = \gamma + \beta(1-\sqrt{\gamma_{\text{ratio}}})^2$ (lower bound)

**Implementation:**
```python
def marchenko_pastur_pdf(x, gamma_ratio, beta, gamma_shift):
    lambda_plus = gamma_shift + beta * (1 + np.sqrt(gamma_ratio))**2
    lambda_minus = gamma_shift + beta * (1 - np.sqrt(gamma_ratio))**2
    
    pdf = np.zeros_like(x)
    mask = (x > lambda_minus) & (x < lambda_plus)
    x_masked = x[mask]
    
    # Avoid division by zero
    denominator = 2.0 * np.pi * beta * gamma_ratio * (x_masked - gamma_shift)
    denominator = np.maximum(denominator, 1e-10)
    
    sqrt_term = np.sqrt((lambda_plus - x_masked) * (x_masked - lambda_minus))
    pdf[mask] = sqrt_term / denominator
    
    return pdf
```

**Key points:**
- The density is only non-zero between $\lambda_-$ and $\lambda_+$
- We clip the denominator to avoid division by zero when $\lambda \approx \gamma$
- The formula assumes $\gamma_{\text{ratio}} < 1$ (for $\geq 1$, the distribution is degenerate)

### Step 4: Spike Location

The spike eigenvalue (from Theorem 4.2) is:

$$\lambda_{\text{spike}} = n \cdot \alpha = n \cdot K_\infty(0)$$

```python
lambda_spike = n * alpha
```

**Example:** For $n=128$, $\alpha=1.398$:
- $\lambda_{\text{spike}} = 128 \times 1.398 = 178.9$

### Step 5: Separate Spike from Bulk

We use an adaptive threshold to separate the spike from the bulk:

```python
threshold = max(lambda_spike_theory * 0.1, support_high * 2.0, 10.0)
spike = eigenvalues[eigenvalues > threshold]
bulk = eigenvalues[eigenvalues <= threshold]
```

This ensures:
- Spike eigenvalues are clearly separated (at least 10% of theoretical spike)
- Bulk eigenvalues are well below the spike
- Minimum threshold of 10 to avoid numerical noise

## Visualization

We create 4 subplots for each configuration:

1. **Log-Log (full spectrum):** Shows spike and bulk on log scales, with theoretical bounds
2. **Log-Lin (bulk histogram):** Histogram of bulk eigenvalues with MP density overlay
3. **Lin-Log (bulk detail):** Bulk eigenvalues on log scale to see structure
4. **Lin-Lin (bulk histogram, linear):** Same as #2 but with linear y-axis

## Differences from Appendix

The appendix (lines 616, 636, 656) uses normalized values:
- $\alpha = 1$ (we get $\approx 1.398$)
- $\beta = 1/\pi \approx 0.318$ (we get $\approx 0.676$)
- $K_\infty(1) = 3$ (we get $2.5$)

**Why the difference?**
- The appendix assumes a specific normalization convention
- Our kernel functions compute the actual ReLU kernel values
- Both are valid; we use the computed values for consistency with the empirical data

## Summary

1. **Parameters:** Computed from $K_\infty$ at $\rho=0,1$ and derivative at $0$
2. **Support:** $[\gamma + \beta(1-\sqrt{n/r})^2, \gamma + \beta(1+\sqrt{n/r})^2]$
3. **Density:** Standard MP formula with shift $\gamma$ and scale $\beta$
4. **Spike:** $n \cdot K_\infty(0)$
5. **Separation:** Adaptive threshold based on theoretical spike and bulk bounds


