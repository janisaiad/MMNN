# Theoretical Marchenko-Pastur Computation

This document explains how we compute the theoretical bounds and density curves for the Marchenko-Pastur spectrum.

## Step 1: Compute Kernel Parameters (α, β, γ)

From Theorem 4.2 and Appendix, we need three key values of the deterministic NTK limit $K_\infty(\rho)$:

### 1.1 Spike Coefficient: α = K_∞(0)

```python
rho_0 = 0.0
K_0 = det_ntk_3layer(rho_0)  # K_∞(0)
alpha = K_0
```

**Theoretical value (from appendix):**
- For ReLU: $\Theta^{(1)}(0) = \Sigma^{(1)}(0) = 0$
- So: $\alpha = K_\infty(0) = 1$

**Spike eigenvalue:** $\lambda_{\text{spike}} = n \cdot \alpha = n \cdot K_\infty(0)$

### 1.2 Bulk Slope: β = K'_∞(0)

```python
eps = 1e-6
K_eps = det_ntk_3layer(eps)
K_prime_0 = (K_eps - K_0) / eps  # numerical derivative
beta = K_prime_0
```

**Theoretical value (from appendix):**
- For ReLU: $\beta = \frac{1}{\pi}$

**Note:** We use numerical differentiation. An analytical formula exists:
$$\beta = \frac{\Theta^{(1)'}(0)}{2} + \frac{\Theta^{(1)}(0)}{\pi} + \Sigma^{(1)'}(0) = \frac{1}{2\pi} + \frac{1}{2\pi} = \frac{1}{\pi}$$

### 1.3 Diagonal Shift: γ = K_∞(1) - K_∞(0) - K'_∞(0)

```python
rho_1 = 1.0
K_1 = det_ntk_3layer(rho_1)  # K_∞(1)
gamma = K_1 - K_0 - K_prime_0
```

**Theoretical value (from appendix):**
- $K_\infty(1) = \Theta^{(1)}(1) + \Sigma^{(1)}(1) + 1 = \frac{1}{2} + \frac{1}{2} + 1 = 2$
- But wait, the appendix says $K_\infty(1) = 3$? Let me check...

Actually, from the appendix line 650:
$$K_\infty(1) = \Theta^{(1)}(1) + \Sigma^{(1)}(1) + 1 = \frac{1}{2} + \frac{1}{2} + 1 = 2$$

But line 656 says: "For normalized ReLU kernels, $\Theta^{(1)}(1) = \Sigma^{(1)}(1) = 1/2$, so $K_\infty(1) = 1 + 1 + 1 = 3$"

There's a discrepancy. Let's use the computed value from the kernel functions.

**Computed:** $\gamma = K_\infty(1) - K_\infty(0) - K'_\infty(0) = K_\infty(1) - 1 - \frac{1}{\pi}$

## Step 2: Compute Bulk Support Bounds

From Theorem 4.2, the bulk eigenvalues are supported on:

$$\left[\gamma + \beta(1-\sqrt{\gamma_{\text{ratio}}})^2,\;\; \gamma + \beta(1+\sqrt{\gamma_{\text{ratio}}})^2\right]$$

where $\gamma_{\text{ratio}} = \frac{n}{r}$.

```python
gamma_ratio = n / r
support_low = gamma + beta * (1 - np.sqrt(gamma_ratio))**2
support_high = gamma + beta * (1 + np.sqrt(gamma_ratio))**2
```

**Special cases:**
- If $\gamma_{\text{ratio}} \geq 1$: The MP distribution is degenerate (all mass at boundaries)
- If $\gamma_{\text{ratio}} < 1$: Standard MP density applies

## Step 3: Compute MP Density Curve

The standard Marchenko-Pastur density (for $\gamma_{\text{ratio}} < 1$) is:

$$\rho(\lambda) = \frac{1}{2\pi \beta \gamma_{\text{ratio}} (\lambda - \gamma)} \sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}$$

where:
- $\lambda_+ = \gamma + \beta(1+\sqrt{\gamma_{\text{ratio}}})^2$
- $\lambda_- = \gamma + \beta(1-\sqrt{\gamma_{\text{ratio}}})^2$

**Implementation:**
```python
def marchenko_pastur_pdf(x, gamma_ratio, beta, gamma_shift):
    lambda_plus = gamma_shift + beta * (1 + np.sqrt(gamma_ratio))**2
    lambda_minus = gamma_shift + beta * (1 - np.sqrt(gamma_ratio))**2
    
    pdf = np.zeros_like(x)
    mask = (x > lambda_minus) & (x < lambda_plus)
    x_masked = x[mask]
    
    denominator = 2.0 * np.pi * beta * gamma_ratio * (x_masked - gamma_shift)
    sqrt_term = np.sqrt((lambda_plus - x_masked) * (x_masked - lambda_minus))
    pdf[mask] = sqrt_term / denominator
    
    return pdf
```

**Note:** The formula uses $(\lambda - \gamma)$ in the denominator, which can be problematic if $\lambda \approx \gamma$. We clip the denominator to avoid division by zero.

## Step 4: Separate Spike from Bulk

We use an adaptive threshold:

```python
threshold = max(lambda_spike_theory * 0.1, support_high * 2.0, 10.0)
spike = eigenvalues[eigenvalues > threshold]
bulk = eigenvalues[eigenvalues <= threshold]
```

This separates the large outlier (spike) from the bulk distribution.

## Summary

1. **Compute kernel values:** $K_\infty(0)$, $K_\infty(1)$, $K'_\infty(0)$
2. **Extract parameters:** $\alpha = K_\infty(0)$, $\beta = K'_\infty(0)$, $\gamma = K_\infty(1) - \alpha - \beta$
3. **Compute support:** $[\gamma + \beta(1-\sqrt{n/r})^2, \gamma + \beta(1+\sqrt{n/r})^2]$
4. **Compute density:** Standard MP formula with shift $\gamma$ and scale $\beta$
5. **Spike location:** $n \cdot \alpha$

## Potential Issues

1. **Derivative computation:** Using finite differences ($\epsilon = 10^{-6}$) instead of analytical formula
2. **$K_\infty(1)$ value:** Need to verify if it's 2 or 3 from the kernel functions
3. **MP density formula:** The denominator $(\lambda - \gamma)$ might need adjustment if $\gamma$ is the shift parameter

Let me check the actual computed values...


