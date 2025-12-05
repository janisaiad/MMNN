# $K_\infty$ Calculation and Data Range Explanation

## How $K_\infty(\rho)$ is Calculated

### Formula

For a 3-layer RF-LR ReLU network, the deterministic NTK limit is:

$$K_\infty(\rho) = \Theta^{(1)}(\rho) \left(1 - \frac{\arccos(\rho)}{\pi}\right) + \Sigma^{(1)}(\rho) + 1$$

where $\rho = \langle x_1, x_2 \rangle$ is the cosine similarity between two normalized inputs.

---

### Building Blocks

#### 1. ReLU Derivative Kernel $\Theta^{(1)}(\rho)$

$$\Theta^{(1)}(\rho) = \frac{1}{2\pi} \left( (\pi - \arccos(\rho)) \rho + \sqrt{1-\rho^2} \right)$$

**Python implementation:**
```python
def relu_kernel_theta1(rho):
    rho = np.clip(rho, -1.0, 1.0)  # we clip to valid range #
    theta = np.arccos(rho)  # we compute angle #
    return (1.0 / (2.0 * np.pi)) * ((np.pi - theta) * rho + np.sqrt(np.maximum(0.0, 1.0 - rho**2)))  # we return derivative kernel #
```

**What it represents**: Correlation of gradients $\nabla_w \text{ReLU}(w^T x_1)$ and $\nabla_w \text{ReLU}(w^T x_2)$.

#### 2. ReLU Base Kernel $\Sigma^{(1)}(\rho)$

$$\Sigma^{(1)}(\rho) = \frac{1}{\pi} \left( \sqrt{1-\rho^2} + \rho(\pi - \arccos(\rho)) \right)$$

**Python implementation:**
```python
def relu_kernel_sigma1(rho):
    rho = np.clip(rho, -1.0, 1.0)  # we clip to valid range #
    theta = np.arccos(rho)  # we compute angle #
    return (1.0 / np.pi) * (np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + rho * (np.pi - theta))  # we return base kernel #
```

**What it represents**: Covariance of pre-activations $\text{ReLU}(w^T x_1)$ and $\text{ReLU}(w^T x_2)$.

#### 3. Angular Damping Factor

$$1 - \frac{\arccos(\rho)}{\pi} = 1 - \frac{\theta}{\pi}$$

Where $\theta \in [0, \pi]$ is the angle between inputs.

- $\rho = 1$ (same input): factor = 1 (full contribution)
- $\rho = 0$ (orthogonal): factor = 0.5  
- $\rho = -1$ (opposite): factor = 0 (no contribution)

#### 4. Bias Term

The constant $+1$ comes from the trainable output bias $c$ in the network.

---

### Full Implementation

```python
def det_ntk_3layer(rho):
    """
    compute deterministic 3-layer ntk limit K_infty(rho)
    """
    rho = np.clip(rho, -1.0, 1.0)  # we clip to valid range #
    theta = np.arccos(rho)  # we compute angle #
    
    # derivative kernel
    theta1 = (1.0 / (2.0 * np.pi)) * ((np.pi - theta) * rho + np.sqrt(np.maximum(0.0, 1.0 - rho**2)))  # we compute theta1 #
    
    # base kernel
    sigma1 = (1.0 / np.pi) * (np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + rho * (np.pi - theta))  # we compute sigma1 #
    
    # 3-layer ntk
    K_infty = theta1 * (1.0 - theta / np.pi) + sigma1 + 1.0  # we compute full kernel #
    
    return K_infty  # we return #
```

---

### Special Values

| $\rho$ | $\theta$ | $\Theta^{(1)}$ | $\Sigma^{(1)}$ | $1 - \theta/\pi$ | $K_\infty(\rho)$ |
|--------|----------|----------------|----------------|------------------|------------------|
| $-1$   | $\pi$    | $0$            | $0$            | $0$              | $1.000$          |
| $0$    | $\pi/2$  | $\frac{1}{2\pi}$ | $\frac{1}{\pi}$ | $0.5$         | $\approx 1.318$  |
| $0.5$  | $\pi/3$  | $\frac{1}{4\pi}$ | $\frac{\pi + \sqrt{3}}{2\pi}$ | $2/3$   | $\approx 1.613$  |
| $1$    | $0$      | $\frac{1}{2}$  | $1$            | $1$              | $2.500$          |

---

## Data Range: $\rho \in [-1, 1]$

### Actual Data Coverage

The NTK-rho data files **DO have the full range** $\rho \in [-1, 1]$:

```bash
$ python check_rho_range.py
Found 343 NTK-rho files

Example: grid_n128_N256_r1024_d1024_ntk_rho.npz
  rho range: [-1.00, 1.00]
  rho shape: (21,)
  first 5: [-1.  -0.9 -0.8 -0.7 -0.6]
  last 5: [0.6 0.7 0.8 0.9 1. ]
```

### How Data is Generated

In `experiments/paper/largescale.py` (line 931):
```python
rho_vals = np.round(np.arange(-1.0, 1.0 + 1e-9, rho_step), 6)  # buckets -1, -0.9, ..., 1
```

With `rho_step = 0.1`, this creates **21 bins** from -1 to 1.

### Why Full Range Matters

1. **Negative correlations** ($\rho < 0$): Anti-correlated inputs, important for understanding kernel behavior across all input relationships

2. **Zero correlation** ($\rho = 0$): Orthogonal inputs, reference point for independence

3. **Positive correlations** ($\rho > 0$): Similar inputs, most common in practice

4. **Boundary behavior** ($\rho \to \pm 1$): Critical for Puiseux expansion analysis near singularities

---

## Verification in Plots

### Plot 2: NTK Concentration

Should show $K_\infty(\rho)$ for **full range** $\rho \in [-1, 1]$:

```python
# load real data with full range
data = np.load("refs/paper/data/grid_*_ntk_rho.npz")
rho_vals = data["rho_vals"]  # should be [-1.0, -0.9, ..., 0.9, 1.0]
ntk_mean = data["ntk_mean"]  # empirical mean
k_infty = det_ntk_3layer(rho_vals)  # deterministic limit

# plot both
plt.plot(rho_vals, k_infty, label="K_∞(ρ)")
plt.plot(rho_vals, ntk_mean, label="Empirical")
plt.xlim(-1, 1)  # full range!
```

### Plot 7: Puiseux Expansion

Focuses on **near-boundary** region $\rho \in [0.9, 1.0]$ to analyze leading term behavior:

```python
# zoom to boundary
rho_vals_boundary = rho_vals[rho_vals >= 0.9]
t = 1.0 - rho_vals_boundary  # distance from boundary
```

This is correct—we only need the upper boundary for Puiseux analysis.

---

## Common Mistakes to Avoid

### ❌ Wrong: Only using $\rho \in [0, 1]$

```python
# BAD: missing negative correlations
rho_vals = np.linspace(0, 1, 11)
```

### ✅ Correct: Full range $\rho \in [-1, 1]$

```python
# GOOD: full correlation spectrum
rho_vals = np.linspace(-1, 1, 21)
```

### ❌ Wrong: Forgetting to clip before arccos

```python
# BAD: can get NaN if rho slightly outside [-1, 1] due to numerical error
theta = np.arccos(rho)
```

### ✅ Correct: Always clip

```python
# GOOD: safe against numerical errors
rho = np.clip(rho, -1.0, 1.0)
theta = np.arccos(rho)
```

---

## Summary

1. **$K_\infty(\rho)$** combines three components:
   - Derivative kernel $\Theta^{(1)}(\rho)$ with angular damping
   - Base kernel $\Sigma^{(1)}(\rho)$  
   - Bias term $+1$

2. **Data range** is correctly $\rho \in [-1, 1]$ with 21 points (step=0.1)

3. **All plots** should use the full range except:
   - Plot 7 (Puiseux): Zooms to $[0.9, 1.0]$ for boundary analysis
   
4. **Python implementation** must clip $\rho$ before arccos to avoid NaN

---

## References

- **`K_INFINITY_CALCULATION.md`**: Detailed mathematical derivation
- **`experiments/paper/largescale.py`**: Data generation code
- **`experiments/paper/plot_all_figures.py`**: Visualization code  
- **`check_rho_range.py`**: Script to verify data coverage

---

**Last updated**: 2025-01-31  
**Status**: ✅ Data has full [-1, 1] range; plots correctly use full range

