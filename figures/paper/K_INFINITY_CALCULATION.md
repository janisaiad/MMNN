# How $K_\infty(\rho)$ is Calculated

## Overview

$K_\infty(\rho)$ is the **deterministic Neural Tangent Kernel (NTK) limit** for a 3-layer ReLU network in the infinite-width regime. It represents the limit of the empirical NTK as the rank $r \to \infty$.

---

## Formula

For a 3-layer RF-LR network with ReLU activations:

$$
K_\infty(\rho) = \Theta^{(1)}(\rho) \left(1 - \frac{\arccos(\rho)}{\pi}\right) + \Sigma^{(1)}(\rho) + 1
$$

where $\rho = \langle x_1, x_2 \rangle$ is the cosine similarity (correlation) between two normalized input vectors.

---

## Building Blocks

### 1. **ReLU Base Kernel** $\Sigma^{(1)}(\rho)$

This is the covariance kernel for a single ReLU layer:

$$
\Sigma^{(1)}(\rho) = \frac{1}{\pi} \left( \sqrt{1-\rho^2} + \rho(\pi - \arccos(\rho)) \right)
$$

**Intuition**: Measures how pre-activations $\text{ReLU}(w^T x_1)$ and $\text{ReLU}(w^T x_2)$ correlate when $w \sim \mathcal{N}(0, I)$.

**Geometric interpretation**: 
- $\theta = \arccos(\rho)$ is the angle between $x_1$ and $x_2$
- $\Sigma^{(1)}(\rho) = \frac{1}{\pi}(\sin\theta + \rho(\pi - \theta))$

### 2. **ReLU Derivative Kernel** $\Theta^{(1)}(\rho)$

This is the derivative (tangent) kernel for a single ReLU layer:

$$
\Theta^{(1)}(\rho) = \frac{1}{2\pi} \left( (\pi - \arccos(\rho))\cos(\arccos(\rho)) + \sin(\arccos(\rho)) \right)
$$

Simplifying using $\cos(\arccos(\rho)) = \rho$ and $\sin(\arccos(\rho)) = \sqrt{1-\rho^2}$:

$$
\Theta^{(1)}(\rho) = \frac{1}{2\pi} \left( (\pi - \arccos(\rho)) \rho + \sqrt{1-\rho^2} \right)
$$

**Intuition**: Measures how gradients $\nabla_w \text{ReLU}(w^T x_1)$ and $\nabla_w \text{ReLU}(w^T x_2)$ correlate.

### 3. **Angular Damping Factor**

The term $\left(1 - \frac{\arccos(\rho)}{\pi}\right)$ dampens the first-layer contribution:

$$
1 - \frac{\arccos(\rho)}{\pi} = 1 - \frac{\theta}{\pi}
$$

where $\theta \in [0, \pi]$ is the angle between inputs.

- When $\rho = 1$ (same input): $\theta = 0$, factor $= 1$ (full contribution)
- When $\rho = 0$ (orthogonal): $\theta = \pi/2$, factor $= 1/2$
- When $\rho = -1$ (opposite): $\theta = \pi$, factor $= 0$ (no contribution)

### 4. **Bias Term**

The constant $+1$ comes from the trainable output bias $c$ in the network architecture.

---

## Full Derivation

### Step 1: First-Layer Features

For input $x$, the first-layer features are:

$$
h^{(1)}(x) = \frac{1}{\sqrt{n_1}} A^{(1)} \sigma(W^{(1)T} x + b^{(1)})
$$

where $\sigma = \text{ReLU}$, $W^{(1)} \in \mathbb{R}^{n_1 \times d}$ is frozen, and $A^{(1)} \in \mathbb{R}^{r \times n_1}$ is trainable.

### Step 2: Empirical Correlation

For two inputs $x_1, x_2$, the empirical correlation of first-layer features is:

$$
\hat{\rho}_r = \frac{\langle h^{(1)}(x_1), h^{(1)}(x_2) \rangle}{\|h^{(1)}(x_1)\| \|h^{(1)}(x_2)\|}
$$

This follows a **Fisher distribution** with parameter $\rho$ (true correlation).

### Step 3: Radial Product

The radial product is:

$$
w_r = \frac{\|h^{(1)}(x_1)\| \|h^{(1)}(x_2)\|}{r}
$$

This follows a **Kibble distribution** and concentrates at $\mathbb{E}[w_r] = 1$ as $r \to \infty$.

### Step 4: Three-Layer NTK

The empirical 3-layer NTK is:

$$
\hat{\Theta}^{(2)}(\hat{\rho}_r, w_r) = \Theta^{(1)}(\hat{\rho}_r) \left(1 - \frac{\arccos(\hat{\rho}_r)}{\pi}\right) + w_r \cdot \Sigma^{(1)}(\hat{\rho}_r) + 1
$$

### Step 5: Infinite-Rank Limit

As $r \to \infty$:
- $\hat{\rho}_r \to \rho$ (Fisher distribution concentrates)
- $w_r \to 1$ (Kibble distribution concentrates)

Therefore:

$$
\lim_{r \to \infty} \hat{\Theta}^{(2)}(\hat{\rho}_r, w_r) = K_\infty(\rho) = \Theta^{(1)}(\rho) \left(1 - \frac{\arccos(\rho)}{\pi}\right) + \Sigma^{(1)}(\rho) + 1
$$

---

## Python Implementation

```python
def compute_theoretical_ntk_limit(rho):
    """
    compute deterministic NTK limit K_infty(rho) for 3-layer ReLU network
    """
    rho = np.clip(rho, -1.0, 1.0)  # we clip rho to valid range #
    theta = np.arccos(rho)  # we compute angle #
    
    # theta^(1) kernel (derivative kernel)
    theta1 = (1.0 / (2 * np.pi)) * ((np.pi - theta) * np.cos(theta) + np.sin(theta))
    # equivalently: (1/(2*pi)) * ((pi - theta) * rho + sqrt(1 - rho^2))
    
    # sigma^(1) kernel (relu base kernel)
    sigma1 = (1.0 / np.pi) * (np.sqrt(1 - rho**2) + rho * (np.pi - theta))
    
    # three-layer ntk: K_infty = Theta^(1)(rho) * (1 - arccos(rho)/pi) + Sigma^(1)(rho) + 1
    K_infty = theta1 * (1.0 - theta / np.pi) + sigma1 + 1.0
    
    return K_infty
```

---

## Special Values

### At $\rho = 1$ (same input):

$$
K_\infty(1) = \Theta^{(1)}(1) \cdot 1 + \Sigma^{(1)}(1) + 1
$$

- $\Theta^{(1)}(1) = \frac{1}{2\pi}(\pi \cdot 1 + 0) = \frac{1}{2}$
- $\Sigma^{(1)}(1) = \frac{1}{\pi}(0 + 1 \cdot \pi) = 1$
- $K_\infty(1) = \frac{1}{2} + 1 + 1 = 2.5$

This is the **on-diagonal** kernel value.

### At $\rho = 0$ (orthogonal inputs):

$$
K_\infty(0) = \Theta^{(1)}(0) \cdot \frac{1}{2} + \Sigma^{(1)}(0) + 1
$$

- $\Theta^{(1)}(0) = \frac{1}{2\pi}(\frac{\pi}{2} \cdot 0 + 1) = \frac{1}{2\pi}$
- $\Sigma^{(1)}(0) = \frac{1}{\pi}(1 + 0) = \frac{1}{\pi}$
- $K_\infty(0) = \frac{1}{2\pi} \cdot \frac{1}{2} + \frac{1}{\pi} + 1 \approx 1.32$

### At $\rho = -1$ (opposite inputs):

$$
K_\infty(-1) = \Theta^{(1)}(-1) \cdot 0 + \Sigma^{(1)}(-1) + 1
$$

- $\Theta^{(1)}(-1) = \frac{1}{2\pi}(0 \cdot (-1) + 0) = 0$
- $\Sigma^{(1)}(-1) = \frac{1}{\pi}(0 + (-1) \cdot 0) = 0$
- $K_\infty(-1) = 0 + 0 + 1 = 1$

---

## Physical Interpretation

1. **$\Theta^{(1)}(\rho) (1 - \arccos(\rho)/\pi)$**: Contribution from gradient alignment at first layer, weighted by angular proximity

2. **$\Sigma^{(1)}(\rho)$**: Fresh basis contribution from second layer (new random features)

3. **$+1$**: Trainable bias at output layer

The kernel grows with correlation: more similar inputs → larger kernel value → stronger training signal correlation.

---

## Verification

In Plot 2 (Three-Layer NTK Concentration), we verify:

1. **Empirical mean** $\mathbb{E}[\hat{\Theta}^{(2)}_r] \to K_\infty(\rho)$ as $r$ increases
2. **Variance** $\text{Var}(\hat{\Theta}^{(2)}_r) = O(1/r)$ decays with rank
3. **Confidence bands** shrink around $K_\infty(\rho)$ (Fisher-Kibble decoupling)

---

## References

- **Theorem 2.1** (paper): Recursive NTK formula for depth-L networks
- **Corollary 2.3** (paper): Explicit 3-layer formula
- **Lemma 2.1** (paper): Fisher-Kibble independence

---

**Summary**: $K_\infty(\rho)$ is the deterministic kernel obtained by taking the infinite-rank limit of the empirical 3-layer NTK, combining derivative kernels, base kernels, and bias terms from the network architecture.

