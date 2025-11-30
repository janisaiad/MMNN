"""
Large-scale NTK matrix computation for Marchenko-Pastur spectrum analysis.

This script computes NTK Gram matrices for RF-LR networks across multiple parameter
configurations and stores eigenvalues for comparison with theoretical predictions.

Key parameters:
- gamma_ratio (l): n/r ratio (several values)
- Dimensions: input dimension d, rank r (p and r are determined)
- Width N: in log scale base 2
- Multiple initializations for averaging

Output: Stores eigenvalues, spike locations, and theoretical predictions.
"""

import numpy as np
import scipy.linalg
from scipy.stats import multivariate_normal
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
import sys
import platform
import hashlib
warnings.filterwarnings('ignore')

# Base seed for reproducibility (do not mutate global RNG)
BASE_SEED = 20250131

# Output directory
OUTPUT_DIR = Path("refs/paper/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    """
    Simple tee for stdout/stderr logging to a file while keeping console output.
    """
    def __init__(self, file_path: Path):
        self.file = open(file_path, "a", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
    
    def write(self, data: str) -> None:
        self._stdout.write(data)
        self.file.write(data)
    
    def flush(self) -> None:
        self._stdout.flush()
        self.file.flush()
    
    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass

def compute_ntk_3layer(x1, x2, w1, w2, b1, b2, A1, A2, c, n1, n2, r, d):
    """
    Compute 3-layer NTK between two inputs x1 and x2.
    
    Architecture: h^(0) = x, h^(1) = (1/sqrt(n1)) * A^(1) * ReLU(W^(1)^T h^(0) + b^(1)), 
                  f = (1/sqrt(n2)) * A^(2) * ReLU(W^(2)^T h^(1) + b^(2)) + c
    
    Formula from paper: Θ^(2)(x,y) = Θ^(1)(ρ₁)(1 - arccos(ρ₁)/π) + w_r·Σ^(1)(ρ₁) + 1
    where ρ₁ = cos(angle(h^(1)(x), h^(1)(x'))) and w_r = ||h^(1)(x)||·||h^(1)(x')||/r
    
    Parameters:
    -----------
    x1, x2 : array-like, shape (d,)
        Input vectors
    w1, w2 : array-like, shape (n1, d) and (n2, r)
        Frozen weight matrices for layers 1 and 2
    b1, b2 : array-like, shape (n1,) and (n2,)
        Frozen bias vectors
    A1, A2 : array-like, shape (r, n1) and (1, n2)
        Trainable weight matrices (low-rank for A1)
    c : float
        Trainable output bias
    n1, n2 : int
        Widths of layers 1 and 2
    r : int
        Rank of bottleneck (rank of A1)
    d : int
        Input dimension
    
    Returns:
    --------
    ntk : float
        Neural Tangent Kernel value
    """
    # Layer 1: h^(1) = (1/sqrt(n1)) * A^(1) * ReLU(W^(1)^T x + b^(1))
    z1_1 = w1 @ x1 + b1  # shape (n1,)
    z1_2 = w1 @ x2 + b1  # shape (n1,)
    
    h1_1 = (1.0 / np.sqrt(n1)) * A1 @ np.maximum(0, z1_1)  # shape (r,)
    h1_2 = (1.0 / np.sqrt(n1)) * A1 @ np.maximum(0, z1_2)  # shape (r,)
    
    # Compute correlation and norms for Fisher-Kibble
    norm_h1_1 = np.linalg.norm(h1_1)
    norm_h1_2 = np.linalg.norm(h1_2)
    
    if norm_h1_1 < 1e-10 or norm_h1_2 < 1e-10:
        # Handle degenerate case
        rho1 = 0.0
        w_r = 0.0
    else:
        rho1 = np.dot(h1_1, h1_2) / (norm_h1_1 * norm_h1_2)
        rho1 = np.clip(rho1, -1.0, 1.0)
        w_r = (norm_h1_1 * norm_h1_2) / r
    
    # Base kernel Sigma^(1)(u) = (1/π)(√(1-u²) + u(1 - arccos u))
    rho1_clipped = np.clip(rho1, -1.0, 1.0)
    theta1 = np.arccos(rho1_clipped)
    sigma1 = (1.0 / np.pi) * (np.sqrt(1 - rho1_clipped**2) + rho1_clipped * (np.pi - theta1))
    
    # One-layer NTK Theta^(1)(ρ) for ReLU: (1/2π)((π - arccos(ρ))cos(ρ) + sin(ρ))
    # This is the derivative kernel
    theta1_ntk = (1.0 / (2 * np.pi)) * ((np.pi - theta1) * np.cos(theta1) + np.sin(theta1))
    
    # Three-layer NTK: Θ^(2) = Θ^(1)(ρ₁)(1 - arccos(ρ₁)/π) + w_r·Σ^(1)(ρ₁) + 1
    ntk = theta1_ntk * (1.0 - theta1 / np.pi) + w_r * sigma1 + 1.0
    
    return ntk


def compute_ntk_gram_matrix(X, w1, w2, b1, b2, A1, A2, c, n1, n2, r, d):
    """
    Compute full NTK Gram matrix for dataset X (vectorized).
    
    Parameters:
    -----------
    X : array-like, shape (n, d)
        Input data matrix
    w1, w2, b1, b2, A1, A2, c : network parameters
    n1, n2, r, d : architecture parameters
    
    Returns:
    --------
    K : array, shape (n, n)
        NTK Gram matrix
    flops : int
        Estimated FLOPs used
    """
    n = X.shape[0]
    # Compute first-layer pre-activations: Z1 = W1 X^T + b1[:,None]  (n1 x n)
    Z1 = (w1 @ X.T) + b1[:, None]
    # ReLU
    Hrelu = np.maximum(0.0, Z1)  # (n1 x n)
    # First-layer features after low-rank A1: H1 = (1/sqrt(n1)) A1 @ Hrelu  (r x n)
    H1 = (A1 @ Hrelu) / np.sqrt(n1)
    # Norms per sample
    norms = np.linalg.norm(H1, axis=0)  # (n,)
    # Inner products matrix G = H1^T H1  (n x n)
    G = H1.T @ H1
    # Avoid division by zero
    eps = 1e-12
    denom = np.outer(norms, norms) + eps  # (n x n)
    # Correlation matrix rho1
    R = G / denom
    R = np.clip(R, -1.0, 1.0)
    # w_r matrix = (||h1(x)|| ||h1(y)||)/r
    Wmat = (np.outer(norms, norms)) / max(r, 1)
    # Compute angle matrix
    Theta = np.arccos(R)
    # Sigma^(1)(R) = (1/pi)(sqrt(1-R^2) + R(pi - arccos R))
    Sigma1 = (1.0 / np.pi) * (np.sqrt(np.maximum(0.0, 1.0 - R**2)) + R * (np.pi - Theta))
    # Theta^(1)(R) = (1/2pi)((pi - arccos R)cos(arccos R) + sin(arccos R))
    # Note: cos(arccos R) = R, sin(arccos R) = sqrt(1 - R^2)
    Theta1 = (1.0 / (2.0 * np.pi)) * ((np.pi - Theta) * R + np.sqrt(np.maximum(0.0, 1.0 - R**2)))
    # Three-layer NTK
    K = Theta1 * (1.0 - Theta / np.pi) + Wmat * Sigma1 + 1.0
    # Symmetrize numerically
    K = 0.5 * (K + K.T)
    # FLOPs estimate (rough): cost dominated by matmuls
    flops = (w1.shape[0] * X.shape[1] * X.shape[0]) + (A1.shape[0] * A1.shape[1] * X.shape[0]) + (X.shape[0] * X.shape[0] * r)
    return K, int(flops)


def initialize_rflr_network(n1, n2, r, d, rng, beta=1.0, sigma_A=1.0, sigma_c=1.0):
    """
    Initialize RF-LR network parameters.
    
    Parameters:
    -----------
    n1, n2 : int
        Widths of layers 1 and 2
    r : int
        Rank of bottleneck
    d : int
        Input dimension
    beta : float
        Bias scale (default: 1.0)
    rng : np.random.Generator
        Random number generator
    sigma_A : float
        Weight scale for A (default: 1.0)
    sigma_c : float
        Output bias scale (default: 1.0)
    
    Returns:
    --------
    w1, w2, b1, b2, A1, A2, c : network parameters
    """
    # Frozen weights: w^(1) ~ N(0, I_d/d), w^(2) ~ N(0, I_r/r)
    w1 = rng.standard_normal((n1, d)) / np.sqrt(d)  # shape (n1, d)
    w2 = rng.standard_normal((n2, r)) / np.sqrt(r)  # shape (n2, r)
    
    # Frozen biases
    b1 = rng.standard_normal(n1) * beta
    b2 = rng.standard_normal(n2) * beta
    
    # Trainable weights: A^(1) has rank r, A^(2) is full
    # A^(1): shape (r, n1) - low-rank bottleneck
    A1 = rng.standard_normal((r, n1)) * sigma_A / np.sqrt(n1)
    
    # A^(2): shape (1, n2) - output layer
    A2 = rng.standard_normal((1, n2)) * sigma_A / np.sqrt(n2)
    
    # Output bias
    c = float(rng.standard_normal() * sigma_c)
    
    return w1, w2, b1, b2, A1, A2, c


def generate_data(n, d, covariance_type='identity', rng=None, ell: float = 1.0):
    """
    Generate input data.
    
    Parameters:
    -----------
    n : int
        Number of data points
    d : int
        Input dimension
    covariance_type : str
        'identity' or 'isotropic' (default: 'identity')
    rng : np.random.Generator, optional
        Random number generator
    ell : float
        Trace normalization per dimension: trace(Sigma_d)/d -> ell. For isotropic, Sigma_d = ell * I.
    
    Returns:
    --------
    X : array, shape (n, d)
        Input data matrix
    """
    if rng is None:
        rng = np.random.default_rng(BASE_SEED)
    
    if covariance_type == 'identity':
        X = rng.standard_normal((n, d))
    elif covariance_type == 'isotropic':
        # Normalize to unit sphere
        X = rng.standard_normal((n, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    elif covariance_type == 'isotropic_ell':
        # Gaussian with covariance ell * I_d: ensures operator norm=ell and trace/d = ell
        X = rng.standard_normal((n, d)) * np.sqrt(max(ell, 0.0))
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")
    
    return X


def compute_theoretical_ntk_limit(rho):
    """
    Compute deterministic NTK limit K_infty(rho) for 3-layer ReLU network.
    
    Parameters:
    -----------
    rho : float or array
        Cosine similarity (correlation)
    
    Returns:
    --------
    K_infty : float or array
        Deterministic NTK limit
    """
    rho = np.clip(rho, -1.0, 1.0)
    theta = np.arccos(rho)
    
    # Theta^(1) kernel (derivative kernel)
    theta1 = (1.0 / (2 * np.pi)) * ((np.pi - theta) * np.cos(theta) + np.sin(theta))
    
    # Sigma^(1) kernel (ReLU base kernel)
    sigma1 = (1.0 / np.pi) * (np.sqrt(1 - rho**2) + rho * (np.pi - theta))
    
    # Three-layer NTK: K_infty = Theta^(1)(rho) * (1 - arccos(rho)/pi) + Sigma^(1)(rho) + 1
    K_infty = theta1 * (1.0 - theta / np.pi) + sigma1 + 1.0
    
    return K_infty


def _generate_pairs_fixed_rho(rho: float, d: int, num_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs (x, y) on the unit sphere in R^d with prescribed cosine similarity rho.  # we generate pairs with fixed cosine similarity #
    We use orthogonal projection: sample x ~ N(0, I), normalize; sample z ~ N(0, I),
    remove projection on x, normalize to z_perp; set y = rho·x + sqrt(1-rho^2)·z_perp.  # we construct y from x and an orthogonal component #
    """
    rho_clipped = float(np.clip(rho, -1.0, 1.0))  # we clip rho #
    x = rng.standard_normal((num_pairs, d))  # we sample x #
    x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12  # we compute norms #
    x_unit = x / x_norm  # we normalize #
    z = rng.standard_normal((num_pairs, d))  # we sample z #
    # subtract projection of z on x_unit
    proj = np.sum(z * x_unit, axis=1, keepdims=True)  # we compute projection scalar #
    z_perp = z - proj * x_unit  # we remove projection #
    z_perp_norm = np.linalg.norm(z_perp, axis=1, keepdims=True) + 1e-12  # we compute norms #
    z_perp_unit = z_perp / z_perp_norm  # we normalize #
    y = rho_clipped * x_unit + np.sqrt(max(0.0, 1.0 - rho_clipped**2)) * z_perp_unit  # we build y #
    return x_unit.astype(np.float64), y.astype(np.float64)  # we return unit vectors #


def _compute_ntk_for_pairs(
    X1: np.ndarray,
    X2: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    A1: np.ndarray,
    r: int,
    n1: int
) -> np.ndarray:
    """
    Vectorized NTK computation for aligned pairs (X1[i], X2[i]). Uses the 3-layer formula based on first-layer features.  # we compute ntk for pairs using first-layer features #
    """
    # first-layer pre-activations
    Z1a = (w1 @ X1.T) + b1[:, None]  # shape (n1, S)  # we compute pre-activations for X1 #
    Z1b = (w1 @ X2.T) + b1[:, None]  # shape (n1, S)  # we compute pre-activations for X2 #
    Hrelu_a = np.maximum(0.0, Z1a)  # we apply relu #
    Hrelu_b = np.maximum(0.0, Z1b)  # we apply relu #
    H1a = (A1 @ Hrelu_a) / np.sqrt(n1)  # shape (r, S)  # we compute first-layer features #
    H1b = (A1 @ Hrelu_b) / np.sqrt(n1)  # shape (r, S)  # we compute first-layer features #
    # per-pair correlations and radial product
    norms_a = np.linalg.norm(H1a, axis=0)  # (S,)  # we compute norms #
    norms_b = np.linalg.norm(H1b, axis=0)  # (S,)  # we compute norms #
    eps = 1e-12  # we set epsilon #
    dots = np.sum(H1a * H1b, axis=0)  # (S,)  # we compute dot products #
    denom = np.maximum(norms_a * norms_b, eps)  # (S,)  # we avoid division by zero #
    rho1 = np.clip(dots / denom, -1.0, 1.0)  # (S,)  # we compute correlation #
    w_r = (norms_a * norms_b) / max(r, 1)  # (S,)  # we compute radial term #
    # kernels
    theta = np.arccos(rho1)  # (S,)  # we compute angle #
    sigma1 = (1.0 / np.pi) * (np.sqrt(np.maximum(0.0, 1.0 - rho1**2)) + rho1 * (np.pi - theta))  # we compute base kernel #
    theta1 = (1.0 / (2.0 * np.pi)) * ((np.pi - theta) * rho1 + np.sqrt(np.maximum(0.0, 1.0 - rho1**2)))  # we compute derivative kernel #
    ntk = theta1 * (1.0 - theta / np.pi) + w_r * sigma1 + 1.0  # (S,)  # we compute 3-layer ntk #
    return ntk  # we return ntk samples #


def _compute_ntk_rho_distributions_for_config(
    n1: int,
    n2: int,
    r: int,
    d: int,
    rho_vals: np.ndarray,
    samples_per_rho: int,
    rng_init: np.random.Generator,
    rng_data: np.random.Generator
) -> dict:
    """
    For one network config (n1, n2, r, d), sample NTK distributions across rho buckets.  # we compute ntk distributions for rho buckets #
    """
    # initialize one network (single init to isolate Fisher-Kibble variability)  # we initialize a single network #
    w1, w2, b1, b2, A1, A2, c = initialize_rflr_network(
        n1, n2, r, d, rng=rng_init, beta=1.0, sigma_A=np.sqrt(2.0), sigma_c=1.0
    )  # we initialize the network under eoc #
    ntk_samples = []  # we collect ntk samples per rho #
    for rho in rho_vals:
        X1, X2 = _generate_pairs_fixed_rho(float(rho), d, samples_per_rho, rng_data)  # we generate pairs with fixed rho #
        ntk_rho = _compute_ntk_for_pairs(X1, X2, w1, b1, A1, r, n1)  # we compute ntk for pairs #
        ntk_samples.append(ntk_rho.astype(np.float64))  # we append ntk samples #
    ntk_samples = np.stack(ntk_samples, axis=0)  # shape (R, S)  # we stack to array #
    ntk_mean = np.mean(ntk_samples, axis=1)  # (R,)  # we compute means #
    ntk_std = np.std(ntk_samples, axis=1)  # (R,)  # we compute stds #
    k_infty = compute_theoretical_ntk_limit(rho_vals.astype(np.float64))  # (R,)  # we compute deterministic kernel #
    return {
        "rho_vals": rho_vals.astype(np.float64),
        "ntk_samples": ntk_samples,
        "ntk_mean": ntk_mean,
        "ntk_std": ntk_std,
        "k_infty": k_infty.astype(np.float64),
        "n1": int(n1),
        "n2": int(n2),
        "r": int(r),
        "d": int(d),
        "samples_per_rho": int(samples_per_rho),
    }  # we return results #


def compute_mp_density_params(gamma_ratio, K_infty_0, K_infty_prime_0, K_infty_diag):
    """
    Compute Marchenko-Pastur density parameters.
    
    Parameters:
    -----------
    gamma_ratio : float
        n/r ratio
    K_infty_0 : float
        K_infty(0) - kernel at zero correlation
    K_infty_prime_0 : float
        K_infty'(0) - derivative at zero
    K_infty_diag : float
        K_infty(1) - diagonal value (or trace/d)
    
    Returns:
    --------
    params : dict
        MP density parameters (beta, gamma, support)
    """
    # From Theorem 4.2:
    # alpha = K_infty(0) (spike coefficient)
    # beta = K_infty'(0) (slope)
    # gamma = K_infty(diag) - alpha - diag * beta (diagonal shift)
    
    alpha = K_infty_0
    beta = K_infty_prime_0
    gamma = K_infty_diag - alpha - 1.0 * beta  # Assuming diag = 1 (normalized)
    
    # Bulk support: [gamma + beta*(1-sqrt(gamma_ratio))^2, gamma + beta*(1+sqrt(gamma_ratio))^2]
    support_low = gamma + beta * (1 - np.sqrt(gamma_ratio))**2
    support_high = gamma + beta * (1 + np.sqrt(gamma_ratio))**2
    
    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'support': (support_low, support_high),
        'gamma_ratio': gamma_ratio
    }


def run_large_scale_computation(
    gamma_ratios=[0.5, 1.0, 2.0],
    n_base=256,  # Base number of data points (will scale)
    width_powers=[4, 5, 6, 7, 8],  # Width N = 2^power (log scale base 2)
    n_init=10,  # Number of random initializations
    d_base=32,  # Base input dimension
    r_base=16,  # Base rank
    output_file="plot4_marchenko_pastur.npz"
):
    """
    Run large-scale NTK computation across multiple parameter configurations.
    
    Parameters:
    -----------
    gamma_ratios : list
        List of n/r ratios to test
    n_base : int
        Base number of data points
    width_powers : list
        List of powers for width: N = 2^power (log scale base 2)
    n_init : int
        Number of random initializations per configuration
    d_base : int
        Base input dimension
    r_base : int
        Base rank
    output_file : str
        Output filename
    """
    print("=" * 80)
    print("Large-Scale NTK Matrix Computation")
    print("=" * 80)
    print(f"Gamma ratios (l): {gamma_ratios}")
    print(f"Width powers (log2 scale): {width_powers}")
    print(f"Number of initializations: {n_init}")
    print(f"Base dimensions: d={d_base}, r={r_base}, n={n_base}")
    print("=" * 80)
    
    # Storage for aggregated results (we also save per-config files)
    results = {
        'gamma_ratios': np.array(gamma_ratios),
        'width_powers': np.array(width_powers),
        'n_vals': [],
        'r_vals': [],
        'd_vals': [],
        'n1_vals': [],
        'n2_vals': [],
        'eigenvalues_all': [],  # List of arrays, one per config
        'lambda_spike_all': [],
        'lambda_spike_mean': [],
        'lambda_spike_std': [],
        'mp_params_all': [],
        'flops_total': 0,
        'metadata': {}
    }
    
    total_configs = len(gamma_ratios) * len(width_powers)
    config_idx = 0
    
    # Compute theoretical kernel values for MP parameters
    K_infty_0 = compute_theoretical_ntk_limit(0.0)
    # Approximate derivative at 0 (finite difference)
    eps = 1e-5
    K_infty_eps = compute_theoretical_ntk_limit(eps)
    K_infty_prime_0 = (K_infty_eps - K_infty_0) / eps
    K_infty_diag = compute_theoretical_ntk_limit(1.0)
    
    print(f"\nTheoretical kernel values:")
    print(f"  K_infty(0) = {K_infty_0:.6f}")
    print(f"  K_infty'(0) = {K_infty_prime_0:.6f}")
    print(f"  K_infty(diag) = {K_infty_diag:.6f}\n")
    
    # Helper to get stable per-config seed
    def get_config_seed(gamma_ratio_val: float, width_power_val: int) -> int:
        s = f"{BASE_SEED}|gamma={gamma_ratio_val:.6f}|width={width_power_val}"
        h = hashlib.blake2b(s.encode('utf-8'), digest_size=8).hexdigest()
        return int(h, 16) & 0xFFFFFFFF
    
    # Main computation loop
    for gamma_ratio in gamma_ratios:
        for width_power in width_powers:
            config_idx += 1
            n1 = 2 ** width_power  # Width of layer 1
            n2 = n1  # Width of layer 2 (can be different)
            
            # Determine n and r from gamma_ratio
            # gamma_ratio = n / r, so if we fix r, n = gamma_ratio * r
            r = r_base * (2 ** (width_power - 4))  # Scale r with width
            n = int(gamma_ratio * r)
            d = d_base * (2 ** (width_power - 4))  # Scale d with width
            
            # Ensure n is reasonable
            n = max(n, 32)  # Minimum n
            n = min(n, 2048)  # Maximum n (for memory)
            
            print(f"\n[{config_idx}/{total_configs}] Config: gamma_ratio={gamma_ratio:.2f}, "
                  f"width_power={width_power} (N={n1})")
            print(f"  Dimensions: n={n}, r={r}, d={d}, n1={n1}, n2={n2}")
            
            # Output filename per configuration (store spectrum only)
            config_file = OUTPUT_DIR / (
                f"mp_gamma{gamma_ratio:g}_N{n1}_n{n}_r{r}_d{d}.npz"
            )
            config_meta_file = OUTPUT_DIR / (
                f"mp_gamma{gamma_ratio:g}_N{n1}_n{n}_r{r}_d{d}_metadata.json"
            )
            
            # Skip if already computed (resume-friendly)
            if config_file.exists() and config_meta_file.exists():
                print(f"  Found existing results for this config. Skipping computation.")
                # Load spike from metadata for aggregation
                try:
                    with open(config_meta_file, 'r') as mf:
                        meta = json.load(mf)
                    results['n_vals'].append(n)
                    results['r_vals'].append(r)
                    results['d_vals'].append(d)
                    results['n1_vals'].append(n1)
                    results['n2_vals'].append(n2)
                    results['lambda_spike_mean'].append(meta.get('lambda_spike_mean', np.nan))
                    results['lambda_spike_std'].append(meta.get('lambda_spike_std', np.nan))
                    results['mp_params_all'].append(meta.get('mp_params', {}))
                except Exception as e:
                    print(f"  Warning: could not read metadata. {e}")
                continue
            
            # Storage for this configuration
            eigenvalues_config = []
            lambda_spike_config = []
            flops_config = 0
            data_seeds = []
            init_seeds = []
            
            # Multiple initializations
            for init_idx in range(n_init):
                # Deterministic seeds per init
                config_seed = get_config_seed(gamma_ratio, width_power)
                seed_data = (config_seed ^ (0x9E3779B9 + init_idx * 0x85EBCA6B)) & 0xFFFFFFFF
                seed_init = (config_seed ^ (0xC2B2AE35 + init_idx * 0x27D4EB2F)) & 0xFFFFFFFF
                data_seeds.append(int(seed_data))
                init_seeds.append(int(seed_init))
                
                rng_data = np.random.default_rng(seed_data)
                rng_init = np.random.default_rng(seed_init)
                
                # Generate data (isotropic on sphere)
                X = generate_data(n, d, covariance_type='isotropic', rng=rng_data)
                
                # Initialize network
                # EOC: sigma_A = sqrt(2) when Tr(Sigma_w) = 1 (Sigma_w = I_d/d)
                w1, w2, b1, b2, A1, A2, c = initialize_rflr_network(
                    n1, n2, r, d, rng=rng_init, beta=1.0, sigma_A=np.sqrt(2.0), sigma_c=1.0
                )
                
                # Compute NTK Gram matrix
                K, flops_k = compute_ntk_gram_matrix(
                    X, w1, w2, b1, b2, A1, A2, c, n1, n2, r, d
                )
                flops_config += flops_k
                
                # Eigendecomposition
                eigenvalues = np.linalg.eigvalsh(K)  # Use eigh for symmetric matrix
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
                
                eigenvalues_config.append(eigenvalues)
                lambda_spike_config.append(eigenvalues[0])  # Largest eigenvalue
                
                if (init_idx + 1) % 5 == 0:
                    print(f"    Completed {init_idx + 1}/{n_init} initializations...")
            
            # Average over initializations
            eigenvalues_mean = np.mean(eigenvalues_config, axis=0)
            lambda_spike_mean = np.mean(lambda_spike_config)
            lambda_spike_std = np.std(lambda_spike_config)
            
            # Compute MP parameters
            mp_params = compute_mp_density_params(
                gamma_ratio, K_infty_0, K_infty_prime_0, K_infty_diag
            )
            
            # Store aggregated results (for master summary)
            results['n_vals'].append(n)
            results['r_vals'].append(r)
            results['d_vals'].append(d)
            results['n1_vals'].append(n1)
            results['n2_vals'].append(n2)
            results['eigenvalues_all'].append(eigenvalues_mean)
            results['lambda_spike_all'].append(lambda_spike_config)
            results['lambda_spike_mean'].append(lambda_spike_mean)
            results['lambda_spike_std'].append(lambda_spike_std)
            results['mp_params_all'].append(mp_params)
            results['flops_total'] += flops_config
            
            print(f"  Spike eigenvalue: {lambda_spike_mean:.4f} ± {lambda_spike_std:.4f}")
            print(f"  Theoretical spike: {n * K_infty_0:.4f}")
            print(f"  FLOPs for this config: {flops_config:.2e}")
            
            # Save per-configuration spectrum only (no Gram matrices)
            np.savez_compressed(
                config_file,
                eigenvalues_per_init=np.array(eigenvalues_config, dtype=object),
                eigenvalues_mean=eigenvalues_mean,
                lambda_spike_per_init=np.array(lambda_spike_config),
                lambda_spike_mean=lambda_spike_mean,
                lambda_spike_std=lambda_spike_std,
                gamma_ratio=gamma_ratio,
                n=n,
                r=r,
                d=d,
                n1=n1,
                n2=n2,
                seeds_data=np.array(data_seeds, dtype=np.uint32),
                seeds_init=np.array(init_seeds, dtype=np.uint32),
                flops_config=flops_config
            )
            meta_cfg = {
                "computation_date": datetime.now().isoformat(),
                "base_seed": BASE_SEED,
                "config_seed": get_config_seed(gamma_ratio, width_power),
                "n_init": n_init,
                "gamma_ratio": gamma_ratio,
                "n": n,
                "r": r,
                "d": d,
                "n1": n1,
                "n2": n2,
                "lambda_spike_mean": float(lambda_spike_mean),
                "lambda_spike_std": float(lambda_spike_std),
                "mp_params": mp_params,
                "flops_config": float(flops_config),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.linalg.__version__ if hasattr(scipy.linalg, "__version__") else "unknown",
            }
            with open(config_meta_file, 'w') as f:
                json.dump(meta_cfg, f, indent=2)
    
    # Convert lists to arrays (eigenvalues_all stays as list due to variable sizes)
    results['n_vals'] = np.array(results['n_vals'])
    results['r_vals'] = np.array(results['r_vals'])
    results['d_vals'] = np.array(results['d_vals'])
    results['n1_vals'] = np.array(results['n1_vals'])
    results['n2_vals'] = np.array(results['n2_vals'])
    results['lambda_spike_mean'] = np.array(results['lambda_spike_mean'])
    results['lambda_spike_std'] = np.array(results['lambda_spike_std'])
    
    # Add metadata
    results['metadata'] = {
        'computation_date': datetime.now().isoformat(),
        'random_seed': 42,
        'n_init': n_init,
        'K_infty_0': float(K_infty_0),
        'K_infty_prime_0': float(K_infty_prime_0),
        'K_infty_diag': float(K_infty_diag),
        'total_configs': total_configs,
        'total_flops': float(results['flops_total'])
    }
    
    # Save aggregated master results
    output_path = OUTPUT_DIR / output_file
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_path}")
    print(f"Total FLOPs: {results['flops_total']:.2e}")
    print(f"{'='*80}\n")
    
    # Save as npz (eigenvalues_all as list, others as arrays)
    np.savez_compressed(
        output_path,
        **{k: v for k, v in results.items() if k != 'metadata'},
        metadata=json.dumps(results['metadata'])
    )
    
    # Also save metadata separately as JSON for easy reading (master index)
    metadata_path = OUTPUT_DIR / (output_file.replace('.npz', '_metadata.json'))
    with open(metadata_path, 'w') as f:
        json.dump(results['metadata'], f, indent=2)
    
    print("Computation complete!")
    return results


def run_grid_computation(
    data_powers=range(4, 11),     # n in {2^4, ..., 2^10}
    width_powers=range(4, 11),    # N in {2^4, ..., 2^10}
    rank_powers=range(4, 11),     # r in {2^4, ..., 2^10}
    n_init=5,                     # initializations per configuration
    d_policy="equal_r",           # how to choose d given r/N
    do_ntk_rho=True,              # also compute NTK-vs-rho distributions
    rho_step=0.1,                 # step for rho buckets (-1 to 1)
    samples_per_rho=2000          # samples per rho bucket
):
    """
    Grid search over (n, N, r) with spectra storage only. Fully reproducible and resume-friendly.
    """
    print("=" * 80)
    print("Grid NTK spectrum computation over (n, N, r)")
    print("=" * 80)
    print(f"n powers:    {list(data_powers)}")
    print(f"N powers:    {list(width_powers)}")
    print(f"r powers:    {list(rank_powers)}")
    print(f"n_init:      {n_init}")
    print(f"d policy:    {d_policy}")
    print("=" * 80)
    
    # Theoretical kernel values (shared)
    K_infty_0 = compute_theoretical_ntk_limit(0.0)
    eps = 1e-5
    K_infty_prime_0 = (compute_theoretical_ntk_limit(eps) - K_infty_0) / eps
    K_infty_diag = compute_theoretical_ntk_limit(1.0)
    
    total_configs = len(list(data_powers)) * len(list(width_powers)) * len(list(rank_powers))
    cfg_idx = 0
    master_total_flops = 0.0
    master_index = []
    ntk_rho_index = []
    
    for n_pow in data_powers:
        for N_pow in width_powers:
            for r_pow in rank_powers:
                cfg_idx += 1
                n = 2 ** n_pow
                n1 = 2 ** N_pow
                n2 = n1
                r = 2 ** r_pow
                if d_policy == "equal_r":
                    d = r
                elif d_policy == "equal_N":
                    d = n1
                else:
                    d = max(16, min(1024, r))
                gamma_ratio = n / max(r, 1)
                
                print(f"\n[{cfg_idx}/{total_configs}] Config: n={n}, N={n1}, r={r}, d={d} (gamma={gamma_ratio:.3f})")
                
                config_file = OUTPUT_DIR / (f"grid_n{n}_N{n1}_r{r}_d{d}.npz")
                config_meta_file = OUTPUT_DIR / (f"grid_n{n}_N{n1}_r{r}_d{d}_metadata.json")
                
                if config_file.exists() and config_meta_file.exists():
                    print("  Found existing results. Skipping.")
                    master_index.append({"file": str(config_file), "meta": str(config_meta_file)})
                    # still attempt NTK-rho if requested and missing
                    if not do_ntk_rho:
                        continue
                    # prepare NTK-rho filenames
                    ntk_file = OUTPUT_DIR / (f"grid_n{n}_N{n1}_r{r}_d{d}_ntk_rho.npz")
                    ntk_meta_file = OUTPUT_DIR / (f"grid_n{n}_N{n1}_r{r}_d{d}_ntk_rho_metadata.json")
                    if ntk_file.exists() and ntk_meta_file.exists():
                        ntk_rho_index.append({"file": str(ntk_file), "meta": str(ntk_meta_file)})
                        continue
                    # compute missing NTK-rho for this config
                    rho_vals = np.round(np.arange(-1.0, 1.0 + 1e-9, rho_step), 6)
                    cfg_seed_str = f"{BASE_SEED}|n={n}|N={n1}|r={r}|d={d}"
                    cfg_seed_hash = hashlib.blake2b(cfg_seed_str.encode("utf-8"), digest_size=8).hexdigest()
                    cfg_seed = int(cfg_seed_hash, 16) & 0xFFFFFFFF
                    seed_data_ntk = (cfg_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
                    seed_init_ntk = (cfg_seed ^ 0x5A5A5A5A) & 0xFFFFFFFF
                    rng_data_ntk = np.random.default_rng(seed_data_ntk)
                    rng_init_ntk = np.random.default_rng(seed_init_ntk)
                    print("  Computing NTK-vs-rho distributions (resume)…")
                    ntk_res = _compute_ntk_rho_distributions_for_config(
                        n1, n1, r, d, rho_vals, samples_per_rho, rng_init_ntk, rng_data_ntk
                    )
                    np.savez_compressed(
                        ntk_file,
                        rho_vals=ntk_res["rho_vals"],
                        ntk_samples=ntk_res["ntk_samples"],
                        ntk_mean=ntk_res["ntk_mean"],
                        ntk_std=ntk_res["ntk_std"],
                        k_infty=ntk_res["k_infty"],
                        n=n,
                        r=r,
                        d=d,
                        n1=n1,
                        n2=n1,
                        samples_per_rho=samples_per_rho
                    )
                    meta_ntk = {
                        "computation_date": datetime.now().isoformat(),
                        "type": "ntk_rho_distribution",
                        "base_seed": BASE_SEED,
                        "config_seed": cfg_seed,
                        "seeds": {"data": int(seed_data_ntk), "init": int(seed_init_ntk)},
                        "n": n,
                        "r": r,
                        "d": d,
                        "n1": n1,
                        "n2": n1,
                        "rho_step": rho_step,
                        "samples_per_rho": samples_per_rho,
                        "python": sys.version,
                        "platform": platform.platform(),
                        "numpy": np.__version__,
                        "scipy": scipy.linalg.__version__ if hasattr(scipy.linalg, "__version__") else "unknown",
                        "notes": "Pairs constructed on unit sphere with exact rho; NTK via first-layer features."
                    }
                    with open(ntk_meta_file, "w") as f:
                        json.dump(meta_ntk, f, indent=2)
                    ntk_rho_index.append({"file": str(ntk_file), "meta": str(ntk_meta_file)})
                    continue
                
                # Per-config deterministic seed
                s = f"{BASE_SEED}|n={n}|N={n1}|r={r}|d={d}"
                cfg_seed_hash = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
                cfg_seed = int(cfg_seed_hash, 16) & 0xFFFFFFFF
                
                eigenvalues_config = []
                lambda_spike_config = []
                seeds_data = []
                seeds_init = []
                flops_config = 0.0
                
                for init_idx in range(n_init):
                    seed_data = (cfg_seed ^ (0x9E3779B9 + init_idx * 0x85EBCA6B)) & 0xFFFFFFFF
                    seed_init = (cfg_seed ^ (0xC2B2AE35 + init_idx * 0x27D4EB2F)) & 0xFFFFFFFF
                    seeds_data.append(int(seed_data))
                    seeds_init.append(int(seed_init))
                    rng_data = np.random.default_rng(seed_data)
                    rng_init = np.random.default_rng(seed_init)
                    
                    # Data on unit sphere
                    X = generate_data(n, d, covariance_type="isotropic", rng=rng_data)
                    # Initialize under EOC
                    w1, w2, b1, b2, A1, A2, c = initialize_rflr_network(
                        n1, n2, r, d, rng=rng_init, beta=1.0, sigma_A=np.sqrt(2.0), sigma_c=1.0
                    )
                    # Compute NTK Gram matrix and eigenvalues
                    K, flops_k = compute_ntk_gram_matrix(X, w1, w2, b1, b2, A1, A2, c, n1, n2, r, d)
                    flops_config += float(flops_k)
                    ev = np.linalg.eigvalsh(K)
                    ev = np.sort(ev)[::-1]
                    eigenvalues_config.append(ev)
                    lambda_spike_config.append(float(ev[0]))
                    
                    if (init_idx + 1) % max(1, n_init // 5) == 0:
                        print(f"  init {init_idx + 1}/{n_init} done")
                
                # Aggregate
                eigenvalues_mean = np.mean(eigenvalues_config, axis=0)
                lambda_spike_mean = float(np.mean(lambda_spike_config))
                lambda_spike_std = float(np.std(lambda_spike_config))
                mp_params = compute_mp_density_params(gamma_ratio, K_infty_0, K_infty_prime_0, K_infty_diag)
                
                # Save spectra only
                np.savez_compressed(
                    config_file,
                    eigenvalues_per_init=np.array(eigenvalues_config, dtype=object),
                    eigenvalues_mean=eigenvalues_mean,
                    lambda_spike_per_init=np.array(lambda_spike_config),
                    lambda_spike_mean=lambda_spike_mean,
                    lambda_spike_std=lambda_spike_std,
                    gamma_ratio=gamma_ratio,
                    n=n,
                    r=r,
                    d=d,
                    n1=n1,
                    n2=n2,
                    seeds_data=np.array(seeds_data, dtype=np.uint32),
                    seeds_init=np.array(seeds_init, dtype=np.uint32),
                    flops_config=float(flops_config)
                )
                meta_cfg = {
                    "computation_date": datetime.now().isoformat(),
                    "base_seed": BASE_SEED,
                    "config_seed": cfg_seed,
                    "n_init": n_init,
                    "gamma_ratio": gamma_ratio,
                    "n": n,
                    "r": r,
                    "d": d,
                    "n1": n1,
                    "n2": n2,
                    "lambda_spike_mean": lambda_spike_mean,
                    "lambda_spike_std": lambda_spike_std,
                    "mp_params": mp_params,
                    "flops_config": float(flops_config),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "scipy": scipy.linalg.__version__ if hasattr(scipy.linalg, "__version__") else "unknown",
                    "notes": "Spectra only; no Gram matrices; EOC sigma_A=sqrt(2); ReLU; isotropic data on sphere."
                }
                with open(config_meta_file, "w") as f:
                    json.dump(meta_cfg, f, indent=2)
                
                master_total_flops += flops_config
                master_index.append({"file": str(config_file), "meta": str(config_meta_file)})
                print(f"  spike={lambda_spike_mean:.4f}±{lambda_spike_std:.4f}, FLOPs={flops_config:.2e}")
                
                # Alternate: NTK kernel vs rho after spectrum (if enabled)
                if do_ntk_rho:
                    ntk_file = OUTPUT_DIR / (f"grid_n{n}_N{n1}_r{r}_d{d}_ntk_rho.npz")
                    ntk_meta_file = OUTPUT_DIR / (f"grid_n{n}_N{n1}_r{r}_d{d}_ntk_rho_metadata.json")
                    if ntk_file.exists() and ntk_meta_file.exists():
                        print("  NTK-vs-rho file exists. Skipping.")
                        ntk_rho_index.append({"file": str(ntk_file), "meta": str(ntk_meta_file)})
                    else:
                        rho_vals = np.round(np.arange(-1.0, 1.0 + 1e-9, rho_step), 6)  # buckets -1, -0.9, ..., 1  # we set rho buckets #
                        seed_data_ntk = (cfg_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF  # we derive data seed #
                        seed_init_ntk = (cfg_seed ^ 0x5A5A5A5A) & 0xFFFFFFFF  # we derive init seed #
                        rng_data_ntk = np.random.default_rng(seed_data_ntk)  # we rng #
                        rng_init_ntk = np.random.default_rng(seed_init_ntk)  # we rng #
                        print("  Computing NTK-vs-rho distributions…")
                        ntk_res = _compute_ntk_rho_distributions_for_config(
                            n1, n2, r, d, rho_vals, samples_per_rho, rng_init_ntk, rng_data_ntk
                        )
                        np.savez_compressed(
                            ntk_file,
                            rho_vals=ntk_res["rho_vals"],
                            ntk_samples=ntk_res["ntk_samples"],
                            ntk_mean=ntk_res["ntk_mean"],
                            ntk_std=ntk_res["ntk_std"],
                            k_infty=ntk_res["k_infty"],
                            n=n,
                            r=r,
                            d=d,
                            n1=n1,
                            n2=n2,
                            samples_per_rho=samples_per_rho
                        )
                        meta_ntk = {
                            "computation_date": datetime.now().isoformat(),
                            "type": "ntk_rho_distribution",
                            "base_seed": BASE_SEED,
                            "config_seed": cfg_seed,
                            "seeds": {"data": int(seed_data_ntk), "init": int(seed_init_ntk)},
                            "n": n,
                            "r": r,
                            "d": d,
                            "n1": n1,
                            "n2": n2,
                            "rho_step": rho_step,
                            "samples_per_rho": samples_per_rho,
                            "python": sys.version,
                            "platform": platform.platform(),
                            "numpy": np.__version__,
                            "scipy": scipy.linalg.__version__ if hasattr(scipy.linalg, "__version__") else "unknown",
                            "notes": "Pairs constructed on unit sphere with exact rho; NTK via first-layer features."
                        }
                        with open(ntk_meta_file, "w") as f:
                            json.dump(meta_ntk, f, indent=2)
                        ntk_rho_index.append({"file": str(ntk_file), "meta": str(ntk_meta_file)})
    
    # Save master index
    master_path = OUTPUT_DIR / "grid_master_index.json"
    with open(master_path, "w") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "base_seed": BASE_SEED,
            "data_powers": list(data_powers),
            "width_powers": list(width_powers),
            "rank_powers": list(rank_powers),
            "n_init": n_init,
            "total_configs": total_configs,
            "total_flops_estimate": float(master_total_flops),
            "index": master_index,
            "ntk_rho_index": ntk_rho_index
        }, f, indent=2)
    
    print("\nGrid computation complete.")
    print(f"  Total configurations: {total_configs}")
    print(f"  Master index saved to: {master_path}")
    print(f"  Total FLOPs (estimate): {master_total_flops:.2e}")
    return {
        "master_index": str(master_path),
        "total_configs": total_configs,
        "total_flops_estimate": master_total_flops
    }


if __name__ == "__main__":
    # Create log file and tee stdout/stderr
    log_name = f"grid_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = OUTPUT_DIR / log_name
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"Logging to: {log_path}")
    
    # Grid configuration
    DATA_POWERS = list(range(4, 11))
    WIDTH_POWERS = list(range(4, 11))
    RANK_POWERS = list(range(4, 11))
    N_INIT = 5
    D_POLICY = "equal_r"
    
    # Run grid computation (overnight run)
    grid_summary = run_grid_computation(
        data_powers=DATA_POWERS,
        width_powers=WIDTH_POWERS,
        rank_powers=RANK_POWERS,
        n_init=N_INIT,
        d_policy=D_POLICY
    )
    print("\nSummary:")
    print(f"  Master index: {grid_summary['master_index']}")
    print(f"  Total configurations: {grid_summary['total_configs']}")
    print(f"  Total FLOPs (estimate): {grid_summary['total_flops_estimate']:.2e}")
    
    # Close log (restore stdout/stderr)
    try:
        tee.flush()
        tee.close()
    except Exception:
        pass

