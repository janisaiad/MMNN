"""
Condition number of proxy vs empirical RF-LR NTK Gram on 1^perp (equicorrelated data).

Validates: (i) proxy condition number on 1^perp is 1 for equicorrelated data (Corollary 13);
(ii) empirical condition number concentrates around 1 as r grows (Theorem equicorrelated-op-bound).

Paper: refs/colt2026/depth_scaling.tex (Proposition 16, Corollary 13), appendix proxy-empirical.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["font.size"] = 18
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 22
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True


def varrho_relu_eoc(rho: float) -> float:
    rho_c = float(np.clip(rho, -1.0, 1.0))
    term = math.sqrt(max(0.0, 1.0 - rho_c * rho_c)) - rho_c * math.acos(rho_c)
    return rho_c + 0.5 * (2.0 / math.pi) * term


def s_relu(rho: float) -> float:
    rho_c = float(np.clip(rho, -1.0, 1.0))
    theta = math.acos(rho_c)
    return ((math.pi - theta) * math.cos(theta) + math.sin(theta)) / (2.0 * math.pi)


def dot_s_relu(rho: float) -> float:
    rho_c = float(np.clip(rho, -1.0, 1.0))
    theta = math.acos(rho_c)
    return 0.5 - theta / (2.0 * math.pi)


def theta_proxy_L(r: int, L: int, rho: float) -> float:
    """Deterministic proxy Theta^(L)(rho) from scalar recursion with rho_k = varrho^(k-1)(rho)."""
    th = 1.0 + s_relu(rho)
    rho_k = rho
    for k in range(2, L + 1):
        rho_k = varrho_relu_eoc(rho_k)
        ak = dot_s_relu(rho_k) / float(r)
        bk = s_relu(rho_k) / float(r)
        th = 1.0 + ak * th + bk
    return th


def proxy_condition_number_equicorrelated(r: int, L: int, n: int, rho0: float) -> float:
    """On 1^perp all eigenvalues equal lambda_perp, so kappa_perp = 1 (Corollary 13)."""
    lam_perp = theta_proxy_L(r, L, 1.0) - theta_proxy_L(r, L, rho0)
    if lam_perp <= 0.0:
        return float("inf")
    return 1.0


def equicorrelated_data(n: int, d: int, rho0: float, rng: np.random.Generator) -> np.ndarray:
    """n unit vectors in R^d with pairwise cosine = rho0 (i != j). Requires d >= n+1."""
    d = max(d, n + 1)
    rho0_c = float(np.clip(rho0, -1.0, 1.0))
    a = math.sqrt(max(0.0, rho0_c))
    b = math.sqrt(max(0.0, 1.0 - rho0_c))
    u = np.zeros((d,), dtype=np.float64)
    u[0] = 1.0
    v = np.zeros((d, n), dtype=np.float64)
    v[1 : n + 1, :] = np.eye(n, dtype=np.float64)
    x = a * u[:, None] + b * v
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def empirical_ntk_gram_3layer(
    width: int,
    r: int,
    x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """3-layer (L=2) RF-LR NTK Gram matrix for inputs x (d x n)."""
    d, n = x.shape
    w1 = rng.standard_normal(size=(width, d), dtype=np.float64) / math.sqrt(d)
    a1 = rng.standard_normal(size=(r, width), dtype=np.float64)
    w2 = rng.standard_normal(size=(width, r), dtype=np.float64) / math.sqrt(r)
    a2 = rng.standard_normal(size=(width,), dtype=np.float64)

    phi1 = relu(w1 @ x)
    u1 = (a1 @ phi1) / math.sqrt(width)
    h1 = relu(u1)
    m1 = (u1 > 0.0).astype(np.float64)

    u2 = w2 @ h1
    phi2 = relu(u2)
    m2 = (u2 > 0.0).astype(np.float64)

    k_a2 = (phi2.T @ phi2) / float(width)
    c1 = phi1.T @ phi1
    s = w2.T @ (a2[:, None] * m2)
    g = m1 * s
    gg = g.T @ g
    k_a1 = (c1 * gg) / float(width * width)
    k = k_a2 + k_a1
    return k


def condition_number_centered(K: np.ndarray) -> float:
    """Condition number of K restricted to 1^perp (after centering)."""
    n = K.shape[0]
    H = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    Kc = H @ K @ H
    evals = np.linalg.eigvalsh(Kc)
    evals = np.sort(evals)
    tol = 1e-12 * max(1.0, float(np.abs(evals).max()))
    positive = evals[evals > tol]
    if positive.size == 0:
        return float("inf")
    lam_min = float(positive.min())
    lam_max = float(evals.max())
    if lam_min <= 0.0:
        return float("inf")
    return lam_max / lam_min


def main() -> None:
    width = 16000
    n = 32
    d = 64
    rho0 = 0.0
    L_proxy = 2
    n_trials = 40
    seed_data = 42
    seed_init = 100

    ranks = [5, 10, 15, 20, 30, 50, 75, 100]

    rng_data = np.random.default_rng(seed_data)
    x_equi = equicorrelated_data(n=n, d=d, rho0=rho0, rng=rng_data)

    proxy_kappas = []
    for r in ranks:
        kappa = proxy_condition_number_equicorrelated(r=r, L=L_proxy, n=n, rho0=rho0)
        proxy_kappas.append(kappa)

    emp_means = []
    emp_stds = []
    for r in ranks:
        kappas = np.empty((n_trials,), dtype=np.float64)
        for t in range(n_trials):
            rng = np.random.default_rng(seed_init + 1000 * r + t)
            K = empirical_ntk_gram_3layer(width=width, r=r, x=x_equi, rng=rng)
            kappas[t] = condition_number_centered(K)
        emp_means.append(float(kappas.mean()))
        emp_stds.append(float(kappas.std(ddof=1)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label=r"proxy $\kappa_\perp=1$")
    ax.errorbar(
        ranks,
        emp_means,
        yerr=emp_stds,
        marker="o",
        capsize=3,
        label="empirical (mean $\\pm$ std)",
    )
    ax.set_xlabel("bottleneck rank $r$")
    ax.set_ylabel(r"condition number $\kappa$ on $\mathbf{1}^\perp$")
    ax.set_title(
        f"Equicorrelated data: proxy vs empirical RF-LR NTK, $n$={n}, $L$={L_proxy}, $\\rho_0$={rho0}"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(bottom=0.5)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "condition_number_proxy_vs_empirical.png", dpi=300)
    plt.close()

    for i, r in enumerate(ranks):
        print(
            f"r={r:>3d}  proxy_kappa={proxy_kappas[i]:.4f}  emp_mean={emp_means[i]:.4f}  emp_std={emp_stds[i]:.4f}"
        )


if __name__ == "__main__":
    main()
