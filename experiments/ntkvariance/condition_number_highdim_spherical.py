"""
Condition number of proxy vs empirical RF-LR NTK for i.i.d. uniform data on S^{d-1} (high d).

Validates Corollary 13 (second item): with high probability max |rho_ij| = O(1/sqrt(d)),
so data are approximately equicorrelated and kappa_perp = 1+o(1); empirical concentrates as r grows.

Paper: refs/colt2026/depth_scaling.tex (Corollary 13, high-dimensional random data).
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
    th = 1.0 + s_relu(rho)
    rho_k = rho
    for k in range(2, L + 1):
        rho_k = varrho_relu_eoc(rho_k)
        ak = dot_s_relu(rho_k) / float(r)
        bk = s_relu(rho_k) / float(r)
        th = 1.0 + ak * th + bk
    return th


def sample_unit_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(size=(d, n), dtype=np.float64)
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
    return k_a2 + k_a1


def condition_number_centered(K: np.ndarray) -> float:
    n = K.shape[0]
    H = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    Kc = H @ K @ H
    evals = np.linalg.eigvalsh(Kc)
    evals = np.sort(evals)
    tol = 1e-12 * max(1.0, float(np.abs(evals).max()))
    positive = evals[evals > tol]
    if positive.size == 0:
        return float("inf")
    return float(evals.max()) / float(positive.min())


def proxy_gram_from_rho(r: int, L: int, rho_mat: np.ndarray) -> np.ndarray:
    """Proxy Gram M_ij = Theta^(L)(rho_ij). rho_mat is n x n with rho_ii=1."""
    n = rho_mat.shape[0]
    M = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = theta_proxy_L(r, L, float(rho_mat[i, j]))
    return M


def main() -> None:
    width = 16000
    n = 32
    L_val = 2
    n_trials = 30
    seed_data = 123
    seed_init = 200

    ranks = [10, 15, 20, 30, 50, 75, 100, 200]
    dims = [(256, "C0", "C0"), (128, "red", "red")]  # (d, proxy_color, empirical_color); d=256 both blue

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label=r"$\kappa_\perp=1$ (equicorrelated limit)")

    for d, proxy_color, emp_color in dims:
        rng_data = np.random.default_rng(seed_data + d)
        x = sample_unit_sphere(n=n, d=d, rng=rng_data)
        rho_mat = x.T @ x
        np.fill_diagonal(rho_mat, 1.0)
        max_off = float(np.max(np.abs(rho_mat - np.eye(n))))
        print(f"high-dim spherical: d={d}, n={n}, max_{{|rho_ij|}} (i!=j) = {max_off:.4f} (expect O(1/sqrt(d)) ~ {1/math.sqrt(d):.4f})")

        proxy_kappas = []
        for r in ranks:
            M = proxy_gram_from_rho(r=r, L=L_val, rho_mat=rho_mat)
            kappa = condition_number_centered(M)
            proxy_kappas.append(kappa)

        emp_means = []
        emp_stds = []
        for r in ranks:
            kappas = np.empty((n_trials,), dtype=np.float64)
            for t in range(n_trials):
                rng = np.random.default_rng(seed_init + 1000 * r + t + d * 10000)
                K = empirical_ntk_gram_3layer(width=width, r=r, x=x, rng=rng)
                kappas[t] = condition_number_centered(K)
            emp_means.append(float(kappas.mean()))
            emp_stds.append(float(kappas.std(ddof=1)))

        ax.plot(ranks, proxy_kappas, marker="s", color=proxy_color, label=f"proxy, d={d}")
        ax.errorbar(
            ranks,
            emp_means,
            yerr=emp_stds,
            marker="o",
            capsize=3,
            color=emp_color,
            label=f"empirical d={d} (mean ± std)",
        )
        for i, r in enumerate(ranks):
            print(f"d={d} r={r:>3d}  proxy_kappa={proxy_kappas[i]:.4f}  emp_mean={emp_means[i]:.4f}  emp_std={emp_stds[i]:.4f}")

    ax.set_xlabel("bottleneck rank $r$", fontsize=10)
    ax.set_ylabel(r"condition number $\kappa$ on $\mathbf{1}^\perp$", fontsize=10)
    ax.set_title(f"High-dim spherical: i.i.d. uniform on $S^{{d-1}}$, $n$={n}, $L$={L_val}", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0.5)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "condition_number_highdim_spherical.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
