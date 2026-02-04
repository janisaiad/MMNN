"""
Condition number of empirical RF-LR NTK for non-equicorrelated (clustered) data.

Illustrates that the proxy lower bound κ ≥ Ω(r·L) can be large and that empirical κ
need not approach 1 when data have varying ρ_ij (e.g. cluster structure).
Contrast with equicorrelated data where κ⊥ = 1.

Paper: refs/colt2026/depth_scaling.tex (Proposition 6, condition number lower bound).
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
    return k_a2 + k_a1


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


def clustered_sphere_data(
    n: int,
    d: int,
    n_clusters: int,
    within_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate n points on S^{d-1} in n_clusters.
    Points within a cluster have high correlation; between clusters, lower.
    within_sd: std of perturbation within cluster (smaller = higher within-cluster corr).
    """
    centers = rng.standard_normal(size=(d, n_clusters), dtype=np.float64)
    norms = np.linalg.norm(centers, axis=0, keepdims=True)
    centers = centers / np.maximum(norms, 1e-12)
    n_per = n // n_clusters
    extra = n - n_per * n_clusters
    sizes = [n_per + (1 if i < extra else 0) for i in range(n_clusters)]
    cols = []
    for c in range(n_clusters):
        center = centers[:, c : c + 1]
        pert = rng.standard_normal(size=(d, sizes[c]), dtype=np.float64) * within_sd
        pts = center + pert
        norms_pt = np.linalg.norm(pts, axis=0, keepdims=True)
        pts = pts / np.maximum(norms_pt, 1e-12)
        cols.append(pts)
    x = np.hstack(cols)
    return x


def main() -> None:
    width = 20000  # 2e4 for slower convergence (non-equicorrelated)
    n = 48
    d = 64
    n_clusters = 4
    within_sd = 0.3  # smaller = higher within-cluster correlation
    n_trials = 30
    seed_data = 444
    seed_init = 555

    ranks = [10, 15, 20, 30, 50, 75, 100, 200, 1000, 2000, 5000, 10000, 100000]  # up to e4, e5

    rng_data = np.random.default_rng(seed_data)
    x = clustered_sphere_data(
        n=n,
        d=d,
        n_clusters=n_clusters,
        within_sd=within_sd,
        rng=rng_data,
    )
    rho_mat = x.T @ x
    np.fill_diagonal(rho_mat, 1.0)
    rho_off = rho_mat - np.eye(n)
    rho_flat = rho_off[np.triu_indices(n, k=1)]
    print(
        f"non-equicorrelated (clustered): n={n}, d={d}, n_clusters={n_clusters}, "
        f"within_sd={within_sd}, rho_ij range [{float(rho_flat.min()):.3f}, {float(rho_flat.max()):.3f}]"
    )

    emp_means = []
    emp_stds = []
    for r in ranks:
        kappas = np.empty((n_trials,), dtype=np.float64)
        for t in range(n_trials):
            rng = np.random.default_rng(seed_init + 1000 * r + t)
            K = empirical_ntk_gram_3layer(width=width, r=r, x=x, rng=rng)
            kappas[t] = condition_number_centered(K)
        emp_means.append(float(kappas.mean()))
        emp_stds.append(float(kappas.std(ddof=1)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label=r"$\kappa_\perp=1$ (equicorrelated)")
    ax.errorbar(
        ranks,
        emp_means,
        yerr=emp_stds,
        marker="o",
        capsize=3,
        label="empirical (mean $\\pm$ std)",
    )
    ax.set_xlabel("bottleneck rank $r$", fontsize=11)
    ax.set_ylabel(r"condition number $\kappa$ on $\mathbf{1}^\perp$", fontsize=11)
    ax.set_title(
        f"Non-equicorrelated (clustered) data: $n$={n}, $d$={d}, "
        f"$L$=2, {n_clusters} clusters, $\\sigma$={within_sd}",
        fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=1.0)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "condition_number_non_equicorrelated.png", dpi=300)
    plt.close()

    for i, r in enumerate(ranks):
        print(f"r={r:>3d}  emp_mean={emp_means[i]:.4f}  emp_std={emp_stds[i]:.4f}")


if __name__ == "__main__":
    main()
