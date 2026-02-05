"""
Kernel regression risk with empirical RF-LR NTK on spherical data.

Fits kernel ridge regression with the empirical 3-layer RF-LR NTK Gram;
plots test risk vs bottleneck rank r (and optionally vs depth L).
Target: linear or low-degree polynomial on the sphere.
Validates that Gram concentration as r grows improves kernel regression performance.

Paper: refs/colt2026/ (RKHS, concentration, optimization implications).
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


def sample_unit_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(size=(d, n), dtype=np.float64)
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


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


def linear_target(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """f*(x) = <w, x>, x is (d, n), w is (d,)."""
    return (w[:, None] * x).sum(axis=0)


def quadratic_target(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """f*(x) = sum_i w_i x_i^2 (low-degree polynomial)."""
    return (w[:, None] * (x**2)).sum(axis=0)


def kernel_ridge_risk(
    K_train: np.ndarray,
    K_test_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    reg: float = 1e-6,
) -> float:
    """Kernel ridge: alpha = (K_train + reg*I)^{-1} y_train, predict = K_test_train @ alpha."""
    n = K_train.shape[0]
    alpha = np.linalg.solve(K_train + reg * np.eye(n, dtype=np.float64), y_train.astype(np.float64))
    y_pred = K_test_train @ alpha
    return float(np.mean((y_pred - y_test) ** 2))


def main() -> None:
    width = 12000
    d = 32
    n_train = 64
    n_test = 256
    n_trials = 20
    reg = 1e-5
    seed_data = 777
    seed_init = 800
    target_type = "linear"  # "linear" or "quadratic"

    ranks = [5, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000]  # extend to reach ~e-3 risk

    rng_data = np.random.default_rng(seed_data)
    x_train = sample_unit_sphere(n=n_train, d=d, rng=rng_data)
    x_test = sample_unit_sphere(n=n_test, d=d, rng=rng_data)

    w_target = rng_data.standard_normal(size=(d,), dtype=np.float64)
    w_target = w_target / np.linalg.norm(w_target)
    if target_type == "linear":
        y_train = linear_target(x_train, w_target)
        y_test = linear_target(x_test, w_target)
    else:
        y_train = quadratic_target(x_train, w_target)
        y_test = quadratic_target(x_test, w_target)

    # Normalize labels to O(1) scale
    scale = np.sqrt(np.mean(y_train**2)) + 1e-10
    y_train = y_train / scale
    y_test = y_test / scale

    risks_mean = []
    risks_std = []
    for r in ranks:
        risks = np.empty((n_trials,), dtype=np.float64)
        for t in range(n_trials):
            rng = np.random.default_rng(seed_init + 1000 * r + t)
            x_all = np.hstack([x_train, x_test])
            K_all = empirical_ntk_gram_3layer(width=width, r=r, x=x_all, rng=rng)
            K_train = K_all[:n_train, :n_train]
            K_test_train = K_all[n_train:, :n_train]
            risks[t] = kernel_ridge_risk(
                K_train=K_train,
                K_test_train=K_test_train,
                y_train=y_train,
                y_test=y_test,
                reg=reg,
            )
        risks_mean.append(float(risks.mean()))
        risks_std.append(float(risks.std(ddof=1)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.errorbar(
        ranks,
        risks_mean,
        yerr=risks_std,
        marker="o",
        capsize=3,
        label="test MSE (mean $\\pm$ std)",
    )
    r0, v0 = ranks[0], risks_mean[0]
    r_ref = np.asarray(ranks, dtype=np.float64)
    v_ref = v0 * (float(r0) / r_ref) ** 0.5  # slope -0.5 in log-log: ~ 1/sqrt(r) from 1st point
    ax.plot(ranks, v_ref, linestyle="--", color="gray", linewidth=1.2, label=r"$\propto r^{-1/2}$ ref")
    ax.set_ylim(bottom=2e-1)
    ax.set_xlabel("bottleneck rank $r$", fontsize=10)
    ax.set_ylabel("test MSE (kernel ridge)", fontsize=10)
    ax.set_title(
        f"Kernel regression: RF-LR NTK, {target_type} target, $n$={n_train}, $d$={d}, $L$=2",
        fontsize=10,
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=4)
    for label in ax.get_yticklabels():
        label.set_fontsize(4)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "kernel_regression_risk_vs_r.png", dpi=300)
    plt.close()

    for i, r in enumerate(ranks):
        print(f"r={r:>3d}  risk_mean={risks_mean[i]:.6e}  risk_std={risks_std[i]:.6e}")


if __name__ == "__main__":
    main()
