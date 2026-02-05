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
plt.rcParams["font.weight"] = "normal"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.rm"] = "serif"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 22
mpl.rcParams["axes.formatter.limits"] = (-6, 6)
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
mpl.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
mpl.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sample_unit_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(size=(d, n), dtype=np.float64)
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def centered_min_eig_from_ntk(width: int, r: int, x: np.ndarray, rng: np.random.Generator) -> float:
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

    h = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    kc = h @ k @ h

    evals = np.linalg.eigvalsh(kc)
    tol = 1e-12 * max(1.0, float(evals.max(initial=1.0)))
    positive = evals[evals > tol]
    if positive.size == 0:
        return 0.0
    return float(positive.min())


def main() -> None:
    width = 16000
    ranks = [5, 10, 20, 50, 100, 200, 500, 1000]
    n = 64
    d = 16
    n_trials = 50
    seed_data = 0
    seed_init = 1

    rng_data = np.random.default_rng(seed_data)
    x = sample_unit_sphere(n=n, d=d, rng=rng_data)

    results = {}
    for r in ranks:
        values = np.empty((n_trials,), dtype=np.float64)
        for t in range(n_trials):
            rng = np.random.default_rng(seed_init + 1000 * r + t)
            values[t] = centered_min_eig_from_ntk(width=width, r=r, x=x, rng=rng)
        results[r] = values

    plt.figure(figsize=(7, 5))
    bins = 30
    for r in ranks:
        vals = results[r]
        plt.hist(vals, bins=bins, alpha=0.35, density=True, label=f"r={r}")
    plt.xlabel(r"$\lambda_{\min}^{+}(HKH)$")
    plt.ylabel("density")
    plt.title(f"distribution of smallest nonzero centered NTK eigenvalue, width={width}, n={n}, d={d}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "min_eig_distribution.png", dpi=300)

    plt.figure(figsize=(6, 4))
    for r in ranks:
        vals = results[r]
        plt.plot(np.sort(vals), np.linspace(0.0, 1.0, num=vals.size, endpoint=False), linewidth=2.0, label=f"r={r}")
    plt.xlabel(r"$\lambda_{\min}^{+}(HKH)$")
    plt.ylabel("empirical cdf")
    plt.title("empirical cdf across initializations")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "min_eig_cdf.png", dpi=300)

    means = [float(results[r].mean()) for r in ranks]
    stds = [float(results[r].std(ddof=1)) for r in ranks]
    plt.figure(figsize=(5, 4))
    plt.errorbar(ranks, means, yerr=stds, marker="o", capsize=3)
    plt.xlabel("rank $r$", fontsize=11)
    plt.ylabel(r"mean of $\lambda_{\min}^{+}(HKH)$", fontsize=11)
    plt.title(f"mean smallest positive centered eigenvalue vs rank, width={width}, n={n}, d={d}", fontsize=11)
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "min_eig_mean_vs_r.png", dpi=300)
    plt.close()

    for r in ranks:
        vals = results[r]
        print(f"r={r:>3d}  mean={float(vals.mean()):.6e}  median={float(np.median(vals)):.6e}  min={float(vals.min()):.6e}")


if __name__ == "__main__":
    main()

