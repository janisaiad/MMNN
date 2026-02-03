import math

import matplotlib.pyplot as plt
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def ntk_entry_pair(width: int, r: int, x: float, xp: float, rng: np.random.Generator) -> float:
    w1 = rng.standard_normal(size=(width,), dtype=np.float64)
    a1 = rng.standard_normal(size=(r, width), dtype=np.float64)
    w2 = rng.standard_normal(size=(width, r), dtype=np.float64) / math.sqrt(r)
    a2 = rng.standard_normal(size=(width,), dtype=np.float64)

    phi1_x = relu(w1 * x)
    phi1_xp = relu(w1 * xp)

    u1_x = (a1 @ phi1_x) / math.sqrt(width)
    u1_xp = (a1 @ phi1_xp) / math.sqrt(width)
    h1_x = relu(u1_x)
    h1_xp = relu(u1_xp)
    m1_x = (u1_x > 0.0).astype(np.float64)
    m1_xp = (u1_xp > 0.0).astype(np.float64)

    u2_x = w2 @ h1_x
    u2_xp = w2 @ h1_xp
    phi2_x = relu(u2_x)
    phi2_xp = relu(u2_xp)
    m2_x = (u2_x > 0.0).astype(np.float64)
    m2_xp = (u2_xp > 0.0).astype(np.float64)

    ntk_a2 = float((phi2_x @ phi2_xp) / width)

    c1 = float(phi1_x @ phi1_xp)
    s_x = w2.T @ (a2 * m2_x)
    s_xp = w2.T @ (a2 * m2_xp)
    ntk_a1 = float((c1 / (width * width)) * ((m1_x * s_x) @ (m1_xp * s_xp)))

    return ntk_a2 + ntk_a1


def main() -> None:
    width = 16000
    ranks = [5, 10, 20, 50]
    n_trials = 100
    x = -1.0
    xp = 1.0
    seed = 0

    variances: list[float] = []
    means: list[float] = []

    for r in ranks:
        values = np.empty((n_trials,), dtype=np.float64)
        for t in range(n_trials):
            rng = np.random.default_rng(seed + 1000 * r + t)
            values[t] = ntk_entry_pair(width=width, r=r, x=x, xp=xp, rng=rng)
        means.append(float(values.mean()))
        variances.append(float(values.var(ddof=1)))

    plt.figure(figsize=(5, 4))
    plt.loglog(ranks, variances, marker="o")
    plt.xlabel("rank r")
    plt.ylabel("var[NTK(-1,1)]")
    plt.title(f"3-layer (2-ReLU) low-rank NTK variance, width={width}, trials={n_trials}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("/home/janis/STG3A/MMNN/experiments/ntkvariance/variance_vs_r.png", dpi=200)

    for r, m, v in zip(ranks, means, variances):
        print(f"r={r:>3d}  mean={m:.6e}  var={v:.6e}")


if __name__ == "__main__":
    main()

