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


def relu_cosine_map_eoc(rho: float, delta_phi: float = 0.5) -> float:
    rho_clipped = float(np.clip(rho, -1.0, 1.0))
    term = math.sqrt(max(0.0, 1.0 - rho_clipped * rho_clipped)) - rho_clipped * math.acos(rho_clipped)
    return rho_clipped + delta_phi * (2.0 / math.pi) * term


def relu_derivative_kernel_bar(rho: float) -> float:
    rho_clipped = float(np.clip(rho, -1.0, 1.0))
    return 0.5 - math.acos(rho_clipped) / (2.0 * math.pi)


def compute_rho_path(rho1: float, depth: int) -> np.ndarray:
    rhos = np.empty((depth + 1,), dtype=np.float64)
    rhos[1] = rho1
    for k in range(2, depth + 1):
        rhos[k] = relu_cosine_map_eoc(float(rhos[k - 1]))
    rhos[0] = np.nan
    return rhos


def compute_top_products(rhos: np.ndarray, r: int, c0: float = 0.5):
    depth = int(len(rhos) - 1)
    products = np.ones((depth,), dtype=np.float64)
    bounds = np.ones((depth,), dtype=np.float64)
    running = 1.0
    for j in range(1, depth):
        k = depth - (j - 1)
        running *= relu_derivative_kernel_bar(float(rhos[k])) / float(r)
        products[j] = running
        bounds[j] = (c0 / float(r)) ** j
    return products, bounds


def main() -> None:
    x = -1.0
    xp = 1.0
    rho1 = float((x * xp) / (abs(x) * abs(xp)))

    depth = 200
    ranks = [5, 10, 20, 50]
    c0 = 0.5

    rhos = compute_rho_path(rho1=rho1, depth=depth)
    j_values = np.arange(depth, dtype=np.int64)

    plt.figure(figsize=(7, 5))
    for r in ranks:
        products, bounds = compute_top_products(rhos=rhos, r=r, c0=c0)
        plt.semilogy(j_values[1:], products[1:], marker=".", linewidth=1.5, label=f"r={r}")
        plt.semilogy(j_values[1:], bounds[1:], linestyle="--", linewidth=1.0, label=f"(c0/r)^j, r={r}")

    plt.xlabel("j = L - ℓ (layers above ℓ)")
    plt.ylabel(r"$\prod_{k=\ell+1}^{L} \dot{\Sigma}^{(k)}$")
    plt.title(f"exponential depth suppression, ReLU EOC, x={x}, x'={xp}, L={depth}, c0={c0}", fontsize=11)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "decay_vs_depth.png", dpi=300)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, depth + 1), rhos[1:], linewidth=2.0)
    plt.xlabel("k")
    plt.ylabel(r"$\rho_k$")
    plt.title(f"correlation alignment, rho1={rho1}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "rho_vs_depth.png", dpi=300)

    ks = np.arange(1, depth + 1, dtype=np.float64)
    one_minus_rho = np.maximum(1.0 - rhos[1:], 1e-300)
    ref = one_minus_rho[0] * (ks[0] / ks) ** 2
    plt.figure(figsize=(6, 4))
    plt.loglog(ks, one_minus_rho, linewidth=2.0, label=r"$1-\rho_k$")
    plt.loglog(ks, ref, linestyle="--", linewidth=1.5, label=r"reference $k^{-2}$")
    plt.xlabel("k")
    plt.ylabel(r"$1-\rho_k$")
    plt.title(f"correlation alignment rate, rho1={rho1}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "one_minus_rho_vs_depth_loglog.png", dpi=300)


if __name__ == "__main__":
    main()

