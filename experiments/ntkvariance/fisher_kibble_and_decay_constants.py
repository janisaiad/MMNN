import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.special import hyp2f1, iv  # type: ignore
except Exception as e:
    hyp2f1 = None
    iv = None
    _SCIPY_IMPORT_ERROR = e
else:
    _SCIPY_IMPORT_ERROR = None


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


def fisher_pdf(rho_hat: np.ndarray, rho: float, r: int) -> np.ndarray:
    if hyp2f1 is None:
        raise RuntimeError(f"scipy is required for hyp2f1, import error: {_SCIPY_IMPORT_ERROR}")
    if r <= 2:
        raise ValueError("we need r > 2 for the Fisher density formula used in the paper")
    rho_hat_c = np.clip(rho_hat, -1.0 + 1e-12, 1.0 - 1e-12)
    rho_c = float(np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12))

    const_log = (
        math.log(r - 2.0)
        + math.lgamma(r - 1.0)
        - 0.5 * math.log(2.0 * math.pi)
        - math.lgamma(r - 0.5)
        + 0.5 * (r - 1.0) * math.log(1.0 - rho_c * rho_c)
    )

    log_part = const_log + 0.5 * (r - 4.0) * np.log(1.0 - rho_hat_c * rho_hat_c) - (r - 1.5) * np.log(
        1.0 - rho_c * rho_hat_c
    )
    z = (1.0 + rho_c * rho_hat_c) / 2.0
    hyper = hyp2f1(0.5, 0.5, r - 0.5, z)
    return np.exp(log_part) * hyper


def kibble_density(u: np.ndarray, v: np.ndarray, rho: float, r: int) -> np.ndarray:
    if iv is None:
        raise RuntimeError(f"scipy is required for Bessel iv, import error: {_SCIPY_IMPORT_ERROR}")
    if r <= 0:
        raise ValueError("we need r >= 1")
    rho_c = float(np.clip(rho, 1e-6, 1.0 - 1e-12))

    uu = np.maximum(u, 1e-300)
    vv = np.maximum(v, 1e-300)
    nu = r / 2.0 - 1.0

    log_num = (r / 4.0 - 0.5) * (np.log(uu) + np.log(vv)) - (uu + vv) / (2.0 * (1.0 - rho_c * rho_c))
    log_den = (
        math.lgamma(r / 2.0)
        + (r / 2.0 + 1.0) * math.log(2.0 * (1.0 - rho_c * rho_c))
        + (r / 4.0 - 0.5) * math.log(rho_c)
    )
    arg = (rho_c * np.sqrt(uu * vv)) / (1.0 - rho_c * rho_c)
    return np.exp(log_num - log_den) * iv(nu, arg)


def I_of_r(r: int) -> float:
    if r <= 2:
        return float("nan")
    rr = float(r)
    log_pref = math.log(rr - 2.0) + (rr - 2.5) * math.log(2.0) - 0.5 * math.log(2.0 * math.pi)
    log_g = (
        math.lgamma(rr / 2.0 - 1.0)
        + 2.0 * math.lgamma(rr - 0.5)
        + math.lgamma(rr - 1.5)
        - 3.0 * math.lgamma(rr - 1.0)
    )
    return float(math.exp(log_pref + log_g))


def c1_of_r(r: int) -> float:
    rr = float(r)
    i_r = I_of_r(r)
    return -(1.0 / rr) * (2.0 / math.pi + (math.sqrt(2.0) / (2.0 * math.pi)) * i_r)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    rho = 0.3
    r_list = [5, 10, 20, 50]

    grid = np.linspace(-0.999, 0.999, 2000, dtype=np.float64)
    plt.figure(figsize=(7, 5))
    for r in r_list:
        pdf = fisher_pdf(grid, rho=rho, r=r)
        pdf = pdf / np.trapz(pdf, grid)
        plt.plot(grid, pdf, linewidth=2.0, label=f"r={r}")
    plt.xlabel(r"$\hat{\rho}$")
    plt.ylabel("density")
    plt.title(rf"Fisher density $p(\hat{{\rho}}\mid \rho={rho})$")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "fisher_density_curves.png", dpi=300)

    r_kibble = 20
    rho_kibble = 0.3
    u_max = 2.5 * r_kibble
    u = np.linspace(1e-6, u_max, 250, dtype=np.float64)
    v = np.linspace(1e-6, u_max, 250, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    dens = kibble_density(uu, vv, rho=rho_kibble, r=r_kibble)
    dens = dens / np.trapz(np.trapz(dens, u, axis=1), v, axis=0)
    plt.figure(figsize=(7, 5))
    levels = np.quantile(dens[dens > 0.0], [0.50, 0.70, 0.85, 0.93, 0.97, 0.99])
    plt.contour(uu, vv, dens, levels=levels, linewidths=1.5)
    plt.xlabel(r"$u=\|x_1\|^2$")
    plt.ylabel(r"$v=\|y_1\|^2$")
    plt.title(rf"Kibble density contours, r={r_kibble}, $\rho$={rho_kibble}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "kibble_density_contours.png", dpi=300)

    r_grid = np.arange(3, 201, dtype=int)
    i_vals = np.array([I_of_r(int(r)) for r in r_grid], dtype=np.float64)
    c1_vals = np.array([c1_of_r(int(r)) for r in r_grid], dtype=np.float64)

    plt.figure(figsize=(7, 5))
    plt.loglog(r_grid, i_vals, linewidth=2.0, label=r"$I(r)$")
    plt.loglog(r_grid, np.sqrt(r_grid) ** (-1), linestyle="--", linewidth=1.5, label=r"reference $r^{-1/2}$")
    plt.xlabel("r")
    plt.ylabel("value")
    plt.title(r"asymptotic scaling $I(r)\sim C r^{-1/2}$")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "I_of_r_scaling.png", dpi=300)

    plt.figure(figsize=(7, 5))
    plt.plot(r_grid, np.sqrt(r_grid.astype(np.float64)) * i_vals, linewidth=2.0)
    plt.xlabel("r")
    plt.ylabel(r"$\sqrt{r}\,I(r)$")
    plt.title(r"checking $\sqrt{r} I(r)$ tends to a constant")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "sqrt_r_I_of_r.png", dpi=300)

    plt.figure(figsize=(7, 5))
    plt.loglog(r_grid, np.abs(2.0 * c1_vals), linewidth=2.0, label=r"$|2c_1(r)|$ (even-parity amplitude)")
    plt.loglog(r_grid, 1.0 / r_grid, linestyle="--", linewidth=1.5, label=r"reference $r^{-1}$")
    plt.xlabel("r")
    plt.ylabel("value")
    plt.title(r"spectral decay prefactor scaling with r (up to a dimension constant)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "spectral_decay_prefactor_vs_r.png", dpi=300)


if __name__ == "__main__":
    main()

