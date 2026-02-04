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


def varrho_relu_eoc(rho: float, delta_phi: float = 0.5) -> float:
    rho_c = float(np.clip(rho, -1.0, 1.0))
    term = math.sqrt(max(0.0, 1.0 - rho_c * rho_c)) - rho_c * math.acos(rho_c)
    return rho_c + delta_phi * (2.0 / math.pi) * term


def s_relu(rho: float) -> float:
    rho_c = float(np.clip(rho, -1.0, 1.0))
    theta = math.acos(rho_c)
    return ((math.pi - theta) * math.cos(theta) + math.sin(theta)) / (2.0 * math.pi)


def dot_s_relu(rho: float) -> float:
    rho_c = float(np.clip(rho, -1.0, 1.0))
    theta = math.acos(rho_c)
    return 0.5 - theta / (2.0 * math.pi)


def theta_star(r: int) -> float:
    rr = float(r)
    return (2.0 * rr + 1.0) / (2.0 * rr - 1.0)


def compute_thm15_sequences(r: int, rho1: float, k_max: int):
    rho = np.empty((k_max + 1,), dtype=np.float64)
    w = np.empty((k_max + 1,), dtype=np.float64)
    theta_off = np.empty((k_max + 1,), dtype=np.float64)
    theta_diag = np.empty((k_max + 1,), dtype=np.float64)

    rho[1] = rho1
    z1 = (1.0 - rho1) / 2.0
    w[1] = 1.0 / math.sqrt(max(z1, 1e-300))
    theta_off[1] = 1.0 + s_relu(rho1)
    theta_diag[1] = 1.0 + s_relu(1.0)

    a_diag = dot_s_relu(1.0) / float(r)
    b_diag = 1.0 + s_relu(1.0) / float(r)

    for k in range(2, k_max + 1):
        rho[k] = varrho_relu_eoc(float(rho[k - 1]))
        zk = (1.0 - float(rho[k])) / 2.0
        w[k] = 1.0 / math.sqrt(max(zk, 1e-300))

        ak = dot_s_relu(float(rho[k])) / float(r)
        bk = 1.0 + s_relu(float(rho[k])) / float(r)
        theta_off[k] = 1.0 + ak * float(theta_off[k - 1]) + (bk - 1.0)

        theta_diag[k] = 1.0 + a_diag * float(theta_diag[k - 1]) + (b_diag - 1.0)

    return {
        "rho": rho,
        "w": w,
        "theta_off": theta_off,
        "theta_diag": theta_diag,
    }


def main() -> None:
    ranks = [5, 10, 20, 50]
    rho1 = 0.0
    k_max = 4000

    fig1, axs1 = plt.subplots(2, 2, figsize=(11, 8))
    axs1 = axs1.reshape(-1)

    for r in ranks:
        seq = compute_thm15_sequences(r=r, rho1=rho1, k_max=k_max)
        ks = np.arange(1, k_max + 1, dtype=np.float64)
        rho = seq["rho"][1:]
        w = seq["w"][1:]
        theta_off = seq["theta_off"][1:]
        theta_diag = seq["theta_diag"][1:]
        delta = theta_diag - theta_off

        axs1[0].plot(ks, w / ks, linewidth=1.8, label=f"r={r}")
        axs1[1].plot(ks, (1.0 - rho) * (ks**2), linewidth=1.8, label=f"r={r}")

        ths = theta_star(r)
        axs1[2].plot(ks, np.abs(theta_off - ths) * ks, linewidth=1.8, label=f"r={r}")
        axs1[3].plot(ks, delta * ks * float(r), linewidth=1.8, label=f"r={r}")

    axs1[0].set_title(r"$w_k/k$ (should stabilize)")
    axs1[0].set_xlabel("k")
    axs1[0].grid(True, linestyle="--", linewidth=0.5)
    axs1[0].legend(fontsize=10)

    axs1[1].set_title(r"$(1-\rho_k)\,k^2$ (bounded)")
    axs1[1].set_xlabel("k")
    axs1[1].grid(True, linestyle="--", linewidth=0.5)
    axs1[1].legend(fontsize=10)

    axs1[2].set_title(r"$k\,|\Theta^{(k)}(\rho_1)-\Theta_\star(r)|$ (bounded)")
    axs1[2].set_xlabel("k")
    axs1[2].grid(True, linestyle="--", linewidth=0.5)
    axs1[2].legend(fontsize=10)

    axs1[3].set_title(r"$r\,k\,(\Theta_{\mathrm{diag}}^{(k)}-\Theta_{\mathrm{off}}^{(k)})$ (stabilizes)")
    axs1[3].set_xlabel("k")
    axs1[3].grid(True, linestyle="--", linewidth=0.5)
    axs1[3].legend(fontsize=10)

    fig1.suptitle(f"theorem 15 diagnostics, rho1={rho1}, k_max={k_max}", y=1.02)
    fig1.tight_layout()
    out_dir = Path(__file__).resolve().parent
    fig1.savefig(out_dir / "confirm_thm15.png", dpi=300)

    n = 64
    rho0 = 0.0
    l_values = np.unique(np.round(np.logspace(1.0, math.log10(k_max), num=40)).astype(int))

    fig2, axs2 = plt.subplots(1, 2, figsize=(11, 4))

    for r in ranks:
        seq = compute_thm15_sequences(r=r, rho1=rho0, k_max=int(l_values.max()))
        theta_off = seq["theta_off"]
        theta_diag = seq["theta_diag"]

        lam1 = theta_diag[l_values] + (n - 1.0) * theta_off[l_values]
        lam_perp = theta_diag[l_values] - theta_off[l_values]

        axs2[0].plot(l_values, np.abs(lam1 / float(n) - theta_star(r)) * l_values / float(n), linewidth=1.8, label=f"r={r}")
        axs2[1].plot(l_values, lam_perp * (l_values.astype(np.float64)) * float(r), linewidth=1.8, label=f"r={r}")

    axs2[0].set_xscale("log")
    axs2[0].set_yscale("log")
    axs2[0].set_title(r"spike saturation: $L\,| \lambda_1/n-\Theta_\star(r)|/n$")
    axs2[0].set_xlabel("L")
    axs2[0].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs2[0].legend(fontsize=10)

    axs2[1].set_xscale("log")
    axs2[1].set_title(r"gap scaling: $r\,L\,\lambda_\perp$")
    axs2[1].set_xlabel("L")
    axs2[1].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs2[1].legend(fontsize=10)

    fig2.suptitle(f"proposition 17 scalings, n={n}, rho0={rho0}", y=1.05)
    fig2.tight_layout()
    fig2.savefig(out_dir / "confirm_prop17.png", dpi=300)


if __name__ == "__main__":
    main()

