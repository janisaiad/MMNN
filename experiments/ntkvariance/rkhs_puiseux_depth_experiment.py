"""
RKHS Puiseux exponent vs depth: numerical test for L >= 4.

For zonal kernels on the sphere, the RKHS is controlled by the Puiseux exponent
at the endpoint rho=1 (Bietti-Bach). If K(1-t) - K(1) ~ c*t^gamma for small t,
then gamma determines the RKHS. The paper proves gamma=1/2 for the mean 3-layer
RF-LR kernel (same as shallow ReLU); extension to L>=4 is open.

This script estimates gamma(L) for the deterministic proxy kernel Theta^{(L)}(rho)
via log-log regression of the gap Theta^{(L)}(1) - Theta^{(L)}(1-t) vs t.
If gamma(L) ~ 1/2 for L=2,3,4,5,6, that suggests RKHS equivalence may extend;
if gamma changes with L, that suggests different RKHS at depth.

Paper: refs/colt2026/rkhs.tex (Corollary thm:no_rkhs_advantage, open for L>=4).
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

plt.rcParams["figure.figsize"] = [7, 5]
plt.rcParams["font.size"] = 16
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True


def varrho_relu_eoc(rho: float, delta_phi: float = 0.5) -> float:
    """ReLU EOC correlation map."""
    rho_c = float(np.clip(rho, -1.0, 1.0))
    term = math.sqrt(max(0.0, 1.0 - rho_c * rho_c)) - rho_c * math.acos(rho_c)
    return rho_c + delta_phi * (2.0 / math.pi) * term


def s_relu(rho: float) -> float:
    """ReLU base kernel (scalar)."""
    rho_c = float(np.clip(rho, -1.0, 1.0))
    theta = math.acos(rho_c)
    return ((math.pi - theta) * math.cos(theta) + math.sin(theta)) / (2.0 * math.pi)


def dot_s_relu(rho: float) -> float:
    """ReLU derivative kernel (scalar)."""
    rho_c = float(np.clip(rho, -1.0, 1.0))
    theta = math.acos(rho_c)
    return 0.5 - theta / (2.0 * math.pi)


def proxy_theta_L(rho1: float, L: int, r: int) -> float:
    """
    Deterministic proxy Theta^{(L)}(rho_1) from the RF-LR recursion.
    Uses the same recursion as confirm_depth_scaling.compute_thm15_sequences.
    """
    rho = np.empty((L + 1,), dtype=np.float64)
    theta_off = np.empty((L + 1,), dtype=np.float64)
    rho[1] = rho1
    theta_off[1] = 1.0 + s_relu(rho1)
    rr = float(r)
    for k in range(2, L + 1):
        rho[k] = varrho_relu_eoc(float(rho[k - 1]))
        ak = dot_s_relu(float(rho[k])) / rr
        bk = 1.0 + s_relu(float(rho[k])) / rr
        theta_off[k] = 1.0 + ak * float(theta_off[k - 1]) + (bk - 1.0)
    return float(theta_off[L])


def shallow_relu_ntk(rho: float) -> float:
    """Shallow (L=1) ReLU NTK: Theta^{(1)}(rho) = 1 + Sigma^{(1)}(rho)."""
    return 1.0 + s_relu(rho)


def estimate_puiseux_exponent(
    t_vals: np.ndarray,
    gap_vals: np.ndarray,
    min_gap: float = 1e-12,
):
    """
    Log-log regression: log(gap) ~ gamma * log(t) + const.
    Returns (gamma, r_squared). Uses only points with gap > min_gap.
    """
    mask = gap_vals > min_gap
    if mask.sum() < 3:
        return float("nan"), float("nan")
    log_t = np.log(t_vals[mask])
    log_gap = np.log(gap_vals[mask])
    # least squares: log_gap = gamma * log_t + c
    A = np.column_stack([log_t, np.ones_like(log_t)])
    coeffs, residuals, rank, s = np.linalg.lstsq(A, log_gap, rcond=None)
    gamma = float(coeffs[0])
    ss_res = np.sum((log_gap - A @ coeffs) ** 2)
    ss_tot = np.sum((log_gap - np.mean(log_gap)) ** 2)
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return gamma, r_sq


def main() -> None:
    r = 20  # we use a single rank; exponent should be independent of r for the proxy
    depths = [2, 3, 4, 5, 6]
    t_vals = np.logspace(-4, -1.3, num=25)  # t from 1e-4 to ~0.05
    gamma_ref = 0.5  # Bietti-Bach: shallow ReLU has exponent 1/2

    # compute Theta^{(L)}(1) and Theta^{(L)}(1-t) for each L
    theta_diag = {L: proxy_theta_L(1.0, L, r) for L in depths}
    shallow_diag = shallow_relu_ntk(1.0)

    results = {}
    for L in depths:
        gaps = np.array(
            [theta_diag[L] - proxy_theta_L(1.0 - t, L, r) for t in t_vals]
        )
        gamma, r_sq = estimate_puiseux_exponent(t_vals, gaps)
        results[L] = {"gamma": gamma, "r_sq": r_sq, "gaps": gaps}
        print(f"L={L}  gamma={gamma:.4f}  R^2={r_sq:.4f}")

    # shallow ReLU (L=1) as reference
    shallow_gaps = np.array(
        [shallow_diag - shallow_relu_ntk(1.0 - t) for t in t_vals]
    )
    gamma_shallow, r_sq_shallow = estimate_puiseux_exponent(t_vals, shallow_gaps)
    results[1] = {"gamma": gamma_shallow, "r_sq": r_sq_shallow, "gaps": shallow_gaps}
    print(f"L=1 (shallow)  gamma={gamma_shallow:.4f}  R^2={r_sq_shallow:.4f}")

    # plot: estimated gamma vs L
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax1 = axes[0]
    L_all = [1] + depths
    gammas = [results[L]["gamma"] for L in L_all]
    r_sqs = [results[L]["r_sq"] for L in L_all]
    ax1.plot(L_all, gammas, "o-", linewidth=2, markersize=10, label=r"$\gamma(L)$")
    ax1.axhline(y=gamma_ref, color="gray", linestyle="--", linewidth=1.5, label=r"$\gamma=1/2$ (ReLU RKHS)")
    ax1.set_xlabel("depth $L$ (number of bottleneck layers)")
    ax1.set_ylabel(r"estimated Puiseux exponent $\gamma$")
    ax1.set_title(r"RKHS Puiseux exponent vs depth ($r$={})".format(r))
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(L_all)

    ax2 = axes[1]
    # log-log of gap vs t for L=2 and L=6 (representative)
    for L in [2, 6]:
        g = results[L]["gaps"]
        ax2.loglog(t_vals, np.maximum(g, 1e-15), "o-", label=f"L={L}", alpha=0.8)
    # reference slope 1/2
    t_ref = np.array([1e-4, 0.05])
    c_ref = results[2]["gaps"][0] / (t_vals[0] ** 0.5)
    ax2.loglog(t_ref, c_ref * t_ref**0.5, "k--", linewidth=1.5, label=r"$t^{1/2}$ ref")
    ax2.set_xlabel(r"$t = 1 - \rho$")
    ax2.set_ylabel(r"$\Theta^{(L)}(1) - \Theta^{(L)}(1-t)$")
    ax2.set_title("Gap near endpoint (log-log)")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "rkhs_puiseux_exponent_vs_depth.png", dpi=300)
    plt.close()

    print("\nConclusion: if gamma(L) ~ 0.5 for all L, the proxy suggests RKHS equivalence extends to L>=4.")
    print("If gamma(L) deviates from 0.5 at large L, the deep kernel may induce a different RKHS.")


if __name__ == "__main__":
    main()
