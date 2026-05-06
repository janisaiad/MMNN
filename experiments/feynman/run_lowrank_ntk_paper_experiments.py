#!/usr/bin/env python3
"""
Run clean paper-style numerical checks for the mixed rank-width low-rank NTK note.

The script produces self-contained plots for:
  1. Haar projector covariance: Cov(G_12, G_12) vs gamma_{n,r}.
  2. Scalar endpoint constants: V/(eps L)->5, C/(eps L^2)->21/20, A/(eps L^3)->173/720.
  3. Log-log slope checks for V,C,A.
  4. RF-LR bottleneck memory suppression: (2r)^(-age).
  5. Operator scaling proxy: ||Z||_op / (L^{3/2} sqrt(m eps)).

Dependencies: numpy, matplotlib.
Example:
  python run_lowrank_ntk_paper_experiments.py --quick
  python run_lowrank_ntk_paper_experiments.py --samples 50000 --haar-samples 10000 --outdir outputs_paper
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_paper_style() -> None:
    # User-requested style block.
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.weight'] = 'normal'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 22
    mpl.rcParams['axes.formatter.limits'] = (-6, 6)
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True


def gamma_exact(n: int, r: int) -> float:
    if r == n:
        return 0.0
    return n * (n - r) / (r * (n - 1) * (n + 2))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def haar_projector_sample(n: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Sample U Haar-Stiefel via QR and return G=(n/r)UU^T."""
    z = rng.normal(size=(n, r))
    q, rr = np.linalg.qr(z, mode="reduced")
    # Fix signs so QR is Haar-Stiefel.
    signs = np.sign(np.diag(rr))
    signs[signs == 0] = 1.0
    q = q * signs
    return (n / r) * (q @ q.T)


def run_haar_covariance(n: int, ranks: Iterable[int], samples: int, rng: np.random.Generator, outdir: Path) -> list[dict]:
    rows = []
    for r in ranks:
        vals = np.empty(samples, dtype=float)
        for s in range(samples):
            g = haar_projector_sample(n, r, rng)
            vals[s] = g[0, 1]
        empirical = float(np.var(vals, ddof=1))
        exact = gamma_exact(n, r)
        rows.append({
            "n": n,
            "r": r,
            "empirical_cov_G01_G01": empirical,
            "exact_gamma": exact,
            "relative_error": abs(empirical - exact) / max(exact, 1e-15),
        })

    with open(outdir / "haar_covariance_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rs = np.array([row["r"] for row in rows], dtype=float)
    emp = np.array([row["empirical_cov_G01_G01"] for row in rows])
    exact = np.array([row["exact_gamma"] for row in rows])

    fig, ax = plt.subplots()
    ax.plot(rs, emp, marker="o", label="Monte Carlo")
    ax.plot(rs, exact, marker="s", linestyle="--", label=r"$\gamma_{n,r}$")
    ax.set_xlabel(r"rank $r$")
    ax.set_ylabel(r"$\mathrm{Cov}(G_{12},G_{12})$")
    ax.set_title(f"Haar projector covariance, n={n}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "haar_covariance")

    fig, ax = plt.subplots()
    rel = np.array([row["relative_error"] for row in rows])
    ax.semilogy(rs, np.maximum(rel, 1e-16), marker="o")
    ax.set_xlabel(r"rank $r$")
    ax.set_ylabel("relative error")
    ax.set_title("Haar covariance relative error")
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "haar_covariance_relative_error")
    return rows


def sample_endpoint_increments(num_samples: int, L: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian increments (xi,zeta) with covariance [[5,1],[1,1]]."""
    cov = np.array([[5.0, 1.0], [1.0, 1.0]])
    chol = np.linalg.cholesky(cov)
    z = rng.normal(size=(num_samples, L, 2))
    inc = z @ chol.T
    return inc[:, :, 0], inc[:, :, 1]


def scalar_mode_statistics(L: int, eps: float, samples: int, rng: np.random.Generator) -> dict:
    """
    Simulate the linearized endpoint scalar mode:
      R_L = sqrt(eps) sum_i xi_i
      T_L = sqrt(eps) sum_i L [ (1-u_i^4)/4 xi_i + u_i^4/4 zeta_i ]
    where Cov(xi,zeta)=[[5,1],[1,1]].
    Then Var(R)/(eps L)->5,
         Cov(R,T)/(eps L^2)->21/20,
         Var(T)/(eps L^3)->173/720.
    """
    xi, zeta = sample_endpoint_increments(samples, L, rng)
    sqrt_eps = math.sqrt(eps)
    R = sqrt_eps * np.sum(xi, axis=1)
    u = (np.arange(1, L + 1, dtype=float)) / float(L)
    a = 0.25 * (1.0 - u ** 4)
    b = 0.25 * u ** 4
    T = sqrt_eps * L * (xi @ a + zeta @ b)

    R_c = R - np.mean(R)
    T_c = T - np.mean(T)
    V = float(np.mean(R_c * R_c))
    C = float(np.mean(R_c * T_c))
    A = float(np.mean(T_c * T_c))
    return {
        "L": L,
        "V": V,
        "C": C,
        "A": A,
        "V_over_eps_L": V / (eps * L),
        "C_over_eps_L2": C / (eps * L ** 2),
        "A_over_eps_L3": A / (eps * L ** 3),
    }


def run_scalar_convergence(L_values: Iterable[int], eps: float, samples: int, rng: np.random.Generator, outdir: Path) -> list[dict]:
    rows = [scalar_mode_statistics(int(L), eps, samples, rng) for L in L_values]

    with open(outdir / "scalar_endpoint_convergence.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    Ls = np.array([row["L"] for row in rows], dtype=float)

    fig, ax = plt.subplots()
    ax.plot(Ls, [row["V_over_eps_L"] for row in rows], marker="o", label=r"$V_L/(\varepsilon L)$")
    ax.axhline(5.0, linestyle="--", label=r"$5$")
    ax.set_xscale("log")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel("normalized value")
    ax.set_title(r"Convergence of $V_L$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "convergence_V")

    fig, ax = plt.subplots()
    ax.plot(Ls, [row["C_over_eps_L2"] for row in rows], marker="o", label=r"$C_L/(\varepsilon L^2)$")
    ax.axhline(21.0 / 20.0, linestyle="--", label=r"$21/20$")
    ax.set_xscale("log")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel("normalized value")
    ax.set_title(r"Convergence of $D,F$ scalar mode")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "convergence_C")

    fig, ax = plt.subplots()
    ax.plot(Ls, [row["A_over_eps_L3"] for row in rows], marker="o", label=r"$A_L/(\varepsilon L^3)$")
    ax.axhline(173.0 / 720.0, linestyle="--", label=r"$173/720$")
    ax.set_xscale("log")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel("normalized value")
    ax.set_title(r"Convergence of $A,B$ scalar mode")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "convergence_A")

    # Log-log slope check.
    values = {
        "V": np.array([row["V"] for row in rows]),
        "C": np.abs(np.array([row["C"] for row in rows])),
        "A": np.array([row["A"] for row in rows]),
    }
    fig, ax = plt.subplots()
    slope_rows = {}
    for name, vals in values.items():
        coeff = np.polyfit(np.log(Ls), np.log(vals), deg=1)
        slope = float(coeff[0])
        slope_rows[name] = slope
        ax.loglog(Ls, vals, marker="o", label=fr"{name}, slope {slope:.2f}")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel("raw statistic")
    ax.set_title("Log-log depth slopes")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "loglog_slopes_scalar_modes")

    with open(outdir / "scalar_loglog_slopes.json", "w") as f:
        json.dump(slope_rows, f, indent=2)
    return rows


def run_rflr_memory(ranks: Iterable[int], max_age: int, outdir: Path) -> None:
    ages = np.arange(max_age + 1)
    fig, ax = plt.subplots()
    for r in ranks:
        memory = (1.0 / (2.0 * r)) ** ages
        ax.semilogy(ages, memory, marker=None, label=fr"$r={r}$")
    ax.set_xlabel(r"age $a=L-k$")
    ax.set_ylabel(r"memory $(2r)^{-a}$")
    ax.set_title("RF-LR geometric memory suppression")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "rflr_memory_suppression")

    # Predicted raw radial coefficients.
    rows = []
    for r in ranks:
        denom = 1.0 - 1.0 / (2.0 * r)
        rows.append({
            "r": r,
            "D_or_F_coefficient_per_eps_L_over_r": 5.0 / denom,
            "A_or_B_coefficient_per_eps_L_over_r2": 5.0 / (denom ** 2),
        })
    with open(outdir / "rflr_coefficients.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def random_symmetric_matrix(m: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(scale=sigma, size=(m, m))
    return (a + a.T) / math.sqrt(2.0)


def run_operator_proxy(L_values: Iterable[int], eps: float, m: int, reps: int, rng: np.random.Generator, outdir: Path) -> list[dict]:
    rows = []
    for L in L_values:
        sigma = math.sqrt((L ** 3) * eps)
        norms = []
        for _ in range(reps):
            z = random_symmetric_matrix(m, sigma, rng)
            norms.append(float(np.linalg.norm(z, ord=2)))
        med = float(np.median(norms))
        scale = (L ** 1.5) * math.sqrt(m * eps)
        rows.append({
            "L": int(L),
            "median_op_norm": med,
            "scale_L32_sqrt_m_eps": scale,
            "ratio": med / scale,
        })
    with open(outdir / "operator_proxy_scaling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    Ls = np.array([row["L"] for row in rows], dtype=float)
    fig, ax = plt.subplots()
    ax.plot(Ls, [row["ratio"] for row in rows], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel(r"$\|Z\|_{\rm op}/(L^{3/2}\sqrt{m\varepsilon})$")
    ax.set_title("Operator scaling proxy")
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "operator_proxy_ratio")
    return rows


def plot_design_gamma(n: int, outdir: Path) -> None:
    ranks = np.arange(1, n + 1)
    gammas = np.array([gamma_exact(n, int(r)) for r in ranks])
    fig, ax = plt.subplots()
    ax.plot(ranks / n, gammas, marker=None)
    ax.set_xlabel(r"rank fraction $r/n$")
    ax.set_ylabel(r"$\gamma_{n,r}$")
    ax.set_title(r"Rank-defect parameter")
    ax.grid(True, alpha=0.25)
    savefig(fig, outdir, "rank_defect_gamma")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="/mnt/data/lowrank_ntk_paper_outputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--quick", action="store_true", help="Use fewer Monte Carlo samples for a fast smoke test.")
    parser.add_argument("--samples", type=int, default=20000, help="Monte Carlo samples for scalar endpoint constants.")
    parser.add_argument("--haar-samples", type=int, default=3000, help="Monte Carlo samples for Haar projector covariance.")
    parser.add_argument("--operator-reps", type=int, default=80)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--m", type=int, default=64, help="Gram matrix size for operator proxy.")
    args = parser.parse_args()

    if args.quick:
        args.samples = min(args.samples, 4000)
        args.haar_samples = min(args.haar_samples, 600)
        args.operator_reps = min(args.operator_reps, 20)

    set_paper_style()
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    L_values = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    if args.quick:
        L_values = np.array([8, 16, 32, 64, 128])

    ranks = [max(1, args.n // 16), max(2, args.n // 8), max(4, args.n // 4), max(8, args.n // 2), args.n]
    ranks = sorted(set(int(r) for r in ranks if 1 <= int(r) <= args.n))

    plot_design_gamma(args.n, outdir)
    haar_rows = run_haar_covariance(args.n, ranks, args.haar_samples, rng, outdir)
    scalar_rows = run_scalar_convergence(L_values, args.eps, args.samples, rng, outdir)
    run_rflr_memory([2, 4, 8, 16], max_age=24, outdir=outdir)
    operator_rows = run_operator_proxy(L_values, args.eps, args.m, args.operator_reps, rng, outdir)

    summary = {
        "n": args.n,
        "eps": args.eps,
        "m": args.m,
        "samples": args.samples,
        "haar_samples": args.haar_samples,
        "operator_reps": args.operator_reps,
        "constants": {
            "V": 5.0,
            "C_DF": 21.0 / 20.0,
            "A_AB": 173.0 / 720.0,
        },
        "outputs": sorted(str(p.name) for p in outdir.glob("*")),
        "haar_last": haar_rows[-1],
        "scalar_last": scalar_rows[-1],
        "operator_last": operator_rows[-1],
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
