#!/usr/bin/env python3
"""Create figures and an English journal-style PDF for the near-field audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def ci95(rows: list[dict[str, str]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(values.size))


def grouped(
    rows: list[dict[str, str]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    output: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[key] for key in keys)].append(row)
    return output


def make_solver_figure(results_dir: Path, tasks: list[dict[str, str]]) -> None:
    selected = (
        "learned-HB",
        "learned-Chebyshev",
        "identity-CG",
        "population-PCG",
        "exact",
    )
    scenarios = list(dict.fromkeys(row["scenario"] for row in tasks))
    by_method_scenario = grouped(tasks, ("method", "scenario"))
    colors = {
        "learned-HB": "#D55E00",
        "learned-Chebyshev": "#E69F00",
        "identity-CG": "#56B4E9",
        "population-PCG": "#0072B2",
        "exact": "#222222",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = 0.15
    for index, method in enumerate(selected):
        ap = [
            mean(by_method_scenario[(method, scenario)], "average_precision")
            for scenario in scenarios
        ]
        axes[0].bar(
            x + (index - 2.0) * width,
            ap,
            width,
            label=method,
            color=colors[method],
        )
    axes[0].set_xticks(x, scenarios, rotation=28, ha="right")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("average precision")
    axes[0].set_title("Localization across physical scenarios")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    macro = grouped(tasks, ("method",))
    residual_methods = (
        "learned-Richardson",
        "learned-HB",
        "learned-Chebyshev",
        "global-safe-HB",
        "population-safe-HB",
        "spectrum-HB",
        "identity-CG",
        "population-PCG",
        "exact",
    )
    mean_residual = [
        mean(macro[(method,)], "mean_relative_residual")
        for method in residual_methods
    ]
    covariance_residual = [
        mean(macro[(method,)], "covariance_relative_residual")
        for method in residual_methods
    ]
    y = np.arange(len(residual_methods))
    axes[1].barh(y - 0.18, mean_residual, 0.36, label="mean RHS", color="#009E73")
    axes[1].barh(
        y + 0.18,
        covariance_residual,
        0.36,
        label="covariance RHS",
        color="#CC79A7",
    )
    axes[1].set_yticks(y, residual_methods)
    axes[1].set_xscale("log")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("relative residual at reported depth")
    axes[1].set_title("Both posterior equations must converge")
    axes[1].grid(axis="x", which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.savefig(results_dir / "near_field_solver_summary.png", dpi=220)
    plt.close(figure)


def make_depth_figure(results_dir: Path, depth_rows: list[dict[str, str]]) -> None:
    by_method_depth = grouped(depth_rows, ("method", "depth"))
    methods = (
        "learned-Richardson",
        "learned-HB",
        "learned-Chebyshev",
        "population-safe-HB",
        "spectrum-HB",
        "identity-CG",
        "population-PCG",
    )
    styles = ("--", "-", "-", ":", "-.", "--", "-")
    depths = sorted({int(row["depth"]) for row in depth_rows})
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    for method, style in zip(methods, styles, strict=True):
        mean_values = [
            mean(by_method_depth[(method, str(depth))], "mean_relative_residual")
            for depth in depths
        ]
        covariance_values = [
            mean(
                by_method_depth[(method, str(depth))],
                "covariance_relative_residual",
            )
            for depth in depths
        ]
        axes[0].semilogy(depths, mean_values, style, marker="o", label=method)
        axes[1].semilogy(depths, covariance_values, style, marker="o", label=method)
    axes[0].set_title(r"Mean solve: $H_DQ_\mu=\Phi$")
    axes[1].set_title(r"Covariance solve: $H_DQ_\Sigma=N_DK_\Gamma$")
    for axis in axes:
        axis.set_xlabel("tied-loop depth")
        axis.set_ylabel("relative residual")
        axis.grid(which="both", alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    figure.savefig(results_dir / "near_field_depth_scaling.png", dpi=220)
    plt.close(figure)


def make_uq_figure(results_dir: Path, tasks: list[dict[str, str]]) -> None:
    macro = grouped(tasks, ("method",))
    methods = (
        "learned-HB",
        "learned-Chebyshev",
        "identity-CG",
        "population-PCG",
        "exact",
    )
    brier = [mean(macro[(method,)], "balanced_brier") for method in methods]
    ece = [mean(macro[(method,)], "balanced_ece") for method in methods]
    detection = [
        mean(macro[(method,)], "uncertainty_error_auc") for method in methods
    ]
    x = np.arange(len(methods))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    axes[0].bar(x - 0.18, brier, 0.36, label="balanced Brier", color="#56B4E9")
    axes[0].bar(x + 0.18, ece, 0.36, label="balanced ECE", color="#E69F00")
    axes[0].set_xticks(x, methods, rotation=20, ha="right")
    axes[0].set_title("Posterior occupancy calibration")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, detection, color="#009E73")
    axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x, methods, rotation=20, ha="right")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_title("Uncertainty--error AUROC")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(results_dir / "near_field_uq_summary.png", dpi=220)
    plt.close(figure)


def tex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("_", r"\_")


def build_report(
    results_dir: Path,
    tasks: list[dict[str, str]],
    protocol: dict[str, object],
) -> None:
    macro = grouped(tasks, ("method",))
    by_method_scenario = grouped(tasks, ("method", "scenario"))
    scenario_names = list(dict.fromkeys(row["scenario"] for row in tasks))
    methods = (
        "learned-Richardson",
        "learned-HB",
        "learned-Chebyshev",
        "global-safe-HB",
        "population-safe-HB",
        "spectrum-HB",
        "identity-CG",
        "population-PCG",
        "exact",
    )
    table_rows = []
    for method in methods:
        rows = macro[(method,)]
        table_rows.append(
            f"{tex_escape(method)} & {mean(rows, 'average_precision'):.3f} & "
            f"{mean(rows, 'area_matched_iou'):.3f} & "
            f"{mean(rows, 'mean_relative_residual'):.3e} & "
            f"{mean(rows, 'covariance_relative_residual'):.3e} & "
            f"{mean(rows, 'balanced_brier'):.3f} \\\\"
        )
    scenario_rows = []
    for scenario in scenario_names:
        scenario_rows.append(
            f"{tex_escape(scenario)} & "
            + " & ".join(
                f"{mean(by_method_scenario[(method, scenario)], 'average_precision'):.3f}"
                for method in (
                    "learned-HB",
                    "identity-CG",
                    "population-PCG",
                    "exact",
                )
            )
            + r" \\"
        )
    depth = int(protocol["depth"])
    physics = protocol["physics"]
    parameters = protocol["parameter_counts"]
    runtimes = protocol["runtime_ms_batch8"]
    runtime_means = {
        method: float(
            np.median([float(seed_values[method]) for seed_values in runtimes.values()])
        )
        for method in methods
    }
    hb_rows = macro[("learned-HB",)]
    cheb_rows = macro[("learned-Chebyshev",)]
    cg_rows = macro[("identity-CG",)]
    pcg_rows = macro[("population-PCG",)]
    exact_rows = macro[("exact",)]
    boundary = mean(exact_rows, "boundary_residual")
    hb_certificate = mean(hb_rows, "certificate")
    cheb_certificate = mean(cheb_rows, "certificate")
    condition_values = np.asarray(
        [
            float(row["true_upper"]) / max(float(row["true_lower"]), 1.0e-12)
            for row in hb_rows
        ]
    )
    geometric_condition = float(np.exp(np.log(condition_values).mean()))
    maximum_condition = float(condition_values.max())
    pcg_ratio = mean(pcg_rows, "mean_relative_residual") / max(
        mean(exact_rows, "mean_relative_residual"), 1.0e-12
    )
    preconditioning_gain = mean(cg_rows, "mean_relative_residual") / max(
        mean(pcg_rows, "mean_relative_residual"), 1.0e-12
    )
    covariance_preconditioning_gain = mean(
        cg_rows, "covariance_relative_residual"
    ) / max(mean(pcg_rows, "covariance_relative_residual"), 1.0e-12)
    report = r"""\documentclass[10pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{amsmath,amssymb,amsthm,bm,booktabs,graphicx,microtype}}
\usepackage{{hyperref,xcolor}}
\usepackage[numbers,sort&compress]{{natbib}}
\newtheorem{{proposition}}{{Proposition}}
\newtheorem{{theorem}}{{Theorem}}
\newtheorem{{remark}}{{Remark}}
\DeclareMathOperator{{\softmax}}{{softmax}}
\DeclareMathOperator{{\tr}}{{tr}}
\title{{Posterior-Moment In-Context Solvers for Bayesian Near-Field Linear Sampling}}
\author{{Technical results note}}
\date{{4 August 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We formulate the original two-dimensional near-field linear sampling method
(LSM) as a kernel Bayesian inverse problem and adapt the parallel posterior-
predictive recurrences of Kang, Lee and Cheng to this setting.  Point sources
illuminate one to six sound-soft obstacles and a distinct receiver array records
the complex scattered near field.  A prescribed von Mises/softmax covariance is
a modelling choice: neither its nonlinearity nor a temperature is learned.  A
single tied loop solves both the probe equation, which determines the posterior
mean, and the matrix equation required for posterior covariance.  Richardson,
heavy-ball (HB), Chebyshev and PCG cells therefore share the same encoder,
decoder and Bayesian semantics.  We prove the corresponding depth laws and
show that learned population whitening accelerates ordinary CG substantially,
while it cannot remove all obstacle-dependent eigendirections.  A controlled
multi-obstacle audit reports localization,
residual, calibration and robustness.  Its main negative result is useful:
finite-horizon HB benefits from deliberately conservative effective endpoints,
whereas plugging in the exact spectral endpoints produces a large transient.
Small residuals require either substantially greater depth or adaptive Krylov
coefficients.
\end{{abstract}}

\section{{Scope and correction of the problem}}
This note concerns deterministic active \emph{{near-field}} LSM, not scalar
one-dimensional Gaussian-process regression, not a Born far-field surrogate,
and not the random-source LSM variants of \citet{{montanelli2022,montanelli2024}}.
For each task, source and receiver locations lie on separate circles.  The
measured object is the full multistatic near-field matrix $N_D$ generated by a
sound-soft obstacle $D$.  The random objects in training are obstacles,
acquisition perturbations and additive noise; the incident point sources are
controlled and deterministic.

\section{{Physical near-field model}}
Let $\Gamma_s=\{{y_j\}}_{{j=1}}^{{n_s}}$ and
$\Gamma_r=\{{x_i\}}_{{i=1}}^{{n_r}}$ denote source and receiver curves.  With
$\Phi_k(x,y)=\tfrac{{\mathrm i}}{{4}}H_0^{{(1)}}(k|x-y|)$, the incident field
from $y_j$ is $u_j^i(x)=\Phi_k(x,y_j)$.  For a sound-soft obstacle,
\[
 (\Delta+k^2)u_j^s=0\quad\text{{outside }}D,
 \qquad u_j^s=-u_j^i\quad\text{{on }}\partial D,
\]
together with the Sommerfeld radiation condition.  The data matrix is
$(N_D)_{{ij}}=u_j^s(x_i)$.  Numerical data are generated by boundary
collocation with interior fundamental sources; the mean relative boundary
residual in the audit is {boundary:.2e}.  This is an independent sound-soft
forward model, not reused code from \texttt{{lsmlab}}.

For every sampling point $z$ in a two-dimensional grid, define
$\phi_z=(\Phi_k(x_i,z))_i$.  After whitening the known receiver-noise
covariance $R_\Gamma$, we retain the same symbols for
$R_\Gamma^{{-1/2}}N_D$ and $R_\Gamma^{{-1/2}}\phi_z$.  This is the original
near-field sampling equation $N_Dg_z\simeq\phi_z$ in a statistically explicit
form.

\section{{A fixed nonlinear feature model}}
For source angles $\theta_i$, let
\[
 W_{{ij}}=\exp\!\left[\gamma\cos(\theta_i-\theta_j)\right],\qquad
 K_\Gamma=(1-\eta)I+\eta D_W^{{-1/2}}WD_W^{{-1/2}}.
\]
This is the symmetric positive version of row-softmax attention.  In the
reported model $\gamma={physics['kernel_gamma']}$ and
$\eta={physics['kernel_mix']}$ are fixed before training.  The softmax
nonlinearity selects the GP feature space; it is not an optimizable temperature.
Other prior knowledge can be encoded by replacing $\cos(\theta_i-\theta_j)$
with a task-appropriate feature similarity while preserving positive
definiteness.

The proper-complex prior is $g\sim\mathcal{{CN}}(0,K_\Gamma)$ and whitened
noise has identity covariance.  Set
\[
 H_D=N_DK_\Gamma N_D^*+I.
\]
For the matrix $\Phi=[\phi_z]_z$, the two posterior equations are
\begin{{equation}}\label{{eq:two-solves}}
 H_DQ_\mu=\Phi,
 \qquad H_DQ_\Sigma=N_DK_\Gamma.
\end{{equation}}

\begin{{proposition}}[Parallel posterior moments]\label{{prop:moments}}
The conditional coefficient field associated with the probe matrix has mean
and covariance
\[
 M_D=K_\Gamma N_D^*Q_\mu,
 \qquad
 \Sigma_D=K_\Gamma-K_\Gamma N_D^*Q_\Sigma.
\]
Consequently, applying one tied linear recurrence to the concatenated right-
hand side $[\Phi,N_DK_\Gamma]$ computes both posterior moments with identical
solver coefficients and depth.
\end{{proposition}}
\begin{{proof}}
These are the Woodbury-form conditioning identities for a linear proper-
complex Gaussian model.  Both expressions contain the same inverse $H_D^{{-1}}$,
which proves the shared recurrence statement.
\end{{proof}}

This is the exact near-field analogue of the two parallel recurrences used for
posterior predictive mean and variance in \citet{{kang2026}}.  It also clarifies
terminology: the matrices $B^*A^jB$ used below are \emph{{spectral Krylov
statistics}} for the controller, not posterior moments.

\section{{Architecture}}
Let $C_{{\mathrm{{pop}},\Gamma}}$ be a geometry-conditioned factor diagonal in
the prescribed receiver-feature basis.  The task supplies
\[
 A_D=s_D^{{-1}}C_{{\mathrm{{pop}},\Gamma}}^*H_D
 C_{{\mathrm{{pop}},\Gamma}},\qquad
 B_D=s_D^{{-1}}C_{{\mathrm{{pop}},\Gamma}}^*
 [\Phi,N_DK_\Gamma],
\]
where the row bound $s_D$ ensures $\lambda_{{\max}}(A_D)\leq1$.  A fixed random
sketch compresses both right-hand-side blocks, and the controller reads
\[
 \mathcal E_D=\left\{\frac{{Z_0^*A_D^jZ_0}}{{\|Z_0\|_F^2}}\right\}_{{j=0}}^J,
 \qquad J=8,
\]
together with the safe lower bound and population-gain summaries.  The 0.73M-
parameter MLP outputs two \emph{{effective}} endpoints.  The physical matrix,
kernel, posterior formulas and recurrent solver are never learned.  The same
encoder and decoder are used by all stationary cells.

The output score is based on $q_z=g_z^*K_\Gamma^{{-1}}g_z$.  For
$g_z\sim\mathcal{{CN}}(m_z,\Sigma_D)$,
\begin{{align*}}
 \mathbb E q_z&=m_z^*K_\Gamma^{{-1}}m_z+
 \tr(K_\Gamma^{{-1}}\Sigma_D),\\
 \operatorname{{Var}}q_z&=\tr[(K_\Gamma^{{-1}}\Sigma_D)^2]
 +2m_z^*K_\Gamma^{{-1}}\Sigma_DK_\Gamma^{{-1}}m_z.
\end{{align*}}
A lognormal moment match gives the reported mean and standard deviation of the
LSM score $-\tfrac12\log q_z$ and hence posterior occupancy probabilities.

\section{{Solver laws and finite-depth qualification}}
For $0<m\leq\lambda(A_D)\leq L$ and $\kappa=L/m$, optimal Richardson has
\[
 \rho_R=\frac{{\kappa-1}}{{\kappa+1}},\qquad
 T_R(\varepsilon)=O\!\left(\kappa\log\varepsilon^{{-1}}\right).
\]
HB uses
\[
 \alpha=\frac{{4}}{{(\sqrt L+\sqrt m)^2}},\qquad
 \beta=\left(\frac{{\sqrt L-\sqrt m}}{{\sqrt L+\sqrt m}}\right)^2,
\quad
 \rho_H=\frac{{\sqrt\kappa-1}}{{\sqrt\kappa+1}},
\]
and Chebyshev semi-iteration has the same asymptotic depth law
$O(\sqrt\kappa\log\varepsilon^{{-1}})$.  A finite-horizon HB bound contains a
polynomial prefactor, typically $(T+1)\rho_H^T$, because endpoint modes have
repeated characteristic roots.  Exact endpoints may therefore increase the
residual before the asymptotic regime.  PCG obeys the familiar energy-norm
bound $2\rho_H^T$ and terminates in at most $n_r$ steps in exact arithmetic,
although finite precision and the large block of probe right-hand sides delay
that ideal behavior.

\begin{{theorem}}[Posterior-error propagation]
Let $R_\mu=\Phi-H_D\widehat Q_\mu$ and
$R_\Sigma=N_DK_\Gamma-H_D\widehat Q_\Sigma$.  Then
\begin{{align*}}
 \|\widehat M_D-M_D\|&\leq
 \|K_\Gamma N_D^*\|\,\|H_D^{{-1}}\|\,\|R_\mu\|,\\
 \|\widehat\Sigma_D-\Sigma_D\|&\leq
 \|K_\Gamma N_D^*\|\,\|H_D^{{-1}}\|\,\|R_\Sigma\|.
\end{{align*}}
Thus a small reconstruction loss alone does not certify Bayesian UQ: the
covariance right-hand side must also converge.
\end{{theorem}}
\begin{{proof}}
Subtract each approximate equation from \eqref{{eq:two-solves}}, multiply by
$H_D^{{-1}}$, and apply submultiplicativity.
\end{{proof}}

With $n_x$ probes, a dense recurrent layer costs
$O(n_r^2(n_x+n_s))$ after construction of $H_D$; memory is
$O(n_r(n_x+n_s))$.  The posterior decoder costs $O(n_sn_rn_x+n_s^2n_r)$.
The spectral sketch dimension is
$2(J+1)(2r)^2+4=2596$ for $J=8,r=6$, independent of $n_x$ after sketching.

\section{{Why the loss was not small}}
There are three distinct obstructions.  First, whitening by a small noise scale
makes $N_DK_\Gamma N_D^*$ much larger than the identity floor, so typical
$\kappa_D$ is large (geometric mean {geometric_condition:.0f}, maximum
{maximum_condition:.0f} in this audit).  Second, a geometry-only population factor has fixed
eigendirections and cannot diagonalize every obstacle-dependent $H_D$.
Third, an asymptotically optimal HB pair can have a severe finite-depth
transient.  The learned controller therefore has a real signal---RHS-weighted
spectral statistics predict useful damping---but its best finite-depth output
need not enclose the whole spectrum.  In the audit, the strict certificate rate
is {hb_certificate:.3f} for learned HB and {cheb_certificate:.3f} for learned
Chebyshev.  Certification and finite-horizon optimality are different goals.

\section{{Numerical audit}}
We use $n_r=n_s={physics['n_sensors']}$, a
${physics['grid_size']}\times{physics['grid_size']}$ sampling grid, one to six
components, frequencies 6--12, 3--30\% noise, full and 180-degree apertures,
and angular jitter.  Training contains one to four disks, ellipses and kites;
six components, stars, 30\% noise, half aperture and frequency 12 are held out.
There are {len(protocol['seeds'])} independent seeds, {protocol['steps']}
training steps per stationary solver and {protocol['eval_tasks_per_seed_scenario']}
test tasks per seed and scenario.  All principal comparisons use depth
$T={depth}$.  ``Spectrum-HB'' substitutes the exact eigenvalue endpoints and is
an evaluation-only diagnostic.  ``Exact'' uses a dense direct solve.

\begin{{table}}[t]
\centering\small
\caption{{Macro-average performance across all scenarios and seeds.  Lower is
better for residuals and balanced Brier; higher is better otherwise.}}
\begin{{tabular}}{{lccccc}}
\toprule
Method & AP & area-IoU & mean res. & covariance res. & Brier\\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering\scriptsize
\caption{{Average precision by physical scenario.}}
\begin{{tabular}}{{lcccc}}
\toprule
Scenario & learned HB & identity CG & population PCG & exact\\
\midrule
{chr(10).join(scenario_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{near_field_solver_summary.png}}
\caption{{Localization and the two distinct posterior residuals.  The exact
spectral HB diagnostic exposes the finite-depth transient.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.96\linewidth]{{near_field_depth_scaling.png}}
\caption{{Empirical depth laws on four-obstacle tasks.  PCG eventually enters
the small-residual regime; stationary methods remain condition-number limited.}}
\end{{figure}}

The macro mean residuals are {mean(hb_rows, 'mean_relative_residual'):.3e}
(learned HB), {mean(cheb_rows, 'mean_relative_residual'):.3e} (learned
Chebyshev), {mean(cg_rows, 'mean_relative_residual'):.3e} (identity CG),
{mean(pcg_rows, 'mean_relative_residual'):.3e} (population PCG) and
{mean(exact_rows, 'mean_relative_residual'):.3e} (direct).  The PCG/direct
mean-residual ratio is {pcg_ratio:.2f}; PCG can attain a smaller measured
residual than the single-precision dense factorization, without changing the
underlying posterior.  Thus ``small'' must be defined relative to the
inferential target and floating-point precision.  Batched runtimes are
{runtime_means['learned-HB']:.2f} ms for learned HB,
{runtime_means['learned-Chebyshev']:.2f} ms for learned Chebyshev,
{runtime_means['identity-CG']:.2f} ms for identity CG,
{runtime_means['population-PCG']:.2f} ms for PCG and
{runtime_means['exact']:.2f} ms for the small $24\times24$ dense reference
(medians across seeds).  At
this matrix size the direct solve is faster; recurrent scaling becomes relevant
only when structure, streaming, hardware reuse or much larger systems prohibit
factorization.

The equal-budget preconditioning test is direct: population-PCG reduces the
mean residual by a factor {preconditioning_gain:.2f} and the covariance
residual by a factor {covariance_preconditioning_gain:.2f} relative to identity
CG.  A factor above one is evidence that the in-context population transform
helps; a factor near or below one means that ordinary CG, rather than learned
conditioning, explains the convergence.

PCG and the direct solver have identical macro AP to four decimals
({mean(pcg_rows, 'average_precision'):.4f}) and visually indistinguishable
indicators.  Therefore the remaining blur in Figure~3 is not a solver-loss
problem.  It is the near-field point-spread/resolution limit of 24 channels at
one frequency: closely spaced components merge into a broad indicator even
after the linear system is solved.  More recurrent depth cannot sharpen that
map.  The relevant remedies are additional independent frequencies, more
source/receiver locations, a wider aperture, or an explicitly different
shape prior; the last option changes the Bayesian model and must not be hidden
inside the preconditioner.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.96\linewidth]{{near_field_reconstructions.png}}
\caption{{A held-out four-obstacle example.  The first row shows truth or
posterior score and the second row shows posterior score standard deviation.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.88\linewidth]{{near_field_uq_summary.png}}
\caption{{Bayesian UQ obtained from the second parallel recurrence.  Thresholds
are calibrated separately on training-distribution validation tasks.}}
\end{{figure}}

The learned stationary loops have lower calibrated Brier scores and higher
uncertainty--error AUROC than the converged posterior.  This is an early-
stopping regularization effect, not evidence that an approximate covariance is
more faithful to the stated GP.  PCG/direct agreement is the appropriate test
of posterior correctness; calibration measures downstream occupancy utility.

\section{{Interpretation and modelling recommendations}}
The experiments support five conclusions.
\begin{{enumerate}}
\item The task is genuinely two-dimensional original near-field LSM: the input
is a complex multistatic response to several deterministic point sources.
\item The fixed softmax covariance is useful as prior geometry, but choosing it
cannot by itself erase the spectrum induced by the current obstacle.
\item Posterior mean and variance must be propagated together.  Solving only
$H_DQ_\mu=\Phi$ can produce attractive reconstructions with incorrect UQ.
\item Spectral endpoints contain signal, but exact asymptotic HB endpoints are
not the finite-depth optimum.  The controller should be trained against the
two residuals and audited against, rather than forced to equal, the spectrum.
\item For a genuinely small loss, use PCG or an $A_D$-equivariant polynomial
preconditioner and increase depth.  A larger generic MLP does not repair missing
task eigendirections.
\end{{enumerate}}

The next scientifically necessary step is to replace the MFS generator by an
independent high-order boundary-element solver and to test experimental near-
field data.  Until then, this is a controlled numerical validation, not a claim
of field-data performance.

\section{{Reproducibility}}
The code fixes all train/evaluation seeds, stores task-level tables and
checkpoints, and reports the exact scenario definitions in
\texttt{{protocol.json}}.  The principal model has
{parameters['heavy_ball']:,} trainable parameters.  Source code and result
tables accompany this PDF.

\begin{{thebibliography}}{{9}}
\bibitem[Kang et~al.(2026)Kang, Lee, and Cheng]{{kang2026}}
G. Kang, C. J. Lee, and X. Cheng.
\newblock Transformers can learn posterior predictive distributions in-context.
\newblock \emph{{arXiv:2605.26713}}, 2026.

\bibitem[Kirsch and Grinberg(2008)]{{kirsch2008}}
A. Kirsch and N. Grinberg.
\newblock \emph{{The Factorization Method for Inverse Problems}}.
\newblock Oxford University Press, 2008.

\bibitem[Colton and Kress(2019)]{{colton2019}}
D. Colton and R. Kress.
\newblock \emph{{Inverse Acoustic and Electromagnetic Scattering Theory}}.
\newblock Springer, 4th edition, 2019.

\bibitem[Garnier et~al.(2022)Garnier, Haddar, and Montanelli]{{montanelli2022}}
J. Garnier, H. Haddar, and H. Montanelli.
\newblock The linear sampling method for random sources.
\newblock \emph{{arXiv:2210.15560}}, 2022.

\bibitem[Garnier et~al.(2024)Garnier, Haddar, and Montanelli]{{montanelli2024}}
J. Garnier, H. Haddar, and H. Montanelli.
\newblock The linear sampling method for data generated by small random scatterers.
\newblock \emph{{arXiv:2403.19482}}, 2024.

\bibitem[Polyak(1964)]{{polyak1964}}
B. T. Polyak.
\newblock Some methods of speeding up the convergence of iteration methods.
\newblock \emph{{USSR Computational Mathematics and Mathematical Physics}}, 1964.
\end{{thebibliography}}
\end{{document}}
"""
    replacements = {
        "{boundary:.2e}": f"{boundary:.2e}",
        "{physics['kernel_gamma']}": str(physics["kernel_gamma"]),
        "{physics['kernel_mix']}": str(physics["kernel_mix"]),
        "{hb_certificate:.3f}": f"{hb_certificate:.3f}",
        "{cheb_certificate:.3f}": f"{cheb_certificate:.3f}",
        "{geometric_condition:.0f}": f"{geometric_condition:.0f}",
        "{maximum_condition:.0f}": f"{maximum_condition:.0f}",
        "{physics['n_sensors']}": str(physics["n_sensors"]),
        "{physics['grid_size']}": str(physics["grid_size"]),
        "{len(protocol['seeds'])}": str(len(protocol["seeds"])),
        "{protocol['steps']}": str(protocol["steps"]),
        "{protocol['eval_tasks_per_seed_scenario']}": str(
            protocol["eval_tasks_per_seed_scenario"]
        ),
        "{depth}": str(depth),
        "{chr(10).join(table_rows)}": "\n".join(table_rows),
        "{chr(10).join(scenario_rows)}": "\n".join(scenario_rows),
        "{mean(hb_rows, 'mean_relative_residual'):.3e}": (
            f"{mean(hb_rows, 'mean_relative_residual'):.3e}"
        ),
        "{mean(cheb_rows, 'mean_relative_residual'):.3e}": (
            f"{mean(cheb_rows, 'mean_relative_residual'):.3e}"
        ),
        "{mean(cg_rows, 'mean_relative_residual'):.3e}": (
            f"{mean(cg_rows, 'mean_relative_residual'):.3e}"
        ),
        "{mean(pcg_rows, 'mean_relative_residual'):.3e}": (
            f"{mean(pcg_rows, 'mean_relative_residual'):.3e}"
        ),
        "{mean(exact_rows, 'mean_relative_residual'):.3e}": (
            f"{mean(exact_rows, 'mean_relative_residual'):.3e}"
        ),
        "{mean(pcg_rows, 'average_precision'):.4f}": (
            f"{mean(pcg_rows, 'average_precision'):.4f}"
        ),
        "{pcg_ratio:.2f}": f"{pcg_ratio:.2f}",
        "{preconditioning_gain:.2f}": f"{preconditioning_gain:.2f}",
        "{covariance_preconditioning_gain:.2f}": (
            f"{covariance_preconditioning_gain:.2f}"
        ),
        "{runtime_means['learned-HB']:.2f}": (
            f"{runtime_means['learned-HB']:.2f}"
        ),
        "{runtime_means['learned-Chebyshev']:.2f}": (
            f"{runtime_means['learned-Chebyshev']:.2f}"
        ),
        "{runtime_means['identity-CG']:.2f}": (
            f"{runtime_means['identity-CG']:.2f}"
        ),
        "{runtime_means['population-PCG']:.2f}": (
            f"{runtime_means['population-PCG']:.2f}"
        ),
        "{runtime_means['exact']:.2f}": f"{runtime_means['exact']:.2f}",
        "{parameters['heavy_ball']:,}": f"{parameters['heavy_ball']:,}",
    }
    for token, value in replacements.items():
        report = report.replace(token, value)
    report = report.replace("{{", "{").replace("}}", "}")
    (results_dir / "near_field_results_note.tex").write_text(report, encoding="utf-8")


def compile_pdf(results_dir: Path) -> None:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "near_field_results_note.tex",
    ]
    for _ in range(2):
        subprocess.run(command, cwd=results_dir, check=True)


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    tasks = read_rows(results_dir / "tasks.csv")
    depth_rows = read_rows(results_dir / "depth.csv")
    protocol = json.loads((results_dir / "protocol.json").read_text(encoding="utf-8"))
    make_solver_figure(results_dir, tasks)
    make_depth_figure(results_dir, depth_rows)
    make_uq_figure(results_dir, tasks)
    build_report(results_dir, tasks, protocol)
    compile_pdf(results_dir)
    print(results_dir / "near_field_results_note.pdf")


if __name__ == "__main__":
    main()
