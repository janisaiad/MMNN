"""Generate the compact, reproducible figures used by the merged report.

The script only reads the machine-readable outputs already produced by the
audits.  It does not rerun the expensive trajectory harvests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "spectral_self_attention"
OUT = ROOT / "refs" / "spectral_self_attention" / "global_report" / "figures_en"

BLUE = "#1565c0"
ORANGE = "#d84315"
GREEN = "#2e7d32"
PURPLE = "#6a1b9a"
GRAY = "#607d8b"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def taxonomy_and_survival() -> None:
    taxonomy = read_json(DATA / "large_taxonomy_summary.json")
    small = read_json(DATA / "small_step_final_results.json")
    types = np.arange(1, 5)
    p3 = [taxonomy["families"][str(i)]["totals"]["period3"] for i in types]
    p4 = [taxonomy["families"][str(i)]["totals"]["period4"] for i in types]
    ode = {int(row["family"]): row for row in small["direct_ode_by_family"]}
    moving = [100 * ode[i]["moving_fraction"] for i in types]
    rigid = [ode[i]["rigid_rotation"] for i in types]
    internal = [ode[i]["internal_shape_motion"] for i in types]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    width = 0.34
    axes[0].bar(types - width / 2, p3, width, color=BLUE, label="period 3")
    axes[0].bar(types + width / 2, p4, width, color=ORANGE, label="period 4")
    axes[0].set_yscale("log")
    axes[0].set_xticks(types, ["type 1", "type 2", "type 3", "type 4"])
    axes[0].set_ylabel("classified trajectories (log scale)")
    axes[0].set_title("Few-layer cycles")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)

    rigid_fraction = [100 * rigid[i - 1] / ode[i]["records"] for i in types]
    internal_fraction = [100 * internal[i - 1] / ode[i]["records"] for i in types]
    axes[1].bar(types, rigid_fraction, color=GREEN, label="rigid rotation")
    axes[1].bar(
        types,
        internal_fraction,
        bottom=rigid_fraction,
        color=PURPLE,
        label="changing internal shape",
    )
    for x, value in zip(types, moving):
        axes[1].text(x, value + 0.7, f"{value:.1f}%", ha="center", fontsize=9)
    axes[1].set_ylim(0, 27)
    axes[1].set_xticks(types, ["type 1", "type 2", "type 3", "type 4"])
    axes[1].set_ylabel("fraction still moving in the flow (%)")
    axes[1].set_title("What survives weak layers")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Finite-layer taxonomy and its continuous limit answer different questions")
    fig.tight_layout()
    finish(fig, "taxonomy_and_survival.png")


def spectra() -> None:
    small = read_json(DATA / "small_step_final_results.json")
    spectra = small["full_lyapunov_spectra"]
    selected = [
        ("cycle type 2\n(3 tokens)", spectra["type2_p3_stable_cycle"]),
        ("cycle type 4\n(3 tokens)", spectra["type4_stable_cycle"]),
        ("chaos type 2\n(3 tokens)", spectra["type2_weak_chaos"]),
        ("chaos type 4\n(3 tokens)", spectra["type4_strong_chaos"]),
        ("hyperchaos type 4\n(8 tokens)", spectra["type4_eight_token_hyperchaos"]),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for y, (label, values) in enumerate(selected):
        values = np.asarray(values)
        colors = np.where(values > 0.01, ORANGE, np.where(values < -0.01, BLUE, GRAY))
        ax.scatter(values, np.full_like(values, y, dtype=float), c=colors, s=52, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(selected)), [item[0] for item in selected])
    ax.invert_yaxis()
    ax.set_xlabel("mean rate: positive = separation, negative = contraction")
    ax.set_title("Complete signatures of certified continuous attractors")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    finish(fig, "continuous_spectra.png")


def multihead_and_dimension() -> None:
    basins = read_json(DATA / "multihead" / "frozen_probe_basins.json")["circle"]
    rows = read_csv(DATA / "multihead" / "high_dimension_decoupling.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))

    scenarios = ["A", "B", "A+B"]
    strips = []
    for scenario in scenarios:
        labels = np.asarray(basins["scenarios"][scenario]["labels"], dtype=int)
        # Relabel by first appearance so colors mean destinations, not JSON ids.
        mapping: dict[int, int] = {}
        next_label = 0
        clean = np.empty_like(labels)
        for index, label in enumerate(labels):
            if int(label) not in mapping:
                mapping[int(label)] = next_label
                next_label += 1
            clean[index] = mapping[int(label)]
        strips.append(clean)
    image = np.vstack(strips)
    cmap = ListedColormap(
        ["#1565c0", "#ef6c00", "#2e7d32", "#8e24aa", "#00838f", "#c62828", "#6d4c41"]
    )
    axes[0].imshow(image, aspect="auto", interpolation="nearest", cmap=cmap)
    axes[0].set_yticks(range(3), ["head A", "head B", "A + B"])
    axes[0].set_xticks([0, 360, 720, 1080, 1439], ["0", "π/2", "π", "3π/2", "2π"])
    axes[0].set_xlabel("initial probe angle")
    axes[0].set_title("Basins: summing heads redraws the map")

    dimensions = np.array([int(row["dimension"]) for row in rows])
    token_cos = np.array([float(row["token_abs_cosine"]) for row in rows])
    head_cos = np.array([float(row["head_abs_cosine"]) for row in rows])
    neff = np.array([float(row["random_head_neff"]) for row in rows])
    axes[1].plot(dimensions, token_cos, "o-", color=BLUE, label="token similarity")
    axes[1].plot(dimensions, head_cos, "s-", color=ORANGE, label="head-force similarity")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("dimension")
    axes[1].set_ylabel("mean absolute inner product")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Random directions become orthogonal")
    axes[2].plot(dimensions, neff, "^-", color=GREEN)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("dimension")
    axes[2].set_ylabel("effective number attended (out of 24)")
    axes[2].set_ylim(0, 25)
    axes[2].grid(alpha=0.2)
    axes[2].set_title("Attention approaches uniform averaging")
    fig.tight_layout()
    finish(fig, "multihead_and_dimension.png")


def meanfield_and_kernels() -> None:
    continuum = read_csv(DATA / "mean_field_extensions" / "continuum_convergence.csv")
    roots = read_csv(DATA / "mean_field_extensions" / "kernel_polygon_roots.csv")
    n = np.array([int(row["n_tokens"]) for row in continuum])
    error = np.array([float(row["velocity_rmse"]) for row in continuum])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].loglog(n, error, "o-", color=BLUE, label="measured error")
    reference = error[0] * (n / n[0]) ** -0.5
    axes[0].loglog(n, reference, "--", color=GRAY, label=r"$n^{-1/2}$ reference")
    axes[0].set_xlabel("number of tokens")
    axes[0].set_ylabel("velocity error")
    axes[0].set_title("The sum becomes an integral")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False)

    normalized = [row for row in roots if row["row_normalized"] == "True"]
    kernel_order = ["exponential", "sigmoid", "softplus", "polynomial4"]
    kernel_names = ["exp", "sigmoid", "softplus", "polynomial"]
    for y, kernel in enumerate(kernel_order):
        subset = [row for row in normalized if row["kernel"] == kernel]
        for row in subset:
            q = float(row["q"])
            stable = row["stable"] == "True"
            axes[1].scatter(
                q,
                y,
                marker="o" if stable else "x",
                s=70,
                color=GREEN if stable else ORANGE,
                linewidth=2,
            )
    axes[1].axvline(0, color=GRAY, lw=0.8)
    axes[1].set_yticks(range(4), kernel_names)
    axes[1].set_xlabel("polygon geometry parameter q")
    axes[1].set_title("Changing the kernel changes polygonal branches")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].scatter([], [], marker="o", color=GREEN, label="stable")
    axes[1].scatter([], [], marker="x", color=ORANGE, label="unstable")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    finish(fig, "meanfield_and_kernels.png")


def training_and_muon() -> None:
    trained = read_csv(DATA / "mean_field_extensions" / "trained_polygon_turnpike.csv")
    muon = read_json(DATA / "one_step_muon" / "summary.json")
    time = np.array([float(row["time"]) for row in trained])
    control = np.array([float(row["control_norm"]) for row in trained])
    distance = np.array([float(row["distance_from_target"]) for row in trained])
    budgets = [0.1, 0.3, 1.0]
    names = ["gradient_descent", "exact_muon", "newton_schulz_5"]
    labels = ["gradient", "exact Muon", "Muon, 5 iterations"]
    colors = [BLUE, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    axes[0].semilogy(time, control, color=PURPLE, label="control strength")
    axes[0].set_xlabel("continuous depth")
    axes[0].set_ylabel("trained MLP strength", color=PURPLE)
    axes[0].tick_params(axis="y", colors=PURPLE)
    axes[0].set_title("The control is concentrated near the output")
    axes[0].grid(alpha=0.18)

    axes[1].plot(time, distance, color=GRAY)
    axes[1].set_xlabel("continuous depth")
    axes[1].set_ylabel("distance to target")
    axes[1].set_title("Terminal control reaches the target")
    axes[1].grid(alpha=0.18)

    for name, label, color in zip(names, labels, colors):
        losses = [muon["outcomes"][f"{budget}:{name}"]["terminal_loss"] for budget in budgets]
        axes[2].plot(budgets, losses, "o-", color=color, label=label)
    axes[2].axhline(muon["untrained_terminal_loss"], ls="--", color=GRAY, label="untrained")
    axes[2].set_xlabel("total budget of the first update")
    axes[2].set_ylabel("terminal loss")
    axes[2].set_title("Update geometry changes one-step loss")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    finish(fig, "training_and_muon.png")


def slow_ou() -> None:
    data = read_json(DATA / "slow_ou_tokens.json")
    rates = np.array([float(row["rate"]) for row in data["rates"] if float(row["rate"]) > 0])
    runs = [row for row in data["rates"] if float(row["rate"]) > 0]
    concentration = np.array([row["late_concentration"] for row in runs])
    tracking = np.array([row["late_tracking"] for row in runs])
    target = np.array([row["late_target_alignment"] for row in runs])
    speed = np.array([row["late_speed"] for row in runs])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    axes[0].semilogx(rates, concentration, "o-", color=BLUE, label="cloud concentration")
    axes[0].semilogx(rates, tracking, "s-", color=ORANGE, label="instantaneous-axis tracking")
    axes[0].semilogx(rates, target, "^-", color=GREEN, label="mean-axis alignment")
    axes[0].set_xlabel("OU weight renewal rate")
    axes[0].set_ylabel("late-time mean score")
    axes[0].set_ylim(0.58, 1.02)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Fast renewal favors the mean axis")
    axes[1].semilogx(rates, speed, "d-", color=PURPLE)
    axes[1].set_xlabel("OU weight renewal rate")
    axes[1].set_ylabel("late-time residual speed")
    axes[1].grid(alpha=0.2)
    axes[1].set_title("Residual motion increases with renewal rate")
    fig.tight_layout()
    finish(fig, "slow_ou.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    taxonomy_and_survival()
    spectra()
    multihead_and_dimension()
    meanfield_and_kernels()
    training_and_muon()
    slow_ou()


if __name__ == "__main__":
    main()
