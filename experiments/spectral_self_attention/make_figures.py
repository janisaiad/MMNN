"""Create compact figures and summaries from the completed experiment atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .mixed_equilibria import (
    embed_mixed_balanced_state,
    mixed_balanced_roots,
    mixed_balanced_state,
)
from .simulator import state_tangent_jacobian, vector_field


def outcome_heatmaps(random_trials: pd.DataFrame, output_dir: Path) -> None:
    labels = (
        "consensus",
        "bipolar",
        "mixed_extreme_stationary",
        "subspace_stationary",
    )
    titles = ("Consensus", "Bipolar", "Mixed extreme", "Extremal eigenspace")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    for label, title, axis in zip(labels, titles, axes.flat, strict=True):
        fractions = (
            random_trials.assign(match=random_trials.geometry.eq(label))
            .groupby(["case", "beta"])
            .match.mean()
            .unstack(fill_value=0)
        )
        sns.heatmap(
            fractions,
            ax=axis,
            cmap="mako",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "fraction over n and trials"},
        )
        axis.set_title(title)
        axis.set_xlabel(r"$\beta$")
        axis.set_ylabel("")
    fig.savefig(output_dir / "outcome_heatmaps.png", dpi=180)
    plt.close(fig)


def indefinite_selection(random_trials: pd.DataFrame, output_dir: Path) -> None:
    indefinite = random_trials[random_trials.family == "indefinite"].copy()
    eigenvalue_min = indefinite.groupby("case").selected_eigenvalue.transform("min")
    indefinite["bottom_selected"] = (
        indefinite.selected_eigenvalue == eigenvalue_min
    )
    summary = (
        indefinite.groupby(["case", "beta"])
        .agg(
            bottom_selected=("bottom_selected", "mean"),
            mixed=("geometry", lambda values: (values == "mixed_extreme_stationary").mean()),
            consensus=("geometry", lambda values: (values == "consensus").mean()),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sns.lineplot(
        data=summary,
        x="beta",
        y="bottom_selected",
        hue="case",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Selection of the most negative eigenspace")
    axes[0].set_ylabel("fraction")
    axes[0].legend_.remove()
    sns.lineplot(
        data=summary,
        x="beta",
        y="mixed",
        hue="case",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Mixed-extreme attractors")
    axes[1].set_ylabel("fraction")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.savefig(output_dir / "indefinite_selection.png", dpi=180)
    plt.close(fig)
    summary.to_csv(output_dir / "indefinite_selection_summary.csv", index=False)


def long_time_plot(long_time: pd.DataFrame, output_dir: Path) -> None:
    selected = long_time[
        (
            (long_time["case"] == "nd_simple")
            & (long_time.beta == 0.03)
            & long_time.n_tokens.isin([3, 5])
        )
        | (
            (long_time["case"] == "mixed_two_positive")
            & (long_time.beta == 8.0)
        )
        | (
            (long_time["case"] == "pd_flat_top")
            & (long_time.beta == 8.0)
        )
    ].copy()
    selected["label"] = (
        selected["case"]
        + ", beta="
        + selected.beta.astype(str)
        + ", n="
        + selected.n_tokens.astype(str)
    )
    summary = (
        selected.groupby(["label", "time"])
        .speed.agg(median="median", upper=lambda values: values.quantile(0.9))
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    sns.lineplot(data=summary, x="time", y="median", hue="label", marker="o", ax=axis)
    axis.set_yscale("log")
    axis.set_ylim(1e-16, None)
    axis.set_ylabel("median max-token speed")
    axis.set_title("Long-horizon convergence and metastability audit")
    axis.legend(fontsize=8)
    fig.savefig(output_dir / "long_time_convergence.png", dpi=180)
    plt.close(fig)


def mixed_geometry_plot(output_dir: Path) -> None:
    examples = (
        (2.0, -3.0, 1.5, 1, 1, "indefinite, n=3"),
        (-0.4, -4.0, 0.03, 1, 1, "negative definite, n=3"),
        (-0.4, -4.0, 0.03, 1, 2, "negative definite, n=5"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), constrained_layout=True)
    angles = np.linspace(0, 2 * np.pi, 400)
    for axis, (a, b, beta, n_center, n_polar, title) in zip(
        axes, examples, strict=True
    ):
        roots = mixed_balanced_roots(a, b, beta, n_center, n_polar)
        q = min(roots, key=abs) if a > 0 else roots[0]
        state = mixed_balanced_state(q, n_center, n_polar)
        axis.plot(np.cos(angles), np.sin(angles), color="0.75", linewidth=1)
        axis.scatter(state[:, 0], state[:, 1], c=np.arange(state.shape[0]), s=75, cmap="viridis")
        axis.axhline(0, color="0.9", linewidth=0.8)
        axis.axvline(0, color="0.9", linewidth=0.8)
        axis.set_aspect("equal")
        axis.set_xlim(-1.1, 1.1)
        axis.set_ylim(-1.1, 1.1)
        axis.set_title(f"{title}\nq={q:.6f}")
        axis.set_xlabel("center eigenmode")
    axes[0].set_ylabel("polar eigenmode")
    fig.savefig(output_dir / "mixed_equilibrium_geometry.png", dpi=180)
    plt.close(fig)


def mixed_stability_certificates(data_dir: Path) -> pd.DataFrame:
    """Full-spectrum residuals and tangent rates for reported exact examples."""
    examples = (
        ("indefinite_n3", (2.0, 0.0, -0.5, -3.0), 1.5, 0, 3, 1, 1),
        ("negative_n3_top", (-0.4, -1.0, -2.0, -4.0), 0.03, 0, 3, 1, 1),
        ("negative_n3_mid", (-0.4, -1.0, -2.0, -4.0), 0.03, 2, 3, 1, 1),
        ("negative_n5", (-0.4, -1.0, -2.0, -4.0), 0.03, 0, 3, 1, 2),
        ("indefinite_n20", (2.0, 1.8, -0.5, -3.0), 8.0, 0, 3, 4, 8),
    )
    rows = []
    for name, eigenvalues_tuple, beta, center_mode, polar_mode, n_center, n_polar in examples:
        eigenvalues = np.asarray(eigenvalues_tuple)
        roots = mixed_balanced_roots(
            eigenvalues[center_mode],
            eigenvalues[polar_mode],
            beta,
            n_center,
            n_polar,
        )
        for root_index, q in enumerate(roots):
            state = embed_mixed_balanced_state(
                q,
                eigenvalues.size,
                center_mode,
                polar_mode,
                n_center,
                n_polar,
            )
            rates = np.linalg.eigvals(
                state_tangent_jacobian(state, eigenvalues, beta)
            ).real
            rows.append(
                {
                    "name": name,
                    "root_index": root_index,
                    "q": q,
                    "max_residual": float(
                        np.linalg.norm(
                            vector_field(state, eigenvalues, beta), axis=1
                        ).max()
                    ),
                    "max_linear_rate": float(rates.max()),
                    "min_linear_rate": float(rates.min()),
                    "linearly_stable": bool(rates.max() < -1e-7),
                    "n_center": n_center,
                    "n_each_polar": n_polar,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "theory" / "mixed_stability_certificates.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/spectral_self_attention")
    )
    args = parser.parse_args()
    output_dir = args.data_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    random_trials = pd.read_csv(args.data_dir / "full" / "random_trials.csv")
    long_time = pd.read_csv(args.data_dir / "long_time" / "long_time_audit.csv")
    summary = (
        random_trials.groupby(["case", "family", "beta", "n_tokens", "geometry"])
        .size()
        .rename("count")
        .reset_index()
    )
    summary["fraction"] = summary["count"] / summary.groupby(
        ["case", "beta", "n_tokens"]
    )["count"].transform("sum")
    summary.to_csv(args.data_dir / "full" / "outcome_summary.csv", index=False)
    outcome_heatmaps(random_trials, output_dir)
    indefinite_selection(random_trials, output_dir)
    long_time_plot(long_time, output_dir)
    mixed_geometry_plot(output_dir)
    certificates = mixed_stability_certificates(args.data_dir)
    print(
        {
            "figures": 4,
            "summary_rows": len(summary),
            "mixed_certificates": len(certificates),
        }
    )


if __name__ == "__main__":
    main()
