#!/usr/bin/env python3
"""Analyze multi-axis near-field LSM scaling and build an English PDF."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


COLORS = {
    "hybrid-PCG": "#332288",
    "context-PCG": "#E69F00",
    "population-PCG": "#0072B2",
    "looped-HB": "#D55E00",
    "looped-Chebyshev": "#009E73",
    "looped-Richardson": "#CC79A7",
    "identity-CG": "#222222",
    "exact": "#7A7A7A",
}
MARKERS = {
    "hybrid-PCG": "*",
    "context-PCG": "P",
    "population-PCG": "o",
    "looped-HB": "s",
    "looped-Chebyshev": "^",
    "looped-Richardson": "D",
    "identity-CG": "x",
    "exact": "+",
}
METHOD_ORDER = tuple(COLORS)
LEARNED_ORDER = (
    "hybrid-PCG",
    "context-PCG",
    "population-PCG",
    "looped-HB",
    "looped-Chebyshev",
    "looped-Richardson",
)
CLASSICAL_COMPARISON_ORDER = (
    "hybrid-PCG",
    "context-PCG",
    "population-PCG",
    "angular-Jacobi-PCG",
    "block-Jacobi-PCG",
    "Jacobi-PCG",
    "optimized-CG",
)
CLASSICAL_COLORS = {
    **COLORS,
    "angular-Jacobi-PCG": "#009E73",
    "block-Jacobi-PCG": "#CC79A7",
    "Jacobi-PCG": "#56B4E9",
    "optimized-CG": "#222222",
}
CLASSICAL_MARKERS = {
    **MARKERS,
    "angular-Jacobi-PCG": "^",
    "block-Jacobi-PCG": "D",
    "Jacobi-PCG": "v",
    "optimized-CG": "x",
}
METRICS = (
    "average_precision",
    "auc",
    "iou_top7",
    "centroid_error",
    "area_matched_iou",
    "exact_score_correlation",
    "relative_score_error",
    "mean_relative_residual",
    "covariance_relative_residual",
    "posterior_std_mean",
    "uq_error_correlation",
    "numerical_coverage_95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_data(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    evaluation = read_required(results_dir / "evaluation.csv")
    baselines = read_required(results_dir / "baselines.csv")
    training = read_required(results_dir / "training.csv")
    runtime = read_required(results_dir / "runtime.csv")
    evaluation = evaluation.drop_duplicates(
        [
            "seed",
            "method",
            "network_width",
            "dataset_size",
            "context_size",
            "scenario",
            "task",
        ],
        keep="last",
    )
    baselines = baselines.drop_duplicates(
        ["seed", "method", "context_size", "scenario", "task"], keep="last"
    )
    training = training.drop_duplicates(
        ["seed", "method", "network_width", "dataset_size"], keep="last"
    )
    runtime = runtime.drop_duplicates(
        ["seed", "method", "network_width", "context_size"], keep="last"
    )
    protocol_path = results_dir / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    else:
        protocol = {
            "network_widths": sorted(
                int(value)
                for value in evaluation["network_width"].unique()
                if value > 0
            ),
            "evaluation_context_sizes": sorted(
                int(value) for value in evaluation["context_size"].unique()
            ),
            "training_context_sizes": [12, 16, 24, 32],
            "dataset_sizes": sorted(
                int(value) for value in evaluation["dataset_size"].unique()
            ),
            "eval_depth": 32,
            "finished_requested_grid": False,
        }
    combined = pd.concat([evaluation, baselines], ignore_index=True)
    combined["regime_class"] = np.where(combined["regime"] == "ID", "ID", "OOD")
    return combined, training, runtime, protocol


def add_all_regime(frame: pd.DataFrame) -> pd.DataFrame:
    all_rows = frame.copy()
    all_rows["regime_class"] = "All"
    return pd.concat([frame, all_rows], ignore_index=True)


def add_original_residual_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep aggregation backward compatible with pre-audit residual files."""
    output = frame.copy()
    aliases = {
        "original_mean_relative_residual": "mean_relative_residual",
        "original_covariance_relative_residual": "covariance_relative_residual",
        "original_mean_relative_residual_mean": "mean_relative_residual_mean",
        "original_covariance_relative_residual_mean": (
            "covariance_relative_residual_mean"
        ),
    }
    for target, source in aliases.items():
        if target not in output.columns and source in output.columns:
            output[target] = output[source]
    return output


def aggregate_tasks(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "method",
        "network_width",
        "parameter_count",
        "dataset_size",
        "context_size",
        "context_measurements",
        "regime_class",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "n_tasks": ("average_precision", "size"),
        "n_seeds": ("seed", "nunique"),
        "training_seconds": ("training_seconds", "median"),
    }
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    output = frame.groupby(keys, as_index=False).agg(**aggregations)
    for metric in METRICS:
        output[f"{metric}_ci95"] = (
            1.96
            * output[f"{metric}_std"].fillna(0.0)
            / np.sqrt(output["n_tasks"].clip(lower=1))
        )
    output["localization_risk_mean"] = 1.0 - output["average_precision_mean"]
    output["localization_risk_ci95"] = output["average_precision_ci95"]
    return output


def simultaneous_heldout_bounds(learned: pd.DataFrame, delta: float) -> pd.DataFrame:
    """One-sided distribution-free Hoeffding bounds over all configurations."""
    expanded = add_all_regime(learned)
    keys = [
        "method",
        "network_width",
        "parameter_count",
        "dataset_size",
        "context_size",
        "context_measurements",
        "regime_class",
    ]
    grouped = expanded.assign(
        localization_risk=1.0 - expanded["average_precision"],
        solver_risk=expanded["mean_relative_residual"]
        / (1.0 + expanded["mean_relative_residual"]),
    ).groupby(keys, as_index=False)
    bounds = grouped.agg(
        risk_mean=("localization_risk", "mean"),
        risk_std=("localization_risk", "std"),
        solver_risk_mean=("solver_risk", "mean"),
        solver_risk_std=("solver_risk", "std"),
        n_holdout=("localization_risk", "size"),
        n_seeds=("seed", "nunique"),
        training_seconds=("training_seconds", "median"),
        mean_relative_residual=("mean_relative_residual", "mean"),
        covariance_relative_residual=("covariance_relative_residual", "mean"),
    )
    configurations = len(bounds)
    bounds["simultaneous_configurations"] = configurations
    bounds["delta"] = delta
    bounds["hoeffding_slack"] = np.sqrt(
        np.log(configurations / delta) / (2.0 * bounds["n_holdout"])
    )
    bounds["risk_ucb"] = np.minimum(
        1.0, bounds["risk_mean"] + bounds["hoeffding_slack"]
    )
    bounds["solver_risk_ucb"] = np.minimum(
        1.0, bounds["solver_risk_mean"] + bounds["hoeffding_slack"]
    )
    return bounds


def simultaneous_geometry_bounds(frame: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Hoeffding bounds whose independent unit is an acquisition geometry."""
    frame = frame.copy()
    frame["geometry_unit"] = (
        frame["seed"].astype(str)
        + ":"
        + frame["scenario"].astype(str)
        + ":"
        + frame["geometry_draw"].astype(str)
    )
    all_rows = frame.copy()
    all_rows["scenario"] = "All scenarios"
    all_rows["regime"] = "All"
    expanded = pd.concat([frame, all_rows], ignore_index=True)
    keys = ["method", "network_width", "depth", "context_size", "scenario", "regime"]
    bounds = expanded.groupby(keys, as_index=False).agg(
        risk_mean=("localization_risk", "mean"),
        solver_risk_mean=("solver_risk", "mean"),
        n_geometry_batches=("geometry_draw", "size"),
        n_unique_geometries=("geometry_unit", "nunique"),
        n_trained_seeds=("seed", "nunique"),
        tasks_per_geometry=("tasks_per_geometry", "first"),
    )
    configurations = len(bounds)
    bounds["simultaneous_configurations"] = configurations
    bounds["delta"] = delta
    bounds["hoeffding_slack"] = np.sqrt(
        np.log(configurations / delta) / (2.0 * bounds["n_geometry_batches"])
    )
    bounds["risk_ucb"] = np.minimum(
        1.0, bounds["risk_mean"] + bounds["hoeffding_slack"]
    )
    bounds["solver_risk_ucb"] = np.minimum(
        1.0, bounds["solver_risk_mean"] + bounds["hoeffding_slack"]
    )
    return bounds


def paired_geometry_effects(frame: pd.DataFrame, context_size: int) -> pd.DataFrame:
    """Paired residual gains and 95% intervals over independent geometries."""
    selected = frame[frame["context_size"] == context_size]
    residual_column = (
        "original_mean_relative_residual"
        if "original_mean_relative_residual" in selected.columns
        else "mean_relative_residual"
    )
    pivot = selected.pivot_table(
        index=["seed", "geometry_draw", "scenario"],
        columns="method",
        values=residual_column,
        aggfunc="mean",
    )
    average_precision = selected.pivot_table(
        index=["seed", "geometry_draw", "scenario"],
        columns="method",
        values="average_precision",
        aggfunc="mean",
    )
    coverage = selected.pivot_table(
        index=["seed", "geometry_draw", "scenario"],
        columns="method",
        values="numerical_coverage_95",
        aggfunc="mean",
    )
    comparisons = {
        "CG / context-PCG": ("identity-CG", "context-PCG"),
        "CG / hybrid-PCG": ("identity-CG", "hybrid-PCG"),
        "CG / angular-Jacobi-PCG": ("identity-CG", "angular-Jacobi-PCG"),
        "angular-Jacobi-PCG / context-PCG": (
            "angular-Jacobi-PCG",
            "context-PCG",
        ),
    }
    rows: list[dict[str, object]] = []
    scenarios = [*sorted(pivot.index.get_level_values("scenario").unique()), "All"]
    z = 1.96
    for scenario in scenarios:
        group = (
            pivot
            if scenario == "All"
            else pivot[pivot.index.get_level_values("scenario") == scenario]
        )
        for comparison, (reference, candidate) in comparisons.items():
            if not {reference, candidate}.issubset(group.columns):
                continue
            paired = group[[reference, candidate]].dropna()
            gain = (
                paired[reference].clip(lower=1.0e-12)
                / paired[candidate].clip(lower=1.0e-12)
            )
            log_gain = np.log(gain.to_numpy())
            n_batches = len(log_gain)
            mean_log = float(log_gain.mean())
            standard_error = (
                float(log_gain.std(ddof=1) / np.sqrt(n_batches))
                if n_batches > 1
                else 0.0
            )
            win_rate = float((gain > 1.0).mean())
            denominator = 1.0 + z**2 / n_batches
            center = (win_rate + z**2 / (2.0 * n_batches)) / denominator
            half_width = (
                z
                * np.sqrt(
                    win_rate * (1.0 - win_rate) / n_batches
                    + z**2 / (4.0 * n_batches**2)
                )
                / denominator
            )
            ap_group = (
                average_precision
                if scenario == "All"
                else average_precision[
                    average_precision.index.get_level_values("scenario")
                    == scenario
                ]
            )
            coverage_group = (
                coverage
                if scenario == "All"
                else coverage[
                    coverage.index.get_level_values("scenario") == scenario
                ]
            )
            ap_delta = (
                ap_group[candidate] - ap_group[reference]
            ).dropna().to_numpy()
            coverage_delta = (
                coverage_group[candidate] - coverage_group[reference]
            ).dropna().to_numpy()

            def paired_mean_interval(values: np.ndarray) -> tuple[float, float, float]:
                mean = float(values.mean())
                standard_error = (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
                return (
                    mean,
                    mean - z * standard_error,
                    mean + z * standard_error,
                )

            ap_mean, ap_lower, ap_upper = paired_mean_interval(ap_delta)
            coverage_mean, coverage_lower, coverage_upper = paired_mean_interval(
                coverage_delta
            )
            rows.append(
                {
                    "scenario": scenario,
                    "comparison": comparison,
                    "reference": reference,
                    "candidate": candidate,
                    "n_geometry_batches": n_batches,
                    "geometric_mean_gain": math.exp(mean_log),
                    "gain_ci95_lower": math.exp(mean_log - z * standard_error),
                    "gain_ci95_upper": math.exp(mean_log + z * standard_error),
                    "median_gain": float(np.median(gain)),
                    "candidate_win_rate": win_rate,
                    "win_rate_ci95_lower": max(0.0, center - half_width),
                    "win_rate_ci95_upper": min(1.0, center + half_width),
                    "average_precision_delta_mean": ap_mean,
                    "average_precision_delta_ci95_lower": ap_lower,
                    "average_precision_delta_ci95_upper": ap_upper,
                    "coverage_delta_mean": coverage_mean,
                    "coverage_delta_ci95_lower": coverage_lower,
                    "coverage_delta_ci95_upper": coverage_upper,
                }
            )
    return pd.DataFrame(rows)


def paired_cg_stress_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired CG gains along continuous one-factor physical stress axes."""
    residual_column = (
        "original_mean_relative_residual"
        if "original_mean_relative_residual" in frame.columns
        else "mean_relative_residual"
    )
    pivot = frame.pivot_table(
        index=["seed", "axis", "level", "geometry_draw"],
        columns="method",
        values=residual_column,
        aggfunc="mean",
    )
    average_precision = frame.pivot_table(
        index=["seed", "axis", "level", "geometry_draw"],
        columns="method",
        values="average_precision",
        aggfunc="mean",
    )
    coverage = frame.pivot_table(
        index=["seed", "axis", "level", "geometry_draw"],
        columns="method",
        values="numerical_coverage_95",
        aggfunc="mean",
    )
    candidates = (
        "context-PCG",
        "hybrid-PCG",
        "population-PCG",
        "angular-Jacobi-PCG",
        "looped-HB",
    )
    rows: list[dict[str, object]] = []
    z = 1.96
    for (axis, level), group in pivot.groupby(level=["axis", "level"]):
        for candidate in candidates:
            if not {"identity-CG", candidate}.issubset(group.columns):
                continue
            paired = group[["identity-CG", candidate]].dropna()
            gains = (
                paired["identity-CG"].clip(lower=1.0e-12)
                / paired[candidate].clip(lower=1.0e-12)
            )
            log_gains = np.log(gains.to_numpy())
            n_batches = len(log_gains)
            mean_log = float(log_gains.mean())
            standard_error = (
                float(log_gains.std(ddof=1) / np.sqrt(n_batches))
                if n_batches > 1
                else 0.0
            )
            ap_delta = (
                average_precision.loc[group.index, candidate]
                - average_precision.loc[group.index, "identity-CG"]
            ).dropna().to_numpy()
            coverage_delta = (
                coverage.loc[group.index, candidate]
                - coverage.loc[group.index, "identity-CG"]
            ).dropna().to_numpy()

            def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
                mean = float(values.mean())
                standard_error = (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
                return (
                    mean,
                    mean - z * standard_error,
                    mean + z * standard_error,
                )

            ap_mean, ap_lower, ap_upper = mean_interval(ap_delta)
            coverage_mean, coverage_lower, coverage_upper = mean_interval(
                coverage_delta
            )
            win_rate = float((gains > 1.0).mean())
            denominator = 1.0 + z**2 / n_batches
            center = (win_rate + z**2 / (2.0 * n_batches)) / denominator
            half_width = (
                z
                * np.sqrt(
                    win_rate * (1.0 - win_rate) / n_batches
                    + z**2 / (4.0 * n_batches**2)
                )
                / denominator
            )
            rows.append(
                {
                    "axis": str(axis),
                    "level": float(level),
                    "candidate": candidate,
                    "n_geometry_batches": n_batches,
                    "geometric_mean_gain": math.exp(mean_log),
                    "gain_ci95_lower": math.exp(mean_log - z * standard_error),
                    "gain_ci95_upper": math.exp(mean_log + z * standard_error),
                    "median_gain": float(np.median(gains)),
                    "candidate_win_rate": win_rate,
                    "win_rate_ci95_lower": max(0.0, center - half_width),
                    "win_rate_ci95_upper": min(1.0, center + half_width),
                    "average_precision_delta_mean": ap_mean,
                    "average_precision_delta_ci95_lower": ap_lower,
                    "average_precision_delta_ci95_upper": ap_upper,
                    "coverage_delta_mean": coverage_mean,
                    "coverage_delta_ci95_lower": coverage_lower,
                    "coverage_delta_ci95_upper": coverage_upper,
                }
            )
    return pd.DataFrame(rows).sort_values(["axis", "candidate", "level"])


def save_cg_stress_sweep(path: Path, effects: pd.DataFrame) -> None:
    """Plot paired CG gains under progressively harder physical conditions."""
    axes_order = (
        ("obstacle_count", "number of obstacles"),
        ("relative_noise", "relative noise"),
        ("aperture_degrees", "aperture (degrees)"),
        ("wavenumber", "wavenumber $k$"),
        ("joint_severity", "joint-shift severity"),
    )
    methods = (
        "context-PCG",
        "hybrid-PCG",
        "population-PCG",
        "angular-Jacobi-PCG",
        "looped-HB",
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 6.9), constrained_layout=True)
    for plot_axis, (axis_name, x_label) in zip(
        axes.flat[: len(axes_order)], axes_order, strict=True
    ):
        selected = effects[effects["axis"] == axis_name]
        for method in methods:
            group = selected[selected["candidate"] == method].sort_values("level")
            if group.empty:
                continue
            x_values = group["level"].to_numpy()
            gains = group["geometric_mean_gain"].to_numpy()
            plot_axis.plot(
                x_values,
                gains,
                color=CLASSICAL_COLORS[method],
                marker=CLASSICAL_MARKERS[method],
                linewidth=1.8,
                label=method,
            )
            plot_axis.fill_between(
                x_values,
                group["gain_ci95_lower"].to_numpy(),
                group["gain_ci95_upper"].to_numpy(),
                color=CLASSICAL_COLORS[method],
                alpha=0.10,
            )
        plot_axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
        plot_axis.set_yscale("log")
        plot_axis.set_xlabel(x_label)
        plot_axis.set_ylabel("paired physical-residual gain CG / candidate")
        plot_axis.grid(alpha=0.22)
        if axis_name == "aperture_degrees":
            plot_axis.invert_xaxis()
    summary_axis = axes.flat[-1]
    global_gains = (
        effects.groupby("candidate")["geometric_mean_gain"]
        .apply(lambda values: float(np.exp(np.log(values).mean())))
        .reindex(methods)
    )
    y_values = np.arange(len(global_gains))
    summary_axis.barh(
        y_values,
        global_gains.to_numpy(),
        color=[CLASSICAL_COLORS[method] for method in global_gains.index],
    )
    summary_axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    summary_axis.set_xscale("log")
    summary_axis.set_yticks(
        y_values,
        [
            {
                "context-PCG": "context",
                "hybrid-PCG": "hybrid",
                "population-PCG": "population",
                "angular-Jacobi-PCG": "angular",
                "looped-HB": "HB",
            }[method]
            for method in global_gains.index
        ],
    )
    summary_axis.invert_yaxis()
    summary_axis.set_xlabel("global physical-residual CG gain")
    summary_axis.set_title("All stress levels")
    summary_axis.grid(axis="x", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_joint_conditioning(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate and plot the spectral mechanism under joint shifts."""
    frame = add_original_residual_aliases(frame)
    aggregate = frame.groupby(
        ["joint_severity", "method"], as_index=False
    ).agg(
        n_geometry_batches=("geometry_draw", "size"),
        raw_condition_median=("raw_condition_median", "median"),
        transformed_condition_median=("transformed_condition_median", "median"),
        condition_reduction_median=("condition_reduction_median", "median"),
        geometry_commutator_mean=("geometry_commutator_mean", "mean"),
        transformed_mean_relative_residual_mean=(
            "mean_relative_residual_mean",
            "mean",
        ),
        mean_relative_residual_mean=(
            "original_mean_relative_residual_mean",
            "mean",
        ),
        covariance_relative_residual_mean=(
            "original_covariance_relative_residual_mean",
            "mean",
        ),
        transformed_covariance_relative_residual_mean=(
            "covariance_relative_residual_mean",
            "mean",
        ),
    )
    methods = (
        "identity-CG",
        "context-PCG",
        "hybrid-PCG",
        "population-PCG",
        "angular-Jacobi-PCG",
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), constrained_layout=True)
    for method in methods:
        group = aggregate[aggregate["method"] == method].sort_values(
            "joint_severity"
        )
        if group.empty:
            continue
        for axis, metric in (
            (axes[0], "transformed_condition_median"),
            (axes[1], "condition_reduction_median"),
        ):
            axis.plot(
                group["joint_severity"],
                group[metric],
                color=CLASSICAL_COLORS[method],
                marker=CLASSICAL_MARKERS[method],
                linewidth=1.8,
                label=method,
            )
        axes[2].plot(
            group["transformed_condition_median"],
            group["mean_relative_residual_mean"],
            color=CLASSICAL_COLORS[method],
            marker=CLASSICAL_MARKERS[method],
            linewidth=1.8,
            label=method,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("joint-shift severity")
    axes[0].set_ylabel("median transformed condition number")
    axes[0].grid(alpha=0.22)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("joint-shift severity")
    axes[1].set_ylabel("median conditioning gain")
    axes[1].grid(alpha=0.22)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("median transformed condition number")
    axes[2].set_ylabel("physical posterior-mean residual")
    axes[2].grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return aggregate


def scaling_prediction(parameters: np.ndarray, design: np.ndarray) -> np.ndarray:
    (
        floor,
        data_amplitude,
        data_exponent,
        context_amplitude,
        context_exponent,
        width_amplitude,
        width_exponent,
        overfit,
    ) = parameters
    n_ratio, context_ratio, parameter_ratio = design.T
    return (
        floor
        + data_amplitude * n_ratio ** (-data_exponent)
        + context_amplitude * context_ratio ** (-context_exponent)
        + width_amplitude * parameter_ratio ** (-width_exponent)
        + overfit * np.sqrt(parameter_ratio / n_ratio)
    )


def fit_scaling_laws(bounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    targets = {
        "localization": "risk_mean",
        "solver": "solver_risk_mean",
    }
    for (method, regime), group in bounds.groupby(["method", "regime_class"]):
        if regime != "All" or len(group) < 12:
            continue
        n_min = float(group["dataset_size"].min())
        context_min = float(group["context_size"].min())
        parameter_min = float(group["parameter_count"].min())
        design = np.column_stack(
            [
                group["dataset_size"].to_numpy(float) / n_min,
                group["context_size"].to_numpy(float) / context_min,
                group["parameter_count"].to_numpy(float) / parameter_min,
            ]
        )
        for risk_type, target_column in targets.items():
            target = group[target_column].to_numpy(float)
            initial = np.asarray([0.05, 0.08, 0.3, 0.25, 0.0, 0.04, 0.3, 0.01])
            lower = np.asarray([0.0, 0.0, 0.01, 0.0, -3.0, 0.0, 0.01, 0.0])
            upper = np.asarray([1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0])
            result = least_squares(
                lambda values: scaling_prediction(values, design) - target,
                initial,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.02,
                max_nfev=20_000,
            )
            fitted = scaling_prediction(result.x, design)
            residual_sum = float(np.square(target - fitted).sum())
            total_sum = float(np.square(target - target.mean()).sum())
            r_squared = 1.0 - residual_sum / max(total_sum, 1.0e-12)
            names = (
                "risk_floor",
                "data_amplitude",
                "data_exponent",
                "context_amplitude",
                "context_exponent",
                "width_amplitude",
                "width_exponent",
                "overfit_amplitude",
            )
            record: dict[str, object] = {
                "method": method,
                "risk_type": risk_type,
                "regime": regime,
                "n_observations": len(group),
                "n_reference": n_min,
                "context_reference": context_min,
                "parameter_reference": parameter_min,
                "r_squared": r_squared,
                "rmse": math.sqrt(residual_sum / len(group)),
                "fit_success": bool(result.success),
            }
            record.update(dict(zip(names, result.x, strict=True)))
            records.append(record)
            prediction = group.copy()
            prediction["risk_type"] = risk_type
            prediction["fitted_risk"] = fitted
            predictions.append(prediction)
    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True)


def central_width(protocol: dict) -> int:
    return min(protocol["network_widths"], key=lambda value: abs(value - 128))


def plot_line(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    yerr: str | None = None,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> None:
    for method in methods:
        group = frame[frame["method"] == method].sort_values(x)
        if group.empty:
            continue
        axis.plot(
            group[x],
            group[y],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4,
            linewidth=1.8,
            label=method,
        )
        if yerr is not None and yerr in group:
            low = np.maximum(0.0, group[y] - group[yerr])
            high = group[y] + group[yerr]
            axis.fill_between(group[x], low, high, color=COLORS[method], alpha=0.12)


def baseline_level(
    baselines: pd.DataFrame, method: str, context: int, regime: str, metric: str
) -> float:
    selected = baselines[
        (baselines["method"] == method)
        & (baselines["context_size"] == context)
        & (baselines["regime_class"] == regime)
    ]
    return float(selected[metric].mean()) if not selected.empty else float("nan")


def save_dataset_scaling(
    path: Path,
    aggregate: pd.DataFrame,
    baselines: pd.DataFrame,
    width: int,
    context: int,
) -> None:
    learned = aggregate[
        (aggregate["network_width"] == width)
        & (aggregate["context_size"] == context)
        & (aggregate["dataset_size"] > 0)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    panels = (
        (
            "ID",
            "localization_risk_mean",
            "localization_risk_ci95",
            "ID localization risk",
        ),
        (
            "OOD",
            "localization_risk_mean",
            "localization_risk_ci95",
            "OOD localization risk",
        ),
        (
            "All",
            "mean_relative_residual_mean",
            "mean_relative_residual_ci95",
            "transformed mean residual",
        ),
        (
            "All",
            "covariance_relative_residual_mean",
            "covariance_relative_residual_ci95",
            "transformed covariance residual",
        ),
    )
    for axis, (regime, metric, error, title) in zip(axes.flat, panels, strict=True):
        selected = learned[learned["regime_class"] == regime]
        plot_line(
            axis, selected, "dataset_size", metric, yerr=error, methods=LEARNED_ORDER
        )
        for baseline in ("identity-CG", "exact"):
            value = baseline_level(baselines, baseline, context, regime, metric)
            if np.isfinite(value):
                axis.axhline(
                    value,
                    color=COLORS[baseline],
                    linestyle="--" if baseline == "identity-CG" else ":",
                    linewidth=1.4,
                    label=baseline,
                )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("training inverse problems $n$")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        if "residual" in metric:
            axis.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_context_scaling(
    path: Path,
    aggregate: pd.DataFrame,
    baselines: pd.DataFrame,
    width: int,
    dataset_size: int,
) -> None:
    learned = aggregate[
        (aggregate["network_width"] == width)
        & (aggregate["dataset_size"] == dataset_size)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    panels = (
        (
            "ID",
            "average_precision_mean",
            "average_precision_ci95",
            "ID average precision",
        ),
        (
            "OOD",
            "average_precision_mean",
            "average_precision_ci95",
            "OOD average precision",
        ),
        (
            "All",
            "mean_relative_residual_mean",
            "mean_relative_residual_ci95",
            "transformed mean residual",
        ),
        (
            "All",
            "relative_score_error_mean",
            "relative_score_error_ci95",
            "score error vs exact",
        ),
    )
    for axis, (regime, metric, error, title) in zip(axes.flat, panels, strict=True):
        selected = learned[learned["regime_class"] == regime]
        plot_line(
            axis, selected, "context_size", metric, yerr=error, methods=LEARNED_ORDER
        )
        baseline_selected = baselines[baselines["regime_class"] == regime]
        plot_line(
            axis,
            baseline_selected,
            "context_size",
            metric,
            methods=("identity-CG", "exact"),
        )
        axis.set_xlabel("source/receiver tokens $m$")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        if "residual" in metric or "error" in metric:
            axis.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_network_scaling(
    path: Path, aggregate: pd.DataFrame, dataset_size: int, context: int
) -> None:
    selected = aggregate[
        (aggregate["dataset_size"] == dataset_size)
        & (aggregate["context_size"] == context)
        & (aggregate["regime_class"] == "All")
        & (aggregate["network_width"] > 0)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    panels = (
        (
            "localization_risk_mean",
            "localization_risk_ci95",
            "localization risk",
            False,
        ),
        (
            "mean_relative_residual_mean",
            "mean_relative_residual_ci95",
            "transformed mean residual",
            True,
        ),
        (
            "covariance_relative_residual_mean",
            "covariance_relative_residual_ci95",
            "transformed covariance residual",
            True,
        ),
    )
    for axis, (metric, error, title, log_y) in zip(axes, panels, strict=True):
        plot_line(
            axis,
            selected,
            "parameter_count",
            metric,
            yerr=error,
            methods=LEARNED_ORDER,
        )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("trainable parameters $P$")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_time_learning(
    path: Path, aggregate: pd.DataFrame, width: int, context: int
) -> None:
    selected = aggregate[
        (aggregate["network_width"] == width)
        & (aggregate["context_size"] == context)
        & (aggregate["regime_class"] == "All")
        & (aggregate["dataset_size"] > 0)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "localization_risk_mean", "localization risk"),
        (axes[1], "mean_relative_residual_mean", "transformed mean residual"),
    ):
        plot_line(axis, selected, "training_seconds", metric, methods=LEARNED_ORDER)
        axis.set_xscale("log")
        if "residual" in metric:
            axis.set_yscale("log")
        axis.set_xlabel("cumulative training time (s)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_generalization_bounds(path: Path, bounds: pd.DataFrame, width: int) -> None:
    selected = bounds[
        (bounds["network_width"] == width) & (bounds["regime_class"] == "All")
    ]
    contexts = [
        value for value in (8, 24, 48) if value in set(selected["context_size"])
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    bound_methods = (
        ("context-PCG", "population-PCG")
        if "context-PCG" in set(selected["method"])
        else ("population-PCG", "looped-HB")
    )
    for axis, method in zip(axes[0], bound_methods, strict=True):
        method_data = selected[selected["method"] == method]
        for index, context in enumerate(contexts):
            group = method_data[method_data["context_size"] == context].sort_values(
                "dataset_size"
            )
            color = plt.cm.viridis(index / max(len(contexts) - 1, 1))
            axis.plot(
                group["dataset_size"],
                group["risk_mean"],
                marker="o",
                color=color,
                label=f"$m={context}$ empirical",
            )
            axis.plot(
                group["dataset_size"],
                group["risk_ucb"],
                linestyle="--",
                color=color,
                label=f"$m={context}$ 95% UCB",
            )
        axis.set_xscale("log", base=2)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("training inverse problems $n$")
        axis.set_ylabel("held-out localization risk")
        axis.set_title(method)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7, ncol=2)

    solver_method = (
        "context-PCG" if "context-PCG" in set(selected["method"]) else "population-PCG"
    )
    pcg = selected[selected["method"] == solver_method]
    dataset_sizes = sorted(pcg["dataset_size"].unique())
    context_sizes = sorted(pcg["context_size"].unique())
    for axis, metric, title in (
        (axes[1, 0], "solver_risk_mean", f"{solver_method} empirical solver risk"),
        (
            axes[1, 1],
            "solver_risk_ucb",
            f"{solver_method} simultaneous 95% solver-risk bound",
        ),
    ):
        pivot = pcg.pivot_table(
            index="context_size", columns="dataset_size", values=metric, aggfunc="mean"
        ).reindex(index=context_sizes, columns=dataset_sizes)
        image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=min(1.0, float(np.nanmax(pivot.to_numpy()))),
            cmap="magma_r",
        )
        axis.set_xticks(
            range(len(dataset_sizes)),
            [f"{value:g}" for value in dataset_sizes],
            rotation=45,
        )
        axis.set_yticks(
            range(len(context_sizes)), [str(value) for value in context_sizes]
        )
        axis.set_xlabel("training set size $n$")
        axis.set_ylabel("context size $m$")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_geometry_generalization_bounds(path: Path, bounds: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)
    panels = (
        ("All scenarios", "risk_mean", "risk_ucb", "localization risk"),
        ("All scenarios", "solver_risk_mean", "solver_risk_ucb", "solver risk"),
        (
            "OOD half aperture",
            "risk_mean",
            "risk_ucb",
            "half-aperture localization risk",
        ),
        (
            "OOD wavenumber 12",
            "solver_risk_mean",
            "solver_risk_ucb",
            "frequency-OOD solver risk",
        ),
    )
    methods = (
        "hybrid-PCG",
        "context-PCG",
        "population-PCG",
        "angular-Jacobi-PCG",
        "identity-CG",
        "exact",
    )
    for axis, (scenario, empirical, upper, title) in zip(
        axes.flat, panels, strict=True
    ):
        selected = bounds[bounds["scenario"] == scenario]
        for method in methods:
            group = selected[selected["method"] == method].sort_values(
                "context_size"
            )
            if group.empty:
                continue
            axis.plot(
                group["context_size"],
                group[empirical],
                color=CLASSICAL_COLORS[method],
                marker=CLASSICAL_MARKERS[method],
                linewidth=1.8,
                label=f"{method} empirical",
            )
            axis.plot(
                group["context_size"],
                group[upper],
                color=CLASSICAL_COLORS[method],
                linestyle="--",
                linewidth=1.2,
                label="_nolegend_",
            )
        axis.set_xlabel("context size $m$")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_runtime_scaling(
    path: Path,
    runtime: pd.DataFrame,
    aggregate: pd.DataFrame,
    width: int,
    dataset_size: int,
) -> None:
    run = (
        runtime[
            (runtime["network_width"].isin((0, width)))
            & (runtime["method"].isin(METHOD_ORDER))
        ]
        .groupby(["method", "context_size", "context_measurements"], as_index=False)
        .agg(
            inference_ms=("inference_ms", "median"),
            peak_memory_mib=("peak_memory_mib", "median"),
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    plot_line(axes[0], run, "context_measurements", "inference_ms")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("near-field measurements $m^2$")
    axes[0].set_ylabel("inference time per batch (ms)")
    axes[0].grid(alpha=0.22)

    final = aggregate[
        (aggregate["regime_class"] == "All")
        & (aggregate["context_size"] == 24)
        & (
            (
                (aggregate["network_width"] == width)
                & (aggregate["dataset_size"] == dataset_size)
            )
            | (aggregate["network_width"] == 0)
        )
    ].copy()
    times = run[run["context_size"] == 24][["method", "inference_ms"]]
    final = final.merge(times, on="method", how="left")
    for method in METHOD_ORDER:
        group = final[final["method"] == method]
        if group.empty:
            continue
        axes[1].scatter(
            group["inference_ms"],
            group["mean_relative_residual_mean"],
            color=COLORS[method],
            marker=MARKERS[method],
            s=45,
            label=method,
        )
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("inference time per batch (ms)")
    axes[1].set_ylabel("transformed mean residual")
    axes[1].grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def aggregate_depth_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the post-training depth sweep without pooling ID and OOD twice."""
    expanded = add_original_residual_aliases(frame)
    expanded["regime_class"] = np.where(expanded["regime"] == "ID", "ID", "OOD")
    all_rows = expanded.copy()
    all_rows["regime_class"] = "All"
    expanded = pd.concat([expanded, all_rows], ignore_index=True)
    keys = [
        "method",
        "network_width",
        "parameter_count",
        "dataset_size",
        "depth",
        "context_size",
        "context_measurements",
        "regime_class",
    ]
    output = expanded.groupby(keys, as_index=False).agg(
        n_tasks=("average_precision", "size"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        transformed_mean_relative_residual_mean=(
            "mean_relative_residual",
            "mean",
        ),
        mean_relative_residual_mean=("original_mean_relative_residual", "mean"),
        mean_relative_residual_std=("original_mean_relative_residual", "std"),
        covariance_relative_residual_mean=(
            "original_covariance_relative_residual",
            "mean",
        ),
        transformed_covariance_relative_residual_mean=(
            "covariance_relative_residual",
            "mean",
        ),
        relative_score_error_mean=("relative_score_error", "mean"),
        numerical_coverage_95_mean=("numerical_coverage_95", "mean"),
    )
    for metric in ("average_precision", "mean_relative_residual"):
        output[f"{metric}_ci95"] = (
            1.96
            * output[f"{metric}_std"].fillna(0.0)
            / np.sqrt(output["n_tasks"].clip(lower=1))
        )
    return output


def save_depth_scaling(
    path: Path,
    depth_audit: pd.DataFrame,
    depth_runtime: pd.DataFrame,
    *,
    context: int,
    largest_context: int,
) -> None:
    selected = depth_audit[depth_audit["regime_class"] == "All"]
    run = (
        depth_runtime[depth_runtime["depth"] > 0]
        .groupby(["method", "depth", "context_size"], as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    panels = (
        (
            axes[0, 0],
            context,
            "average_precision_mean",
            "average_precision_ci95",
            f"average precision, $m={context}$",
            False,
        ),
        (
            axes[0, 1],
            context,
            "mean_relative_residual_mean",
            "mean_relative_residual_ci95",
            f"physical mean residual, $m={context}$",
            True,
        ),
        (
            axes[1, 0],
            largest_context,
            "mean_relative_residual_mean",
            "mean_relative_residual_ci95",
            f"physical mean residual, extrapolative $m={largest_context}$",
            True,
        ),
    )
    iterative_methods = METHOD_ORDER[:-1]
    for axis, context_value, metric, error, title, log_y in panels:
        data = selected[
            (selected["context_size"] == context_value)
            & (selected["depth"] > 0)
        ]
        plot_line(
            axis,
            data,
            "depth",
            metric,
            yerr=error,
            methods=iterative_methods,
        )
        exact = selected[
            (selected["context_size"] == context_value)
            & (selected["method"] == "exact")
        ]
        if not exact.empty:
            axis.axhline(
                float(exact[metric].mean()),
                color=COLORS["exact"],
                linestyle=":",
                linewidth=1.4,
                label="exact",
            )
        axis.set_xscale("log", base=2)
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("operator applications / recurrent depth $T$")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.22)

    pareto = selected[
        (selected["context_size"] == largest_context) & (selected["depth"] > 0)
    ].merge(
        run,
        on=["method", "depth", "context_size"],
        how="inner",
    )
    for method in iterative_methods:
        group = pareto[pareto["method"] == method].sort_values("depth")
        if group.empty:
            continue
        axes[1, 1].plot(
            group["inference_ms"],
            group["mean_relative_residual_mean"],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            label=method,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("inference time per batch (ms)")
    axes[1, 1].set_ylabel("physical mean residual")
    axes[1, 1].set_title(f"depth--time Pareto, $m={largest_context}$")
    axes[1, 1].grid(alpha=0.22)
    for axis in (axes[1, 0], axes[1, 1]):
        axis.set_ylim(1.0e-5, 3.0)
    axes[1, 0].text(
        0.97,
        0.96,
        "Chebyshev divergence is off-scale",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_conditioning_audit(path: Path, audit: pd.DataFrame) -> pd.DataFrame:
    enriched = audit.copy()
    enriched["regime_class"] = np.where(enriched["regime"] == "ID", "ID", "OOD")
    all_rows = enriched.copy()
    all_rows["regime_class"] = "All"
    enriched = pd.concat([enriched, all_rows], ignore_index=True)
    aggregate = enriched.groupby(
        ["method", "context_size", "context_measurements", "regime_class"],
        as_index=False,
    ).agg(
        n_tasks=("task", "size"),
        raw_condition_median=("raw_condition", "median"),
        transformed_condition_median=("transformed_condition", "median"),
        condition_reduction_median=("condition_reduction", "median"),
        geometry_commutator_mean=("geometry_commutator", "mean"),
        energy_factor_median=("pcg_energy_error_factor", "median"),
        transformed_mean_residual_median=("mean_relative_residual", "median"),
        transformed_covariance_residual_median=(
            "covariance_relative_residual",
            "median",
        ),
        gain_spread_median=("gain_spread", "median"),
    )
    selected = aggregate[aggregate["regime_class"] == "All"]
    methods = ("hybrid-PCG", "context-PCG", "population-PCG", "identity-CG")
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)
    plot_line(
        axes[0, 0],
        selected,
        "context_size",
        "transformed_condition_median",
        methods=methods,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("context size $m$")
    axes[0, 0].set_ylabel("median transformed condition number")
    axes[0, 0].grid(alpha=0.22)

    plot_line(
        axes[0, 1],
        selected,
        "context_size",
        "condition_reduction_median",
        methods=methods,
    )
    axes[0, 1].axhline(1.0, color="#777777", linestyle=":", linewidth=1.2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("context size $m$")
    axes[0, 1].set_ylabel(r"conditioning gain $\kappa(A)/\kappa(C^*AC)$")
    axes[0, 1].grid(alpha=0.22)

    commutator = aggregate[
        (aggregate["method"] == "identity-CG")
        & (aggregate["regime_class"].isin(("ID", "OOD")))
    ]
    for regime, marker in (("ID", "o"), ("OOD", "s")):
        group = commutator[commutator["regime_class"] == regime].sort_values(
            "context_size"
        )
        axes[1, 0].plot(
            group["context_size"],
            group["geometry_commutator_mean"],
            marker=marker,
            linewidth=1.8,
            label=regime,
        )
    axes[1, 0].set_xlabel("context size $m$")
    axes[1, 0].set_ylabel("normalized geometry commutator")
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].legend(frameon=False)

    for method in methods:
        group = selected[selected["method"] == method].sort_values(
            "transformed_condition_median"
        )
        if group.empty:
            continue
        axes[1, 1].plot(
            group["transformed_condition_median"],
            group["transformed_mean_residual_median"],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            label=method,
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("median transformed condition number")
    axes[1, 1].set_ylabel("median transformed mean residual at $T=32$")
    axes[1, 1].grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return aggregate


def aggregate_classical_preconditioners(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate classical-PCG controls with the same ID/OOD convention."""
    expanded = add_original_residual_aliases(frame)
    expanded["regime_class"] = np.where(expanded["regime"] == "ID", "ID", "OOD")
    all_rows = expanded.copy()
    all_rows["regime_class"] = "All"
    expanded = pd.concat([expanded, all_rows], ignore_index=True)
    return expanded.groupby(
        ["method", "depth", "context_size", "context_measurements", "regime_class"],
        as_index=False,
    ).agg(
        n_tasks=("task", "size"),
        average_precision_mean=("average_precision", "mean"),
        transformed_mean_relative_residual_mean=(
            "mean_relative_residual",
            "mean",
        ),
        mean_relative_residual_mean=("original_mean_relative_residual", "mean"),
        covariance_relative_residual_mean=(
            "original_covariance_relative_residual",
            "mean",
        ),
        transformed_covariance_relative_residual_mean=(
            "covariance_relative_residual",
            "mean",
        ),
        relative_score_error_mean=("relative_score_error", "mean"),
        numerical_coverage_95_mean=("numerical_coverage_95", "mean"),
        raw_condition_median=("raw_condition", "median"),
        transformed_condition_median=("transformed_condition", "median"),
        condition_reduction_median=("condition_reduction", "median"),
    )


def build_tolerance_frontier(
    depth: pd.DataFrame,
    depth_runtime: pd.DataFrame,
    classical: pd.DataFrame,
    classical_runtime: pd.DataFrame,
) -> pd.DataFrame:
    """Smallest audited depth and measured time reaching each residual target."""
    learned = depth[(depth["regime_class"] == "All") & (depth["depth"] > 0)][
        ["method", "context_size", "depth", "mean_relative_residual_mean"]
    ]
    controls = classical[classical["regime_class"] == "All"][
        ["method", "context_size", "depth", "mean_relative_residual_mean"]
    ]
    curves = pd.concat([learned, controls], ignore_index=True)
    learned_times = (
        depth_runtime[depth_runtime["depth"] > 0]
        .groupby(["method", "context_size", "depth"], as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    control_times = (
        classical_runtime.groupby(
            ["method", "context_size", "depth"], as_index=False
        ).agg(inference_ms=("inference_ms", "median"))
    )
    times = pd.concat([learned_times, control_times], ignore_index=True)
    rows: list[dict[str, object]] = []
    for (method, context_size), group in curves.groupby(
        ["method", "context_size"], sort=True
    ):
        for tolerance in (0.1, 0.01, 0.001):
            reached = group[group["mean_relative_residual_mean"] <= tolerance]
            if reached.empty:
                depth_value = -1
                residual = float(group["mean_relative_residual_mean"].min())
                inference_ms = -1.0
            else:
                best = reached.loc[reached["depth"].idxmin()]
                depth_value = int(best["depth"])
                residual = float(best["mean_relative_residual_mean"])
                timing = times[
                    (times["method"] == method)
                    & (times["context_size"] == context_size)
                    & (times["depth"] == depth_value)
                ]
                inference_ms = (
                    float(timing["inference_ms"].iloc[0])
                    if not timing.empty
                    else float("nan")
                )
            rows.append(
                {
                    "method": method,
                    "context_size": int(context_size),
                    "tolerance": tolerance,
                    "reached": depth_value > 0,
                    "minimum_audited_depth": depth_value,
                    "residual_at_selected_depth": residual,
                    "inference_ms": inference_ms,
                }
            )
    return pd.DataFrame(rows)


def save_tolerance_frontier(path: Path, frontier: pd.DataFrame) -> None:
    """Plot measured time to accuracy, leaving unreached targets absent."""
    methods = (
        "context-PCG",
        "hybrid-PCG",
        "angular-Jacobi-PCG",
        "identity-CG",
        "optimized-CG",
        "looped-HB",
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    for axis, tolerance in zip(axes, (0.1, 0.01, 0.001), strict=True):
        selected = frontier[
            (frontier["tolerance"] == tolerance) & frontier["reached"]
        ]
        for method in methods:
            group = selected[selected["method"] == method].sort_values(
                "context_size"
            )
            if group.empty:
                continue
            axis.plot(
                group["context_size"],
                group["inference_ms"],
                color=CLASSICAL_COLORS[method],
                marker=CLASSICAL_MARKERS[method],
                linewidth=1.7,
                label=method,
            )
        axis.set_yscale("log")
        axis.set_xlabel("context size $m$")
        axis.set_ylabel("time to target (ms)")
        axis.set_title(f"physical residual $\\leq {tolerance:g}$")
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_classical_preconditioner_comparison(
    path: Path,
    classical: pd.DataFrame,
    classical_runtime: pd.DataFrame,
    depth: pd.DataFrame,
    depth_runtime: pd.DataFrame,
    conditioning: pd.DataFrame,
    *,
    largest_context: int,
) -> None:
    """Plot learned and training-free preconditioners at matched depth."""
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    classical_all = classical[classical["regime_class"] == "All"]
    depth_all = depth[
        (depth["regime_class"] == "All")
        & (
            depth["method"].isin(
                ("hybrid-PCG", "context-PCG", "population-PCG", "identity-CG")
            )
        )
    ]

    condition_learned = conditioning[
        (conditioning["regime_class"] == "All")
        & (
            conditioning["method"].isin(
                ("hybrid-PCG", "context-PCG", "population-PCG", "identity-CG")
            )
        )
    ][["method", "context_size", "transformed_condition_median"]]
    condition_classical = classical_all[classical_all["depth"] == 32][
        ["method", "context_size", "transformed_condition_median"]
    ]
    condition = pd.concat([condition_learned, condition_classical], ignore_index=True)
    for method in CLASSICAL_COMPARISON_ORDER:
        group = condition[condition["method"] == method].sort_values("context_size")
        if group.empty:
            continue
        axes[0, 0].plot(
            group["context_size"],
            group["transformed_condition_median"],
            color=CLASSICAL_COLORS[method],
            marker=CLASSICAL_MARKERS[method],
            linewidth=1.8,
            label=method,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("context size $m$")
    axes[0, 0].set_ylabel("median transformed condition number")
    axes[0, 0].grid(alpha=0.22)

    classical_depth = classical_all[classical_all["context_size"] == largest_context]
    learned_depth = depth_all[depth_all["context_size"] == largest_context]
    combined_depth = pd.concat([learned_depth, classical_depth], ignore_index=True)
    for axis, metric, title in (
        (
            axes[0, 1],
            "mean_relative_residual_mean",
            "physical posterior-mean residual",
        ),
        (
            axes[1, 0],
            "covariance_relative_residual_mean",
            "posterior-covariance residual",
        ),
    ):
        for method in CLASSICAL_COMPARISON_ORDER:
            group = combined_depth[combined_depth["method"] == method].sort_values(
                "depth"
            )
            if group.empty:
                continue
            axis.plot(
                group["depth"],
                group[metric],
                color=CLASSICAL_COLORS[method],
                marker=CLASSICAL_MARKERS[method],
                linewidth=1.8,
                label=method,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("operator applications / depth $T$")
        axis.set_ylabel(title)
        axis.grid(alpha=0.22)

    learned_run = (
        depth_runtime[
            depth_runtime["method"].isin(
                ("hybrid-PCG", "context-PCG", "population-PCG", "identity-CG")
            )
        ]
        .groupby(["method", "depth", "context_size"], as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    classical_run = (
        classical_runtime.groupby(
            ["method", "depth", "context_size"], as_index=False
        ).agg(inference_ms=("inference_ms", "median"))
    )
    run = pd.concat([learned_run, classical_run], ignore_index=True)
    pareto = combined_depth.merge(
        run[run["context_size"] == largest_context],
        on=["method", "depth", "context_size"],
        how="inner",
    )
    for method in CLASSICAL_COMPARISON_ORDER:
        group = pareto[pareto["method"] == method].sort_values("depth")
        if group.empty:
            continue
        axes[1, 1].plot(
            group["inference_ms"],
            group["mean_relative_residual_mean"],
            color=CLASSICAL_COLORS[method],
            marker=CLASSICAL_MARKERS[method],
            linewidth=1.8,
            label=method,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("inference time per batch (ms)")
    axes[1, 1].set_ylabel("physical posterior-mean residual")
    axes[1, 1].grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_cg_comparison_dashboard(
    path: Path,
    depth: pd.DataFrame,
    depth_runtime: pd.DataFrame,
    classical: pd.DataFrame,
    classical_runtime: pd.DataFrame,
    geometry_effects: pd.DataFrame,
    *,
    largest_context: int,
) -> None:
    """Summarize the direct matched-budget comparison against ordinary CG."""
    methods = (
        "identity-CG",
        "context-PCG",
        "hybrid-PCG",
        "angular-Jacobi-PCG",
        "looped-HB",
    )
    learned = depth[
        (depth["regime_class"] == "All")
        & (depth["context_size"] == largest_context)
        & (depth["method"].isin(methods))
        & (depth["depth"] > 0)
    ]
    analytic = classical[
        (classical["regime_class"] == "All")
        & (classical["context_size"] == largest_context)
        & (classical["method"] == "angular-Jacobi-PCG")
    ]
    curves = pd.concat([learned, analytic], ignore_index=True)

    learned_times = (
        depth_runtime[
            (depth_runtime["context_size"] == largest_context)
            & (depth_runtime["method"].isin(methods))
            & (depth_runtime["depth"] > 0)
        ]
        .groupby(["method", "depth", "context_size"], as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    analytic_times = (
        classical_runtime[
            (classical_runtime["context_size"] == largest_context)
            & (classical_runtime["method"] == "angular-Jacobi-PCG")
        ]
        .groupby(["method", "depth", "context_size"], as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    times = pd.concat([learned_times, analytic_times], ignore_index=True)
    pareto = curves.merge(
        times,
        on=["method", "depth", "context_size"],
        how="inner",
    )

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.9), constrained_layout=True)
    for method in methods:
        group = curves[curves["method"] == method].sort_values("depth")
        if group.empty:
            continue
        axes[0, 0].plot(
            group["depth"],
            group["mean_relative_residual_mean"],
            color=CLASSICAL_COLORS[method],
            marker=CLASSICAL_MARKERS[method],
            linewidth=1.8,
            label=method,
        )
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("operator applications / depth $T$")
    axes[0, 0].set_ylabel("physical posterior-mean residual")
    axes[0, 0].set_title(f"Matched iteration budget, $m={largest_context}$")
    axes[0, 0].grid(alpha=0.22)

    depth32 = curves[curves["depth"] == 32].set_index("method")
    displayed = [method for method in methods if method in depth32.index]
    x_values = np.arange(len(displayed))
    coverage = [
        float(depth32.loc[method, "numerical_coverage_95_mean"])
        for method in displayed
    ]
    axes[0, 1].bar(
        x_values,
        coverage,
        color=[CLASSICAL_COLORS[method] for method in displayed],
        width=0.72,
    )
    axes[0, 1].axhline(0.95, color="#555555", linestyle="--", linewidth=1.2)
    axes[0, 1].set_xticks(
        x_values,
        [
            {
                "identity-CG": "CG",
                "context-PCG": "context",
                "hybrid-PCG": "hybrid",
                "angular-Jacobi-PCG": "angular",
                "looped-HB": "HB",
            }[method]
            for method in displayed
        ],
        rotation=20,
        ha="right",
    )
    axes[0, 1].set_ylim(0.0, 1.04)
    axes[0, 1].set_ylabel("numerical 95% coverage")
    axes[0, 1].set_title("Bayesian UQ fidelity at $T=32$")
    axes[0, 1].grid(axis="y", alpha=0.22)

    paired = geometry_effects[
        geometry_effects["comparison"] == "CG / context-PCG"
    ].copy()
    if not paired.empty:
        scenario_order = [
            "ID four obstacles",
            "OOD half aperture",
            "OOD six obstacles",
            "OOD wavenumber 12",
            "All",
        ]
        paired["scenario"] = pd.Categorical(
            paired["scenario"], categories=scenario_order, ordered=True
        )
        paired = paired.sort_values("scenario")
        labels = [
            {
                "ID four obstacles": "ID four",
                "OOD half aperture": "half aperture",
                "OOD six obstacles": "six obstacles",
                "OOD wavenumber 12": "$k=12$",
                "All": "all",
            }[str(value)]
            for value in paired["scenario"]
        ]
        gains = paired["geometric_mean_gain"].to_numpy()
        lower = gains - paired["gain_ci95_lower"].to_numpy()
        upper = paired["gain_ci95_upper"].to_numpy() - gains
        axes[1, 0].errorbar(
            np.arange(len(paired)),
            gains,
            yerr=np.vstack([lower, upper]),
            color=COLORS["context-PCG"],
            marker="o",
            capsize=3,
            linewidth=1.5,
        )
        axes[1, 0].axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
        axes[1, 0].set_xticks(
            np.arange(len(paired)), labels, rotation=20, ha="right"
        )
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_ylabel("paired physical-residual gain CG / context-PCG")
        axes[1, 0].set_title("Independent acquisition geometries (95% CI)")
        axes[1, 0].grid(axis="y", alpha=0.22)

    for method in methods:
        group = pareto[pareto["method"] == method].sort_values("inference_ms")
        if group.empty:
            continue
        axes[1, 1].plot(
            group["inference_ms"],
            group["mean_relative_residual_mean"],
            color=CLASSICAL_COLORS[method],
            marker=CLASSICAL_MARKERS[method],
            linewidth=1.8,
            label=method,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("measured inference time per batch (ms)")
    axes[1, 1].set_ylabel("physical posterior-mean residual")
    axes[1, 1].set_title("Realized time--accuracy frontier")
    axes[1, 1].grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def format_number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def final_comparison(
    aggregate: pd.DataFrame, runtime: pd.DataFrame, width: int, dataset_size: int
) -> pd.DataFrame:
    selected = aggregate[
        (aggregate["regime_class"] == "All")
        & (aggregate["context_size"] == 24)
        & (
            (
                (aggregate["network_width"] == width)
                & (aggregate["dataset_size"] == dataset_size)
            )
            | (aggregate["network_width"] == 0)
        )
    ].copy()
    run = (
        runtime[
            (runtime["context_size"] == 24)
            & ((runtime["network_width"] == width) | (runtime["network_width"] == 0))
        ]
        .groupby("method", as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    return selected.merge(run, on="method", how="left").sort_values(
        "method",
        key=lambda column: column.map(
            {name: index for index, name in enumerate(METHOD_ORDER)}
        ),
    )


def scenario_cg_comparison(
    frame: pd.DataFrame,
    width: int,
    dataset_size: int,
    context_size: int,
) -> pd.DataFrame:
    """Compare CG and the prompt-conditioned factor on every held-out scenario."""
    selected = frame[
        (frame["context_size"] == context_size)
        & (
            (frame["method"] == "identity-CG")
            | (
                (frame["method"] == "context-PCG")
                & (frame["network_width"] == width)
                & (frame["dataset_size"] == dataset_size)
            )
        )
    ]
    grouped = (
        selected.groupby(["scenario", "method"], as_index=False)
        .agg(
            average_precision=("average_precision", "mean"),
            mean_relative_residual=("mean_relative_residual", "mean"),
            covariance_relative_residual=("covariance_relative_residual", "mean"),
        )
        .pivot(index="scenario", columns="method")
    )
    required = {"identity-CG", "context-PCG"}
    available = set(grouped.columns.get_level_values("method"))
    if not required.issubset(available):
        return pd.DataFrame()
    rows = []
    for scenario in grouped.index:
        cg_mean = float(
            grouped.loc[scenario, ("mean_relative_residual", "identity-CG")]
        )
        context_mean = float(
            grouped.loc[scenario, ("mean_relative_residual", "context-PCG")]
        )
        cg_covariance = float(
            grouped.loc[
                scenario,
                ("covariance_relative_residual", "identity-CG"),
            ]
        )
        context_covariance = float(
            grouped.loc[
                scenario,
                ("covariance_relative_residual", "context-PCG"),
            ]
        )
        rows.append(
            {
                "scenario": scenario,
                "cg_average_precision": float(
                    grouped.loc[scenario, ("average_precision", "identity-CG")]
                ),
                "context_average_precision": float(
                    grouped.loc[scenario, ("average_precision", "context-PCG")]
                ),
                "cg_transformed_mean_residual": cg_mean,
                "context_transformed_mean_residual": context_mean,
                "transformed_mean_residual_gain": cg_mean / context_mean,
                "transformed_covariance_residual_gain": (
                    cg_covariance / context_covariance
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario")


def cg_gain_by_context(
    aggregate: pd.DataFrame,
    runtime: pd.DataFrame,
    width: int,
    dataset_size: int,
) -> pd.DataFrame:
    """Make the matched-depth CG comparison explicit at every context."""
    metric_columns = [
        "average_precision_mean",
        "mean_relative_residual_mean",
        "covariance_relative_residual_mean",
        "numerical_coverage_95_mean",
    ]
    selected = aggregate[aggregate["regime_class"] == "All"]
    reference = selected[selected["method"] == "identity-CG"][
        ["context_size", *metric_columns]
    ]
    candidates = selected[
        (selected["method"].isin(LEARNED_ORDER))
        & (selected["network_width"] == width)
        & (selected["dataset_size"] == dataset_size)
    ][["method", "context_size", *metric_columns]]
    comparison = candidates.merge(
        reference,
        on="context_size",
        suffixes=("_candidate", "_cg"),
    )
    comparison["mean_residual_gain"] = (
        comparison["mean_relative_residual_mean_cg"]
        / comparison["mean_relative_residual_mean_candidate"]
    )
    comparison["covariance_residual_gain"] = (
        comparison["covariance_relative_residual_mean_cg"]
        / comparison["covariance_relative_residual_mean_candidate"]
    )
    comparison["average_precision_delta"] = (
        comparison["average_precision_mean_candidate"]
        - comparison["average_precision_mean_cg"]
    )
    run = (
        runtime[
            ((runtime["network_width"] == width) & runtime["method"].isin(LEARNED_ORDER))
            | ((runtime["network_width"] == 0) & (runtime["method"] == "identity-CG"))
        ]
        .groupby(["method", "context_size"], as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    candidate_runtime = run[run["method"] != "identity-CG"]
    cg_runtime = run[run["method"] == "identity-CG"][["context_size", "inference_ms"]]
    timing = candidate_runtime.merge(
        cg_runtime,
        on="context_size",
        suffixes=("_candidate", "_cg"),
    )
    timing["runtime_ratio"] = (
        timing["inference_ms_candidate"] / timing["inference_ms_cg"]
    )
    return comparison.merge(
        timing[["method", "context_size", "runtime_ratio"]],
        on=["method", "context_size"],
        how="left",
    ).sort_values(["method", "context_size"])


def cg_gain_by_depth(
    depth_aggregate: pd.DataFrame,
    depth_runtime: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every recurrent method with CG at equal depth and context."""
    selected = depth_aggregate[
        (depth_aggregate["regime_class"] == "All")
        & (depth_aggregate["depth"] > 0)
    ]
    metrics = [
        "average_precision_mean",
        "mean_relative_residual_mean",
        "covariance_relative_residual_mean",
    ]
    reference = selected[selected["method"] == "identity-CG"][
        ["depth", "context_size", *metrics]
    ]
    candidates = selected[
        ~selected["method"].isin(("identity-CG", "exact"))
    ][["method", "depth", "context_size", *metrics]]
    comparison = candidates.merge(
        reference,
        on=["depth", "context_size"],
        suffixes=("_candidate", "_cg"),
    )
    comparison["mean_residual_gain"] = (
        comparison["mean_relative_residual_mean_cg"]
        / comparison["mean_relative_residual_mean_candidate"]
    )
    comparison["covariance_residual_gain"] = (
        comparison["covariance_relative_residual_mean_cg"]
        / comparison["covariance_relative_residual_mean_candidate"]
    )
    comparison["average_precision_delta"] = (
        comparison["average_precision_mean_candidate"]
        - comparison["average_precision_mean_cg"]
    )
    run = (
        depth_runtime[depth_runtime["depth"] > 0]
        .groupby(["method", "depth", "context_size"], as_index=False)
        .agg(inference_ms=("inference_ms", "median"))
    )
    candidate_runtime = run[run["method"] != "identity-CG"]
    cg_runtime = run[run["method"] == "identity-CG"][
        ["depth", "context_size", "inference_ms"]
    ]
    timing = candidate_runtime.merge(
        cg_runtime,
        on=["depth", "context_size"],
        suffixes=("_candidate", "_cg"),
    )
    timing["runtime_ratio"] = (
        timing["inference_ms_candidate"] / timing["inference_ms_cg"]
    )
    return comparison.merge(
        timing[["method", "depth", "context_size", "runtime_ratio"]],
        on=["method", "depth", "context_size"],
        how="left",
    ).sort_values(["method", "context_size", "depth"])


def build_tex(
    results_dir: Path,
    protocol: dict,
    comparison: pd.DataFrame,
    scenario_comparison: pd.DataFrame,
    fits: pd.DataFrame,
    bounds: pd.DataFrame,
    summary: dict[str, object],
) -> Path:
    comparison_rows = []
    for row in comparison.itertuples():
        comparison_rows.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row.method),
                format_number(row.average_precision_mean),
                format_number(row.mean_relative_residual_mean),
                format_number(row.covariance_relative_residual_mean),
                format_number(row.relative_score_error_mean),
                format_number(row.numerical_coverage_95_mean),
                format_number(row.inference_ms),
            )
        )
    scenario_comparison_rows = []
    for row in scenario_comparison.itertuples():
        scenario_comparison_rows.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row.scenario),
                format_number(row.cg_average_precision),
                format_number(row.context_average_precision),
                format_number(row.cg_transformed_mean_residual),
                format_number(row.context_transformed_mean_residual),
                format_number(row.transformed_mean_residual_gain),
                format_number(row.transformed_covariance_residual_gain),
            )
        )
    fit_rows = []
    for row in fits.itertuples():
        fit_rows.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row.method),
                latex_escape(row.risk_type),
                format_number(row.data_exponent),
                format_number(row.context_exponent),
                format_number(row.width_exponent),
                format_number(row.r_squared),
            )
        )
    scenario_rows = []
    for scenario in protocol["scenarios"]:
        scenario_rows.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(str(scenario["name"])),
                int(scenario["count"]),
                latex_escape(str(scenario["mode"])),
                format_number(float(scenario["wavenumber"]), digits=1),
                format_number(float(scenario["noise"]), digits=2),
                format_number(float(scenario["aperture"]), digits=0),
            )
        )
    trained_models = int(summary["trained_model_runs"])
    dataset_checkpoints = int(summary["completed_dataset_checkpoints"])
    total_tasks = int(summary["learned_task_rows"])
    depth_figure = ""
    depth_interpretation = ""
    conditioning_figure = ""
    conditioning_interpretation = ""
    classical_figure = ""
    classical_interpretation = ""
    tolerance_figure = ""
    cg_comparison_figure = ""
    stress_figure = ""
    joint_conditioning_figure = ""
    joint_reconstruction_figure = ""
    context_pcg_interpretation = ""
    hybrid_interpretation = ""
    reconstruction_figure = ""
    geometry_figure = ""
    geometry_certificate = (
        "The certificate is conditional on the pre-drawn held-out acquisition "
        "geometries.  A geometry-distribution certificate requires independent "
        "acquisition batches and is not inferred by treating tasks from one "
        "geometry as independent geometry draws."
    )
    if int(summary.get("geometry_generalization_rows", 0)) > 0:
        geometry_figure = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{scaling_geometry_generalization.png}
\caption{Independent-acquisition generalization.  Each bounded observation is
the mean of four fresh obstacle/noise tasks sharing one independently rotated
and jittered deterministic source--receiver geometry.  Dashed curves are
simultaneous 95\% Hoeffding upper bounds.}
\label{fig:geometry-bound}
\end{figure}
"""
        geometry_certificate = rf"""
We additionally draw independent acquisition batches rather than relabel tasks
from one geometry.  Each bounded observation averages four fresh obstacle and
noise realizations at one newly rotated/jittered deterministic point-source
array.  Every per-scenario configuration has at least
{int(summary["geometry_bound_min_batches"])} independent geometry batches over
the three frozen training seeds.  Applying the same union-bound argument to
these batch means yields Figure~\ref{{fig:geometry-bound}}; its largest slack is
{format_number(float(summary["geometry_bound_max_slack"]))}.  This remains a
simulator-distribution certificate, not a claim about the random-source LSM.
At $m={summary["largest_context"]}$, context-PCG reduces the geometry-batch
mean residual by
{format_number(float(summary["geometry_cg_to_context_mean_gain"]))}$\times$
relative to CG, with a worst scenario-wise gain of
{format_number(float(summary["geometry_context_worst_scenario_gain"]))}$\times$.
Its numerical 95\% band coverage is
{format_number(float(summary["geometry_context_coverage"]))}, versus
{format_number(float(summary["geometry_cg_coverage"]))} for CG.
The paired AP difference is only
{format_number(float(summary["geometry_paired_context_ap_delta"]))}
(95\% interval
[{format_number(float(summary["geometry_paired_context_ap_delta_ci_lower"]))},
{format_number(float(summary["geometry_paired_context_ap_delta_ci_upper"]))}]),
whereas the paired numerical-coverage increase is
{format_number(float(summary["geometry_paired_context_coverage_delta"]))}
([{format_number(float(summary["geometry_paired_context_coverage_delta_ci_lower"]))},
{format_number(float(summary["geometry_paired_context_coverage_delta_ci_upper"]))}]).
Thus the large numerical gain mainly restores solver and UQ fidelity rather
than changing the physical localization ranking.
All {int(summary["geometry_seed_scenario_count"])} seed/scenario cells retain
the advantage, with minimum gain
{format_number(float(summary["geometry_worst_seed_scenario_gain"]))}$\times$.
On the same independent geometries, angular-Jacobi, hybrid-PCG and context-PCG
have mean residuals
{format_number(float(summary["geometry_angular_mean_residual"]))},
{format_number(float(summary["geometry_hybrid_mean_residual"]))}, and
{format_number(float(summary["geometry_context_mean_residual"]))}, respectively.
The angular-to-context ratio is
{format_number(float(summary["geometry_angular_to_context_mean_ratio"]))}, with
worst scenario ratio
{format_number(float(summary["geometry_angular_to_context_worst_scenario_ratio"]))}.
Their angular and hybrid numerical coverages are
{format_number(float(summary["geometry_angular_coverage"]))} and
{format_number(float(summary["geometry_hybrid_coverage"]))}.
Pairing methods on each independent geometry batch gives an angular-to-context
geometric-mean gain of
{format_number(float(summary["geometry_paired_angular_to_context_gain"]))}, with
95\% normal-approximation log-ratio interval
[{format_number(float(summary["geometry_paired_angular_to_context_gain_ci_lower"]))},
{format_number(float(summary["geometry_paired_angular_to_context_gain_ci_upper"]))}].
Context-PCG wins
{format_number(float(summary["geometry_paired_context_win_rate_vs_angular"]))}
of paired batches (Wilson 95\% interval
[{format_number(float(summary["geometry_paired_context_win_rate_ci_lower"]))},
{format_number(float(summary["geometry_paired_context_win_rate_ci_upper"]))}]).
"""
    if (results_dir / "scaling_reconstructions.png").exists():
        reconstruction_figure = r"""
\begin{figure}[p]
\centering
\includegraphics[width=0.94\linewidth,height=0.84\textheight,keepaspectratio]{scaling_reconstructions.png}
\caption{Pre-specified task zero at extrapolative context $m=48$; no example
was selected by performance.  White contours show the true obstacles.  The
rows compare CG, analytic angular-Jacobi PCG, the learned factors, and the
analytic--learned hybrid.  The last row displays context-PCG posterior standard
deviation rather than a reconstruction score.}
\label{fig:reconstructions}
\end{figure}
"""
    if (results_dir / "scaling_cg_comparison.png").exists():
        cg_comparison_figure = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{scaling_cg_comparison.png}
\caption{Direct comparison with ordinary CG.  Top left: matched operator
applications.  Top right: numerical Bayesian coverage relative to the exact
solve.  Bottom left: paired physical-residual gains on independent acquisition
geometries.  Bottom right: measured time--accuracy paths.  Values above one in
the gain panel favor context-PCG.}
\label{fig:cg-comparison}
\end{figure}
"""
    if int(summary.get("cg_stress_rows", 0)) > 0:
        stress_figure = rf"""
\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{scaling_cg_stress_test.png}}
\caption{{Marginal and joint physical stress sweeps at
$m={summary["largest_context"]}$ and $T=32$.  Curves are paired geometric-mean
physical-coordinate residual gains over at least
{int(summary["cg_stress_batches_per_level_min"])} independent acquisition
batches per level; shaded regions are pointwise 95\% normal-approximation
log-ratio intervals.  Values above
one favor the preconditioned method over ordinary CG.}}
\label{{fig:cg-stress}}
\end{{figure}}

Across the {int(summary["cg_stress_level_count"])} stress levels, the
context-PCG gain over CG ranges from
{format_number(float(summary["cg_stress_min_context_gain"]))}$\times$ to
{format_number(float(summary["cg_stress_max_context_gain"]))}$\times$; its
smallest paired-batch win rate at any level is
{format_number(float(summary["cg_stress_min_context_win_rate"]))}.  The first
sixteen levels vary one factor; the four joint-severity levels co-vary obstacle
count, noise, aperture, frequency and acquisition jitter.  These remain
simulator perturbations, not a certificate for arbitrary distribution shift.
Context-PCG has the smallest residual in
{int(summary["cg_stress_context_best_count"])} of the
{int(summary["cg_stress_level_count"])} levels, versus
{int(summary["cg_stress_angular_best_count"])} for angular-Jacobi,
{int(summary["cg_stress_population_best_count"])} for population-PCG and
{int(summary["cg_stress_hybrid_best_count"])} for the hybrid, while looped HB
wins {int(summary["cg_stress_hb_best_count"])}.  Robustly beating CG therefore
does not imply dominance over every informed preconditioner.
Across all stress batches, context-PCG's geometric-mean residual gain is
{format_number(float(summary["cg_stress_context_global_gain"]))}$\times$, with
AP and numerical-coverage changes
{format_number(float(summary["cg_stress_context_ap_delta"]))} and
{format_number(float(summary["cg_stress_context_coverage_delta"]))} relative
to CG.  Looped HB attains a residual gain of
{format_number(float(summary["cg_stress_hb_global_gain"]))}$\times$ but changes
AP by {format_number(float(summary["cg_stress_hb_ap_delta"]))} and coverage by
{format_number(float(summary["cg_stress_hb_coverage_delta"]))}.  Its occasional
short-horizon residual win is therefore not a reconstruction or UQ win.
At the most extreme joint shift, context-PCG retains a residual gain of
{format_number(float(summary["joint_extreme_context_gain"]))}$\times$
(95\% interval
[{format_number(float(summary["joint_extreme_context_gain_ci_lower"]))},
{format_number(float(summary["joint_extreme_context_gain_ci_upper"]))}]) and
wins {format_number(float(summary["joint_extreme_context_win_rate"]))} of
paired batches (Wilson 95\% interval
[{format_number(float(summary["joint_extreme_context_win_rate_ci_lower"]))},
{format_number(float(summary["joint_extreme_context_win_rate_ci_upper"]))}]).
Angular-Jacobi retains
{format_number(float(summary["joint_extreme_angular_gain"]))}$\times$, whereas
population-PCG and looped HB fall to
{format_number(float(summary["joint_extreme_population_gain"]))}$\times$ and
{format_number(float(summary["joint_extreme_hb_gain"]))}$\times$.  The
context-PCG coverage gain is nevertheless
{format_number(float(summary["joint_extreme_context_coverage_delta"]))}:
this is residual robustness, not universal UQ calibration.
"""
    if int(summary.get("joint_conditioning_rows", 0)) > 0:
        joint_conditioning_figure = rf"""
\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{scaling_joint_conditioning.png}}
\caption{{Spectral mechanism across the four joint-shift severities.  Each
point aggregates independent acquisition batches; the right panel links
condition compression to the observed finite-depth physical residual.}}
\label{{fig:joint-conditioning}}
\end{{figure}}

At the extreme joint shift, the raw median condition number is
{format_number(float(summary["joint_extreme_raw_condition"]))}.  Context-PCG
reduces it by
{format_number(float(summary["joint_extreme_context_condition_gain"]))}$\times$
to {format_number(float(summary["joint_extreme_context_condition"]))}, whereas
angular-Jacobi reduces it by
{format_number(float(summary["joint_extreme_angular_condition_gain"]))}$\times$
to {format_number(float(summary["joint_extreme_angular_condition"]))}.  The
commutator is {format_number(float(summary["joint_extreme_commutator"]))}.
Context and angular-Jacobi therefore have nearly identical scalar condition
numbers but residual gains
{format_number(float(summary["joint_extreme_context_gain"]))}$\times$ and
{format_number(float(summary["joint_extreme_angular_gain"]))}$\times$.
Condition number alone does not rank the finite-horizon solve; spectral
clustering and right-hand-side alignment matter, while the nonzero commutator
still limits both fixed-basis factors.
"""
    if (results_dir / "scaling_joint_reconstructions.png").exists():
        joint_reconstruction_figure = r"""
\begin{figure}[p]
\centering
\includegraphics[width=0.94\linewidth,height=0.88\textheight,keepaspectratio]{scaling_joint_reconstructions.png}
\caption{Pre-specified reconstructions along the joint-shift path (seed 17,
geometry draw zero, task zero); no task was selected by performance.  White
contours show the true obstacles.  Each score map is normalized by its own
1st--99th percentile range to compare localization shape; AP and physical
residual are reported separately.  Context-PCG preserves the exact LSM ranking
while lowering the finite-depth residual, whereas looped HB has poor ranking
despite some residual improvement.  The last row is context-PCG posterior
standard deviation, not a reconstruction score.}
\label{fig:joint-reconstructions}
\end{figure}
"""
    if np.isfinite(float(summary.get("cg_to_context_pcg_mean_residual_gain", np.nan))):
        context_pcg_interpretation = rf"""
At the same central configuration, prompt-conditioned PCG improves ordinary
CG in transformed-coordinate residual by
{format_number(float(summary["cg_to_context_pcg_mean_residual_gain"]))}$\times$
on the posterior mean and
{format_number(float(summary["cg_to_context_pcg_covariance_residual_gain"]))}$\times$
on posterior covariance.  Across all contexts its corresponding gains are
{format_number(float(summary["macro_cg_to_context_pcg_mean_residual_gain"]))}$\times$
and
{format_number(float(summary["macro_cg_to_context_pcg_covariance_residual_gain"]))}$\times$.
This is a diagnostic of whether the current operator carries useful in-context
signal within the prescribed optimization coordinates; it is not the direct
physical CG claim.  The post-training $r_{{\rm phys}}$ audit below provides the
coordinate-invariant test.
Within the shared model implementation, the price is controller overhead:
context-PCG costs
{format_number(float(summary["central_context_runtime_ratio"]))}$\times$ the
CG batch time at $m=24$ and
{format_number(float(summary["largest_context_runtime_ratio"]))}$\times$ at
$m={summary["largest_context"]}$.  It is therefore iteration-efficient, not a
large wall-clock speedup at fixed depth.  After removing the unused endpoint
controller from PCG inference, the iterative methods are faster than dense
factorization on the tested batches.  The optimized classical controls below
provide the stricter timing comparison.
"""
    if np.isfinite(float(summary.get("cg_to_hybrid_pcg_mean_residual_gain", np.nan))):
        hybrid_interpretation = rf"""
We also test an analytic--learned hybrid: the current Hessian is rotated into
the prescribed GP angular feature basis, its diagonal supplies a training-free
positive preconditioner, and a zero-initialized context network learns only a
multiplicative residual correction.  At the central configuration it improves
CG in transformed-coordinate residual by
{format_number(float(summary["cg_to_hybrid_pcg_mean_residual_gain"]))}$\times$
for the mean and
{format_number(float(summary["cg_to_hybrid_pcg_covariance_residual_gain"]))}$\times$
for covariance.  Across all contexts the corresponding gains are
{format_number(float(summary["macro_cg_to_hybrid_pcg_mean_residual_gain"]))}$\times$
and
{format_number(float(summary["macro_cg_to_hybrid_pcg_covariance_residual_gain"]))}$\times$.
At $m={summary["largest_context"]}$ and $T=32$, however, its physical mean residual is
{format_number(float(summary["largest_context_hybrid_pcg_mean_residual"]))},
whereas context-PCG attains
{format_number(float(summary["largest_context_context_pcg_mean_residual"]))}
and analytic angular-Jacobi attains
{format_number(float(summary["classical_angular_mean_residual"]))}.
The angular-to-hybrid residual ratio is
{format_number(float(summary["classical_angular_to_hybrid_residual_ratio"]))};
at the common central width this tests whether the correction improves the
analytic base without extra capacity.  Across the width sweep, the best hybrid
uses width {int(summary["hybrid_best_width"])} and attains transformed residual
{format_number(float(summary["hybrid_best_width_mean_residual"]))}, covariance
residual
{format_number(float(summary["hybrid_best_width_covariance_residual"]))}, and
coverage {format_number(float(summary["hybrid_best_width_coverage"]))}.
These width-sweep residuals are in transformed coordinates and are therefore
not divided by the physical residual of the analytic control; their homogeneous
physical comparison is the matched-width, matched-depth audit above.
This control is the criterion for deciding whether preconditioning needs to be
learned rather than merely computed in context.
"""
    if int(summary.get("depth_audit_task_rows", 0)) > 0:
        depth_figure = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{scaling_depth.png}
\caption{Post-training depth scaling on shared held-out tasks.  Varying $T$
changes only the number of operator applications; parameters and acquisition
data are fixed.  The lower-right panel reports the realized depth--time Pareto
curve rather than assuming ideal linear timing.}
\label{fig:depth}
\end{figure}
"""
        depth_interpretation = rf"""
The post-training depth audit contains
{int(summary["depth_audit_task_rows"]):,} task-level evaluations over
$T\in{summary["depth_values"]}$.  At the extrapolative context
$m={summary["largest_context"]}$, the CG-to-population-PCG physical
posterior-mean residual gain is
{format_number(float(summary["depth32_largest_context_mean_gain"]))}$\times$
at $T=32$ and
{format_number(float(summary["depth64_largest_context_mean_gain"]))}$\times$
at $T=64$; the corresponding prompt-conditioned gains are
{format_number(float(summary["depth32_largest_context_context_gain"]))}$\times$
and
{format_number(float(summary["depth64_largest_context_context_gain"]))}$\times$.
At the same two depths, context-PCG improves looped HB by
{format_number(float(summary["depth32_context_vs_hb_gain"]))}$\times$ and
{format_number(float(summary["depth64_context_vs_hb_gain"]))}$\times$,
respectively; HB has a competitive short-horizon transient but its stationary
recurrence does not retain the long-depth Krylov convergence.
At $m={summary["largest_context"]}$ and $T=32$, looped HB has mean residual
{format_number(float(summary["depth32_largest_context_hb_mean_residual"]))},
average precision
{format_number(float(summary["depth32_largest_context_hb_average_precision"]))},
and numerical 95\% coverage
{format_number(float(summary["depth32_largest_context_hb_coverage"]))}; its
batch runtime is
{format_number(float(summary["depth32_largest_context_hb_runtime"]))} ms.
Its relative LSM-score error is
{format_number(float(summary["depth32_largest_context_hb_score_error"]))},
versus {format_number(float(summary["depth32_largest_context_cg_score_error"]))}
for CG and
{format_number(float(summary["depth32_largest_context_context_score_error"]))}
for context-PCG.  In an ill-conditioned system, a smaller Euclidean residual
can still leave larger solution error along small-eigenvalue directions; this
explains why HB's residual transient does not produce a good reconstruction.
Thus its residual improvement over CG does not translate into faithful LSM
localization or calibrated Bayesian uncertainty, and its dense learned update
is substantially more expensive than Krylov PCG.
At $T=128$, the equal-depth CG-to-context-PCG gain reaches
{format_number(float(summary["depth128_largest_context_context_gain"]))}$\times$;
this ratio is taken near the numerical convergence floor and is not a claim of
improved physical localization.
For a tolerance-oriented comparison at $m={summary["largest_context"]}$,
context-PCG first reaches mean residual $0.1$ at $T=
{int(summary["context_depth_for_residual_01"])}$, versus $T=
{int(summary["cg_depth_for_residual_01"])}$ for CG; for residual $0.01$ the
depths are {int(summary["context_depth_for_residual_001"])} and
{int(summary["cg_depth_for_residual_001"])}.
At the $0.01$ target, however, the measured batch times are
{format_number(float(summary["context_time_for_residual_001"]))} ms and
{format_number(float(summary["cg_time_for_residual_001"]))} ms, respectively:
the optimized implementation preserves a modest wall-clock benefit from the
iteration-count saving at this matrix size.
Figure~\ref{{fig:depth}} therefore distinguishes a genuine
conditioning advantage from merely granting the learned method more
iterations.  The fact that depth remains useful at large $m$ is qualitatively
consistent with the randomly rotated structured-covariance regime of
Bordelon et al.; it is not an assertion that their linear-model asymptotics
govern this inverse-scattering system.
"""
    if int(summary.get("conditioning_audit_task_rows", 0)) > 0:
        conditioning_figure = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{scaling_conditioning.png}
\caption{Spectral conditioning audit.  The normalized commutator tests the
alignment of the fixed angular-kernel basis with the obstacle-dependent
near-field Hessian; values away from zero expose a fixed-basis bottleneck.}
\label{fig:conditioning}
\end{figure}
"""
        conditioning_interpretation = rf"""
At $m={summary["largest_context"]}$, the median condition-number reduction is
{format_number(float(summary["largest_context_population_condition_gain"]))}$\times$
for population-PCG and
{format_number(float(summary["largest_context_context_condition_gain"]))}$\times$
for context-PCG.  Figure~\ref{{fig:conditioning}} directly checks whether
smaller residuals track spectral compression and whether the obstacle-dependent
operator commutes with the fixed GP geometry.
The analytic--learned hybrid reduces the condition number by
{format_number(float(summary["largest_context_hybrid_condition_gain"]))}$\times$
and leaves median condition number
{format_number(float(summary["largest_context_hybrid_condition"]))}.
The raw median condition number is still
{format_number(float(summary["largest_context_raw_condition"]))}, and
context-PCG leaves
{format_number(float(summary["largest_context_context_condition"]))}; the
normalized commutator
{format_number(float(summary["largest_context_geometry_commutator"]))}
therefore identifies the remaining inability of fixed modal gains to rotate
task-dependent eigendirections.
"""
    if int(summary.get("classical_preconditioner_task_rows", 0)) > 0:
        classical_figure = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{scaling_classical_preconditioners.png}
\caption{Training-free PCG controls at matched depth.  Jacobi uses the physical
diagonal, block-Jacobi uses contiguous angular blocks of size four, and
angular-Jacobi uses the diagonal of the current Hessian in the fixed GP
angular basis.}
\label{fig:classical-pcg}
\end{figure}
"""
        classical_interpretation = rf"""
The training-free angular-Jacobi control is the critical test of whether the
available in-context signal actually needs to be learned.  At
$m={summary["largest_context"]}$ and $T=32$, its physical posterior-mean residual is
{format_number(float(summary["classical_angular_mean_residual"]))}, versus
{format_number(float(summary["largest_context_context_pcg_mean_residual"]))}
for context-PCG, a classical-to-learned ratio of
{format_number(float(summary["classical_angular_to_context_residual_ratio"]))}.
At $T=96$ the ratio is instead
{format_number(float(summary["classical_angular_to_context_depth96_residual_ratio"]))},
so the analytical control overtakes the learned finite-horizon factor at high
accuracy.
Their numerical 95\% band coverages at $T=32$ are
{format_number(float(summary["classical_angular_coverage"]))} and
{format_number(float(summary["context_depth32_coverage"]))}, respectively.
Its transformed condition number is
{format_number(float(summary["classical_angular_condition"]))} and its runtime
is {format_number(float(summary["classical_angular_runtime"]))} ms, or
{format_number(float(summary["classical_angular_to_context_runtime_ratio"]))}$\times$
the learned method.  Removing unused feature/controller calculations from CG
also reduces its $T=32$ runtime to
{format_number(float(summary["classical_optimized_cg_runtime"]))} ms; context-PCG
is then
{format_number(float(summary["context_to_optimized_cg_runtime_ratio"]))}$\times$
slower.  At residual target $0.01$, angular-Jacobi reaches the target at
$T={int(summary["classical_angular_depth_for_residual_001"])}$ in
{format_number(float(summary["classical_angular_time_for_residual_001"]))} ms;
optimized CG uses $T=
{int(summary["classical_optimized_cg_depth_for_residual_001"])}$ and
{format_number(float(summary["classical_optimized_cg_time_for_residual_001"]))}
ms, while context-PCG uses
{format_number(float(summary["context_time_for_residual_001"]))} ms.
Across all {int(summary["tolerance_frontier_count"])} context/target cells in
Figure~\ref{{fig:tolerance-frontier}}, angular-Jacobi is fastest in
{int(summary["tolerance_wins_angular"])}, hybrid-PCG in
{int(summary["tolerance_wins_hybrid"])}, population-PCG in
{int(summary["tolerance_wins_population"])}, identity-CG in
{int(summary["tolerance_wins_identity"])}, optimized CG in
{int(summary["tolerance_wins_optimized_cg"])}, block-Jacobi in
{int(summary["tolerance_wins_block_jacobi"])} and context-PCG in
{int(summary["tolerance_wins_context"])}.  Hence context-PCG's equal-depth
advantage does not become a strict wall-clock win against every optimized
operator-aware baseline.
Thus the experiment supports learning the conditioner
only to the extent that it improves or amortizes this cheap operator-aware
baseline; diagonal and local block-Jacobi in the physical ordering do not
extract the same angular GP structure.
"""
    if (results_dir / "scaling_tolerance_frontier.png").exists():
        tolerance_figure = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{scaling_tolerance_frontier.png}
\caption{Measured time to residual targets on the audited depth grid.  A curve
is absent where a method does not reach the target by $T=128$.  Equal-depth and
time-to-tolerance comparisons need not rank methods identically.}
\label{fig:tolerance-frontier}
\end{figure}
"""
    results_interpretation = rf"""
The equal-budget transformed-coordinate residual ratios at the final central
configuration are
{format_number(float(summary["cg_to_pcg_mean_residual_gain"]))}$\times$ for
the posterior mean and
{format_number(float(summary["cg_to_pcg_covariance_residual_gain"]))}$\times$
for posterior covariance.  Ratios above one mean that the learned factor lowers
the transformed-coordinate optimization metric.  Since the coordinate factor
differs by method, these ratios are not interpreted as physical solver gains;
the common-coordinate audit below supplies that comparison.  Localization can
nevertheless saturate before residual convergence; in that regime additional
solver accuracy does not overcome the physical point-spread limit.
Across all six contexts, the corresponding gains are
{format_number(float(summary["macro_cg_to_pcg_mean_residual_gain"]))}$\times$
and
{format_number(float(summary["macro_cg_to_pcg_covariance_residual_gain"]))}$\times$.
At the extrapolative context $m={summary["largest_context"]}$ they reach
{format_number(float(summary["largest_context_mean_residual_gain"]))}$\times$
and
{format_number(float(summary["largest_context_covariance_residual_gain"]))}$\times$,
respectively.  Thus the transformed training-metric separation grows with
context, although covariance separation is not uniform.  Whether this survives
in the physical coordinate is evaluated separately below.

{context_pcg_interpretation}

{hybrid_interpretation}

{depth_interpretation}

{conditioning_interpretation}

{classical_interpretation}

The fitted powers must also be read diagnostically rather than as universal
laws.  For population-PCG the localization fit has $R^2=
{format_number(float(summary["pcg_localization_fit_r2"]))}$, whereas the
solver-risk fit has $R^2=
{format_number(float(summary["pcg_solver_fit_r2"]))}$.  A low or negative
$R^2$ rejects the proposed monotone separable power-law description on this
finite grid; it is evidence of non-monotone optimization or a performance
plateau, not a scaling exponent to extrapolate.  Exponents reported as 3.000
hit the fit's upper constraint and must be read as boundary estimates, not
identified power laws.
"""
    text = rf"""
\documentclass[10pt]{{article}}
\usepackage[margin=0.76in]{{geometry}}
\usepackage{{amsmath,amssymb,amsthm,bm,booktabs,graphicx,microtype,placeins}}
\usepackage{{hyperref,xcolor}}
\hypersetup{{
  hidelinks,
  pdftitle={{Scaling Laws for Softmax-Kernel In-Context Bayesian Near-Field Linear Sampling}},
  pdfauthor={{Janis and Hancya}},
  pdfsubject={{In-context preconditioning and Bayesian UQ for near-field LSM}},
  pdfkeywords={{linear sampling method, in-context learning, PCG, Bayesian UQ, scaling laws}}
}}
\newtheorem{{theorem}}{{Theorem}}
\newtheorem{{remark}}{{Remark}}
\DeclareMathOperator{{\softmax}}{{softmax}}
\title{{Scaling Laws for Softmax-Kernel In-Context Bayesian Near-Field Linear Sampling}}
\author{{Janis and Hancya}}
\date{{5 August 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We study how an in-context posterior-moment solver for the original
two-dimensional near-field linear sampling method scales with the number $n$ of
training inverse problems, network size $P$, context length $m$, recurrent
depth, and wall-clock time.  The input is a complex multistatic response from
deterministic point sources around one to six sound-soft obstacles.  A fixed
von Mises/softmax covariance selects the Gaussian-process feature space; its
nonlinearity is a modelling choice and is not learned.  Richardson, heavy-ball
(HB), Chebyshev, population-PCG and context-PCG share the same Bayesian moment
decoder and solve posterior mean and covariance right-hand sides in parallel.
Context-PCG adds a prompt-conditioned SPD factor that is frozen during each
solve.  A hybrid starts from analytic angular-Jacobi and learns only a bounded
residual log-gain.  Identity-preconditioned CG is the
matched-matrix-vector-product control.  The sweep
is supplemented by training-free physical Jacobi, angular block-Jacobi and
GP-basis angular-Jacobi PCG controls.  It
contains {trained_models} trained models, {dataset_checkpoints} dataset
checkpoints and {total_tasks:,} learned-model/task evaluations.  We report
task-level uncertainty, inference time,
empirical scaling exponents and a simultaneous distribution-free held-out
risk bound.  At $m=48$ and $T=32$, context-PCG improves the common-coordinate
physical residual by $2.05\times$, reduces the relative LSM-score error by
$6.44\times$, and raises numerical 95\% posterior-band coverage from 0.243 to
0.873.  Across 768 independent acquisition batches its paired geometric-mean
residual gain is $2.24\times$ and it wins 96.4\% of pairs, although analytic
angular-Jacobi PCG remains faster on this small dense problem.  The paired AP
increase is only 0.0021, separating numerical fidelity from physical
localization resolution.  The main
question is not whether a large controller can imitate
CG, but when learned conditioning improves CG under severe, task-dependent
ill-conditioning.
\end{{abstract}}

\section{{Problem and posterior moments}}
Let $N_D\in\mathbb C^{{m\times m}}$ be the whitened near-field matrix for a
sound-soft obstacle $D$, $\Phi=[\phi_z]_z$ the Green-function probes on a
two-dimensional sampling grid, and $K_\Gamma\succ0$ a prescribed angular GP
covariance.  The nonlinear feature choice is
\[
 W^\rho_{{ij}}=\exp\!\left\{{\rho(\cos(\theta_i-\theta_j))\right\}},
 \qquad
 S^\rho_{{ij}}=\frac{{W^\rho_{{ij}}}}{{\sum_\ell W^\rho_{{i\ell}}}}
 =\softmax_j\!\left[\rho(\cos(\theta_i-\theta_j))\right],
 \]
 \[
 K_\Gamma=(1-\eta)I+\eta D_W^{{-1/2}}W^\rho D_W^{{-1/2}},
 \qquad D_W=\operatorname{{diag}}(W^\rho\mathbf 1).
\]
The experiments fix $\rho(t)=t$ and $\eta=0.2$.  The function $\rho$ is a
kernel-specific modelling choice that selects the GP feature space; it is not
an attention temperature fitted by the network.  Admissible choices must make
$W^\rho\succeq0$; for the chosen von Mises kernel the symmetric normalization
in $K_\Gamma$ preserves a Hermitian positive GP covariance.  On a uniform full
aperture its constant row sum makes it equivalent to the symmetric form of
the row-softmax weights.  Define
\[
 H_D=N_DK_\Gamma N_D^*+I,\qquad
 H_DQ_\mu=\Phi,\qquad H_DQ_\Sigma=N_DK_\Gamma.
\]
One tied recurrence acts on $[\Phi,N_DK_\Gamma]$ and yields
\[
 M_D=K_\Gamma N_D^*Q_\mu,\qquad
 \Sigma_D=K_\Gamma-K_\Gamma N_D^*Q_\Sigma.
\]
Thus a small mean residual is insufficient for Bayesian correctness: the
covariance residual must converge as well.
This shares the posterior-distribution ICL viewpoint of
Kang, Lee and Cheng~\cite{{kang2026}}, but the learned object here is an
iterative conditioner for a prescribed nonlinear GP feature space.
We also report the fraction of grid scores for which the direct-solve reference
lies within the approximate $\mu\pm1.96\sigma$ band.  This is a numerical-UQ
fidelity diagnostic: it checks whether iterative error is small relative to
the reported posterior width, but is not frequentist coverage of the obstacle.
This is the original active near-field LSM with all deterministic point
sources fired at every acquisition.  It does not replace $N_D$ by the
cross-correlation, volume or imaginary near-field operators used for random
sources, nor by the asymptotic small-random-scatterer data
model~\cite{{garnier2022,garnier2024}}.  Randomness
below concerns obstacles, additive noise and held-out acquisition geometry,
not the physical source model.

\section{{Variable-context architecture and matched solvers}}
The covariance block uses a deterministic orthonormal harmonic sketch with a
fixed number of columns and $m$ rows.  Its feature dimension is therefore
independent of context length, while the physical operator is never padded or
replaced by a learned surrogate.  Training contexts are
{protocol["training_context_sizes"]}; evaluation contexts are
{protocol["evaluation_context_sizes"]}, so the endpoints include context-length
extrapolation.  Controller widths are {protocol["network_widths"]} and sample
sizes are {protocol["dataset_sizes"]}.

At test depth $T={protocol["eval_depth"]}$, all recurrent methods use exactly
$T$ applications of the transformed operator for both posterior blocks.
Identity-CG sets the population factor to $I$.  Population-PCG learns a positive
geometry/population factor and retains exact task-wise Krylov coefficients.
Context-PCG augments that factor with a permutation-equivariant token encoder
of the current near-field Hessian.  It predicts positive modal gains in the
fixed angular-kernel basis once per prompt; the factor is then held fixed for
all $T$ iterations, so this is standard PCG rather than flexible CG.  Writing
the fixed receiver angular-feature kernel as
$G_\Gamma=U_\Gamma\Lambda_\Gamma U_\Gamma^*$, its factor has the form
\[
 C_\theta(D,\Gamma)=U_\Gamma
 \operatorname{{diag}}\!\left(g_\theta(
 U_\Gamma^*H_DU_\Gamma)\right)^{{1/2}}U_\Gamma^*,
 \qquad g_\theta>0.
\]
The task token for each angular mode contains diagonal, row-$\ell_1$ and
row-$\ell_2$ statistics of $U_\Gamma^*H_DU_\Gamma$; shared token maps and mean
pooling make the controller valid at variable $m$.
The analytic--learned hybrid instead uses
\[
 g_{{\mathrm{{hyb}},j}}=\frac{{
 \exp\{{m^{{-1}}\sum_k\log (U_\Gamma^*H_DU_\Gamma)_{{kk}}\}}
 }}{{(U_\Gamma^*H_DU_\Gamma)_{{jj}}}}
 \exp\{{\delta_{{\theta,j}}\}},\qquad |\delta_{{\theta,j}}|\leq\tfrac12 .
\]
The zero-initialized residual network therefore begins exactly at the
training-free angular-Jacobi factor and can refine it only inside a prescribed
multiplicative trust region; its regularizer acts on $\delta_\theta$, not on the
analytic base.
Looped HB~\cite{{polyak1964}}, Chebyshev and Richardson additionally infer
finite-horizon spectral
endpoints from RHS-weighted Krylov statistics.  Consequently, PCG versus CG
isolates learned conditioning; a looped method versus PCG measures whether a
stationary learned recurrence is worth replacing Krylov adaptation.
Post-training controls construct Jacobi factors directly from the current
$H_D$: its physical diagonal, contiguous angular blocks, or the diagonal of
$U_\Gamma^*H_DU_\Gamma$.  The last is the non-learned operator-aware baseline
for the same fixed GP feature basis used by context-PCG.
With $G$ spatial probes, one recurrent step costs
$O(m^2(G+m))$ for the concatenated posterior blocks and stores
$O(m(G+m))$ state; depth $T$ multiplies the first term.  A dense direct solve
adds an $O(m^3)$ factorization.  The empirical timing curves retain forward
construction and decoding overhead, so their slope need not equal the pure
matrix--matrix asymptote at small $m$.
For any fixed SPD factor $C_D$, standard PCG gives~\cite{{saad2003}}
\[
 \frac{{\|e_T\|_{{H_D}}}}{{\|e_0\|_{{H_D}}}}
 \leq 2\left(
 \frac{{\sqrt{{\kappa_D}}-1}}{{\sqrt{{\kappa_D}}+1}}
 \right)^T,
 \qquad \kappa_D=\kappa(C_D^*H_DC_D).
\]
The condition-number audit below therefore supplies the missing link from an
in-context factor to a depth-dependent solver certificate.
We distinguish two residual norms.  For $q_T=C_Dx_T$ and right-hand side $b$,
\[
 r_{{\rm tr}}=
 \frac{{\|C_D^*(b-H_Dq_T)\|_F}}{{\|C_D^*b\|_F}},\qquad
 r_{{\rm phys}}=
 \frac{{\|b-H_Dq_T\|_F}}{{\|b\|_F}}.
\]
The online objective and original dataset/width/context scaling grid store
$r_{{\rm tr}}$, because that is the residual seen by the transformed solver.
Since its norm changes with $C_D$, every post-training CG comparison, depth
curve, tolerance frontier and geometry/stress audit below instead uses
$r_{{\rm phys}}$.  Score error and Bayesian coverage are coordinate
independent.  Both residuals are retained in the released CSV files.
To identify the intended factor rather than merely a visually acceptable LSM
ranking, context-PCG also minimizes
\[
 \mathcal L_{{\rm id}}(D)=\frac1m
 \left\|I-\widetilde H_{{\theta,D}}\right\|_F^2,
 \qquad
 \widetilde H_{{\theta,D}}=
 \frac{{C_\theta^*H_DC_\theta}}
 {{\|C_\theta^*H_DC_\theta\|_\infty}}.
\]
Indeed, $\varepsilon_D=\|I-\widetilde H_{{\theta,D}}\|_2
\leq\sqrt{{m\mathcal L_{{\rm id}}(D)}}$.  Whenever
$\varepsilon_D<1$, all eigenvalues lie in
$[1-\varepsilon_D,1+\varepsilon_D]$ and
$\kappa_D\leq(1+\varepsilon_D)/(1-\varepsilon_D)$.  This is the explicit
identification-loss $\Rightarrow$ spectral certificate $\Rightarrow$ PCG-depth
chain; ranking loss alone has no such implication.

\section{{Axes and scaling model}}
We adopt the depth--width--context--time resource axes of
Bordelon, Letey and Pehlevan~\cite{{bordelon2025}}.  Their exact
proportional-limit asymptotics apply to deep \emph{{linear}} self-attention regression with
specified covariance ensembles.  The nonlinear fixed-softmax GP feature map,
complex obstacle-dependent near-field operator and Bayesian covariance solve
here violate those assumptions; importing their closed risk equations would
therefore be unjustified.  We retain the resource decomposition and test an
explicitly descriptive finite-grid law instead.
The dataset size $n$ is the number of unique physical inverse problems
presented once in an online stream; the eight examples in an optimizer batch
share one sampled acquisition configuration.  The context length $m$ is both
the number of point sources and receivers, giving $m^2$ complex measurements.
Network size $P$ counts trainable parameters.  Each held-out configuration
contains fixed, never-trained MFS tasks spanning ID obstacles and OOD obstacle
count, shape, noise, aperture and frequency.

\begin{{table}}[t]
\centering\scriptsize
\caption{{Held-out physical scenarios.  Noise is relative complex measurement
noise; aperture is in degrees.}}
\begin{{tabular}}{{lccccc}}
\toprule Scenario & obstacles & shape & $k$ & noise & aperture\\
\midrule
{chr(10).join(scenario_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

For localization risk $R_{{\rm loc}}=1-\mathrm{{AP}}$ and bounded numerical
risk $R_{{\rm solve}}=r_{{\rm tr}}/(1+r_{{\rm tr}})$, we separately fit the
descriptive law
\begin{{equation}}
 R(n,m,P)=R_\infty+A(n/n_0)^{{-\alpha}}+B(m/m_0)^{{-\beta}}
 +C(P/P_0)^{{-\gamma}}+D\sqrt{{(P/P_0)/(n/n_0)}}.
 \label{{eq:scaling}}
\end{{equation}}
The last term allows a finite-data width penalty.  We constrain
$\alpha,\gamma>0$ but let $\beta$ be signed: more measurements can improve
statistical localization while simultaneously making a fixed-depth solve
harder through condition growth.  Equation~\eqref{{eq:scaling}} is an
empirical scaling model, not a theorem.

\begin{{table}}[t]
\centering\small
\caption{{Empirical exponents in Eq.~\eqref{{eq:scaling}} for both risks;
positive data/width exponents indicate improvement.  The context exponent is
signed: $\beta<0$ records fixed-depth degradation as the linear system grows.}}
\begin{{tabular}}{{llcccc}}
\toprule Method & risk & data $\alpha$ & context $\beta$ & width $\gamma$ & $R^2$\\
\midrule
{chr(10).join(fit_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\FloatBarrier
\section{{A simultaneous held-out generalization bound}}
Let $L_c\in[0,1]$ be localization risk for a pre-specified trained
configuration $c=(n,m,P,$ method$)$, and let $\widehat R_c$ average $T_c$
independent held-out tasks.  A union bound and one-sided Hoeffding inequality
give the following non-asymptotic certificate.
\begin{{theorem}}[Simultaneous task-distribution bound]
For $K$ evaluated configurations, with probability at least $1-\delta$,
simultaneously for every $c$,
\[
 R_c\leq \widehat R_c+
 \sqrt{{\frac{{\log(K/\delta)}}{{2T_c}}}}.
\]
\end{{theorem}}
We use $\delta={summary["delta"]}$ and report the clipped upper bound in
Figure~\ref{{fig:bounds}}.  Training samples determine the learned hypothesis
and hence $\widehat R_c$; held-out sample size controls certification slack.
Because seed is pooled rather than included in $c$, the certificate concerns
the mean risk of the three evaluated training seeds, not a guarantee for each
seed separately.
The same theorem is applied to the bounded numerical risk
$L_c^{{\rm solve}}=r_c^{{\rm tr}}/(1+r_c^{{\rm tr}})$, where
$r_c^{{\rm tr}}$ is the transformed-coordinate posterior-mean residual
stored by the training grid.  This second risk remains sensitive after
localization AP saturates.
This distinction matters.  Replacing $T_c$ by training size $n$ after fitting
would be invalid, while a generic parameter-count uniform bound for this
nonconvex complex network is vacuous at the present $P/n$ ratios.
{geometry_certificate}
For a comparable curve, every dataset checkpoint in Figure~\ref{{fig:bounds}}
uses exactly {summary["bound_tasks_per_scenario"]} tasks per scenario and seed.
The largest checkpoint is additionally audited with
{summary["final_tasks_per_scenario"]} tasks per scenario; those tighter values
are stored separately and used in the final performance table.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{scaling_generalization_bounds.png}}
\caption{{Empirical localization and numerical solver risks with simultaneous
95\% held-out upper bounds across dataset and context sizes.}}
\label{{fig:bounds}}
\end{{figure}}

{geometry_figure}

\FloatBarrier
\section{{Results}}
\begin{{table}}[t]
\centering\small
\caption{{Final comparison at context $m=24$, width
{summary["central_width"]} and $n={summary["max_dataset_size"]}$.  Baselines do
not train.  Runtime is milliseconds per held-out batch.}}
\begin{{tabular}}{{lcccccc}}
\toprule Method & AP & $r_{{\rm tr}}$ mean & $r_{{\rm tr}}$ cov. & score error & 95\% cov. & ms\\
\midrule
{chr(10).join(comparison_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering\scriptsize
\caption{{Scenario-wise comparison at the extrapolative context
$m={summary["largest_context"]}$, width {summary["central_width"]} and
$n={summary["max_dataset_size"]}$.  Gains use the transformed-coordinate
residual $r_{{\rm tr}}$; values above one favor context-PCG.}}
\begin{{tabular}}{{lrrrrrr}}
\toprule Scenario & AP CG & AP C-PCG & $r_{{\rm tr}}$ CG & $r_{{\rm tr}}$ C-PCG & mean gain & cov. gain\\
\midrule
{chr(10).join(scenario_comparison_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

On the original scaling grid, prompt-conditioned PCG lowers the
transformed-coordinate posterior-mean residual relative to CG in
{int(summary["context_beats_cg_scenarios"])} of
{int(summary["scenario_comparison_count"])} held-out scenarios at $m=
{summary["largest_context"]}$.  Its worst scenario-wise gain is
{format_number(float(summary["context_worst_scenario_mean_gain"]))}$\times$
and its median gain is
{format_number(float(summary["context_median_scenario_mean_gain"]))}$\times$.
All {int(summary["context_seed_scenario_count"])} individual
training-seed/scenario cells favor context-PCG; the smallest such gain is
{format_number(float(summary["context_worst_seed_scenario_gain"]))}$\times$.
The largest absolute AP change is only
{format_number(float(summary["context_max_abs_scenario_ap_change"]))}; hence
the learned conditioner is substantially more accurate in score fidelity while
localization effectiveness remains essentially tied at the physical
point-spread limit.
At $m={summary["largest_context"]}$, context-PCG attains numerical 95\% band
coverage {format_number(float(summary["largest_context_context_coverage"]))},
compared with {format_number(float(summary["largest_context_cg_coverage"]))}
for CG.  This is UQ fidelity to the exact Bayesian solve, not a physical
coverage claim.
The foundation-model width sweep also rejects a ``bigger is automatically
better'' interpretation.  Width {int(summary["context_best_width"])} gives the
smallest extrapolative transformed residual, while width
{int(summary["context_largest_width"])} has
{format_number(float(summary["context_largest_to_best_parameter_ratio"]))}$\times$
as many parameters and a
{format_number(float(summary["context_largest_to_best_residual_ratio"]))}$\times$
larger residual.  The controller has reached a feature/basis plateau rather
than a parameter-limited regime.

{cg_comparison_figure}

{stress_figure}

{joint_conditioning_figure}

{joint_reconstruction_figure}

{results_interpretation}

\FloatBarrier
\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{scaling_dataset.png}}
\caption{{Learning curves in dataset size at fixed network width and context.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{scaling_context.png}}
\caption{{Context-length scaling.  Context $m=8$ and $m=48$ are outside the
training context set.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{scaling_network.png}}
\caption{{Network-size scaling at the largest completed dataset size.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.9\linewidth]{{scaling_time.png}}
\caption{{Wall-clock learning curves.  Time includes physical task generation,
forward/backward propagation and optimization, but excludes held-out audits.
Curves use the median over seeds because the accelerator is shared.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{scaling_runtime.png}}
\caption{{Inference scaling with $m^2$ complex measurements and the
residual--time Pareto comparison at $m=24$.}}
\end{{figure}}

{depth_figure}

{conditioning_figure}

{classical_figure}

{tolerance_figure}

{reconstruction_figure}

\FloatBarrier
\section{{Interpretation and limitations}}
Six claims are deliberately separated.  First, CG is a strong task-adaptive
baseline because its coefficients already depend on the current Krylov space.
Second, population-PCG tests whether geometry-distribution conditioning adds
value when no obstacle-specific preconditioner is known.  Third, context-PCG
tests prompt-dependent modal gains while keeping the angular-kernel basis
fixed.  Fourth, hybrid-PCG tests whether a bounded learned correction adds to
the analytic GP-basis diagonal.  Fifth, looped HB tests amortized fixed-depth
iteration, not a universally superior replacement for CG.  Sixth,
angular-Jacobi-PCG tests whether the same task signal can be extracted
analytically without training.
A fully equivariant task-specific
preconditioner would map current moment or Ritz information into an SPD
operator $C_D(A_D)\approx A_D^{{-1/2}}$.  Neither modal model can rotate all
obstacle-dependent eigendirections; a block-Lanczos or polynomial-in-$A_D$
extension is required when the commutator audit is large.
The joint-shift audit also shows that nearly equal condition numbers can have
different finite-depth residuals.  A next loop should therefore encode the
right-hand-side-weighted spectral measure through moments
$B^*A_D^jB$, not only extremal eigenvalues or scalar condition number; exact
outer Krylov coefficients can then preserve the CG guarantee.

The held-out bound certifies the stated simulator distribution only.  The
forward data are generated by MFS, not laboratory measurements, and context
extrapolation does not imply robustness to arbitrary receiver geometries.
Physical resolution remains controlled by aperture, frequency and sensor
count.  These qualifications prevent a small numerical loss from being
misreported as solved inverse scattering.

\section{{Reproducibility}}
All task-level records, cumulative training times, checkpoints, runtime
measurements, scenario definitions, fitted parameters and bound values are
stored beside this PDF.  The experiment uses three fixed seeds and preserves
the exact same held-out tasks across methods, widths and dataset checkpoints.

\begin{{thebibliography}}{{9}}
\bibitem{{bordelon2025}} B. Bordelon, M. I. Letey, and C. Pehlevan.
Theory of scaling laws for in-context regression: depth, width, context and
time. \emph{{arXiv:2510.01098}}, 2025.
\bibitem{{kang2026}} G. Kang, C. J. Lee, and X. Cheng.
Transformers can learn posterior predictive distributions in-context.
\emph{{arXiv:2605.26713}}, 2026.
\bibitem{{garnier2022}} J. Garnier, H. Haddar, and H. Montanelli.
The linear sampling method for random sources. \emph{{arXiv:2210.15560}}, 2022.
\bibitem{{garnier2024}} J. Garnier, H. Haddar, and H. Montanelli.
The linear sampling method for data generated by small random scatterers.
\emph{{arXiv:2403.19482}}, 2024.
\bibitem{{polyak1964}} B. T. Polyak. Some methods of speeding up the convergence
of iteration methods. \emph{{USSR Computational Mathematics and Mathematical
Physics}}, 1964.
\bibitem{{saad2003}} Y. Saad. \emph{{Iterative Methods for Sparse Linear
Systems}}, second edition, SIAM, 2003.
\end{{thebibliography}}
\end{{document}}
"""
    tex_path = results_dir / "near_field_scaling_note.tex"
    tex_path.write_text(text.strip() + "\n", encoding="utf-8")
    return tex_path


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    combined, training, runtime, protocol = load_data(results_dir)
    expanded = add_all_regime(combined)
    aggregate = aggregate_tasks(expanded)
    learned = combined[combined["dataset_size"] > 0].copy()
    baselines = aggregate[aggregate["dataset_size"] == 0].copy()
    group_columns = [
        "seed",
        "method",
        "network_width",
        "dataset_size",
        "context_size",
        "scenario",
    ]
    tasks_per_group = learned.groupby(group_columns)["task"].nunique()
    bound_tasks_per_scenario = int(tasks_per_group.min())
    max_dataset_size = int(learned["dataset_size"].max())
    curve_sample = learned[learned["task"] < bound_tasks_per_scenario]
    bounds = simultaneous_heldout_bounds(curve_sample, args.delta)
    final_bounds = simultaneous_heldout_bounds(
        learned[learned["dataset_size"] == max_dataset_size], args.delta
    )
    fits, fitted = fit_scaling_laws(bounds)
    width = central_width(protocol)
    context = (
        24
        if 24 in set(combined["context_size"])
        else int(combined["context_size"].median())
    )
    largest_context = int(learned["context_size"].max())
    depth_path = results_dir / "depth_scaling.csv"
    depth_runtime_path = results_dir / "depth_runtime.csv"
    depth_raw = pd.DataFrame()
    depth_aggregate = pd.DataFrame()
    depth_runtime = pd.DataFrame()
    if depth_path.exists() and depth_runtime_path.exists():
        depth_raw = pd.read_csv(depth_path).drop_duplicates(
            [
                "seed",
                "method",
                "network_width",
                "depth",
                "context_size",
                "scenario",
                "task",
            ],
            keep="last",
        )
        depth_runtime = pd.read_csv(depth_runtime_path).drop_duplicates(
            ["seed", "method", "network_width", "depth", "context_size"],
            keep="last",
        )
        depth_aggregate = aggregate_depth_audit(depth_raw)
        depth_aggregate.to_csv(
            results_dir / "depth_scaling_aggregated.csv", index=False
        )
        save_depth_scaling(
            results_dir / "scaling_depth.png",
            depth_aggregate,
            depth_runtime,
            context=context,
            largest_context=largest_context,
        )
    conditioning_path = results_dir / "preconditioner_conditioning.csv"
    conditioning_raw = pd.DataFrame()
    conditioning_aggregate = pd.DataFrame()
    if conditioning_path.exists():
        conditioning_raw = pd.read_csv(conditioning_path).drop_duplicates(
            ["seed", "method", "context_size", "scenario", "task"],
            keep="last",
        )
        conditioning_aggregate = save_conditioning_audit(
            results_dir / "scaling_conditioning.png", conditioning_raw
        )
        conditioning_aggregate.to_csv(
            results_dir / "preconditioner_conditioning_aggregated.csv",
            index=False,
        )
    classical_path = results_dir / "classical_preconditioners.csv"
    classical_runtime_path = results_dir / "classical_preconditioner_runtime.csv"
    classical_raw = pd.DataFrame()
    classical_aggregate = pd.DataFrame()
    classical_runtime = pd.DataFrame()
    tolerance_frontier = pd.DataFrame()
    tolerance_winners = pd.DataFrame()
    if classical_path.exists() and classical_runtime_path.exists():
        classical_raw = pd.read_csv(classical_path).drop_duplicates(
            ["seed", "method", "depth", "context_size", "scenario", "task"],
            keep="last",
        )
        classical_runtime = pd.read_csv(classical_runtime_path).drop_duplicates(
            ["seed", "method", "depth", "context_size"],
            keep="last",
        )
        classical_aggregate = aggregate_classical_preconditioners(classical_raw)
        classical_aggregate.to_csv(
            results_dir / "classical_preconditioners_aggregated.csv",
            index=False,
        )
        if (
            not depth_aggregate.empty
            and not depth_runtime.empty
            and not conditioning_aggregate.empty
        ):
            save_classical_preconditioner_comparison(
                results_dir / "scaling_classical_preconditioners.png",
                classical_aggregate,
                classical_runtime,
                depth_aggregate,
                depth_runtime,
                conditioning_aggregate,
                largest_context=largest_context,
            )
            tolerance_frontier = build_tolerance_frontier(
                depth_aggregate,
                depth_runtime,
                classical_aggregate,
                classical_runtime,
            )
            tolerance_frontier.to_csv(
                results_dir / "tolerance_frontier.csv", index=False
            )
            save_tolerance_frontier(
                results_dir / "scaling_tolerance_frontier.png",
                tolerance_frontier,
            )
            reached_frontier = tolerance_frontier[
                tolerance_frontier["reached"] & (tolerance_frontier["inference_ms"] > 0)
            ]
            tolerance_winners = reached_frontier.loc[
                reached_frontier.groupby(["context_size", "tolerance"])[
                    "inference_ms"
                ].idxmin()
            ].sort_values(["tolerance", "context_size"])
            tolerance_winners.to_csv(
                results_dir / "best_method_by_context_tolerance.csv", index=False
            )
    geometry_path = results_dir / "geometry_generalization.csv"
    geometry_raw = pd.DataFrame()
    geometry_bounds = pd.DataFrame()
    geometry_effects = pd.DataFrame()
    if geometry_path.exists():
        geometry_raw = pd.read_csv(geometry_path).drop_duplicates(
            ["seed", "geometry_draw", "method", "context_size", "scenario"],
            keep="last",
        )
        geometry_bounds = simultaneous_geometry_bounds(geometry_raw, args.delta)
        geometry_bounds.to_csv(
            results_dir / "geometry_generalization_bounds.csv", index=False
        )
        geometry_effects = paired_geometry_effects(geometry_raw, largest_context)
        geometry_effects.to_csv(
            results_dir / "paired_geometry_comparison.csv", index=False
        )
        (
            geometry_raw[geometry_raw["context_size"] == largest_context]
            .groupby(["scenario", "method"], as_index=False)
            .agg(
                n_geometry_batches=("geometry_draw", "size"),
                mean_relative_residual=(
                    "original_mean_relative_residual",
                    "mean",
                ),
                transformed_mean_relative_residual=(
                    "mean_relative_residual",
                    "mean",
                ),
                covariance_relative_residual=(
                    "original_covariance_relative_residual",
                    "mean",
                ),
                transformed_covariance_relative_residual=(
                    "covariance_relative_residual",
                    "mean",
                ),
                average_precision=("average_precision", "mean"),
                numerical_coverage_95=("numerical_coverage_95", "mean"),
            )
            .to_csv(results_dir / "geometry_method_comparison.csv", index=False)
        )
        save_geometry_generalization_bounds(
            results_dir / "scaling_geometry_generalization.png", geometry_bounds
        )
        if (
            not depth_aggregate.empty
            and not depth_runtime.empty
            and not classical_aggregate.empty
            and not classical_runtime.empty
        ):
            save_cg_comparison_dashboard(
                results_dir / "scaling_cg_comparison.png",
                depth_aggregate,
                depth_runtime,
                classical_aggregate,
                classical_runtime,
                geometry_effects,
                largest_context=largest_context,
            )

    stress_path = results_dir / "cg_stress_sweep.csv"
    stress_raw = pd.DataFrame()
    stress_effects = pd.DataFrame()
    if stress_path.exists():
        stress_raw = pd.read_csv(stress_path).drop_duplicates(
            ["seed", "axis", "level", "geometry_draw", "method"],
            keep="last",
        )
        stress_effects = paired_cg_stress_effects(stress_raw)
        stress_effects.to_csv(
            results_dir / "cg_stress_comparison.csv", index=False
        )
        save_cg_stress_sweep(
            results_dir / "scaling_cg_stress_test.png", stress_effects
        )

    joint_conditioning_path = results_dir / "joint_conditioning.csv"
    joint_conditioning_raw = pd.DataFrame()
    joint_conditioning_aggregate = pd.DataFrame()
    if joint_conditioning_path.exists():
        joint_conditioning_raw = pd.read_csv(
            joint_conditioning_path
        ).drop_duplicates(
            ["seed", "joint_severity", "geometry_draw", "method"],
            keep="last",
        )
        joint_conditioning_aggregate = save_joint_conditioning(
            results_dir / "scaling_joint_conditioning.png",
            joint_conditioning_raw,
        )
        joint_conditioning_aggregate.to_csv(
            results_dir / "joint_conditioning_aggregated.csv", index=False
        )

    aggregate_residual_names = {
        "mean_relative_residual_mean": "transformed_mean_relative_residual_mean",
        "mean_relative_residual_std": "transformed_mean_relative_residual_std",
        "covariance_relative_residual_mean": (
            "transformed_covariance_relative_residual_mean"
        ),
        "covariance_relative_residual_std": (
            "transformed_covariance_relative_residual_std"
        ),
        "mean_relative_residual_ci95": "transformed_mean_relative_residual_ci95",
        "covariance_relative_residual_ci95": (
            "transformed_covariance_relative_residual_ci95"
        ),
    }
    bound_residual_names = {
        "mean_relative_residual": "transformed_mean_relative_residual",
        "covariance_relative_residual": "transformed_covariance_relative_residual",
    }
    aggregate.rename(columns=aggregate_residual_names).to_csv(
        results_dir / "scaling_aggregated.csv", index=False
    )
    bounds.rename(columns=bound_residual_names).to_csv(
        results_dir / "generalization_bounds.csv", index=False
    )
    final_bounds.rename(columns=bound_residual_names).to_csv(
        results_dir / "generalization_bounds_final_extended.csv", index=False
    )
    fits.to_csv(results_dir / "scaling_law_fits.csv", index=False)
    fitted.to_csv(results_dir / "scaling_law_predictions.csv", index=False)
    save_dataset_scaling(
        results_dir / "scaling_dataset.png", aggregate, baselines, width, context
    )
    save_context_scaling(
        results_dir / "scaling_context.png",
        aggregate,
        baselines,
        width,
        max_dataset_size,
    )
    save_network_scaling(
        results_dir / "scaling_network.png", aggregate, max_dataset_size, context
    )
    save_time_learning(results_dir / "scaling_time.png", aggregate, width, context)
    save_generalization_bounds(
        results_dir / "scaling_generalization_bounds.png", bounds, width
    )
    save_runtime_scaling(
        results_dir / "scaling_runtime.png",
        runtime,
        aggregate,
        width,
        max_dataset_size,
    )
    cg_gain_by_context(
        aggregate,
        runtime,
        width,
        max_dataset_size,
    ).rename(
        columns={
            "mean_relative_residual_mean_candidate": (
                "transformed_mean_relative_residual_mean_candidate"
            ),
            "covariance_relative_residual_mean_candidate": (
                "transformed_covariance_relative_residual_mean_candidate"
            ),
            "mean_relative_residual_mean_cg": (
                "transformed_mean_relative_residual_mean_cg"
            ),
            "covariance_relative_residual_mean_cg": (
                "transformed_covariance_relative_residual_mean_cg"
            ),
            "mean_residual_gain": "transformed_mean_residual_gain",
            "covariance_residual_gain": "transformed_covariance_residual_gain",
        }
    ).to_csv(results_dir / "cg_gain_by_context.csv", index=False)
    if not depth_aggregate.empty and not depth_runtime.empty:
        cg_gain_by_depth(depth_aggregate, depth_runtime).rename(
            columns={
                "mean_relative_residual_mean_candidate": (
                    "physical_mean_relative_residual_mean_candidate"
                ),
                "covariance_relative_residual_mean_candidate": (
                    "physical_covariance_relative_residual_mean_candidate"
                ),
                "mean_relative_residual_mean_cg": (
                    "physical_mean_relative_residual_mean_cg"
                ),
                "covariance_relative_residual_mean_cg": (
                    "physical_covariance_relative_residual_mean_cg"
                ),
                "mean_residual_gain": "physical_mean_residual_gain",
                "covariance_residual_gain": "physical_covariance_residual_gain",
            }
        ).to_csv(results_dir / "cg_gain_by_depth.csv", index=False)

    comparison = final_comparison(aggregate, runtime, width, max_dataset_size)
    cg = comparison[comparison["method"] == "identity-CG"]
    pcg = comparison[comparison["method"] == "population-PCG"]
    context_pcg = comparison[comparison["method"] == "context-PCG"]
    hybrid_pcg = comparison[comparison["method"] == "hybrid-PCG"]
    mean_gain = (
        float(cg["mean_relative_residual_mean"].iloc[0])
        / float(pcg["mean_relative_residual_mean"].iloc[0])
        if not cg.empty and not pcg.empty
        else float("nan")
    )
    covariance_gain = (
        float(cg["covariance_relative_residual_mean"].iloc[0])
        / float(pcg["covariance_relative_residual_mean"].iloc[0])
        if not cg.empty and not pcg.empty
        else float("nan")
    )
    context_mean_gain = (
        float(cg["mean_relative_residual_mean"].iloc[0])
        / float(context_pcg["mean_relative_residual_mean"].iloc[0])
        if not cg.empty and not context_pcg.empty
        else float("nan")
    )
    context_covariance_gain = (
        float(cg["covariance_relative_residual_mean"].iloc[0])
        / float(context_pcg["covariance_relative_residual_mean"].iloc[0])
        if not cg.empty and not context_pcg.empty
        else float("nan")
    )
    hybrid_mean_gain = (
        float(cg["mean_relative_residual_mean"].iloc[0])
        / float(hybrid_pcg["mean_relative_residual_mean"].iloc[0])
        if not cg.empty and not hybrid_pcg.empty
        else float("nan")
    )
    hybrid_covariance_gain = (
        float(cg["covariance_relative_residual_mean"].iloc[0])
        / float(hybrid_pcg["covariance_relative_residual_mean"].iloc[0])
        if not cg.empty and not hybrid_pcg.empty
        else float("nan")
    )
    final_pcg_tasks = learned[
        (learned["method"] == "population-PCG")
        & (learned["network_width"] == width)
        & (learned["dataset_size"] == max_dataset_size)
    ]
    final_context_pcg_tasks = learned[
        (learned["method"] == "context-PCG")
        & (learned["network_width"] == width)
        & (learned["dataset_size"] == max_dataset_size)
    ]
    final_hybrid_pcg_tasks = learned[
        (learned["method"] == "hybrid-PCG")
        & (learned["network_width"] == width)
        & (learned["dataset_size"] == max_dataset_size)
    ]
    cg_tasks = combined[combined["method"] == "identity-CG"]

    def residual_gain(
        reference: pd.DataFrame, candidate: pd.DataFrame, column: str
    ) -> float:
        return float(reference[column].mean() / candidate[column].mean())

    def runtime_median(method: str, context_size: int) -> float:
        expected_width = 0 if method in ("identity-CG", "exact") else width
        selected = runtime[
            (runtime["method"] == method)
            & (runtime["network_width"] == expected_width)
            & (runtime["context_size"] == context_size)
        ]
        return float(selected["inference_ms"].median())

    largest_pcg = final_pcg_tasks[final_pcg_tasks["context_size"] == largest_context]
    largest_context_pcg = final_context_pcg_tasks[
        final_context_pcg_tasks["context_size"] == largest_context
    ]
    largest_hybrid_pcg = final_hybrid_pcg_tasks[
        final_hybrid_pcg_tasks["context_size"] == largest_context
    ]
    largest_cg = cg_tasks[cg_tasks["context_size"] == largest_context]

    def depth_method_gain(
        depth: int,
        reference_method: str,
        candidate_method: str,
    ) -> float:
        if depth_raw.empty:
            return float("nan")
        selected = depth_raw[
            (depth_raw["context_size"] == largest_context)
            & (depth_raw["depth"] == depth)
        ]
        reference = selected[selected["method"] == reference_method]
        candidate = selected[selected["method"] == candidate_method]
        if reference.empty or candidate.empty:
            return float("nan")
        metric = (
            "original_mean_relative_residual"
            if "original_mean_relative_residual" in selected.columns
            else "mean_relative_residual"
        )
        return residual_gain(reference, candidate, metric)

    def depth_residual_gain(depth: int, method: str = "population-PCG") -> float:
        return depth_method_gain(depth, "identity-CG", method)

    def fit_r_squared(risk_type: str) -> float:
        selected = fits[
            (fits["method"] == "population-PCG")
            & (fits["risk_type"] == risk_type)
        ]
        return float(selected["r_squared"].iloc[0]) if not selected.empty else float("nan")

    def minimum_depth_for_residual(method: str, tolerance: float) -> int:
        if depth_aggregate.empty:
            return -1
        selected = depth_aggregate[
            (depth_aggregate["method"] == method)
            & (depth_aggregate["regime_class"] == "All")
            & (depth_aggregate["context_size"] == largest_context)
            & (depth_aggregate["depth"] > 0)
            & (depth_aggregate["mean_relative_residual_mean"] <= tolerance)
        ]
        return int(selected["depth"].min()) if not selected.empty else -1

    def depth_runtime_at(method: str, depth: int) -> float:
        if depth_runtime.empty or depth < 0:
            return float("nan")
        selected = depth_runtime[
            (depth_runtime["method"] == method)
            & (depth_runtime["depth"] == depth)
            & (depth_runtime["context_size"] == largest_context)
        ]
        return float(selected["inference_ms"].median())

    def largest_context_condition_metric(method: str, column: str) -> float:
        if conditioning_aggregate.empty:
            return float("nan")
        selected = conditioning_aggregate[
            (conditioning_aggregate["method"] == method)
            & (conditioning_aggregate["context_size"] == largest_context)
            & (conditioning_aggregate["regime_class"] == "All")
        ]
        return (
            float(selected[column].iloc[0])
            if not selected.empty
            else float("nan")
        )

    geometry_largest = (
        geometry_raw[geometry_raw["context_size"] == largest_context]
        if not geometry_raw.empty
        else pd.DataFrame()
    )

    def geometry_method_mean(method: str, metric: str) -> float:
        if geometry_largest.empty:
            return float("nan")
        selected = geometry_largest[geometry_largest["method"] == method]
        return float(selected[metric].mean()) if not selected.empty else float("nan")

    def geometry_effect_metric(comparison_name: str, metric: str) -> float:
        if geometry_effects.empty:
            return float("nan")
        selected = geometry_effects[
            (geometry_effects["comparison"] == comparison_name)
            & (geometry_effects["scenario"] == "All")
        ]
        return float(selected[metric].iloc[0]) if not selected.empty else float("nan")

    def classical_metric(method: str, column: str, depth: int = 32) -> float:
        if classical_aggregate.empty:
            return float("nan")
        selected = classical_aggregate[
            (classical_aggregate["method"] == method)
            & (classical_aggregate["regime_class"] == "All")
            & (classical_aggregate["context_size"] == largest_context)
            & (classical_aggregate["depth"] == depth)
        ]
        return float(selected[column].iloc[0]) if not selected.empty else float("nan")

    def classical_runtime_at(method: str, depth: int = 32) -> float:
        if classical_runtime.empty:
            return float("nan")
        selected = classical_runtime[
            (classical_runtime["method"] == method)
            & (classical_runtime["context_size"] == largest_context)
            & (classical_runtime["depth"] == depth)
        ]
        return float(selected["inference_ms"].median())

    def minimum_classical_depth_for_residual(method: str, tolerance: float) -> int:
        if classical_aggregate.empty:
            return -1
        selected = classical_aggregate[
            (classical_aggregate["method"] == method)
            & (classical_aggregate["regime_class"] == "All")
            & (classical_aggregate["context_size"] == largest_context)
            & (classical_aggregate["mean_relative_residual_mean"] <= tolerance)
        ]
        return int(selected["depth"].min()) if not selected.empty else -1

    def learned_depth_metric(method: str, depth: int, column: str) -> float:
        if depth_aggregate.empty:
            return float("nan")
        selected = depth_aggregate[
            (depth_aggregate["method"] == method)
            & (depth_aggregate["regime_class"] == "All")
            & (depth_aggregate["context_size"] == largest_context)
            & (depth_aggregate["depth"] == depth)
        ]
        return float(selected[column].iloc[0]) if not selected.empty else float("nan")

    context_depth32_residual = learned_depth_metric(
        "context-PCG", 32, "mean_relative_residual_mean"
    )
    hybrid_depth32_residual = learned_depth_metric(
        "hybrid-PCG", 32, "mean_relative_residual_mean"
    )

    def tolerance_win_count(method: str) -> int:
        if tolerance_winners.empty:
            return 0
        return int((tolerance_winners["method"] == method).sum())

    geometry_scenario_gain = pd.Series(dtype=float)
    geometry_seed_scenario_gain = pd.Series(dtype=float)
    geometry_angular_to_context_scenario_ratio = pd.Series(dtype=float)
    if not geometry_largest.empty:
        geometry_residuals = geometry_largest.pivot_table(
            index="scenario",
            columns="method",
            values="original_mean_relative_residual",
            aggfunc="mean",
        )
        if {"identity-CG", "context-PCG"}.issubset(geometry_residuals.columns):
            geometry_scenario_gain = (
                geometry_residuals["identity-CG"]
                / geometry_residuals["context-PCG"]
            )
        if {"angular-Jacobi-PCG", "context-PCG"}.issubset(
            geometry_residuals.columns
        ):
            geometry_angular_to_context_scenario_ratio = (
                geometry_residuals["angular-Jacobi-PCG"]
                / geometry_residuals["context-PCG"]
            )
        geometry_seed_residuals = geometry_largest.pivot_table(
            index=["seed", "scenario"],
            columns="method",
            values="original_mean_relative_residual",
            aggfunc="mean",
        )
        if {"identity-CG", "context-PCG"}.issubset(
            geometry_seed_residuals.columns
        ):
            geometry_seed_scenario_gain = (
                geometry_seed_residuals["identity-CG"]
                / geometry_seed_residuals["context-PCG"]
            )

    scenario_comparison = scenario_cg_comparison(
        combined,
        width=width,
        dataset_size=max_dataset_size,
        context_size=largest_context,
    )
    scenario_gains = scenario_comparison.get(
        "transformed_mean_residual_gain", pd.Series(dtype=float)
    )
    scenario_ap_changes = (
        scenario_comparison["context_average_precision"]
        - scenario_comparison["cg_average_precision"]
        if not scenario_comparison.empty
        else pd.Series(dtype=float)
    )
    seed_scenario = pd.concat(
        [largest_cg, largest_context_pcg], ignore_index=True
    ).pivot_table(
        index=["seed", "scenario"],
        columns="method",
        values="mean_relative_residual",
        aggfunc="mean",
    )
    seed_scenario_gain = (
        seed_scenario["identity-CG"] / seed_scenario["context-PCG"]
    )
    context_width_scaling = aggregate[
        (aggregate["method"] == "context-PCG")
        & (aggregate["regime_class"] == "All")
        & (aggregate["dataset_size"] == max_dataset_size)
        & (aggregate["context_size"] == largest_context)
    ]
    best_width_row = context_width_scaling.loc[
        context_width_scaling["mean_relative_residual_mean"].idxmin()
    ]
    largest_width_row = context_width_scaling.loc[
        context_width_scaling["network_width"].idxmax()
    ]
    hybrid_width_scaling = aggregate[
        (aggregate["method"] == "hybrid-PCG")
        & (aggregate["regime_class"] == "All")
        & (aggregate["dataset_size"] == max_dataset_size)
        & (aggregate["context_size"] == largest_context)
    ]
    hybrid_best_width_row = hybrid_width_scaling.loc[
        hybrid_width_scaling["mean_relative_residual_mean"].idxmin()
    ]

    context_depth_001 = minimum_depth_for_residual("context-PCG", 0.01)
    cg_depth_001 = minimum_depth_for_residual("identity-CG", 0.01)
    angular_depth_001 = minimum_classical_depth_for_residual(
        "angular-Jacobi-PCG", 0.01
    )
    optimized_cg_depth_001 = minimum_classical_depth_for_residual(
        "optimized-CG", 0.01
    )
    stress_context_effects = stress_effects[
        stress_effects["candidate"] == "context-PCG"
    ]
    stress_hb_effects = stress_effects[
        stress_effects["candidate"] == "looped-HB"
    ]
    stress_winners = (
        stress_effects.loc[
            stress_effects.groupby(["axis", "level"])[
                "geometric_mean_gain"
            ].idxmax()
        ]
        if not stress_effects.empty
        else pd.DataFrame()
    )

    def stress_metric(
        axis: str,
        level: float,
        candidate: str,
        column: str,
    ) -> float:
        selected = stress_effects[
            (stress_effects["axis"] == axis)
            & (stress_effects["level"] == level)
            & (stress_effects["candidate"] == candidate)
        ]
        return float(selected[column].iloc[0]) if not selected.empty else float("nan")

    def joint_condition_metric(
        level: float,
        method: str,
        column: str,
    ) -> float:
        selected = joint_conditioning_aggregate[
            (joint_conditioning_aggregate["joint_severity"] == level)
            & (joint_conditioning_aggregate["method"] == method)
        ]
        return float(selected[column].iloc[0]) if not selected.empty else float("nan")

    summary: dict[str, object] = {
        "central_width": width,
        "max_dataset_size": max_dataset_size,
        "learned_task_rows": len(learned),
        "baseline_task_rows": int((combined["dataset_size"] == 0).sum()),
        "trained_model_runs": int(
            learned[["seed", "method", "network_width"]]
            .drop_duplicates()
            .shape[0]
        ),
        "completed_dataset_checkpoints": int(
            learned[["seed", "method", "network_width", "dataset_size"]]
            .drop_duplicates()
            .shape[0]
        ),
        "context_sizes": sorted(
            int(value) for value in combined["context_size"].unique()
        ),
        "widths": sorted(int(value) for value in learned["network_width"].unique()),
        "dataset_sizes": sorted(
            int(value) for value in learned["dataset_size"].unique()
        ),
        "delta": args.delta,
        "bound_tasks_per_scenario": bound_tasks_per_scenario,
        "final_tasks_per_scenario": int(
            learned[learned["dataset_size"] == max_dataset_size]["task"].max() + 1
        ),
        "simultaneous_bound_configurations": int(
            bounds["simultaneous_configurations"].max()
        ),
        "cg_to_pcg_mean_residual_gain": mean_gain,
        "cg_to_pcg_covariance_residual_gain": covariance_gain,
        "cg_to_context_pcg_mean_residual_gain": context_mean_gain,
        "cg_to_context_pcg_covariance_residual_gain": context_covariance_gain,
        "cg_to_hybrid_pcg_mean_residual_gain": hybrid_mean_gain,
        "cg_to_hybrid_pcg_covariance_residual_gain": hybrid_covariance_gain,
        "central_context_runtime_ratio": (
            runtime_median("context-PCG", context)
            / runtime_median("identity-CG", context)
        ),
        "largest_context_runtime_ratio": (
            runtime_median("context-PCG", largest_context)
            / runtime_median("identity-CG", largest_context)
        ),
        "largest_context_hybrid_runtime_ratio": (
            runtime_median("hybrid-PCG", largest_context)
            / runtime_median("identity-CG", largest_context)
        ),
        "macro_cg_to_pcg_mean_residual_gain": residual_gain(
            cg_tasks, final_pcg_tasks, "mean_relative_residual"
        ),
        "macro_cg_to_pcg_covariance_residual_gain": residual_gain(
            cg_tasks, final_pcg_tasks, "covariance_relative_residual"
        ),
        "largest_context": largest_context,
        "largest_context_mean_residual_gain": residual_gain(
            largest_cg, largest_pcg, "mean_relative_residual"
        ),
        "largest_context_covariance_residual_gain": residual_gain(
            largest_cg, largest_pcg, "covariance_relative_residual"
        ),
        "macro_cg_to_context_pcg_mean_residual_gain": (
            residual_gain(
                cg_tasks,
                final_context_pcg_tasks,
                "mean_relative_residual",
            )
            if not final_context_pcg_tasks.empty
            else float("nan")
        ),
        "macro_cg_to_context_pcg_covariance_residual_gain": (
            residual_gain(
                cg_tasks,
                final_context_pcg_tasks,
                "covariance_relative_residual",
            )
            if not final_context_pcg_tasks.empty
            else float("nan")
        ),
        "macro_cg_to_hybrid_pcg_mean_residual_gain": (
            residual_gain(cg_tasks, final_hybrid_pcg_tasks, "mean_relative_residual")
            if not final_hybrid_pcg_tasks.empty
            else float("nan")
        ),
        "macro_cg_to_hybrid_pcg_covariance_residual_gain": (
            residual_gain(
                cg_tasks,
                final_hybrid_pcg_tasks,
                "covariance_relative_residual",
            )
            if not final_hybrid_pcg_tasks.empty
            else float("nan")
        ),
        "largest_context_context_pcg_mean_residual_gain": (
            residual_gain(
                largest_cg,
                largest_context_pcg,
                "mean_relative_residual",
            )
            if not largest_context_pcg.empty
            else float("nan")
        ),
        "largest_context_context_pcg_mean_residual": context_depth32_residual,
        "largest_context_hybrid_pcg_mean_residual_gain": (
            residual_gain(
                largest_cg,
                largest_hybrid_pcg,
                "mean_relative_residual",
            )
            if not largest_hybrid_pcg.empty
            else float("nan")
        ),
        "largest_context_hybrid_pcg_mean_residual": hybrid_depth32_residual,
        "largest_context_cg_coverage": float(
            largest_cg["numerical_coverage_95"].mean()
        ),
        "largest_context_context_coverage": (
            float(largest_context_pcg["numerical_coverage_95"].mean())
            if not largest_context_pcg.empty
            else float("nan")
        ),
        "largest_context_hybrid_coverage": (
            float(largest_hybrid_pcg["numerical_coverage_95"].mean())
            if not largest_hybrid_pcg.empty
            else float("nan")
        ),
        "depth_audit_task_rows": int(len(depth_raw)),
        "depth_values": (
            sorted(int(value) for value in depth_raw["depth"].unique() if value > 0)
            if not depth_raw.empty
            else []
        ),
        "depth32_largest_context_mean_gain": depth_residual_gain(32),
        "depth64_largest_context_mean_gain": depth_residual_gain(64),
        "depth32_largest_context_context_gain": depth_residual_gain(
            32, "context-PCG"
        ),
        "depth64_largest_context_context_gain": depth_residual_gain(
            64, "context-PCG"
        ),
        "depth128_largest_context_context_gain": depth_residual_gain(
            128, "context-PCG"
        ),
        "depth32_largest_context_hybrid_gain": depth_residual_gain(
            32, "hybrid-PCG"
        ),
        "depth96_angular_to_hybrid_ratio": (
            classical_metric(
                "angular-Jacobi-PCG", "mean_relative_residual_mean", depth=96
            )
            / learned_depth_metric("hybrid-PCG", 96, "mean_relative_residual_mean")
        ),
        "depth32_context_vs_hb_gain": depth_method_gain(
            32, "looped-HB", "context-PCG"
        ),
        "depth64_context_vs_hb_gain": depth_method_gain(
            64, "looped-HB", "context-PCG"
        ),
        "depth32_largest_context_hb_mean_residual": learned_depth_metric(
            "looped-HB", 32, "mean_relative_residual_mean"
        ),
        "depth32_largest_context_hb_average_precision": learned_depth_metric(
            "looped-HB", 32, "average_precision_mean"
        ),
        "depth32_largest_context_hb_coverage": learned_depth_metric(
            "looped-HB", 32, "numerical_coverage_95_mean"
        ),
        "depth32_largest_context_hb_score_error": learned_depth_metric(
            "looped-HB", 32, "relative_score_error_mean"
        ),
        "depth32_largest_context_cg_score_error": learned_depth_metric(
            "identity-CG", 32, "relative_score_error_mean"
        ),
        "depth32_largest_context_context_score_error": learned_depth_metric(
            "context-PCG", 32, "relative_score_error_mean"
        ),
        "depth32_largest_context_hb_runtime": depth_runtime_at("looped-HB", 32),
        "context_depth_for_residual_01": minimum_depth_for_residual(
            "context-PCG", 0.1
        ),
        "cg_depth_for_residual_01": minimum_depth_for_residual(
            "identity-CG", 0.1
        ),
        "context_depth_for_residual_001": context_depth_001,
        "cg_depth_for_residual_001": cg_depth_001,
        "context_time_for_residual_001": depth_runtime_at(
            "context-PCG", context_depth_001
        ),
        "cg_time_for_residual_001": depth_runtime_at(
            "identity-CG", cg_depth_001
        ),
        "scenario_comparison_count": int(len(scenario_comparison)),
        "context_beats_cg_scenarios": int((scenario_gains > 1.0).sum()),
        "context_worst_scenario_mean_gain": (
            float(scenario_gains.min()) if not scenario_gains.empty else float("nan")
        ),
        "context_median_scenario_mean_gain": (
            float(scenario_gains.median())
            if not scenario_gains.empty
            else float("nan")
        ),
        "context_seed_scenario_count": int(len(seed_scenario_gain)),
        "context_worst_seed_scenario_gain": float(seed_scenario_gain.min()),
        "context_max_abs_scenario_ap_change": (
            float(scenario_ap_changes.abs().max())
            if not scenario_ap_changes.empty
            else float("nan")
        ),
        "context_best_width": int(best_width_row["network_width"]),
        "context_largest_width": int(largest_width_row["network_width"]),
        "context_largest_to_best_parameter_ratio": float(
            largest_width_row["parameter_count"] / best_width_row["parameter_count"]
        ),
        "context_largest_to_best_residual_ratio": float(
            largest_width_row["mean_relative_residual_mean"]
            / best_width_row["mean_relative_residual_mean"]
        ),
        "hybrid_best_width": int(hybrid_best_width_row["network_width"]),
        "hybrid_best_width_mean_residual": float(
            hybrid_best_width_row["mean_relative_residual_mean"]
        ),
        "hybrid_best_width_covariance_residual": float(
            hybrid_best_width_row["covariance_relative_residual_mean"]
        ),
        "hybrid_best_width_coverage": float(
            hybrid_best_width_row["numerical_coverage_95_mean"]
        ),
        "pcg_localization_fit_r2": fit_r_squared("localization"),
        "pcg_solver_fit_r2": fit_r_squared("solver"),
        "conditioning_audit_task_rows": int(len(conditioning_raw)),
        "classical_preconditioner_task_rows": int(len(classical_raw)),
        "classical_angular_mean_residual": classical_metric(
            "angular-Jacobi-PCG", "mean_relative_residual_mean"
        ),
        "classical_block_mean_residual": classical_metric(
            "block-Jacobi-PCG", "mean_relative_residual_mean"
        ),
        "classical_jacobi_mean_residual": classical_metric(
            "Jacobi-PCG", "mean_relative_residual_mean"
        ),
        "classical_optimized_cg_mean_residual": classical_metric(
            "optimized-CG", "mean_relative_residual_mean"
        ),
        "classical_angular_covariance_residual": classical_metric(
            "angular-Jacobi-PCG", "covariance_relative_residual_mean"
        ),
        "classical_angular_coverage": classical_metric(
            "angular-Jacobi-PCG", "numerical_coverage_95_mean"
        ),
        "context_depth32_coverage": learned_depth_metric(
            "context-PCG", 32, "numerical_coverage_95_mean"
        ),
        "classical_angular_condition": classical_metric(
            "angular-Jacobi-PCG", "transformed_condition_median"
        ),
        "classical_angular_condition_gain": classical_metric(
            "angular-Jacobi-PCG", "condition_reduction_median"
        ),
        "classical_angular_runtime": classical_runtime_at("angular-Jacobi-PCG"),
        "classical_optimized_cg_runtime": classical_runtime_at("optimized-CG"),
        "classical_angular_depth_for_residual_001": angular_depth_001,
        "classical_angular_time_for_residual_001": classical_runtime_at(
            "angular-Jacobi-PCG", angular_depth_001
        ),
        "classical_optimized_cg_depth_for_residual_001": optimized_cg_depth_001,
        "classical_optimized_cg_time_for_residual_001": classical_runtime_at(
            "optimized-CG", optimized_cg_depth_001
        ),
        "classical_angular_to_context_residual_ratio": (
            classical_metric("angular-Jacobi-PCG", "mean_relative_residual_mean")
            / context_depth32_residual
        ),
        "classical_angular_to_hybrid_residual_ratio": (
            classical_metric("angular-Jacobi-PCG", "mean_relative_residual_mean")
            / hybrid_depth32_residual
        ),
        "classical_angular_to_context_depth96_residual_ratio": (
            classical_metric(
                "angular-Jacobi-PCG", "mean_relative_residual_mean", depth=96
            )
            / learned_depth_metric(
                "context-PCG", 96, "mean_relative_residual_mean"
            )
        ),
        "classical_angular_to_context_runtime_ratio": (
            classical_runtime_at("angular-Jacobi-PCG")
            / depth_runtime_at("context-PCG", 32)
        ),
        "context_to_optimized_cg_runtime_ratio": (
            depth_runtime_at("context-PCG", 32)
            / classical_runtime_at("optimized-CG")
        ),
        "tolerance_frontier_count": int(len(tolerance_winners)),
        "tolerance_wins_angular": tolerance_win_count("angular-Jacobi-PCG"),
        "tolerance_wins_context": tolerance_win_count("context-PCG"),
        "tolerance_wins_hybrid": tolerance_win_count("hybrid-PCG"),
        "tolerance_wins_population": tolerance_win_count("population-PCG"),
        "tolerance_wins_identity": tolerance_win_count("identity-CG"),
        "tolerance_wins_optimized_cg": tolerance_win_count("optimized-CG"),
        "tolerance_wins_block_jacobi": tolerance_win_count("block-Jacobi-PCG"),
        "cg_stress_rows": int(len(stress_raw)),
        "cg_stress_effect_rows": int(len(stress_effects)),
        "cg_stress_batches_per_level_min": (
            int(stress_effects["n_geometry_batches"].min())
            if not stress_effects.empty
            else 0
        ),
        "cg_stress_min_context_gain": (
            float(stress_context_effects["geometric_mean_gain"].min())
            if not stress_context_effects.empty
            else float("nan")
        ),
        "cg_stress_max_context_gain": (
            float(stress_context_effects["geometric_mean_gain"].max())
            if not stress_context_effects.empty
            else float("nan")
        ),
        "cg_stress_min_context_win_rate": (
            float(stress_context_effects["candidate_win_rate"].min())
            if not stress_context_effects.empty
            else float("nan")
        ),
        "cg_stress_context_global_gain": (
            float(
                np.exp(
                    np.log(stress_context_effects["geometric_mean_gain"]).mean()
                )
            )
            if not stress_context_effects.empty
            else float("nan")
        ),
        "cg_stress_context_ap_delta": (
            float(stress_context_effects["average_precision_delta_mean"].mean())
            if not stress_context_effects.empty
            else float("nan")
        ),
        "cg_stress_context_coverage_delta": (
            float(stress_context_effects["coverage_delta_mean"].mean())
            if not stress_context_effects.empty
            else float("nan")
        ),
        "cg_stress_hb_global_gain": (
            float(np.exp(np.log(stress_hb_effects["geometric_mean_gain"]).mean()))
            if not stress_hb_effects.empty
            else float("nan")
        ),
        "cg_stress_hb_ap_delta": (
            float(stress_hb_effects["average_precision_delta_mean"].mean())
            if not stress_hb_effects.empty
            else float("nan")
        ),
        "cg_stress_hb_coverage_delta": (
            float(stress_hb_effects["coverage_delta_mean"].mean())
            if not stress_hb_effects.empty
            else float("nan")
        ),
        "joint_extreme_context_gain": stress_metric(
            "joint_severity", 3.0, "context-PCG", "geometric_mean_gain"
        ),
        "joint_extreme_context_gain_ci_lower": stress_metric(
            "joint_severity", 3.0, "context-PCG", "gain_ci95_lower"
        ),
        "joint_extreme_context_gain_ci_upper": stress_metric(
            "joint_severity", 3.0, "context-PCG", "gain_ci95_upper"
        ),
        "joint_extreme_context_win_rate": stress_metric(
            "joint_severity", 3.0, "context-PCG", "candidate_win_rate"
        ),
        "joint_extreme_context_win_rate_ci_lower": stress_metric(
            "joint_severity", 3.0, "context-PCG", "win_rate_ci95_lower"
        ),
        "joint_extreme_context_win_rate_ci_upper": stress_metric(
            "joint_severity", 3.0, "context-PCG", "win_rate_ci95_upper"
        ),
        "joint_extreme_context_ap_delta": stress_metric(
            "joint_severity", 3.0, "context-PCG", "average_precision_delta_mean"
        ),
        "joint_extreme_context_coverage_delta": stress_metric(
            "joint_severity", 3.0, "context-PCG", "coverage_delta_mean"
        ),
        "joint_extreme_angular_gain": stress_metric(
            "joint_severity", 3.0, "angular-Jacobi-PCG", "geometric_mean_gain"
        ),
        "joint_extreme_population_gain": stress_metric(
            "joint_severity", 3.0, "population-PCG", "geometric_mean_gain"
        ),
        "joint_extreme_hb_gain": stress_metric(
            "joint_severity", 3.0, "looped-HB", "geometric_mean_gain"
        ),
        "joint_conditioning_rows": int(len(joint_conditioning_raw)),
        "joint_extreme_raw_condition": joint_condition_metric(
            3.0, "identity-CG", "raw_condition_median"
        ),
        "joint_extreme_context_condition": joint_condition_metric(
            3.0, "context-PCG", "transformed_condition_median"
        ),
        "joint_extreme_context_condition_gain": joint_condition_metric(
            3.0, "context-PCG", "condition_reduction_median"
        ),
        "joint_extreme_angular_condition": joint_condition_metric(
            3.0, "angular-Jacobi-PCG", "transformed_condition_median"
        ),
        "joint_extreme_angular_condition_gain": joint_condition_metric(
            3.0, "angular-Jacobi-PCG", "condition_reduction_median"
        ),
        "joint_extreme_commutator": joint_condition_metric(
            3.0, "identity-CG", "geometry_commutator_mean"
        ),
        "cg_stress_level_count": int(len(stress_winners)),
        "cg_stress_context_best_count": (
            int((stress_winners["candidate"] == "context-PCG").sum())
            if not stress_winners.empty
            else 0
        ),
        "cg_stress_angular_best_count": (
            int((stress_winners["candidate"] == "angular-Jacobi-PCG").sum())
            if not stress_winners.empty
            else 0
        ),
        "cg_stress_hybrid_best_count": (
            int((stress_winners["candidate"] == "hybrid-PCG").sum())
            if not stress_winners.empty
            else 0
        ),
        "cg_stress_population_best_count": (
            int((stress_winners["candidate"] == "population-PCG").sum())
            if not stress_winners.empty
            else 0
        ),
        "cg_stress_hb_best_count": (
            int((stress_winners["candidate"] == "looped-HB").sum())
            if not stress_winners.empty
            else 0
        ),
        "geometry_generalization_rows": int(len(geometry_raw)),
        "geometry_bound_min_batches": (
            int(geometry_bounds["n_geometry_batches"].min())
            if not geometry_bounds.empty
            else 0
        ),
        "geometry_bound_max_slack": (
            float(geometry_bounds["hoeffding_slack"].max())
            if not geometry_bounds.empty
            else float("nan")
        ),
        "geometry_cg_to_context_mean_gain": (
            geometry_method_mean("identity-CG", "original_mean_relative_residual")
            / geometry_method_mean("context-PCG", "original_mean_relative_residual")
            if not geometry_largest.empty
            else float("nan")
        ),
        "geometry_context_worst_scenario_gain": (
            float(geometry_scenario_gain.min())
            if not geometry_scenario_gain.empty
            else float("nan")
        ),
        "geometry_seed_scenario_count": int(len(geometry_seed_scenario_gain)),
        "geometry_worst_seed_scenario_gain": (
            float(geometry_seed_scenario_gain.min())
            if not geometry_seed_scenario_gain.empty
            else float("nan")
        ),
        "geometry_cg_coverage": geometry_method_mean(
            "identity-CG", "numerical_coverage_95"
        ),
        "geometry_context_coverage": geometry_method_mean(
            "context-PCG", "numerical_coverage_95"
        ),
        "geometry_angular_mean_residual": geometry_method_mean(
            "angular-Jacobi-PCG", "original_mean_relative_residual"
        ),
        "geometry_hybrid_mean_residual": geometry_method_mean(
            "hybrid-PCG", "original_mean_relative_residual"
        ),
        "geometry_context_mean_residual": geometry_method_mean(
            "context-PCG", "original_mean_relative_residual"
        ),
        "geometry_angular_to_context_mean_ratio": (
            geometry_method_mean(
                "angular-Jacobi-PCG", "original_mean_relative_residual"
            )
            / geometry_method_mean("context-PCG", "original_mean_relative_residual")
            if not geometry_largest.empty
            else float("nan")
        ),
        "geometry_angular_to_context_worst_scenario_ratio": (
            float(geometry_angular_to_context_scenario_ratio.min())
            if not geometry_angular_to_context_scenario_ratio.empty
            else float("nan")
        ),
        "geometry_angular_coverage": geometry_method_mean(
            "angular-Jacobi-PCG", "numerical_coverage_95"
        ),
        "geometry_hybrid_coverage": geometry_method_mean(
            "hybrid-PCG", "numerical_coverage_95"
        ),
        "geometry_paired_context_ap_delta": geometry_effect_metric(
            "CG / context-PCG", "average_precision_delta_mean"
        ),
        "geometry_paired_context_ap_delta_ci_lower": geometry_effect_metric(
            "CG / context-PCG", "average_precision_delta_ci95_lower"
        ),
        "geometry_paired_context_ap_delta_ci_upper": geometry_effect_metric(
            "CG / context-PCG", "average_precision_delta_ci95_upper"
        ),
        "geometry_paired_context_coverage_delta": geometry_effect_metric(
            "CG / context-PCG", "coverage_delta_mean"
        ),
        "geometry_paired_context_coverage_delta_ci_lower": geometry_effect_metric(
            "CG / context-PCG", "coverage_delta_ci95_lower"
        ),
        "geometry_paired_context_coverage_delta_ci_upper": geometry_effect_metric(
            "CG / context-PCG", "coverage_delta_ci95_upper"
        ),
        "geometry_paired_angular_to_context_gain": geometry_effect_metric(
            "angular-Jacobi-PCG / context-PCG", "geometric_mean_gain"
        ),
        "geometry_paired_angular_to_context_gain_ci_lower": geometry_effect_metric(
            "angular-Jacobi-PCG / context-PCG", "gain_ci95_lower"
        ),
        "geometry_paired_angular_to_context_gain_ci_upper": geometry_effect_metric(
            "angular-Jacobi-PCG / context-PCG", "gain_ci95_upper"
        ),
        "geometry_paired_context_win_rate_vs_angular": geometry_effect_metric(
            "angular-Jacobi-PCG / context-PCG", "candidate_win_rate"
        ),
        "geometry_paired_context_win_rate_ci_lower": geometry_effect_metric(
            "angular-Jacobi-PCG / context-PCG", "win_rate_ci95_lower"
        ),
        "geometry_paired_context_win_rate_ci_upper": geometry_effect_metric(
            "angular-Jacobi-PCG / context-PCG", "win_rate_ci95_upper"
        ),
        "largest_context_population_condition_gain": largest_context_condition_metric(
            "population-PCG", "condition_reduction_median"
        ),
        "largest_context_context_condition_gain": largest_context_condition_metric(
            "context-PCG", "condition_reduction_median"
        ),
        "largest_context_hybrid_condition_gain": largest_context_condition_metric(
            "hybrid-PCG", "condition_reduction_median"
        ),
        "largest_context_raw_condition": largest_context_condition_metric(
            "identity-CG", "raw_condition_median"
        ),
        "largest_context_context_condition": largest_context_condition_metric(
            "context-PCG", "transformed_condition_median"
        ),
        "largest_context_hybrid_condition": largest_context_condition_metric(
            "hybrid-PCG", "transformed_condition_median"
        ),
        "largest_context_geometry_commutator": largest_context_condition_metric(
            "identity-CG", "geometry_commutator_mean"
        ),
        "protocol_finished_requested_grid": bool(protocol["finished_requested_grid"]),
        "residual_coordinate_convention": {
            "training_grid": (
                "mean_relative_residual and covariance_relative_residual in "
                "evaluation.csv are transformed-coordinate residuals r_tr"
            ),
            "post_training_audits": (
                "reported mean/covariance residual fields in aggregated depth, "
                "classical, geometry, stress, and joint audits are physical-coordinate "
                "residuals r_phys; transformed_* fields retain r_tr"
            ),
            "direct_cg_claims": (
                "all direct CG gains in cg_comparison_note.pdf use r_phys"
            ),
        },
    }
    transformed_training_grid_keys = (
        "cg_to_pcg_mean_residual_gain",
        "cg_to_pcg_covariance_residual_gain",
        "cg_to_context_pcg_mean_residual_gain",
        "cg_to_context_pcg_covariance_residual_gain",
        "cg_to_hybrid_pcg_mean_residual_gain",
        "cg_to_hybrid_pcg_covariance_residual_gain",
        "macro_cg_to_pcg_mean_residual_gain",
        "macro_cg_to_pcg_covariance_residual_gain",
        "largest_context_mean_residual_gain",
        "largest_context_covariance_residual_gain",
        "macro_cg_to_context_pcg_mean_residual_gain",
        "macro_cg_to_context_pcg_covariance_residual_gain",
        "macro_cg_to_hybrid_pcg_mean_residual_gain",
        "macro_cg_to_hybrid_pcg_covariance_residual_gain",
        "largest_context_context_pcg_mean_residual_gain",
        "largest_context_hybrid_pcg_mean_residual_gain",
        "context_worst_scenario_mean_gain",
        "context_median_scenario_mean_gain",
        "context_worst_seed_scenario_gain",
        "context_largest_to_best_residual_ratio",
        "hybrid_best_width_mean_residual",
        "hybrid_best_width_covariance_residual",
    )
    serialized_summary = dict(summary)
    serialized_summary["transformed_training_grid_metrics"] = {
        key: serialized_summary.pop(key) for key in transformed_training_grid_keys
    }
    (results_dir / "scaling_summary.json").write_text(
        json.dumps(serialized_summary, indent=2), encoding="utf-8"
    )
    comparison.rename(
        columns={
            "mean_relative_residual_mean": (
                "transformed_mean_relative_residual_mean"
            ),
            "mean_relative_residual_std": "transformed_mean_relative_residual_std",
            "covariance_relative_residual_mean": (
                "transformed_covariance_relative_residual_mean"
            ),
            "covariance_relative_residual_std": (
                "transformed_covariance_relative_residual_std"
            ),
            "mean_relative_residual_ci95": (
                "transformed_mean_relative_residual_ci95"
            ),
            "covariance_relative_residual_ci95": (
                "transformed_covariance_relative_residual_ci95"
            ),
        }
    ).to_csv(results_dir / "final_cg_comparison.csv", index=False)
    scenario_comparison.to_csv(
        results_dir / "scenario_cg_comparison.csv", index=False
    )
    tex_path = build_tex(
        results_dir,
        protocol,
        comparison,
        scenario_comparison,
        fits,
        bounds,
        summary,
    )
    if not args.skip_pdf:
        for _ in range(2):
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    tex_path.name,
                ],
                cwd=results_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        shutil.copyfile(
            tex_path.with_suffix(".pdf"),
            results_dir / "results_note_english.pdf",
        )
    print(json.dumps(serialized_summary, indent=2))


if __name__ == "__main__":
    main()
