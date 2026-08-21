#!/usr/bin/env python3
"""Strict structural and numerical audit for the completed scaling artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def deduplicated(path: Path, keys: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.drop_duplicates(keys, keep="last")


def finite_count(frame: pd.DataFrame) -> int:
    numeric = frame.select_dtypes(include=[np.number])
    return int((~np.isfinite(numeric.to_numpy())).sum())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    results = args.results_dir
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    evaluation = deduplicated(
        results / "evaluation.csv",
        [
            "seed",
            "method",
            "network_width",
            "dataset_size",
            "context_size",
            "scenario",
            "task",
        ],
    )
    baselines = deduplicated(
        results / "baselines.csv",
        ["seed", "method", "context_size", "scenario", "task"],
    )
    training = deduplicated(
        results / "training.csv",
        ["seed", "method", "network_width", "dataset_size"],
    )
    runtime = deduplicated(
        results / "runtime.csv",
        ["seed", "method", "network_width", "context_size"],
    )
    seeds = len(protocol["seeds"])
    widths = len(protocol["network_widths"])
    methods = len(protocol["learned_methods"])
    contexts = len(protocol["evaluation_context_sizes"])
    datasets = len(protocol["dataset_sizes"])
    scenarios = len(protocol["scenarios"])
    intermediate_tasks = int(protocol["intermediate_eval_tasks_per_seed_scenario"])
    final_tasks = int(protocol["final_eval_tasks_per_seed_scenario"])
    expected_training = seeds * widths * methods * datasets
    expected_evaluation = seeds * widths * methods * contexts * scenarios * (
        (datasets - 1) * intermediate_tasks + final_tasks
    )
    expected_baselines = seeds * 2 * contexts * scenarios * final_tasks
    expected_runtime = seeds * (widths * methods + 2) * contexts
    checkpoint_count = len(list((results / "checkpoints").glob("*.pt")))
    expected_checkpoints = seeds * widths * methods
    checks = {
        "protocol_finished": bool(protocol["finished_requested_grid"]),
        "training_rows": len(training) == expected_training,
        "evaluation_rows": len(evaluation) == expected_evaluation,
        "baseline_rows": len(baselines) == expected_baselines,
        "runtime_rows": len(runtime) == expected_runtime,
        "checkpoint_count": checkpoint_count == expected_checkpoints,
        "no_skipped_updates": int(training["skipped_updates"].max()) == 0,
        "finite_training": finite_count(training) == 0,
        "finite_evaluation": finite_count(evaluation) == 0,
        "finite_baselines": finite_count(baselines) == 0,
        "finite_runtime": finite_count(runtime) == 0,
    }
    report: dict[str, object] = {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "training_rows": len(training),
            "evaluation_rows": len(evaluation),
            "baseline_rows": len(baselines),
            "runtime_rows": len(runtime),
            "checkpoints": checkpoint_count,
        },
        "expected": {
            "training_rows": expected_training,
            "evaluation_rows": expected_evaluation,
            "baseline_rows": expected_baselines,
            "runtime_rows": expected_runtime,
            "checkpoints": expected_checkpoints,
        },
        "nonfinite": {
            "training": finite_count(training),
            "evaluation": finite_count(evaluation),
            "baselines": finite_count(baselines),
            "runtime": finite_count(runtime),
        },
    }
    depth_values = 10
    audit_tasks = intermediate_tasks
    expected_depth = seeds * (
        (methods + 1) * depth_values * contexts * scenarios * audit_tasks
        + contexts * scenarios * audit_tasks
    )
    expected_depth_runtime = seeds * (
        (methods + 1) * depth_values * contexts + contexts
    )
    optional_specs = {
        "depth_scaling.csv": {
            "keys": [
                "seed",
                "method",
                "network_width",
                "depth",
                "context_size",
                "scenario",
                "task",
            ],
            "expected": expected_depth,
        },
        "depth_runtime.csv": {
            "keys": ["seed", "method", "network_width", "depth", "context_size"],
            "expected": expected_depth_runtime,
        },
        "preconditioner_conditioning.csv": {
            "keys": [
                "seed",
                "method",
                "context_size",
                "scenario",
                "task",
            ],
            "expected": seeds * 4 * contexts * scenarios * audit_tasks,
        },
        "geometry_generalization.csv": {
            "keys": [
                "seed",
                "geometry_draw",
                "method",
                "context_size",
                "scenario",
            ],
            "expected": seeds * 64 * 6 * 3 * 4,
        },
        "cg_stress_sweep.csv": {
            "keys": ["seed", "axis", "level", "geometry_draw", "method"],
            "expected": seeds * 5 * 4 * 64 * 7,
        },
        "cg_stress_comparison.csv": {
            "keys": ["axis", "level", "candidate"],
            "expected": 5 * 4 * 5,
        },
        "joint_conditioning.csv": {
            "keys": ["seed", "joint_severity", "geometry_draw", "method"],
            "expected": seeds * 4 * 64 * 5,
        },
        "joint_conditioning_aggregated.csv": {
            "keys": ["joint_severity", "method"],
            "expected": 4 * 5,
        },
        "geometry_method_comparison.csv": {
            "keys": ["scenario", "method"],
            "expected": 4 * 6,
        },
        "paired_geometry_comparison.csv": {
            "keys": ["scenario", "comparison"],
            "expected": 5 * 4,
        },
        "tolerance_frontier.csv": {
            "keys": ["method", "context_size", "tolerance"],
            "expected": 11 * contexts * 3,
        },
        "best_method_by_context_tolerance.csv": {
            "keys": ["context_size", "tolerance"],
            "expected": contexts * 3,
        },
        "scenario_cg_comparison.csv": {
            "keys": ["scenario"],
            "expected": scenarios,
        },
        "final_cg_comparison.csv": {
            "keys": ["method"],
            "expected": methods + 2,
        },
        "cg_gain_by_context.csv": {
            "keys": ["method", "context_size"],
            "expected": methods * contexts,
        },
        "cg_gain_by_depth.csv": {
            "keys": ["method", "depth", "context_size"],
            "expected": methods * depth_values * contexts,
        },
        "classical_preconditioners.csv": {
            "keys": [
                "seed",
                "method",
                "depth",
                "context_size",
                "scenario",
                "task",
            ],
            "expected": seeds * 4 * depth_values * contexts * scenarios * audit_tasks,
        },
        "classical_preconditioner_runtime.csv": {
            "keys": ["seed", "method", "depth", "context_size"],
            "expected": seeds * 4 * depth_values * contexts,
        },
    }
    optional = {}
    for name, spec in optional_specs.items():
        path = results / name
        if not path.exists():
            optional[name] = {"missing": True, "expected_rows": spec["expected"]}
            report["passed"] = False
            continue
        frame = deduplicated(path, spec["keys"])
        complete = len(frame) == spec["expected"]
        optional[name] = {
            "rows": len(frame),
            "expected_rows": spec["expected"],
            "complete": complete,
            "nonfinite": finite_count(frame),
        }
        if finite_count(frame) != 0 or not complete:
            report["passed"] = False
    report["optional_artifacts"] = optional
    summary = json.loads(
        (results / "scaling_summary.json").read_text(encoding="utf-8")
    )
    final_columns = set(
        pd.read_csv(results / "final_cg_comparison.csv", nrows=0).columns
    )
    scenario_columns = set(
        pd.read_csv(results / "scenario_cg_comparison.csv", nrows=0).columns
    )
    context_gain_columns = set(
        pd.read_csv(results / "cg_gain_by_context.csv", nrows=0).columns
    )
    depth_gain_columns = set(
        pd.read_csv(results / "cg_gain_by_depth.csv", nrows=0).columns
    )
    conditioning_columns = set(
        pd.read_csv(
            results / "preconditioner_conditioning_aggregated.csv", nrows=0
        ).columns
    )
    aggregate_columns = set(
        pd.read_csv(results / "scaling_aggregated.csv", nrows=0).columns
    )
    bound_columns = set(
        pd.read_csv(results / "generalization_bounds.csv", nrows=0).columns
    )
    manuscript_tex = (results / "near_field_scaling_note.tex").read_text(
        encoding="utf-8"
    )
    cg_note_tex = (results / "cg_comparison_note.tex").read_text(encoding="utf-8")
    coordinate_checks = {
        "coordinate_contract_recorded": bool(
            summary.get("residual_coordinate_convention")
        ),
        "transformed_grid_metrics_namespaced": bool(
            summary.get("transformed_training_grid_metrics")
        ),
        "final_grid_comparison_labels_transformed_residuals": {
            "transformed_mean_relative_residual_mean",
            "transformed_covariance_relative_residual_mean",
        }.issubset(final_columns),
        "scenario_comparison_labels_transformed_residuals": {
            "cg_transformed_mean_residual",
            "context_transformed_mean_residual",
            "transformed_mean_residual_gain",
            "transformed_covariance_residual_gain",
        }.issubset(scenario_columns),
        "context_gain_labels_transformed_residuals": {
            "transformed_mean_residual_gain",
            "transformed_covariance_residual_gain",
        }.issubset(context_gain_columns),
        "depth_gain_labels_physical_residuals": {
            "physical_mean_residual_gain",
            "physical_covariance_residual_gain",
        }.issubset(depth_gain_columns),
        "conditioning_audit_labels_transformed_residuals": {
            "transformed_mean_residual_median",
            "transformed_covariance_residual_median",
        }.issubset(conditioning_columns),
        "scaling_aggregate_labels_transformed_residuals": {
            "transformed_mean_relative_residual_mean",
            "transformed_covariance_relative_residual_mean",
        }.issubset(aggregate_columns),
        "generalization_bound_labels_transformed_residuals": {
            "transformed_mean_relative_residual",
            "transformed_covariance_relative_residual",
        }.issubset(bound_columns),
        "manuscript_abstract_uses_physical_cg_gain": (
            "physical residual by $2.05\\times$" in manuscript_tex
            and "$6.44\\times$" in manuscript_tex
        ),
        "manuscript_omits_mixed_coordinate_ratio": "3.540" not in manuscript_tex,
        "manuscript_disclaims_transformed_as_physical_gain": (
            "these ratios are not interpreted as physical solver gains"
            in manuscript_tex
        ),
        "cg_note_uses_physical_coordinate_claim": (
            "physical-coordinate posterior-mean residual" in cg_note_tex
            and "$2.05\\times$" in cg_note_tex
        ),
    }
    report["checks"].update(coordinate_checks)
    if not all(coordinate_checks.values()):
        report["passed"] = False
    stress = deduplicated(
        results / "cg_stress_sweep.csv",
        ["seed", "axis", "level", "geometry_draw", "method"],
    )
    stress_protocol = json.loads(
        (results / "cg_stress_protocol.json").read_text(encoding="utf-8")
    )
    stress_group_sizes = stress.groupby(["axis", "level", "method"]).size()
    stress_checks = {
        "stress_tasks_per_geometry": set(stress["tasks_per_geometry"]) == {4},
        "stress_context": set(stress["context_size"]) == {48},
        "stress_seven_methods": stress["method"].nunique() == 7,
        "stress_192_batches_per_level_method": bool(
            (stress_group_sizes == seeds * 64).all()
        ),
        "stress_protocol": (
            stress_protocol["geometry_draws"] == 64
            and stress_protocol["tasks_per_geometry"] == 4
            and len(stress_protocol["stress_levels"]) == 20
            and len(stress_protocol["methods"]) == 7
        ),
    }
    joint = deduplicated(
        results / "joint_conditioning.csv",
        ["seed", "joint_severity", "geometry_draw", "method"],
    )
    joint_group_sizes = joint.groupby(["joint_severity", "method"]).size()
    stress_joint = stress[stress["axis"] == "joint_severity"].rename(
        columns={
            "level": "joint_severity",
            "mean_relative_residual": "stress_residual",
        }
    )
    joint_consistency = stress_joint.merge(
        joint,
        on=["seed", "joint_severity", "geometry_draw", "method"],
        how="inner",
    )
    stress_checks.update(
        {
            "joint_conditioning_tasks_per_geometry": set(
                joint["tasks_per_geometry"]
            )
            == {4},
            "joint_conditioning_192_batches": bool(
                (joint_group_sizes == seeds * 64).all()
            ),
            "joint_conditioning_matches_stress_tasks": (
                len(joint_consistency) == seeds * 4 * 64 * 5
                and float(
                    np.abs(
                        joint_consistency["stress_residual"]
                        - joint_consistency["mean_relative_residual_mean"]
                    ).max()
                )
                < 2.0e-6
            ),
        }
    )
    depth = deduplicated(
        results / "depth_scaling.csv",
        [
            "seed",
            "method",
            "network_width",
            "depth",
            "context_size",
            "scenario",
            "task",
        ],
    )
    matched = depth[(depth["context_size"] == 48) & (depth["depth"] == 32)]
    matched_means = matched.groupby("method").agg(
        mean_residual=("original_mean_relative_residual", "mean"),
        transformed_mean_residual=("mean_relative_residual", "mean"),
        score_error=("relative_score_error", "mean"),
        average_precision=("average_precision", "mean"),
        numerical_coverage=("numerical_coverage_95", "mean"),
    )
    cg = matched_means.loc["identity-CG"]
    context = matched_means.loc["context-PCG"]
    hb = matched_means.loc["looped-HB"]
    stress_effects = deduplicated(
        results / "cg_stress_comparison.csv",
        ["axis", "level", "candidate"],
    )
    context_stress = stress_effects[
        stress_effects["candidate"] == "context-PCG"
    ]
    geometry = deduplicated(
        results / "geometry_generalization.csv",
        ["seed", "geometry_draw", "method", "context_size", "scenario"],
    )
    geometry_pivot = geometry[geometry["context_size"] == 48].pivot(
        index=["seed", "geometry_draw", "scenario"],
        columns="method",
        values="original_mean_relative_residual",
    )
    claim_checks = {
        "matched_depth_context_residual_beats_cg": bool(
            context["mean_residual"] < cg["mean_residual"]
        ),
        "matched_depth_context_score_error_beats_cg": bool(
            context["score_error"] < cg["score_error"]
        ),
        "matched_depth_context_coverage_beats_cg": bool(
            context["numerical_coverage"] > cg["numerical_coverage"]
        ),
        "matched_depth_reported_numbers_match_raw_data": (
            round(float(cg["mean_residual"] / context["mean_residual"]), 2) == 2.05
            and round(float(cg["score_error"] / context["score_error"]), 2)
            == 6.44
            and round(float(cg["numerical_coverage"]), 3) == 0.243
            and round(float(context["numerical_coverage"]), 3) == 0.873
        ),
        "matched_depth_hb_reconstruction_worse_than_cg": bool(
            hb["average_precision"] < cg["average_precision"]
        ),
        "stress_context_lower_ci_beats_cg_all_levels": bool(
            len(context_stress) == 20
            and (context_stress["gain_ci95_lower"] > 1.0).all()
        ),
        "independent_geometry_context_beats_cg_at_least_95_percent": bool(
            len(geometry_pivot) == seeds * 64 * 4
            and float(
                (
                    geometry_pivot["identity-CG"]
                    > geometry_pivot["context-PCG"]
                ).mean()
            )
            >= 0.95
        ),
        "english_pdf_matches_full_pdf": (
            sha256_file(results / "results_note_english.pdf")
            == sha256_file(results / "near_field_scaling_note.pdf")
        ),
    }
    report["claim_values"] = {
        "matched_depth_cg_residual": float(cg["mean_residual"]),
        "matched_depth_context_residual": float(context["mean_residual"]),
        "matched_depth_context_transformed_residual": float(
            context["transformed_mean_residual"]
        ),
        "matched_depth_residual_gain": float(
            cg["mean_residual"] / context["mean_residual"]
        ),
        "matched_depth_cg_score_error": float(cg["score_error"]),
        "matched_depth_context_score_error": float(context["score_error"]),
        "matched_depth_cg_coverage": float(cg["numerical_coverage"]),
        "matched_depth_context_coverage": float(context["numerical_coverage"]),
        "stress_min_context_gain_ci95_lower": float(
            context_stress["gain_ci95_lower"].min()
        ),
        "independent_geometry_pairs": int(len(geometry_pivot)),
        "independent_geometry_context_win_rate": float(
            (
                geometry_pivot["identity-CG"]
                > geometry_pivot["context-PCG"]
            ).mean()
        ),
    }
    stress_checks.update(claim_checks)
    report["checks"].update(stress_checks)
    if not all(stress_checks.values()):
        report["passed"] = False
    render_names = (
        "scaling_dataset.png",
        "scaling_context.png",
        "scaling_network.png",
        "scaling_time.png",
        "scaling_runtime.png",
        "scaling_generalization_bounds.png",
        "scaling_depth.png",
        "scaling_conditioning.png",
        "scaling_classical_preconditioners.png",
        "scaling_tolerance_frontier.png",
        "scaling_geometry_generalization.png",
        "scaling_cg_comparison.png",
        "scaling_cg_stress_test.png",
        "scaling_joint_conditioning.png",
        "scaling_joint_reconstructions.png",
        "scaling_reconstructions.png",
        "cg_comparison_note.pdf",
        "near_field_scaling_note.pdf",
        "results_note_english.pdf",
    )
    render_artifacts = {}
    for name in render_names:
        path = results / name
        size = path.stat().st_size if path.exists() else 0
        valid = size >= 10_000
        render_artifacts[name] = {"bytes": size, "valid": valid}
        if not valid:
            report["passed"] = False
    report["render_artifacts"] = render_artifacts
    hash_names = (
        "protocol.json",
        "training.csv",
        "evaluation.csv",
        "baselines.csv",
        "runtime.csv",
        "depth_scaling.csv",
        "preconditioner_conditioning.csv",
        "classical_preconditioners.csv",
        "geometry_generalization.csv",
        "cg_stress_sweep.csv",
        "cg_stress_protocol.json",
        "joint_conditioning.csv",
        "cg_comparison_note.pdf",
        "near_field_scaling_note.pdf",
        "results_note_english.pdf",
    )
    hashes = {name: sha256_file(results / name) for name in hash_names}
    checkpoint_digest = hashlib.sha256()
    for path in sorted((results / "checkpoints").glob("*.pt")):
        checkpoint_digest.update(path.name.encode("utf-8"))
        checkpoint_digest.update(sha256_file(path).encode("ascii"))
    hashes["checkpoints_tree"] = checkpoint_digest.hexdigest()
    report["sha256"] = hashes
    (results / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
