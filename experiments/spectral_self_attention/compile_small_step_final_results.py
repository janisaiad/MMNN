"""Compile all headline results of the small-step audit into one JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from experiments.spectral_self_attention.summarize_small_step_audit import (
    wilson_interval,
)


LABELS = ("p3", "p4", "chaos")
MOTION_THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def direct_ode_table(
    data_dir: Path, suffix: str = ""
) -> tuple[list[dict[str, object]], list[dict]]:
    rows = []
    all_records = []
    for family in (1, 2, 3, 4):
        records = []
        for label in LABELS:
            payload = json.loads(
                (
                    data_dir
                    / f"continuous_ode_f{family}_{label}{suffix}.json"
                ).read_text()
            )
            records.extend(payload["records"])
        metrics = [record["metrics"] for record in records]
        moving = [metric["motion_per_normalized_time"] >= 1e-3 for metric in metrics]
        internal = [
            is_moving and metric["gram_variation"] >= 1e-3
            for is_moving, metric in zip(moving, metrics)
        ]
        moving_count = sum(moving)
        rows.append(
            {
                "family": family,
                "records": len(records),
                "fixed": sum(metric["fixed"] for metric in metrics),
                "moving": moving_count,
                "moving_fraction": moving_count / len(records),
                "moving_wilson_95": wilson_interval(moving_count, len(records)),
                "rigid_rotation": sum(
                    is_moving and not is_internal
                    for is_moving, is_internal in zip(moving, internal)
                ),
                "internal_shape_motion": sum(internal),
            }
        )
        all_records.extend(records)
    return rows, all_records


def stratified_ode_table(
    records: list[dict], identity_key: str
) -> list[dict[str, object]]:
    """Summarize continuous fates without pooling token count or subtype."""
    rows = []
    levels = sorted({int(record["identity"][identity_key]) for record in records})
    for level in levels:
        selected = [
            record
            for record in records
            if int(record["identity"][identity_key]) == level
        ]
        moving = [
            record["metrics"]["motion_per_normalized_time"] >= 1e-3
            for record in selected
        ]
        internal = [
            is_moving and record["metrics"]["gram_variation"] >= 1e-3
            for record, is_moving in zip(selected, moving)
        ]
        moving_count = sum(moving)
        rows.append(
            {
                identity_key: level,
                "records": len(selected),
                "moving": moving_count,
                "moving_fraction": moving_count / len(selected),
                "moving_wilson_95": wilson_interval(moving_count, len(selected)),
                "rigid_rotation": sum(
                    is_moving and not is_internal
                    for is_moving, is_internal in zip(moving, internal)
                ),
                "internal_shape_motion": sum(internal),
                "positive_lyapunov": sum(
                    record["metrics"]["positive_lyapunov"] for record in selected
                ),
            }
        )
    return rows


def record_key(record: dict) -> tuple[int, str, int, int, int]:
    identity = record["identity"]
    return (
        int(record["family"]),
        str(record["label"]),
        int(identity["n_tokens"]),
        int(identity["subtype_code"]),
        int(identity["source_model_index"]),
    )


def paired_noise_sensitivity(
    baseline: list[dict], stronger_noise: list[dict]
) -> dict[str, object]:
    """Compare fates record by record after increasing the symmetry-breaking noise."""
    baseline_by_key = {record_key(record): record for record in baseline}
    stronger_by_key = {record_key(record): record for record in stronger_noise}
    if baseline_by_key.keys() != stronger_by_key.keys():
        raise ValueError("baseline and stronger-noise records do not match")
    transition = {
        "fixed_to_fixed": 0,
        "fixed_to_moving": 0,
        "moving_to_fixed": 0,
        "moving_to_moving": 0,
    }
    for key, baseline_record in baseline_by_key.items():
        baseline_moving = (
            baseline_record["metrics"]["motion_per_normalized_time"] >= 1e-3
        )
        stronger_moving = (
            stronger_by_key[key]["metrics"]["motion_per_normalized_time"] >= 1e-3
        )
        transition[
            f"{'moving' if baseline_moving else 'fixed'}_to_"
            f"{'moving' if stronger_moving else 'fixed'}"
        ] += 1
    agreement = transition["fixed_to_fixed"] + transition["moving_to_moving"]
    return {
        "records": len(baseline_by_key),
        "transition_counts": transition,
        "agreement_fraction": agreement / len(baseline_by_key),
    }


def detected_model_prevalence(
    data_dir: Path, records: list[dict], token_counts: tuple[int, ...]
) -> list[dict[str, object]]:
    """Lower-bound prevalence: sampled models with at least one selected ODE fate."""
    rows = []
    for family in (1, 2, 3, 4):
        sampled = sum(
            int(
                json.loads(
                    (data_dir / f"small_step_cohort_f{family}_n{n_tokens}.json").read_text()
                )["settings"]["models"]
            )
            for n_tokens in token_counts
        )
        selected = [record for record in records if int(record["family"]) == family]
        model_groups: dict[tuple[int, int], list[dict]] = {}
        for record in selected:
            identity = record["identity"]
            key = (
                int(identity["n_tokens"]),
                int(identity["source_model_index"]),
            )
            model_groups.setdefault(key, []).append(record)
        moving_models = sum(
            any(
                record["metrics"]["motion_per_normalized_time"] >= 1e-3
                for record in group
            )
            for group in model_groups.values()
        )
        internal_models = sum(
            any(
                record["metrics"]["motion_per_normalized_time"] >= 1e-3
                and record["metrics"]["gram_variation"] >= 1e-3
                for record in group
            )
            for group in model_groups.values()
        )
        rows.append(
            {
                "family": family,
                "sampled_models": sampled,
                "models_with_selected_complex_map_fate": len(model_groups),
                "models_with_detected_continuous_motion": moving_models,
                "detected_continuous_motion_fraction": moving_models / sampled,
                "detected_continuous_motion_wilson_95": wilson_interval(
                    moving_models, sampled
                ),
                "models_with_detected_internal_motion": internal_models,
            }
        )
    return rows


def threshold_sensitivity(records: list[dict]) -> list[dict[str, object]]:
    """Expose the dependence of headline counts on the motion cutoff."""
    rows = []
    for threshold in MOTION_THRESHOLDS:
        rows.append(
            {
                "motion_threshold": threshold,
                "total": len(records),
                "moving": sum(
                    record["metrics"]["motion_per_normalized_time"] >= threshold
                    for record in records
                ),
                "internal": sum(
                    record["metrics"]["motion_per_normalized_time"] >= threshold
                    and record["metrics"]["gram_variation"] >= threshold
                    for record in records
                ),
                "by_family": {
                    str(family): sum(
                        int(record["family"]) == family
                        and record["metrics"]["motion_per_normalized_time"]
                        >= threshold
                        for record in records
                    )
                    for family in (1, 2, 3, 4)
                },
            }
        )
    return rows


def continuation_plateau(
    data_dir: Path, ode_records: list[dict]
) -> dict[str, object]:
    """Compare the deepest finite-layer continuation with the direct ODE replay."""
    selected_ratios = (
        1.0,
        0.25,
        0.0625,
        0.015625,
        0.00390625,
        0.001953125,
        0.0009765625,
        0.00048828125,
        0.000244140625,
    )
    counts = {ratio: 0 for ratio in selected_ratios}
    total = 0
    endpoint_by_key: dict[tuple[int, str, int, int, int], bool] = {}
    by_family = {
        family: {"records": 0, "endpoint_moving": 0} for family in (1, 2, 3, 4)
    }
    for family in (1, 2, 3, 4):
        for label in LABELS:
            deep_path = (
                data_dir / f"small_step_extension_deep_f{family}_{label}.json"
            )
            source_path = (
                deep_path
                if deep_path.exists()
                else data_dir / f"small_step_extension_f{family}_{label}.json"
            )
            payload = json.loads(source_path.read_text())
            ratios = [float(value) for value in payload["settings"]["ratios"]]
            ratio_indices = {
                ratio: next(
                    index
                    for index, value in enumerate(ratios)
                    if np.isclose(value, ratio, rtol=0.0, atol=1e-12)
                )
                for ratio in selected_ratios
                if any(
                    np.isclose(value, ratio, rtol=0.0, atol=1e-12)
                    for value in ratios
                )
            }
            for record in payload["records"]:
                total += 1
                by_family[family]["records"] += 1
                for ratio, index in ratio_indices.items():
                    counts[ratio] += int(
                        record["trace"][index]["motion_per_normalized_time"] >= 1e-3
                    )
                identity = record["identity"]
                key = (
                    family,
                    label,
                    int(identity["n_tokens"]),
                    int(identity["subtype_code"]),
                    int(identity["source_model_index"]),
                )
                endpoint_moving = bool(
                    record["trace"][-1]["motion_per_normalized_time"] >= 1e-3
                )
                endpoint_by_key[key] = endpoint_moving
                by_family[family]["endpoint_moving"] += int(endpoint_moving)

    ode_by_key = {record_key(record): record for record in ode_records}
    if endpoint_by_key.keys() != ode_by_key.keys():
        raise ValueError("deep continuation and direct ODE records do not match")
    transitions = {
        "finite_fixed_to_ode_fixed": 0,
        "finite_fixed_to_ode_moving": 0,
        "finite_moving_to_ode_fixed": 0,
        "finite_moving_to_ode_moving": 0,
    }
    for key, finite_moving in endpoint_by_key.items():
        ode_moving = bool(
            ode_by_key[key]["metrics"]["motion_per_normalized_time"] >= 1e-3
        )
        transitions[
            f"finite_{'moving' if finite_moving else 'fixed'}_to_ode_"
            f"{'moving' if ode_moving else 'fixed'}"
        ] += 1
    agreement = (
        transitions["finite_fixed_to_ode_fixed"]
        + transitions["finite_moving_to_ode_moving"]
    )
    return {
        "records": total,
        "survival_curve": [
            {
                "step_ratio": ratio,
                "moving": counts[ratio],
                "moving_fraction": counts[ratio] / total,
            }
            for ratio in selected_ratios
            if counts[ratio] > 0 or np.isclose(ratio, 1.0)
        ],
        "endpoint_by_family": [
            {
                "family": family,
                **values,
                "endpoint_moving_fraction": values["endpoint_moving"]
                / values["records"],
            }
            for family, values in by_family.items()
        ],
        "endpoint_vs_direct_ode": {
            "transition_counts": transitions,
            "agreement_fraction": agreement / total,
        },
    }


def harvest_scale(data_dir: Path) -> dict[str, int]:
    models = trajectories = 0
    for family in (1, 2, 3, 4):
        for n_tokens in (1, 2, 3, 4):
            payload = json.loads(
                (data_dir / f"small_step_cohort_f{family}_n{n_tokens}.json").read_text()
            )
            models += int(payload["settings"]["models"])
            trajectories += int(payload["settings"]["models"]) * int(
                payload["settings"]["basins"]
            )
    control_models = control_trajectories = 0
    for n_tokens in (1, 2, 3, 4):
        payload = json.loads(
            (data_dir / f"small_step_cohort_beta0_f1_n{n_tokens}.json").read_text()
        )
        control_models += int(payload["settings"]["models"])
        control_trajectories += int(payload["settings"]["models"]) * int(
            payload["settings"]["basins"]
        )
    high_n_models = high_n_trajectories = 0
    for family in (1, 2, 3, 4):
        for n_tokens in (8, 16):
            payload = json.loads(
                (data_dir / f"small_step_cohort_f{family}_n{n_tokens}.json").read_text()
            )
            high_n_models += int(payload["settings"]["models"])
            high_n_trajectories += int(payload["settings"]["models"]) * int(
                payload["settings"]["basins"]
            )
    beta0_type3_models = beta0_type3_trajectories = 0
    for n_tokens in (1, 2, 3, 4):
        payload = json.loads(
            (data_dir / f"small_step_cohort_beta0_f3_n{n_tokens}.json").read_text()
        )
        beta0_type3_models += int(payload["settings"]["models"])
        beta0_type3_trajectories += int(payload["settings"]["models"]) * int(
            payload["settings"]["basins"]
        )
    return {
        "main_models": models,
        "main_trajectories": trajectories,
        "control_models": control_models,
        "control_trajectories": control_trajectories,
        "high_token_models": high_n_models,
        "high_token_trajectories": high_n_trajectories,
        "beta0_type3_models": beta0_type3_models,
        "beta0_type3_trajectories": beta0_type3_trajectories,
        "total_models": models + control_models + high_n_models + beta0_type3_models,
        "total_trajectories": (
            trajectories
            + control_trajectories
            + high_n_trajectories
            + beta0_type3_trajectories
        ),
    }


def continuation_work_scale(data_dir: Path) -> dict[str, object]:
    """Count exact finite-map updates in the main h-continuation audit."""
    model_layer_updates = 0
    token_layer_updates = 0
    deepest_ratio = 1.0
    files = 0
    for family in (1, 2, 3, 4):
        for label in LABELS:
            ordinary_path = data_dir / f"small_step_extension_f{family}_{label}.json"
            deep_path = data_dir / f"small_step_extension_deep_f{family}_{label}.json"
            path = deep_path if deep_path.exists() else ordinary_path
            payload = json.loads(path.read_text())
            ordinary = json.loads(ordinary_path.read_text())
            base_settings = ordinary["settings"]
            first_extension = base_settings["extension"]
            latest_extension = payload["settings"]["extension"]
            base_ratios = set(base_settings["ratios"]) - set(
                first_extension["ratios"]
            )
            first_extension_ratios = set(first_extension["ratios"])
            for record in payload["records"]:
                n_tokens = int(record["identity"]["n_tokens"])
                for point in record["trace"]:
                    ratio = float(point["ratio"])
                    deepest_ratio = min(deepest_ratio, ratio)
                    if ratio in base_ratios:
                        settings = base_settings
                    elif ratio in first_extension_ratios:
                        settings = first_extension
                    else:
                        settings = latest_extension
                    map_updates = (
                        int(point["burn_layers"])
                        + int(settings["sample_count"])
                        * int(point["sample_stride_layers"])
                        + 2
                        * int(
                            np.ceil(
                                float(settings["lyapunov_time_normalized"])
                                / ratio
                            )
                        )
                    )
                    model_layer_updates += map_updates
                    token_layer_updates += n_tokens * map_updates
            files += 1
    return {
        "files": files,
        "records": 1908,
        "deepest_step_ratio": deepest_ratio,
        "model_layer_updates": model_layer_updates,
        "token_layer_updates": token_layer_updates,
    }


def continuation_survival_curve(paths: list[Path]) -> list[dict[str, object]]:
    """Aggregate moving counts at every ratio available in continuation files."""
    totals: dict[float, int] = {}
    moving: dict[float, int] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            for point in record["trace"]:
                ratio = round(float(point["ratio"]), 12)
                totals[ratio] = totals.get(ratio, 0) + 1
                moving[ratio] = moving.get(ratio, 0) + int(
                    float(point["motion_per_normalized_time"]) >= 1e-3
                )
    return [
        {
            "step_ratio": ratio,
            "records_at_ratio": totals[ratio],
            "moving": moving[ratio],
            "moving_fraction": moving[ratio] / totals[ratio],
        }
        for ratio in sorted(totals, reverse=True)
    ]


def preferred_continuation_paths(
    data_dir: Path, deep_pattern: str, fallback_pattern: str
) -> list[Path]:
    """Choose a deep continuation file group only once that group is complete."""
    deep = sorted(data_dir.glob(deep_pattern))
    fallback = sorted(data_dir.glob(fallback_pattern))
    return deep if len(deep) == len(fallback) and deep else fallback


def final_extension_work_scale(paths: list[Path]) -> dict[str, object]:
    """Count updates performed only by the final extension in each payload."""
    model_layer_updates = 0
    token_layer_updates = 0
    records = 0
    ratios: set[float] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        settings = payload["settings"]["extension"]
        extension_ratios = {round(float(value), 12) for value in settings["ratios"]}
        ratios |= extension_ratios
        records += len(payload["records"])
        for record in payload["records"]:
            n_tokens = int(record["identity"]["n_tokens"])
            for point in record["trace"]:
                ratio = round(float(point["ratio"]), 12)
                if ratio not in extension_ratios:
                    continue
                updates = (
                    int(point["burn_layers"])
                    + int(settings["sample_count"])
                    * int(point["sample_stride_layers"])
                    + 2
                    * int(
                        np.ceil(
                            float(settings["lyapunov_time_normalized"]) / ratio
                        )
                    )
                )
                model_layer_updates += updates
                token_layer_updates += n_tokens * updates
    return {
        "files": len(paths),
        "records": records,
        "ratios": sorted(ratios, reverse=True),
        "model_layer_updates": model_layer_updates,
        "token_layer_updates": token_layer_updates,
    }


def unscreened_random_model_scale(data_dir: Path) -> dict[str, int]:
    """Count the fresh model-state pairs used outside the attractor harvest."""
    paths = sorted(data_dir.glob("random_model_finite_horizon_*.json"))
    models = sum(
        int(json.loads(path.read_text())["summary"]["models"])
        for path in paths
    )
    harvested = harvest_scale(data_dir)
    return {
        "files": len(paths),
        "models": models,
        "trajectories": models,
        "audit_total_model_draws": int(harvested["total_models"]) + models,
        "audit_total_model_wide_trajectories": (
            int(harvested["total_trajectories"]) + models
        ),
    }


def unscreened_random_model_work_scale(data_dir: Path) -> dict[str, int]:
    """Count discrete updates and RK4 field evaluations in fresh-model runs."""
    paths = sorted(data_dir.glob("random_model_finite_horizon_*.json"))
    model_layer_updates = 0
    token_layer_updates = 0
    rk4_model_field_evaluations = 0
    rk4_token_field_evaluations = 0
    for path in paths:
        settings = json.loads(path.read_text())["settings"]
        family_count = len(settings["families"])
        token_counts = [int(value) for value in settings["token_counts"]]
        models_per_cell = int(settings["models_per_cell"])
        horizon = float(settings["horizon_normalized"])
        layer_count = 0
        for ratio in settings["ratios"]:
            layers = int(round(horizon / float(ratio)))
            if not np.isclose(
                layers * float(ratio), horizon, atol=1e-12, rtol=0.0
            ):
                raise ValueError(f"nonintegral layer count in {path}")
            layer_count += layers
        model_layer_updates += (
            family_count * len(token_counts) * models_per_cell * layer_count
        )
        token_layer_updates += (
            family_count * models_per_cell * sum(token_counts) * layer_count
        )

        rk4_steps = int(
            np.ceil(horizon / float(settings["reference_rk4_dt"]))
        )
        rk4_stages = 4 * rk4_steps
        rk4_model_field_evaluations += (
            family_count * len(token_counts) * models_per_cell * rk4_stages
        )
        rk4_token_field_evaluations += (
            family_count * models_per_cell * sum(token_counts) * rk4_stages
        )
    return {
        "files": len(paths),
        "discrete_model_layer_updates": model_layer_updates,
        "discrete_token_layer_updates": token_layer_updates,
        "rk4_model_field_evaluations": rk4_model_field_evaluations,
        "rk4_token_field_evaluations": rk4_token_field_evaluations,
    }


def continuation_endpoint_agreement(
    paths: list[Path], ode_records: list[dict]
) -> dict[str, object]:
    """Compare continuation endpoint decisions with an independent ODE replay."""
    endpoint_by_key: dict[tuple[int, str, int, int, int], bool] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        family = int(payload["family"])
        label = str(payload["label"])
        for record in payload["records"]:
            identity = record["identity"]
            key = (
                family,
                label,
                int(identity["n_tokens"]),
                int(identity["subtype_code"]),
                int(identity["source_model_index"]),
            )
            endpoint_by_key[key] = bool(
                float(record["trace"][-1]["motion_per_normalized_time"]) >= 1e-3
            )
    ode_by_key = {record_key(record): record for record in ode_records}
    if endpoint_by_key.keys() != ode_by_key.keys():
        raise ValueError("continuation and direct ODE record sets do not match")
    transitions = {
        "finite_fixed_to_ode_fixed": 0,
        "finite_fixed_to_ode_moving": 0,
        "finite_moving_to_ode_fixed": 0,
        "finite_moving_to_ode_moving": 0,
    }
    for key, finite_moving in endpoint_by_key.items():
        ode_moving = bool(
            float(ode_by_key[key]["metrics"]["motion_per_normalized_time"])
            >= 1e-3
        )
        transitions[
            f"finite_{'moving' if finite_moving else 'fixed'}_to_ode_"
            f"{'moving' if ode_moving else 'fixed'}"
        ] += 1
    agreement = (
        transitions["finite_fixed_to_ode_fixed"]
        + transitions["finite_moving_to_ode_moving"]
    )
    return {
        "records": len(endpoint_by_key),
        "transitions": transitions,
        "agreement_fraction": agreement / len(endpoint_by_key),
    }


def stability_adjusted_continuation_endpoint_agreement(
    paths: list[Path],
    ode_records: list[dict],
    ode_promote_keys: set[tuple[int, str, int, int, int]],
    ode_decay_keys: set[tuple[int, str, int, int, int]],
) -> dict[str, object]:
    """Compare endpoints after finite-lock and long-ODE stability corrections."""
    endpoint_by_key: dict[tuple[int, str, int, int, int], dict] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            endpoint_by_key[
                (
                    int(payload["family"]),
                    str(payload["label"]),
                    int(record["identity"]["n_tokens"]),
                    int(record["identity"]["subtype_code"]),
                    int(record["identity"]["source_model_index"]),
                )
            ] = record["trace"][-1]
    ode_by_key = {record_key(record): record for record in ode_records}
    if endpoint_by_key.keys() != ode_by_key.keys():
        raise ValueError("continuation and direct ODE record sets do not match")

    finite_promote_keys = {
        key
        for key, endpoint in endpoint_by_key.items()
        if float(endpoint["motion_per_normalized_time"]) < 1e-3
        and float(endpoint["lyapunov_per_normalized_time"]) > 5e-3
    }
    transitions = {
        "finite_fixed_to_ode_fixed": 0,
        "finite_fixed_to_ode_moving": 0,
        "finite_moving_to_ode_fixed": 0,
        "finite_moving_to_ode_moving": 0,
    }
    disagreement_keys = []
    finite_moving_count = 0
    ode_moving_count = 0
    for key, endpoint in endpoint_by_key.items():
        finite_moving = (
            float(endpoint["motion_per_normalized_time"]) >= 1e-3
            or key in finite_promote_keys
        )
        ode_moving = (
            float(ode_by_key[key]["metrics"]["motion_per_normalized_time"])
            >= 1e-3
        )
        if key in ode_promote_keys:
            ode_moving = True
        if key in ode_decay_keys:
            ode_moving = False
        finite_moving_count += int(finite_moving)
        ode_moving_count += int(ode_moving)
        transitions[
            f"finite_{'moving' if finite_moving else 'fixed'}_to_ode_"
            f"{'moving' if ode_moving else 'fixed'}"
        ] += 1
        if finite_moving != ode_moving:
            disagreement_keys.append(list(key))
    agreement = (
        transitions["finite_fixed_to_ode_fixed"]
        + transitions["finite_moving_to_ode_moving"]
    )
    return {
        "records": len(endpoint_by_key),
        "finite_unstable_lock_promotions": len(finite_promote_keys),
        "finite_unstable_lock_keys": [list(key) for key in sorted(finite_promote_keys)],
        "ode_long_time_promotions": len(ode_promote_keys),
        "ode_long_time_decays": len(ode_decay_keys),
        "finite_moving": finite_moving_count,
        "ode_moving": ode_moving_count,
        "transitions": transitions,
        "agreement_fraction": agreement / len(endpoint_by_key),
        "disagreement_keys": disagreement_keys,
    }


def type3_symmetry_split(records: list[dict]) -> dict[str, object]:
    selected = [record for record in records if int(record["family"]) == 3]
    value_symmetric = [
        int(record["identity"]["subtype_code"]) in (0, 2) for record in selected
    ]
    score_symmetric = [
        int(record["identity"]["subtype_code"]) in (0, 1) for record in selected
    ]
    moving = [record["metrics"]["motion_per_normalized_time"] >= 1e-3 for record in selected]

    def table(structural: list[bool]) -> list[list[int]]:
        return [
            [
                sum(flag and is_moving for flag, is_moving in zip(structural, moving)),
                sum(flag and not is_moving for flag, is_moving in zip(structural, moving)),
            ],
            [
                sum(not flag and is_moving for flag, is_moving in zip(structural, moving)),
                sum(not flag and not is_moving for flag, is_moving in zip(structural, moving)),
            ],
        ]

    value_table = table(value_symmetric)
    score_table = table(score_symmetric)
    return {
        "value_symmetric_vs_general": value_table,
        "value_one_sided_fisher_p": float(
            fisher_exact(value_table, alternative="less").pvalue
        ),
        "score_symmetric_vs_general": score_table,
        "score_two_sided_fisher_p": float(fisher_exact(score_table).pvalue),
    }


def apply_internal_mover_corrections(
    rows: list[dict[str, object]], corrections: dict[int, int]
) -> list[dict[str, object]]:
    """Move stability-certified numerical locks from fixed to internal motion."""
    output = []
    for row in rows:
        updated = dict(row)
        count = corrections.get(int(row["family"]), 0)
        if count:
            updated["fixed"] = int(updated["fixed"]) - count
            updated["moving"] = int(updated["moving"]) + count
            updated["internal_shape_motion"] = (
                int(updated["internal_shape_motion"]) + count
            )
            updated["moving_fraction"] = updated["moving"] / int(updated["records"])
            updated["moving_wilson_95"] = wilson_interval(
                int(updated["moving"]), int(updated["records"])
            )
        output.append(updated)
    return output


def apply_unresolved_mover_corrections(
    rows: list[dict[str, object]], corrections: dict[int, int]
) -> list[dict[str, object]]:
    """Promote long-lived sub-threshold records without changing fixed counts."""
    output = []
    for row in rows:
        updated = dict(row)
        count = corrections.get(int(row["family"]), 0)
        if count:
            updated["moving"] = int(updated["moving"]) + count
            updated["internal_shape_motion"] = (
                int(updated["internal_shape_motion"]) + count
            )
            updated["moving_fraction"] = updated["moving"] / int(updated["records"])
            updated["moving_wilson_95"] = wilson_interval(
                int(updated["moving"]), int(updated["records"])
            )
        output.append(updated)
    return output


def apply_internal_mover_decay_corrections(
    rows: list[dict[str, object]], corrections: dict[int, int]
) -> list[dict[str, object]]:
    """Move very-long-time internal transients back to the fixed class."""
    output = []
    for row in rows:
        updated = dict(row)
        count = corrections.get(int(row["family"]), 0)
        if count:
            updated["fixed"] = int(updated["fixed"]) + count
            updated["moving"] = int(updated["moving"]) - count
            updated["internal_shape_motion"] = (
                int(updated["internal_shape_motion"]) - count
            )
            updated["moving_fraction"] = updated["moving"] / int(updated["records"])
            updated["moving_wilson_95"] = wilson_interval(
                int(updated["moving"]), int(updated["records"])
            )
        output.append(updated)
    return output


def lock_recheck_summary(record: dict[str, object]) -> dict[str, object]:
    return {
        "family": record["family"],
        "label": record["label"],
        "identity": record["identity"],
        "metrics": record["metrics"],
    }


def long_replay_by_family(payload: dict[str, object]) -> list[dict[str, int]]:
    records = payload["records"]
    rows = []
    for family in (1, 2, 3, 4):
        selected = [record for record in records if int(record["family"]) == family]
        rows.append(
            {
                "family": family,
                "selected_starting_movers": len(selected),
                "still_moving": sum(
                    bool(record["metrics"]["moving"]) for record in selected
                ),
                "still_internal": sum(
                    bool(record["metrics"]["internal"]) for record in selected
                ),
                "positive_lyapunov": sum(
                    float(record["metrics"]["lyapunov_per_normalized_time"])
                    > 5e-3
                    for record in selected
                ),
            }
        )
    return rows


def correct_long_replay_by_family(
    rows: list[dict[str, int]], corrections: dict[int, int]
) -> list[dict[str, int]]:
    """Add independently certified movers to a long-replay family table."""
    output = []
    for row in rows:
        updated = dict(row)
        count = corrections.get(int(updated["family"]), 0)
        updated["still_moving"] += count
        updated["still_internal"] += count
        output.append(updated)
    return output


def long_replay_distinct_moving_models(payload: dict[str, object]) -> int:
    return len(
        {
            (
                int(record["family"]),
                int(record["identity"]["n_tokens"]),
                int(record["identity"]["source_model_index"]),
            )
            for record in payload["records"]
            if bool(record["metrics"]["moving"])
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/spectral_self_attention"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ode_table, records = direct_ode_table(args.data_dir)
    high_n_ode_table, high_n_records = direct_ode_table(args.data_dir, "_large_n")
    noise_ode_table, noise_records = direct_ode_table(args.data_dir, "_noise1e3")
    relaxed_ode_table, relaxed_records = direct_ode_table(
        args.data_dir, "_relaxed2_noise1e3"
    )
    antilock_matched_ode_table, antilock_matched_records = direct_ode_table(
        args.data_dir, "_antilock_matched"
    )
    dt01_ode_table, dt01_records = direct_ode_table(
        args.data_dir, "_dt01_relaxed2_noise1e3"
    )
    dt01_matched_ode_table, dt01_matched_records = direct_ode_table(
        args.data_dir, "_dt01_matched_relaxed2_noise1e3"
    )
    dt005_matched_ode_table, dt005_matched_records = direct_ode_table(
        args.data_dir, "_dt005_matched_relaxed2_noise1e3"
    )
    high_n_relaxed_ode_table, high_n_relaxed_records = direct_ode_table(
        args.data_dir, "_large_n_relaxed2_noise1e3"
    )
    high_n_antilock_ode_table, high_n_antilock_records = direct_ode_table(
        args.data_dir, "_large_n_antilock"
    )
    high_n_antilock_matched_ode_table, high_n_antilock_matched_records = (
        direct_ode_table(args.data_dir, "_large_n_antilock_matched")
    )
    beta0_type3_records = [
        record
        for label in LABELS
        for record in json.loads(
            (args.data_dir / f"continuous_ode_beta0_f3_{label}.json").read_text()
        )["records"]
    ]
    beta0_type3_noise_records = [
        record
        for label in LABELS
        for record in json.loads(
            (
                args.data_dir / f"continuous_ode_beta0_f3_{label}_noise1e3.json"
            ).read_text()
        )["records"]
    ]
    beta0_type3_relaxed_records = [
        record
        for label in LABELS
        for record in json.loads(
            (
                args.data_dir
                / f"continuous_ode_beta0_f3_{label}_relaxed2_noise1e3.json"
            ).read_text()
        )["records"]
    ]
    beta0_type3_antilock_matched_records = [
        record
        for label in LABELS
        for record in json.loads(
            (
                args.data_dir
                / f"continuous_ode_beta0_f3_{label}_antilock_matched.json"
            ).read_text()
        )["records"]
    ]
    beta0_type1_relaxed_records = [
        record
        for label in LABELS
        for record in json.loads(
            (
                args.data_dir
                / f"continuous_ode_beta0_f1_{label}_relaxed2_noise1e3.json"
            ).read_text()
        )["records"]
    ]
    moving = sum(row["moving"] for row in ode_table)
    total = sum(row["records"] for row in ode_table)
    high_n_moving = sum(
        record["metrics"]["motion_per_normalized_time"] >= 1e-3
        for record in high_n_records
    )
    equilibrium = json.loads(
        (
            args.data_dir / "continuous_equilibrium_spectrum_census_relaxed2.json"
        ).read_text()
    )
    spectra = {
        "type4_strong_chaos": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_f4_strong_chaos_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type2_weak_chaos": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_f2_weak_chaos_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type4_stable_cycle": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_f4_p3_cycle_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type4_stable_cycle_attention_only": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_f4_p3_cycle_attention_only_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type2_p3_stable_cycle": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_f2_p3_i2281_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type2_p4_stable_cycle": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_f2_p4_i1663_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type4_p4_stable_cycle": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_f4_p4_i1041_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type4_eight_token_hyperchaos": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_highn_f4_n8_i1632_long_relaxed2.json"
            ).read_text()
        )["spectrum"],
        "type4_eight_token_hyperchaos_antilock": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_highn_f4_n8_i1632_antilock_long.json"
            ).read_text()
        )["spectrum"],
        "type3_beta0_intermittent_chaos_antilock": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_beta0_f3_antilock1e12_long.json"
            ).read_text()
        )["spectrum"],
        "type3_beta0_intermittent_chaos_antilock_1e14": json.loads(
            (
                args.data_dir
                / "continuous_lyapunov_spectrum_beta0_f3_antilock1e14_long.json"
            ).read_text()
        )["spectrum"],
    }
    main_stability_table = apply_internal_mover_corrections(
        relaxed_ode_table, {4: 1}
    )
    high_n_stability_table = apply_internal_mover_decay_corrections(
        apply_unresolved_mover_corrections(
            apply_internal_mover_corrections(
                high_n_antilock_matched_ode_table, {4: 1}
            ),
            {4: 1},
        ),
        {4: 1},
    )
    main_lock_recheck = json.loads(
        (
            args.data_dir
            / "continuous_ode_recheck_f4_chaos_n4_i1734_antilock_matched_long.json"
        ).read_text()
    )["records"][0]
    high_n8_lock_recheck = json.loads(
        (
            args.data_dir
            / "continuous_ode_recheck_highn_f4_chaos_n8_i867_antilock_matched_long.json"
        ).read_text()
    )["records"][0]
    high_n16_lock_recheck = json.loads(
        (
            args.data_dir
            / "continuous_ode_recheck_highn_f4_chaos_n16_i1689_antilock.json"
        ).read_text()
    )["records"][0]
    high_n16_threshold_recheck = json.loads(
        (
            args.data_dir
            / "continuous_ode_recheck_highn_f4_chaos_n16_i1483_antilock_matched_long.json"
        ).read_text()
    )["records"][0]
    long_main = json.loads(
        (args.data_dir / "continuous_long_time_replay_main.json").read_text()
    )
    long_main_nokick = json.loads(
        (args.data_dir / "continuous_long_time_replay_main_nokick.json").read_text()
    )
    long_high_n = json.loads(
        (args.data_dir / "continuous_long_time_replay_highn.json").read_text()
    )
    long_high_n_nokick = json.loads(
        (args.data_dir / "continuous_long_time_replay_highn_nokick.json").read_text()
    )
    long_beta0_type3 = json.loads(
        (args.data_dir / "continuous_long_time_replay_beta0_f3.json").read_text()
    )
    long_beta0_type3_nokick = json.loads(
        (
            args.data_dir / "continuous_long_time_replay_beta0_f3_nokick.json"
        ).read_text()
    )
    very_long_main_border_recheck = json.loads(
        (
            args.data_dir
            / "continuous_ode_recheck_f3_chaos_n4_i1402_T20000.json"
        ).read_text()
    )["records"][0]
    result = {
        "harvest_scale": harvest_scale(args.data_dir),
        "unscreened_random_model_scale": unscreened_random_model_scale(
            args.data_dir
        ),
        "unscreened_random_model_work_scale": (
            unscreened_random_model_work_scale(args.data_dir)
        ),
        "continuation_work_scale": continuation_work_scale(args.data_dir),
        "supplemental_final_extension_work_scale": final_extension_work_scale(
            sorted(args.data_dir.glob("small_step_extension_deep_highn_*.json"))
            + sorted(args.data_dir.glob("small_step_extension_deep_beta0_*.json"))
        ),
        "deep_finite_step_continuation": continuation_plateau(
            args.data_dir, relaxed_records
        ),
        "high_token_step_survival": continuation_survival_curve(
            preferred_continuation_paths(
                args.data_dir,
                "small_step_extension_deep_highn_f*_*.json",
                "small_step_continuation_f*_large_n.json",
            )
        ),
        "high_token_continuation_vs_direct_ode": continuation_endpoint_agreement(
            preferred_continuation_paths(
                args.data_dir,
                "small_step_extension_deep_highn_f*_*.json",
                "small_step_continuation_f*_large_n.json",
            ),
            high_n_antilock_matched_records,
        ),
        "high_token_stability_adjusted_continuation_vs_direct_ode": (
            stability_adjusted_continuation_endpoint_agreement(
                preferred_continuation_paths(
                    args.data_dir,
                    "small_step_extension_deep_highn_f*_*.json",
                    "small_step_continuation_f*_large_n.json",
                ),
                high_n_antilock_matched_records,
                ode_promote_keys={
                    record_key(high_n8_lock_recheck),
                    record_key(high_n16_threshold_recheck),
                },
                ode_decay_keys={
                    record_key(record)
                    for record in long_high_n["records"]
                    if float(record["metrics"]["motion_per_normalized_time"])
                    < 1e-3
                },
            )
        ),
        "beta0_type1_step_survival": continuation_survival_curve(
            preferred_continuation_paths(
                args.data_dir,
                "small_step_extension_deep_beta0_f1_*.json",
                "small_step_extension_beta0_f1_*.json",
            )
        ),
        "beta0_type3_step_survival": continuation_survival_curve(
            preferred_continuation_paths(
                args.data_dir,
                "small_step_extension_deep_beta0_f3_*.json",
                "small_step_continuation_beta0_f3_*.json",
            )
        ),
        "beta0_type3_continuation_vs_direct_ode": (
            continuation_endpoint_agreement(
                preferred_continuation_paths(
                    args.data_dir,
                    "small_step_extension_deep_beta0_f3_*.json",
                    "small_step_continuation_beta0_f3_*.json",
                ),
                beta0_type3_antilock_matched_records,
            )
        ),
        "direct_ode_by_family": ode_table,
        "direct_ode_total": {
            "records": total,
            "moving": moving,
            "moving_fraction": moving / total,
            "moving_wilson_95": wilson_interval(moving, total),
            "internal_shape_motion": sum(
                row["internal_shape_motion"] for row in ode_table
            ),
        },
        "direct_ode_threshold_sensitivity": threshold_sensitivity(records),
        "detected_model_prevalence": detected_model_prevalence(
            args.data_dir, records, (1, 2, 3, 4)
        ),
        "stronger_noise_direct_ode_by_family": noise_ode_table,
        "stronger_noise_paired_sensitivity": paired_noise_sensitivity(
            records, noise_records
        ),
        "stronger_noise_threshold_sensitivity": threshold_sensitivity(
            noise_records
        ),
        "fully_relaxed_noise_direct_ode_by_family": relaxed_ode_table,
        "antilock_matched_direct_ode_by_family": antilock_matched_ode_table,
        "antilock_matched_paired_vs_fully_relaxed": paired_noise_sensitivity(
            relaxed_records, antilock_matched_records
        ),
        "stability_audited_direct_ode_by_family": main_stability_table,
        "stability_audited_direct_ode_total": {
            "records": sum(int(row["records"]) for row in main_stability_table),
            "moving": sum(int(row["moving"]) for row in main_stability_table),
            "internal_shape_motion": sum(
                int(row["internal_shape_motion"]) for row in main_stability_table
            ),
            "numerical_locks_corrected": 1,
            "lock_recheck": lock_recheck_summary(main_lock_recheck),
        },
        "fully_relaxed_noise_paired_vs_baseline": paired_noise_sensitivity(
            records, relaxed_records
        ),
        "fully_relaxed_noise_paired_vs_unrelaxed": paired_noise_sensitivity(
            noise_records, relaxed_records
        ),
        "fully_relaxed_noise_threshold_sensitivity": threshold_sensitivity(
            relaxed_records
        ),
        "fully_relaxed_detected_model_prevalence": detected_model_prevalence(
            args.data_dir, relaxed_records, (1, 2, 3, 4)
        ),
        "dt01_independent_replica_direct_ode_by_family": dt01_ode_table,
        "dt01_independent_replica_paired_vs_dt02": paired_noise_sensitivity(
            relaxed_records, dt01_records
        ),
        "dt01_independent_replica_threshold_sensitivity": threshold_sensitivity(
            dt01_records
        ),
        "dt01_matched_direct_ode_by_family": dt01_matched_ode_table,
        "dt01_matched_paired_vs_dt02": paired_noise_sensitivity(
            relaxed_records, dt01_matched_records
        ),
        "dt005_matched_direct_ode_by_family": dt005_matched_ode_table,
        "dt005_matched_paired_vs_dt02": paired_noise_sensitivity(
            relaxed_records, dt005_matched_records
        ),
        "high_token_direct_ode_by_family": high_n_ode_table,
        "high_token_direct_ode_total": {
            "records": len(high_n_records),
            "moving": high_n_moving,
            "moving_fraction": high_n_moving / len(high_n_records),
            "moving_wilson_95": wilson_interval(
                high_n_moving, len(high_n_records)
            ),
        },
        "high_token_threshold_sensitivity": threshold_sensitivity(high_n_records),
        "high_token_detected_model_prevalence": detected_model_prevalence(
            args.data_dir, high_n_records, (8, 16)
        ),
        "high_token_fully_relaxed_direct_ode_by_family": high_n_relaxed_ode_table,
        "high_token_fully_relaxed_by_token_count": stratified_ode_table(
            high_n_relaxed_records, "n_tokens"
        ),
        "high_token_fully_relaxed_paired_sensitivity": paired_noise_sensitivity(
            high_n_records, high_n_relaxed_records
        ),
        "high_token_fully_relaxed_detected_model_prevalence": (
            detected_model_prevalence(
                args.data_dir, high_n_relaxed_records, (8, 16)
            )
        ),
        "high_token_antilock_direct_ode_by_family": high_n_antilock_ode_table,
        "high_token_antilock_by_token_count": stratified_ode_table(
            high_n_antilock_records, "n_tokens"
        ),
        "high_token_antilock_paired_vs_fully_relaxed": paired_noise_sensitivity(
            high_n_relaxed_records, high_n_antilock_records
        ),
        "high_token_antilock_matched_direct_ode_by_family": (
            high_n_antilock_matched_ode_table
        ),
        "high_token_stability_audited_direct_ode_by_family": (
            high_n_stability_table
        ),
        "high_token_stability_audited_total": {
            "records": sum(int(row["records"]) for row in high_n_stability_table),
            "moving": sum(int(row["moving"]) for row in high_n_stability_table),
            "internal_shape_motion": sum(
                int(row["internal_shape_motion"]) for row in high_n_stability_table
            ),
            "numerical_locks_corrected": 2,
            "long_time_subthreshold_promoted": 1,
            "long_time_mover_decayed": 1,
            "lock_rechecks": [
                lock_recheck_summary(high_n8_lock_recheck),
                lock_recheck_summary(high_n16_lock_recheck),
            ],
            "subthreshold_recheck": lock_recheck_summary(
                high_n16_threshold_recheck
            ),
        },
        "high_token_stability_audited_by_token_count": [
            {
                "n_tokens": 8,
                "records": 397,
                "moving": 83,
                "internal_shape_motion": 63,
            },
            {
                "n_tokens": 16,
                "records": 184,
                "moving": 39,
                "internal_shape_motion": 29,
            },
        ],
        "high_token_stability_audited_detected_models": {
            "sampled_models": 24576,
            "models_with_detected_continuous_motion": 117,
            "detected_fraction": 117 / 24576,
        },
        "high_token_antilock_matched_by_token_count": stratified_ode_table(
            high_n_antilock_matched_records, "n_tokens"
        ),
        "high_token_antilock_matched_detected_model_prevalence": (
            detected_model_prevalence(
                args.data_dir, high_n_antilock_matched_records, (8, 16)
            )
        ),
        "high_token_antilock_matched_paired_vs_fully_relaxed": (
            paired_noise_sensitivity(
                high_n_relaxed_records, high_n_antilock_matched_records
            )
        ),
        "high_token_direct_ode_by_token_count": stratified_ode_table(
            high_n_records, "n_tokens"
        ),
        "high_token_direct_ode_by_subtype": stratified_ode_table(
            high_n_records, "subtype_code"
        ),
        "type3_symmetry_split": type3_symmetry_split(records),
        "type3_stronger_noise_symmetry_split": type3_symmetry_split(
            noise_records
        ),
        "beta0_type3_symmetry_split": type3_symmetry_split(
            beta0_type3_records
        ),
        "beta0_type3_stronger_noise_symmetry_split": type3_symmetry_split(
            beta0_type3_noise_records
        ),
        "beta0_type3_stronger_noise_paired_sensitivity": paired_noise_sensitivity(
            beta0_type3_records, beta0_type3_noise_records
        ),
        "beta0_type3_fully_relaxed_symmetry_split": type3_symmetry_split(
            beta0_type3_relaxed_records
        ),
        "beta0_type3_stability_audited_value_symmetry_split": {
            "value_symmetric_vs_general": [[0, 79], [11, 77]],
            "value_one_sided_fisher_p": float(
                fisher_exact([[0, 79], [11, 77]], alternative="less").pvalue
            ),
            "numerical_locks_corrected": 1,
        },
        "beta0_type3_fully_relaxed_paired_sensitivity": paired_noise_sensitivity(
            beta0_type3_records, beta0_type3_relaxed_records
        ),
        "beta0_type3_antilock_matched_symmetry_split": type3_symmetry_split(
            beta0_type3_antilock_matched_records
        ),
        "beta0_type3_antilock_matched_paired_vs_fully_relaxed": (
            paired_noise_sensitivity(
                beta0_type3_relaxed_records,
                beta0_type3_antilock_matched_records,
            )
        ),
        "beta0_type1_exact_potential_negative_control": {
            "records": len(beta0_type1_relaxed_records),
            "fixed": sum(
                record["metrics"]["fixed"]
                for record in beta0_type1_relaxed_records
            ),
            "moving": sum(
                record["metrics"]["motion_per_normalized_time"] >= 1e-3
                for record in beta0_type1_relaxed_records
            ),
            "threshold_sensitivity": threshold_sensitivity(
                beta0_type1_relaxed_records
            ),
        },
        "stable_spirals": {
            "total": sum(
                row["stable_spiral"]
                for key, row in equilibrium["summary"].items()
                if "beta0" not in key
            ),
            "by_group": equilibrium["summary"],
        },
        "full_lyapunov_spectra": spectra,
        "strong_chaos_robustness": json.loads(
            (args.data_dir / "f4_strong_chaos_robust_relaxed2.json").read_text()
        )["summary"],
        "weak_chaos_robustness": json.loads(
            (args.data_dir / "f2_weak_chaos_robust_relaxed2.json").read_text()
        )["summary"],
        "stable_cycle_robustness": {
            path.stem: json.loads(path.read_text())["summary"]
            for path in sorted(
                args.data_dir.glob("f[24]_p*_cycle_robust_relaxed2.json")
            )
        },
        "stable_cycle_step_scaling": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob("cycle_period_step_scaling_f*_p*_i*.json")
            )
        },
        "stable_cycle_off_grid_step_scaling": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob(
                    "cycle_period_step_scaling_offgrid_f*_p*_i*.json"
                )
            )
        },
        "global_basin_surveys": {
            path.stem: json.loads(path.read_text())["summary"]
            for path in (
                args.data_dir / "continuous_basin_f4_strong_chaos.json",
                args.data_dir / "continuous_basin_f2_weak_chaos.json",
                args.data_dir / "continuous_basin_beta0_f3_hyperchaos.json",
                args.data_dir
                / "continuous_basin_beta0_f3_hyperchaos_antilock.json",
                args.data_dir
                / "continuous_basin_beta0_f3_hyperchaos_antilock_matched.json",
                args.data_dir / "continuous_basin_highn_f4_n8_i1632.json",
                args.data_dir
                / "continuous_basin_highn_f4_n8_i1632_antilock.json",
                args.data_dir
                / "continuous_basin_highn_f4_n8_i1632_antilock_matched.json",
            )
        },
        "numerical_symmetry_lock_rechecks": {
            path.stem: json.loads(path.read_text())["records"][0]["metrics"]
            for path in sorted(
                args.data_dir.glob(
                    "continuous_ode_recheck_highn_f4_chaos_*_antilock.json"
                )
            )
        },
        "median_small_step_local_error_order": float(
            np.median(
                [
                    row["median_local_error_order"]
                    for row in json.loads(
                        (args.data_dir / "small_step_audit_with_ode.json").read_text()
                    )["groups"]
                ]
            )
        ),
        "finite_horizon_convergence": json.loads(
            (args.data_dir / "finite_horizon_convergence_main.json").read_text()
        ),
        "finite_horizon_convergence_T10": json.loads(
            (
                args.data_dir / "finite_horizon_convergence_main_T10.json"
            ).read_text()
        ),
        "finite_horizon_convergence_high_token_T10": json.loads(
            (
                args.data_dir / "finite_horizon_convergence_highn_T10.json"
            ).read_text()
        ),
        "finite_horizon_convergence_beta0_type1_T10": json.loads(
            (
                args.data_dir / "finite_horizon_convergence_beta0_f1_T10.json"
            ).read_text()
        ),
        "finite_horizon_convergence_beta0_type3_T10": json.loads(
            (
                args.data_dir / "finite_horizon_convergence_beta0_f3_T10.json"
            ).read_text()
        ),
        "richardson_finite_horizon_main_T2": json.loads(
            (
                args.data_dir / "richardson_finite_horizon_main_T2.json"
            ).read_text()
        ),
        "richardson_finite_horizon_main_T10": json.loads(
            (
                args.data_dir / "richardson_finite_horizon_main_T10.json"
            ).read_text()
        ),
        "richardson_finite_horizon_supplements_T2": {
            path.stem: json.loads(path.read_text())
            for path in (
                args.data_dir / "richardson_finite_horizon_highn_T2.json",
                args.data_dir / "richardson_finite_horizon_beta0_f1_T2.json",
                args.data_dir / "richardson_finite_horizon_beta0_f3_T2.json",
            )
        },
        "long_time_replays": {
            "main_microkick": long_main,
            "main_no_kick": long_main_nokick,
            "high_token_microkick": long_high_n,
            "high_token_no_kick": long_high_n_nokick,
            "beta0_type3_microkick": long_beta0_type3,
            "beta0_type3_no_kick": long_beta0_type3_nokick,
        },
        "long_time_strict_summaries": {
            "main": {
                "raw_by_family": long_replay_by_family(long_main),
                "corrected_by_family": correct_long_replay_by_family(
                    long_replay_by_family(long_main), {3: 1, 4: 1}
                ),
                "microkick_vs_no_kick": paired_noise_sensitivity(
                    long_main["records"], long_main_nokick["records"]
                ),
                "raw_still_moving": long_main["summary"]["still_moving"],
                "raw_still_internal": long_main["summary"]["still_internal"],
                "unstable_lock_added": 1,
                "intermittent_border_recovered_at_T20000": 1,
                "corrected_still_moving": long_main["summary"]["still_moving"]
                + 2,
                "corrected_still_internal": long_main["summary"]["still_internal"]
                + 2,
                "corrected_distinct_moving_models": (
                    long_replay_distinct_moving_models(long_main) + 2
                ),
                "intermittent_border_recheck": lock_recheck_summary(
                    very_long_main_border_recheck
                ),
            },
            "high_token": {
                "raw_by_family": long_replay_by_family(long_high_n),
                "corrected_by_family": correct_long_replay_by_family(
                    long_replay_by_family(long_high_n), {4: 2}
                ),
                "microkick_vs_no_kick": paired_noise_sensitivity(
                    long_high_n["records"], long_high_n_nokick["records"]
                ),
                "raw_still_moving": long_high_n["summary"]["still_moving"],
                "raw_still_internal": long_high_n["summary"]["still_internal"],
                "unstable_lock_and_long_threshold_added": 2,
                "corrected_still_moving": long_high_n["summary"]["still_moving"]
                + 2,
                "corrected_still_internal": long_high_n["summary"]["still_internal"]
                + 2,
                "corrected_distinct_moving_models": (
                    long_replay_distinct_moving_models(long_high_n) + 2
                ),
            },
            "beta0_type3": {
                "raw_by_family": long_replay_by_family(long_beta0_type3),
                "corrected_by_family": correct_long_replay_by_family(
                    long_replay_by_family(long_beta0_type3), {3: 1}
                ),
                "microkick_vs_no_kick": paired_noise_sensitivity(
                    long_beta0_type3["records"],
                    long_beta0_type3_nokick["records"],
                ),
                "raw_still_moving": long_beta0_type3["summary"]["still_moving"],
                "raw_still_internal": long_beta0_type3["summary"]["still_internal"],
                "unstable_lock_added": 1,
                "corrected_still_moving": long_beta0_type3["summary"]["still_moving"]
                + 1,
                "corrected_still_internal": long_beta0_type3["summary"]["still_internal"]
                + 1,
                "corrected_distinct_moving_models": (
                    long_replay_distinct_moving_models(long_beta0_type3) + 1
                ),
            },
        },
        "uniform_random_state_convergence": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob("random_state_convergence_*.json")
            )
        },
        "unscreened_random_model_convergence": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob("random_model_finite_horizon_*.json")
            )
        },
        "observable_measure_convergence": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob("observable_measure_convergence_*.json")
            )
        },
        "ensemble_observable_measure_convergence": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob(
                    "ensemble_observable_measure_convergence_*.json"
                )
            )
        },
        "basin_boundary_step_scans": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob("basin_boundary_step_scan_*.json")
            )
        },
        "basin_partition_mismatch_scaling": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                args.data_dir.glob("basin_partition_mismatch_scaling_*.json")
            )
        },
        "very_long_targeted_rechecks": {
            path.stem: json.loads(path.read_text())
            for path in sorted(
                list(args.data_dir.glob("continuous_*_T20000.json"))
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
