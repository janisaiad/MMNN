"""Fail-closed audit for the full-training Fourier/Muon campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "full_training_results"
MASKED_RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT.parents[1] / "refs" / "mup_dmft_frequency"
OPTIMIZERS = ("mup_gd", "muon_p0", "muon_p1_3", "muon_p2_3")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_numeric_cells_finite(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for value in row.values():
            if value in ("", "None", "True", "False"):
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            if not math.isfinite(numeric):
                return False
    return True


def all_json_numbers_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_json_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_json_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def numeric_equal(left: Any, right: Any) -> bool:
    """Compare derived JSON scalars without silently equating missing values."""
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1.0e-10, abs_tol=1.0e-12
        )
    except (TypeError, ValueError):
        return False


def first_numeric_crossing(
    rows: list[dict[str, str]],
    *,
    time_key: str,
    value_key: str,
    threshold: float,
    direction: str,
) -> float | None:
    """Return the first observed threshold crossing while preserving blanks."""
    if direction not in ("above", "below"):
        raise ValueError(f"unknown crossing direction: {direction}")
    for row in rows:
        encoded = row.get(value_key, "")
        if encoded in ("", "None"):
            continue
        value = float(encoded)
        crossed = value >= threshold if direction == "above" else value <= threshold
        if crossed:
            return float(row[time_key])
    return None


def median_event(rows: list[dict[str, str]], frequency: int) -> float | None:
    key = f"t50_{frequency}"
    observed = sorted(
        float(row[key]) for row in rows if row.get(key, "") not in ("", "None")
    )
    if len(observed) < math.ceil(len(rows) / 2):
        return None
    return observed[math.ceil(len(rows) / 2) - 1]


def expected_campaign_tags() -> set[str]:
    tags: set[str] = set()
    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            for seed in range(8):
                tags.add(f"hierarchy_{architecture}_d{depth}_mup_gd_s{seed}")
        for optimizer in OPTIMIZERS:
            for seed in range(8):
                tags.add(f"hierarchy_{architecture}_d5_{optimizer}_s{seed}")
                tags.add(f"powerlaw_{architecture}_d5_{optimizer}_s{seed}")
        for optimizer in ("mup_gd", "muon_p1_3"):
            for width in (64, 128, 256):
                for seed in (20, 21):
                    tags.add(f"width_{architecture}_{optimizer}_m{width}_s{seed}")
    return tags


def expected_masked_tags() -> set[str]:
    tags: set[str] = set()
    for name in ("muP", "lazy"):
        for seed in range(3):
            tags.add(f"hierarchy_{name}_s{seed}")
        for frequency in (2, 4, 6, 8, 12, 16, 24):
            for seed in range(3):
                tags.add(f"gap_{name}_q{frequency}_s{seed}")
        for frequency in (4, 8, 12, 16, 24):
            tags.add(f"fixed_gap_{name}_p{frequency - 3}_q{frequency}")
            tags.add(
                f"gap_control_{name}_p{frequency // 2}_q{frequency}"
            )
    for width in (64, 128, 256, 512):
        tags.add(f"width_muP_n{width}")
    for ratio in (0.0625, 0.125, 0.25, 0.5, 1.0):
        for seed in range(3):
            tags.add(f"rank_muP_rho{ratio:g}_s{seed}")
    return tags


def expected_masked_spectrum_tags() -> set[str]:
    return {
        f"spectrum_{name}_s{seed}"
        for name in ("muP", "lazy")
        for seed in range(3)
    }


def expected_calibration_tags() -> set[str]:
    tags: set[str] = set()
    for architecture in ("fc", "mmnn"):
        keys = [(depth, "mup_gd") for depth in (3, 5, 7)]
        keys.extend(
            (5, optimizer) for optimizer in ("muon_p0", "muon_p1_3", "muon_p2_3")
        )
        for depth, optimizer in keys:
            rates = (0.1, 0.3, 1.0) if optimizer == "mup_gd" else (0.003, 0.01, 0.03)
            for rate in rates:
                for seed in (101, 102):
                    tags.add(
                        f"cal_{architecture}_d{depth}_{optimizer}_lr{rate:g}_s{seed}"
                    )
    return tags


def expected_discretization_tags() -> set[str]:
    tags: set[str] = set()
    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            for seed in range(30, 33):
                tags.add(f"dt_{architecture}_{optimizer}_base_s{seed}")
                tags.add(f"dt_{architecture}_{optimizer}_half_s{seed}")
                tags.add(f"dt_{architecture}_{optimizer}_quarter_s{seed}")
                tags.add(f"dt_{architecture}_{optimizer}_eighth_s{seed}")
                tags.add(f"grid_{architecture}_{optimizer}_g256_s{seed}")
    return tags


def matched_discretization_log_ratios(
    summaries: list[dict[str, str]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    by_tag = {row["tag"]: row for row in summaries}
    base_half_ratios: list[float] = []
    half_quarter_ratios: list[float] = []
    quarter_eighth_ratios: list[float] = []
    grid_ratios: list[float] = []
    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            for seed in range(30, 33):
                prefix = f"dt_{architecture}_{optimizer}"
                base = float(by_tag[f"{prefix}_base_s{seed}"]["final_loss"])
                half = float(by_tag[f"{prefix}_half_s{seed}"]["final_loss"])
                quarter = float(
                    by_tag[f"{prefix}_quarter_s{seed}"]["final_loss"]
                )
                eighth = float(
                    by_tag[f"{prefix}_eighth_s{seed}"]["final_loss"]
                )
                fine = float(
                    by_tag[f"grid_{architecture}_{optimizer}_g256_s{seed}"][
                        "final_loss"
                    ]
                )
                base_half_ratios.append(abs(math.log(half / base)))
                half_quarter_ratios.append(abs(math.log(quarter / half)))
                quarter_eighth_ratios.append(abs(math.log(eighth / quarter)))
                grid_ratios.append(abs(math.log(fine / base)))
    return (
        base_half_ratios,
        half_quarter_ratios,
        quarter_eighth_ratios,
        grid_ratios,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-discretization", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {"checks": {}, "metrics": {}, "failures": []}

    def check(name: str, condition: bool, detail: Any) -> None:
        report["checks"][name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            report["failures"].append(name)

    required = (
        RESULTS / "calibration_runs.csv",
        RESULTS / "selected_learning_rates.json",
        RESULTS / "full_training_traces.csv",
        RESULTS / "full_training_summaries.csv",
        RESULTS / "full_training_metadata.json",
        RESULTS / "full_training_analysis.json",
        RESULTS / "spectral_backend_audit.json",
        MASKED_RESULTS / "metadata.json",
        MASKED_RESULTS / "run_summaries.csv",
        MASKED_RESULTS / "training_traces.csv",
        MASKED_RESULTS / "saddle_spectra.csv",
        MASKED_RESULTS / "saddle_spectrum_summaries.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    check("required_outputs", not missing, missing)
    if missing:
        output = RESULTS / "full_training_audit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)

    calibration = read_csv(RESULTS / "calibration_runs.csv")
    summaries = read_csv(RESULTS / "full_training_summaries.csv")
    traces = read_csv(RESULTS / "full_training_traces.csv")
    learning_rates = json.loads((RESULTS / "selected_learning_rates.json").read_text())
    metadata = json.loads((RESULTS / "full_training_metadata.json").read_text())
    analysis_path = RESULTS / "full_training_analysis.json"
    analysis = json.loads(analysis_path.read_text())
    backend_audit = json.loads((RESULTS / "spectral_backend_audit.json").read_text())
    masked_metadata = json.loads((MASKED_RESULTS / "metadata.json").read_text())
    masked_summaries = read_csv(MASKED_RESULTS / "run_summaries.csv")
    masked_traces = read_csv(MASKED_RESULTS / "training_traces.csv")
    masked_spectra = read_csv(MASKED_RESULTS / "saddle_spectra.csv")
    masked_spectrum_summaries = read_csv(
        MASKED_RESULTS / "saddle_spectrum_summaries.csv"
    )

    masked_tags = expected_masked_tags()
    masked_spectrum_tags = expected_masked_spectrum_tags()
    observed_masked_tags = {row["tag"] for row in masked_summaries}
    observed_masked_trace_tags = {row["tag"] for row in masked_traces}
    observed_spectrum_tags = {row["tag"] for row in masked_spectrum_summaries}
    check(
        "masked_campaign_manifest",
        masked_metadata.get("number_of_runs") == 93
        and masked_metadata.get("quick") is False
        and len(masked_summaries) == 87
        and observed_masked_tags == masked_tags
        and observed_masked_trace_tags == masked_tags
        and len(masked_spectrum_summaries) == 6
        and observed_spectrum_tags == masked_spectrum_tags
        and {row["tag"] for row in masked_spectra} == masked_spectrum_tags,
        {
            "metadata_runs": masked_metadata.get("number_of_runs"),
            "primary_runs": len(masked_summaries),
            "spectrum_runs": len(masked_spectrum_summaries),
            "missing_primary": sorted(masked_tags - observed_masked_tags),
            "unexpected_primary": sorted(observed_masked_tags - masked_tags),
            "missing_spectrum": sorted(
                masked_spectrum_tags - observed_spectrum_tags
            ),
            "unexpected_spectrum": sorted(
                observed_spectrum_tags - masked_spectrum_tags
            ),
        },
    )
    check(
        "masked_numeric_cells_finite",
        all_numeric_cells_finite(masked_summaries)
        and all_numeric_cells_finite(masked_traces)
        and all_numeric_cells_finite(masked_spectra)
        and all_numeric_cells_finite(masked_spectrum_summaries),
        {
            "summary_rows": len(masked_summaries),
            "trace_rows": len(masked_traces),
            "spectrum_rows": len(masked_spectra),
        },
    )
    masked_fourier_failures: list[str] = []
    for row in masked_traces:
        encoded = row.get("fourier_matrix", "")
        if not encoded:
            continue
        matrix = np.asarray(json.loads(encoded), dtype=float)
        symmetric = (
            matrix.ndim == 2
            and matrix.shape[0] == matrix.shape[1]
            and np.allclose(matrix, matrix.T, rtol=1.0e-6, atol=1.0e-9)
        )
        positive_semidefinite = symmetric and float(
            np.linalg.eigvalsh(matrix).min()
        ) >= -1.0e-7
        if (
            not np.all(np.isfinite(matrix))
            or not symmetric
            or not positive_semidefinite
        ):
            masked_fourier_failures.append(
                f"{row['tag']}|step={row['step']}"
            )
    check(
        "masked_fourier_matrices_finite_symmetric_psd",
        not masked_fourier_failures,
        masked_fourier_failures,
    )
    masked_traces_by_tag: dict[str, list[dict[str, str]]] = {}
    for row in masked_traces:
        masked_traces_by_tag.setdefault(row["tag"], []).append(row)
    masked_event_failures: list[str] = []
    for summary_row in masked_summaries:
        tag = summary_row["tag"]
        tag_traces = sorted(
            masked_traces_by_tag.get(tag, []),
            key=lambda row: float(row["time"]),
        )
        for key, value in summary_row.items():
            if not key.startswith(("t20_", "t50_", "t90_")):
                continue
            threshold_text, frequency = key.split("_", maxsplit=1)
            threshold = float(threshold_text.removeprefix("t")) / 100.0
            observed = first_numeric_crossing(
                tag_traces,
                time_key="time",
                value_key=f"recovery_{frequency}",
                threshold=threshold,
                direction="above",
            )
            recorded = None if value in ("", "None") else float(value)
            if observed != recorded:
                masked_event_failures.append(f"{tag}|{key}")
    check(
        "masked_event_times_recomputed_without_imputation",
        not masked_event_failures,
        masked_event_failures,
    )

    observed_tags = {row["tag"] for row in summaries}
    expected_tags = expected_campaign_tags()
    check(
        "campaign_manifest",
        observed_tags == expected_tags and len(summaries) == 184,
        {
            "observed_runs": len(summaries),
            "missing": sorted(expected_tags - observed_tags),
            "unexpected": sorted(observed_tags - expected_tags),
        },
    )
    check(
        "metadata_manifest",
        metadata.get("number_of_runs") == 184
        and metadata.get("quick") is False
        and metadata.get("dtype") == "torch.float32"
        and isinstance(metadata.get("torch_version"), str)
        and isinstance(metadata.get("numpy_version"), str),
        {
            "number_of_runs": metadata.get("number_of_runs"),
            "quick": metadata.get("quick"),
            "dtype": metadata.get("dtype"),
            "torch_version": metadata.get("torch_version"),
            "numpy_version": metadata.get("numpy_version"),
            "cuda_version": metadata.get("cuda_version"),
            "gpu_name": metadata.get("gpu_name"),
        },
    )
    unstable = [row["tag"] for row in summaries if row["stable"] != "True"]
    check("all_confirmation_runs_stable", not unstable, unstable)
    traces_by_tag: dict[str, list[dict[str, str]]] = {}
    for row in traces:
        traces_by_tag.setdefault(row["tag"], []).append(row)
    event_time_failures: list[str] = []
    for summary_row in summaries:
        tag = summary_row["tag"]
        tag_traces = sorted(
            traces_by_tag.get(tag, []), key=lambda row: float(row["step"])
        )
        for key, value in summary_row.items():
            if not key.startswith("t50_"):
                continue
            frequency = key.removeprefix("t50_")
            observed = first_numeric_crossing(
                tag_traces,
                time_key="step",
                value_key=f"relative_error_{frequency}",
                threshold=0.5,
                direction="below",
            )
            recorded = None if value in ("", "None") else float(value)
            if observed != recorded:
                event_time_failures.append(f"{tag}|{frequency}")
    check(
        "event_times_recomputed_without_imputation",
        not event_time_failures,
        event_time_failures,
    )
    stable_calibration = [row for row in calibration if row["stable"] == "True"]
    nonfinite_calibration = [
        row["tag"]
        for row in calibration
        if row["stable"] == "True"
        and not math.isfinite(float(row["calibration_score"]))
    ]
    check(
        "all_raw_numeric_cells_finite",
        not nonfinite_calibration
        and all_numeric_cells_finite(stable_calibration)
        and all_numeric_cells_finite(summaries)
        and all_numeric_cells_finite(traces)
        and all_json_numbers_finite(analysis),
        {
            "calibration_rows": len(calibration),
            "stable_calibration_rows": len(stable_calibration),
            "nonfinite_stable_calibration": nonfinite_calibration,
            "summary_rows": len(summaries),
            "trace_rows": len(traces),
        },
    )
    observed_calibration_tags = {row["tag"] for row in calibration}
    calibration_manifest = expected_calibration_tags()
    check(
        "calibration_manifest",
        len(calibration) == 72 and observed_calibration_tags == calibration_manifest,
        {
            "runs": len(calibration),
            "missing": sorted(calibration_manifest - observed_calibration_tags),
            "unexpected": sorted(observed_calibration_tags - calibration_manifest),
        },
    )

    calibration_seeds = {int(row["seed"]) for row in calibration}
    confirmation_seeds = {
        int(row["seed"])
        for row in summaries
        if row["tag"].startswith(("hierarchy_", "powerlaw_"))
    }
    check(
        "calibration_confirmation_seed_separation",
        calibration_seeds == {101, 102}
        and calibration_seeds.isdisjoint(confirmation_seeds),
        {
            "calibration": sorted(calibration_seeds),
            "confirmation": sorted(confirmation_seeds),
        },
    )
    paired_failures: dict[str, list[int]] = {}
    for target in ("hierarchy", "powerlaw"):
        for architecture in ("fc", "mmnn"):
            for optimizer in OPTIMIZERS:
                seeds = sorted(
                    int(row["seed"])
                    for row in summaries
                    if row["tag"].startswith(
                        f"{target}_{architecture}_d5_{optimizer}_s"
                    )
                )
                if seeds != list(range(8)):
                    paired_failures[f"{target}|{architecture}|{optimizer}"] = seeds
    check("paired_confirmation_seeds", not paired_failures, paired_failures)

    backend_failures = [
        row["tag"]
        for row in summaries
        if row["optimizer"] != "mup_gd"
        and (
            row.get("spectral_backend") != "direct_torch_svd"
            or row.get("spectral_cuda_driver") != "gesvd"
            or float(row.get("spectral_relative_floor", "nan")) != 1.0e-7
        )
    ]
    check(
        "direct_svd_backend",
        metadata.get("spectral_backend") == "direct_torch_svd"
        and metadata.get("spectral_cuda_driver") == "gesvd"
        and metadata.get("spectral_relative_floor") == 1.0e-7
        and not backend_failures,
        backend_failures,
    )
    polar_audit = backend_audit["aggregate"]["p=0"]
    check(
        "gram_shortcut_rejected_quantitatively",
        polar_audit["median_cosine"] < 0.8
        and polar_audit["median_relative_error"] > 0.5,
        polar_audit,
    )

    learning_rate_failures: list[str] = []
    for row in summaries:
        key = f"{row['architecture']}|{row['affine_depth']}|{row['optimizer']}"
        if row["tag"].startswith("width_"):
            key = f"{row['architecture']}|5|{row['optimizer']}"
        if not math.isclose(
            float(row["learning_rate"]),
            float(learning_rates[key]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            learning_rate_failures.append(row["tag"])
    check(
        "selected_rates_and_width_reuse",
        len(learning_rates) == 12 and not learning_rate_failures,
        learning_rate_failures,
    )
    calibration_selection_failures: list[str] = []
    for key, selected_rate in learning_rates.items():
        architecture, depth_text, optimizer = key.split("|")
        rows = [
            row
            for row in calibration
            if row["architecture"] == architecture
            and row["affine_depth"] == depth_text
            and row["optimizer"] == optimizer
        ]
        by_rate: dict[float, list[float]] = {}
        for row in rows:
            by_rate.setdefault(float(row["learning_rate"]), []).append(
                float(row["calibration_score"])
            )
        medians = {rate: float(np.median(scores)) for rate, scores in by_rate.items()}
        expected_rate = min(medians, key=medians.get)
        selected_rows = [
            row for row in rows if float(row["learning_rate"]) == float(selected_rate)
        ]
        if expected_rate != float(selected_rate) or not all(
            row["stable"] == "True" for row in selected_rows
        ):
            calibration_selection_failures.append(key)
    check(
        "calibration_selection_reproduced",
        not calibration_selection_failures,
        calibration_selection_failures,
    )

    feature_motion: dict[str, float] = {}
    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            values = [
                float(row["final_feature_displacement"])
                for row in summaries
                if row["tag"].startswith(
                    f"hierarchy_{architecture}_d{depth}_mup_gd_s"
                )
            ]
            feature_motion[f"{architecture}|{depth}"] = float(np.median(values))
    check(
        "order_one_feature_motion_at_all_depths",
        all(value > 0.1 for value in feature_motion.values()),
        feature_motion,
    )

    curvature_change: dict[str, float] = {}
    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            log_ratios: list[float] = []
            for seed in range(8):
                tag = f"hierarchy_{architecture}_d{depth}_mup_gd_s{seed}"
                rows = sorted(
                    (row for row in traces if row["tag"] == tag),
                    key=lambda row: float(row["step"]),
                )
                for frequency in (1, 4, 8, 16):
                    initial = float(rows[0][f"lambda_{frequency}"])
                    final = float(rows[-1][f"lambda_{frequency}"])
                    log_ratios.append(
                        abs(
                            math.log(
                                max(final, 1.0e-30) / max(initial, 1.0e-30)
                            )
                        )
                    )
            curvature_change[f"{architecture}|{depth}"] = float(
                np.max(log_ratios)
            )
    check(
        "dynamic_non_kernel_curvature_at_all_depths",
        all(value > math.log(1.2) for value in curvature_change.values()),
        curvature_change,
    )

    hierarchy_metrics: dict[str, Any] = {}
    hierarchy_failures: list[str] = []
    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            rows = [
                row
                for row in summaries
                if row["tag"].startswith(
                    f"hierarchy_{architecture}_d{depth}_mup_gd_s"
                )
            ]
            medians = {
                frequency: median_event(rows, frequency)
                for frequency in (1, 4, 8, 16)
            }
            coverage = {
                frequency: sum(
                    row.get(f"t50_{frequency}", "") not in ("", "None")
                    for row in rows
                )
                / len(rows)
                for frequency in (1, 4, 8, 16)
            }
            observed = [
                medians[frequency]
                for frequency in (1, 4, 8, 16)
                if medians[frequency] is not None
            ]
            passed = (
                len(rows) == 8
                and all(coverage[frequency] >= 0.5 for frequency in (1, 4, 8))
                and all(
                    later > earlier
                    for earlier, later in zip(observed, observed[1:])
                )
            )
            key = f"{architecture}|{depth}"
            hierarchy_metrics[key] = {
                "medians": {str(key): value for key, value in medians.items()},
                "coverage": {str(key): value for key, value in coverage.items()},
            }
            if not passed:
                hierarchy_failures.append(key)
    report["metrics"]["depth_hierarchy"] = hierarchy_metrics
    check(
        "ordered_full_training_hierarchy_at_all_depths",
        not hierarchy_failures,
        hierarchy_failures,
    )

    muon_advancement: dict[str, Any] = {}
    muon_advancement_failures: list[str] = []
    for target_kind, frequencies in (
        ("hierarchy", (8, 16)),
        ("powerlaw", tuple(range(8, 25))),
    ):
        for architecture in ("fc", "mmnn"):
            prefix = f"{target_kind}_{architecture}_d5_"
            baseline = [
                row
                for row in summaries
                if row["tag"].startswith(f"{prefix}mup_gd_s")
            ]
            improvements: dict[str, list[int]] = {}
            for optimizer in ("muon_p0", "muon_p1_3", "muon_p2_3"):
                compared = [
                    row
                    for row in summaries
                    if row["tag"].startswith(f"{prefix}{optimizer}_s")
                ]
                advanced: list[int] = []
                for frequency in frequencies:
                    baseline_time = median_event(baseline, frequency)
                    compared_time = median_event(compared, frequency)
                    if compared_time is not None and (
                        baseline_time is None or compared_time < baseline_time
                    ):
                        advanced.append(frequency)
                improvements[optimizer] = advanced
            key = f"{target_kind}|{architecture}"
            muon_advancement[key] = improvements
            if not any(improvements.values()):
                muon_advancement_failures.append(key)
    report["metrics"]["muon_clock_advancement"] = muon_advancement
    check(
        "spectral_power_advances_at_least_one_weak_sector",
        not muon_advancement_failures,
        muon_advancement_failures,
    )

    # Recompute every derived scalar used by the empirical narrative.  A fresh
    # timestamp is not sufficient evidence that an analysis file matches its
    # raw trajectories.
    analysis_consistency_failures: list[str] = []
    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            key = f"{architecture}|{depth}"
            recorded = analysis.get("depth_hierarchy", {}).get(key, {})
            raw = [
                row
                for row in summaries
                if row["tag"].startswith(
                    f"hierarchy_{architecture}_d{depth}_mup_gd_s"
                )
            ]
            expected_medians = {
                str(frequency): median_event(raw, frequency)
                for frequency in (1, 4, 8, 16)
            }
            expected_coverage = {
                str(frequency): sum(
                    row.get(f"t50_{frequency}", "") not in ("", "None")
                    for row in raw
                )
                / len(raw)
                for frequency in (1, 4, 8, 16)
            }
            for frequency in (1, 4, 8, 16):
                frequency_key = str(frequency)
                if not numeric_equal(
                    recorded.get("median_half_error_step", {}).get(frequency_key),
                    expected_medians[frequency_key],
                ):
                    analysis_consistency_failures.append(
                        f"depth_hierarchy|{key}|median|{frequency}"
                    )
                if not numeric_equal(
                    recorded.get("event_coverage", {}).get(frequency_key),
                    expected_coverage[frequency_key],
                ):
                    analysis_consistency_failures.append(
                        f"depth_hierarchy|{key}|coverage|{frequency}"
                    )
            if not numeric_equal(
                recorded.get("median_final_feature_displacement"),
                feature_motion[key],
            ):
                analysis_consistency_failures.append(
                    f"depth_hierarchy|{key}|feature_displacement"
                )

    for target_kind, frequencies in (
        ("hierarchy", (1, 4, 8, 16)),
        ("powerlaw", tuple(range(1, 25))),
    ):
        for architecture in ("fc", "mmnn"):
            baseline = {
                int(row["seed"]): row
                for row in summaries
                if row["tag"].startswith(
                    f"{target_kind}_{architecture}_d5_mup_gd_s"
                )
            }
            for optimizer in OPTIMIZERS:
                key = f"{target_kind}|{architecture}|{optimizer}"
                raw = [
                    row
                    for row in summaries
                    if row["tag"].startswith(
                        f"{target_kind}_{architecture}_d5_{optimizer}_s"
                    )
                ]
                recorded = analysis.get("sector_clocks", {}).get(key, {})
                for frequency in frequencies:
                    frequency_key = str(frequency)
                    expected_median = median_event(raw, frequency)
                    expected_coverage = sum(
                        row.get(f"t50_{frequency}", "") not in ("", "None")
                        for row in raw
                    ) / len(raw)
                    if not numeric_equal(
                        recorded.get("median_half_error_step", {}).get(
                            frequency_key
                        ),
                        expected_median,
                    ):
                        analysis_consistency_failures.append(
                            f"sector_clocks|{key}|median|{frequency}"
                        )
                    if not numeric_equal(
                        recorded.get("event_coverage", {}).get(frequency_key),
                        expected_coverage,
                    ):
                        analysis_consistency_failures.append(
                            f"sector_clocks|{key}|coverage|{frequency}"
                        )
                expected_feature_motion = float(
                    np.median(
                        [float(row["final_feature_displacement"]) for row in raw]
                    )
                )
                if not numeric_equal(
                    recorded.get("median_final_feature_displacement"),
                    expected_feature_motion,
                ):
                    analysis_consistency_failures.append(
                        f"sector_clocks|{key}|feature_displacement"
                    )

                if optimizer == "mup_gd":
                    continue
                compared = {int(row["seed"]): row for row in raw}
                seeds = sorted(set(baseline) & set(compared))
                differences = np.asarray(
                    [
                        math.log(max(float(compared[seed]["final_loss"]), 1.0e-30))
                        - math.log(
                            max(float(baseline[seed]["final_loss"]), 1.0e-30)
                        )
                        for seed in seeds
                    ]
                )
                paired = analysis.get("paired_endpoint", {}).get(key, {})
                if (
                    paired.get("pairs") != len(seeds)
                    or paired.get("wins") != int(np.sum(differences < 0.0))
                    or not numeric_equal(
                        paired.get("median_log_loss_difference"),
                        float(np.median(differences)),
                    )
                ):
                    analysis_consistency_failures.append(f"paired_endpoint|{key}")

    for architecture in ("fc", "mmnn"):
        for stage in ("initial", "final"):
            selected = [
                row
                for row in traces
                if row["tag"].startswith(
                    f"hierarchy_{architecture}_d5_mup_gd_s"
                )
            ]
            tags = sorted({row["tag"] for row in selected})
            values: list[list[float]] = []
            for tag in tags:
                tag_rows = sorted(
                    (row for row in selected if row["tag"] == tag),
                    key=lambda row: float(row["step"]),
                )
                row = tag_rows[0] if stage == "initial" else tag_rows[-1]
                values.append(
                    [float(row[f"lambda_{q}"]) for q in (1, 4, 8, 16)]
                )
            mean = np.mean(values, axis=0)
            slope = float(
                np.polyfit(
                    np.log(np.asarray((4, 8, 16), dtype=float)),
                    np.log(mean[1:]),
                    1,
                )[0]
            )
            recorded = analysis.get("curvature", {}).get(
                f"{architecture}|{stage}", {}
            )
            if not numeric_equal(recorded.get("slope_q_ge_4"), slope):
                analysis_consistency_failures.append(
                    f"curvature|{architecture}|{stage}|slope"
                )
            for frequency, value in zip((1, 4, 8, 16), mean, strict=True):
                if not numeric_equal(
                    recorded.get("mean", {}).get(str(frequency)), float(value)
                ):
                    analysis_consistency_failures.append(
                        f"curvature|{architecture}|{stage}|mean|{frequency}"
                    )

    check(
        "analysis_values_recomputed_from_raw_data",
        not analysis_consistency_failures,
        analysis_consistency_failures,
    )

    figure_stems = (
        "full_training_depth_hierarchy",
        "muon_hierarchy_clocks",
        "muon_powerlaw_front",
        "muon_paired_endpoints",
        "muon_mup_width_transfer",
        "full_training_dynamic_diagnostics",
        "full_training_step_convergence",
    )
    masked_figure_stems = (
        "hierarchy_and_kernel_evolution",
        "dynamic_saddle_spectrum",
        "frequency_rank_controls",
        "mup_width_collapse",
    )
    figure_paths = [
        FIGURES / f"{stem}.{suffix}"
        for stem in figure_stems
        for suffix in ("pdf", "png")
    ]
    absent_figures = [str(path) for path in figure_paths if not path.exists()]
    source_time = max(
        (RESULTS / "full_training_traces.csv").stat().st_mtime,
        (RESULTS / "full_training_summaries.csv").stat().st_mtime,
    )
    check(
        "analysis_source_consistency",
        analysis_path.stat().st_mtime >= source_time,
        {
            "analysis_mtime": analysis_path.stat().st_mtime,
            "source_mtime": source_time,
        },
    )
    stale_figures = [
        str(path)
        for path in figure_paths
        if path.exists() and path.stat().st_mtime < source_time
    ]
    check(
        "plot_source_consistency",
        not absent_figures and not stale_figures,
        {
            "absent": absent_figures,
            "stale": stale_figures,
        },
    )
    masked_source_time = max(
        (MASKED_RESULTS / "training_traces.csv").stat().st_mtime,
        (MASKED_RESULTS / "run_summaries.csv").stat().st_mtime,
        (MASKED_RESULTS / "saddle_spectra.csv").stat().st_mtime,
        (MASKED_RESULTS / "saddle_spectrum_summaries.csv").stat().st_mtime,
    )
    stale_masked_figures = [
        str(FIGURES / f"{stem}.{suffix}")
        for stem in masked_figure_stems
        for suffix in ("pdf", "png")
        if not (FIGURES / f"{stem}.{suffix}").exists()
        or (FIGURES / f"{stem}.{suffix}").stat().st_mtime < masked_source_time
    ]
    check(
        "masked_plot_source_consistency",
        not stale_masked_figures,
        stale_masked_figures,
    )
    mismatched_paper_figures = [
        stem
        for stem in figure_stems
        if not (FIGURES / f"{stem}.pdf").exists()
        or not (PAPER / "figures" / f"{stem}.pdf").exists()
        or (
            sha256(FIGURES / f"{stem}.pdf") != sha256(PAPER / "figures" / f"{stem}.pdf")
        )
    ]
    check(
        "paper_figure_consistency",
        not mismatched_paper_figures,
        mismatched_paper_figures,
    )
    mismatched_masked_paper_figures = [
        stem
        for stem in masked_figure_stems
        if not (FIGURES / f"{stem}.pdf").exists()
        or not (PAPER / "figures" / f"{stem}.pdf").exists()
        or sha256(FIGURES / f"{stem}.pdf")
        != sha256(PAPER / "figures" / f"{stem}.pdf")
    ]
    check(
        "masked_paper_figure_consistency",
        not mismatched_masked_paper_figures,
        mismatched_masked_paper_figures,
    )

    paper_source = (PAPER / "main.tex").read_text()
    forbidden = (r"\begin{table", r"\begin{tabular")
    check(
        "paper_has_no_tables",
        not any(token in paper_source for token in forbidden),
        [token for token in forbidden if token in paper_source],
    )
    paper_dependencies = [
        PAPER / "main.tex",
        PAPER / "references.bib",
        *[
            PAPER / "figures" / f"{stem}.pdf"
            for stem in (*figure_stems, *masked_figure_stems)
        ],
    ]
    newest_paper_dependency = max(
        path.stat().st_mtime for path in paper_dependencies if path.exists()
    )
    check(
        "paper_pdf_exists",
        (PAPER / "main.pdf").exists()
        and (PAPER / "main.pdf").stat().st_size > 0
        and (PAPER / "main.pdf").stat().st_mtime >= newest_paper_dependency,
        {
            "pdf": str(PAPER / "main.pdf"),
            "newest_dependency_mtime": newest_paper_dependency,
        },
    )

    if args.require_discretization:
        discretization_path = RESULTS / "discretization_summaries.csv"
        discretization_trace_path = RESULTS / "discretization_traces.csv"
        discretization_outputs_exist = (
            discretization_path.exists() and discretization_trace_path.exists()
        )
        check(
            "discretization_output_exists",
            discretization_outputs_exist,
            [str(discretization_path), str(discretization_trace_path)],
        )
        if discretization_outputs_exist:
            discretization = read_csv(discretization_path)
            unstable_discretization = [
                row["tag"] for row in discretization if row["stable"] != "True"
            ]
            observed_discretization_tags = {row["tag"] for row in discretization}
            discretization_manifest = expected_discretization_tags()
            manifest_passed = (
                len(discretization) == 60
                and observed_discretization_tags == discretization_manifest
            )
            check(
                "discretization_stability_and_manifest",
                manifest_passed and not unstable_discretization,
                {
                    "runs": len(discretization),
                    "unstable": unstable_discretization,
                    "missing": sorted(
                        discretization_manifest - observed_discretization_tags
                    ),
                    "unexpected": sorted(
                        observed_discretization_tags - discretization_manifest
                    ),
                },
            )
            if manifest_passed:
                (
                    base_half_ratios,
                    half_quarter_ratios,
                    quarter_eighth_ratios,
                    grid_ratios,
                ) = matched_discretization_log_ratios(discretization)
                discretization_by_tag = {
                    row["tag"]: row for row in discretization
                }
                grouped_base_half: list[float] = []
                grouped_half_quarter: list[float] = []
                grouped_quarter_eighth: list[float] = []
                grouped_grid: list[float] = []
                for architecture in ("fc", "mmnn"):
                    for optimizer in ("mup_gd", "muon_p1_3"):
                        base_half_group: list[float] = []
                        half_quarter_group: list[float] = []
                        quarter_eighth_group: list[float] = []
                        grid_group: list[float] = []
                        for seed in range(30, 33):
                            prefix = f"dt_{architecture}_{optimizer}"
                            base = float(
                                discretization_by_tag[
                                    f"{prefix}_base_s{seed}"
                                ]["final_loss"]
                            )
                            half = float(
                                discretization_by_tag[
                                    f"{prefix}_half_s{seed}"
                                ]["final_loss"]
                            )
                            quarter = float(
                                discretization_by_tag[
                                    f"{prefix}_quarter_s{seed}"
                                ]["final_loss"]
                            )
                            eighth = float(
                                discretization_by_tag[
                                    f"{prefix}_eighth_s{seed}"
                                ]["final_loss"]
                            )
                            fine = float(
                                discretization_by_tag[
                                    f"grid_{architecture}_{optimizer}_g256_s{seed}"
                                ]["final_loss"]
                            )
                            base_half_group.append(abs(math.log(half / base)))
                            half_quarter_group.append(
                                abs(math.log(quarter / half))
                            )
                            quarter_eighth_group.append(
                                abs(math.log(eighth / quarter))
                            )
                            grid_group.append(abs(math.log(fine / base)))
                        grouped_base_half.append(
                            float(np.median(base_half_group))
                        )
                        grouped_half_quarter.append(
                            float(np.median(half_quarter_group))
                        )
                        grouped_quarter_eighth.append(
                            float(np.median(quarter_eighth_group))
                        )
                        grouped_grid.append(float(np.median(grid_group)))
                report["metrics"]["discretization"] = {
                    "median_abs_log_base_half_ratio": float(
                        np.median(base_half_ratios)
                    ),
                    "median_abs_log_half_quarter_ratio": float(
                        np.median(half_quarter_ratios)
                    ),
                    "median_abs_log_quarter_eighth_ratio": float(
                        np.median(quarter_eighth_ratios)
                    ),
                    "median_abs_log_grid_ratio": float(np.median(grid_ratios)),
                    "max_group_median_abs_log_base_half_ratio": max(
                        grouped_base_half
                    ),
                    "max_group_median_abs_log_half_quarter_ratio": max(
                        grouped_half_quarter
                    ),
                    "max_group_median_abs_log_quarter_eighth_ratio": max(
                        grouped_quarter_eighth
                    ),
                    "max_group_median_abs_log_grid_ratio": max(grouped_grid),
                    "max_group_median_abs_log_quarter_eighth_residual_norm_ratio": (
                        0.5 * max(grouped_quarter_eighth)
                    ),
                    "max_group_median_abs_log_grid_residual_norm_ratio": (
                        0.5 * max(grouped_grid)
                    ),
                }
                check(
                    "refined_step_and_grid_residual_norm_agreement",
                    0.5 * max(grouped_quarter_eighth) <= math.log(2.0)
                    and 0.5 * max(grouped_grid) <= math.log(2.0),
                    report["metrics"]["discretization"],
                )
                clock_metrics: dict[str, Any] = {}
                ordering_failures: list[str] = []
                acceleration_failures: list[str] = []
                for architecture in ("fc", "mmnn"):
                    for variant in (
                        "base",
                        "half",
                        "quarter",
                        "eighth",
                        "grid",
                    ):
                        by_optimizer: dict[str, dict[int, float | None]] = {}
                        for optimizer in ("mup_gd", "muon_p1_3"):
                            if variant == "grid":
                                prefix = (
                                    f"grid_{architecture}_{optimizer}_g256_s"
                                )
                            else:
                                prefix = (
                                    f"dt_{architecture}_{optimizer}_{variant}_s"
                                )
                            rows = [
                                row
                                for row in discretization
                                if row["tag"].startswith(prefix)
                            ]
                            medians = {
                                frequency: median_event(rows, frequency)
                                for frequency in (4, 8, 16, 24)
                            }
                            by_optimizer[optimizer] = medians
                            observed_core = [
                                medians[frequency] for frequency in (4, 8, 16)
                            ]
                            key = f"{architecture}|{optimizer}|{variant}"
                            clock_metrics[key] = {
                                "median_half_error_step": {
                                    str(frequency): medians[frequency]
                                    for frequency in (4, 8, 16, 24)
                                }
                            }
                            if any(value is None for value in observed_core) or not all(
                                later > earlier
                                for earlier, later in zip(
                                    observed_core, observed_core[1:]
                                )
                            ):
                                ordering_failures.append(key)
                        baseline = by_optimizer["mup_gd"]
                        compared = by_optimizer["muon_p1_3"]
                        advances = any(
                            compared[frequency] is not None
                            and (
                                baseline[frequency] is None
                                or compared[frequency] < baseline[frequency]
                            )
                            for frequency in (8, 16, 24)
                        )
                        clock_metrics[f"{architecture}|comparison|{variant}"] = {
                            "muon_p1_3_advances_a_weak_sector": advances
                        }
                        if architecture == "fc" and not advances:
                            acceleration_failures.append(
                                f"{architecture}|{variant}"
                            )
                report["metrics"]["discretization_clocks"] = clock_metrics
                check(
                    "discretization_preserves_core_frequency_order",
                    not ordering_failures,
                    ordering_failures,
                )
                check(
                    "dense_muon_clock_advancement_survives_controls",
                    not acceleration_failures,
                    acceleration_failures,
                )
                control_source_time = max(
                    discretization_path.stat().st_mtime,
                    discretization_trace_path.stat().st_mtime,
                )
                control_figures = [
                    FIGURES / f"full_training_step_convergence.{suffix}"
                    for suffix in ("pdf", "png")
                ]
                check(
                    "discretization_plot_source_consistency",
                    all(
                        path.exists() and path.stat().st_mtime >= control_source_time
                        for path in control_figures
                    ),
                    [str(path) for path in control_figures],
                )

    workspace = ROOT.parents[1]
    artifact_paths = [
        *required,
        MASKED_RESULTS / "metadata.json",
        MASKED_RESULTS / "run_summaries.csv",
        MASKED_RESULTS / "training_traces.csv",
        MASKED_RESULTS / "saddle_spectra.csv",
        MASKED_RESULTS / "saddle_spectrum_summaries.csv",
        *figure_paths,
        *[
            FIGURES / f"{stem}.{suffix}"
            for stem in masked_figure_stems
            for suffix in ("pdf", "png")
        ],
        *paper_dependencies,
        PAPER / "main.tex",
        PAPER / "references.bib",
        PAPER / "main.pdf",
        ROOT / "run_study.py",
        ROOT / "run_full_training_muon.py",
        ROOT / "audit_spectral_backend.py",
        ROOT / "audit_full_training_results.py",
        workspace / "model" / "mmnn" / "mup_right_factor.py",
        workspace / "model" / "mmnn" / "full_training_frequency.py",
        workspace / "model" / "mmnn" / "spectral_power.py",
    ]
    report["sha256"] = {
        str(path.relative_to(workspace)): sha256(path)
        for path in artifact_paths
        if path.exists()
    }
    report["passed"] = not report["failures"]
    output = RESULTS / "full_training_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
