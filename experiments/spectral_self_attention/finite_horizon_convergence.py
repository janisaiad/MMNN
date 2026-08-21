"""Whole-cohort finite-horizon convergence of layers to the continuous flow."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.large_scale_cycle_census import (
    map_angles,
    wrap,
)
from experiments.spectral_self_attention.small_step_continuation import stack_models


def run(
    inputs: list[Path],
    ratios: list[float],
    horizon: float,
    reference_dt: float,
) -> dict[str, object]:
    records = [
        {
            "family": int(payload["family"]),
            "label": str(payload["label"]),
            **record,
        }
        for path in inputs
        for payload in [json.loads(path.read_text())]
        for record in payload["records"]
    ]
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[
            (int(record["family"]), int(record["identity"]["n_tokens"]))
        ].append(record)
    errors: dict[tuple[int, float], list[np.ndarray]] = defaultdict(list)
    all_errors: dict[float, list[np.ndarray]] = defaultdict(list)
    identified_errors: dict[float, list[dict[str, object]]] = defaultdict(list)
    for (family, _), selected in groups.items():
        models = stack_models(selected)
        original_steps = models["step_size"].astype(float).copy()
        initial = np.asarray(
            [record["initial_angle"] for record in selected], dtype=float
        )[:, None, :]
        reference = initial.copy()
        reference_steps = int(np.ceil(horizon / reference_dt))
        effective_dt = horizon / reference_steps
        for _ in range(reference_steps):
            reference = rk4_step(
                reference, models, original_steps, effective_dt
            )
        for ratio in ratios:
            models["step_size"] = original_steps * ratio
            discrete = initial.copy()
            layers = int(round(horizon / ratio))
            if not np.isclose(layers * ratio, horizon, atol=1e-12, rtol=0.0):
                raise ValueError("horizon must be an integer multiple of every ratio")
            for _ in range(layers):
                discrete = map_angles(discrete, models)
            error = np.sqrt(np.mean(wrap(discrete - reference)[:, 0, :] ** 2, axis=-1))
            errors[(family, ratio)].append(error)
            all_errors[ratio].append(error)
            identified_errors[ratio].extend(
                {
                    "family": family,
                    "label": str(record["label"]),
                    "identity": record["identity"],
                    "angular_rms_error": float(value),
                }
                for record, value in zip(selected, error)
            )
        models["step_size"] = original_steps

    rows = []
    families = sorted({family for family, _ in groups})
    for family in families:
        medians = []
        for ratio in ratios:
            values = np.concatenate(errors[(family, ratio)])
            median = float(np.median(values))
            medians.append(median)
            rows.append(
                {
                    "family": family,
                    "step_ratio": ratio,
                    "records": len(values),
                    "median_error": median,
                    "mean_error": float(np.mean(values)),
                    "q90_error": float(np.quantile(values, 0.9)),
                    "maximum_error": float(np.max(values)),
                }
            )
        slope, _ = np.polyfit(np.log(ratios), np.log(medians), 1)
        for row in rows:
            if row["family"] == family:
                row["family_median_error_order"] = float(slope)
    aggregate = []
    aggregate_medians = []
    for ratio in ratios:
        values = np.concatenate(all_errors[ratio])
        median = float(np.median(values))
        aggregate_medians.append(median)
        aggregate.append(
            {
                "step_ratio": ratio,
                "records": len(values),
                "median_error": median,
                "mean_error": float(np.mean(values)),
                "q90_error": float(np.quantile(values, 0.9)),
                "maximum_error": float(np.max(values)),
                "count_above_0_01": int(np.sum(values > 1e-2)),
                "count_above_0_1": int(np.sum(values > 1e-1)),
                "count_above_1": int(np.sum(values > 1.0)),
            }
        )
    aggregate_slope, _ = np.polyfit(
        np.log(ratios), np.log(aggregate_medians), 1
    )
    return {
        "settings": {
            "ratios": ratios,
            "horizon_normalized": horizon,
            "reference_rk4_dt": reference_dt,
        },
        "summary": {
            "records": len(records),
            "aggregate_median_error_order": float(aggregate_slope),
            "family_median_error_orders": {
                str(family): next(
                    row["family_median_error_order"]
                    for row in rows
                    if row["family"] == family
                )
                for family in families
            },
        },
        "aggregate": aggregate,
        "by_family": rows,
        "largest_errors": {
            str(ratio): sorted(
                identified_errors[ratio],
                key=lambda row: float(row["angular_rms_error"]),
                reverse=True,
            )[:20]
            for ratio in ratios
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--ratios",
        default=".0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625",
    )
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--reference-dt", type=float, default=0.001)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        args.inputs,
        [float(value) for value in args.ratios.split(",")],
        args.horizon,
        args.reference_dt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
