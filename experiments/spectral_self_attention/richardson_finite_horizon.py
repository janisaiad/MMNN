"""Richardson cancellation of the leading finite-layer trajectory error."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.large_scale_cycle_census import map_angles, wrap
from experiments.spectral_self_attention.small_step_continuation import stack_models


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q90": float(np.quantile(values, 0.9)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def fitted_order(ratios: list[float], values: list[float]) -> float:
    return float(np.polyfit(np.log(ratios), np.log(values), 1)[0])


def run(
    inputs: list[Path],
    ratios: list[float],
    horizon: float,
    reference_dt: float,
) -> dict[str, object]:
    records = [
        {"family": int(payload["family"]), "label": str(payload["label"]), **record}
        for path in inputs
        for payload in [json.loads(path.read_text())]
        for record in payload["records"]
    ]
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[(int(record["family"]), int(record["identity"]["n_tokens"]))].append(
            record
        )

    raw: dict[tuple[int, float], list[np.ndarray]] = defaultdict(list)
    accelerated: dict[tuple[int, float], list[np.ndarray]] = defaultdict(list)
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
            reference = rk4_step(reference, models, original_steps, effective_dt)

        endpoints: dict[float, np.ndarray] = {}
        for ratio in sorted(set(ratios) | {ratio / 2.0 for ratio in ratios}, reverse=True):
            layers = int(round(horizon / ratio))
            if not np.isclose(layers * ratio, horizon, atol=1e-12, rtol=0.0):
                raise ValueError("horizon must be divisible by every ratio")
            models["step_size"] = original_steps * ratio
            discrete = initial.copy()
            for _ in range(layers):
                discrete = map_angles(discrete, models)
            endpoints[ratio] = discrete

        for ratio in ratios:
            coarse_error = wrap(endpoints[ratio] - reference)[:, 0]
            fine_error = wrap(endpoints[ratio / 2.0] - reference)[:, 0]
            raw_error = np.sqrt(np.mean(fine_error**2, axis=-1))
            richardson_error = np.sqrt(
                np.mean(wrap(2.0 * fine_error - coarse_error) ** 2, axis=-1)
            )
            raw[(family, ratio)].append(raw_error)
            accelerated[(family, ratio)].append(richardson_error)
        models["step_size"] = original_steps

    rows = []
    all_raw: dict[float, list[np.ndarray]] = defaultdict(list)
    all_accelerated: dict[float, list[np.ndarray]] = defaultdict(list)
    for family in sorted({family for family, _ in groups}):
        family_raw_medians = []
        family_accelerated_medians = []
        family_rows = []
        for ratio in ratios:
            raw_values = np.concatenate(raw[(family, ratio)])
            accelerated_values = np.concatenate(accelerated[(family, ratio)])
            all_raw[ratio].append(raw_values)
            all_accelerated[ratio].append(accelerated_values)
            family_raw_medians.append(float(np.median(raw_values)))
            family_accelerated_medians.append(float(np.median(accelerated_values)))
            family_rows.append(
                {
                    "family": family,
                    "coarse_step_ratio": ratio,
                    "fine_step_ratio": ratio / 2.0,
                    "records": len(raw_values),
                    "fine_error": summary(raw_values),
                    "richardson_error": summary(accelerated_values),
                }
            )
        raw_order = fitted_order(ratios, family_raw_medians)
        accelerated_order = fitted_order(ratios, family_accelerated_medians)
        for row in family_rows:
            row["family_fine_error_order"] = raw_order
            row["family_richardson_error_order"] = accelerated_order
        rows.extend(family_rows)

    aggregate = []
    raw_medians = []
    accelerated_medians = []
    for ratio in ratios:
        raw_values = np.concatenate(all_raw[ratio])
        accelerated_values = np.concatenate(all_accelerated[ratio])
        raw_medians.append(float(np.median(raw_values)))
        accelerated_medians.append(float(np.median(accelerated_values)))
        aggregate.append(
            {
                "coarse_step_ratio": ratio,
                "fine_step_ratio": ratio / 2.0,
                "records": len(raw_values),
                "fine_error": summary(raw_values),
                "richardson_error": summary(accelerated_values),
                "median_improvement_factor": float(
                    np.median(raw_values) / np.median(accelerated_values)
                ),
            }
        )
    return {
        "settings": {
            "ratios": ratios,
            "horizon_normalized": horizon,
            "reference_rk4_dt": reference_dt,
        },
        "summary": {
            "records": len(records),
            "fine_error_order": fitted_order(ratios, raw_medians),
            "richardson_error_order": fitted_order(
                ratios, accelerated_medians
            ),
        },
        "aggregate": aggregate,
        "by_family": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--ratios", default=".0625,.03125,.015625,.0078125,.00390625,.001953125"
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
