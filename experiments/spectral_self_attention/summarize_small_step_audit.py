"""Summarize the small-step continuation and direct-ODE audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return [0.0 if lower < 1e-15 else lower, 1.0 if 1.0 - upper < 1e-15 else upper]


def source_name(path: Path, payload: dict[str, object]) -> str:
    if "beta0" in path.stem:
        return "type1_beta0"
    return f"type{payload['family']}"


def local_error_order(trace: list[dict[str, object]]) -> float:
    tail = trace[-6:]
    ratios = np.asarray([float(point["ratio"]) for point in tail])
    errors = np.asarray([float(point["absolute_local_error"]) for point in tail])
    valid = np.isfinite(errors) & (errors > 1e-13)
    if valid.sum() < 3:
        return math.nan
    return float(np.polyfit(np.log(ratios[valid]), np.log(errors[valid]), 1)[0])


def classify_limit(trace: list[dict[str, object]]) -> str:
    tail = trace[-3:]
    moving = [float(point["motion_per_normalized_time"]) >= 1e-3 for point in tail]
    recurrent = [bool(point["moving_recurrent"]) for point in tail]
    positive = [bool(point["positive_lyapunov"]) for point in tail]
    stationary = [bool(point["map_stationary"]) for point in tail]
    internal = [float(point["gram_variation"]) >= 1e-3 for point in tail]
    if all(stationary) and any(positive):
        return "unstable_equilibrium_lock"
    if all(stationary) and not any(positive):
        return "collapsed_to_equilibrium_branch"
    if all(moving) and all(recurrent) and sum(internal) >= 2:
        periods = np.asarray([float(point["return_time_normalized"]) for point in tail])
        if np.mean(periods) > 0 and np.std(periods) / np.mean(periods) < 0.25:
            return "robust_internal_cycle"
    if all(moving) and all(recurrent):
        return "robust_rigid_rotation"
    if all(moving) and all(positive):
        return "robust_positive_lyapunov"
    if sum(moving) >= 2:
        return "moving_shape_dynamics" if sum(internal) >= 2 else "moving_unresolved"
    return "transition_unresolved"


def count_with_interval(values: list[bool]) -> dict[str, object]:
    successes = sum(values)
    total = len(values)
    return {
        "count": successes,
        "total": total,
        "fraction": successes / total if total else math.nan,
        "wilson_95": wilson_interval(successes, total),
    }


def summarize_continuations(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    curves = []
    record_rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        source = source_name(path, payload)
        records = payload["records"]
        for record in records:
            trace = record["trace"]
            identity = record["identity"]
            record_rows.append(
                {
                    "source": source,
                    "label": payload["label"],
                    "n_tokens": int(identity["n_tokens"]),
                    "subtype_code": int(identity["subtype_code"]),
                    "source_model_index": int(identity["source_model_index"]),
                    "fate": classify_limit(trace),
                    "local_error_order": local_error_order(trace),
                    "final_motion": float(trace[-1]["motion_per_normalized_time"]),
                    "final_field": float(trace[-1]["continuous_field_rms"]),
                    "final_recurrence_error": float(trace[-1]["recurrence_error"]),
                    "final_return_time": float(trace[-1]["return_time_normalized"]),
                    "final_lyapunov": float(trace[-1]["lyapunov_per_normalized_time"]),
                    "final_gram_variation": float(trace[-1]["gram_variation"]),
                }
            )
        for ratio_index, ratio in enumerate(payload["settings"]["ratios"]):
            points = [record["trace"][ratio_index] for record in records]
            moving = [float(point["motion_per_normalized_time"]) >= 1e-3 for point in points]
            curves.append(
                {
                    "source": source,
                    "label": payload["label"],
                    "ratio": float(ratio),
                    "records": len(points),
                    "moving": count_with_interval(moving),
                    "moving_recurrent": count_with_interval(
                        [bool(point["moving_recurrent"]) for point in points]
                    ),
                    "positive_lyapunov": count_with_interval(
                        [bool(point["positive_lyapunov"]) for point in points]
                    ),
                    "median_motion": float(
                        np.median([point["motion_per_normalized_time"] for point in points])
                    ),
                    "median_local_error": float(
                        np.median([point["absolute_local_error"] for point in points])
                    ),
                    "median_relative_local_error": float(
                        np.median([point["relative_local_error"] for point in points])
                    ),
                }
            )
    return curves, record_rows


def grouped_record_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["label"])].append(row)
    output = []
    fates = sorted({row["fate"] for row in rows})
    for (source, label), selected in sorted(groups.items()):
        orders = np.asarray([row["local_error_order"] for row in selected])
        output.append(
            {
                "source": source,
                "label": label,
                "records": len(selected),
                "fates": {
                    fate: count_with_interval([row["fate"] == fate for row in selected])
                    for fate in fates
                },
                "median_local_error_order": float(np.nanmedian(orders)),
                "by_n_tokens": {
                    str(n_tokens): len(
                        [row for row in selected if row["n_tokens"] == n_tokens]
                    )
                    for n_tokens in sorted({row["n_tokens"] for row in selected})
                },
            }
        )
    return output


def summarize_odes(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        payload = json.loads(path.read_text())
        records = payload["records"]
        metrics = [row["metrics"] for row in records]
        moving = [row["motion_per_normalized_time"] >= 1e-3 for row in metrics]
        internal = [row["gram_variation"] >= 1e-3 for row in metrics]
        kernel_recurrent = [
            row.get("score_kernel_recurrence_error", math.inf) < 3e-2
            for row in metrics
        ]
        output.append(
            {
                "file": path.name,
                "start_mode": payload["settings"]["start_mode"],
                "summary": payload["summary"],
                "moving": count_with_interval(moving),
                "rigid_rotation": count_with_interval(
                    [is_moving and not is_internal for is_moving, is_internal in zip(moving, internal)]
                ),
                "internal_shape_motion": count_with_interval(
                    [is_moving and is_internal for is_moving, is_internal in zip(moving, internal)]
                ),
                "kernel_recurrent": count_with_interval(
                    [
                        is_moving and is_internal and is_recurrent
                        for is_moving, is_internal, is_recurrent in zip(
                            moving, internal, kernel_recurrent
                        )
                    ]
                ),
                "median_motion": float(
                    np.median(
                        [row["metrics"]["motion_per_normalized_time"] for row in records]
                    )
                ),
                "median_lyapunov": float(
                    np.median(
                        [row["metrics"]["lyapunov_per_normalized_time"] for row in records]
                    )
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("continuations", nargs="+", type=Path)
    parser.add_argument("--odes", nargs="*", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    curves, records = summarize_continuations(args.continuations)
    result = {
        "groups": grouped_record_summary(records),
        "curves": curves,
        "records": records,
        "direct_ode": summarize_odes(args.odes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
