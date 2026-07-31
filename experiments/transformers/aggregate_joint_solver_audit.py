#!/usr/bin/env python3
"""Aggregate the controlled joint encoder/controller audit.

The input root is produced by the paired commands documented in
``THREE_CONTROLLER_ARCHITECTURE.md``.  The script deliberately reads only the
held-out ``final`` rows and the paired controller summaries; training logs and
checkpoints remain outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


CONTROLLERS = (
    "exact",
    "learned_hb",
    "richardson_same_step",
    "oracle_richardson",
    "oracle_chebyshev",
    "pcg",
)
SCENARIOS = ("nominal", "ood")
SEEDS = (0, 1, 2)
POLICIES = (
    ("joint_training", "cross"),
    ("tail_calibrated", "robust_cross"),
    ("spectral_prediction", "spectral_predicted_matched_cross"),
)


def final_training_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["tag"] == "final"]
    if len(matches) != 1:
        raise ValueError(f"expected one final row in {path}, found {len(matches)}")
    return matches[0]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_training(root: Path) -> list[dict]:
    rows = []
    for controller in ("heavy_ball", "richardson"):
        for seed in SEEDS:
            source = (
                root
                / f"{controller}_seed{seed}_step1000"
                / "eval_metrics.csv"
            )
            row = final_training_row(source)
            rows.append(
                {
                    "seed": seed,
                    "controller": controller,
                    "steps": row["step"],
                    "depth": 40,
                    "eval_examples": 2048,
                    "u_mse": row["u_mse"],
                    "u_relative": row["u_rel"],
                    "operator_relative": row["A_rel"],
                    "subspace_overlap": row["Abasis_subspace_overlap"],
                    "step_scale": row["decoder_step_scale"],
                    "momentum": row["decoder_momentum"],
                }
            )
    return rows


def aggregate_controllers(root: Path) -> list[dict]:
    rows = []
    for policy, prefix in POLICIES:
        for scenario in SCENARIOS:
            seed_rows = []
            for seed in SEEDS:
                source = root / f"{prefix}_seed{seed}_{scenario}" / "summary.json"
                result = json.loads(source.read_text())
                pcg_mse = result["controllers"]["pcg"]["u_mse"]["mean"]
                for controller in CONTROLLERS:
                    metrics = result["controllers"][controller]
                    row = {
                        "seed": seed,
                        "policy": policy,
                        "scenario": scenario,
                        "z_scale": result["z_scale"],
                        "controller": controller,
                        "hb_or_polynomial_depth": result["hb_depth"],
                        "pcg_depth": result["pcg_depth"],
                        "examples": result["examples"],
                        "u_mse": metrics["u_mse"]["mean"],
                        "u_mse_ci95_halfwidth": metrics["u_mse"]["ci95_halfwidth"],
                        "solver_u_mse": metrics["solver_u_mse"]["mean"],
                        "u_mse_over_pcg": metrics["u_mse"]["mean"] / pcg_mse,
                        "jury_margin_min": result["heavy_ball_jury_margin_min"],
                    }
                    rows.append(row)
                    seed_rows.append(row)
            for controller in CONTROLLERS:
                selected = [r for r in seed_rows if r["controller"] == controller]
                pcg_selected = [r for r in seed_rows if r["controller"] == "pcg"]
                mean_u_mse = fmean(float(r["u_mse"]) for r in selected)
                mean_pcg_mse = fmean(float(r["u_mse"]) for r in pcg_selected)
                rows.append(
                    {
                        "seed": "mean",
                        "policy": policy,
                        "scenario": scenario,
                        "z_scale": selected[0]["z_scale"],
                        "controller": controller,
                        "hb_or_polynomial_depth": selected[0]["hb_or_polynomial_depth"],
                        "pcg_depth": selected[0]["pcg_depth"],
                        "examples": sum(int(r["examples"]) for r in selected),
                        "u_mse": mean_u_mse,
                        "u_mse_ci95_halfwidth": "NA",
                        "solver_u_mse": fmean(float(r["solver_u_mse"]) for r in selected),
                        "u_mse_over_pcg": mean_u_mse / mean_pcg_mse,
                        "jury_margin_min": min(float(r["jury_margin_min"]) for r in selected),
                    }
                )
    return rows


def aggregate_predictive_hyperparameters(root: Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        prediction = json.loads(
            (root / f"spectral_predicted_hb_seed{seed}" / "summary.json").read_text()
        )
        held_out = json.loads(
            (root / f"spectral_predicted_cross_seed{seed}_ood" / "summary.json").read_text()
        )
        adam = json.loads(
            (root / f"robust_cross_seed{seed}_ood_new" / "summary.json").read_text()
        )
        predicted_hb = held_out["controllers"]["learned_hb"]
        predicted_pcg = held_out["controllers"]["pcg"]
        adam_hb = adam["controllers"]["learned_hb"]
        adam_pcg = adam["controllers"]["pcg"]
        rows.append(
            {
                "seed": seed,
                "calibration_tasks": prediction["calibration_tasks"],
                "held_out_tasks": held_out["examples"],
                "depth": prediction["depth"],
                "spectral_min": prediction["spectral_min"],
                "spectral_max": prediction["spectral_max"],
                "minimax_step": prediction["minimax_step"],
                "minimax_momentum": prediction["minimax_momentum"],
                "predicted_step": prediction["predicted"]["step"],
                "predicted_momentum": prediction["predicted"]["momentum"],
                "predicted_hb_over_pcg": (
                    predicted_hb["u_mse"]["mean"] / predicted_pcg["u_mse"]["mean"]
                ),
                "adam_hb_over_pcg": (
                    adam_hb["u_mse"]["mean"] / adam_pcg["u_mse"]["mean"]
                ),
                "predicted_solver_u_mse": predicted_hb["solver_u_mse"]["mean"],
                "adam_solver_u_mse": adam_hb["solver_u_mse"]["mean"],
                "solver_mse_reduction_vs_adam": (
                    1.0
                    - predicted_hb["solver_u_mse"]["mean"]
                    / adam_hb["solver_u_mse"]["mean"]
                ),
                "jury_margin_min": held_out["heavy_ball_jury_margin_min"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    write_rows(
        args.outdir / "joint_solver_training_results.csv",
        aggregate_training(args.root),
    )
    write_rows(
        args.outdir / "joint_learned_space_controller_results.csv",
        aggregate_controllers(args.root),
    )
    write_rows(
        args.outdir / "predictive_hb_hyperparameter_results.csv",
        aggregate_predictive_hyperparameters(args.root),
    )


if __name__ == "__main__":
    main()
