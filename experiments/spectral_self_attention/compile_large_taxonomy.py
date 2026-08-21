"""Compile compact metrics from the large equilibrium/cycle audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUBTYPE_NAMES = {
    "0": "symmetric_score_and_value",
    "1": "symmetric_score_general_value",
    "2": "general_score_symmetric_value",
    "3": "fully_general",
}


def percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def condensed_counts(counts: dict[str, int]) -> dict[str, int | float]:
    basins = counts["basins"]
    high_period = sum(counts.get(f"p{period}", 0) for period in range(5, 13))
    return {
        "basins": basins,
        "fixed": counts.get("p1", 0),
        "period2": counts.get("p2", 0),
        "period3": counts.get("p3", 0),
        "period4": counts.get("p4", 0),
        "period5_to_12": high_period,
        "rigid_or_slow_rotation": counts.get("rotation", 0),
        "unresolved": counts.get("unresolved", 0),
        "period3_percent": percent(counts.get("p3", 0), basins),
        "period4_percent": percent(counts.get("p4", 0), basins),
        "any_period2_to_12_percent": percent(
            sum(counts.get(f"p{period}", 0) for period in range(2, 13)), basins
        ),
    }


def run(directory: Path) -> dict[str, object]:
    families: dict[str, object] = {}
    beta_zero: dict[str, object] = {}
    mlp_only: dict[str, object] = {}
    total_models = 0
    total_basins = 0
    total_p3 = 0
    total_p4 = 0
    for family in (1, 2, 3, 4):
        census = json.loads((directory / f"large_census_type{family}.json").read_text())
        lyapunov = json.loads((directory / f"lyapunov_type{family}.json").read_text())
        roots = json.loads((directory / f"root_census_type{family}.json").read_text())
        totals = condensed_counts(census["totals"])
        total_models += int(census["totals"]["models"])
        total_basins += int(census["totals"]["basins"])
        total_p3 += int(census["totals"]["p3"])
        total_p4 += int(census["totals"]["p4"])
        unresolved_lyapunov = lyapunov["by_attractor"].get("unresolved", {"count": 0})
        subtypes = {
            SUBTYPE_NAMES[key]: condensed_counts(value)
            for key, value in census["grouped"]["subtype_code"].items()
            if key in SUBTYPE_NAMES
        }
        families[str(family)] = {
            "name": census["family_name"],
            "totals": totals,
            "model_incidence": census["model_incidence"],
            "by_tokens": {
                key: condensed_counts(value) for key, value in census["by_tokens"].items()
            },
            "by_step_bin": {
                key: condensed_counts(value)
                for key, value in census["grouped"]["step_bin"].items()
            },
            "by_width": {
                key: condensed_counts(value)
                for key, value in census["grouped"]["width"].items()
            },
            "untied_subtypes": subtypes,
            "lyapunov_unresolved": unresolved_lyapunov,
            "fixed_root_census": {
                tokens: {
                    key: row[key]
                    for key in (
                        "all_roots",
                        "stable_roots",
                        "stable_irregular_roots",
                        "stable_spiral_roots",
                    )
                }
                for tokens, row in roots["by_tokens"].items()
            },
        }
        beta_zero_census = json.loads(
            (directory / f"beta0_census_type{family}.json").read_text()
        )
        beta_zero[str(family)] = condensed_counts(beta_zero_census["totals"])
        if family in (1, 2):
            mlp_only_census = json.loads(
                (directory / f"mlp_only_census_type{family}.json").read_text()
            )
            mlp_only[str(family)] = condensed_counts(mlp_only_census["totals"])
    rotation = json.loads((directory / "type3_continuous_rotation.json").read_text())
    certificates = json.loads((directory / "verified_periodic_orbits.json").read_text())
    bifurcations = json.loads((directory / "period_bifurcation_sweep.json").read_text())
    beta_zero_certificates = json.loads(
        (directory / "beta0_verified_periodic_orbits.json").read_text()
    )
    mlp_only_certificates = json.loads(
        (directory / "mlp_only_verified_periodic_orbits.json").read_text()
    )
    verified_lyapunov = json.loads(
        (directory / "verified_positive_lyapunov.json").read_text()
    )
    return {
        "global": {
            "sampled_models": total_models,
            "sampled_trajectories": total_basins,
            "period3_trajectories": total_p3,
            "period4_trajectories": total_p4,
        },
        "families": families,
        "uniform_attention_beta_zero": beta_zero,
        "mlp_only_controls": mlp_only,
        "type3_continuous_rotation": rotation,
        "periodic_orbit_certificates": certificates,
        "period_bifurcation_sweeps": bifurcations,
        "uniform_attention_certificates": beta_zero_certificates,
        "mlp_only_certificates": mlp_only_certificates,
        "verified_positive_lyapunov": verified_lyapunov,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"global": result["global"]}, indent=2))


if __name__ == "__main__":
    main()
