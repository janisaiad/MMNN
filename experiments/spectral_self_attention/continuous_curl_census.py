"""Measure the non-gradient part of the continuous angular field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import draw_models
from experiments.spectral_self_attention.small_step_continuation import (
    continuous_angular_field,
)


def field_jacobian(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    epsilon: float = 1e-6,
) -> np.ndarray:
    models_count, _, n_tokens = angles.shape
    jacobian = np.empty((models_count, n_tokens, n_tokens))
    for column in range(n_tokens):
        direction = np.zeros_like(angles)
        direction[:, 0, column] = epsilon
        plus = continuous_angular_field(angles + direction, models)[:, 0, :]
        minus = continuous_angular_field(angles - direction, models)[:, 0, :]
        jacobian[:, :, column] = (plus - minus) / (2.0 * epsilon)
    return jacobian


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def beta0_harmonic_drift(
    models: dict[str, np.ndarray], n_tokens: int
) -> np.ndarray:
    """Exact mean angular drift over the full token torus when beta=0."""
    attention = 0.5 * (models["value"][:, 1, 0] - models["value"][:, 0, 1])
    attention /= n_tokens
    linear = 0.5 * (models["linear"][:, 1, 0] - models["linear"][:, 0, 1])
    hidden_rotated = np.stack(
        (-models["hidden"][:, :, 1], models["hidden"][:, :, 0]), axis=-1
    )
    quadratic = np.einsum(
        "mdr,mrd,mr->m",
        models["output"],
        hidden_rotated,
        models["hidden_bias"],
        optimize=True,
    )
    return attention + linear + quadratic


def beta0_pairwise_fourier_force(
    theta_i: np.ndarray, theta_j: np.ndarray, value: np.ndarray
) -> np.ndarray:
    """Exact Fourier form of t(theta_i)^T V x(theta_j)."""
    a = value[..., 0, 0]
    b = value[..., 0, 1]
    c = value[..., 1, 0]
    d = value[..., 1, 1]
    return (
        0.5 * (a + d) * np.sin(theta_j - theta_i)
        + 0.5 * (c - b) * np.cos(theta_i - theta_j)
        + 0.5 * (d - a) * np.sin(theta_i + theta_j)
        + 0.5 * (b + c) * np.cos(theta_i + theta_j)
    )


def run(models_count: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    output: dict[str, object] = {}
    for family in (1, 2, 3, 4):
        family_rows: dict[str, object] = {}
        for n_tokens in (2, 3, 4):
            models = draw_models(rng, family, models_count)
            angles = rng.uniform(-np.pi, np.pi, size=(models_count, 1, n_tokens))
            original_beta = models["beta"].copy()
            cases = {}
            for name, beta in (
                ("sampled_softmax", original_beta),
                ("uniform_attention_beta0", np.zeros_like(original_beta)),
            ):
                models["beta"] = beta
                jacobian = field_jacobian(angles, models)
                antisymmetric = 0.5 * (jacobian - np.swapaxes(jacobian, -1, -2))
                ratio = np.linalg.norm(antisymmetric, axis=(1, 2)) / np.maximum(
                    np.linalg.norm(jacobian, axis=(1, 2)), 1e-12
                )
                harmonic = (
                    np.abs(beta0_harmonic_drift(models, n_tokens))
                    if name == "uniform_attention_beta0"
                    else np.full(models_count, np.nan)
                )
                cases[name] = {
                    "antisymmetric_fraction": summarize(ratio),
                    "near_gradient_count": int(np.sum(ratio < 1e-6)),
                    "models": models_count,
                    "harmonic_drift_absolute": (
                        summarize(harmonic) if np.all(np.isfinite(harmonic)) else None
                    ),
                    "near_zero_harmonic_count": (
                        int(np.sum(harmonic < 1e-10))
                        if np.all(np.isfinite(harmonic))
                        else None
                    ),
                    "positive_beta_only": (
                        {
                            "antisymmetric_fraction": summarize(
                                ratio[original_beta > 0.0]
                            ),
                            "near_gradient_count": int(
                                np.sum(ratio[original_beta > 0.0] < 1e-6)
                            ),
                            "models": int(np.sum(original_beta > 0.0)),
                        }
                        if name == "sampled_softmax"
                        else None
                    ),
                    "by_subtype": {
                        str(int(code)): {
                            "antisymmetric_fraction": summarize(
                                ratio[models["subtype_code"] == code]
                            ),
                            "near_gradient_count": int(
                                np.sum(ratio[models["subtype_code"] == code] < 1e-6)
                            ),
                            "models": int(np.sum(models["subtype_code"] == code)),
                            "harmonic_drift_absolute": (
                                summarize(harmonic[models["subtype_code"] == code])
                                if np.all(np.isfinite(harmonic))
                                else None
                            ),
                            "near_zero_harmonic_count": (
                                int(
                                    np.sum(
                                        harmonic[models["subtype_code"] == code] < 1e-10
                                    )
                                )
                                if np.all(np.isfinite(harmonic))
                                else None
                            ),
                        }
                        for code in np.unique(models["subtype_code"])
                    },
                }
            family_rows[str(n_tokens)] = cases
        output[str(family)] = family_rows
    return {"settings": {"models_per_case": models_count, "seed": seed}, "families": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=260814301)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.models, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
