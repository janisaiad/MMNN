"""Independent geometric and divergence checks for a saved continuous trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_curl_census import field_jacobian
from experiments.spectral_self_attention.continuous_ode_robustness import select_record
from experiments.spectral_self_attention.small_step_continuation import stack_models


def run(
    payload: dict[str, object],
    trace: dict[str, object],
    spectrum: dict[str, object] | None,
    chunk_size: int,
) -> dict[str, object]:
    identity = trace["identity"]
    record = select_record(
        payload,
        int(identity["n_tokens"]),
        int(identity["source_model_index"]),
    )
    history = np.asarray(trace["angles"], dtype=float)
    divergences = []
    for first in range(0, len(history), chunk_size):
        angles = history[first : first + chunk_size]
        repeated = [record] * len(angles)
        models = stack_models(repeated)
        jacobian = field_jacobian(angles[:, None, :], models)
        jacobian *= models["step_size"][:, None, None]
        divergences.append(np.trace(jacobian, axis1=1, axis2=2))
    divergence = np.concatenate(divergences)
    relative = np.asarray(trace["relative_angles"], dtype=float)
    covariance_eigenvalues = np.linalg.eigvalsh(np.cov(relative, rowvar=False))[::-1]
    gram = np.asarray(trace["gram"], dtype=float)
    attention = np.asarray(trace["attention_weights"], dtype=float)
    result = {
        "identity": identity,
        "samples": len(history),
        "mean_divergence": float(np.mean(divergence)),
        "divergence_quantiles": {
            str(quantile): float(np.quantile(divergence, quantile))
            for quantile in (0.01, 0.05, 0.5, 0.95, 0.99)
        },
        "fraction_locally_expanding_volume": float(np.mean(divergence > 0.0)),
        "relative_angle_covariance_eigenvalues": covariance_eigenvalues.tolist(),
        "gram_standard_deviation_max": float(np.max(np.std(gram, axis=0))),
        "attention_standard_deviation_max": float(
            np.max(np.std(attention, axis=0))
        ),
    }
    if spectrum is not None:
        spectrum_sum = float(np.sum(spectrum["spectrum"]))
        result["lyapunov_spectrum_sum"] = spectrum_sum
        result["divergence_minus_spectrum_sum"] = float(
            np.mean(divergence) - spectrum_sum
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("continuation", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--spectrum", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.continuation.read_text()),
        json.loads(args.trace.read_text()),
        json.loads(args.spectrum.read_text()) if args.spectrum else None,
        args.chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
