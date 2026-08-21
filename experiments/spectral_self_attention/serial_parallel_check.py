"""Finite-depth comparison of serial Attention->MLP and a parallel drift."""

from __future__ import annotations

import json

import numpy as np

from experiments.spectral_self_attention.slow_ou_tokens import normalize, token_field


def run_check() -> dict[str, object]:
    rng = np.random.default_rng(19)
    initial = normalize(rng.normal(size=(1, 8, 3)))
    attention_matrix = np.array(
        [[1.2, 0.25, 0.0], [0.25, 0.6, 0.1], [0.0, 0.1, -0.4]]
    )[None, None]
    mlp_matrix = np.array(
        [[0.1, 1.1, -0.3], [-0.7, 0.2, 0.8], [0.4, -0.6, 0.1]]
    )
    mlp_bias = np.array([0.2, -0.1, 0.15])

    def mlp_field(tokens: np.ndarray) -> np.ndarray:
        output = np.tanh(np.einsum("de,rne->rnd", mlp_matrix, tokens) + mlp_bias)
        return output - np.sum(tokens * output, axis=-1, keepdims=True) * tokens

    def integrate(step_size: float, serial: bool) -> np.ndarray:
        tokens = initial.copy()
        steps = round(4.0 / step_size)
        step_size = 4.0 / steps
        for _ in range(steps):
            if serial:
                after_attention = normalize(
                    tokens + step_size * token_field(tokens, attention_matrix)
                )
                tokens = normalize(
                    after_attention + step_size * mlp_field(after_attention)
                )
            else:
                tokens = normalize(
                    tokens
                    + step_size
                    * (
                        token_field(tokens, attention_matrix)
                        + mlp_field(tokens)
                    )
                )
        return tokens

    step_sizes = np.asarray([0.2, 0.1, 0.05, 0.025, 0.0125])
    differences = []
    for step_size in step_sizes:
        serial = integrate(float(step_size), serial=True)
        parallel = integrate(float(step_size), serial=False)
        differences.append(float(np.sqrt(np.mean((serial - parallel) ** 2))))
    slope = float(np.polyfit(np.log(step_sizes), np.log(differences), 1)[0])
    return {
        "step_sizes": step_sizes.tolist(),
        "serial_parallel_rms_difference": differences,
        "log_log_slope": slope,
    }


if __name__ == "__main__":
    print(json.dumps(run_check(), indent=2))
