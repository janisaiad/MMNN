"""Small-system taxonomy for serial spherical Attention->quadratic-MLP blocks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def angles_to_tokens(angles: np.ndarray) -> np.ndarray:
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def tokens_to_angles(tokens: np.ndarray) -> np.ndarray:
    return np.arctan2(tokens[..., 1], tokens[..., 0])


def wrap(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class QuadraticMLP:
    """u(x)=b+Bx+C((Wx+c)^2); tied C gives a potential MLP."""

    bias: np.ndarray
    linear: np.ndarray
    hidden: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    kind: str

    def __call__(self, tokens: np.ndarray) -> np.ndarray:
        hidden_values = np.einsum("rd,...nd->...nr", self.hidden, tokens)
        hidden_values += self.hidden_bias
        return (
            self.bias
            + np.einsum("de,...ne->...nd", self.linear, tokens)
            + np.einsum("dr,...nr->...nd", self.output, hidden_values**2)
        )


@dataclass(frozen=True)
class SerialBlock:
    score: np.ndarray
    value: np.ndarray
    beta: float
    step_size: float
    mlp: QuadraticMLP | None

    def attention(self, tokens: np.ndarray) -> np.ndarray:
        scores = np.einsum("...id,de,...je->...ij", tokens, self.score, tokens)
        logits = self.beta * scores
        logits -= np.max(logits, axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        values = np.einsum("de,...je->...jd", self.value, tokens)
        return np.einsum("...ij,...jd->...id", weights, values)

    def map_tokens(self, tokens: np.ndarray) -> np.ndarray:
        after_attention = normalize(tokens + self.step_size * self.attention(tokens))
        if self.mlp is None:
            return after_attention
        return normalize(after_attention + self.step_size * self.mlp(after_attention))

    def map_angles(self, angles: np.ndarray) -> np.ndarray:
        return tokens_to_angles(self.map_tokens(angles_to_tokens(angles)))

    def fixed_residual(self, angles: np.ndarray) -> np.ndarray:
        return wrap(self.map_angles(angles) - angles)


def potential_mlp(
    bias: np.ndarray,
    symmetric_linear: np.ndarray,
    hidden: np.ndarray,
    hidden_bias: np.ndarray,
    coefficients: np.ndarray,
) -> QuadraticMLP:
    output = hidden.T * coefficients[None, :]
    return QuadraticMLP(
        bias=np.asarray(bias, dtype=float),
        linear=np.asarray(symmetric_linear, dtype=float),
        hidden=np.asarray(hidden, dtype=float),
        hidden_bias=np.asarray(hidden_bias, dtype=float),
        output=output,
        kind="potential",
    )


def triwell_mlp(gain: float = 1.0) -> QuadraticMLP:
    # grad[(gain/3) Re((x+iy)^3)] = gain*(x^2-y^2, -2xy).
    hidden = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    coefficients = gain * np.array([2.0, 0.0, -0.5, -0.5])
    return potential_mlp(
        np.zeros(2), np.zeros((2, 2)), hidden, np.zeros(4), coefficients
    )


def rotor_mlp(rotation: float, triwell_gain: float) -> QuadraticMLP:
    base = triwell_mlp(triwell_gain)
    return QuadraticMLP(
        bias=base.bias,
        linear=np.array([[0.0, -rotation], [rotation, 0.0]]),
        hidden=base.hidden,
        hidden_bias=base.hidden_bias,
        output=base.output,
        kind="general",
    )


def map_jacobian(block: SerialBlock, angles: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    n = angles.size
    jacobian = np.empty((n, n))
    for column in range(n):
        direction = np.zeros(n)
        direction[column] = epsilon
        plus = block.map_angles(angles + direction)
        minus = block.map_angles(angles - direction)
        jacobian[:, column] = wrap(plus - minus) / (2.0 * epsilon)
    return jacobian


def solve_fixed_point(block: SerialBlock, start: np.ndarray) -> tuple[np.ndarray, float]:
    result = least_squares(
        block.fixed_residual,
        start,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=1600,
    )
    angles = np.mod(result.x, 2.0 * np.pi)
    residual = float(np.max(np.abs(block.fixed_residual(angles))))
    return angles, residual


def equilibrium_record(block: SerialBlock, angles: np.ndarray) -> dict[str, object]:
    eigenvalues = np.linalg.eigvals(map_jacobian(block, angles))
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if angles.size == 1:
        geometry = "single"
    else:
        correlations = np.cos(angles[:, None] - angles[None, :])
        off_diagonal = correlations[~np.eye(angles.size, dtype=bool)]
        if np.min(off_diagonal) > 1.0 - 1e-5:
            geometry = "consensus"
        elif angles.size == 2 and off_diagonal[0] < -1.0 + 1e-5:
            geometry = "bipolar"
        else:
            geometry = "irregular"
    return {
        "angles": angles.tolist(),
        "spectral_radius": spectral_radius,
        "stable": spectral_radius < 1.0 - 2e-5,
        "complex_linearization": bool(np.max(np.abs(eigenvalues.imag)) > 2e-4),
        "eigenvalues": [[float(value.real), float(value.imag)] for value in eigenvalues],
        "geometry": geometry,
    }


def canonical_labeled_key(angles: np.ndarray, decimals: int = 5) -> tuple[float, ...]:
    values = np.round(np.mod(angles, 2.0 * np.pi), decimals)
    values[np.isclose(values, 2.0 * np.pi, atol=10.0 ** (-decimals))] = 0.0
    return tuple(values.tolist())


def canonical_unlabeled_key(angles: np.ndarray, decimals: int = 5) -> tuple[float, ...]:
    values = np.round(np.mod(angles, 2.0 * np.pi), decimals)
    values[np.isclose(values, 2.0 * np.pi, atol=10.0 ** (-decimals))] = 0.0
    return tuple(np.sort(values).tolist())


def discover_equilibria(
    block: SerialBlock,
    n_tokens: int,
    rng: np.random.Generator,
    random_starts: int = 180,
) -> list[dict[str, object]]:
    grid = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    if n_tokens == 1:
        starts = [
            np.array([value])
            for value in np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
        ]
    elif n_tokens == 2:
        starts = [np.asarray(values) for values in product(grid, repeat=2)]
    else:
        starts = []
    starts.extend(rng.uniform(0.0, 2.0 * np.pi, n_tokens) for _ in range(random_starts))
    found: dict[tuple[float, ...], dict[str, object]] = {}
    for start in starts:
        angles, residual = solve_fixed_point(block, start)
        if residual > 2e-8:
            continue
        key = canonical_unlabeled_key(angles)
        if key not in found:
            record = equilibrium_record(block, angles)
            record["residual"] = residual
            found[key] = record
    return list(found.values())


def triwell_scaling() -> list[dict[str, object]]:
    records = []
    wells = 2.0 * np.pi * np.arange(3) / 3.0
    for n_tokens in range(1, 6):
        block = SerialBlock(
            score=0.03 * np.eye(2),
            value=0.03 * np.eye(2),
            beta=2.0,
            step_size=0.06,
            mlp=triwell_mlp(1.2),
        )
        equilibria: dict[tuple[float, ...], dict[str, object]] = {}
        for assignment in product(wells, repeat=n_tokens):
            angles, residual = solve_fixed_point(block, np.asarray(assignment) + 0.01)
            if residual > 2e-8:
                continue
            key = canonical_labeled_key(angles)
            equilibria[key] = equilibrium_record(block, angles)
        stable = sum(bool(record["stable"]) for record in equilibria.values())
        unlabeled = {canonical_unlabeled_key(np.asarray(key)) for key in equilibria}
        records.append(
            {
                "tokens": n_tokens,
                "predicted_labeled": 3**n_tokens,
                "found_labeled": len(equilibria),
                "stable_labeled": stable,
                "found_up_to_token_permutation": len(unlabeled),
                "predicted_up_to_token_permutation": (n_tokens + 2) * (n_tokens + 1) // 2,
            }
        )
    return records


def coupling_sweep() -> list[dict[str, object]]:
    rng = np.random.default_rng(260426086)
    output = []
    single_token_roots = np.pi * np.arange(6) / 3.0
    starts = [
        np.asarray(values) + 0.005
        for values in product(single_token_roots, repeat=3)
    ]
    starts.extend(rng.uniform(0.0, 2.0 * np.pi, 3) for _ in range(250))
    for coupling in (0.0, 0.03, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0):
        block = SerialBlock(
            score=coupling * np.eye(2),
            value=coupling * np.eye(2),
            beta=1.2,
            step_size=0.06,
            mlp=triwell_mlp(0.2),
        )
        found: dict[tuple[float, ...], dict[str, object]] = {}
        for start in starts:
            angles, residual = solve_fixed_point(block, start)
            if residual > 2e-8:
                continue
            key = canonical_unlabeled_key(angles)
            if key not in found:
                found[key] = equilibrium_record(block, angles)
        output.append(
            {
                "coupling": coupling,
                "equilibria": len(found),
                "stable_equilibria": sum(bool(record["stable"]) for record in found.values()),
                "stable_spirals": sum(
                    bool(record["stable"] and record["complex_linearization"])
                    for record in found.values()
                ),
            }
        )
    return output


def random_mlp(rng: np.random.Generator, kind: str, scale: float = 0.7) -> QuadraticMLP:
    width = 4
    hidden = rng.normal(size=(width, 2))
    hidden_bias = rng.normal(scale=0.35, size=width)
    bias = rng.normal(scale=0.25, size=2)
    if kind == "potential":
        raw = rng.normal(size=(2, 2))
        linear = scale * (raw + raw.T) / 2.0
        coefficients = scale * rng.normal(size=width) / np.sqrt(width)
        return potential_mlp(bias, linear, hidden, hidden_bias, coefficients)
    linear = scale * rng.normal(size=(2, 2))
    output = scale * rng.normal(size=(2, width)) / np.sqrt(width)
    return QuadraticMLP(bias, linear, hidden, hidden_bias, output, "general")


def random_block(rng: np.random.Generator, relation: str, mlp_kind: str) -> SerialBlock:
    if relation == "equal":
        raw = rng.normal(scale=0.55, size=(2, 2))
        value = (raw + raw.T) / 2.0
        score = value.copy()
    else:
        score = rng.normal(scale=0.55, size=(2, 2))
        value = rng.normal(scale=0.55, size=(2, 2))
    return SerialBlock(
        score=score,
        value=value,
        beta=1.4,
        step_size=0.11,
        mlp=random_mlp(rng, mlp_kind),
    )


def ensemble_survey(models: int) -> list[dict[str, object]]:
    rows = []
    for relation, mlp_kind in product(("equal", "unequal"), ("potential", "general")):
        stable_counts = []
        total_counts = []
        focus_counts = []
        irregular_counts = []
        max_example: dict[str, object] | None = None
        focus_example: dict[str, object] | None = None
        for model in range(models):
            rng = np.random.default_rng(
                np.random.SeedSequence([260426087, relation == "unequal", mlp_kind == "general", model])
            )
            block = random_block(rng, relation, mlp_kind)
            equilibria = discover_equilibria(block, 2, rng)
            stable = sum(bool(record["stable"]) for record in equilibria)
            focuses = sum(
                bool(record["stable"] and record["complex_linearization"])
                for record in equilibria
            )
            stable_counts.append(stable)
            total_counts.append(len(equilibria))
            focus_counts.append(focuses)
            irregular_counts.append(
                sum(bool(record["stable"] and record["geometry"] == "irregular") for record in equilibria)
            )
            if focus_example is None and focuses:
                focus_record = next(
                    record
                    for record in equilibria
                    if record["stable"] and record["complex_linearization"]
                )
                focus_example = {
                    "model": model,
                    "equilibrium": focus_record,
                    "score": block.score.tolist(),
                    "value": block.value.tolist(),
                }
            if max_example is None or stable > int(max_example["stable_equilibria"]):
                max_example = {
                    "model": model,
                    "stable_equilibria": stable,
                    "equilibria": len(equilibria),
                    "stable_spirals": focuses,
                    "score": block.score.tolist(),
                    "value": block.value.tolist(),
                    "mlp": {
                        "bias": block.mlp.bias.tolist(),
                        "linear": block.mlp.linear.tolist(),
                        "hidden": block.mlp.hidden.tolist(),
                        "hidden_bias": block.mlp.hidden_bias.tolist(),
                        "output": block.mlp.output.tolist(),
                    },
                }
        rows.append(
            {
                "relation": relation,
                "mlp": mlp_kind,
                "models": models,
                "mean_equilibria": float(np.mean(total_counts)),
                "max_equilibria": int(np.max(total_counts)),
                "mean_stable_equilibria": float(np.mean(stable_counts)),
                "max_stable_equilibria": int(np.max(stable_counts)),
                "models_with_multiple_stable": int(np.sum(np.asarray(stable_counts) >= 2)),
                "models_with_stable_spiral": int(np.sum(np.asarray(focus_counts) >= 1)),
                "models_with_stable_irregular_equilibrium": int(
                    np.sum(np.asarray(irregular_counts) >= 1)
                ),
                "max_example": max_example,
                "focus_example": focus_example,
            }
        )
    return rows


def single_token_survey(models: int = 200) -> list[dict[str, object]]:
    rows = []
    for mlp_kind in ("potential", "general"):
        counts = []
        stable_counts = []
        for model in range(models):
            rng = np.random.default_rng(
                np.random.SeedSequence([260426089, mlp_kind == "general", model])
            )
            block = SerialBlock(
                score=np.zeros((2, 2)),
                value=np.zeros((2, 2)),
                beta=1.0,
                step_size=0.08,
                mlp=random_mlp(rng, mlp_kind),
            )
            equilibria = discover_equilibria(block, 1, rng, random_starts=40)
            counts.append(len(equilibria))
            stable_counts.append(sum(bool(record["stable"]) for record in equilibria))
        rows.append(
            {
                "mlp": mlp_kind,
                "models": models,
                "zero_equilibrium_models": int(np.sum(np.asarray(counts) == 0)),
                "max_equilibria": int(np.max(counts)),
                "max_stable_equilibria": int(np.max(stable_counts)),
                "mean_equilibria": float(np.mean(counts)),
                "mean_stable_equilibria": float(np.mean(stable_counts)),
            }
        )
    return rows


def rotor_experiments() -> list[dict[str, object]]:
    configurations = [
        (
            "equal_general_mlp",
            SerialBlock(
                score=0.5 * np.eye(2),
                value=0.5 * np.eye(2),
                beta=1.5,
                step_size=0.05,
                mlp=rotor_mlp(rotation=1.4, triwell_gain=0.7),
            ),
        ),
        (
            "unequal_attention_potential_mlp",
            SerialBlock(
                score=np.zeros((2, 2)),
                value=np.array([[0.25, -1.3], [1.3, 0.25]]),
                beta=1.5,
                step_size=0.05,
                mlp=triwell_mlp(0.3),
            ),
        ),
    ]
    rng = np.random.default_rng(260426088)
    rows = []
    for name, block in configurations:
        tokens = angles_to_tokens(rng.uniform(-np.pi, np.pi, size=4))
        unwrapped_mean_angle = []
        concentration = []
        pairwise_change = []
        previous_gram = tokens @ tokens.T
        for step in range(8000):
            tokens = block.map_tokens(tokens)
            if step >= 5000 and step % 5 == 0:
                mean = np.mean(tokens, axis=0)
                unwrapped_mean_angle.append(float(np.arctan2(mean[1], mean[0])))
                concentration.append(float(np.linalg.norm(mean)))
                gram = tokens @ tokens.T
                pairwise_change.append(float(np.linalg.norm(gram - previous_gram)))
                previous_gram = gram
        winding = np.unwrap(np.asarray(unwrapped_mean_angle))
        elapsed = (len(winding) - 1) * 5 * block.step_size
        angular_velocity = float((winding[-1] - winding[0]) / elapsed)
        equilibria = discover_equilibria(block, 2, rng, random_starts=300)
        rows.append(
            {
                "name": name,
                "late_concentration": float(np.mean(concentration)),
                "angular_velocity": angular_velocity,
                "turns_observed": float(abs(winding[-1] - winding[0]) / (2.0 * np.pi)),
                "late_pairwise_shape_change": float(np.mean(pairwise_change[-100:])),
                "fixed_equilibria_found_for_two_tokens": len(equilibria),
                "stable_fixed_equilibria_found": sum(bool(record["stable"]) for record in equilibria),
            }
        )
    return rows


def serial_potential_effects() -> dict[str, object]:
    """Examples created by finite Attention->MLP composition despite tied QK/V."""
    focus_rng = np.random.default_rng(np.random.SeedSequence([999, 80, 37]))
    focus_block = replace(
        random_block(focus_rng, "equal", "potential"), step_size=0.8
    )
    focus_equilibria = discover_equilibria(
        focus_block, 2, focus_rng, random_starts=240
    )
    focus = next(
        record
        for record in focus_equilibria
        if record["stable"] and record["complex_linearization"]
    )

    period_rng = np.random.default_rng(np.random.SeedSequence([1234, 6, 33]))
    period_block = replace(
        random_block(period_rng, "equal", "potential"), step_size=0.6
    )
    period_fixed_equilibria = discover_equilibria(
        period_block, 2, period_rng, random_starts=500
    )
    angles = period_rng.uniform(-np.pi, np.pi, size=(128, 2))
    for _ in range(2500):
        angles = period_block.map_angles(angles)
    one_step = period_block.map_angles(angles)
    two_step = period_block.map_angles(one_step)
    one_error = np.max(np.abs(wrap(one_step - angles)), axis=1)
    two_error = np.max(np.abs(wrap(two_step - angles)), axis=1)
    period_mask = (two_error < 1e-8) & (one_error > 1e-3)
    cycle_index = int(np.flatnonzero(period_mask)[0])
    cycle_a = angles[cycle_index]
    cycle_b = one_step[cycle_index]

    def two_step_map(state: np.ndarray) -> np.ndarray:
        return period_block.map_angles(period_block.map_angles(state))

    epsilon = 1e-6
    two_step_jacobian = np.column_stack(
        [
            wrap(
                two_step_map(cycle_a + epsilon * np.eye(2)[column])
                - two_step_map(cycle_a - epsilon * np.eye(2)[column])
            )
            / (2.0 * epsilon)
            for column in range(2)
        ]
    )
    cycle_eigenvalues = np.linalg.eigvals(two_step_jacobian)
    small_step_results = []
    for step_size in (0.6, 0.3, 0.15, 0.075, 0.03):
        block = replace(period_block, step_size=step_size)
        state = cycle_a[None, :].copy()
        for _ in range(round(300.0 / step_size)):
            state = block.map_angles(state)
        first = block.map_angles(state)
        second = block.map_angles(first)
        small_step_results.append(
            {
                "step_size": step_size,
                "one_step_motion": float(np.max(np.abs(wrap(first - state)))),
                "two_step_return_error": float(np.max(np.abs(wrap(second - state)))),
            }
        )

    def parallel_map(state: np.ndarray) -> np.ndarray:
        tokens = angles_to_tokens(state)
        output = period_block.attention(tokens) + period_block.mlp(tokens)
        return tokens_to_angles(normalize(tokens + period_block.step_size * output))

    def mlp_only_map(state: np.ndarray) -> np.ndarray:
        tokens = angles_to_tokens(state)
        return tokens_to_angles(
            normalize(tokens + period_block.step_size * period_block.mlp(tokens))
        )

    def reverse_serial_map(state: np.ndarray) -> np.ndarray:
        tokens = angles_to_tokens(state)
        after_mlp = normalize(
            tokens + period_block.step_size * period_block.mlp(tokens)
        )
        return tokens_to_angles(
            normalize(
                after_mlp + period_block.step_size * period_block.attention(after_mlp)
            )
        )

    architecture_comparison = []
    for name, block_map in (
        ("attention_then_mlp", period_block.map_angles),
        ("parallel_sum", parallel_map),
        ("mlp_only", mlp_only_map),
        ("mlp_then_attention", reverse_serial_map),
    ):
        state = cycle_a[None, :].copy()
        for _ in range(3000):
            state = block_map(state)
        first = block_map(state)
        second = block_map(first)
        architecture_comparison.append(
            {
                "architecture": name,
                "one_step_motion": float(np.max(np.abs(wrap(first - state)))),
                "two_step_return_error": float(np.max(np.abs(wrap(second - state)))),
            }
        )
    return {
        "stable_focus": focus,
        "stable_period_two": {
            "cycle_point_a": cycle_a.tolist(),
            "cycle_point_b": cycle_b.tolist(),
            "basin_hits": int(np.sum(period_mask)),
            "basin_trials": int(period_mask.size),
            "coexisting_fixed_equilibria": len(period_fixed_equilibria),
            "coexisting_stable_fixed_equilibria": sum(
                bool(record["stable"]) for record in period_fixed_equilibria
            ),
            "one_step_motion": float(one_error[cycle_index]),
            "two_step_return_error": float(two_error[cycle_index]),
            "two_step_eigenvalues": [
                [float(value.real), float(value.imag)] for value in cycle_eigenvalues
            ],
        },
        "step_size_sweep": small_step_results,
        "architecture_comparison": architecture_comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spectral_self_attention/mlp_equilibrium_taxonomy.json"),
    )
    args = parser.parse_args()
    result = {
        "model": "serial normalized Attention then one-hidden-layer quadratic MLP on S1",
        "triwell_scaling": triwell_scaling(),
        "coupling_sweep": coupling_sweep(),
        "single_token_survey": single_token_survey(),
        "ensemble_survey": ensemble_survey(args.models),
        "rotors": rotor_experiments(),
        "serial_potential_effects": serial_potential_effects(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
