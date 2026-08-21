"""Small-step test for a stable type-3 rotating attractor on the circle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.mlp_equilibrium_taxonomy import (
    SerialBlock,
    triwell_mlp,
    wrap,
)


def order_parameter(angles: np.ndarray) -> np.ndarray:
    return np.mean(np.exp(1j * angles), axis=-1)


def simulate(
    block: SerialBlock,
    initial: np.ndarray,
    burn_time: float,
    measure_time: float,
) -> dict[str, float]:
    angles = initial.copy()
    burn_steps = int(np.ceil(burn_time / block.step_size))
    measure_steps = int(np.ceil(measure_time / block.step_size))
    for _ in range(burn_steps):
        angles = block.map_angles(angles)
    phase_advance = np.zeros(angles.shape[0])
    coherence_sum = np.zeros(angles.shape[0])
    previous_order = order_parameter(angles)
    motion_sum = np.zeros(angles.shape[0])
    for _ in range(measure_steps):
        previous_angles = angles
        angles = block.map_angles(angles)
        current_order = order_parameter(angles)
        phase_advance += np.angle(current_order * np.conj(previous_order))
        coherence_sum += np.abs(current_order)
        motion_sum += np.mean(np.abs(wrap(angles - previous_angles)), axis=-1)
        previous_order = current_order
    elapsed = measure_steps * block.step_size
    angular_speed = phase_advance / elapsed
    motion_per_time = motion_sum / elapsed
    coherence = coherence_sum / measure_steps
    return {
        "mean_angular_speed": float(np.mean(angular_speed)),
        "std_angular_speed": float(np.std(angular_speed)),
        "mean_absolute_motion_per_time": float(np.mean(motion_per_time)),
        "mean_coherence": float(np.mean(coherence)),
        "minimum_coherence": float(np.min(coherence)),
    }


def run(seed: int, basins: int, tokens: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-np.pi, np.pi, size=(basins, tokens))
    radial_gain = 1.2
    rotation = 0.8
    triwell_gain = 0.25
    rotor_value = np.array(
        [[radial_gain, -rotation], [rotation, radial_gain]], dtype=float
    )
    symmetric_value = radial_gain * np.eye(2)
    steps = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
    rotating = []
    symmetric_control = []
    tied_control = []
    for step in steps:
        common = {
            "beta": 1.0,
            "step_size": step,
            "mlp": triwell_mlp(triwell_gain),
        }
        rotating.append(
            {
                "step_size": step,
                **simulate(
                    SerialBlock(score=np.zeros((2, 2)), value=rotor_value, **common),
                    initial,
                    burn_time=180.0,
                    measure_time=80.0,
                ),
            }
        )
        symmetric_control.append(
            {
                "step_size": step,
                **simulate(
                    SerialBlock(
                        score=np.zeros((2, 2)), value=symmetric_value, **common
                    ),
                    initial,
                    burn_time=80.0,
                    measure_time=30.0,
                ),
            }
        )
        tied_control.append(
            {
                "step_size": step,
                **simulate(
                    SerialBlock(score=symmetric_value, value=symmetric_value, **common),
                    initial,
                    burn_time=80.0,
                    measure_time=30.0,
                ),
            }
        )
    continuum_speed = float(np.sqrt(rotation**2 - triwell_gain**2))
    return {
        "settings": {
            "seed": seed,
            "basins": basins,
            "tokens": tokens,
            "radial_gain": radial_gain,
            "rotation": rotation,
            "triwell_gain": triwell_gain,
        },
        "continuum_prediction": {
            "consensus_equation": "dtheta/dt = rotation - triwell_gain*sin(3*theta)",
            "mean_angular_speed": continuum_speed,
        },
        "type3_antisymmetric_value": rotating,
        "type3_symmetric_value_control": symmetric_control,
        "type1_tied_control": tied_control,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=260813031)
    parser.add_argument("--basins", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.basins, args.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
