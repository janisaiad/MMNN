from dataclasses import replace

import numpy as np

from experiments.spectral_self_attention.mlp_equilibrium_taxonomy import (
    SerialBlock,
    map_jacobian,
    potential_mlp,
    random_block,
    solve_fixed_point,
    triwell_mlp,
    wrap,
)


def test_tied_quadratic_mlp_has_symmetric_jacobian() -> None:
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(2, 2))
    symmetric = (raw + raw.T) / 2.0
    hidden = rng.normal(size=(5, 2))
    hidden_bias = rng.normal(size=5)
    mlp = potential_mlp(
        rng.normal(size=2),
        symmetric,
        hidden,
        hidden_bias,
        rng.normal(size=5),
    )
    point = rng.normal(size=2)
    epsilon = 1e-6
    jacobian = np.column_stack(
        [
            (
                mlp((point + epsilon * np.eye(2)[column])[None, :])[0]
                - mlp((point - epsilon * np.eye(2)[column])[None, :])[0]
            )
            / (2.0 * epsilon)
            for column in range(2)
        ]
    )
    np.testing.assert_allclose(jacobian, jacobian.T, atol=2e-9)


def test_quadratic_potential_realizes_three_stable_wells_on_circle() -> None:
    block = SerialBlock(
        score=np.zeros((2, 2)),
        value=np.zeros((2, 2)),
        beta=1.0,
        step_size=0.06,
        mlp=triwell_mlp(1.2),
    )
    wells = 2.0 * np.pi * np.arange(3) / 3.0
    for well in wells:
        angle, residual = solve_fixed_point(block, np.array([well + 0.02]))
        assert residual < 1e-10
        assert abs(np.linalg.eigvals(map_jacobian(block, angle))[0]) < 1.0


def test_serial_potential_substeps_can_have_a_stable_period_two_cycle() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([1234, 6, 33]))
    block = replace(random_block(rng, "equal", "potential"), step_size=0.6)
    point = np.array([-0.937020126712106, 1.134772039425787])
    after_one = block.map_angles(point)
    after_two = block.map_angles(after_one)

    assert np.max(np.abs(wrap(after_one - point))) > 0.2
    np.testing.assert_allclose(wrap(after_two - point), 0.0, atol=2e-12)

    epsilon = 1e-6
    jacobian = np.column_stack(
        [
            wrap(
                block.map_angles(
                    block.map_angles(point + epsilon * np.eye(2)[column])
                )
                - block.map_angles(
                    block.map_angles(point - epsilon * np.eye(2)[column])
                )
            )
            / (2.0 * epsilon)
            for column in range(2)
        ]
    )
    assert np.max(np.abs(np.linalg.eigvals(jacobian))) < 1.0
