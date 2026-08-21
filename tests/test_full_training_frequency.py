from __future__ import annotations

import math

import torch

from mmnn.full_training_frequency import (
    FullyTrainedPeriodicMLP,
    FullyTrainedPeriodicMMNN,
)
from mmnn.spectral_power import spectral_power_direction


def periodic_grid(size: int) -> torch.Tensor:
    return 2.0 * math.pi * torch.arange(size) / size


def test_fully_trained_models_are_centered_and_all_blocks_train() -> None:
    x = periodic_grid(16)
    mlp = FullyTrainedPeriodicMLP(x, width=12, affine_depth=5, seed=3)
    mmnn = FullyTrainedPeriodicMMNN(
        x, width=12, affine_depth=5, rank=4, seed=3
    )
    for model in (mlp, mmnn):
        torch.testing.assert_close(model(), torch.zeros_like(x))
        assert all(parameter.requires_grad for parameter in model.parameters())
        assert len(model.relative_feature_displacements()) == 4
        target = torch.cos(x) + 0.4 * torch.cos(5.0 * x)
        loss = 0.5 * torch.mean((model() - target).square())
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        assert all(
            bool(torch.all(torch.isfinite(gradient)))
            and float(torch.linalg.vector_norm(gradient)) > 0.0
            for gradient in gradients
        )


def test_mmnn_factorized_forward_matches_materialized_matrices() -> None:
    x = periodic_grid(11)
    model = FullyTrainedPeriodicMMNN(
        x, width=9, affine_depth=4, rank=3, seed=5
    )
    state = torch.relu(
        model.input @ model.input_weight.T / math.sqrt(2.0) + model.input_bias
    )
    for left, right, bias in zip(
        model.left_factors,
        model.right_factors,
        model.hidden_biases,
        strict=True,
    ):
        materialized = left @ right.T / math.sqrt(model.rank)
        state = torch.relu(state @ materialized.T / math.sqrt(model.width) + bias)
    expected = state @ model.readout / model.width + model.output_bias
    torch.testing.assert_close(model._uncentered_forward(), expected)


def test_metric_scaled_gradient_is_a_descent_direction() -> None:
    x = periodic_grid(16)
    target = torch.cos(x) + 0.3 * torch.cos(5.0 * x)
    model = FullyTrainedPeriodicMLP(x, width=10, affine_depth=3, seed=7)
    loss = 0.5 * torch.mean((model() - target) ** 2)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    directional_derivative = sum(
        model.metric_scale(name) * torch.sum(gradient.square())
        for (name, _), gradient in zip(
            model.named_parameters(), gradients, strict=True
        )
    )
    assert float(directional_derivative) > 0.0


def test_exact_muon_power_decreases_loss_at_small_step() -> None:
    x = periodic_grid(32)
    target = torch.cos(x) + 0.4 * torch.cos(7.0 * x)
    for power in (0.0, 1.0 / 3.0, 2.0 / 3.0):
        model = FullyTrainedPeriodicMMNN(
            x, width=10, affine_depth=4, rank=3, seed=11
        )
        named_parameters = tuple(model.named_parameters())
        names = tuple(name for name, _ in named_parameters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        initial_loss = 0.5 * torch.mean((model() - target) ** 2)
        gradients = torch.autograd.grad(initial_loss, parameters)
        directions = tuple(
            spectral_power_direction(gradient, power)
            if parameter.ndim == 2
            else gradient * model.metric_scale(name)
            for name, parameter, gradient in zip(
                names, parameters, gradients, strict=True
            )
        )
        pairing = sum(
            torch.sum(gradient * direction)
            for gradient, direction in zip(gradients, directions, strict=True)
        )
        assert float(pairing) > 0.0
        with torch.no_grad():
            for parameter, direction in zip(parameters, directions, strict=True):
                parameter.add_(direction, alpha=-1.0e-6)
        final_loss = 0.5 * torch.mean((model() - target) ** 2)
        assert float(final_loss) < float(initial_loss)


def test_dense_hidden_tangent_is_forward_backward_product() -> None:
    x = periodic_grid(9)
    model = FullyTrainedPeriodicMLP(x, width=7, affine_depth=3, seed=13)
    first = torch.relu(
        model.input @ model.input_weight.T / math.sqrt(2.0) + model.input_bias
    )
    preactivation = (
        first @ model.hidden_weights[0].T / math.sqrt(model.width)
        + model.hidden_biases[0]
    )
    hidden = torch.relu(preactivation)
    output = hidden @ model.readout / model.width + model.output_bias

    for sample in (0, 4, 8):
        backward = torch.autograd.grad(
            output[sample],
            preactivation,
            retain_graph=True,
        )[0][sample]
        weight_gradient = torch.autograd.grad(
            output[sample],
            model.hidden_weights[0],
            retain_graph=True,
        )[0]
        expected = backward[:, None] * first[sample][None, :]
        torch.testing.assert_close(
            math.sqrt(model.metric_scale("hidden_weights.0")) * weight_gradient,
            expected,
        )


def test_factorized_tangents_have_exact_channel_products() -> None:
    x = periodic_grid(9)
    model = FullyTrainedPeriodicMMNN(
        x, width=7, affine_depth=3, rank=3, seed=17
    )
    first = torch.relu(
        model.input @ model.input_weight.T / math.sqrt(2.0) + model.input_bias
    )
    channel = first @ model.right_factors[0]
    preactivation = (
        channel @ model.left_factors[0].T
        / math.sqrt(model.width * model.rank)
        + model.hidden_biases[0]
    )
    hidden = torch.relu(preactivation)
    output = hidden @ model.readout / model.width + model.output_bias

    for sample in (0, 4, 8):
        backward = torch.autograd.grad(
            output[sample],
            preactivation,
            retain_graph=True,
        )[0][sample]
        left_gradient, right_gradient = torch.autograd.grad(
            output[sample],
            (model.left_factors[0], model.right_factors[0]),
            retain_graph=True,
        )
        expected_left = (
            backward[:, None] * channel[sample][None, :] / math.sqrt(model.rank)
        )
        projected_backward = backward @ model.left_factors[0]
        expected_right = (
            first[sample][:, None]
            * projected_backward[None, :]
            / math.sqrt(model.rank)
        )
        scale = math.sqrt(model.metric_scale("left_factors.0"))
        torch.testing.assert_close(scale * left_gradient, expected_left)
        torch.testing.assert_close(scale * right_gradient, expected_right)
