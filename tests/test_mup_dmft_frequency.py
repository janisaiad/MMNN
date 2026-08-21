from __future__ import annotations

import math

import torch

from mmnn.mup_right_factor import CenteredRightFactorMuP


def periodic_grid(size: int, device: torch.device) -> torch.Tensor:
    return 2.0 * math.pi * torch.arange(size, device=device) / size


def cosine_coefficient(values: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    return 2.0 * torch.mean(values * torch.cos(k * x))


def tangent_fourier_matrix(
    model: CenteredRightFactorMuP,
    x: torch.Tensor,
    frequencies: tuple[int, ...],
) -> torch.Tensor:
    prediction = model()
    gradients = []
    for index, frequency in enumerate(frequencies):
        coefficient = cosine_coefficient(prediction, x, frequency)
        gradient = torch.autograd.grad(
            coefficient,
            model.V,
            retain_graph=index + 1 < len(frequencies),
        )[0]
        gradients.append(gradient.flatten() / math.sqrt(2.0))
    stacked = torch.stack(gradients)
    return model.metric_scale * (stacked @ stacked.T)


def make_small_model() -> tuple[torch.Tensor, CenteredRightFactorMuP]:
    device = torch.device("cpu")
    x = periodic_grid(16, device)
    model = CenteredRightFactorMuP(
        x,
        width=12,
        rank=4,
        gamma=0.7,
        seed=3,
        bias_scale_1=0.35,
        bias_scale_2=0.15,
    )
    return x, model


def test_centering_and_only_right_factor_is_trainable() -> None:
    _, model = make_small_model()
    assert torch.equal(model(), torch.zeros_like(model()))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == ["V"]


def test_exact_kernel_factorization() -> None:
    _, model = make_small_model()
    prediction = model()
    gradients = []
    for index in range(prediction.numel()):
        gradients.append(
            torch.autograd.grad(
                prediction[index],
                model.V,
                retain_graph=index + 1 < prediction.numel(),
            )[0].flatten()
        )
    jacobian = torch.stack(gradients)
    direct = model.metric_scale * (jacobian @ jacobian.T)

    phi = model.h @ model.h.T / model.width
    gates = (model.s0 > 0).to(model.h.dtype)
    g = (gates * model.readout[None, :]) @ model.U / math.sqrt(model.width)
    backpropagated = g @ g.T / model.rank
    factorized = phi * backpropagated
    torch.testing.assert_close(direct, factorized, rtol=2.0e-5, atol=2.0e-6)


def test_fourier_tangent_matrix_is_psd() -> None:
    x, model = make_small_model()
    matrix = tangent_fourier_matrix(model, x, (1, 2, 3, 4))
    torch.testing.assert_close(matrix, matrix.T)
    assert float(torch.linalg.eigvalsh(matrix).min()) >= -1.0e-10
