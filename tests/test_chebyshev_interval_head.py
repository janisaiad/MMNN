import importlib.util
import sys
from pathlib import Path

import torch

TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "experiments" / "transformers"
for module_name in [
    "constructive_weakform_richardson_transformer",
    "first_principles_decoder_cells",
    "low_rank_subspace_preconditioner",
    "first_principles_inverse_decoder",
    "structured_one_head_heavyball",
    "train_chebyshev_interval_head",
]:
    module_path = TRANSFORMER_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

LAB = sys.modules["train_chebyshev_interval_head"]


def test_effective_spectrum_features_are_finite_and_have_fixed_width() -> None:
    torch.manual_seed(401)
    factor = torch.randn(5, 8, 8, dtype=torch.float64)
    operator = factor.transpose(-1, -2) @ factor
    operator = operator + 0.2 * torch.eye(8, dtype=torch.float64)

    features = LAB.effective_spectrum_features(operator, prompt_length=32)

    assert features.shape == (5, 7)
    assert torch.isfinite(features).all()


def test_symmetric_effective_operator_has_same_spectrum_as_bh() -> None:
    torch.manual_seed(409)
    factor_h = torch.randn(4, 7, 7, dtype=torch.float64)
    factor_b = torch.randn(4, 7, 7, dtype=torch.float64)
    normal = factor_h.transpose(-1, -2) @ factor_h
    normal = normal + torch.eye(7, dtype=torch.float64)
    preconditioner = factor_b.transpose(-1, -2) @ factor_b
    preconditioner = preconditioner + torch.eye(7, dtype=torch.float64)

    symmetric = LAB.symmetric_effective_operator(preconditioner, normal)
    expected = torch.linalg.eigvals(preconditioner @ normal).real.sort().values

    torch.testing.assert_close(torch.linalg.eigvalsh(symmetric), expected)


def test_relative_h_loss_is_differentiable() -> None:
    torch.manual_seed(419)
    normal = torch.eye(5, dtype=torch.float64).expand(3, -1, -1)
    prediction = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    target = torch.randn(3, 5, dtype=torch.float64)

    loss = LAB.relative_h_loss(prediction, target, normal)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
