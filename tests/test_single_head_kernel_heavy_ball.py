import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "transformers"
    / "richardson_transformer_weak_krr_lab.py"
)
SPEC = importlib.util.spec_from_file_location("single_head_kernel_lab", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LAB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAB
SPEC.loader.exec_module(LAB)

NoMLPKernelAttentionLoop = LAB.NoMLPKernelAttentionLoop


def test_single_head_attention_is_exact_row_normalized_rbf_kernel() -> None:
    model = NoMLPKernelAttentionLoop(
        depth=1,
        lam=0.02,
        iteration="richardson",
        init_lengthscale=0.4,
        learn_kernel=False,
        step_init=0.8,
    )
    x = torch.tensor([[[-1.0], [-0.25], [0.5], [1.0]]])

    attention, degree, lengthscale = model.kernel_attention(x)
    pairwise_distance = torch.cdist(x / lengthscale, x / lengthscale).pow(2)
    kernel = torch.exp(-0.5 * pairwise_distance)

    torch.testing.assert_close(attention, kernel / kernel.sum(dim=-1, keepdim=True))
    torch.testing.assert_close(degree, kernel.sum(dim=-1))


def test_first_loop_is_one_attention_correction_without_an_mlp() -> None:
    model = NoMLPKernelAttentionLoop(
        depth=1,
        lam=0.02,
        iteration="heavy_ball",
        init_lengthscale=0.3,
        learn_kernel=False,
        step_init=1.2,
        momentum_init=0.2,
    )
    x_context = torch.tensor([[[-0.8], [-0.1], [0.4], [0.9]]])
    y_context = torch.tensor([[0.5, -0.2, 1.0, 0.3]])
    x_query = torch.tensor([[0.2]])

    prediction, state, info = model(x_context, y_context, x_query)
    expected_state = info["step"] * torch.einsum("bij,bj->bi", info["attention"], y_context)
    lengthscale = info["lengthscale"]
    query_logits = -0.5 * (
        x_query[:, None, :] / lengthscale - x_context / lengthscale
    ).pow(2).sum(dim=-1)
    expected_prediction = info["step"] * torch.einsum(
        "bi,bi->b", torch.softmax(query_logits, dim=-1), y_context
    )

    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(prediction, expected_prediction)
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
    assert not any(isinstance(module, torch.nn.Sequential) for module in model.modules())
