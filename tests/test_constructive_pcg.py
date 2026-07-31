import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "transformers"
    / "constructive_weakform_richardson_transformer.py"
)
SPEC = importlib.util.spec_from_file_location("constructive_pcg_lab", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LAB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAB
SPEC.loader.exec_module(LAB)

TaskConfig = LAB.TaskConfig
run_constructive_loop = LAB.run_constructive_loop
sample_weak_batch = LAB.sample_weak_batch
set_seed = LAB.set_seed


def test_pcg_reaches_batched_spd_solution_in_at_most_k_steps() -> None:
    set_seed(7)
    cfg = TaskConfig(K=6, prompt_len=32, design="correlated", cond=100.0, dtype="float64")
    batch = sample_weak_batch(16, cfg, torch.device("cpu"))

    result = run_constructive_loop(
        batch,
        cfg,
        depth=cfg.K,
        precond="jacobi",
        solver="pcg",
    )

    assert torch.mean((result.beta_L - batch.beta_post) ** 2).item() < 1e-16


def test_pcg_improves_over_paired_richardson_at_fixed_depth() -> None:
    set_seed(11)
    cfg = TaskConfig(K=16, prompt_len=128, design="correlated", cond=1000.0, dtype="float64")
    batch = sample_weak_batch(32, cfg, torch.device("cpu"))

    richardson = run_constructive_loop(
        batch,
        cfg,
        depth=8,
        precond="jacobi",
        solver="richardson",
    )
    pcg = run_constructive_loop(
        batch,
        cfg,
        depth=8,
        precond="jacobi",
        solver="pcg",
    )

    richardson_mse = torch.mean((richardson.beta_L - batch.beta_post) ** 2).item()
    pcg_mse = torch.mean((pcg.beta_L - batch.beta_post) ** 2).item()
    assert pcg_mse < 1e-3 * richardson_mse
    assert pcg.theory_factor_mean < richardson.theory_factor_mean


def test_elliptic_pde_prompt_satisfies_weak_linear_identity() -> None:
    set_seed(17)
    cfg = TaskConfig(
        K=4,
        prompt_len=24,
        design="pde_elliptic",
        noise_var=1e-8,
        dtype="float64",
        pde_state_dim=12,
    )
    batch = sample_weak_batch(10, cfg, torch.device("cpu"))

    weak_residual = batch.b - torch.einsum("bmk,bk->bm", batch.G, batch.beta_true)

    assert batch.G.shape == (10, 24, 4)
    assert weak_residual.square().mean().sqrt().item() < 5e-4
    assert torch.linalg.eigvalsh(batch.H).min().item() > 0.0


def test_heavy_ball_with_zero_momentum_is_exactly_richardson() -> None:
    set_seed(13)
    cfg = TaskConfig(K=8, prompt_len=48, design="correlated", cond=100.0, dtype="float64")
    batch = sample_weak_batch(12, cfg, torch.device("cpu"))

    richardson = run_constructive_loop(
        batch,
        cfg,
        depth=5,
        precond="jacobi",
        solver="richardson",
    )
    heavy_ball = run_constructive_loop(
        batch,
        cfg,
        depth=5,
        precond="jacobi",
        solver="heavy_ball",
        hb_mode="manual",
        hb_alpha=1.0,
        hb_beta=0.0,
    )

    torch.testing.assert_close(heavy_ball.beta_L, richardson.beta_L, rtol=0.0, atol=0.0)
    torch.testing.assert_close(heavy_ball.alpha_L, richardson.alpha_L, rtol=0.0, atol=0.0)


def test_oracle_heavy_ball_accelerates_paired_richardson_stably() -> None:
    set_seed(17)
    cfg = TaskConfig(K=16, prompt_len=128, design="correlated", cond=1000.0, dtype="float64")
    batch = sample_weak_batch(32, cfg, torch.device("cpu"))

    richardson = run_constructive_loop(
        batch,
        cfg,
        depth=8,
        precond="jacobi",
        solver="richardson",
    )
    heavy_ball = run_constructive_loop(
        batch,
        cfg,
        depth=8,
        precond="jacobi",
        solver="heavy_ball",
        hb_mode="oracle",
    )

    richardson_mse = torch.mean((richardson.beta_L - batch.beta_post) ** 2).item()
    heavy_ball_mse = torch.mean((heavy_ball.beta_L - batch.beta_post) ** 2).item()
    assert heavy_ball_mse < 1e-2 * richardson_mse
    assert heavy_ball.theory_factor_max < 1.0


def test_chebyshev_accelerates_paired_richardson_without_adaptive_coefficients() -> None:
    set_seed(19)
    cfg = TaskConfig(K=16, prompt_len=128, design="correlated", cond=1000.0, dtype="float64")
    batch = sample_weak_batch(32, cfg, torch.device("cpu"))

    richardson = run_constructive_loop(
        batch,
        cfg,
        depth=8,
        precond="jacobi",
        solver="richardson",
    )
    chebyshev = run_constructive_loop(
        batch,
        cfg,
        depth=8,
        precond="jacobi",
        solver="chebyshev",
    )

    richardson_mse = torch.mean((richardson.beta_L - batch.beta_post) ** 2).item()
    chebyshev_mse = torch.mean((chebyshev.beta_L - batch.beta_post) ** 2).item()
    assert chebyshev_mse < 1e-2 * richardson_mse
    assert chebyshev.theory_factor_max < 1.0
