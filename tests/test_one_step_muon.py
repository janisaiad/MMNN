import torch

from experiments.spectral_self_attention.one_step_muon import (
    newton_schulz_scalar_magnitudes,
    run_experiment,
)


def test_newton_schulz_keeps_exact_zeros_zero_and_respects_cutoff() -> None:
    norms = torch.tensor([0.0, 1e-13, 1e-7, 1.0], dtype=torch.float64)
    mapped = newton_schulz_scalar_magnitudes(norms)
    assert mapped[0] == 0.0
    assert 0.0 < mapped[1] < mapped[2]
    assert mapped[2] == mapped[3]


def test_muon_flattens_depth_profile_in_one_step_experiment() -> None:
    rows, summary = run_experiment()
    assert summary["last_to_first_gradient_ratio"] > 1e10
    selected = [
        row
        for row in rows
        if row["budget"] == 0.3 and row["optimizer"] == "exact_muon"
    ]
    control_norms = [float(row["control_norm"]) for row in selected]
    assert max(control_norms) - min(control_norms) < 1e-12
