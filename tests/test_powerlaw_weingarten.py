import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "feynman" / "run_powerlaw_early_stopping.py"
SPEC = importlib.util.spec_from_file_location("powerlaw_early_stopping", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

aligned_risk_curve = MODULE.aligned_risk_curve
block_haar_risk_moments = MODULE.block_haar_risk_moments
kernel_risk_curve = MODULE.kernel_risk_curve
orthogonal_twirl_moments = MODULE.orthogonal_twirl_moments
powerlaw_spectrum = MODULE.powerlaw_spectrum


def test_aligned_kernel_reproduces_powerlaw_risk_formula():
    eta = powerlaw_spectrum(24, alpha=0.5)
    times = np.array([0.01, 0.1, 0.5])
    expected, expected_bias, expected_variance = aligned_risk_curve(
        eta, sample_size=32, beta=1.7, times=times
    )
    actual, actual_bias, actual_variance, _, _ = kernel_risk_curve(
        np.diag(eta), eta, sample_size=32, beta=1.7, times=times
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_bias, expected_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_variance, expected_variance, rtol=1e-12, atol=1e-12)


def test_orthogonal_twirl_is_deterministic_for_constant_spectrum():
    left = np.arange(1.0, 9.0)
    right = np.full(8, 3.25)
    mean, variance = orthogonal_twirl_moments(left, right)
    np.testing.assert_allclose(mean, 3.25 * np.sum(left))
    np.testing.assert_allclose(variance, 0.0, atol=1e-14)


def test_block_haar_with_constant_tail_cost_has_zero_variance():
    eta = powerlaw_spectrum(20, alpha=0.7)
    # At t=0 the total spectral cost is identically 1/2 across modes.
    mean, variance = block_haar_risk_moments(
        eta,
        eta,
        aligned_head=5,
        sample_size=16,
        beta=2.0,
        time=0.0,
    )
    np.testing.assert_allclose(mean, 0.5 * np.sum(eta), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(variance, 0.0, atol=1e-14)
