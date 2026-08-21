"""Finite representatives of every spectral order/sign/degeneracy class."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralCase:
    name: str
    family: str
    eigenvalues: tuple[float, ...]
    purpose: str


# Permuting eigenvalues or rotating eigenvectors does not change the dynamics up
# to an orthogonal conjugacy.  These representatives therefore cover the finite
# qualitative taxonomy: inertia, extreme-value dominance, zeros, and degeneracy.
SPECTRAL_CASES = (
    SpectralCase("pd_simple", "positive_definite", (3.0, 2.0, 1.0, 0.4), "simple positive maximum"),
    SpectralCase("pd_flat_top", "positive_definite", (3.0, 3.0, 1.0, 0.4), "repeated positive maximum"),
    SpectralCase("psd_kernel", "positive_semidefinite", (3.0, 1.0, 0.0, 0.0), "nontrivial kernel"),
    SpectralCase("zero", "zero", (0.0, 0.0, 0.0, 0.0), "fully stationary control"),
    SpectralCase("nd_simple", "negative_definite", (-0.4, -1.0, -2.0, -4.0), "paper regime, simple minimum"),
    SpectralCase("nd_flat_bottom", "negative_definite", (-0.4, -1.0, -4.0, -4.0), "repeated negative minimum"),
    SpectralCase("nsd_kernel", "negative_semidefinite", (0.0, -1.0, -2.0, -4.0), "zero maximum and kernel"),
    SpectralCase("mixed_positive_dominant", "indefinite", (3.0, 1.0, -0.5, -2.0), "paper cone condition"),
    SpectralCase("mixed_negative_dominant", "indefinite", (1.0, 0.4, -1.0, -4.0), "stable negative-mode polarization window"),
    SpectralCase("mixed_equal_extremes", "indefinite", (2.0, 0.5, -0.5, -2.0), "equal spectral radius at both signs"),
    SpectralCase("mixed_two_positive", "indefinite", (2.0, 1.8, -0.5, -3.0), "small positive spectral gap"),
    SpectralCase("mixed_single_positive", "indefinite", (1.0, -0.2, -0.8, -2.0), "no positive transverse modes"),
    SpectralCase("mixed_flat_positive", "indefinite", (2.0, 2.0, -0.5, -3.0), "degenerate positive maximum"),
    SpectralCase("mixed_flat_negative", "indefinite", (1.0, 0.3, -3.0, -3.0), "degenerate negative minimum"),
    SpectralCase("mixed_with_zero", "indefinite", (2.0, 0.0, -0.5, -3.0), "interior zero mode"),
)

