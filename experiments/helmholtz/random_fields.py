import numpy as np


def sample_smooth_random_field(
    n: int,
    rng: np.random.Generator,
    alpha: float = 2.5,
    amplitude: float = 1.0,
    mean: float = 0.0,
) -> np.ndarray:  # we sample a smooth random field using spectral decay #
    noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))  # we sample complex noise #
    kx = np.fft.fftfreq(n).reshape(-1, 1)  # we build kx #
    ky = np.fft.fftfreq(n).reshape(1, -1)  # we build ky #
    k2 = kx**2 + ky**2  # we build squared frequency #
    filt = (1.0 + (2.0 * np.pi) ** 2 * k2) ** (-float(alpha) / 2.0)  # we build decay filter #
    field_hat = noise * filt  # we filter in fourier domain #
    field = np.fft.ifft2(field_hat).real  # we invert fft #
    field = field - np.mean(field)  # we center #
    std = float(np.std(field) + 1e-12)  # we compute std #
    field = field / std  # we normalize #
    field = float(amplitude) * field + float(mean)  # we scale and shift #
    return field.astype(np.float64)  # we return field #


def sample_positive_field_from_gaussian(
    base: np.ndarray,
    min_value: float,
    max_value: float,
) -> np.ndarray:  # we map a gaussian-like field to a positive bounded range #
    z = base.astype(np.float64)  # we cast #
    z = (z - np.mean(z)) / (np.std(z) + 1e-12)  # we standardize #
    s = 1.0 / (1.0 + np.exp(-z))  # we squash #
    out = float(min_value) + (float(max_value) - float(min_value)) * s  # we scale to range #
    return out.astype(np.float64)  # we return #


def sample_boundary_forcing(
    x: np.ndarray,
    y: np.ndarray,
    boundary_mask: np.ndarray,
    rng: np.random.Generator,
    n_terms: int = 6,
    amplitude: float = 1.0,
) -> np.ndarray:  # we sample a boundary forcing g on the boundary and extend by zeros #
    theta = np.arctan2(y, x)  # we compute polar angle about origin #
    g = np.zeros_like(x, dtype=np.float64)  # we allocate #
    for _ in range(int(n_terms)):  # we sum a few harmonics #
        m = int(rng.integers(1, 9))  # we sample frequency #
        phase = float(rng.uniform(0.0, 2.0 * np.pi))  # we sample phase #
        w = float(rng.normal(0.0, 1.0))  # we sample weight #
        g = g + w * np.sin(m * theta + phase)  # we add harmonic #
    g = g / (float(np.std(g[boundary_mask]) + 1e-12))  # we normalize on boundary #
    g = float(amplitude) * g  # we scale #
    gout = np.zeros_like(g)  # we allocate #
    gout[boundary_mask] = g[boundary_mask]  # we write boundary values #
    return gout.astype(np.float64)  # we return #

