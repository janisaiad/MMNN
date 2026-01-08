import numpy as np


def make_grid(n: int, x_min: float = -1.0, x_max: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:  # we build a cartesian grid #
    xs = np.linspace(x_min, x_max, n, dtype=np.float64)  # we make a 1d axis #
    x, y = np.meshgrid(xs, xs, indexing="ij")  # we make a 2d grid #
    h = float((x_max - x_min) / (n - 1))  # we store grid spacing #
    return x, y, h  # we return grid arrays #


def boundary_mask_from_domain(domain_mask: np.ndarray) -> np.ndarray:  # we compute a 4-neighbor boundary mask #
    m = domain_mask.astype(bool)  # we ensure boolean mask #
    up = np.zeros_like(m)  # we allocate #
    down = np.zeros_like(m)  # we allocate #
    left = np.zeros_like(m)  # we allocate #
    right = np.zeros_like(m)  # we allocate #
    up[1:, :] = m[:-1, :]  # we shift #
    down[:-1, :] = m[1:, :]  # we shift #
    left[:, 1:] = m[:, :-1]  # we shift #
    right[:, :-1] = m[:, 1:]  # we shift #
    neighbor_all_inside = up & down & left & right  # we detect interior by 4-neighbors #
    boundary = m & (~neighbor_all_inside)  # we mark boundary as non-interior domain points #
    return boundary  # we return boundary mask #


def shape_disk(x: np.ndarray, y: np.ndarray, center: tuple[float, float], radius: float) -> np.ndarray:  # we create a disk mask #
    cx, cy = center  # we unpack center #
    r2 = (x - cx) ** 2 + (y - cy) ** 2  # we compute squared radius #
    return r2 <= float(radius) ** 2  # we return mask #


def shape_ellipse(x: np.ndarray, y: np.ndarray, center: tuple[float, float], axes: tuple[float, float], angle: float) -> np.ndarray:  # we create an ellipse mask #
    cx, cy = center  # we unpack center #
    a, b = axes  # we unpack axes #
    ca = float(np.cos(angle))  # we compute cos #
    sa = float(np.sin(angle))  # we compute sin #
    xr = ca * (x - cx) + sa * (y - cy)  # we rotate #
    yr = -sa * (x - cx) + ca * (y - cy)  # we rotate #
    val = (xr / float(a)) ** 2 + (yr / float(b)) ** 2  # we compute ellipse equation #
    return val <= 1.0  # we return mask #


def shape_superellipse(x: np.ndarray, y: np.ndarray, center: tuple[float, float], axes: tuple[float, float], power: float, angle: float) -> np.ndarray:  # we create a superellipse mask #
    cx, cy = center  # we unpack center #
    a, b = axes  # we unpack axes #
    ca = float(np.cos(angle))  # we compute cos #
    sa = float(np.sin(angle))  # we compute sin #
    xr = ca * (x - cx) + sa * (y - cy)  # we rotate #
    yr = -sa * (x - cx) + ca * (y - cy)  # we rotate #
    p = float(power)  # we cast #
    val = (np.abs(xr) / float(a)) ** p + (np.abs(yr) / float(b)) ** p  # we compute superellipse #
    return val <= 1.0  # we return mask #


def shape_lshape(x: np.ndarray, y: np.ndarray, center: tuple[float, float], half_extent: float, cut_half_extent: float, angle: float) -> np.ndarray:  # we create an L-shaped mask #
    cx, cy = center  # we unpack center #
    ca = float(np.cos(angle))  # we compute cos #
    sa = float(np.sin(angle))  # we compute sin #
    xr = ca * (x - cx) + sa * (y - cy)  # we rotate #
    yr = -sa * (x - cx) + ca * (y - cy)  # we rotate #
    he = float(half_extent)  # we cast #
    che = float(cut_half_extent)  # we cast #
    outer = (np.abs(xr) <= he) & (np.abs(yr) <= he)  # we build outer square #
    cut = (xr >= 0.0) & (yr >= 0.0) & (xr <= che) & (yr <= che)  # we cut top-right quadrant corner #
    return outer & (~cut)  # we return l-shape mask #


def shape_star(x: np.ndarray, y: np.ndarray, center: tuple[float, float], r0: float, r1: float, m: int, angle: float) -> np.ndarray:  # we create a star-shaped mask #
    cx, cy = center  # we unpack center #
    ca = float(np.cos(angle))  # we compute cos #
    sa = float(np.sin(angle))  # we compute sin #
    xr = ca * (x - cx) + sa * (y - cy)  # we rotate #
    yr = -sa * (x - cx) + ca * (y - cy)  # we rotate #
    theta = np.arctan2(yr, xr)  # we compute angle #
    radius = np.sqrt(xr**2 + yr**2)  # we compute radius #
    rr = float(r0) + (float(r1) - float(r0)) * 0.5 * (1.0 + np.cos(int(m) * theta))  # we compute radial boundary #
    return radius <= rr  # we return mask #


def sample_shape_mask(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    shape_family: str,
) -> np.ndarray:  # we sample a random shape mask from a family #
    if shape_family == "disk":  # we dispatch #
        center = (float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.2, 0.2)))  # we sample center #
        radius = float(rng.uniform(0.45, 0.85))  # we sample radius #
        return shape_disk(x, y, center=center, radius=radius)  # we return #
    if shape_family == "ellipse":  # we dispatch #
        center = (float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.2, 0.2)))  # we sample center #
        axes = (float(rng.uniform(0.45, 0.9)), float(rng.uniform(0.35, 0.85)))  # we sample axes #
        angle = float(rng.uniform(0.0, np.pi))  # we sample angle #
        return shape_ellipse(x, y, center=center, axes=axes, angle=angle)  # we return #
    if shape_family == "superellipse":  # we dispatch #
        center = (float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.2, 0.2)))  # we sample center #
        axes = (float(rng.uniform(0.5, 0.95)), float(rng.uniform(0.5, 0.95)))  # we sample axes #
        power = float(rng.uniform(2.5, 6.0))  # we sample power #
        angle = float(rng.uniform(0.0, np.pi))  # we sample angle #
        return shape_superellipse(x, y, center=center, axes=axes, power=power, angle=angle)  # we return #
    if shape_family == "lshape":  # we dispatch #
        center = (float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.15, 0.15)))  # we sample center #
        he = float(rng.uniform(0.6, 0.9))  # we sample half extent #
        che = float(rng.uniform(0.25, 0.55) * he)  # we sample cut size #
        angle = float(rng.uniform(0.0, np.pi))  # we sample angle #
        return shape_lshape(x, y, center=center, half_extent=he, cut_half_extent=che, angle=angle)  # we return #
    if shape_family == "star":  # we dispatch #
        center = (float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.15, 0.15)))  # we sample center #
        r0 = float(rng.uniform(0.35, 0.55))  # we sample inner radius #
        r1 = float(rng.uniform(0.65, 0.95))  # we sample outer radius #
        m = int(rng.integers(3, 8))  # we sample spikes count #
        angle = float(rng.uniform(0.0, np.pi))  # we sample rotation #
        return shape_star(x, y, center=center, r0=r0, r1=r1, m=m, angle=angle)  # we return #
    raise ValueError(f"unknown shape_family={shape_family!r}")  # we fail loudly #

