import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

from experiments.helmholtz.shapes import make_grid, boundary_mask_from_domain, sample_shape_mask
from experiments.helmholtz.random_fields import (
    sample_smooth_random_field,
    sample_positive_field_from_gaussian,
    sample_boundary_forcing,
)
from experiments.helmholtz.pde_solve import solve_helmholtz_dirichlet


@dataclass(frozen=True)
class HelmholtzDatasetConfig:  # we group dataset hyperparameters #
    n_grid: int = 64  # we set grid size #
    n_train: int = 256  # we set number of train samples #
    n_test: int = 64  # we set number of test samples #
    shape_families_train: tuple[str, ...] = ("disk", "ellipse", "superellipse", "lshape")  # we set train shapes #
    shape_families_test: tuple[str, ...] = ("disk", "ellipse", "superellipse", "lshape", "star")  # we set test shapes #
    forcing_alpha: float = 2.5  # we set smoothness for forcing #
    forcing_amplitude: float = 1.0  # we set forcing scale #
    k2_alpha: float = 2.0  # we set smoothness for k^2 field #
    k_min: float = 2.0  # we set k range #
    k_max: float = 12.0  # we set k range #
    boundary_n_terms: int = 6  # we set boundary harmonics #
    boundary_amplitude: float = 1.0  # we set boundary amplitude #
    base_seed: int = 20260108  # we set base seed #


def _stable_rng(seed: int) -> np.random.Generator:  # we build a deterministic rng #
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)  # we return rng #


def _make_sample(
    x: np.ndarray,
    y: np.ndarray,
    h: float,
    rng: np.random.Generator,
    shape_family: str,
    cfg: HelmholtzDatasetConfig,
) -> dict[str, np.ndarray]:  # we generate one helmholtz sample #
    domain = sample_shape_mask(x, y, rng=rng, shape_family=shape_family)  # we sample domain #
    boundary = boundary_mask_from_domain(domain)  # we compute boundary mask #

    f_raw = sample_smooth_random_field(
        n=int(cfg.n_grid),
        rng=rng,
        alpha=float(cfg.forcing_alpha),
        amplitude=float(cfg.forcing_amplitude),
        mean=0.0,
    )  # we sample forcing #
    f = f_raw * domain.astype(np.float64)  # we zero outside domain #

    k_base = sample_smooth_random_field(
        n=int(cfg.n_grid),
        rng=rng,
        alpha=float(cfg.k2_alpha),
        amplitude=1.0,
        mean=0.0,
    )  # we sample base field #
    k = sample_positive_field_from_gaussian(k_base, min_value=float(cfg.k_min), max_value=float(cfg.k_max))  # we map to positive range #
    k2 = (k**2) * domain.astype(np.float64)  # we make k^2 and zero outside #

    g = sample_boundary_forcing(
        x=x,
        y=y,
        boundary_mask=boundary,
        rng=rng,
        n_terms=int(cfg.boundary_n_terms),
        amplitude=float(cfg.boundary_amplitude),
    )  # we sample boundary forcing #

    u = solve_helmholtz_dirichlet(k2=k2, f=f, domain_mask=domain, boundary_mask=boundary, g=g, h=float(h))  # we solve pde #

    return {
        "f": f.astype(np.float32),
        "k2": k2.astype(np.float32),
        "g": g.astype(np.float32),
        "u": u.astype(np.float32),
        "mask": domain.astype(np.float32),
        "bmask": boundary.astype(np.float32),
    }  # we return sample dict #


def generate_dataset_npz(
    out_file: Path,
    cfg: HelmholtzDatasetConfig,
) -> Path:  # we generate and save a dataset to npz #
    out_file = Path(out_file)  # we normalize path #
    out_file.parent.mkdir(parents=True, exist_ok=True)  # we ensure directory #

    x, y, h = make_grid(int(cfg.n_grid), x_min=-1.0, x_max=1.0)  # we make grid #
    x_f = x.astype(np.float32)  # we cast #
    y_f = y.astype(np.float32)  # we cast #

    def gen_split(n_samples: int, shape_families: tuple[str, ...], split_seed_offset: int) -> dict[str, np.ndarray]:  # we build split arrays #
        fs = []  # we collect #
        k2s = []  # we collect #
        gs = []  # we collect #
        us = []  # we collect #
        masks = []  # we collect #
        bmasks = []  # we collect #
        shape_ids = []  # we collect #
        for i in range(int(n_samples)):  # we loop samples #
            seed_i = int(cfg.base_seed) + int(split_seed_offset) + 100000 * i  # we derive seed #
            rng = _stable_rng(seed_i)  # we build rng #
            shape_family = str(shape_families[int(rng.integers(0, len(shape_families)))])  # we sample shape family #
            sample = _make_sample(x, y, h, rng=rng, shape_family=shape_family, cfg=cfg)  # we build sample #
            fs.append(sample["f"])  # we append #
            k2s.append(sample["k2"])  # we append #
            gs.append(sample["g"])  # we append #
            us.append(sample["u"])  # we append #
            masks.append(sample["mask"])  # we append #
            bmasks.append(sample["bmask"])  # we append #
            shape_ids.append(np.int64(shape_families.index(shape_family)))  # we encode shape id #
        return {
            "f": np.stack(fs, axis=0),
            "k2": np.stack(k2s, axis=0),
            "g": np.stack(gs, axis=0),
            "u": np.stack(us, axis=0),
            "mask": np.stack(masks, axis=0),
            "bmask": np.stack(bmasks, axis=0),
            "shape_id": np.stack(shape_ids, axis=0),
        }  # we return dict #

    train = gen_split(int(cfg.n_train), tuple(cfg.shape_families_train), split_seed_offset=0)  # we generate train #
    test = gen_split(int(cfg.n_test), tuple(cfg.shape_families_test), split_seed_offset=777777)  # we generate test #

    meta = {
        "config": asdict(cfg),
        "grid": {"n": int(cfg.n_grid), "x_min": -1.0, "x_max": 1.0, "h": float(h)},
        "channels": ["f", "k2", "g", "mask", "bmask"],
        "shape_families_train": list(cfg.shape_families_train),
        "shape_families_test": list(cfg.shape_families_test),
    }  # we build metadata #

    np.savez_compressed(
        out_file,
        x=x_f,
        y=y_f,
        train_f=train["f"],
        train_k2=train["k2"],
        train_g=train["g"],
        train_u=train["u"],
        train_mask=train["mask"],
        train_bmask=train["bmask"],
        train_shape_id=train["shape_id"],
        test_f=test["f"],
        test_k2=test["k2"],
        test_g=test["g"],
        test_u=test["u"],
        test_mask=test["mask"],
        test_bmask=test["bmask"],
        test_shape_id=test["shape_id"],
        meta=json.dumps(meta),
    )  # we save to npz #

    return out_file  # we return dataset path #

