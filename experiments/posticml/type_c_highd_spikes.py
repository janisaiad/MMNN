#!/usr/bin/env python3
"""
Type C (post-ICML): regression on [-1,1]^d with overlapping Gaussian bumps; M=1024; ranks[0]=d.

Usage:
  python experiments/posticml/type_c_highd_spikes.py --quick
  python experiments/posticml/type_c_highd_spikes.py --d 10 --n-train 200 --hidden-rank 15
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from experiments.table.run_scaling_law_depth_width import (  # noqa: E402
    SWEEP_LR_STOP_BELOW,
    SWEEP_MIN_LOSS_DIVISOR,
    _sweep_lr_sequence,
)
from experiments.table.mmnn_vs import MMNN  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "results" / "type_c_highd_spikes"
M_FIXED = 1024


def sample_x_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    """Quasi-uniform in [-1,1]^d: Sobol on power-of-2 length (avoids scipy warning), else uniform."""
    rng = np.random.default_rng(seed)
    try:
        from scipy.stats import qmc

        n2 = 1 << int(np.ceil(np.log2(max(n, 1))))
        eng = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = eng.random(n2)[:n]
        return 2.0 * u - 1.0
    except Exception:
        return rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float64)


def gaussian_mixture_target(x: np.ndarray, centers: np.ndarray, sigma: float, amps: np.ndarray) -> np.ndarray:
    """x: (n,d), centers: (K,d), amps: (K,) -> (n,)"""
    # (n,1,d) - (1,K,d)
    diff = x[:, None, :] - centers[None, :, :]
    sq = np.sum(diff**2, axis=-1)
    g = np.exp(-sq / (2 * sigma**2))
    return (g * amps).sum(axis=1)


def sample_x_mixed(n: int, d: int, centers: np.ndarray, sigma: float, seed: int, frac_near: float = 0.5):
    """Half uniform on cube (hard off-manifold), half near random bump centers (signal)."""
    rng = np.random.default_rng(seed)
    k = centers.shape[0]
    n_near = int(round(n * frac_near))
    n_uni = n - n_near
    x_uni = sample_x_hypercube(n_uni, d, seed + 1)
    idx = rng.integers(0, k, size=n_near)
    noise = rng.normal(0, sigma, size=(n_near, d))
    x_near = np.clip(centers[idx] + noise, -1.0, 1.0)
    x = np.vstack([x_uni, x_near])
    perm = rng.permutation(n)
    return x[perm]


def train_run(
    *,
    d: int,
    n_train: int,
    n_test: int,
    hidden_rank: int,
    num_layers: int,
    num_epochs: int,
    batch_size: int,
    momentum: float,
    seed: int,
    n_bumps: int,
    sigma: float,
    out_dir: Path,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32

    rng = np.random.default_rng(seed + 17)
    centers = rng.uniform(-0.7, 0.7, size=(n_bumps, d)).astype(np.float64)
    amps = rng.uniform(0.5, 1.5, size=(n_bumps,)).astype(np.float64)

    x_train = sample_x_mixed(n_train, d, centers, sigma, seed, frac_near=0.5)
    y_train = gaussian_mixture_target(x_train, centers, sigma, amps)
    x_test = sample_x_mixed(n_test, d, centers, sigma, seed + 999, frac_near=0.5)
    y_test = gaussian_mixture_target(x_test, centers, sigma, amps)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) + 1e-8
    y_train_z = (y_train - y_mean) / y_std
    y_test_z = (y_test - y_mean) / y_std

    x_train_t = torch.tensor(x_train, device=device, dtype=mydtype)
    y_train_t = torch.tensor(y_train_z.reshape(-1, 1), device=device, dtype=mydtype)
    x_test_t = torch.tensor(x_test, device=device, dtype=mydtype)
    y_test_t = torch.tensor(y_test_z.reshape(-1, 1), device=device, dtype=mydtype)

    ranks = [d] + [hidden_rank] * num_layers + [1]
    widths = [M_FIXED] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)

    lr_seq = _sweep_lr_sequence()
    lr_init = lr_seq[0]
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)

    current_lr_index = 0
    last_reduction_epoch = -1
    window_size = 10
    min_epochs_before_reduce = 20
    all_losses = []
    init_loss = None
    min_loss_so_far = float("inf")

    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        idx = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n_train, batch_size):
            b = idx[i : i + batch_size]
            xb = x_train_t[b]
            yb = y_train_t[b]
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        if nb == 0:
            break
        epoch_loss /= nb
        all_losses.append(epoch_loss)
        if init_loss is None:
            init_loss = epoch_loss
        min_loss_so_far = min(min_loss_so_far, epoch_loss)

        if (
            epoch >= min_epochs_before_reduce
            and epoch - last_reduction_epoch >= min_epochs_before_reduce
            and len(all_losses) >= 2 * window_size
            and current_lr_index < len(lr_seq) - 1
        ):
            recent = np.mean(all_losses[-window_size:])
            prev = np.mean(all_losses[-2 * window_size : -window_size])
            if recent >= prev:
                current_lr_index += 1
                for g in optimizer.param_groups:
                    g["lr"] = lr_seq[current_lr_index]
                last_reduction_epoch = epoch

        if optimizer.param_groups[0]["lr"] < SWEEP_LR_STOP_BELOW:
            break

        if (epoch + 1) % max(1, num_epochs // 5) == 0 or epoch == 0:
            print(f"  ep {epoch+1}/{num_epochs} loss={epoch_loss:.4e} lr={optimizer.param_groups[0]['lr']:.2e}")

    model.eval()
    with torch.no_grad():
        pred_z = model(x_test_t)
        te_z = nn.MSELoss()(pred_z, y_test_t).item()
        pred_raw = pred_z * y_std + y_mean
        y_test_raw = torch.tensor(y_test.reshape(-1, 1), device=device, dtype=mydtype)
        te_raw = nn.MSELoss()(pred_raw, y_test_raw).item()
        vy = float(torch.var(y_test_raw))
        nmse = float(te_raw / (vy + 1e-12))

    payload = {
        "d": d,
        "M_width": M_FIXED,
        "hidden_rank": hidden_rank,
        "num_layers": num_layers,
        "n_train": n_train,
        "n_test": n_test,
        "n_bumps": n_bumps,
        "sigma": sigma,
        "momentum": momentum,
        "y_mean": y_mean,
        "y_std": y_std,
        "final_train_loss_z": float(all_losses[-1]) if all_losses else None,
        "final_test_mse_z": float(te_z),
        "final_test_mse_raw": float(te_raw),
        "test_nmse_var": nmse,
        "epochs_run": len(all_losses),
        "params": sum(p.numel() for p in model.parameters()),
    }
    torch.save(model.state_dict(), out_dir / "model_parameters.pth")
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  done test_mse_raw={te_raw:.4e} test_mse_z={te_z:.4e} -> {out_dir}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Type C: high-d overlapping Gaussian bumps, M=1024")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--d", type=int, default=5)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--hidden-rank", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--bumps", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="re-run even if results.json exists")
    args = p.parse_args()

    if args.quick:
        epochs = args.epochs or 300
        n_train = min(args.n_train, 120)
        d = min(args.d, 5)
    else:
        epochs = args.epochs or 4000
        n_train = args.n_train
        d = args.d

    name = f"d{d}_N{n_train}_M{M_FIXED}_r{args.hidden_rank}_L{args.num_layers}_mixed"
    out_dir = OUT_ROOT / name
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and (out_dir / "results.json").exists():
        print(f"skip (done): {name}")
        return

    train_run(
        d=d,
        n_train=n_train,
        n_test=args.n_test,
        hidden_rank=args.hidden_rank,
        num_layers=args.num_layers,
        num_epochs=epochs,
        batch_size=args.batch_size,
        momentum=args.momentum,
        seed=args.seed,
        n_bumps=args.bumps,
        sigma=args.sigma,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
