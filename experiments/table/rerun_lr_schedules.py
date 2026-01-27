#!/usr/bin/env python3
"""
Re-run factor1 rank10 SGD configs (different LR schedules), save results.json with
all_losses, all_lrs, lr_reduction_epochs. AdaptiveStagnation: lr_init=1e-2, on
stagnation (10-mean criteria) LR steps 1e-2 -> 5e-3 -> 1e-3 -> 5e-4 -> 1e-4.
Saves loss_evolution.png with red vertical bars at LR reduction epochs.

Usage:
  uv run python experiments/table/rerun_lr_schedules.py [--epochs N]
  (default 10000; use --epochs 200 for a quick check)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.table.mmnn_vs import MMNN

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------
FACTOR = 1
HIDDEN_WIDTH = 1024
HIDDEN_RANK = 10
NUM_LAYERS = 2
INTERVAL = [-1, 1]

# lr_sequence for AdaptiveStagnation: 1e-2 -> 5e-3 -> 1e-3 -> 5e-4 -> 1e-4
LR_SEQUENCE = [0.01, 0.005, 0.001, 0.0005, 0.0001]
ADAPTIVE_WINDOW = 10
ADAPTIVE_MIN_EPOCHS = 20

BASE = Path(__file__).resolve().parent / "results_lr_schedules_rerun"

# (momentum, lr_init, scheduler_type, scheduler_params or None)
CONFIGS = [
    (0.3, 0.005, None, None),
    (0.4, 0.01, "AdaptiveStagnation", {"lr_sequence": LR_SEQUENCE, "window_size": ADAPTIVE_WINDOW, "min_epochs_before_reduce": ADAPTIVE_MIN_EPOCHS}),
    (0.4, 0.001, None, None),
    (0.4, 0.005, None, None),
    (0.5, 0.01, "AdaptiveStagnation", {"lr_sequence": LR_SEQUENCE, "window_size": ADAPTIVE_WINDOW, "min_epochs_before_reduce": ADAPTIVE_MIN_EPOCHS}),
]


def target_function(x, factor):
    return np.cos(2 * factor * np.pi * x) + np.cos(2 * np.pi * x)


def run_one(momentum, lr_init, scheduler_type, scheduler_params, output_dir, num_epochs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # model
    ranks = [1] + [HIDDEN_RANK] * NUM_LAYERS + [1]
    widths = [HIDDEN_WIDTH] * (NUM_LAYERS + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)

    # data
    n_train = max(1, int(FACTOR * HIDDEN_WIDTH))
    batch_size = max(1, int(4 * FACTOR * 10))
    x_train = np.linspace(INTERVAL[0], INTERVAL[1], n_train)
    y_train = target_function(x_train, FACTOR)
    x_t = torch.tensor(x_train.reshape(-1, 1), device=device, dtype=dtype)
    y_t = torch.tensor(y_train.reshape(-1, 1), device=device, dtype=dtype)

    # optimizer: SGD only
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)

    # adaptive scheduler state (only when scheduler_type == AdaptiveStagnation)
    adaptive = None
    if scheduler_type == "AdaptiveStagnation" and scheduler_params:
        seq = scheduler_params.get("lr_sequence", LR_SEQUENCE)
        adaptive = {
            "lr_sequence": seq,
            "current_lr_index": 0,
            "window_size": scheduler_params.get("window_size", ADAPTIVE_WINDOW),
            "min_epochs_before_reduce": scheduler_params.get("min_epochs_before_reduce", ADAPTIVE_MIN_EPOCHS),
            "last_reduction_epoch": -1,
            "lr_reduction_epochs": [],
        }

    all_losses = []
    all_lrs = []
    min_loss = float("inf")
    min_loss_epoch = 0

    start = time.time()
    pbar = tqdm(range(num_epochs), desc=output_dir.name, unit="ep")

    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(x_t[idx])
            loss = nn.MSELoss()(pred, y_t[idx])
            if torch.isnan(loss) or torch.isinf(loss):
                pbar.write(f"NaN/Inf at epoch {epoch}, stopping")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if n_batches == 0:
            break
        epoch_loss /= n_batches
        if np.isnan(epoch_loss) or np.isinf(epoch_loss):
            break
        all_losses.append(epoch_loss)
        lr = optimizer.param_groups[0]["lr"]
        all_lrs.append(lr)

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            min_loss_epoch = epoch

        # AdaptiveStagnation: reduce LR when loss stagnates (mean of last window >= mean of previous window)
        if adaptive is not None:
            wi = adaptive["window_size"]
            min_ep = adaptive["min_epochs_before_reduce"]
            last = adaptive["last_reduction_epoch"]
            idx_cur = adaptive["current_lr_index"]
            seq = adaptive["lr_sequence"]
            if (
                epoch >= min_ep
                and epoch - last >= min_ep
                and len(all_losses) >= 2 * wi
                and idx_cur < len(seq) - 1
            ):
                recent = np.mean(all_losses[-wi:])
                prev = np.mean(all_losses[-2 * wi : -wi])
                if recent >= prev:
                    idx_cur += 1
                    new_lr = seq[idx_cur]
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    adaptive["current_lr_index"] = idx_cur
                    adaptive["last_reduction_epoch"] = epoch
                    adaptive["lr_reduction_epochs"].append(epoch)
                    pbar.write(f"  LR -> {new_lr:.2e} at epoch {epoch}")

        pbar.set_postfix(loss=f"{epoch_loss:.3e}", lr=f"{lr:.2e}")

    elapsed = time.time() - start
    lr_reduction_epochs = adaptive["lr_reduction_epochs"] if adaptive else []

    # results.json
    res = {
        "factor": FACTOR,
        "num_layers": NUM_LAYERS,
        "hidden_rank": HIDDEN_RANK,
        "hidden_width": HIDDEN_WIDTH,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr_init": lr_init,
        "optimizer_type": "SGD",
        "momentum": momentum,
        "scheduler_type": scheduler_type,
        "scheduler_params": scheduler_params,
        "min_loss": min_loss,
        "min_loss_epoch": min_loss_epoch,
        "final_loss": all_losses[-1] if all_losses else None,
        "training_time_seconds": elapsed,
        "all_losses": all_losses,
        "all_lrs": all_lrs,
        "lr_reduction_epochs": lr_reduction_epochs,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)

    # loss_evolution.png with red bars at lr_reduction_epochs (when AdaptiveStagnation)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(all_losses, "b-", linewidth=1.2, alpha=0.8, label="Loss")
    for i, e in enumerate(lr_reduction_epochs):
        if e < len(all_losses):
            ax.axvline(x=e, color="r", linestyle="--", linewidth=1.2, alpha=0.8, label="LR decay" if i == 0 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"mom {momentum}, lr {lr_init}, {scheduler_type or 'NoScheduler'}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # config.json
    cfg = {
        "factor": FACTOR,
        "num_layers": NUM_LAYERS,
        "hidden_rank": HIDDEN_RANK,
        "hidden_width": HIDDEN_WIDTH,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr_init": lr_init,
        "optimizer_type": "SGD",
        "momentum": momentum,
        "scheduler_type": scheduler_type,
        "scheduler_params": scheduler_params,
        "function": f"cos(2*{FACTOR}*pi*x)+cos(2*pi*x)",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return res


def dirname(mom, lr_init, sched):
    s = "NoScheduler" if sched is None else sched
    return f"factor{FACTOR}_rank{HIDDEN_RANK}_SGD_mom{mom}_lr{lr_init}_{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10000, help="epochs per config (default 10000)")
    args = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    for mom, lr_init, sched_type, sched_params in CONFIGS:
        name = dirname(mom, lr_init, sched_type)
        out = BASE / name
        run_one(mom, lr_init, sched_type, sched_params, out, args.epochs)
    print("done. results in", BASE)


if __name__ == "__main__":
    main()
