#!/usr/bin/env python3
"""
Rerun 4 setups (4 momentum: 0.0, 0.3, 0.6, 0.7) with e8a6947-like settings:
  factor=1, rank=10, n_train=5000, batch_size=4*factor, num_epochs=1000,
  SGD, lr_init=0.01, AdaptiveStagnation.

Saves per-config JSON: all_losses, num_epochs, lr_reduction_epochs, all_lrs.
Produces one plot: 4 loss curves + red vertical bars at LR reductions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.table.mmnn_vs import MMNN

OUTPUT_DIR = Path("experiments/table/rerun_four_setups")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def target_function(x, factor):
    return np.cos(2 * factor * np.pi * x)


# e8a6947-like: factor=1, rank=10, n_train=5000, batch_size=4*factor, 1000 epochs
FACTOR = 1
HIDDEN_RANK = 10
HIDDEN_WIDTH = 1024
NUM_LAYERS = 2
N_TRAIN = 5000
BATCH_SIZE = max(1, 4 * FACTOR)
NUM_EPOCHS = 1000


def train_one(cfg):
    """Train one config. Returns (name, all_losses, num_epochs, lr_reduction_epochs, all_lrs)."""
    name = cfg["name"]
    lr_init = cfg["lr_init"]
    momentum = cfg["momentum"]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    n_train = cfg["n_train"]
    scheduler_params = cfg.get("scheduler_params", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32

    ranks = [1] + [HIDDEN_RANK] * NUM_LAYERS + [1]
    widths = [HIDDEN_WIDTH] * (NUM_LAYERS + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)

    interval = [-1, 1]
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_function(x_train, FACTOR)
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)

    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)

    adaptive_scheduler = {
        "lr_sequence": scheduler_params.get("lr_sequence", [0.01, 0.005, 0.001, 0.0005, 0.0001]),
        "current_lr_index": 0,
        "window_size": scheduler_params.get("window_size", 10),
        "min_epochs_before_reduce": scheduler_params.get("min_epochs_before_reduce", 20),
        "last_reduction_epoch": -1,
        "lr_reduction_epochs": [],
    }

    all_losses = []
    all_lrs = []

    pbar = tqdm(range(num_epochs), desc=name, unit="epoch")
    for epoch in pbar:
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i : i + batch_size]
            x_batch = x_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.MSELoss()(y_pred, y_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                pbar.write(f"  NaN/Inf at epoch {epoch}, stopping")
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
        all_losses.append(float(epoch_loss))
        all_lrs.append(float(optimizer.param_groups[0]["lr"]))

        # AdaptiveStagnation
        lr_seq = adaptive_scheduler["lr_sequence"]
        wi = adaptive_scheduler["window_size"]
        min_ep = adaptive_scheduler["min_epochs_before_reduce"]
        last_red = adaptive_scheduler["last_reduction_epoch"]
        cur_idx = adaptive_scheduler["current_lr_index"]
        if (
            epoch >= min_ep
            and epoch - last_red >= min_ep
            and len(all_losses) >= 2 * wi
            and cur_idx < len(lr_seq) - 1
        ):
            recent = np.mean(all_losses[-wi:])
            prev = np.mean(all_losses[-2 * wi : -wi])
            if recent >= prev:
                cur_idx += 1
                new_lr = lr_seq[cur_idx]
                for g in optimizer.param_groups:
                    g["lr"] = new_lr
                adaptive_scheduler["current_lr_index"] = cur_idx
                adaptive_scheduler["last_reduction_epoch"] = epoch
                adaptive_scheduler["lr_reduction_epochs"].append(epoch)

        pbar.set_postfix(loss=f"{epoch_loss:.4e}")

    return name, all_losses, num_epochs, adaptive_scheduler["lr_reduction_epochs"], all_lrs


def main():
    configs = [
        {
            "name": "factor1_rank10_SGD_mom0.0_lr0.01_AdaptiveStagnation",
            "lr_init": 0.01,
            "momentum": 0.0,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "n_train": N_TRAIN,
            "scheduler_params": {
                "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "window_size": 10,
                "min_epochs_before_reduce": 20,
            },
        },
        {
            "name": "factor1_rank10_SGD_mom0.3_lr0.01_AdaptiveStagnation",
            "lr_init": 0.01,
            "momentum": 0.3,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "n_train": N_TRAIN,
            "scheduler_params": {
                "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "window_size": 10,
                "min_epochs_before_reduce": 20,
            },
        },
        {
            "name": "factor1_rank10_SGD_mom0.6_lr0.01_AdaptiveStagnation",
            "lr_init": 0.01,
            "momentum": 0.6,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "n_train": N_TRAIN,
            "scheduler_params": {
                "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "window_size": 10,
                "min_epochs_before_reduce": 20,
            },
        },
        {
            "name": "factor1_rank10_SGD_mom0.7_lr0.01_AdaptiveStagnation",
            "lr_init": 0.01,
            "momentum": 0.7,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "n_train": N_TRAIN,
            "scheduler_params": {
                "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "window_size": 10,
                "min_epochs_before_reduce": 20,
            },
        },
    ]

    results = []
    for cfg in configs:
        name, all_losses, num_epochs, lr_reduction_epochs, all_lrs = train_one(cfg)
        meta = {
            "config": name,
            "num_epochs": num_epochs,
            "all_losses": all_losses,
            "lr_reduction_epochs": lr_reduction_epochs,
            "all_lrs": all_lrs,
        }
        p = OUTPUT_DIR / f"{name}_losses.json"
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)
        results.append(meta)
        print(f"  Saved {p.name} ({len(all_losses)} points, {len(lr_reduction_epochs)} LR reductions)")

    # Plot 4 curves + red bars at LR reductions
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    first_lr_bar = True
    for i, meta in enumerate(results):
        losses = meta["all_losses"]
        lbl = meta["config"].replace("factor1_rank10_SGD_", "").replace("_AdaptiveStagnation", "")
        ax.semilogy(range(len(losses)), losses, color=colors[i % len(colors)], label=lbl, linewidth=1.5, alpha=0.8)
        for ep in meta.get("lr_reduction_epochs", []):
            if ep < len(losses):
                ax.axvline(x=ep, color="r", linestyle="--", linewidth=1, alpha=0.6, 
                           label="LR reduction" if first_lr_bar else None)
                first_lr_bar = False
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Four setups: factor=1, rank=10, SGD lr=0.01, AdaptiveStagnation (red bars = LR reduction)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "four_setups_loss_with_lr_bars.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved {plot_path}")


if __name__ == "__main__":
    main()
