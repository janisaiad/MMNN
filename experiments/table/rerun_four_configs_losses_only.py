#!/usr/bin/env python3
"""
Rerun the 4 configs and save only loss JSON (no PNGs, no GIFs).
At the end, plot the 4 loss curves on one figure.

Configs:
  1. factor1_rank10_SGD_mom0.6_lr0.01_AdaptiveStagnation
  2. factor1_rank10_SGD_mom0.6_lr0.001_NoScheduler
  3. factor1_rank10_SGD_mom0.6_lr0.005_NoScheduler
  4. factor1_rank10_SGD_mom0.7_lr0.01_AdaptiveStagnation
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

# Output: only loss JSONs + one final plot
OUTPUT_DIR = Path("experiments/table/rerun_four_losses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def target_function(x, factor):
    return np.cos(2 * factor * np.pi * x)


# Factor, rank, width, L common to all
FACTOR = 1
HIDDEN_RANK = 10
HIDDEN_WIDTH = 1024
NUM_LAYERS = 2


def train_one(cfg):
    """Train one config. Returns (config_name, all_losses). No plots, no extra files."""
    name = cfg["name"]
    lr_init = cfg["lr_init"]
    momentum = cfg["momentum"]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    scheduler_type = cfg.get("scheduler_type")
    scheduler_params = cfg.get("scheduler_params", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32

    # Model
    ranks = [1] + [HIDDEN_RANK] * NUM_LAYERS + [1]
    widths = [HIDDEN_WIDTH] * (NUM_LAYERS + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)

    # Data
    n_train = max(1, int(FACTOR * HIDDEN_WIDTH))
    interval = [-1, 1]
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_function(x_train, FACTOR)
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)

    # Optimizer: SGD only
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)

    # Scheduler
    adaptive_scheduler = None
    if scheduler_type == "AdaptiveStagnation":
        adaptive_scheduler = {
            "lr_sequence": scheduler_params.get("lr_sequence", [0.01, 0.005, 0.001, 0.0005, 0.0001]),
            "current_lr_index": 0,
            "window_size": scheduler_params.get("window_size", 10),
            "min_epochs_before_reduce": scheduler_params.get("min_epochs_before_reduce", 20),
            "last_reduction_epoch": -1,
        }

    all_losses = []

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

        # AdaptiveStagnation
        if adaptive_scheduler is not None:
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

        pbar.set_postfix(loss=f"{epoch_loss:.4e}")

    return name, all_losses, num_epochs


def main():
    configs = [
        {
            "name": "factor1_rank10_SGD_mom0.6_lr0.01_AdaptiveStagnation",
            "lr_init": 0.01,
            "momentum": 0.6,
            "num_epochs": 10000,
            "batch_size": 40,
            "scheduler_type": "AdaptiveStagnation",
            "scheduler_params": {
                "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "window_size": 10,
                "min_epochs_before_reduce": 20,
            },
        },
        {
            "name": "factor1_rank10_SGD_mom0.6_lr0.001_NoScheduler",
            "lr_init": 0.001,
            "momentum": 0.6,
            "num_epochs": 250,
            "batch_size": 4,
            "scheduler_type": None,
            "scheduler_params": {},
        },
        {
            "name": "factor1_rank10_SGD_mom0.6_lr0.005_NoScheduler",
            "lr_init": 0.005,
            "momentum": 0.6,
            "num_epochs": 250,
            "batch_size": 4,
            "scheduler_type": None,
            "scheduler_params": {},
        },
        {
            "name": "factor1_rank10_SGD_mom0.7_lr0.01_AdaptiveStagnation",
            "lr_init": 0.01,
            "momentum": 0.7,
            "num_epochs": 10000,
            "batch_size": 40,
            "scheduler_type": "AdaptiveStagnation",
            "scheduler_params": {
                "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "window_size": 10,
                "min_epochs_before_reduce": 20,
            },
        },
    ]

    json_paths = []
    for cfg in configs:
        name, all_losses, num_epochs = train_one(cfg)
        meta = {
            "config": name,
            "num_epochs": num_epochs,
            "all_losses": all_losses,
        }
        p = OUTPUT_DIR / f"{name}_losses.json"
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)
        json_paths.append(p)
        print(f"  Saved {p.name} ({len(all_losses)} points)")

    # Plot the 4 losses
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, p in enumerate(json_paths):
        with open(p) as f:
            d = json.load(f)
        losses = d["all_losses"]
        lbl = d["config"]
        ax.semilogy(range(len(losses)), losses, color=colors[i % len(colors)], label=lbl, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Four configs: factor=1, rank=10, SGD")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "four_losses_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved {plot_path}")


if __name__ == "__main__":
    main()
