#!/usr/bin/env python3
"""
Rerun 4 setups (4 momentum: 0.0, 0.3, 0.6, 0.7) with e8a6947-like settings:
  factor=1, rank=10, n_train=5000, batch_size=4*factor, num_epochs=250,
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
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.table.mmnn_vs import MMNN


def _setup_mpl_rc():
    """LaTeX-style plotting (from meanfield_cosine_multifreq_experiment)."""
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "normal"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 22
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
    mpl.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
    mpl.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.top"] = True


def _legend_momentum(meta):
    """Label for legend: η (eta)=momentum, R=rank, e.g. r'$\\eta$ 0.0, R=10'."""
    m = meta.get("momentum")
    if m is None:
        for p in (meta.get("config") or "").split("_"):
            if p.startswith("mom"):
                m = p[3:] if len(p) > 3 else p
                break
        if m is None:
            m = "?"
    r = meta.get("rank")
    if r is None:
        cfg = meta.get("config") or ""
        if "fullrank" in cfg:
            r = HIDDEN_WIDTH
        else:
            for p in cfg.split("_"):
                if p.startswith("rank") and len(p) > 4 and p[4:].isdigit():
                    r = int(p[4:])
                    break
            if r is None:
                r = HIDDEN_RANK
    return rf"$\eta$ = {m}, R = {r}"

OUTPUT_DIR = Path("experiments/table/rerun_four_setups")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def target_function(x, factor):
    return np.cos(2 * factor * np.pi * x)


# e8a6947-like: factor=1, rank=10, n_train=5000, batch_size=4*factor, 250 epochs
FACTOR = 1
HIDDEN_RANK = 10
HIDDEN_WIDTH = 1024
NUM_LAYERS = 2
N_TRAIN = 5000
BATCH_SIZE = max(1, 4 * FACTOR)
NUM_EPOCHS = 250
MAX_EPOCHS_PLOT = 800  # plot only up to this epoch


def train_one(cfg):
    """Train one config. Returns (name, all_losses, num_epochs, lr_reduction_epochs, all_lrs)."""
    name = cfg["name"]
    lr_init = cfg["lr_init"]
    momentum = cfg["momentum"]
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    n_train = cfg["n_train"]
    scheduler_params = cfg.get("scheduler_params", {})
    hidden_rank = cfg.get("hidden_rank", HIDDEN_RANK)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32

    ranks = [1] + [hidden_rank] * NUM_LAYERS + [1]
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

    desc = "fullrank" if hidden_rank == HIDDEN_WIDTH else f"mom {momentum}"
    pbar = tqdm(range(num_epochs), desc=desc, unit="epoch")
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
            "momentum": cfg["momentum"],
            "rank": cfg.get("hidden_rank", HIDDEN_RANK),
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

    # Plot 4 low-rank curves + full-rank if available; red bars only for mom 0.0
    _setup_mpl_rc()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    lw = 3.0
    for i, meta in enumerate(results):
        losses = meta["all_losses"][:MAX_EPOCHS_PLOT]
        lbl = _legend_momentum(meta)
        ax.semilogy(range(len(losses)), losses, color=colors[i % len(colors)], label=lbl, linewidth=lw, alpha=0.8)
        if i == 0:
            for j, ep in enumerate(meta.get("lr_reduction_epochs", [])):
                if ep < MAX_EPOCHS_PLOT and ep < len(meta["all_losses"]):
                    ax.axvline(x=ep, color="r", linestyle="--", linewidth=1, alpha=0.6,
                               label="LR reduction" if j == 0 else None)
    # Add full-rank curve if JSON exists
    fullrank_path = OUTPUT_DIR / f"{FULLRANK_NAME}_losses.json"
    if fullrank_path.exists():
        with open(fullrank_path) as f:
            fr = json.load(f)
        losses = fr["all_losses"][:MAX_EPOCHS_PLOT]
        ax.semilogy(range(len(losses)), losses, color=colors[4], label=_legend_momentum(fr), linewidth=lw, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Low rank versus full rank")
    ax.legend(loc="upper right", fontsize=18)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "four_setups_loss_with_lr_bars.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved {plot_path}")


FULLRANK_NAME = "factor1_fullrank_SGD_mom0.0_lr0.01_AdaptiveStagnation"


FULLRANK_NUM_EPOCHS = 1000  # full-rank (MLP-like) training length


def run_fullrank(out_dir=None):
    """Train one full-rank (R=width) config and save fullrank_loss.png."""
    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": FULLRANK_NAME,
        "hidden_rank": HIDDEN_WIDTH,
        "lr_init": 0.01,
        "momentum": 0.0,
        "num_epochs": FULLRANK_NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_train": N_TRAIN,
        "scheduler_params": {
            "lr_sequence": [0.01, 0.005, 0.001, 0.0005, 0.0001],
            "window_size": 10,
            "min_epochs_before_reduce": 20,
        },
    }
    name, all_losses, num_epochs, lr_reduction_epochs, all_lrs = train_one(cfg)
    meta = {
        "config": name,
        "momentum": 0.0,
        "rank": HIDDEN_WIDTH,
        "num_epochs": num_epochs,
        "all_losses": all_losses,
        "lr_reduction_epochs": lr_reduction_epochs,
        "all_lrs": all_lrs,
    }
    p = out_dir / f"{name}_losses.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {p.name} ({len(all_losses)} points, {len(lr_reduction_epochs)} LR reductions)")
    plot_fullrank(out_dir)


def plot_fullrank(out_dir=None):
    """Plot only the full-rank loss curve → fullrank_loss.png."""
    out_dir = Path(out_dir or OUTPUT_DIR)
    _setup_mpl_rc()
    candidates = sorted(out_dir.glob(f"{FULLRANK_NAME}_losses.json"))
    if not candidates:
        print(f"No {FULLRANK_NAME}_losses.json in {out_dir}. Run --full-rank first.")
        return
    with open(candidates[0]) as f:
        meta = json.load(f)
    losses = meta["all_losses"][:MAX_EPOCHS_PLOT]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.semilogy(range(len(losses)), losses, color="#1f77b4", label=_legend_momentum(meta), linewidth=3.0, alpha=0.8)
    for j, ep in enumerate(meta.get("lr_reduction_epochs", [])):
        if ep < MAX_EPOCHS_PLOT and ep < len(meta["all_losses"]):
            ax.axvline(x=ep, color="r", linestyle="--", linewidth=1, alpha=0.6, label="LR reduction" if j == 0 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Full rank: factor=1, R=1024, SGD mom=0.0 lr=0.01, AdaptiveStagnation")
    ax.legend(loc="upper right", fontsize=18)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "fullrank_loss.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved {plot_path}")


def plot_only(out_dir=None):
    """Regenerate the 4-setups plot from existing *_losses.json in out_dir (excludes fullrank)."""
    out_dir = Path(out_dir or OUTPUT_DIR)
    _setup_mpl_rc()
    pattern = sorted(p for p in out_dir.glob("*_losses.json") if "fullrank" not in p.name)
    if not pattern:
        print(f"No *_losses.json in {out_dir}. Run without --plot-only first.")
        return
    results = []
    for p in pattern:
        with open(p) as f:
            results.append(json.load(f))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    lw = 3.0
    for i, meta in enumerate(results):
        losses = meta["all_losses"][:MAX_EPOCHS_PLOT]
        lbl = _legend_momentum(meta)
        ax.semilogy(range(len(losses)), losses, color=colors[i % len(colors)], label=lbl, linewidth=lw, alpha=0.8)
        if i == 0:
            for j, ep in enumerate(meta.get("lr_reduction_epochs", [])):
                if ep < MAX_EPOCHS_PLOT and ep < len(meta["all_losses"]):
                    ax.axvline(x=ep, color="r", linestyle="--", linewidth=1, alpha=0.6,
                               label="LR reduction" if j == 0 else None)
    # Add full-rank curve if JSON exists
    fullrank_path = out_dir / f"{FULLRANK_NAME}_losses.json"
    if fullrank_path.exists():
        with open(fullrank_path) as f:
            fr = json.load(f)
        losses = fr["all_losses"][:MAX_EPOCHS_PLOT]
        ax.semilogy(range(len(losses)), losses, color=colors[4], label=_legend_momentum(fr), linewidth=lw, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Low rank versus full rank")
    ax.legend(loc="upper right", fontsize=18)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "four_setups_loss_with_lr_bars.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved {plot_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="Only regenerate 4-setups plot from existing *_losses.json")
    ap.add_argument("--full-rank", action="store_true", help="Train full-rank (R=1024) and plot fullrank_loss.png")
    ap.add_argument("--plot-fullrank-only", action="store_true", help="Only regenerate fullrank_loss.png from existing fullrank JSON")
    ap.add_argument("--out-dir", default=None, help="Output (or input for --plot-only) directory")
    args = ap.parse_args()
    if args.out_dir:
        OUTPUT_DIR = Path(args.out_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.full_rank:
        run_fullrank(OUTPUT_DIR)
    elif args.plot_fullrank_only:
        plot_fullrank(OUTPUT_DIR)
    elif args.plot_only:
        plot_only(OUTPUT_DIR)
    else:
        main()
