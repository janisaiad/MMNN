#!/usr/bin/env python3
"""
we plot the 10 functions that form the low-rank representation after the 1st layer (the R bottleneck
activations right after the first width→rank projection) for configs f3_N768_bs8_L2 and f3_N768_bs8_L3:
before and after each LR reduce (same rule: just before and 10 epochs after), using saved params.
We plot those R curves plus full prediction and target.
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(_REPO))

from experiments.table.mmnn_vs import MMNN
from experiments.table.run_selected_sumcos_configs import (
    EPOCHS_AFTER_LR_DIVIDE,
    RESULTS_SELECTED_DIR,
)

RUN_NAMES = ["f3_N768_bs8_L2", "f3_N768_bs8_L3"]


def target_baseline(x, factor=1.0):
    """sum_{k=1}^{factor} cos(2 pi k x)"""
    if factor < 1:
        return np.cos(2 * np.pi * x)
    out = np.zeros_like(x, dtype=float)
    for k in range(1, int(factor) + 1):
        out += np.cos(2 * k * np.pi * x)
    return out


def forward_after_first_layer(model, x_tensor):
    """we run only the first layer (rank→width, ReLU, width→rank) and return the R bottleneck activations (batch, R)."""
    x = model.fcs[0](x_tensor)
    x = torch.relu(x)
    x = model.fcs[1](x)
    return x


def load_model_and_get_first_layer_activations(config, params_path, device, x_plot, hidden_rank):
    """we build model, load state; return full prediction and the R activations after the 1st layer (as list of R curves)."""
    hidden_width = int(config.get("hidden_width", 1024))
    num_layers = int(config.get("num_layers", 2))
    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [hidden_width] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    model.load_state_dict(torch.load(params_path, map_location=device))
    model.eval()
    x_t = torch.tensor(x_plot.reshape(-1, 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        full = model(x_t).cpu().numpy().ravel()
        after_first = forward_after_first_layer(model, x_t)
        partials = [after_first[:, r].cpu().numpy().ravel() for r in range(hidden_rank)]
    return full, partials


def setup_mpl():
    plt.rcParams["font.size"] = 11
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True


def plot_one_config(run_name):
    """we plot loss + first-layer bottleneck activations before/after each LR divide for one config."""
    run_dir = RESULTS_SELECTED_DIR / run_name
    if not run_dir.is_dir():
        print("run dir not found:", run_dir)
        return
    config_path = run_dir / "config.json"
    losses_path = run_dir / "losses.json"
    if not config_path.exists() or not losses_path.exists():
        print("config or losses not found in", run_dir)
        return
    with open(config_path) as f:
        config = json.load(f)
    with open(losses_path) as f:
        data = json.load(f)
    lr_reduction_epochs = data.get("lr_reduction_epochs", [])
    all_losses = data.get("all_losses", [])
    factor = int(config.get("factor", 3))
    hidden_rank = int(config.get("hidden_rank", 10))
    num_layers = int(config.get("num_layers", 2))
    n_train = int(config.get("n_train", 768))
    batch_size = int(config.get("batch_size", 8))
    setup_mpl()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_plot = np.linspace(-1, 1, 1000)
    y_target = target_baseline(x_plot, factor)

    n_divides = len(lr_reduction_epochs)
    n_rows = 1 + max(0, n_divides)
    fig = plt.figure(figsize=(14, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, figure=fig)
    ax_loss = fig.add_subplot(gs[0, :])
    epochs = range(1, len(all_losses) + 1)
    ax_loss.semilogy(epochs, all_losses, color="C2", linewidth=1.2)
    for ep in lr_reduction_epochs:
        ax_loss.axvline(ep, color="red", linestyle="--", alpha=0.8, linewidth=1)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (train)")
    ax_loss.set_title(f"{run_name} — loss curve (red = LR divide)")
    ax_loss.grid(True, alpha=0.3)

    try:
        cmap = mpl.colormaps["tab10"].resampled(hidden_rank)
    except Exception:
        cmap = plt.cm.tab10
    for idx, e_before in enumerate(lr_reduction_epochs):
        row = idx + 1
        ax_before = fig.add_subplot(gs[row, 0])
        ax_after = fig.add_subplot(gs[row, 1])
        before_dir = run_dir / f"params_before_lr_divide_epoch_{e_before}"
        after_dir = run_dir / f"params_after_lr_divide_epoch_{e_before}_plus{EPOCHS_AFTER_LR_DIVIDE}"
        before_pth = before_dir / "model_parameters.pth"
        after_pth = after_dir / "model_parameters.pth"

        for ax, pth, title_prefix, epoch_label in [
            (ax_before, before_pth, "Before", str(e_before)),
            (ax_after, after_pth, "After", str(e_before + EPOCHS_AFTER_LR_DIVIDE)),
        ]:
            ax.plot(x_plot, y_target, color="black", linewidth=2, label="target")
            if pth.exists():
                full, partials = load_model_and_get_first_layer_activations(config, pth, device, x_plot, hidden_rank)
                ax.plot(x_plot, full, color="C0", linewidth=1.8, linestyle="-", label="full pred")
                for r in range(hidden_rank):
                    ax.plot(x_plot, partials[r], color=cmap(r / max(1, hidden_rank - 1)), linewidth=0.9, alpha=0.85, label=f"rank {r} (after L1)")
                if idx == 0:
                    ax.legend(loc="upper left", fontsize=7, ncol=3)
            else:
                ax.text(0.5, 0.5, "no params", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1, 1)
            ax.set_ylim(np.min(y_target) - 0.5, np.max(y_target) + 0.5)
            ax.set_title(f"{title_prefix} LR divide (epoch {epoch_label})")
    plt.suptitle(f"{run_name} — 10 bottleneck activations after 1st layer (rank r = 0..{hidden_rank - 1})  factor={factor}, N={n_train}, bs={batch_size}, L={num_layers}", y=1.002)
    plt.tight_layout()
    out_path = run_dir / "plot_partial_before_after_lr_divides.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out_path)


def main():
    for run_name in RUN_NAMES:
        print("plotting", run_name)
        plot_one_config(run_name)
    print("done.")


if __name__ == "__main__":
    main()
