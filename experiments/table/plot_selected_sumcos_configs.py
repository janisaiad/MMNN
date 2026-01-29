#!/usr/bin/env python3
"""
we plot for each selected sumcos config: (1) loss curve with red bars at LR divide epochs;
(2) for each LR divide, target vs prediction just before and 10 epochs after, using saved params.
usage: python plot_selected_sumcos_configs.py [run_name] [run_name ...]
  with no args we plot all configs in results_sumcos_selected_rerun.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.table.mmnn_vs import MMNN
from experiments.table.run_selected_sumcos_configs import (
    EPOCHS_AFTER_LR_DIVIDE,
    RESULTS_SELECTED_DIR,
    SELECTED_SUMCOS_CONFIGS,
)


def target_baseline(x, factor=1.0):
    """sum_{k=1}^{factor} cos(2 pi k x)"""
    if factor < 1:
        return np.cos(2 * np.pi * x)
    out = np.zeros_like(x, dtype=float)
    for k in range(1, int(factor) + 1):
        out += np.cos(2 * k * np.pi * x)
    return out


def setup_mpl():
    plt.rcParams["font.size"] = 12
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True


def load_model_and_predict(config, params_path, device, x_plot):
    """we build model from config, load state from params_path, return prediction on x_plot"""
    hidden_width = int(config.get("hidden_width", 1024))
    hidden_rank = int(config.get("hidden_rank", 10))
    num_layers = int(config.get("num_layers", 2))
    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [hidden_width] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    model.load_state_dict(torch.load(params_path, map_location=device))
    model.eval()
    x_t = torch.tensor(x_plot.reshape(-1, 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(x_t).cpu().numpy().ravel()
    return y_pred


def plot_one_config(run_dir, run_name):
    """we plot loss curve and before/after fit for each LR divide for one config"""
    config_path = run_dir / "config.json"
    losses_path = run_dir / "losses.json"
    if not config_path.exists() or not losses_path.exists():
        print("skip (no config/losses):", run_name)
        return
    with open(config_path) as f:
        config = json.load(f)
    with open(losses_path) as f:
        data = json.load(f)
    all_losses = data.get("all_losses", [])
    lr_reduction_epochs = data.get("lr_reduction_epochs", [])
    if not all_losses:
        print("skip (no losses):", run_name)
        return
    factor = int(config.get("factor", 3))
    setup_mpl()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_plot = np.linspace(-1, 1, 1000)
    y_target = target_baseline(x_plot, factor)

    n_divides = len(lr_reduction_epochs)
    n_rows = 1 + max(0, n_divides)
    fig = plt.figure(figsize=(12, 3.2 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, figure=fig)
    ax_loss = fig.add_subplot(gs[0, :])  # loss curve spans both columns

    # row 0: loss curve with red bars at LR divide
    epochs = range(1, len(all_losses) + 1)
    ax_loss.semilogy(epochs, all_losses, color="C2", linewidth=1.2)
    for ep in lr_reduction_epochs:
        ax_loss.axvline(ep, color="red", linestyle="--", alpha=0.8, linewidth=1)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (train)")
    ax_loss.set_title(f"{run_name} — loss curve (red = LR divide)")
    ax_loss.grid(True, alpha=0.3)

    # rows 1..n_divides: before (col 0) and after (col 1) for each LR divide
    for idx, e_before in enumerate(lr_reduction_epochs):
        row = idx + 1
        ax_before = fig.add_subplot(gs[row, 0])
        ax_after = fig.add_subplot(gs[row, 1])
        before_dir = run_dir / f"params_before_lr_divide_epoch_{e_before}"
        after_dir = run_dir / f"params_after_lr_divide_epoch_{e_before}_plus{EPOCHS_AFTER_LR_DIVIDE}"
        before_pth = before_dir / "model_parameters.pth"
        after_pth = after_dir / "model_parameters.pth"
        if before_pth.exists():
            y_pred_before = load_model_and_predict(config, before_pth, device, x_plot)
            ax_before.plot(x_plot, y_target, color="C0", linewidth=1.5, label="target")
            ax_before.plot(x_plot, y_pred_before, color="C1", linestyle="--", linewidth=1, label="pred")
            ax_before.legend(loc="upper right", fontsize=9)
            ax_before.set_title(f"Before LR divide (epoch {e_before})")
        else:
            ax_before.text(0.5, 0.5, f"no params\nepoch {e_before}", ha="center", va="center", transform=ax_before.transAxes)
        ax_before.set_xlabel("$x$")
        ax_before.set_ylabel("$y$")
        ax_before.grid(True, alpha=0.3)
        ax_before.set_xlim(-1, 1)
        if after_pth.exists():
            y_pred_after = load_model_and_predict(config, after_pth, device, x_plot)
            ax_after.plot(x_plot, y_target, color="C0", linewidth=1.5, label="target")
            ax_after.plot(x_plot, y_pred_after, color="C1", linestyle="--", linewidth=1, label="pred")
            ax_after.legend(loc="upper right", fontsize=9)
            ax_after.set_title(f"After LR divide (epoch {e_before + EPOCHS_AFTER_LR_DIVIDE})")
        else:
            ax_after.text(0.5, 0.5, f"no params\nepoch {e_before}+{EPOCHS_AFTER_LR_DIVIDE}", ha="center", va="center", transform=ax_after.transAxes)
        ax_after.set_xlabel("$x$")
        ax_after.set_ylabel("$y$")
        ax_after.grid(True, alpha=0.3)
        ax_after.set_xlim(-1, 1)

    plt.suptitle(f"{run_name}  factor={factor}, N={config.get('n_train')}, bs={config.get('batch_size')}, L={config.get('num_layers')}", y=1.002)
    plt.tight_layout()
    out_path = run_dir / "plot_loss_and_fit_before_after_lr_divides.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out_path)


def main():
    base = Path(__file__).resolve().parent
    results_dir = base / "results_sumcos_selected_rerun"
    if not results_dir.is_dir():
        print("results dir not found:", results_dir)
        return
    # run names: from argv or all selected config names that exist
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if argv:
        run_names = argv
    else:
        run_names = [c["name"] for c in SELECTED_SUMCOS_CONFIGS]
    for run_name in run_names:
        run_dir = results_dir / run_name
        if not run_dir.is_dir():
            print("run dir not found:", run_dir)
            continue
        print("plotting", run_name)
        plot_one_config(run_dir, run_name)
    print("done.")


if __name__ == "__main__":
    main()
