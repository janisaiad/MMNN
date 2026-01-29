#!/usr/bin/env python3
"""
we plot for a baseline sweep run: (1) function to fit, (2) model prediction if params saved, (3) loss curve.
usage: python plot_baseline_sweep_run.py f3_N768_bs4_L3 [--results-dir results_baseline_sweep_sumcos]
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


def target_baseline(x, factor=1.0):
    """sum_{k=1}^{factor} cos(2 pi k x)"""
    if factor < 1:
        return np.cos(2 * np.pi * x)
    out = np.zeros_like(x, dtype=float)
    for k in range(1, int(factor) + 1):
        out += np.cos(2 * k * np.pi * x)
    return out


def setup_mpl():
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.rcParams["font.size"] = 18
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 22
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "f3_N768_bs4_L3"
    base = Path(__file__).resolve().parent
    results_dir = base / "results_baseline_sweep_sumcos"
    if "--results-dir" in sys.argv:
        i = sys.argv.index("--results-dir")
        if i + 1 < len(sys.argv):
            results_dir = base / sys.argv[i + 1]
    run_dir = results_dir / run_name
    if not run_dir.is_dir():
        print("run dir not found:", run_dir)
        return
    config_path = run_dir / "config.json"
    losses_path = run_dir / "losses.json"
    if not config_path.exists() or not losses_path.exists():
        print("config.json or losses.json not found in", run_dir)
        return
    with open(config_path) as f:
        config = json.load(f)
    with open(losses_path) as f:
        data = json.load(f)
    all_losses = data.get("all_losses", [])
    lr_reduction_epochs = data.get("lr_reduction_epochs", [])
    factor = int(config.get("factor", 1))
    n_train = int(config.get("n_train", 100))
    hidden_width = int(config.get("hidden_width", 1024))
    hidden_rank = int(config.get("hidden_rank", 10))
    num_layers = int(config.get("num_layers", 2))

    setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x_plot = np.linspace(-1, 1, 1000)
    y_target = target_baseline(x_plot, factor)

    # (1) function to fit
    ax = axes[0]
    ax.plot(x_plot, y_target, color="C0", linewidth=2, label=r"target $\sum_{k=1}^{%d} \cos(2\pi k x)$" % factor)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Function to fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)

    # (2) prediction if we have saved parameters
    ax = axes[1]
    model_pth = run_dir / "model_parameters.pth"
    if not model_pth.exists():
        params_dirs = sorted(run_dir.glob("params_at_div_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0)
        if params_dirs:
            model_pth = params_dirs[-1] / "model_parameters.pth"
    if model_pth.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ranks = [1] + [hidden_rank] * num_layers + [1]
        widths = [hidden_width] * (num_layers + 1)
        model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
        model.load_state_dict(torch.load(model_pth, map_location=device))
        model.eval()
        x_t = torch.tensor(x_plot.reshape(-1, 1), device=device, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(x_t).cpu().numpy().ravel()
        ax.plot(x_plot, y_target, color="C0", linewidth=2, label="target")
        ax.plot(x_plot, y_pred, color="C1", linestyle="--", linewidth=1.5, label="prediction")
        ax.set_title("Target vs prediction")
    else:
        ax.text(0.5, 0.5, "No saved parameters\n(final model not saved for this run)", ha="center", va="center", transform=ax.transAxes)
        ax.plot(x_plot, y_target, color="C0", linewidth=2, label="target")
        ax.set_title("Target (no prediction)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)

    # (3) loss curve
    ax = axes[2]
    epochs = range(1, len(all_losses) + 1)
    ax.semilogy(epochs, all_losses, color="C2", linewidth=1.5)
    for ep in lr_reduction_epochs:
        ax.axvline(ep, color="red", linestyle="--", alpha=0.8, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (train)")
    ax.set_title("Loss curve")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{run_name}  (factor={factor}, N={n_train}, bs={config.get('batch_size')}, L={num_layers})")
    plt.tight_layout()
    out_path = run_dir / "plot_target_prediction_loss.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out_path)


if __name__ == "__main__":
    main()
