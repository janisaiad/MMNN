#!/usr/bin/env python3
"""
we build gifs from params_epoch_* checkpoints saved during a low-lr-long run (fixed lr, save every 10 epochs).
run after training is stopped or finished.
usage: python make_gif_lowlr_run.py [run_dir]
  run_dir defaults to results_sumcos_selected_rerun_lowlr/f3_N768_bs4_L3 (relative to script dir).
output: run_dir/fit_predict_epochs.gif, run_dir/partial_functions_epochs.gif (after L1), run_dir/partial_functions_L2_epochs.gif (after L2)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
try:
    import imageio
except ImportError:
    imageio = None
import matplotlib as mpl
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from experiments.table.mmnn_vs import MMNN

_BASE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = _BASE / "results_sumcos_selected_rerun_lowlr" / "f3_N768_bs4_L3"
GIF_DPI = 100
GIF_DURATION_SEC = 0.12
GIF_FIGSIZE_FIT = (8, 4)
GIF_FIGSIZE_PARTIAL = (10, 4)


def target_baseline(x, factor=1.0):
    if factor < 1:
        return np.cos(2 * np.pi * x)
    out = np.zeros_like(x, dtype=float)
    for k in range(1, int(factor) + 1):
        out += np.cos(2 * k * np.pi * x)
    return out


def forward_after_first_layer(model, x_tensor):
    x = model.fcs[0](x_tensor)
    x = torch.relu(x)
    x = model.fcs[1](x)
    return x


def forward_after_second_layer(model, x_tensor):
    x = model.fcs[0](x_tensor)
    x = torch.relu(x)
    x = model.fcs[1](x)
    x = model.fcs[2](x)
    x = torch.relu(x)
    x = model.fcs[3](x)
    return x


def load_model_and_predict(config, params_path, device, x_plot):
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
        full = model(x_t).cpu().numpy().ravel()
    return full


def load_model_and_get_partials(config, params_path, device, x_plot, hidden_rank, after_layer=1):
    """after_layer=1 -> activations after 1st layer; after_layer=2 -> after 2nd layer."""
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
        if after_layer == 1:
            after = forward_after_first_layer(model, x_t)
        else:
            after = forward_after_second_layer(model, x_t)
        partials = [after[:, r].cpu().numpy().ravel() for r in range(hidden_rank)]
    return full, partials


def fig_to_rgba(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))
    return buf


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        print("run dir not found:", run_dir)
        return
    run_name = run_dir.name
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # infer config from run name (e.g. f3_N768_bs4_L3 -> factor=3, n_train=768, batch_size=4, num_layers=3)
        m = re.match(r"f(\d+)_N(\d+)_bs(\d+)_L(\d+)$", run_name)
        if not m:
            print("config.json not found and cannot infer config from run name:", run_name)
            return
        factor = int(m.group(1))
        n_train = int(m.group(2))
        batch_size = int(m.group(3))
        num_layers = int(m.group(4))
        config = {
            "factor": factor,
            "n_train": n_train,
            "batch_size": batch_size,
            "num_layers": num_layers,
            "hidden_width": 1024,
            "hidden_rank": 10,
        }
        print("config.json missing; inferred from run name:", config)
    factor = int(config.get("factor", 3))
    hidden_rank = int(config.get("hidden_rank", 10))

    param_dirs = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("params_epoch_"):
            m = re.match(r"params_epoch_(\d+)", d.name)
            if m and (d / "model_parameters.pth").exists():
                param_dirs.append((int(m.group(1)), d))
    param_dirs.sort(key=lambda x: x[0])
    if not param_dirs:
        print("no params_epoch_* dirs with model_parameters.pth in", run_dir)
        return
    print(f"found {len(param_dirs)} checkpoints (epochs {param_dirs[0][0]} to {param_dirs[-1][0]})")

    plt.rcParams["font.size"] = 10
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["font.family"] = "sans-serif"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_plot = np.linspace(-1, 1, 500)
    y_target = target_baseline(x_plot, factor)
    try:
        cmap = mpl.colormaps["tab10"].resampled(hidden_rank)
    except Exception:
        cmap = plt.cm.tab10

    # gif 1: fit/predict (target vs full pred)
    frames_fit = []
    for epoch, pdir in param_dirs:
        pth = pdir / "model_parameters.pth"
        full = load_model_and_predict(config, pth, device, x_plot)
        fig, ax = plt.subplots(figsize=GIF_FIGSIZE_FIT, dpi=GIF_DPI)
        ax.plot(x_plot, y_target, color="black", linewidth=1.5, label="target")
        ax.plot(x_plot, full, color="C0", linewidth=1.2, linestyle="--", label="pred")
        ax.set_xlim(-1, 1)
        ax.set_ylim(np.min(y_target) - 0.5, np.max(y_target) + 0.5)
        ax.set_title(f"{run_name}  epoch {epoch}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$x$")
        frames_fit.append(fig_to_rgba(fig))
        plt.close(fig)
    out_fit = run_dir / "fit_predict_epochs.gif"
    if imageio is None:
        print("imageio not available; install with: pip install imageio")
        return
    imageio.mimsave(str(out_fit), frames_fit, duration=GIF_DURATION_SEC, loop=0)
    print("saved", out_fit)

    # gif 2: partial functions after L1 (10 bottleneck activations + full + target)
    frames_partial = []
    for epoch, pdir in param_dirs:
        pth = pdir / "model_parameters.pth"
        full, partials = load_model_and_get_partials(config, pth, device, x_plot, hidden_rank, after_layer=1)
        fig, ax = plt.subplots(figsize=GIF_FIGSIZE_PARTIAL, dpi=GIF_DPI)
        ax.plot(x_plot, y_target, color="black", linewidth=1.5, label="target")
        ax.plot(x_plot, full, color="C0", linewidth=1.2, linestyle="-", label="full pred")
        for r in range(hidden_rank):
            ax.plot(x_plot, partials[r], color=cmap(r / max(1, hidden_rank - 1)), linewidth=0.7, alpha=0.8, label=f"r{r}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(np.min(y_target) - 0.5, np.max(y_target) + 0.5)
        ax.set_title(f"{run_name}  partial (after L1)  epoch {epoch}")
        ax.legend(loc="upper left", fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$x$")
        frames_partial.append(fig_to_rgba(fig))
        plt.close(fig)
    out_partial = run_dir / "partial_functions_epochs.gif"
    imageio.mimsave(str(out_partial), frames_partial, duration=GIF_DURATION_SEC, loop=0)
    print("saved", out_partial)

    # gif 3: partial functions after L2 (10 bottleneck activations + full + target)
    frames_partial_L2 = []
    for epoch, pdir in param_dirs:
        pth = pdir / "model_parameters.pth"
        full, partials = load_model_and_get_partials(config, pth, device, x_plot, hidden_rank, after_layer=2)
        fig, ax = plt.subplots(figsize=GIF_FIGSIZE_PARTIAL, dpi=GIF_DPI)
        ax.plot(x_plot, y_target, color="black", linewidth=1.5, label="target")
        ax.plot(x_plot, full, color="C0", linewidth=1.2, linestyle="-", label="full pred")
        for r in range(hidden_rank):
            ax.plot(x_plot, partials[r], color=cmap(r / max(1, hidden_rank - 1)), linewidth=0.7, alpha=0.8, label=f"r{r}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(np.min(y_target) - 0.5, np.max(y_target) + 0.5)
        ax.set_title(f"{run_name}  partial (after L2)  epoch {epoch}")
        ax.legend(loc="upper left", fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$x$")
        frames_partial_L2.append(fig_to_rgba(fig))
        plt.close(fig)
    out_partial_L2 = run_dir / "partial_functions_L2_epochs.gif"
    imageio.mimsave(str(out_partial_L2), frames_partial_L2, duration=GIF_DURATION_SEC, loop=0)
    print("saved", out_partial_L2)
    print("done.")


if __name__ == "__main__":
    main()
