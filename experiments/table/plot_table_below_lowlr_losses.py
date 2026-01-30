#!/usr/bin/env python3
"""
we plot loss curves for all completed runs in results_sumcos_lowlr_table_below_1e2
(or another dir). For each run dir that has losses.json we save run_dir/loss_curve.png
(x = gradient steps, y = loss; comparable across batch sizes).
usage: python plot_table_below_lowlr_losses.py [results_dir]
  run after training or just before sleeping to plot whatever has finished.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

_BASE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = _BASE / "results_sumcos_lowlr_table_below_1e2"


def main(results_dir=None):
    """plot loss curves for all runs in results_dir that have losses.json; results_dir defaults to DEFAULT_RESULTS_DIR or argv[1]."""
    if results_dir is None:
        results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    results_dir = Path(results_dir).resolve()
    if not results_dir.is_dir():
        print("results dir not found:", results_dir)
        return

    # LaTeX-style formatting (mathtext, serif, math-formatted axis labels)
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True

    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    n_plotted = 0
    for run_dir in run_dirs:
        losses_path = run_dir / "losses.json"
        if not losses_path.exists():
            continue
        try:
            with open(losses_path) as f:
                data = json.load(f)
        except Exception as e:
            print(run_dir.name, "load error:", e)
            continue
        all_losses = data.get("all_losses")
        if not all_losses:
            continue
        config = data.get("config") or {}
        n_train = int(config.get("n_train", 0))
        batch_size = int(config.get("batch_size", 1))
        num_epochs_planned = int(config.get("num_epochs", 0)) or len(all_losses)
        steps_per_epoch = (n_train + batch_size - 1) // batch_size
        # x = cumulative steps at end of each epoch (comparable across bs)
        steps = [(i + 1) * steps_per_epoch for i in range(len(all_losses))]
        steps = np.array(steps, dtype=float)
        losses = np.array(all_losses, dtype=float)
        expected_total_steps = num_epochs_planned * steps_per_epoch

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.plot(steps, losses, color="C0", linewidth=0.9)
        ax.set_xlim(0, expected_total_steps)
        ax.set_xlabel("gradient steps")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        epochs_run = len(all_losses)
        title = run_dir.name if epochs_run >= num_epochs_planned else f"{run_dir.name} ({epochs_run}/{num_epochs_planned} ep)"
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        out_path = run_dir / "loss_curve.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_plotted += 1
    print("plotted", n_plotted, "loss curves in", results_dir)


if __name__ == "__main__":
    main()
