#!/usr/bin/env python3
"""
Plot MNIST: test accuracy vs log10(trainable parameters) for MLP and MMNN.

Uses matplotlib config from meanfield_cosine_multifreq_experiment.
MMNN with fixWb: only width→rank trained; MMNN fixWb=False: all params trained.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# we use same styling as meanfield_cosine_multifreq_experiment (smaller fonts)
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["font.size"] = 11
plt.rcParams["font.weight"] = "normal"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.rm"] = "serif"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 11
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

DIR = Path(__file__).resolve().parent

# we consolidate all runs: (trainable_params, test_acc, label, fixWb)
# R=5, R=10 from earlier run with identical training params
DATA = [
    (7_695, 93.72, "MMNN R=5 fixWb", True),
    (10_260, 96.24, "MMNN R=10 fixWb", True),
    (12_825, 96.97, "MMNN R=15 fixWb", True),
    (17_955, 97.00, "MMNN R=25 fixWb", True),
    (30_780, 97.01, "MMNN R=50 fixWb", True),
    (440_362, 98.30, "MMNN R=32 fixWb=False", False),
]
MLP_TRAINABLE = 669_706
MLP_ACC = 98.39


def main():
    # we split into fixWb (low-rank, random features) and full MMNN
    mmnn_fixwb = [(p, a, l) for (p, a, l, fw) in DATA if fw]
    mmnn_full = [(p, a, l) for (p, a, l, fw) in DATA if not fw]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # we plot MMNN fixWb curve (log10 params vs acc)
    if mmnn_fixwb:
        px = [np.log10(p) for p, _, _ in mmnn_fixwb]
        py = [a for _, a, _ in mmnn_fixwb]
        ax.plot(px, py, "o-", color="#1f77b4", linewidth=2, markersize=8, label="RF-LR (random features)")
        for (p, a, l) in mmnn_fixwb:
            r = l.replace("MMNN R=", "").replace(" fixWb", "")
            ax.annotate(f"R={r}", (np.log10(p), a), textcoords="offset points", xytext=(6, 4), fontsize=9, ha="left")
    # we plot MMNN full (one point: all params trained)
    if mmnn_full:
        p, a, l = mmnn_full[0]
        ax.plot(np.log10(p), a, "s", color="#ff7f0e", markersize=10, zorder=5, label="LR R=32 (low rank only)")
        ax.annotate("R=32", (np.log10(p), a), textcoords="offset points", xytext=(8, 0), fontsize=9, ha="left")
    # we plot MLP reference
    ax.axhline(MLP_ACC, color="#2ca02c", linestyle="--", linewidth=2, label=f"MLP ({MLP_ACC}%, {MLP_TRAINABLE:,})")
    ax.axvline(np.log10(MLP_TRAINABLE), color="#2ca02c", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel(r"$\log_{10}$(trainable parameters)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST: test accuracy vs trainable parameters", fontsize=11)
    ax.set_xlim(3.8, 5.9)
    ax.set_ylim(92, 99.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = DIR / "mnist_perf_vs_params.png"
    plt.savefig(out)
    plt.close()
    print(f"  saved {out}")

    # we save machine-readable data for the plot
    dump = {
        "mmnn_fixwb": [{"trainable": p, "log10_trainable": round(np.log10(p), 4), "test_acc": a, "label": l} for (p, a, l) in mmnn_fixwb],
        "mmnn_full": [{"trainable": p, "log10_trainable": round(np.log10(p), 4), "test_acc": a, "label": l} for (p, a, l) in mmnn_full],
        "mlp": {"trainable": MLP_TRAINABLE, "test_acc": MLP_ACC},
    }
    with open(DIR / "plot_data.json", "w") as f:
        json.dump(dump, f, indent=2)


if __name__ == "__main__":
    main()
