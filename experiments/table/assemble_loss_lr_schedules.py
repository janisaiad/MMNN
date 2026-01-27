#!/usr/bin/env python3
"""
Assemble 5 loss_evolution.png (factor1 rank10, different LR scheduling) into one 2x3 grid.
Each panel labelled with mom, lr, scheduler. Saves to refs/icml_sgdadamlandscapedynamical/figures/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/table/experiments/table/results_tune_lr_decay_L2"
OUT = ROOT / "refs/icml_sgdadamlandscapedynamical/figures"
OUT.mkdir(parents=True, exist_ok=True)

# (relative path under BASE, short label for LR scheduling)
PANELS = [
    ("factor1_rank10_SGD_mom0.3_lr0.005_NoScheduler/loss_evolution.png", "mom 0.3, lr 0.005, NoScheduler"),
    ("factor1_rank10_SGD_mom0.4_lr0.01_AdaptiveStagnation/loss_evolution.png", "mom 0.4, lr 0.01, AdaptiveStagnation"),
    ("factor1_rank10_SGD_mom0.4_lr0.001_NoScheduler/loss_evolution.png", "mom 0.4, lr 0.001, NoScheduler"),
    ("factor1_rank10_SGD_mom0.4_lr0.005_NoScheduler/loss_evolution.png", "mom 0.4, lr 0.005, NoScheduler"),
    ("factor1_rank10_SGD_mom0.5_lr0.01_AdaptiveStagnation/loss_evolution.png", "mom 0.5, lr 0.01, AdaptiveStagnation"),
]

def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for i, (rel, label) in enumerate(PANELS):
        p = BASE / rel
        if not p.exists():
            axes.flat[i].text(0.5, 0.5, f"missing:\n{p.name}", ha="center", va="center", fontsize=10)
            axes.flat[i].set_xlim(0, 1)
            axes.flat[i].set_ylim(0, 1)
            axes.flat[i].axis("off")
            continue
        img = plt.imread(p)
        axes.flat[i].imshow(img)
        axes.flat[i].set_title(label, fontsize=9)
        axes.flat[i].axis("off")
    axes.flat[5].axis("off")
    fig.suptitle("L2 loss evolution, factor1 rank10 SGD, different LR scheduling", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "loss_evolution_lr_schedules_2x3.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved:", out)

if __name__ == "__main__":
    main()
