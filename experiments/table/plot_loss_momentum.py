#!/usr/bin/env python3
"""
Plot L2 (MSE) training losses for SGD momentum 0.3--0.7 on the same axes.
Uses factor4_rank10 runs (n=1024, r=10) with all_losses from results.json.
Saves to refs/icml_sgdadamlandscapedynamical/figures/loss_training_curve_momentum.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# we set paths relative to project root
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/table/experiments/table/results_tune_lr_decay_L2"
OUT_DIR = ROOT / "refs/icml_sgdadamlandscapedynamical/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# available in [0.3,0.7]: factor4_rank10 mom 0.3,0.7; factor1_rank10 mom 0.5
CONFIGS = [
    (RESULTS / "factor4_rank10_SGD_mom0.3_lr0.01_AdaptiveStagnation/results.json", 0.3),
    (RESULTS / "factor1_rank10_SGD_mom0.5_lr0.001_NoScheduler/results.json", 0.5),
    (RESULTS / "factor4_rank10_SGD_mom0.7_lr0.01_AdaptiveStagnation/results.json", 0.7),
]

def main():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for p, mom in CONFIGS:
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        loss = np.array(d["all_losses"], dtype=float)
        # we truncate at first NaN for plotting
        valid = ~(np.isnan(loss) | np.isinf(loss))
        if not np.all(valid):
            idx = np.where(~valid)[0]
            if len(idx):
                loss = loss[: idx[0]]
        if len(loss) == 0:
            continue
        ax.semilogy(loss, label=f"mom {mom}", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "loss_training_curve_momentum.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", out)

if __name__ == "__main__":
    main()
