import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_jsonl(metrics_path: Path, out_png: Path) -> None:  # we plot training curves from metrics.jsonl #
    metrics_path = Path(metrics_path)  # we normalize #
    out_png = Path(out_png)  # we normalize #
    rows = []  # we collect rows #
    with open(metrics_path, "r") as f:  # we open #
        for line in f:  # we loop #
            line = line.strip()  # we strip #
            if not line:  # we skip #
                continue  # we continue #
            rows.append(json.loads(line))  # we parse #
    if len(rows) == 0:  # we guard #
        return  # we return #
    epochs = np.array([r["epoch"] for r in rows], dtype=np.int64)  # we build epochs #
    train_loss = np.array([r["train_loss"] for r in rows], dtype=np.float64)  # we store #
    test_loss = np.array([r["test_loss"] for r in rows], dtype=np.float64)  # we store #
    train_rel = np.array([r["train_rel_l2"] for r in rows], dtype=np.float64)  # we store #
    test_rel = np.array([r["test_rel_l2"] for r in rows], dtype=np.float64)  # we store #

    out_png.parent.mkdir(parents=True, exist_ok=True)  # we ensure dir #
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # we create plots #
    axes[0].plot(epochs, train_loss, label="train")  # we plot #
    axes[0].plot(epochs, test_loss, label="test")  # we plot #
    axes[0].set_yscale("log")  # we log scale #
    axes[0].set_xlabel("epoch")  # we label #
    axes[0].set_ylabel("masked mse")  # we label #
    axes[0].set_title("loss")  # we title #
    axes[0].grid(True, alpha=0.3)  # we grid #
    axes[0].legend()  # we legend #

    axes[1].plot(epochs, train_rel, label="train")  # we plot #
    axes[1].plot(epochs, test_rel, label="test")  # we plot #
    axes[1].set_yscale("log")  # we log scale #
    axes[1].set_xlabel("epoch")  # we label #
    axes[1].set_ylabel("relative l2")  # we label #
    axes[1].set_title("relative error")  # we title #
    axes[1].grid(True, alpha=0.3)  # we grid #
    axes[1].legend()  # we legend #

    plt.tight_layout()  # we layout #
    plt.savefig(out_png, dpi=150)  # we save #
    plt.close(fig)  # we close #

