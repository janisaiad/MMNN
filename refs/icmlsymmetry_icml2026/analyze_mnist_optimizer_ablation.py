#!/usr/bin/env python3
"""Merge MNIST SGD/Adam ablations and plot optimizer comparisons."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "results" / "mnist_batch_symmetry"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.glob("*/metrics.json")):
        with open(path) as f:
            row = json.load(f)
        if "optimizer" not in row:
            row["optimizer"] = "sgd"
        row["run_dir"] = str(path.parent)
        rows.append(row)
    return rows


def save_merged(rows: list[dict[str, object]]) -> None:
    fields = [
        "name", "model_kind", "optimizer", "seed", "effective_batch_size", "full_batch",
        "width", "rank", "depth", "epochs", "train_subset", "final_train_loss",
        "final_test_acc", "final_test_loss", "ntrainable", "logit_defect_mean",
        "partial_defect_mean", "prediction_consistency_mean",
        "first_weight_self_symmetry_mean", "first_weight_pair_symmetry_mean",
    ]
    with open(ROOT / "merged_optimizer_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(ROOT / "merged_optimizer_summary.json", "w") as f:
        json.dump(rows, f, indent=2)


def select_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if int(row.get("epochs", 0)) != 20:
            continue
        if int(row.get("width", 0)) != 128:
            continue
        if int(row.get("train_subset", 0)) != 2000:
            continue
        selected.append(row)
    return selected


def plot_accuracy_loss(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(9, 5))
    markers = {("mlp", "sgd"): "o", ("mlp", "adam"): "s", ("mmnn", "sgd"): "^", ("mmnn", "adam"): "D"}
    for (kind, opt), marker in markers.items():
        sub = [r for r in rows if r["model_kind"] == kind and r["optimizer"] == opt]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: (int(r["effective_batch_size"]), int(r["rank"])))
        xs = [float(r["effective_batch_size"]) for r in sub]
        ys = [float(r["final_test_acc"]) for r in sub]
        labels = [f"r{r['rank']}" for r in sub]
        plt.scatter(xs, ys, label=f"{kind}-{opt}", marker=marker, s=70, alpha=0.85)
        for x, y, label in zip(xs, ys, labels):
            if kind == "mmnn":
                plt.annotate(label, (x, y), fontsize=7, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("batch size")
    plt.ylabel("test accuracy")
    plt.title("MNIST: SGD vs Adam on large-batch regimes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "optimizer_accuracy_vs_batch.png", dpi=240)
    plt.close()
    plt.figure(figsize=(9, 5))
    for (kind, opt), marker in markers.items():
        sub = [r for r in rows if r["model_kind"] == kind and r["optimizer"] == opt]
        if not sub:
            continue
        xs = [float(r["effective_batch_size"]) for r in sub]
        ys = [float(r["final_test_loss"]) for r in sub]
        plt.scatter(xs, ys, label=f"{kind}-{opt}", marker=marker, s=70, alpha=0.85)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("batch size")
    plt.ylabel("test CE loss")
    plt.title("MNIST: final loss, SGD vs Adam")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "optimizer_loss_vs_batch.png", dpi=240)
    plt.close()


def plot_symmetry(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(8, 5))
    for opt, marker in [("sgd", "o"), ("adam", "s")]:
        sub = [r for r in rows if r["optimizer"] == opt]
        xs = [float(r["final_test_loss"]) for r in sub]
        ys = [float(r["partial_defect_mean"]) for r in sub]
        colors = ["#1f77b4" if r["model_kind"] == "mlp" else "#ff7f0e" for r in sub]
        plt.scatter(xs, ys, label=opt, marker=marker, s=70, c=colors, alpha=0.85)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("test CE loss")
    plt.ylabel("mean partial transform defect")
    plt.title("Do not trust symmetry metrics when loss is high")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "optimizer_loss_vs_partial_defect.png", dpi=240)
    plt.close()


def print_key_comparisons(rows: list[dict[str, object]]) -> None:
    key = {}
    for row in rows:
        label = (row["model_kind"], row["optimizer"], int(row["effective_batch_size"]), int(row["rank"]))
        key[label] = row
    comparisons = [
        ("MLP batch512", ("mlp", "sgd", 512, 128), ("mlp", "adam", 512, 128)),
        ("MLP full", ("mlp", "sgd", 2000, 128), ("mlp", "adam", 2000, 128)),
        ("MMNN r10 batch512", ("mmnn", "sgd", 512, 10), ("mmnn", "adam", 512, 10)),
        ("MMNN r25 batch512", ("mmnn", "sgd", 512, 25), ("mmnn", "adam", 512, 25)),
        ("MMNN r10 full", ("mmnn", "sgd", 2000, 10), ("mmnn", "adam", 2000, 10)),
        ("MMNN r25 full", ("mmnn", "sgd", 2000, 25), ("mmnn", "adam", 2000, 25)),
    ]
    for name, sgd_key, adam_key in comparisons:
        sgd = key.get(sgd_key)
        adam = key.get(adam_key)
        if not sgd or not adam:
            continue
        print(
            f"{name}: SGD acc={sgd['final_test_acc']:.3f} loss={sgd['final_test_loss']:.3f} -> "
            f"Adam acc={adam['final_test_acc']:.3f} loss={adam['final_test_loss']:.3f}"
        )


def main() -> None:
    rows = select_rows(load_rows())
    save_merged(rows)
    plot_accuracy_loss(rows)
    plot_symmetry(rows)
    print_key_comparisons(rows)
    print(f"wrote merged optimizer analysis for {len(rows)} runs -> {ROOT}")


if __name__ == "__main__":
    main()
