#!/usr/bin/env python3
"""Rerun and analyze the low-loss sumcos configs cited in the ICLR merged draft."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.mmnn_vs import MMNN  # noqa: E402
from experiments.table.run_scaling_law_depth_width import (  # noqa: E402
    SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
    SWEEP_LR_STOP_BELOW,
    SWEEP_LR_WINDOW,
    SWEEP_MIN_LOSS_DIVISOR,
    SWEEP_NUM_EPOCHS_MAX,
    SWEEP_WIDTH,
    _sweep_lr_sequence,
    target_baseline,
    train_baseline_sweep_one,
)


OUT_ROOT = Path(__file__).resolve().parent / "results" / "good_sumcos_low_loss"
EPS = 1e-12


GOOD_CONFIGS = [
    {"name": "rank5_f3_N768_bs8_L3", "factor": 3, "n_train": 768, "batch_size": 8, "num_layers": 3, "hidden_rank": 5},
    {"name": "rank5_f4_N1024_bs4_L3", "factor": 4, "n_train": 1024, "batch_size": 4, "num_layers": 3, "hidden_rank": 5},
    {"name": "rank5_f5_N1280_bs4_L3", "factor": 5, "n_train": 1280, "batch_size": 4, "num_layers": 3, "hidden_rank": 5},
    {"name": "rank10_f2_N512_bs2_L3", "factor": 2, "n_train": 512, "batch_size": 2, "num_layers": 3, "hidden_rank": 10},
    {"name": "rank10_f3_N768_bs4_L3", "factor": 3, "n_train": 768, "batch_size": 4, "num_layers": 3, "hidden_rank": 10},
    {"name": "rank10_f3_N768_bs8_L3", "factor": 3, "n_train": 768, "batch_size": 8, "num_layers": 3, "hidden_rank": 10},
]


def make_train_config(base: dict, epochs: int) -> dict:
    return {
        "name": base["name"],
        "n_train": base["n_train"],
        "batch_size": base["batch_size"],
        "hidden_width": SWEEP_WIDTH,
        "hidden_rank": base["hidden_rank"],
        "num_layers": base["num_layers"],
        "num_epochs": epochs,
        "lr_sequence": _sweep_lr_sequence(),
        "lr_window": SWEEP_LR_WINDOW,
        "min_epochs_before_reduce": SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
        "min_loss_divisor": SWEEP_MIN_LOSS_DIVISOR,
        "lr_stop_below": SWEEP_LR_STOP_BELOW,
        "momentum": 0.0,
        "factor": base["factor"],
        "seed": 42,
    }


def load_model(config: dict, path: Path, device: torch.device) -> MMNN:
    rank = int(config["hidden_rank"])
    layers = int(config["num_layers"])
    width = int(config["hidden_width"])
    ranks = [1] + [rank] * layers + [1]
    widths = [width] * (layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def collect_partials(model: MMNN, x: torch.Tensor) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    z = x
    depth = int(getattr(model, "depth"))
    for j in range(depth - 1):
        z = model.fcs[2 * j](z)
        z = torch.relu(z)
        z = model.fcs[2 * j + 1](z)
        values.append(z)
    return values


def strict_positive_minima(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 3:
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.sum((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]) & (values[1:-1] > 1e-4), axis=0).astype(np.float64)


def analyze_run(run_dir: Path) -> dict[str, object]:
    with open(run_dir / "losses.json") as f:
        losses = json.load(f)
    config = losses["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, run_dir / "model_parameters.pth", device)
    x_pos = torch.linspace(0.02, 0.98, 768, device=device, dtype=torch.float32).reshape(-1, 1)
    x_all = torch.linspace(-1.0, 1.0, 1536, device=device, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        y_pos = model(x_pos)
        y_neg = model(-x_pos)
        out_even = float((torch.mean((y_pos - y_neg) ** 2) / (torch.mean(y_pos ** 2) + EPS)).item())
        p_pos = collect_partials(model, x_pos)
        p_neg = collect_partials(model, -x_pos)
        p_all = collect_partials(model, x_all)
    layer_rows: list[dict[str, float | int]] = []
    all_active: list[np.ndarray] = []
    for idx, (a, b, full) in enumerate(zip(p_pos, p_neg, p_all), start=1):
        energy = torch.mean(a ** 2, dim=0).detach().cpu().numpy()
        defect = (torch.mean((a - b) ** 2, dim=0) / (torch.mean(a ** 2, dim=0) + EPS)).detach().cpu().numpy()
        mask = energy >= max(float(np.quantile(energy, 0.50)), 1e-8)
        active = defect[mask] if np.any(mask) else defect
        minima = strict_positive_minima(full.detach().cpu().numpy())
        all_active.append(active)
        layer_rows.append({
            "layer": idx,
            "active_even_mean": float(np.mean(active)),
            "active_even_median": float(np.median(active)),
            "active_even_p90": float(np.quantile(active, 0.90)),
            "minima_mean": float(np.mean(minima)),
            "minima_p90": float(np.quantile(minima, 0.90)),
        })
    mirror = mirror_stats(model)
    plot_run(run_dir, losses, layer_rows, all_active)
    return {
        "name": config["name"],
        "factor": int(config["factor"]),
        "rank": int(config["hidden_rank"]),
        "N": int(config["n_train"]),
        "bs": int(config["batch_size"]),
        "L": int(config["num_layers"]),
        "final_train_error": losses.get("final_train_error"),
        "final_test_error": losses.get("final_test_error"),
        "min_loss": float(np.min(losses.get("all_losses", [np.nan]))),
        "epochs_run": losses.get("epochs_run"),
        "output_even_defect": out_even,
        "last_layer_active_even_mean": layer_rows[-1]["active_even_mean"] if layer_rows else None,
        "mean_layer_active_even": float(np.mean([row["active_even_mean"] for row in layer_rows])) if layer_rows else None,
        "mean_minima": float(np.mean([row["minima_mean"] for row in layer_rows])) if layer_rows else None,
        "layers": layer_rows,
        **mirror,
    }


def mirror_stats(model: MMNN) -> dict[str, float]:
    first = model.fcs[0]
    next_weight = model.fcs[1].weight.detach().cpu().numpy()
    slopes = first.weight.detach().cpu().numpy().reshape(-1)
    biases = first.bias.detach().cpu().numpy().reshape(-1)
    distances = []
    mismatch = []
    corr = []
    for j in range(slopes.shape[0]):
        d = (slopes + slopes[j]) ** 2 + (biases - biases[j]) ** 2
        d[j] = np.inf
        k = int(np.argmin(d))
        cj = next_weight[:, j]
        ck = next_weight[:, k]
        denom = float(np.mean(cj ** 2 + ck ** 2) + EPS)
        distances.append(float(np.sqrt(d[k])))
        mismatch.append(float(np.mean((cj - ck) ** 2) / denom))
        corr.append(float(np.dot(cj, ck) / (np.linalg.norm(cj) * np.linalg.norm(ck) + EPS)))
    distances_np = np.asarray(distances)
    mismatch_np = np.asarray(mismatch)
    corr_np = np.asarray(corr)
    close = distances_np <= np.quantile(distances_np, 0.20)
    return {
        "mirror_distance_p20": float(np.quantile(distances_np, 0.20)),
        "mirror_mismatch_best20": float(np.mean(mismatch_np[close])),
        "mirror_corr_best20": float(np.mean(corr_np[close])),
    }


def plot_run(run_dir: Path, losses: dict, layer_rows: list[dict[str, float | int]], all_active: list[np.ndarray]) -> None:
    vals = losses.get("all_losses", [])
    plt.figure(figsize=(7, 4))
    plt.semilogy(np.arange(1, len(vals) + 1), vals)
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title(losses["config"]["name"])
    plt.tight_layout()
    plt.savefig(run_dir / "low_loss_curve.png", dpi=220)
    plt.close()
    layers = [int(row["layer"]) for row in layer_rows]
    defects = [float(row["active_even_mean"]) for row in layer_rows]
    minima = [float(row["minima_mean"]) for row in layer_rows]
    plt.figure(figsize=(7, 4))
    plt.plot(layers, defects, marker="o", label="active even defect")
    plt.yscale("log")
    plt.xlabel("partial layer")
    plt.ylabel("defect")
    plt.title("Low-loss partial symmetry")
    plt.tight_layout()
    plt.savefig(run_dir / "low_loss_layerwise_symmetry.png", dpi=220)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot(layers, minima, marker="o")
    plt.xlabel("partial layer")
    plt.ylabel("mean minima")
    plt.title("Low-loss oscillatory complexity")
    plt.tight_layout()
    plt.savefig(run_dir / "low_loss_layerwise_minima.png", dpi=220)
    plt.close()
    if all_active:
        x = np.concatenate(all_active)
        plt.figure(figsize=(7, 4))
        plt.hist(np.log10(x + EPS), bins=32)
        plt.xlabel("log10 active partial even defect")
        plt.ylabel("count")
        plt.title("Active partial symmetry distribution")
        plt.tight_layout()
        plt.savefig(run_dir / "low_loss_partial_defect_distribution.png", dpi=220)
        plt.close()


def write_summary(rows: list[dict[str, object]]) -> None:
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    fields = [
        "name", "factor", "rank", "N", "bs", "L", "final_train_error", "final_test_error", "min_loss",
        "epochs_run", "output_even_defect", "last_layer_active_even_mean", "mean_layer_active_even",
        "mean_minima", "mirror_distance_p20", "mirror_mismatch_best20", "mirror_corr_best20",
    ]
    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plt.figure(figsize=(7, 5))
    for row in rows:
        plt.scatter(float(row["final_test_error"]), float(row["last_layer_active_even_mean"]), s=70)
        plt.annotate(str(row["name"]).replace("rank", "r"), (float(row["final_test_error"]), float(row["last_layer_active_even_mean"])), fontsize=7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("final test MSE")
    plt.ylabel("last-layer active partial even defect")
    plt.title("Low-loss ICLR configs: loss vs internal symmetry")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "low_loss_test_error_vs_partial_symmetry.png", dpi=240)
    plt.close()
    plt.figure(figsize=(7, 5))
    ranks = sorted({int(row["rank"]) for row in rows})
    for rank in ranks:
        vals = [float(row["last_layer_active_even_mean"]) for row in rows if int(row["rank"]) == rank]
        plt.scatter([rank] * len(vals), vals, s=70, label=f"rank {rank}")
    plt.yscale("log")
    plt.xlabel("rank")
    plt.ylabel("last-layer active even defect")
    plt.title("Low-loss configs by rank")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "low_loss_rank_vs_partial_symmetry.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=SWEEP_NUM_EPOCHS_MAX)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for base in GOOD_CONFIGS:
        out_dir = OUT_ROOT / base["name"]
        if not args.analyze_only:
            if args.overwrite or not (out_dir / "model_parameters.pth").exists():
                cfg = make_train_config(base, args.epochs)
                print(f"training {base['name']}", flush=True)
                train_baseline_sweep_one(cfg, out_dir)
            else:
                print(f"skip train (checkpoint exists): {base['name']}", flush=True)
        if (out_dir / "model_parameters.pth").exists():
            print(f"analyzing {base['name']}", flush=True)
            rows.append(analyze_run(out_dir))
    write_summary(rows)
    print(f"done -> {OUT_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
