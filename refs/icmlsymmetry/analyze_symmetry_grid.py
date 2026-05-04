#!/usr/bin/env python3
"""Post-process symmetry_grid_long outputs with active-channel robust metrics and plots."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "results" / "symmetry_grid_long"
EPS = 1e-12


def active_mask(energy: np.ndarray) -> np.ndarray:
    if energy.size == 0:
        return np.zeros_like(energy, dtype=bool)
    finite = np.isfinite(energy)
    if not np.any(finite):
        return np.zeros_like(energy, dtype=bool)
    positive = energy[finite]
    cutoff = max(float(np.quantile(positive, 0.50)), 1e-8)
    return finite & (energy >= cutoff)


def load_run(run_dir: Path) -> dict[str, object]:
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    data = np.load(run_dir / "distributions.npz")
    layer_ids = sorted(
        int(k.replace("partial_even_layer", ""))
        for k in data.files
        if k.startswith("partial_even_layer")
    )
    active_even_layers: list[float] = []
    active_even_values: list[np.ndarray] = []
    active_minima_layers: list[float] = []
    for layer_id in layer_ids:
        even = np.asarray(data[f"partial_even_layer{layer_id}"], dtype=np.float64)
        energy = np.asarray(data[f"partial_energy_layer{layer_id}"], dtype=np.float64)
        minima = np.asarray(data[f"partial_minima_layer{layer_id}"], dtype=np.float64)
        mask = active_mask(energy)
        if np.any(mask):
            active_even_layers.append(float(np.mean(even[mask])))
            active_even_values.append(even[mask])
            active_minima_layers.append(float(np.mean(minima[mask])))
        else:
            active_even_layers.append(float("nan"))
            active_even_values.append(np.array([], dtype=np.float64))
            active_minima_layers.append(float("nan"))
    all_active_even = np.concatenate(active_even_values) if active_even_values else np.array([], dtype=np.float64)
    mirror_distance = np.asarray(data["mirror_distance"], dtype=np.float64)
    mismatch_same = np.asarray(data["mismatch_same"], dtype=np.float64)
    signed_corr = np.asarray(data["signed_corr"], dtype=np.float64)
    close = mirror_distance <= np.quantile(mirror_distance, 0.20)
    row = {
        **metrics,
        "active_last_layer_even_mean": float(active_even_layers[-1]) if active_even_layers else float("nan"),
        "active_mean_layer_even": float(np.nanmean(active_even_layers)) if active_even_layers else float("nan"),
        "active_mean_layer_minima": float(np.nanmean(active_minima_layers)) if active_minima_layers else float("nan"),
        "active_all_even_median": float(np.median(all_active_even)) if all_active_even.size else float("nan"),
        "active_all_even_p90": float(np.quantile(all_active_even, 0.90)) if all_active_even.size else float("nan"),
        "weight_mismatch_median": float(np.median(mismatch_same)),
        "weight_mismatch_p90": float(np.quantile(mismatch_same, 0.90)),
        "weight_signed_corr_close_mean": float(np.mean(signed_corr[close])),
        "run_dir": str(run_dir),
    }
    return row


def save_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "name", "model_kind", "seed", "width", "rank", "depth", "final_train_mse",
        "output_even_defect", "active_last_layer_even_mean", "active_mean_layer_even",
        "active_all_even_median", "active_all_even_p90", "active_mean_layer_minima",
        "mirror_distance_p20", "mismatch_same_best20", "weight_mismatch_median",
        "weight_mismatch_p90", "weight_signed_corr_close_mean",
    ]
    with open(ROOT / "active_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(ROOT / "active_summary.json", "w") as f:
        json.dump(rows, f, indent=2)


def grouped(rows: list[dict[str, object]], key: str, value: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        label = str(row[key])
        val = float(row[value])
        if np.isfinite(val):
            out.setdefault(label, []).append(val)
    return out


def boxplot_metric(rows: list[dict[str, object]], key: str, value: str, path: Path, title: str, ylabel: str) -> None:
    groups = grouped(rows, key, value)
    labels = sorted(groups)
    values = [groups[label] for label in labels]
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(values, labels=labels, showfliers=True)
    plt.yscale("log")
    plt.ylabel(ylabel)
    plt.xlabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def scatter(rows: list[dict[str, object]], x_key: str, y_key: str, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    for kind, marker in [("mmnn", "o"), ("mlp", "s")]:
        xs = [float(row[x_key]) for row in rows if row["model_kind"] == kind]
        ys = [float(row[y_key]) for row in rows if row["model_kind"] == kind]
        labels = [f"L{row['depth']} r{row['rank']}" for row in rows if row["model_kind"] == kind]
        plt.scatter(xs, ys, label=kind, marker=marker, s=70, alpha=0.8)
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), fontsize=7, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def distribution_overlay(rows: list[dict[str, object]]) -> None:
    plt.figure(figsize=(8, 5))
    for kind in ["mmnn", "mlp"]:
        vals = [float(row["active_last_layer_even_mean"]) for row in rows if row["model_kind"] == kind]
        vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if vals.size:
            plt.hist(np.log10(vals + EPS), bins=24, alpha=0.55, label=kind)
    plt.xlabel("log10 active last-layer partial even defect")
    plt.ylabel("config count")
    plt.title("Active-channel internal symmetry distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "active_last_layer_even_distribution.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5))
    for kind in ["mmnn", "mlp"]:
        vals = [float(row["mismatch_same_best20"]) for row in rows if row["model_kind"] == kind]
        vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if vals.size:
            plt.hist(vals, bins=20, alpha=0.55, label=kind)
    plt.xlabel("close mirror-pair outgoing mismatch")
    plt.ylabel("config count")
    plt.title("Weight-space mirror mismatch distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "active_weight_mismatch_distribution.png", dpi=240)
    plt.close()


def make_layer_heatmap(rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda r: (str(r["model_kind"]), int(r["depth"]), int(r["rank"]), int(r["seed"])))
    labels: list[str] = []
    matrix: list[list[float]] = []
    max_layers = max(len(row.get("layers", [])) for row in ordered)
    for row in ordered:
        values = [float(layer["even_mean"]) for layer in row.get("layers", [])]
        values += [np.nan] * (max_layers - len(values))
        matrix.append(values)
        labels.append(f"{row['model_kind']} L{row['depth']} r{row['rank']} s{row['seed']}")
    arr = np.asarray(matrix, dtype=np.float64)
    plt.figure(figsize=(7, max(5, 0.35 * len(labels))))
    plt.imshow(np.log10(arr + EPS), aspect="auto", cmap="viridis")
    plt.colorbar(label="log10 layer even defect")
    plt.yticks(np.arange(len(labels)), labels, fontsize=7)
    plt.xticks(np.arange(max_layers), [f"layer {i + 1}" for i in range(max_layers)])
    plt.title("Layerwise partial symmetry heatmap")
    plt.tight_layout()
    plt.savefig(ROOT / "layerwise_partial_symmetry_heatmap.png", dpi=240)
    plt.close()


def main() -> None:
    run_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "metrics.json").exists() and (p / "distributions.npz").exists())
    rows = [load_run(path) for path in run_dirs]
    save_csv(rows)
    boxplot_metric(
        rows,
        "model_kind",
        "active_last_layer_even_mean",
        ROOT / "box_active_last_layer_by_model.png",
        "Active last-layer partial symmetry by model",
        "active last-layer even defect",
    )
    boxplot_metric(
        rows,
        "depth",
        "active_last_layer_even_mean",
        ROOT / "box_active_last_layer_by_depth.png",
        "Active last-layer partial symmetry by depth",
        "active last-layer even defect",
    )
    scatter(
        rows,
        "output_even_defect",
        "active_last_layer_even_mean",
        ROOT / "active_output_vs_partial_symmetry.png",
        "Output symmetry can hide internal asymmetry",
    )
    scatter(
        rows,
        "mismatch_same_best20",
        "active_last_layer_even_mean",
        ROOT / "active_weightspace_vs_partial_symmetry.png",
        "Weight-space mirror mismatch vs partial symmetry",
    )
    distribution_overlay(rows)
    make_layer_heatmap(rows)
    print(f"wrote active analysis for {len(rows)} runs -> {ROOT}")


if __name__ == "__main__":
    main()
