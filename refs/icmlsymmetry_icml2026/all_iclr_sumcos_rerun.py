#!/usr/bin/env python3
"""Rerun all available ICLR sumcos configs and analyze symmetry on low-loss checkpoints."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.run_scaling_law_depth_width import (  # noqa: E402
    SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
    SWEEP_LR_STOP_BELOW,
    SWEEP_LR_WINDOW,
    SWEEP_MIN_LOSS_DIVISOR,
    SWEEP_NUM_EPOCHS_MAX,
    SWEEP_WIDTH,
    _sweep_lr_sequence,
    train_baseline_sweep_one,
)
from refs.icmlsymmetry.good_sumcos_low_loss import analyze_run  # noqa: E402


OUT_ROOT = Path(__file__).resolve().parent / "results" / "all_iclr_sumcos_rerun"
CSV_SPECS = [
    (ROOT / "experiments" / "table" / "baseline_sweep_sumcos_rank5_results.csv", 5),
    (ROOT / "experiments" / "table" / "baseline_sweep_sumcos_results.csv", 10),
]


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_configs(include_failed: bool, max_test_error: float | None) -> list[dict]:
    configs: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for csv_path, rank in CSV_SPECS:
        if not csv_path.exists():
            print(f"missing CSV: {csv_path}")
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = str(row["name"]).strip()
                key = (rank, name)
                if key in seen:
                    continue
                worked = boolish(row.get("worked", "False"))
                if not include_failed and not worked:
                    continue
                final_test = float(row.get("final_test_error") or "nan")
                if max_test_error is not None and not (np.isfinite(final_test) and final_test <= max_test_error):
                    continue
                seen.add(key)
                configs.append({
                    "source_csv": str(csv_path),
                    "source_name": name,
                    "name": f"rank{rank}_{name}",
                    "factor": int(row["factor"]),
                    "n_train": int(row["N"]),
                    "batch_size": int(row["bs"]),
                    "num_layers": int(row["L"]),
                    "hidden_rank": rank,
                    "source_final_test_error": final_test,
                    "source_min_loss": float(row.get("min_loss") or "nan"),
                    "source_worked": worked,
                })
    configs.sort(key=lambda c: (c["hidden_rank"], c["factor"], c["n_train"], c["batch_size"], c["num_layers"]))
    return configs


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


def summary_paths() -> tuple[Path, Path]:
    return OUT_ROOT / "summary.json", OUT_ROOT / "summary.csv"


def load_existing_summary() -> list[dict]:
    json_path, _ = summary_paths()
    if not json_path.exists():
        return []
    with open(json_path) as f:
        return json.load(f)


def write_summary(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = summary_paths()
    rows = sorted(rows, key=lambda r: (int(r.get("rank", 0)), int(r.get("factor", 0)), float(r.get("final_test_error", np.inf)), str(r.get("name", ""))))
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    fields = [
        "name", "factor", "rank", "N", "bs", "L", "final_train_error", "final_test_error", "min_loss",
        "epochs_run", "output_even_defect", "last_layer_active_even_mean", "mean_layer_active_even",
        "mean_minima", "mirror_distance_p20", "mirror_mismatch_best20", "mirror_corr_best20",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        plot_summary(rows)


def plot_summary(rows: list[dict]) -> None:
    good_rows = [
        row for row in rows
        if row.get("final_test_error") is not None
        and row.get("last_layer_active_even_mean") is not None
        and np.isfinite(float(row["final_test_error"]))
        and np.isfinite(float(row["last_layer_active_even_mean"]))
    ]
    if not good_rows:
        return
    plt.figure(figsize=(8, 5))
    for rank, marker in [(5, "o"), (10, "s"), (20, "^")]:
        sub = [row for row in good_rows if int(row["rank"]) == rank]
        if not sub:
            continue
        plt.scatter(
            [float(row["final_test_error"]) for row in sub],
            [float(row["last_layer_active_even_mean"]) for row in sub],
            label=f"rank {rank}",
            marker=marker,
            s=55,
            alpha=0.8,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("final test MSE")
    plt.ylabel("last-layer active partial even defect")
    plt.title("All rerun ICLR worked configs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "all_test_error_vs_partial_symmetry.png", dpi=240)
    plt.close()
    plt.figure(figsize=(8, 5))
    for rank, marker in [(5, "o"), (10, "s"), (20, "^")]:
        sub = [row for row in good_rows if int(row["rank"]) == rank]
        if not sub:
            continue
        plt.scatter(
            [int(row["factor"]) for row in sub],
            [float(row["last_layer_active_even_mean"]) for row in sub],
            label=f"rank {rank}",
            marker=marker,
            s=55,
            alpha=0.8,
        )
    plt.yscale("log")
    plt.xlabel("sumcos factor")
    plt.ylabel("last-layer active partial even defect")
    plt.title("Internal symmetry across frequencies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "all_factor_vs_partial_symmetry.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=SWEEP_NUM_EPOCHS_MAX)
    parser.add_argument("--include-failed", action="store_true", help="Also rerun failed CSV rows. This can be very long.")
    parser.add_argument("--max-test-error", type=float, default=None, help="Optional filter on source final_test_error.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = load_configs(include_failed=args.include_failed, max_test_error=args.max_test_error)
    if args.max_runs is not None:
        configs = configs[: args.max_runs]
    print(f"selected {len(configs)} configs -> {OUT_ROOT}", flush=True)
    rows_by_name = {str(row["name"]): row for row in load_existing_summary()}
    for index, base in enumerate(configs, start=1):
        out_dir = OUT_ROOT / base["name"]
        print(f"[{index}/{len(configs)}] {base['name']} source_test={base['source_final_test_error']:.4e}", flush=True)
        if not args.analyze_only:
            if args.overwrite or not (out_dir / "model_parameters.pth").exists():
                train_baseline_sweep_one(make_train_config(base, args.epochs), out_dir)
            else:
                print("  skip train: checkpoint exists", flush=True)
        if (out_dir / "model_parameters.pth").exists():
            row = analyze_run(out_dir)
            row["source_final_test_error"] = base["source_final_test_error"]
            row["source_min_loss"] = base["source_min_loss"]
            rows_by_name[str(row["name"])] = row
            write_summary(list(rows_by_name.values()))
        else:
            print("  no checkpoint, skip analysis", flush=True)
    write_summary(list(rows_by_name.values()))
    print(f"done -> {OUT_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
