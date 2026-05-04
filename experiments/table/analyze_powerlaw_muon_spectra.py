#!/usr/bin/env python3
"""Analyze whether HT-Muon runs exhibit heavy-tail spectral coefficients."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table.mmnn_vs import MMNN


EPS = 1e-12


def fit_powerlaw(singulars: np.ndarray, top_k: int) -> dict[str, float]:
    values = singulars[: min(top_k, singulars.size)]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 3:
        return {
            "alpha_hat": float("nan"),
            "r2": float("nan"),
            "top_singular": float(values[0]) if values.size else float("nan"),
            "tail_ratio": float("nan"),
        }
    x = np.log(np.arange(1, values.size + 1, dtype=float))
    y = np.log(values + EPS)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "alpha_hat": float(-slope),
        "r2": float(1.0 - ss_res / (ss_tot + EPS)),
        "top_singular": float(values[0]),
        "tail_ratio": float(values[-1] / (values[0] + EPS)),
    }


def build_model(config: dict[str, object], device: torch.device) -> torch.nn.Module:
    model_name = str(config["model"])
    if model_name != "mmnn":
        raise ValueError(f"only mmnn checkpoints are supported, got {model_name}")
    dim = int(config["dim"])
    rank = int(config["rank"])
    depth = int(config["depth"])
    width = int(config["width"])
    fix_wb = bool(config["fix_wb"])
    ranks = [dim] + [rank] * depth + [1]
    widths = [width] * (depth + 1)
    return MMNN(ranks=ranks, widths=widths, device=str(device), ResNet=False, fixWb=fix_wb).to(device)


def analyze_run(run_dir: Path, top_k: int, device: torch.device) -> list[dict[str, object]]:
    config_path = run_dir / "config.json"
    model_path = run_dir / "model_parameters.pth"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists() or not model_path.exists() or not metrics_path.exists():
        return []
    with open(config_path) as f:
        config = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    model = build_model(config, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or param.ndim != 2:
                continue
            singulars = torch.linalg.svdvals(param.detach().float()).cpu().numpy()
            fit = fit_powerlaw(singulars, top_k)
            target_alpha = float(config.get("ht_alpha", float("nan")))
            rows.append({
                "run": run_dir.name,
                "optimizer": config.get("optimizer"),
                "target": config.get("target"),
                "freq": config.get("freq"),
                "depth": config.get("depth"),
                "rank": config.get("rank"),
                "fix_wb": config.get("fix_wb"),
                "layer": name,
                "top_k": min(top_k, singulars.size),
                "target_alpha": target_alpha,
                "alpha_hat": fit["alpha_hat"],
                "alpha_error": abs(fit["alpha_hat"] - target_alpha) if math.isfinite(fit["alpha_hat"]) and math.isfinite(target_alpha) else float("nan"),
                "r2": fit["r2"],
                "top_singular": fit["top_singular"],
                "tail_ratio": fit["tail_ratio"],
                "test_mse": metrics.get("test_mse"),
                "final_train_mse": metrics.get("final_train_mse"),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze heavy-tail spectra in HT-Muon checkpoints")
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("experiments/table/powerlaw_muon_spectra_summary.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for results_dir in args.results_dirs:
        for run_dir in sorted(results_dir.iterdir()) if results_dir.exists() else []:
            if run_dir.is_dir():
                rows.extend(analyze_run(run_dir, args.top_k, device))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "optimizer",
        "target",
        "freq",
        "depth",
        "rank",
        "fix_wb",
        "layer",
        "top_k",
        "target_alpha",
        "alpha_hat",
        "alpha_error",
        "r2",
        "top_singular",
        "tail_ratio",
        "test_mse",
        "final_train_mse",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} layer spectra)")


if __name__ == "__main__":
    main()
