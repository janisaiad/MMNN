#!/usr/bin/env python3
"""
Train MMNN on an *even* 1D target (sum of cosines), then measure output asymmetry:
  D_sym = E_x[(f(x) - f(-x))^2]  on a grid of x>0.

Fills the rebuttal table: rank r × seed → test MSE, D_sym, relative D_sym / E[f(x)^2].

Usage:
  python experiments/posticml/symmetry_rank_sweep.py --quick
  python experiments/posticml/symmetry_rank_sweep.py --seeds 0 1 2 3 4 --ranks 10 20
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from experiments.table.mmnn_vs import MMNN  # noqa: E402
from experiments.table.run_scaling_law_depth_width import (  # noqa: E402
    SWEEP_LR_STOP_BELOW,
    SWEEP_MIN_LOSS_DIVISOR,
    _sweep_lr_sequence,
    train_baseline_sweep_one,
)

OUT_ROOT = Path(__file__).resolve().parent / "results" / "symmetry_rank_sweep"
M_FIXED = 1024


def measure_symmetry_defect(
    model: torch.nn.Module,
    device: torch.device,
    *,
    n_grid: int = 1024,
    x_min: float = 0.02,
    x_max: float = 0.98,
) -> dict:
    """Even-target symmetry: f(x) should equal f(-x)."""
    mydtype = torch.float32
    x = torch.linspace(x_min, x_max, n_grid, device=device, dtype=mydtype).reshape(-1, 1)
    xm = -x
    model.eval()
    with torch.no_grad():
        fp = model(x)
        fn = model(xm)
        diff2 = (fp - fn) ** 2
        sym_mse = float(diff2.mean().item())
        energy = float((fp ** 2).mean().item()) + 1e-12
        rel = sym_mse / energy
    return {
        "symmetry_mse": sym_mse,
        "relative_symmetry_mse": rel,
        "output_energy_mean": float(energy),
    }


def load_model_checkpoint(
    *,
    hidden_rank: int,
    num_layers: int,
    ckpt_path: Path,
    device: torch.device,
) -> MMNN:
    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [M_FIXED] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="2 seeds, 2 ranks, ~600 epochs")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--ranks", type=int, nargs="*", default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--n-train", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.quick:
        seeds = args.seeds or [100, 101]
        ranks = args.ranks or [10, 20]
        num_epochs = args.num_epochs or 900
    else:
        seeds = args.seeds or [42, 43, 44, 45, 46]
        ranks = args.ranks or [10, 20]
        num_epochs = args.num_epochs or 5000

    lr_seq = _sweep_lr_sequence()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    table_csv = OUT_ROOT / "symmetry_table.csv"
    summary_json = OUT_ROOT / "symmetry_summary.json"

    for seed in seeds:
        for r in ranks:
            name = f"f{args.factor}_N{args.n_train}_bs{args.batch_size}_L{args.num_layers}_M{M_FIXED}_r{r}_seed{seed}"
            out_dir = OUT_ROOT / name
            if args.overwrite and out_dir.exists():
                shutil.rmtree(out_dir)

            cfg = {
                "name": name,
                "n_train": args.n_train,
                "batch_size": args.batch_size,
                "hidden_width": M_FIXED,
                "hidden_rank": r,
                "num_layers": args.num_layers,
                "num_epochs": num_epochs,
                "lr_sequence": lr_seq,
                "lr_window": 10,
                "min_epochs_before_reduce": 20,
                "min_loss_divisor": SWEEP_MIN_LOSS_DIVISOR,
                "lr_stop_below": SWEEP_LR_STOP_BELOW,
                "momentum": 0.0,
                "factor": args.factor,
                "seed": seed,
            }

            if (out_dir / "symmetry_metrics.json").exists():
                print(f"skip (done): {name}")
                with open(out_dir / "symmetry_metrics.json") as f:
                    m = json.load(f)
                rows.append(m)
                continue

            print(f"train: {name}")
            payload = train_baseline_sweep_one(cfg, out_dir)
            ckpt = out_dir / "model_parameters.pth"
            model = load_model_checkpoint(
                hidden_rank=r,
                num_layers=args.num_layers,
                ckpt_path=ckpt,
                device=device,
            )
            sym = measure_symmetry_defect(model, device)

            row = {
                "name": name,
                "hidden_rank": r,
                "seed": seed,
                "final_test_mse": payload.get("final_test_error"),
                "epochs_run": payload.get("epochs_run"),
                **sym,
            }
            with open(out_dir / "symmetry_metrics.json", "w") as f:
                json.dump(row, f, indent=2)
            rows.append(row)
            print(
                f"  test_mse={row['final_test_mse']:.4e} "
                f"sym_mse={row['symmetry_mse']:.4e} rel={row['relative_symmetry_mse']:.4e}"
            )

    rows.sort(key=lambda r: (r.get("hidden_rank", 0), r.get("seed", 0)))

    # aggregate by rank
    by_rank: dict[int, list] = {}
    for row in rows:
        by_rank.setdefault(row["hidden_rank"], []).append(row)

    agg = {}
    for r, lst in sorted(by_rank.items()):
        sym_vals = [x["symmetry_mse"] for x in lst]
        rel_vals = [x["relative_symmetry_mse"] for x in lst]
        te = [x["final_test_mse"] for x in lst if x.get("final_test_mse") is not None]
        agg[str(r)] = {
            "n_seeds": len(lst),
            "symmetry_mse_mean": float(np.mean(sym_vals)),
            "symmetry_mse_std": float(np.std(sym_vals)),
            "relative_symmetry_mse_mean": float(np.mean(rel_vals)),
            "relative_symmetry_mse_std": float(np.std(rel_vals)),
            "final_test_mse_mean": float(np.mean(te)) if te else None,
            "final_test_mse_std": float(np.std(te)) if te else None,
        }

    summary = {"per_run": rows, "aggregate_by_rank": agg, "target": "sum_{k=1}^{factor} cos(2π k x) (even in x)"}
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "hidden_rank",
        "seed",
        "final_test_mse",
        "symmetry_mse",
        "relative_symmetry_mse",
        "epochs_run",
        "name",
    ]
    with open(table_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"wrote {table_csv}")
    print(f"wrote {summary_json}")
    print("aggregate_by_rank:", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
