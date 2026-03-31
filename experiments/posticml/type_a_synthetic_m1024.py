#!/usr/bin/env python3
"""
Type A (post-ICML): controlled 1D sum-of-cosines, hidden width M=1024 fixed.
Sweeps rank r and/or momentum; marks sqrt(M)=32 as reference scale for r.

Usage:
  python experiments/posticml/type_a_synthetic_m1024.py --quick
  python experiments/posticml/type_a_synthetic_m1024.py --ranks 5 10 20 --momenta 0.0 0.9
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from experiments.table.run_scaling_law_depth_width import (  # noqa: E402
    SWEEP_LR_STOP_BELOW,
    SWEEP_MIN_LOSS_DIVISOR,
    train_baseline_sweep_one,
    _sweep_lr_sequence,
)

OUT_ROOT = Path(__file__).resolve().parent / "results" / "type_a_synthetic_m1024"
M_FIXED = 1024
SQRT_M = int(M_FIXED**0.5)  # 32


def main() -> None:
    p = argparse.ArgumentParser(description="Type A: sumcos MMNN, M=1024, sweep r / momentum")
    p.add_argument("--quick", action="store_true", help="tiny sweep, few epochs")
    p.add_argument("--ranks", type=int, nargs="*", default=None, help="hidden ranks (default: 5 10 20 or 10 in quick)")
    p.add_argument("--momenta", type=float, nargs="*", default=[0.0], help="SGD momentum values")
    p.add_argument("--factor", type=int, default=3, help="sumcos factor (1..5)")
    p.add_argument("--n-train", type=int, default=768, help="training grid points")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-epochs", type=int, default=None, help="override max epochs")
    p.add_argument("--overwrite", action="store_true", help="re-run even if losses.json exists")
    args = p.parse_args()

    if args.quick:
        ranks = [10] if args.ranks is None else args.ranks
        num_epochs = args.num_epochs or 400
        momenta = [0.0] if args.momenta == [0.0] else args.momenta
    else:
        ranks = args.ranks if args.ranks is not None else [5, 10, 20, 30]
        num_epochs = args.num_epochs or 8000
        momenta = args.momenta

    lr_seq = _sweep_lr_sequence()
    summary_path = OUT_ROOT / "summary_type_a.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    meta = {
        "M_width": M_FIXED,
        "sqrt_M": SQRT_M,
        "note": f"Reference rank scale r ~ sqrt(M) => r ~ {SQRT_M} (times constant c)",
        "runs": [],
    }

    for mom in momenta:
        for r in ranks:
            name = f"f{args.factor}_N{args.n_train}_bs{args.batch_size}_L{args.num_layers}_M{M_FIXED}_r{r}_mom{mom}"
            out_dir = OUT_ROOT / name
            if args.overwrite and out_dir.exists():
                shutil.rmtree(out_dir)
            if (out_dir / "losses.json").exists():
                print(f"skip (done): {name}")
                with open(out_dir / "losses.json") as f:
                    payload = json.load(f)
                meta["runs"].append({"name": name, "final_test_error": payload.get("final_test_error")})
                continue

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
                "momentum": mom,
                "factor": args.factor,
            }
            print(f"train: {name} (sqrt(M)={SQRT_M})")
            payload = train_baseline_sweep_one(cfg, out_dir)
            meta["runs"].append(
                {
                    "name": name,
                    "hidden_rank": r,
                    "momentum": mom,
                    "final_test_error": payload.get("final_test_error"),
                    "epochs_run": payload.get("epochs_run"),
                }
            )

    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
