#!/usr/bin/env python3
"""
we run table-below-1e-2 configs (min_loss < 1e-2 from baseline sumcos CSV) with fixed lr=1e-4,
300k epochs (default), batch sizes 512/128/64/16 (high to low). Save params every 10 epochs for gif.
Same output dir as --from-table-below: results_sumcos_lowlr_table_below_1e2.
usage: python run_table_below_lowlr_100k.py [--epochs N] [--table-csv PATH] [--adam] [--batch-sizes 16] [--plot-losses]
  Or run plot_table_below_lowlr_losses.py [results_dir] after training to plot loss curves.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from experiments.table.run_selected_sumcos_configs import (
    load_configs_from_table_below,
    train_one_selected,
    TABLE_SUMCOS_CSV,
    TABLE_BATCH_SIZES,
    TABLE_INCLUDE_FACTORS,
    RESULTS_TABLE_BELOW_1E2_DIR,
    RESULTS_TABLE_BELOW_1E2_ADAM_DIR,
    LOWLR_LR,
    LOWLR_SAVE_EVERY_N_EPOCHS,
)
from experiments.table.run_scaling_law_depth_width import (
    SWEEP_LR_WINDOW,
    SWEEP_MIN_LOSS_DIVISOR,
    SWEEP_LR_STOP_BELOW,
    SWEEP_WIDTH,
    SWEEP_RANK,
)

NUM_EPOCHS = 300_000
MIN_LOSS_BELOW = 0.01


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Table-below 1e-2 configs: lr=1e-4, 300k ep (default), save every 10 ep for gif.")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"num epochs (default {NUM_EPOCHS})")
    ap.add_argument("--table-csv", type=Path, default=TABLE_SUMCOS_CSV, help="baseline sumcos CSV path")
    ap.add_argument("--adam", action="store_true", help="use Adam optimizer instead of SGD; output to results_sumcos_lowlr_table_below_1e2_adam")
    ap.add_argument("--batch-sizes", type=str, default=None, help="comma-separated batch sizes (e.g. 16 or 16,32); default 512,128,64,16")
    ap.add_argument("--plot-losses", action="store_true", help="plot loss curves at the end for all completed runs")
    args = ap.parse_args()
    num_epochs = args.epochs
    table_csv = args.table_csv
    use_adam = args.adam
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
        batch_sizes = sorted(batch_sizes, reverse=True)
    else:
        batch_sizes = TABLE_BATCH_SIZES

    if not table_csv.exists():
        print("table CSV not found:", table_csv)
        return

    rows = load_configs_from_table_below(table_csv, MIN_LOSS_BELOW, include_factors=TABLE_INCLUDE_FACTORS, sort_bs_high_to_low=True)
    if not rows:
        print("no configs with min_loss <", MIN_LOSS_BELOW, "in", table_csv)
        return

    lr_seq = [LOWLR_LR]
    fixed_lr_only = True
    save_checkpoint_every_n_epochs = LOWLR_SAVE_EVERY_N_EPOCHS
    out_dir_base = RESULTS_TABLE_BELOW_1E2_ADAM_DIR if use_adam else RESULTS_TABLE_BELOW_1E2_DIR
    out_dir_base.mkdir(parents=True, exist_ok=True)

    # one run per (factor, N, L) per batch size; table has one row per (factor, N, bs_orig, L) so dedupe by (factor, N, L)
    seen_base = set()
    configs = []
    for r in rows:
        key = (r["factor"], r["n_train"], r["num_layers"])
        if key in seen_base:
            continue
        seen_base.add(key)
        n_train = r["n_train"]
        num_layers = r["num_layers"]
        factor = r["factor"]
        for bs in batch_sizes:
            if bs > n_train:
                continue
            name = f"f{factor}_N{n_train}_bs{bs}_L{num_layers}"
            configs.append({
                "name": name,
                "n_train": n_train,
                "batch_size": bs,
                "hidden_width": SWEEP_WIDTH,
                "hidden_rank": SWEEP_RANK,
                "num_layers": num_layers,
                "num_epochs": num_epochs,
                "lr_sequence": lr_seq,
                "lr_window": SWEEP_LR_WINDOW,
                "min_epochs_before_reduce": 0,
                "min_loss_divisor": SWEEP_MIN_LOSS_DIVISOR,
                "lr_stop_below": SWEEP_LR_STOP_BELOW,
                "momentum": 0.0,
                "factor": factor,
                "fixed_lr_only": fixed_lr_only,
                "save_checkpoint_every_n_epochs": save_checkpoint_every_n_epochs,
                "optimizer": "adam" if use_adam else "sgd",
            })
    # high factor first (4, 3, 2, 1), then high bs, then name
    configs.sort(key=lambda c: (-c["factor"], -c["batch_size"], c["name"]))

    opt_str = "Adam" if use_adam else "SGD"
    print(f"TABLE BELOW {MIN_LOSS_BELOW}: {opt_str}, lr={LOWLR_LR:.0e}, epochs={num_epochs}; fixed lr; save every {save_checkpoint_every_n_epochs} ep (for gif).")
    print(f"  include factors {TABLE_INCLUDE_FACTORS} regardless of min_loss; {len(seen_base)} unique (factor,N,L) -> {len(configs)} runs with bs={batch_sizes} (factor 4 first, then high-to-low bs).")
    print(f"Output: {out_dir_base}")

    for i, cfg in enumerate(configs, 1):
        out_name = cfg["name"]
        out_dir = out_dir_base / out_name
        print(f"[{i}/{len(configs)}] {out_name}")
        try:
            train_one_selected(cfg, out_dir)
        except Exception as e:
            print(f"  error: {e}")
            import traceback
            traceback.print_exc()
    if args.plot_losses:
        from experiments.table.plot_table_below_lowlr_losses import main as plot_losses
        print("plotting loss curves...")
        plot_losses(results_dir=out_dir_base)
    print("done.")


if __name__ == "__main__":
    main()
