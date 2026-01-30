#!/usr/bin/env python3
"""
we rerun a fixed list of sumcos configs (factor 3) that worked well; we save parameters just before
each LR divide and 10 epochs after each LR divide (2 saves per divide).
usage: python run_selected_sumcos_configs.py           # default: lr 1e-2 with divide-by-2, 10k max epochs
        python run_selected_sumcos_configs.py --low-lr-long --only f3_N768_bs4_L3   # lr=1e-4 fixed, 300k ep, save params every 10 ep (for gif); run one config
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_TABLE = _REPO_ROOT / "experiments" / "table"
RESULTS_SELECTED_DIR = _TABLE / "results_sumcos_selected_rerun"
RESULTS_SELECTED_LOWLR_DIR = _TABLE / "results_sumcos_selected_rerun_lowlr"

# low-lr long run: fixed lr 1e-4, 300k epochs (no LR decay); save params every 10 epochs for gif
LOWLR_LR = 1e-4
LOWLR_NUM_EPOCHS = 300_000
LOWLR_SAVE_EVERY_N_EPOCHS = 10

from experiments.table.mmnn_vs import MMNN

# we reuse sweep constants from run_scaling_law_depth_width
from experiments.table.run_scaling_law_depth_width import (
    SWEEP_LR_DIVISOR,
    SWEEP_LR_INIT,
    SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
    SWEEP_LR_N_STEPS,
    SWEEP_LR_STOP_BELOW,
    SWEEP_LR_WINDOW,
    SWEEP_MIN_LOSS_DIVISOR,
    SWEEP_NUM_EPOCHS_MAX,
    SWEEP_RANK,
    SWEEP_WIDTH,
    target_baseline,
    _sweep_lr_sequence,
)

# selected configs from sumcos sweep (factor 3): name, N, bs, L, final_test_error, epochs_run (reference)
SELECTED_SUMCOS_CONFIGS = [
    {"name": "f3_N768_bs4_L3", "n_train": 768, "batch_size": 4, "num_layers": 3},
    {"name": "f3_N768_bs2_L3", "n_train": 768, "batch_size": 2, "num_layers": 3},
    {"name": "f3_N768_bs8_L3", "n_train": 768, "batch_size": 8, "num_layers": 3},
    {"name": "f3_N384_bs2_L2", "n_train": 384, "batch_size": 2, "num_layers": 2},
    {"name": "f3_N768_bs8_L2", "n_train": 768, "batch_size": 8, "num_layers": 2},
    {"name": "f3_N768_bs1_L1", "n_train": 768, "batch_size": 1, "num_layers": 1},
    {"name": "f3_N768_bs2_L2", "n_train": 768, "batch_size": 2, "num_layers": 2},
    {"name": "f3_N768_bs4_L2", "n_train": 768, "batch_size": 4, "num_layers": 2},
    {"name": "f3_N768_bs1_L2", "n_train": 768, "batch_size": 1, "num_layers": 2},
    {"name": "f3_N384_bs1_L2", "n_train": 384, "batch_size": 1, "num_layers": 2},
]
FACTOR = 3
EPOCHS_AFTER_LR_DIVIDE = 10


def train_one_selected(config, output_dir):
    """we train one selected config (sumcos, factor 3); save params just before each LR divide and 10 epochs after each divide"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    n_train = config["n_train"]
    batch_size = config["batch_size"]
    hidden_width = config["hidden_width"]
    hidden_rank = config["hidden_rank"]
    num_layers = config["num_layers"]
    num_epochs = config["num_epochs"]
    lr_sequence = config["lr_sequence"]
    window_size = config["lr_window"]
    min_epochs_before_reduce = config["min_epochs_before_reduce"]
    min_loss_divisor = config["min_loss_divisor"]
    momentum = config.get("momentum", 0.0)
    factor = config.get("factor", FACTOR)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [hidden_width] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)

    interval = [-1, 1]
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_baseline(x_train, factor)
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)

    lr_init = lr_sequence[0]
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)

    current_lr_index = 0
    last_reduction_epoch = -1
    lr_reduction_epochs = []  # epochs where we divided LR (we save before at E, after at E+10)

    all_losses = []
    all_lrs = []
    init_loss = None
    min_loss_so_far = float("inf")
    min_loss_counter = 0
    min_loss_checkpoints = []
    lr_divide_saves = []  # list of {epoch_before, epoch_after_plus10} for logging
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    LOG_INTERVAL_SEC = 5.0
    last_log_time = time.time()
    pbar = tqdm(range(num_epochs), desc=output_dir.name, unit="ep", mininterval=LOG_INTERVAL_SEC)
    for epoch in pbar:
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i : i + batch_size]
            x_batch = x_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.MSELoss()(y_pred, y_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                tqdm.write(f"  NaN/Inf at epoch {epoch}, stopping")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if n_batches == 0:
            break
        epoch_loss /= n_batches
        if np.isnan(epoch_loss) or np.isinf(epoch_loss):
            break
        all_losses.append(float(epoch_loss))
        all_lrs.append(float(optimizer.param_groups[0]["lr"]))

        if init_loss is None:
            init_loss = epoch_loss

        if epoch_loss < min_loss_so_far:
            min_loss_so_far = epoch_loss
        threshold = init_loss / (min_loss_divisor ** (min_loss_counter + 1))
        if min_loss_so_far < threshold:
            min_loss_counter += 1
            ckpt_dir = output_dir / f"params_at_div_{min_loss_divisor}_{min_loss_counter}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "model_parameters.pth")
            min_loss_checkpoints.append({
                "counter": min_loss_counter,
                "epoch": epoch,
                "loss": float(min_loss_so_far),
                "threshold": float(threshold),
            })

        fixed_lr_only = config.get("fixed_lr_only", False)
        if not fixed_lr_only:
            # save 10 epochs after a previous LR divide (we do this before checking new divide so order is clear)
            if epoch >= EPOCHS_AFTER_LR_DIVIDE and (epoch - EPOCHS_AFTER_LR_DIVIDE) in lr_reduction_epochs:
                e_before = epoch - EPOCHS_AFTER_LR_DIVIDE
                after_dir = output_dir / f"params_after_lr_divide_epoch_{e_before}_plus{EPOCHS_AFTER_LR_DIVIDE}"
                after_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), after_dir / "model_parameters.pth")
                lr_divide_saves.append({"after_epoch": epoch, "divide_epoch": e_before})
                if (epoch + 1) % 200 == 0 or epoch < 50:
                    tqdm.write(f"  saved after_lr_divide epoch {e_before}+{EPOCHS_AFTER_LR_DIVIDE} -> {after_dir.name}")

            # AdaptiveStagnation: reduce lr when stagnating; save params *before* applying new lr
            if (
                epoch >= min_epochs_before_reduce
                and epoch - last_reduction_epoch >= min_epochs_before_reduce
                and len(all_losses) >= 2 * window_size
                and current_lr_index < len(lr_sequence) - 1
            ):
                recent = np.mean(all_losses[-window_size:])
                prev = np.mean(all_losses[-2 * window_size : -window_size])
                if recent >= prev:
                    # save just before this LR divide (current state, still with old lr for this epoch)
                    before_dir = output_dir / f"params_before_lr_divide_epoch_{epoch}"
                    before_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), before_dir / "model_parameters.pth")
                    lr_reduction_epochs.append(epoch)
                    current_lr_index += 1
                    new_lr = lr_sequence[current_lr_index]
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    last_reduction_epoch = epoch
                    if (epoch + 1) % 200 == 0 or epoch < 50:
                        tqdm.write(f"  saved before_lr_divide epoch {epoch} -> {before_dir.name}")

            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < config.get("lr_stop_below", SWEEP_LR_STOP_BELOW):
                tqdm.write(f"  stopping: lr={current_lr:.2e} < {config.get('lr_stop_below', SWEEP_LR_STOP_BELOW):.0e}")
                break

        current_lr = optimizer.param_groups[0]["lr"]
        now = time.time()
        if now - last_log_time >= LOG_INTERVAL_SEC:
            pbar.set_postfix(loss=f"{epoch_loss:.4e}", min_loss=f"{min_loss_so_far:.4e}", lr=f"{current_lr:.2e}", lr_div=len(lr_reduction_epochs))
            last_log_time = now
        # save params every N epochs when fixed_lr_only (for gif later)
        save_every = config.get("save_checkpoint_every_n_epochs", 0)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            ckpt_dir = output_dir / f"params_epoch_{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "model_parameters.pth")

    training_time = time.time() - start_time

    n_test = 500
    x_test = np.linspace(interval[0], interval[1], n_test)
    y_test = target_baseline(x_test, factor)
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_pred = model(x_test_tensor)
        final_test_error = nn.MSELoss()(y_pred, y_test_tensor).item()
        final_test_error_max = torch.max(torch.abs(y_pred - y_test_tensor)).item()

    total_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), output_dir / "model_parameters.pth")
    losses_payload = {
        "config": config,
        "final_train_error": float(all_losses[-1]) if all_losses else None,
        "final_test_error": float(final_test_error),
        "final_test_error_max": float(final_test_error_max),
        "training_time_seconds": float(training_time),
        "total_parameters": int(total_params),
        "epochs_run": len(all_losses),
        "all_losses": [float(x) for x in all_losses],
        "all_lrs": [float(x) for x in all_lrs],
        "lr_reduction_epochs": list(lr_reduction_epochs),
        "lr_divide_saves": lr_divide_saves,
        "init_loss": float(init_loss) if init_loss is not None else None,
        "min_loss_checkpoints": min_loss_checkpoints,
    }
    with open(output_dir / "losses.json", "w") as f:
        json.dump(losses_payload, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    n_before = len([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("params_before_lr_divide")])
    n_after = len([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("params_after_lr_divide")])
    print(f"  completed: {output_dir.name} test_err={final_test_error:.4e} lr_divides={len(lr_reduction_epochs)} before_saves={n_before} after_saves={n_after}")
    return losses_payload


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rerun selected sumcos configs; optional low-lr long run.")
    parser.add_argument("--low-lr-long", action="store_true", help="fixed lr=1e-4; save params every 10 ep; output to results_sumcos_selected_rerun_lowlr")
    parser.add_argument("--only", type=str, default=None, metavar="CONFIG", help="run only this config (e.g. f3_N768_bs4_L3)")
    parser.add_argument("--epochs", type=int, default=None, help="override num_epochs (e.g. 2000)")
    parser.add_argument("--batch-sizes", type=str, default=None, help="comma-separated batch sizes (e.g. 1,16,32,128); with --only BASE run BASE N,L with each bs")
    args = parser.parse_args()
    low_lr_long = args.low_lr_long
    only_config = args.only
    epochs_override = args.epochs
    batch_sizes_str = args.batch_sizes

    if low_lr_long:
        out_dir_base = RESULTS_SELECTED_LOWLR_DIR
        lr_seq = [LOWLR_LR]
        num_epochs = epochs_override if epochs_override is not None else LOWLR_NUM_EPOCHS
        fixed_lr_only = True
        save_checkpoint_every_n_epochs = LOWLR_SAVE_EVERY_N_EPOCHS
        print(f"SELECTED SUMCOS RERUN (low-lr long): lr={LOWLR_LR:.0e}, epochs={num_epochs}")
        print(f"  Fixed lr only; no LR divide policy. Save params every {save_checkpoint_every_n_epochs} epochs (for gif).")
        print(f"Output: {out_dir_base}")
    else:
        out_dir_base = RESULTS_SELECTED_DIR
        lr_seq = _sweep_lr_sequence()
        num_epochs = SWEEP_NUM_EPOCHS_MAX
        fixed_lr_only = False
        save_checkpoint_every_n_epochs = 0
        print(f"SELECTED SUMCOS RERUN: {len(SELECTED_SUMCOS_CONFIGS)} configs (factor={FACTOR})")
        print(f"  Saves: params_before_lr_divide_epoch_E and params_after_lr_divide_epoch_E_plus{EPOCHS_AFTER_LR_DIVIDE}")
        print(f"Output: {out_dir_base}")

    out_dir_base.mkdir(parents=True, exist_ok=True)
    configs = []
    if batch_sizes_str and only_config and low_lr_long:
        batch_sizes_list = [int(x.strip()) for x in batch_sizes_str.split(",") if x.strip()]
        base = next((c for c in SELECTED_SUMCOS_CONFIGS if c["name"] == only_config), None)
        if base is None:
            print("--only config not in SELECTED_SUMCOS_CONFIGS:", only_config)
            return
        n_train = base["n_train"]
        num_layers = base["num_layers"]
        for bs in batch_sizes_list:
            if bs > n_train:
                continue
            name = f"f{FACTOR}_N{n_train}_bs{bs}_L{num_layers}"
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
                "min_epochs_before_reduce": SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
                "min_loss_divisor": SWEEP_MIN_LOSS_DIVISOR,
                "lr_stop_below": SWEEP_LR_STOP_BELOW,
                "momentum": 0.0,
                "factor": FACTOR,
                "fixed_lr_only": fixed_lr_only,
                "save_checkpoint_every_n_epochs": save_checkpoint_every_n_epochs if low_lr_long else 0,
            })
        print(f"  batch_sizes={batch_sizes_list} (base {only_config})")
    else:
        for c in SELECTED_SUMCOS_CONFIGS:
            if only_config is not None and c["name"] != only_config:
                continue
            configs.append({
                "name": c["name"],
                "n_train": c["n_train"],
                "batch_size": c["batch_size"],
                "hidden_width": SWEEP_WIDTH,
                "hidden_rank": SWEEP_RANK,
                "num_layers": c["num_layers"],
                "num_epochs": num_epochs,
                "lr_sequence": lr_seq,
                "lr_window": SWEEP_LR_WINDOW,
                "min_epochs_before_reduce": SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
                "min_loss_divisor": SWEEP_MIN_LOSS_DIVISOR,
                "lr_stop_below": SWEEP_LR_STOP_BELOW,
                "momentum": 0.0,
                "factor": FACTOR,
                "fixed_lr_only": fixed_lr_only,
                "save_checkpoint_every_n_epochs": save_checkpoint_every_n_epochs if low_lr_long else 0,
            })
    if not configs:
        print("no configs to run (check --only and --batch-sizes)")
        return

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
    print("Selected sumcos rerun done.")


if __name__ == "__main__":
    main()
