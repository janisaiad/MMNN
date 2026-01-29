#!/usr/bin/env python3
"""
main experiment runner: stable baseline (cos 2pi x) and frequency/layer scaling (depth and width).
goal: find scaling laws for N (training data), width, L (depth), freq; stable baseline first.
usage:
  python run_scaling_law_depth_width.py --baseline           # stable baseline: cos(2pi x), N=width=1024
  python run_scaling_law_depth_width.py --baseline-sweep     # sweep sumcos: sum_{k=1}^{f} cos(2 pi k x), factor 1..5
  python run_scaling_law_depth_width.py --baseline-sweep-expcos  # sweep expcos: sum_{k=0}^{f} cos(2^k pi x), factor 3 and 4 only; N = mult*2^factor (Nyquist)
  python run_scaling_law_depth_width.py --train              # scaling-law training only
  python run_scaling_law_depth_width.py --analyze            # scaling-law analysis only
  python run_scaling_law_depth_width.py                      # scaling-law: train then analyze (default)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# we add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from experiments.table.mmnn_vs import MMNN

# we use same output dir as original experiments (paths relative to repo root)
_TABLE = _REPO_ROOT / "experiments" / "table"
RESULTS_DIR = _TABLE / "results_frequency_layer_scaling"
RESULTS_BASELINE_DIR = _TABLE / "results_stable_baseline"
RESULTS_BASELINE_SWEEP_DIR = _TABLE / "results_baseline_sweep_sumcos"
RESULTS_BASELINE_SWEEP_EXPCOS_DIR = _TABLE / "results_baseline_sweep_expcos"
ANALYSIS_CSV = _TABLE / "frequency_layer_scaling_analysis.csv"
SUMMARY_TXT = _TABLE / "frequency_layer_scaling_summary.txt"

# stable baseline: cos(2 pi x), N_train = width = 1024 (from rerun_four_setups)
BASELINE_N_TRAIN = 1024
BASELINE_WIDTH = 1024
BASELINE_LAYERS = 2
BASELINE_RANK = 10
BASELINE_FACTOR = 1
BASELINE_BATCH_SIZE = 4
BASELINE_NUM_EPOCHS = 250
BASELINE_LR_SEQUENCE = [0.01, 0.005, 0.001, 0.0005, 0.0001]
BASELINE_LR_WINDOW = 10
BASELINE_LR_MIN_EPOCHS_BEFORE_REDUCE = 20

# baseline sweep: factor 1..5, N = base * factor, batch [1,2,4,8,16], layers 1..2*factor, epochs until 10K or lr < 1e-6
SWEEP_BASE_N = [16, 32, 64, 128, 256]
SWEEP_FACTORS = [1, 2, 3, 4, 5]
SWEEP_BATCH_SIZES = [1, 2, 4, 8, 16]
SWEEP_NUM_EPOCHS_MAX = 10_000
SWEEP_LR_STOP_BELOW = 1e-6
SWEEP_LR_INIT = 1e-2
SWEEP_LR_DIVISOR = 2
SWEEP_LR_N_STEPS = 25
SWEEP_MIN_LOSS_DIVISOR = 1.2
SWEEP_WIDTH = 1024
SWEEP_RANK = 10
SWEEP_LR_WINDOW = 10
SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE = 20

# expcos sweep: target = sum_{k=0}^{factor} cos(2^k pi x); factors 3 and 4 only.
# How to choose N_samples: highest mode is cos(2^factor pi x), with 2^factor periods on [-1,1]. Nyquist: need at least 2 points per period => N >= 2 * 2^factor. We use N = mult * 2^factor with mult in {4, 8, 16} for safety (factor 3: N in {32, 64, 128}; factor 4: N in {64, 128, 256}).
SWEEP_EXPCOS_FACTORS = [3, 4]
SWEEP_EXPCOS_N_MULTIPLIERS = [4, 8, 16]  # N = mult * 2^factor for each factor

# we configure matplotlib for LaTeX-style plots
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["font.size"] = 18
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 22
mpl.rcParams["axes.formatter.limits"] = (-6, 6)
mpl.rcParams["axes.formatter.use_mathtext"] = True


def target_baseline(x, factor=1.0):
    """for factor n we fit sum_{k=1}^{n} cos(2 pi k x) on [-1,1]"""
    if factor < 1:
        return np.cos(2 * np.pi * x)
    out = np.zeros_like(x, dtype=float)
    for k in range(1, int(factor) + 1):
        out += np.cos(2 * k * np.pi * x)
    return out


def target_baseline_exp(x, factor=1.0):
    """for factor n we fit sum_{k=0}^{n} cos(2^k pi x) on [-1,1]; highest mode has 2^factor periods in [-1,1], so Nyquist needs N >= 2 * 2^factor; we use N in {4*2^factor, 8*2^factor, 16*2^factor}"""
    if factor < 0:
        return np.cos(np.pi * x)
    out = np.zeros_like(x, dtype=float)
    for k in range(0, int(factor) + 1):
        out += np.cos((2 ** k) * np.pi * x)
    return out


def target_function(x, freq_multiplier=1.0):
    """we define the multi-frequency function with phase shifts, scaled by freq_multiplier"""
    base_freqs = [12, 24, 36, 72]
    scaled_freqs = [f * freq_multiplier for f in base_freqs]
    return (
        np.cos(scaled_freqs[0] * np.pi * x)
        + np.cos(scaled_freqs[1] * np.pi * x + 0.5)
        + np.cos(scaled_freqs[2] * np.pi * x)
        + np.cos(scaled_freqs[3] * np.pi * x + 0.5)
    )


def generate_configs():
    """we generate configurations for frequency and layer scaling (depth) and rank (width)"""
    configs = []
    base_layers = 8
    base_ranks = [10, 15, 25]
    batch_size = 100
    fixWb = False
    freq_multipliers = [0.3, 0.6, 1.5, 2, 3, 5, 7, 10]

    base_config = {
        "hidden_width": 777,
        "input_rank": 1,
        "output_rank": 1,
        "use_resnet": False,
        "batch_size": batch_size,
        "lr_init": 0.001,
        "lr_gamma": 0.9,
        "lr_step_size": 100,
        "interval": [-1, 1],
        "show_plot": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": "torch.float32",
        "num_training_samples": 1000,
        "num_test_samples": 1234,
        "fixWb": fixWb,
    }

    for freq_mult in freq_multipliers:
        base_layer_count = math.ceil(freq_mult * base_layers)
        if freq_mult == 0.3:
            layer_counts = [math.ceil(0.3 * base_layers), math.ceil(0.6 * base_layers)]
        elif freq_mult == 0.6:
            layer_counts = [
                math.ceil(0.3 * base_layers),
                math.ceil(0.6 * base_layers),
                math.ceil(1.5 * 0.6 * base_layers),
            ]
        elif freq_mult == 1.5:
            layer_counts = [
                math.ceil(0.6 * 1.5 * base_layers),
                math.ceil(1.5 * base_layers),
                math.ceil(2 * 1.5 * base_layers),
            ]
        elif freq_mult == 2:
            layer_counts = [math.ceil(1.5 * base_layers), math.ceil(3 * base_layers)]
        elif freq_mult == 3:
            layer_counts = [math.ceil(2 * base_layers), math.ceil(5 * base_layers)]
        elif freq_mult == 5:
            layer_counts = [math.ceil(3 * base_layers), math.ceil(7 * base_layers)]
        elif freq_mult == 7:
            layer_counts = [math.ceil(5 * base_layers), math.ceil(10 * base_layers)]
        elif freq_mult == 10:
            layer_counts = [math.ceil(7 * base_layers), math.ceil(10 * base_layers)]
        else:
            layer_counts = [base_layer_count]
        if base_layer_count not in layer_counts:
            layer_counts.append(base_layer_count)
        layer_counts = sorted(set(layer_counts))

        num_epochs = int(2 * freq_mult * 10000)
        base_freqs = [12, 24, 36, 72]
        scaled_freqs = [f * freq_mult for f in base_freqs]
        func_str = (
            f"cos({scaled_freqs[0]}*pi*x) + cos({scaled_freqs[1]}*pi*x + 0.5) + "
            f"cos({scaled_freqs[2]}*pi*x) + cos({scaled_freqs[3]}*pi*x + 0.5)"
        )

        for rank in base_ranks:
            for num_layers in layer_counts:
                config = base_config.copy()
                config.update({
                    "hidden_rank": rank,
                    "num_layers": num_layers,
                    "num_epochs": num_epochs,
                    "freq_multiplier": freq_mult,
                    "function": func_str,
                })
                configs.append(config)

    return configs


def train_one_config(config, output_dir, target_func):
    """we train one configuration with loss threshold checkpoints"""
    device = torch.device(config["device"])
    mydtype = torch.get_default_dtype()

    arch_label = (
        "FULL_RANK"
        if config["hidden_rank"] == config["hidden_width"]
        else f"rank={config['hidden_rank']}"
    )
    print(f"\n{'='*80}")
    print(f"Training: {arch_label}, fixWb={config['fixWb']}, layers={config['num_layers']}")
    print(f"Frequency multiplier: {config['freq_multiplier']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    ranks = [config["input_rank"]] + [config["hidden_rank"]] * config["num_layers"] + [config["output_rank"]]
    widths = [config["hidden_width"]] * (config["num_layers"] + 1)
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=config["use_resnet"],
        fixWb=config["fixWb"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}, trainable: {trainable_params:,}")

    x_train = np.linspace(*config["interval"], config["num_training_samples"]).reshape([-1, 1])
    y_train = target_func(x_train, config["freq_multiplier"])
    x_train = torch.tensor(x_train, device=device, dtype=mydtype)
    y_train = torch.tensor(y_train, device=device, dtype=mydtype)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    x_test = np.random.rand(config["num_test_samples"]) * 2 - 1
    y_test = target_func(x_test, config["freq_multiplier"])
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)

    optimizer = optim.Adam(model.parameters(), lr=config["lr_init"])
    scheduler = StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])
    criterion = nn.MSELoss()

    checkpoint_path = output_dir / "checkpoint.pth"
    start_epoch = 1
    all_losses = []
    errors_train = []
    errors_test = []
    errors_test_max = []

    loss_thresholds = [
        1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3,
        5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5,
        5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7,
        5e-8, 2e-8, 1e-8, 5e-9, 2e-9, 1e-9,
    ]
    thresholds_reached = set()

    if checkpoint_path.exists():
        print(f"loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        all_losses = checkpoint.get("all_losses", [])
        errors_train = checkpoint.get("errors_train", [])
        errors_test = checkpoint.get("errors_test", [])
        errors_test_max = checkpoint.get("errors_test_max", [])
        thresholds_reached = set(checkpoint.get("thresholds_reached", []))
        print(f"resuming from epoch {start_epoch}")
    else:
        print("starting fresh training")

    start_time = time.time()

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        epoch_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        all_losses.append(avg_loss)
        scheduler.step()

        for threshold in loss_thresholds:
            if threshold not in thresholds_reached and avg_loss < threshold:
                thresholds_reached.add(threshold)
                threshold_dir = output_dir / f"model_at_loss_{threshold:.0e}"
                threshold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), threshold_dir / "model_parameters.pth")
                with open(threshold_dir / "epoch_info.json", "w") as f:
                    json.dump({"epoch": epoch, "loss": float(avg_loss), "threshold": float(threshold)}, f, indent=4)
                print(f"  loss {avg_loss:.6e} < {threshold:.0e} at epoch {epoch} -> saved")

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                y_pred = model(x_test_tensor)
                test_error = criterion(y_pred, y_test_tensor).item()
                test_error_max = torch.max(torch.abs(y_pred - y_test_tensor)).item()
                errors_train.append(avg_loss)
                errors_test.append(test_error)
                errors_test_max.append(test_error_max)
                print(f"Epoch {epoch}/{config['num_epochs']}: train={avg_loss:.4e}, test={test_error:.4e}, max={test_error_max:.4e}")

        if epoch % 500 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "all_losses": all_losses,
                    "errors_train": errors_train,
                    "errors_test": errors_test,
                    "errors_test_max": errors_test_max,
                    "thresholds_reached": list(thresholds_reached),
                },
                checkpoint_path,
            )
            print(f"checkpoint saved at epoch {epoch}")

    training_time = time.time() - start_time

    torch.save(
        {
            "epoch": config["num_epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "all_losses": all_losses,
            "errors_train": errors_train,
            "errors_test": errors_test,
            "errors_test_max": errors_test_max,
            "thresholds_reached": list(thresholds_reached),
        },
        checkpoint_path,
    )

    results = {
        "config": config,
        "final_train_error": float(errors_train[-1]) if errors_train else None,
        "final_test_error": float(errors_test[-1]) if errors_test else None,
        "final_test_error_max": float(errors_test_max[-1]) if errors_test_max else None,
        "training_time_seconds": float(training_time),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "epochs_run": int(len(all_losses)),
        "thresholds_reached": list(thresholds_reached),
        "all_losses": [float(x) for x in all_losses],
        "errors_train": [float(x) for x in errors_train],
        "errors_test": [float(x) for x in errors_test],
        "errors_test_max": [float(x) for x in errors_test_max],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    torch.save(model.state_dict(), output_dir / "model_parameters.pth")

    # we plot final prediction and loss
    x_plot = np.linspace(*config["interval"], 1000)
    x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
    y_plot_true = target_func(x_plot, config["freq_multiplier"])
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot_true, "b-", label="True function", linewidth=2)
    plt.plot(x_plot, y_plot_nn, "r--", label="Learned network", linewidth=2)
    plt.xlabel("$x$", fontsize=22)
    plt.ylabel("$f(x)$", fontsize=22)
    config_str = f"{arch_label}, L={config['num_layers']}, freq×{config['freq_multiplier']}, epoch {len(all_losses)}"
    plt.title(f"Final Prediction\n{config_str}", fontsize=20)
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / "final_prediction.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(all_losses) + 1), all_losses, "b-", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=22)
    plt.ylabel("Loss (log scale)", fontsize=22)
    plt.title(f"Training Loss Evolution\n{config_str}", fontsize=20)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  completed: {output_dir.name}")
    return results


def run_training():
    """we run all scaling-law training configs"""
    configs = generate_configs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FREQUENCY AND LAYER SCALING BENCHMARK — {len(configs)} configs -> {RESULTS_DIR}")

    for i, config in enumerate(configs, 1):
        output_dir_name = f"freq{config['freq_multiplier']}_rank{config['hidden_rank']}_L{config['num_layers']}"
        output_dir = RESULTS_DIR / output_dir_name

        checkpoint_file = output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            ckpt = torch.load(checkpoint_file, map_location="cpu")
            if ckpt.get("epoch", 0) >= config["num_epochs"]:
                print(f"[{i}/{len(configs)}] skip (done): {output_dir_name}")
                continue

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_one_config(config, output_dir, target_function)
        except Exception as e:
            print(f"  error in {output_dir_name}: {e}")
            import traceback
            traceback.print_exc()
    print("Training phase done.")


def train_baseline_one(config, output_dir):
    """we train one stable-baseline config: cos(2pi x), SGD + AdaptiveStagnation (from rerun_four_setups)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    n_train = config["n_train"]
    hidden_width = config["hidden_width"]
    hidden_rank = config["hidden_rank"]
    num_layers = config["num_layers"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr_init = config["lr_init"]
    momentum = config.get("momentum", 0.0)
    factor = config.get("factor", BASELINE_FACTOR)
    lr_sequence = config.get("lr_sequence", BASELINE_LR_SEQUENCE)
    window_size = config.get("lr_window", BASELINE_LR_WINDOW)
    min_epochs_before_reduce = config.get("min_epochs_before_reduce", BASELINE_LR_MIN_EPOCHS_BEFORE_REDUCE)

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

    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)

    current_lr_index = 0
    last_reduction_epoch = -1
    lr_reduction_epochs = []

    all_losses = []
    all_lrs = []
    start_time = time.time()

    for epoch in range(num_epochs):
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
                print(f"  NaN/Inf at epoch {epoch}, stopping")
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

        # AdaptiveStagnation
        if (
            epoch >= min_epochs_before_reduce
            and epoch - last_reduction_epoch >= min_epochs_before_reduce
            and len(all_losses) >= 2 * window_size
            and current_lr_index < len(lr_sequence) - 1
        ):
            recent = np.mean(all_losses[-window_size:])
            prev = np.mean(all_losses[-2 * window_size : -window_size])
            if recent >= prev:
                current_lr_index += 1
                new_lr = lr_sequence[current_lr_index]
                for g in optimizer.param_groups:
                    g["lr"] = new_lr
                last_reduction_epoch = epoch
                lr_reduction_epochs.append(epoch)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{num_epochs} loss={epoch_loss:.4e} lr={optimizer.param_groups[0]['lr']:.6f}")

    training_time = time.time() - start_time

    # we evaluate on a fixed test set
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
    results = {
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
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f"  completed: {output_dir.name} test_err={final_test_error:.4e}")
    return results


def run_baseline():
    """we run stable baseline: cos(2pi x), N_train=width=1024, SGD+AdaptiveStagnation, 4 momenta (from rerun_four_setups)"""
    RESULTS_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    print("STABLE BASELINE: cos(2 pi x), N_train=width=1024, SGD lr=0.01 AdaptiveStagnation")
    print(f"Output: {RESULTS_BASELINE_DIR}")

    base_cfg = {
        "n_train": BASELINE_N_TRAIN,
        "hidden_width": BASELINE_WIDTH,
        "hidden_rank": BASELINE_RANK,
        "num_layers": BASELINE_LAYERS,
        "batch_size": BASELINE_BATCH_SIZE,
        "num_epochs": BASELINE_NUM_EPOCHS,
        "lr_init": 0.01,
        "factor": BASELINE_FACTOR,
        "lr_sequence": BASELINE_LR_SEQUENCE,
        "lr_window": BASELINE_LR_WINDOW,
        "min_epochs_before_reduce": BASELINE_LR_MIN_EPOCHS_BEFORE_REDUCE,
    }

    configs = [
        {**base_cfg, "name": "cos2pi_N1024_W1024_rank10_L2_SGD_mom0.0_AdaptiveStagnation", "momentum": 0.0},
        {**base_cfg, "name": "cos2pi_N1024_W1024_rank10_L2_SGD_mom0.3_AdaptiveStagnation", "momentum": 0.3},
        {**base_cfg, "name": "cos2pi_N1024_W1024_rank10_L2_SGD_mom0.6_AdaptiveStagnation", "momentum": 0.6},
        {**base_cfg, "name": "cos2pi_N1024_W1024_rank10_L2_SGD_mom0.7_AdaptiveStagnation", "momentum": 0.7},
    ]

    for i, cfg in enumerate(configs, 1):
        out_name = cfg["name"]
        out_dir = RESULTS_BASELINE_DIR / out_name
        if (out_dir / "results.json").exists():
            print(f"[{i}/{len(configs)}] skip (done): {out_name}")
            continue
        print(f"[{i}/{len(configs)}] {out_name}")
        try:
            train_baseline_one(cfg, out_dir)
        except Exception as e:
            print(f"  error: {e}")
            import traceback
            traceback.print_exc()
    print("Baseline phase done.")


def _sweep_lr_sequence():
    """we build lr sequence: start 1e-2, divide by 2 each step"""
    return [SWEEP_LR_INIT / (SWEEP_LR_DIVISOR ** i) for i in range(SWEEP_LR_N_STEPS)]


def train_baseline_sweep_one(config, output_dir):
    """we train one sweep config: cos(2pi x), N and batch_size from config, lr start 1e-2 divide by 2, save params when min_loss < init_loss/1.2^k"""
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
    factor = config.get("factor", BASELINE_FACTOR)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [hidden_width] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)

    target_fn = target_baseline_exp if config.get("target") == "expcos" else target_baseline
    interval = [-1, 1]
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_fn(x_train, factor)
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)

    lr_init = lr_sequence[0]
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)

    current_lr_index = 0
    last_reduction_epoch = -1
    lr_reduction_epochs = []

    all_losses = []
    all_lrs = []
    init_loss = None
    min_loss_so_far = float("inf")
    min_loss_counter = 0
    min_loss_checkpoints = []
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
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
                print(f"  NaN/Inf at epoch {epoch}, stopping")
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

        # AdaptiveStagnation: reduce lr when stagnating (divide-by-2 sequence)
        if (
            epoch >= min_epochs_before_reduce
            and epoch - last_reduction_epoch >= min_epochs_before_reduce
            and len(all_losses) >= 2 * window_size
            and current_lr_index < len(lr_sequence) - 1
        ):
            recent = np.mean(all_losses[-window_size:])
            prev = np.mean(all_losses[-2 * window_size : -window_size])
            if recent >= prev:
                current_lr_index += 1
                new_lr = lr_sequence[current_lr_index]
                for g in optimizer.param_groups:
                    g["lr"] = new_lr
                last_reduction_epoch = epoch
                lr_reduction_epochs.append(epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < config.get("lr_stop_below", SWEEP_LR_STOP_BELOW):
            print(f"  stopping: lr={current_lr:.2e} < {config.get('lr_stop_below', SWEEP_LR_STOP_BELOW):.0e}")
            break

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{num_epochs} loss={epoch_loss:.4e} min_loss={min_loss_so_far:.4e} lr={current_lr:.6f}")

    training_time = time.time() - start_time

    n_test = 500
    x_test = np.linspace(interval[0], interval[1], n_test)
    y_test = target_fn(x_test, factor)
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_pred = model(x_test_tensor)
        final_test_error = nn.MSELoss()(y_pred, y_test_tensor).item()
        final_test_error_max = torch.max(torch.abs(y_pred - y_test_tensor)).item()

    total_params = sum(p.numel() for p in model.parameters())
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
        "init_loss": float(init_loss) if init_loss is not None else None,
        "min_loss_checkpoints": min_loss_checkpoints,
    }
    torch.save(model.state_dict(), output_dir / "model_parameters.pth")
    with open(output_dir / "losses.json", "w") as f:
        json.dump(losses_payload, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  completed: {output_dir.name} test_err={final_test_error:.4e} min_loss_saves={len(min_loss_checkpoints)}")
    return losses_payload


def run_baseline_sweep():
    """we run sweep: target = sum_{k=1}^{factor} cos(2 pi k x); factor 1..5, N = base*factor, batch_size [1,2,4,8,16], layers 1..2*factor; epochs until 10K or lr < 1e-6; save losses.json and params at min_loss/1.2^k"""
    RESULTS_BASELINE_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for f in RESULTS_BASELINE_SWEEP_DIR.rglob("epoch_info.json"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    lr_seq = _sweep_lr_sequence()
    print(f"BASELINE SWEEP: factors {SWEEP_FACTORS}, N = {SWEEP_BASE_N} * factor, batch_size {SWEEP_BATCH_SIZES}, layers 1..2*factor, max_epochs {SWEEP_NUM_EPOCHS_MAX}, stop if lr < {SWEEP_LR_STOP_BELOW}")
    print(f"Output: {RESULTS_BASELINE_SWEEP_DIR}")

    configs = []
    for factor in SWEEP_FACTORS:
        n_samples_for_factor = [n * factor for n in SWEEP_BASE_N]
        layer_counts = list(range(1, 2 * factor + 1))
        for n_train in n_samples_for_factor:
            for batch_size in SWEEP_BATCH_SIZES:
                if batch_size > n_train:
                    continue
                for num_layers in layer_counts:
                    name = f"f{factor}_N{n_train}_bs{batch_size}_L{num_layers}"
                    configs.append({
                        "name": name,
                        "n_train": n_train,
                        "batch_size": batch_size,
                        "hidden_width": SWEEP_WIDTH,
                        "hidden_rank": SWEEP_RANK,
                        "num_layers": num_layers,
                        "num_epochs": SWEEP_NUM_EPOCHS_MAX,
                        "lr_sequence": lr_seq,
                        "lr_window": SWEEP_LR_WINDOW,
                        "min_epochs_before_reduce": SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
                        "min_loss_divisor": SWEEP_MIN_LOSS_DIVISOR,
                        "lr_stop_below": SWEEP_LR_STOP_BELOW,
                        "momentum": 0.0,
                        "factor": factor,
                    })

    # we load rerun list: configs that must not be skipped and should be run again (e.g. to save final params)
    rerun_file = _TABLE / "baseline_sweep_rerun.txt"
    rerun_set = set()
    if rerun_file.exists():
        for line in rerun_file.read_text().strip().splitlines():
            line = line.split("#")[0].strip()
            if line:
                rerun_set.add(line)

    for i, cfg in enumerate(configs, 1):
        out_name = cfg["name"]
        out_dir = RESULTS_BASELINE_SWEEP_DIR / out_name
        if (out_dir / "losses.json").exists() and out_name not in rerun_set:
            print(f"[{i}/{len(configs)}] skip (done): {out_name}")
            continue
        if out_name in rerun_set:
            print(f"[{i}/{len(configs)}] rerun (in rerun list): {out_name}")
        else:
            print(f"[{i}/{len(configs)}] {out_name}")
        try:
            train_baseline_sweep_one(cfg, out_dir)
        except Exception as e:
            print(f"  error: {e}")
            import traceback
            traceback.print_exc()
    print("Baseline sweep done.")


def run_baseline_sweep_expcos():
    """we run sweep: target = sum_{k=0}^{factor} cos(2^k pi x); factor 3 and 4 only; N = mult*2^factor with mult in {4,8,16} (Nyquist: highest mode 2^factor periods on [-1,1] => N >= 2*2^factor); batch_size [1,2,4,8,16], layers 1..2*factor; same lr/epochs as baseline sweep"""
    RESULTS_BASELINE_SWEEP_EXPCOS_DIR.mkdir(parents=True, exist_ok=True)
    for f in RESULTS_BASELINE_SWEEP_EXPCOS_DIR.rglob("epoch_info.json"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    lr_seq = _sweep_lr_sequence()
    print(f"BASELINE SWEEP EXPCOS: target = sum_{{k=0}}^{{factor}} cos(2^k pi x); factors {SWEEP_EXPCOS_FACTORS}; N = mult*2^factor with mult in {SWEEP_EXPCOS_N_MULTIPLIERS} (Nyquist: N >= 2*2^factor); batch_size {SWEEP_BATCH_SIZES}, layers 1..2*factor")
    print(f"  N_samples for factor 3: {[m * 2**3 for m in SWEEP_EXPCOS_N_MULTIPLIERS]}; factor 4: {[m * 2**4 for m in SWEEP_EXPCOS_N_MULTIPLIERS]}")
    print(f"Output: {RESULTS_BASELINE_SWEEP_EXPCOS_DIR}")

    configs = []
    for factor in SWEEP_EXPCOS_FACTORS:
        n_samples_for_factor = [m * (2 ** factor) for m in SWEEP_EXPCOS_N_MULTIPLIERS]
        layer_counts = list(range(1, 2 * factor + 1))
        for n_train in n_samples_for_factor:
            for batch_size in SWEEP_BATCH_SIZES:
                if batch_size > n_train:
                    continue
                for num_layers in layer_counts:
                    name = f"f{factor}_N{n_train}_bs{batch_size}_L{num_layers}"
                    configs.append({
                        "name": name,
                        "n_train": n_train,
                        "batch_size": batch_size,
                        "hidden_width": SWEEP_WIDTH,
                        "hidden_rank": SWEEP_RANK,
                        "num_layers": num_layers,
                        "num_epochs": SWEEP_NUM_EPOCHS_MAX,
                        "lr_sequence": lr_seq,
                        "lr_window": SWEEP_LR_WINDOW,
                        "min_epochs_before_reduce": SWEEP_LR_MIN_EPOCHS_BEFORE_REDUCE,
                        "min_loss_divisor": SWEEP_MIN_LOSS_DIVISOR,
                        "lr_stop_below": SWEEP_LR_STOP_BELOW,
                        "momentum": 0.0,
                        "factor": factor,
                        "target": "expcos",
                    })

    rerun_file = _TABLE / "baseline_sweep_expcos_rerun.txt"
    rerun_set = set()
    if rerun_file.exists():
        for line in rerun_file.read_text().strip().splitlines():
            line = line.split("#")[0].strip()
            if line:
                rerun_set.add(line)

    for i, cfg in enumerate(configs, 1):
        out_name = cfg["name"]
        out_dir = RESULTS_BASELINE_SWEEP_EXPCOS_DIR / out_name
        if (out_dir / "losses.json").exists() and out_name not in rerun_set:
            print(f"[{i}/{len(configs)}] skip (done): {out_name}")
            continue
        if out_name in rerun_set:
            print(f"[{i}/{len(configs)}] rerun (in rerun list): {out_name}")
        else:
            print(f"[{i}/{len(configs)}] {out_name}")
        try:
            train_baseline_sweep_one(cfg, out_dir)
        except Exception as e:
            print(f"  error: {e}")
            import traceback
            traceback.print_exc()
    print("Baseline sweep expcos done.")


def load_all_results():
    """we load all completed training results from RESULTS_DIR"""
    all_results = []
    for config_dir in sorted(RESULTS_DIR.iterdir()):
        if not config_dir.is_dir():
            continue
        results_file = config_dir / "results.json"
        config_file = config_dir / "config.json"
        checkpoint_file = config_dir / "checkpoint.pth"
        if not results_file.exists() or not config_file.exists():
            continue
        try:
            with open(results_file) as f:
                results = json.load(f)
            with open(config_file) as f:
                config = json.load(f)
            if checkpoint_file.exists():
                ckpt = torch.load(checkpoint_file, map_location="cpu")
                if ckpt.get("epoch", 0) < config.get("num_epochs", 0):
                    continue
            entry = {
                "config_name": config_dir.name,
                "freq_multiplier": config.get("freq_multiplier", 0),
                "rank": config.get("hidden_rank", 0),
                "layers": config.get("num_layers", 0),
                "epochs": config.get("num_epochs", 0),
                "final_train_error": results.get("final_train_error"),
                "final_test_error": results.get("final_test_error"),
                "final_test_error_max": results.get("final_test_error_max"),
                "training_time_seconds": results.get("training_time_seconds"),
                "total_parameters": results.get("total_parameters"),
                "epochs_run": results.get("epochs_run", 0),
                "thresholds_reached": len(results.get("thresholds_reached", [])),
            }
            all_losses = results.get("all_losses", [])
            if all_losses:
                entry["initial_loss"] = all_losses[0]
                entry["final_loss"] = all_losses[-1]
                entry["loss_reduction_factor"] = all_losses[0] / all_losses[-1] if all_losses[-1] > 0 else None
            all_results.append(entry)
        except Exception as e:
            print(f"Error loading {config_dir.name}: {e}")
    return all_results


def run_analysis():
    """we run analysis and write CSV + summary (depth and width scaling)"""
    all_results = load_all_results()
    if not all_results:
        print("No completed results found in", RESULTS_DIR)
        return

    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} completed configurations")

    # summaries by frequency, rank (width), layers (depth)
    freq_summary = df.groupby("freq_multiplier").agg({
        "final_test_error": ["mean", "std", "min", "max"],
        "final_test_error_max": ["mean", "std", "min", "max"],
        "training_time_seconds": ["mean", "sum"],
        "thresholds_reached": "mean",
        "config_name": "count",
    }).round(6)
    freq_summary.columns = ["_".join(c).strip() for c in freq_summary.columns.values]
    freq_summary = freq_summary.rename(columns={"config_name_count": "num_configs"})

    rank_summary = df.groupby("rank").agg({
        "final_test_error": ["mean", "std", "min", "max"],
        "final_test_error_max": ["mean", "std", "min", "max"],
        "training_time_seconds": ["mean", "sum"],
        "thresholds_reached": "mean",
        "config_name": "count",
    }).round(6)
    rank_summary.columns = ["_".join(c).strip() for c in rank_summary.columns.values]
    rank_summary = rank_summary.rename(columns={"config_name_count": "num_configs"})

    layer_summary = df.groupby("layers").agg({
        "final_test_error": ["mean", "std", "min", "max"],
        "final_test_error_max": ["mean", "std", "min", "max"],
        "training_time_seconds": ["mean", "sum"],
        "thresholds_reached": "mean",
        "config_name": "count",
    }).round(6)
    layer_summary.columns = ["_".join(c).strip() for c in layer_summary.columns.values]
    layer_summary = layer_summary.rename(columns={"config_name_count": "num_configs"})

    df.to_csv(ANALYSIS_CSV, index=False)
    ANALYSIS_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"  CSV -> {ANALYSIS_CSV}")

    with open(SUMMARY_TXT, "w") as f:
        f.write("FREQUENCY AND LAYER SCALING EXPERIMENTS - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total completed configurations: {len(df)}\n\n")
        f.write("FREQUENCY SUMMARY:\n")
        f.write(str(freq_summary) + "\n\n")
        f.write("RANK SUMMARY:\n")
        f.write(str(rank_summary) + "\n\n")
        f.write("LAYER SUMMARY:\n")
        f.write(str(layer_summary) + "\n")
    print(f"  Summary -> {SUMMARY_TXT}")
    print("Analysis done.")


def main():
    parser = argparse.ArgumentParser(
        description="Main experiments: stable baseline (cos 2pi x) and scaling law (depth/width/freq)."
    )
    parser.add_argument("--baseline", action="store_true", help="run stable baseline only (N=width=1024, SGD+AdaptiveStagnation)")
    parser.add_argument("--baseline-sweep", action="store_true", help="sweep sumcos: sum_{k=1}^{f} cos(2 pi k x), factor 1..5; N = base*factor")
    parser.add_argument("--baseline-sweep-expcos", action="store_true", help="sweep expcos: sum_{k=0}^{f} cos(2^k pi x), factor 3 and 4 only; N = mult*2^factor (Nyquist)")
    parser.add_argument("--train", action="store_true", help="run scaling-law training only")
    parser.add_argument("--analyze", action="store_true", help="run scaling-law analysis only")
    args = parser.parse_args()

    if args.baseline:
        run_baseline()
    elif args.baseline_sweep:
        run_baseline_sweep()
    elif args.baseline_sweep_expcos:
        run_baseline_sweep_expcos()
    elif args.train:
        run_training()
    elif args.analyze:
        run_analysis()
    else:
        run_training()
        run_analysis()


if __name__ == "__main__":
    main()
