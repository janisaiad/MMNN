#!/usr/bin/env python3
"""
we run depth and width scaling law experiments for the SinQuad function.
target: cos(36*pi*x^2) - 0.8*cos(17*pi*(x+0.5)^2) on [-2, 2]
width: 1024, ranks: 5, 10, 20
depths: 8, 10, 12, 15
lr: 1e-2 with AdaptiveStagnation (divide by 2 when stagnating)
goal: understand training stability for large depth/width
usage:
  python run_depth_width_scaling_sinquad.py           # run all configs
  python run_depth_width_scaling_sinquad.py --plot    # plot results only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# we add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from experiments.table.mmnn_vs import MMNN

# we set output directory
_TABLE = _REPO_ROOT / "experiments" / "table"
RESULTS_DIR = _TABLE / "results_sinquad_depth_width_scaling"

# we define constants
WIDTH = 1024  # width of hidden layers
RANKS = [5, 10, 20]  # ranks to test
DEPTHS = [8, 10, 12, 15]  # depths to test
INTERVAL = [-2, 2]  # domain
N_TRAIN = 2000  # number of training samples (proportional to domain size)
N_TEST = 500  # number of test samples
BATCH_SIZE = 64  # batch size
NUM_EPOCHS_MAX = 10_000  # max epochs

# we define lr schedule parameters (AdaptiveStagnation)
LR_INIT = 1e-2
LR_DIVISOR = 2
LR_N_STEPS = 25  # we generate 25 lr values
LR_STOP_BELOW = 1e-7  # we stop training when lr < this
LR_WINDOW = 10  # window for stagnation detection
LR_MIN_EPOCHS_BEFORE_REDUCE = 20  # min epochs before reducing lr
MIN_LOSS_DIVISOR = 1.2  # we save params when min_loss < init_loss/1.2^k

# we configure matplotlib for LaTeX-style plots
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["font.size"] = 14
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["savefig.dpi"] = 300


def target_sinquad(x):
    """we evaluate the SinQuad function: cos(36*pi*x^2) - 0.8*cos(17*pi*(x+0.5)^2)"""
    return np.cos(36 * np.pi * x**2) - 0.8 * np.cos(17 * np.pi * (x + 0.5)**2)


def _lr_sequence():
    """we build lr sequence: start LR_INIT, divide by LR_DIVISOR each step"""
    return [LR_INIT / (LR_DIVISOR ** i) for i in range(LR_N_STEPS)]


def train_one_config(config, output_dir):
    """we train one config with AdaptiveStagnation and checkpoint saves at min_loss/1.2^k"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    
    n_train = config["n_train"]
    n_test = config["n_test"]
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
    interval = config["interval"]
    
    print(f"\n{'='*80}")
    print(f"Training: rank={hidden_rank}, depth={num_layers}, width={hidden_width}")
    print(f"N_train={n_train}, batch_size={batch_size}, interval={interval}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    # we set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # we build model
    ranks = [1] + [hidden_rank] * num_layers + [1]
    widths = [hidden_width] * (num_layers + 1)
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}, trainable: {trainable_params:,}")
    
    # we generate training data
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_sinquad(x_train)
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we generate test data
    x_test = np.linspace(interval[0], interval[1], n_test)
    y_test = target_sinquad(x_test)
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we set optimizer
    lr_init = lr_sequence[0]
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
    
    # we track training
    current_lr_index = 0
    last_reduction_epoch = -1
    lr_reduction_epochs = []
    
    all_losses = []
    all_lrs = []
    all_test_losses = []
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
        
        # we compute test loss periodically
        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                y_pred_test = model(x_test_tensor)
                test_loss = nn.MSELoss()(y_pred_test, y_test_tensor).item()
                all_test_losses.append({"epoch": epoch, "loss": float(test_loss)})
        
        if init_loss is None:
            init_loss = epoch_loss
        
        # we save params at min_loss / 1.2^k thresholds
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
        
        # we apply AdaptiveStagnation: reduce lr when stagnating
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
        if current_lr < LR_STOP_BELOW:
            print(f"  stopping: lr={current_lr:.2e} < {LR_STOP_BELOW:.0e}")
            break
        
        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{num_epochs} loss={epoch_loss:.4e} min_loss={min_loss_so_far:.4e} lr={current_lr:.6f}")
    
    training_time = time.time() - start_time
    
    # we evaluate final test error
    with torch.no_grad():
        y_pred = model(x_test_tensor)
        final_test_error = nn.MSELoss()(y_pred, y_test_tensor).item()
        final_test_error_max = torch.max(torch.abs(y_pred - y_test_tensor)).item()
    
    # we save results
    losses_payload = {
        "config": config,
        "final_train_error": float(all_losses[-1]) if all_losses else None,
        "final_test_error": float(final_test_error),
        "final_test_error_max": float(final_test_error_max),
        "training_time_seconds": float(training_time),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "epochs_run": len(all_losses),
        "all_losses": [float(x) for x in all_losses],
        "all_lrs": [float(x) for x in all_lrs],
        "all_test_losses": all_test_losses,
        "lr_reduction_epochs": list(lr_reduction_epochs),
        "init_loss": float(init_loss) if init_loss is not None else None,
        "min_loss_checkpoints": min_loss_checkpoints,
    }
    torch.save(model.state_dict(), output_dir / "model_parameters.pth")
    with open(output_dir / "losses.json", "w") as f:
        json.dump(losses_payload, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # we plot loss curve
    plot_loss_curve(output_dir, all_losses, lr_reduction_epochs, config)
    
    # we plot final prediction
    plot_final_prediction(output_dir, model, config, device, mydtype)
    
    print(f"  completed: {output_dir.name} test_err={final_test_error:.4e} min_loss_saves={len(min_loss_checkpoints)}")
    return losses_payload


def plot_loss_curve(output_dir, all_losses, lr_reduction_epochs, config):
    """we plot loss curve with lr reduction markers"""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = list(range(1, len(all_losses) + 1))
    ax.semilogy(epochs, all_losses, "b-", linewidth=1.5, label="Train loss")
    
    # we mark lr reduction epochs with vertical lines
    for epoch in lr_reduction_epochs:
        ax.axvline(epoch, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss (log scale)", fontsize=14)
    title = f"rank={config['hidden_rank']}, L={config['num_layers']}, width={config['hidden_width']}"
    ax.set_title(f"Training Loss\n{title}", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_final_prediction(output_dir, model, config, device, mydtype):
    """we plot final prediction vs true function"""
    interval = config["interval"]
    x_plot = np.linspace(interval[0], interval[1], 1000)
    x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
    y_plot_true = target_sinquad(x_plot)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_plot, y_plot_true, "b-", label="True function", linewidth=2)
    ax.plot(x_plot, y_plot_nn, "r--", label="Learned network", linewidth=1.5)
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$f(x)$", fontsize=14)
    title = f"rank={config['hidden_rank']}, L={config['num_layers']}, width={config['hidden_width']}"
    ax.set_title(f"Final Prediction\n{title}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "final_prediction.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_all():
    """we run all depth x rank configurations"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lr_seq = _lr_sequence()
    
    print(f"SINQUAD DEPTH/WIDTH SCALING EXPERIMENT")
    print(f"Target: cos(36*pi*x^2) - 0.8*cos(17*pi*(x+0.5)^2) on {INTERVAL}")
    print(f"Width: {WIDTH}, Ranks: {RANKS}, Depths: {DEPTHS}")
    print(f"N_train: {N_TRAIN}, batch_size: {BATCH_SIZE}")
    print(f"LR: start {LR_INIT}, divide by {LR_DIVISOR} until {LR_STOP_BELOW}")
    print(f"Output: {RESULTS_DIR}")
    print(f"{'='*80}")
    
    configs = []
    for rank in RANKS:
        for depth in DEPTHS:
            name = f"rank{rank}_L{depth}_W{WIDTH}"
            configs.append({
                "name": name,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "batch_size": BATCH_SIZE,
                "hidden_width": WIDTH,
                "hidden_rank": rank,
                "num_layers": depth,
                "num_epochs": NUM_EPOCHS_MAX,
                "lr_sequence": lr_seq,
                "lr_window": LR_WINDOW,
                "min_epochs_before_reduce": LR_MIN_EPOCHS_BEFORE_REDUCE,
                "min_loss_divisor": MIN_LOSS_DIVISOR,
                "lr_stop_below": LR_STOP_BELOW,
                "momentum": 0.0,
                "interval": INTERVAL,
            })
    
    print(f"Total configs: {len(configs)}")
    
    for i, cfg in enumerate(configs, 1):
        out_name = cfg["name"]
        out_dir = RESULTS_DIR / out_name
        if (out_dir / "losses.json").exists():
            print(f"[{i}/{len(configs)}] skip (done): {out_name}")
            continue
        print(f"[{i}/{len(configs)}] {out_name}")
        try:
            train_one_config(cfg, out_dir)
        except Exception as e:
            print(f"  error: {e}")
            import traceback
            traceback.print_exc()
    
    print("All configs done.")
    
    # we generate summary
    generate_summary()


def generate_summary():
    """we generate summary of all results"""
    summary_data = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        losses_file = run_dir / "losses.json"
        if not losses_file.exists():
            continue
        with open(losses_file) as f:
            data = json.load(f)
        cfg = data["config"]
        summary_data.append({
            "name": cfg["name"],
            "rank": cfg["hidden_rank"],
            "depth": cfg["num_layers"],
            "width": cfg["hidden_width"],
            "final_train_error": data["final_train_error"],
            "final_test_error": data["final_test_error"],
            "epochs_run": data["epochs_run"],
            "training_time_seconds": data["training_time_seconds"],
            "n_lr_reductions": len(data["lr_reduction_epochs"]),
            "n_checkpoints": len(data["min_loss_checkpoints"]),
        })
    
    # we save summary
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    # we print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Name':<25} {'Rank':<6} {'Depth':<6} {'Train Err':<12} {'Test Err':<12} {'Epochs':<8} {'Time(s)':<10}")
    print("-"*100)
    for row in summary_data:
        train_err = f"{row['final_train_error']:.4e}" if row['final_train_error'] else "N/A"
        test_err = f"{row['final_test_error']:.4e}" if row['final_test_error'] else "N/A"
        print(f"{row['name']:<25} {row['rank']:<6} {row['depth']:<6} {train_err:<12} {test_err:<12} {row['epochs_run']:<8} {row['training_time_seconds']:<10.1f}")
    print("="*100)
    
    # we plot summary heatmaps
    plot_summary_heatmaps(summary_data)


def plot_summary_heatmaps(summary_data):
    """we plot heatmaps of test error by rank x depth"""
    import pandas as pd
    
    df = pd.DataFrame(summary_data)
    
    # we create heatmap of final test error
    pivot_test = df.pivot(index="depth", columns="rank", values="final_test_error")
    pivot_train = df.pivot(index="depth", columns="rank", values="final_train_error")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # we plot test error heatmap
    ax = axes[0]
    im = ax.imshow(np.log10(pivot_test.values + 1e-12), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot_test.columns)))
    ax.set_xticklabels(pivot_test.columns)
    ax.set_yticks(range(len(pivot_test.index)))
    ax.set_yticklabels(pivot_test.index)
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Depth", fontsize=12)
    ax.set_title("log10(Test Error)", fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    # we add values to cells
    for i in range(len(pivot_test.index)):
        for j in range(len(pivot_test.columns)):
            val = pivot_test.values[i, j]
            if val is not None and not np.isnan(val):
                ax.text(j, i, f"{val:.1e}", ha="center", va="center", fontsize=9, color="white")
    
    # we plot train error heatmap
    ax = axes[1]
    im = ax.imshow(np.log10(pivot_train.values + 1e-12), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot_train.columns)))
    ax.set_xticklabels(pivot_train.columns)
    ax.set_yticks(range(len(pivot_train.index)))
    ax.set_yticklabels(pivot_train.index)
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Depth", fontsize=12)
    ax.set_title("log10(Train Error)", fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    # we add values to cells
    for i in range(len(pivot_train.index)):
        for j in range(len(pivot_train.columns)):
            val = pivot_train.values[i, j]
            if val is not None and not np.isnan(val):
                ax.text(j, i, f"{val:.1e}", ha="center", va="center", fontsize=9, color="white")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "summary_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved summary heatmaps to {RESULTS_DIR / 'summary_heatmaps.png'}")


def plot_only():
    """we regenerate plots from existing losses.json files"""
    print("Regenerating plots from existing results...")
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        losses_file = run_dir / "losses.json"
        if not losses_file.exists():
            continue
        with open(losses_file) as f:
            data = json.load(f)
        cfg = data["config"]
        all_losses = data["all_losses"]
        lr_reduction_epochs = data["lr_reduction_epochs"]
        
        # we replot loss curve
        plot_loss_curve(run_dir, all_losses, lr_reduction_epochs, cfg)
        print(f"  plotted: {run_dir.name}")
    
    # we regenerate summary
    generate_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SinQuad depth/width scaling experiments")
    parser.add_argument("--plot", action="store_true", help="regenerate plots only")
    args = parser.parse_args()
    
    if args.plot:
        plot_only()
    else:
        run_all()
