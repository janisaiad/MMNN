#!/usr/bin/env python3
"""
we train MMNN on multi-frequency function: cos(12πx) + cos(24πx + 0.5) + cos(36πx) + cos(72πx + 0.5)
we save model parameters every time loss goes below thresholds: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, etc.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN

# we configure matplotlib for LaTeX formatting
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'normal'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True


def target_function(x):
    """we define the multi-frequency function with phase shifts"""
    return np.cos(12 * np.pi * x) + np.cos(24 * np.pi * x + 0.5) + np.cos(36 * np.pi * x) + np.cos(72 * np.pi * x + 0.5)


def generate_configs():
    """we generate configurations to test"""
    configs = []
    
    # we test different ranks and fixWb options
    ranks = [10, 15, 20, 25, 50, 100]
    fixWb_options = [False, True]
    
    base_config = {
        "num_layers": 8,
        "hidden_width": 777,
        "input_rank": 1,
        "output_rank": 1,
        "use_resnet": False,
        "num_epochs": 30000,
        "batch_size": 100,
        "lr_init": 0.001,
        "lr_gamma": 0.9,
        "lr_step_size": 100,
        "interval": [-1, 1],
        "show_plot": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": "torch.float32",
        "function": "cos(12*pi*x) + cos(24*pi*x + 0.5) + cos(36*pi*x) + cos(72*pi*x + 0.5)",
        "num_training_samples": 5000,  # we use more samples for multi-frequency
        "num_test_samples": 6000,
    }
    
    for rank in ranks:
        for fixWb in fixWb_options:
            config = base_config.copy()
            config.update({
                "hidden_rank": rank,
                "fixWb": fixWb,
            })
            configs.append(config)
    
    return configs


def train_one_config(config, output_dir):
    """we train one configuration with loss threshold checkpoints"""
    device = torch.device(config["device"])
    mydtype = torch.get_default_dtype()
    
    arch_label = "FULL_RANK" if config['hidden_rank'] == config['hidden_width'] else f"rank={config['hidden_rank']}"
    print(f"\n{'='*80}")
    print(f"Training: {arch_label}, fixWb={config['fixWb']}")
    print(f"Function: {config['function']}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    # we set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # we build model
    ranks = [config["input_rank"]] + [config["hidden_rank"]] * config["num_layers"] + [config["output_rank"]]
    widths = [config["hidden_width"]] * (config["num_layers"] + 1)
    
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=config["use_resnet"],
        fixWb=config["fixWb"]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    
    # we create datasets
    x_train = np.linspace(*config["interval"], config["num_training_samples"]).reshape([-1, 1])
    y_train = target_function(x_train)
    x_train = torch.tensor(x_train, device=device, dtype=mydtype)
    y_train = torch.tensor(y_train, device=device, dtype=mydtype)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # we create test data
    x_test = np.random.rand(config["num_test_samples"]) * 2 - 1
    y_test = target_function(x_test)
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we setup training
    optimizer = optim.Adam(model.parameters(), lr=config["lr_init"])
    scheduler = StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])
    criterion = nn.MSELoss()
    
    # we check for existing checkpoint
    checkpoint_path = output_dir / "checkpoint.pth"
    start_epoch = 1
    all_losses = []
    errors_train = []
    errors_test = []
    errors_test_max = []
    
    # we track which loss thresholds have been reached
    loss_thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    thresholds_reached = set()  # we track which thresholds we've already saved
    
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
        print(f"thresholds already reached: {sorted(thresholds_reached)}")
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
        
        # we check if we've crossed any loss thresholds and save model
        for threshold in loss_thresholds:
            if threshold not in thresholds_reached and avg_loss < threshold:
                thresholds_reached.add(threshold)
                # we save model at this threshold
                threshold_dir = output_dir / f"model_at_loss_{threshold:.0e}"
                threshold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), threshold_dir / "model_parameters.pth")
                # we also save epoch info
                with open(threshold_dir / "epoch_info.json", "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "loss": float(avg_loss),
                        "threshold": float(threshold)
                    }, f, indent=4)
                print(f"✓ Loss {avg_loss:.6e} < {threshold:.0e} at epoch {epoch} - model saved to {threshold_dir.name}")
        
        # we evaluate on test set
        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                y_pred = model(x_test_tensor)
                test_error = criterion(y_pred, y_test_tensor).item()
                test_error_max = torch.max(torch.abs(y_pred - y_test_tensor)).item()
                
                errors_train.append(avg_loss)
                errors_test.append(test_error)
                errors_test_max.append(test_error_max)
                
                print(f"Epoch {epoch}/{config['num_epochs']}: "
                      f"train={avg_loss:.4e}, test={test_error:.4e}, test_max={test_error_max:.4e}, "
                      f"thresholds_reached={len(thresholds_reached)}/{len(loss_thresholds)}")
        
        # we save checkpoint every 500 epochs
        if epoch % 500 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "all_losses": all_losses,
                "errors_train": errors_train,
                "errors_test": errors_test,
                "errors_test_max": errors_test_max,
                "thresholds_reached": list(thresholds_reached),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"checkpoint saved at epoch {epoch}")
    
    training_time = time.time() - start_time
    
    # we save final checkpoint
    checkpoint = {
        "epoch": config["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "all_losses": all_losses,
        "errors_train": errors_train,
        "errors_test": errors_test,
        "errors_test_max": errors_test_max,
        "thresholds_reached": list(thresholds_reached),
    }
    torch.save(checkpoint, checkpoint_path)
    
    # we save results
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
        "all_losses": [float(l) for l in all_losses],
        "errors_train": [float(e) for e in errors_train],
        "errors_test": [float(e) for e in errors_test],
        "errors_test_max": [float(e) for e in errors_test_max],
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # we save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # we save final model
    torch.save(model.state_dict(), output_dir / "model_parameters.pth")
    
    # we plot final prediction
    x_plot = np.linspace(*config["interval"], 1000)
    x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
    y_plot_true = target_function(x_plot)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot_true, 'b-', label='True function', linewidth=2)
    plt.plot(x_plot, y_plot_nn, 'r--', label='Learned network', linewidth=2)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$f(x)$', fontsize=22)
    arch_label = "FULL_RANK" if config['hidden_rank'] == config['hidden_width'] else f"rank={config['hidden_rank']}"
    config_str = f"{arch_label}, fixWb={config['fixWb']}, epoch {len(all_losses)}"
    plt.title(f'Final Prediction\n{config_str}', fontsize=20)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / 'final_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # we plot loss evolution
    fig = plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1.5)
    # we mark threshold crossings
    for threshold in sorted(thresholds_reached):
        # we find first epoch where loss < threshold
        for i, loss in enumerate(all_losses):
            if loss < threshold:
                plt.axvline(x=i+1, color='r', linestyle='--', alpha=0.5, linewidth=1)
                plt.text(i+1, threshold, f'  {threshold:.0e}', rotation=90, fontsize=14, va='bottom')
                break
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss (log scale)', fontsize=22)
    plt.title(f'Training Loss Evolution\n{config_str}', fontsize=20)
    plt.grid(True, alpha=0.3, which='both')
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ completed: {output_dir.name}")
    print(f"  Thresholds reached: {sorted(thresholds_reached)}")
    return results


def main():
    """we run multi-frequency benchmark"""
    print("="*80)
    print("MULTI-FREQUENCY BENCHMARK")
    print("Function: cos(12πx) + cos(24πx + 0.5) + cos(36πx) + cos(72πx + 0.5)")
    print("="*80)
    
    configs = generate_configs()
    print(f"\ngenerated {len(configs)} configurations")
    print(f"  ranks: {sorted(set(c['hidden_rank'] for c in configs))}")
    print(f"  fixWb: True, False")
    print(f"  epochs: 30,000")
    print(f"  loss thresholds: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9")
    
    # we create output directory
    base_output_dir = Path("experiments/table/results_multi_frequency_benchmark")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {base_output_dir}")
    print(f"Total configurations: {len(configs)}\n")
    
    # we run each configuration
    for i, config in enumerate(configs, 1):
        rank = config['hidden_rank']
        fixWb = config['fixWb']
        output_dir_name = f"rank{rank}_fixWb{fixWb}"
        output_dir = base_output_dir / output_dir_name
        
        print(f"\n{'='*80}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"Rank: {rank}, fixWb: {fixWb}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}")
        
        # we check if already completed
        checkpoint_file = output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            epoch = ckpt.get('epoch', 0)
            if epoch >= 30000:
                print(f"✓ Already completed (epoch {epoch}/30000), skipping...")
                continue
        
        # we create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # we train
        try:
            results = train_one_config(config, output_dir)
            print(f"✓ Completed: {output_dir_name}")
        except Exception as e:
            print(f"✗ Error in {output_dir_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("MULTI-FREQUENCY BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
