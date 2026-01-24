#!/usr/bin/env python3
"""
we test frequency and layer scaling laws
baseline: R=15, bs=100, 8 layers
frequency multipliers: 0.3, 0.6, 1.5, 2, 3, 5, 7, 10
for each factor: multiply frequencies, use approximately factor*8 layers (rounded up), and test nearby layers
epochs: 2 * factor * 10k
ranks: 10, 15, 25
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
import math

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


def target_function(x, freq_multiplier=1.0):
    """we define the multi-frequency function with phase shifts, scaled by freq_multiplier"""
    base_freqs = [12, 24, 36, 72]
    scaled_freqs = [f * freq_multiplier for f in base_freqs]
    return (np.cos(scaled_freqs[0] * np.pi * x) + 
            np.cos(scaled_freqs[1] * np.pi * x + 0.5) + 
            np.cos(scaled_freqs[2] * np.pi * x) + 
            np.cos(scaled_freqs[3] * np.pi * x + 0.5))


def generate_configs():
    """we generate configurations for frequency and layer scaling"""
    configs = []
    
    # baseline parameters
    base_layers = 8
    base_ranks = [10, 15, 25]  # we test ranks 10, 15, 25
    batch_size = 100
    fixWb = False  # baseline fixWb
    
    # frequency multipliers
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
        # we calculate base layer count (rounded up)
        base_layer_count = math.ceil(freq_mult * base_layers)
        
        # we determine nearby layer counts for diagonal testing
        # for each factor, test approximately factor*8 layers and nearby ones
        if freq_mult == 0.3:
            # test 0.3L and 0.6L
            layer_counts = [math.ceil(0.3 * base_layers), math.ceil(0.6 * base_layers)]  # 3, 5
        elif freq_mult == 0.6:
            # test 0.3L, 0.6L, and maybe 1.5*0.6L
            layer_counts = [math.ceil(0.3 * base_layers), math.ceil(0.6 * base_layers), math.ceil(1.5 * 0.6 * base_layers)]  # 3, 5, 8
        elif freq_mult == 1.5:
            # test 0.6*1.5L, 1.5L, 2*1.5L
            layer_counts = [math.ceil(0.6 * 1.5 * base_layers), math.ceil(1.5 * base_layers), math.ceil(2 * 1.5 * base_layers)]  # 8, 12, 24
        elif freq_mult == 2:
            # test 1.5L and 3L (as user specified)
            layer_counts = [math.ceil(1.5 * base_layers), math.ceil(3 * base_layers)]  # 12, 24
        elif freq_mult == 3:
            # test 2*3L and 5*3L (diagonal)
            layer_counts = [math.ceil(2 * base_layers), math.ceil(5 * base_layers)]  # 16, 40
        elif freq_mult == 5:
            # test 3*5L and 7*5L (diagonal)
            layer_counts = [math.ceil(3 * base_layers), math.ceil(7 * base_layers)]  # 24, 56
        elif freq_mult == 7:
            # test 5*7L and 10*7L (diagonal)
            layer_counts = [math.ceil(5 * base_layers), math.ceil(10 * base_layers)]  # 40, 80
        elif freq_mult == 10:
            # test 7*10L and 10L (diagonal)
            layer_counts = [math.ceil(7 * base_layers), math.ceil(10 * base_layers)]  # 56, 80
        else:
            layer_counts = [base_layer_count]  # fallback
        
        # we ensure base layer count is included
        if base_layer_count not in layer_counts:
            layer_counts.append(base_layer_count)
        layer_counts = sorted(set(layer_counts))  # we remove duplicates and sort
        
        # we calculate epochs: 2 * factor * 10k
        num_epochs = int(2 * freq_mult * 10000)
        
        # we create function string
        base_freqs = [12, 24, 36, 72]
        scaled_freqs = [f * freq_mult for f in base_freqs]
        func_str = f"cos({scaled_freqs[0]}*pi*x) + cos({scaled_freqs[1]}*pi*x + 0.5) + cos({scaled_freqs[2]}*pi*x) + cos({scaled_freqs[3]}*pi*x + 0.5)"
        
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
    
    arch_label = "FULL_RANK" if config['hidden_rank'] == config['hidden_width'] else f"rank={config['hidden_rank']}"
    print(f"\n{'='*80}")
    print(f"Training: {arch_label}, fixWb={config['fixWb']}, layers={config['num_layers']}")
    print(f"Frequency multiplier: {config['freq_multiplier']}")
    print(f"Function: {config['function']}")
    print(f"Epochs: {config['num_epochs']}")
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
    y_train = target_func(x_train, config["freq_multiplier"])
    x_train = torch.tensor(x_train, device=device, dtype=mydtype)
    y_train = torch.tensor(y_train, device=device, dtype=mydtype)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # we create test data
    x_test = np.random.rand(config["num_test_samples"]) * 2 - 1
    y_test = target_func(x_test, config["freq_multiplier"])
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
    
    # we track which loss thresholds have been reached (sorted descending: largest to smallest)
    loss_thresholds = [
        1e-1, 5e-2, 2e-2, 1e-2,
        5e-3, 2e-3, 1e-3,
        5e-4, 2e-4, 1e-4,
        5e-5, 2e-5, 1e-5,
        5e-6, 2e-6, 1e-6,
        5e-7, 2e-7, 1e-7,
        5e-8, 2e-8, 1e-8,
        5e-9, 2e-9, 1e-9
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
                threshold_dir = output_dir / f"model_at_loss_{threshold:.0e}"
                threshold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), threshold_dir / "model_parameters.pth")
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
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    torch.save(model.state_dict(), output_dir / "model_parameters.pth")
    
    # we plot final prediction
    x_plot = np.linspace(*config["interval"], 1000)
    x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
    y_plot_true = target_func(x_plot, config["freq_multiplier"])
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot_true, 'b-', label='True function', linewidth=2)
    plt.plot(x_plot, y_plot_nn, 'r--', label='Learned network', linewidth=2)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$f(x)$', fontsize=22)
    arch_label = "FULL_RANK" if config['hidden_rank'] == config['hidden_width'] else f"rank={config['hidden_rank']}"
    config_str = f"{arch_label}, L={config['num_layers']}, freq×{config['freq_multiplier']}, epoch {len(all_losses)}"
    plt.title(f'Final Prediction\n{config_str}', fontsize=20)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / 'final_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # we plot loss evolution (NO vertical red lines as requested)
    fig = plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1.5)
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
    """we run frequency and layer scaling benchmark"""
    print("="*80)
    print("FREQUENCY AND LAYER SCALING BENCHMARK")
    print("Baseline: R=15, bs=100, 8 layers")
    print("Frequency multipliers: 0.3, 0.6, 1.5, 2, 3, 5, 7, 10")
    print("Ranks: 10, 15, 25")
    print("="*80)
    
    configs = generate_configs()
    print(f"\ngenerated {len(configs)} configurations")
    
    # we create output directory
    base_output_dir = Path("experiments/table/results_frequency_layer_scaling")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {base_output_dir}")
    print(f"Total configurations: {len(configs)}\n")
    
    # we run each configuration
    for i, config in enumerate(configs, 1):
        freq_mult = config['freq_multiplier']
        rank = config['hidden_rank']
        num_layers = config['num_layers']
        output_dir_name = f"freq{freq_mult}_rank{rank}_L{num_layers}"
        output_dir = base_output_dir / output_dir_name
        
        print(f"\n{'='*80}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"Freq multiplier: {freq_mult}, Rank: {rank}, Layers: {num_layers}")
        print(f"Epochs: {config['num_epochs']}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}")
        
        # we check if already completed
        checkpoint_file = output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            epoch = ckpt.get('epoch', 0)
            if epoch >= config['num_epochs']:
                print(f"✓ Already completed (epoch {epoch}/{config['num_epochs']}), skipping...")
                continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = train_one_config(config, output_dir, target_function)
            print(f"✓ Completed: {output_dir_name}")
        except Exception as e:
            print(f"✗ Error in {output_dir_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("FREQUENCY AND LAYER SCALING BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
