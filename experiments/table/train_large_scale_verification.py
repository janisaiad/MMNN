#!/usr/bin/env python3
"""
Large scale run to verify: loss = g(L/freq) with optimal range 7-12
We test many L values for each frequency to build the precise curve
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# we configure matplotlib
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['font.size'] = 18
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

def generate_configs():
    """we generate configurations to verify L/freq optimal range 7-20 with very dense coverage"""
    configs = []
    
    # we focus on frequencies where we can test L/freq in range 7-20 (expanded from 7-12)
    # We want VERY DENSE coverage to get a smooth curve
    
    # we use more frequencies for better coverage
    freq_multipliers = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    ranks = [10, 15, 25]
    batch_size = 100
    
    for freq_mult in freq_multipliers:
        # we compute L range to cover L/freq from 7 to 20 (the optimal range)
        L_min = max(3, int(np.ceil(7 * freq_mult)))  # start from L/freq = 7
        L_max = int(np.ceil(20 * freq_mult))  # go up to L/freq = 20
        
        # we generate L values VERY DENSELY for smooth curve
        if freq_mult <= 0.5:
            # for very low frequencies, test every layer
            L_values = list(range(L_min, min(L_max + 1, 30), 1))
        elif freq_mult <= 1.0:
            # for low-medium frequencies, test every layer
            L_values = list(range(L_min, min(L_max + 1, 50), 1))
        elif freq_mult <= 2.0:
            # for medium frequencies, test every layer
            L_values = list(range(L_min, min(L_max + 1, 60), 1))
        else:
            # for higher frequencies, test every 1-2 layers
            L_values = list(range(L_min, min(L_max + 1, 80), 1))
        
        # we ensure optimal range 7-20 is VERY DENSELY covered
        # test every 0.25 in ratio for smoothness
        for target_ratio in np.arange(7, 20.25, 0.25):  # every 0.25 in ratio
            target_L = int(np.round(target_ratio * freq_mult))
            if target_L >= 3 and target_L <= 100 and target_L not in L_values:
                L_values.append(target_L)
        
        # we also add some values below 7 and above 20 for context
        # below 7: add a few points
        for target_ratio in np.arange(4, 7, 0.5):
            target_L = int(np.round(target_ratio * freq_mult))
            if target_L >= 3 and target_L <= 100 and target_L not in L_values:
                L_values.append(target_L)
        
        # above 20: add a few points
        for target_ratio in np.arange(20, 25, 1.0):
            target_L = int(np.round(target_ratio * freq_mult))
            if target_L >= 3 and target_L <= 100 and target_L not in L_values:
                L_values.append(target_L)
        
        L_values = sorted(set(L_values))
        
        # we compute epochs
        for rank in ranks:
            for num_layers in L_values:
                num_epochs = min(int(2 * freq_mult * 10000), 200000)
                num_epochs = max(num_epochs, 5000)
                
                config = {
                    'freq_multiplier': freq_mult,
                    'hidden_rank': rank,
                    'num_layers': num_layers,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'L_over_freq': num_layers / freq_mult,
                }
                configs.append(config)
    
    return configs

def target_function(x, freq_multiplier):
    """we create target function with frequency multiplier"""
    base_freqs = [12, 24, 36, 72]
    result = np.zeros_like(x)
    for base_freq in base_freqs:
        freq = base_freq * freq_multiplier
        if base_freq in [24, 72]:
            result += np.cos(freq * np.pi * x + 0.5)
        else:
            result += np.cos(freq * np.pi * x)
    return result

def train_one_config(config, output_dir, target_func):
    """we train one configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    
    # we set up model
    hidden_width = 777
    hidden_rank = config['hidden_rank']
    num_layers = config['num_layers']
    input_rank = 1
    output_rank = 1
    
    ranks = [input_rank] + [hidden_rank] * num_layers + [output_rank]
    widths = [hidden_width] * (num_layers + 1)
    
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=False,
        fixWb=True
    )
    
    # we create training data
    interval = [-1, 1]
    n_train = 5000
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_func(x_train, config['freq_multiplier'])
    
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we create test data
    n_test = 1000
    x_test = np.linspace(interval[0], interval[1], n_test)
    y_test = target_func(x_test, config['freq_multiplier'])
    
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we set up optimizer
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    
    # we check for checkpoint
    checkpoint_path = output_dir / "checkpoint.pth"
    start_epoch = 0
    all_losses = []
    errors_train = []
    errors_test = []
    errors_test_max = []
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        all_losses = checkpoint.get("all_losses", [])
        errors_train = checkpoint.get("errors_train", [])
        errors_test = checkpoint.get("errors_test", [])
        errors_test_max = checkpoint.get("errors_test_max", [])
    
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = x_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.MSELoss()(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        epoch_loss /= (n_train // batch_size + 1)
        all_losses.append(epoch_loss)
        
        # we evaluate every 50 epochs
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                y_pred_test = model(x_test_tensor)
                error_test = nn.MSELoss()(y_pred_test, y_test_tensor).item()
                error_test_max = torch.max(torch.abs(y_pred_test - y_test_tensor)).item()
                
                y_pred_train = model(x_train_tensor)
                error_train = nn.MSELoss()(y_pred_train, y_train_tensor).item()
                
                errors_train.append(error_train)
                errors_test.append(error_test)
                errors_test_max.append(error_test_max)
        
        # we save checkpoint every 500 epochs
        if epoch % 500 == 0 and epoch > 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "all_losses": all_losses,
                "errors_train": errors_train,
                "errors_test": errors_test,
                "errors_test_max": errors_test_max,
            }
            torch.save(checkpoint, checkpoint_path)
    
    training_time = time.time() - start_time
    
    # we save final checkpoint
    checkpoint = {
        "epoch": config["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "all_losses": all_losses,
        "errors_test": errors_test,
        "errors_test_max": errors_test_max,
    }
    torch.save(checkpoint, checkpoint_path)
    
    # we save results
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results = {
        "config": config,
        "final_train_error": float(errors_train[-1]) if errors_train else None,
        "final_test_error": float(errors_test[-1]) if errors_test else None,
        "final_test_error_max": float(errors_test_max[-1]) if errors_test_max else None,
        "training_time_seconds": float(training_time),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "epochs_run": int(len(all_losses)),
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
    
    # we plot loss evolution
    fig = plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1.5)
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss (log scale)', fontsize=22)
    config_str = f"freq×{config['freq_multiplier']:.2f}, rank={config['hidden_rank']}, L={config['num_layers']}, L/freq={config['L_over_freq']:.2f}"
    plt.title(f'Training Loss Evolution\n{config_str}', fontsize=20)
    plt.grid(True, alpha=0.3, which='both')
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ completed: {output_dir.name}")
    return results

def main():
    """we run large scale verification"""
    print("="*80)
    print("LARGE SCALE VERIFICATION: loss = g(L/freq) with optimal range 7-12")
    print("="*80)
    
    configs = generate_configs()
    print(f"\nTotal configurations: {len(configs)}")
    
    # we show frequency distribution
    freqs = sorted(set(c['freq_multiplier'] for c in configs))
    print(f"\nFrequencies: {freqs}")
    print(f"Ranks: [10, 15, 25]")
    
    # we show L/freq coverage
    print(f"\nL/freq ratio coverage per frequency:")
    for freq in freqs:
        freq_configs = [c for c in configs if c['freq_multiplier'] == freq]
        ratios = sorted(set(c['L_over_freq'] for c in freq_configs))
        optimal_count = sum(1 for r in ratios if 7 <= r <= 12)
        print(f"  freq×{freq:.2f}: {len(ratios)} ratios, {optimal_count} in range 7-12, "
              f"range=[{ratios[0]:.1f}, {ratios[-1]:.1f}]")
    
    results_dir = Path("experiments/table/results_large_scale_verification")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    target_function_lambda = target_function
    
    for i, config in enumerate(configs, 1):
        freq_mult = config['freq_multiplier']
        rank = config['hidden_rank']
        num_layers = config['num_layers']
        L_over_freq = config['L_over_freq']
        
        output_dir_name = f"freq{freq_mult:.2f}_rank{rank}_L{num_layers}_ratio{L_over_freq:.2f}"
        output_dir = results_dir / output_dir_name
        
        print("\n" + "="*80)
        print(f"Configuration {i}/{len(configs)}")
        print(f"Freq: {freq_mult:.2f}, Rank: {rank}, Layers: {num_layers}, L/freq: {L_over_freq:.2f}")
        print(f"Epochs: {config['num_epochs']}")
        print(f"{'='*80}")
        
        # we check if already completed
        checkpoint_file = output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            epoch = ckpt.get('epoch', 0)
            if epoch >= config['num_epochs']:
                print(f"✓ Already completed, skipping...")
                continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = train_one_config(config, output_dir, target_function_lambda)
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("LARGE SCALE VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
