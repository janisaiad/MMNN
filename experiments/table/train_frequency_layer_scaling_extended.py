#!/usr/bin/env python3
"""
Extended frequency and layer scaling experiments
We test frequency multipliers in log space from 0.3 to 100
Using the scaling law: L ≈ round(5.16 × freq + 2.55)
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
plt.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

def generate_freq_multipliers_log_space(min_freq=0.05, max_freq=500, num_points=40):
    """we generate frequency multipliers in log space"""
    # we use log space from min_freq to max_freq
    log_min = np.log10(min_freq)
    log_max = np.log10(max_freq)
    log_freqs = np.linspace(log_min, log_max, num_points)
    freqs = 10 ** log_freqs
    
    # we ensure we have specific low frequencies: 0.05, 0.1, 0.2, 0.3
    required_low_freqs = [0.05, 0.1, 0.2, 0.3]
    for req_freq in required_low_freqs:
        if req_freq not in freqs:
            # we find closest and replace or add
            closest_idx = np.argmin(np.abs(freqs - req_freq))
            if np.abs(freqs[closest_idx] - req_freq) > 0.01:
                # we insert if not close enough
                freqs = np.insert(freqs, closest_idx, req_freq)
            else:
                freqs[closest_idx] = req_freq
    
    # we add very high frequencies: 200, 300, 400, 500
    required_high_freqs = [200, 300, 400, 500]
    for req_freq in required_high_freqs:
        if req_freq not in freqs:
            closest_idx = np.argmin(np.abs(freqs - req_freq))
            if np.abs(freqs[closest_idx] - req_freq) > 5:
                freqs = np.insert(freqs, closest_idx, req_freq)
            else:
                freqs[closest_idx] = req_freq
    
    # we sort and remove duplicates
    freqs = np.unique(np.sort(freqs))
    return freqs

def compute_optimal_layers(freq_mult, scaling_law='linear'):
    """we compute optimal layer count based on scaling law"""
    if scaling_law == 'linear':
        # L = 5.16 × freq + 2.55
        L = round(5.16 * freq_mult + 2.55)
    elif scaling_law == 'toeplitz':
        # L = round(freq × 8)
        L = round(freq_mult * 8)
    else:
        L = round(5.16 * freq_mult + 2.55)
    
    # we ensure minimum layers (at least 3 for very low frequencies)
    L = max(L, 3)
    
    # we also test nearby layers (diagonal pattern as before)
    if freq_mult in [0.05, 0.1, 0.2, 0.3]:
        # for very low frequencies (0.05, 0.1, 0.2, 0.3), test more thoroughly: L, 1.5L, 2L, 2.5L
        layers_to_test = [L, max(round(1.5 * L), L+1), max(round(2 * L), L+2), max(round(2.5 * L), L+3)]
    elif freq_mult < 0.5:
        # for other very low frequencies, test L, 1.5*L, 2*L
        layers_to_test = [L, max(round(1.5 * L), L+1), max(round(2 * L), L+2)]
    elif freq_mult < 1.0:
        # for low frequencies, test L and 1.5*L
        layers_to_test = [L, max(round(1.5 * L), L+1)]
    elif freq_mult < 3.0:
        # for medium frequencies, test 0.75*L, L, 1.5*L
        layers_to_test = [max(round(0.75 * L), 3), L, round(1.5 * L)]
    elif freq_mult >= 200:
        # for very high frequencies (200+), test 0.9*L, L, 1.1*L (tighter range)
        layers_to_test = [max(round(0.9 * L), 3), L, round(1.1 * L)]
    else:
        # for high frequencies, test 0.8*L, L, 1.2*L
        layers_to_test = [max(round(0.8 * L), 3), L, round(1.2 * L)]
    
    # we remove duplicates and sort
    layers_to_test = sorted(list(set(layers_to_test)))
    return layers_to_test

def generate_configs():
    """we generate all configurations for extended frequency range"""
    configs = []
    
    # we generate frequency multipliers in log space
    freq_multipliers = generate_freq_multipliers_log_space(min_freq=0.3, max_freq=100, num_points=25)
    
    # we use ranks from previous experiments
    ranks = [10, 15, 25]
    
    # baseline parameters
    batch_size = 100
    
    for freq_mult in freq_multipliers:
        # we compute optimal layers using scaling law
        layers_to_test = compute_optimal_layers(freq_mult)
        
        for rank in ranks:
            for num_layers in layers_to_test:
                # we compute epochs: 2 * freq_mult * 10k (but cap at reasonable values)
                num_epochs = min(int(2 * freq_mult * 10000), 200000)  # cap at 200k epochs
                num_epochs = max(num_epochs, 5000)  # minimum 5k epochs
                
                config = {
                    'freq_multiplier': freq_mult,
                    'hidden_rank': rank,
                    'num_layers': num_layers,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                }
                configs.append(config)
    
    return configs

def target_function(x, freq_multiplier):
    """we create target function with frequency multiplier"""
    # base frequencies: 12π, 24π, 36π, 72π
    # we multiply all by freq_multiplier
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
    """we train one configuration with loss threshold checkpoints"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    
    # we set up model architecture
    hidden_width = 777
    hidden_rank = config['hidden_rank']
    num_layers = config['num_layers']
    input_rank = 1
    output_rank = 1
    
    # we build ranks and widths lists
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
    
    # we create test data (finer grid)
    n_test = 1000
    x_test = np.linspace(interval[0], interval[1], n_test)
    y_test = target_func(x_test, config['freq_multiplier'])
    
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we set up optimizer
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    
    # we define loss thresholds
    thresholds = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6,
                  5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9, 5e-10, 1e-10, 5e-11, 1e-11,
                  5e-12, 1e-12, 5e-13, 1e-13]
    
    # we check for existing checkpoint
    checkpoint_path = output_dir / "checkpoint.pth"
    start_epoch = 0
    all_losses = []
    errors_train = []
    errors_test = []
    errors_test_max = []
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
        print(f"thresholds already reached: {sorted(thresholds_reached)}")
    
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    
    start_time = time.time()
    
    print(f"Starting training: {num_epochs} epochs, batch_size={batch_size}")
    
    for epoch in range(start_epoch, num_epochs):
        # we train one epoch
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
        
        # we evaluate on test set every 50 epochs
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
                
                # we check thresholds
                for thresh in thresholds:
                    if thresh not in thresholds_reached and error_test < thresh:
                        thresholds_reached.add(thresh)
                        # we save model at threshold
                        threshold_dir = output_dir / f"model_at_loss_{thresh:.0e}"
                        threshold_dir.mkdir(exist_ok=True)
                        torch.save(model.state_dict(), threshold_dir / "model_parameters.pth")
                        with open(threshold_dir / "epoch_info.json", 'w') as f:
                            json.dump({'epoch': epoch, 'test_error': error_test}, f)
        
        # we print progress
        if epoch % 500 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}: train={epoch_loss:.4e}, test={errors_test[-1]:.4e}, "
                  f"test_max={errors_test_max[-1]:.4e}, thresholds_reached={len(thresholds_reached)}/{len(thresholds)}")
        
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
        "errors_test": errors_test,
        "errors_test_max": errors_test_max,
        "thresholds_reached": list(thresholds_reached),
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
    x_plot = np.linspace(*interval, 1000)
    x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
    y_plot_true = target_func(x_plot, config["freq_multiplier"])
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot_true, 'b-', label='True function', linewidth=2)
    plt.plot(x_plot, y_plot_nn, 'r--', label='Learned network', linewidth=2)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$f(x)$', fontsize=22)
    arch_label = "FULL_RANK" if config['hidden_rank'] == 777 else f"rank={config['hidden_rank']}"
    config_str = f"{arch_label}, L={config['num_layers']}, freq×{config['freq_multiplier']:.2f}, epoch {len(all_losses)}"
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
    """we run extended frequency scaling experiments"""
    print("="*80)
    print("EXTENDED FREQUENCY AND LAYER SCALING EXPERIMENTS")
    print("Frequency range: 0.3 to 100 (log space)")
    print("="*80)
    
    configs = generate_configs()
    print(f"\nTotal configurations: {len(configs)}")
    
    # we show frequency distribution
    freqs = sorted(set(c['freq_multiplier'] for c in configs))
    print(f"\nFrequency multipliers ({len(freqs)}):")
    for i, f in enumerate(freqs):
        if i % 5 == 0 or i == len(freqs) - 1:
            print(f"  {f:.3f}", end="")
            if (i + 1) % 5 == 0 or i == len(freqs) - 1:
                print()
        elif i == len(freqs) - 1:
            print(f"  {f:.3f}")
    
    results_dir = Path("experiments/table/results_frequency_layer_scaling_extended")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    target_function_lambda = target_function
    
    for i, config in enumerate(configs, 1):
        freq_mult = config['freq_multiplier']
        rank = config['hidden_rank']
        num_layers = config['num_layers']
        
        output_dir_name = f"freq{freq_mult:.3f}_rank{rank}_L{num_layers}"
        output_dir = results_dir / output_dir_name
        
        print("\n" + "="*80)
        print(f"Configuration {i}/{len(configs)}")
        print(f"Freq multiplier: {freq_mult:.3f}, Rank: {rank}, Layers: {num_layers}")
        print(f"Epochs: {config['num_epochs']}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}")
        
        # we check if already completed
        checkpoint_file = output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            import torch
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            epoch = ckpt.get('epoch', 0)
            if epoch >= config['num_epochs']:
                print(f"✓ Already completed (epoch {epoch}/{config['num_epochs']}), skipping...")
                continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = train_one_config(config, output_dir, target_function_lambda)
            print(f"✓ Completed: {output_dir_name}")
        except Exception as e:
            print(f"✗ Error in {output_dir_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("EXTENDED FREQUENCY AND LAYER SCALING BENCHMARK COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
