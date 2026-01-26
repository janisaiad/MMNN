#!/usr/bin/env python3
"""
Mean-field analysis: Save weights of first trainable layer for each epoch
and generate mean-field density plots (GIFs and PNGs) for the first 250 epochs.

Configurations extracted from tofill.tex figures:
1. factor=4 (cos(8πx)), rank=50, momentum=0.0
2. factor=4 (cos(8πx)), rank=50, momentum=0.3
3. factor=4 (cos(8πx)), rank=15, momentum=0.0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import json
import time
from tqdm import tqdm
import sys
import imageio.v2 as imageio
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.table.mmnn_vs import MMNN

# Configure matplotlib
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['savefig.dpi'] = 150

def target_function(x, factor):
    """Cosine function: cos(2*factor*pi*x)"""
    return np.cos(2 * factor * np.pi * x)

def plot_loss_evolution(all_losses, output_dir, config_name):
    """Plot loss evolution for tracking"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(f'Loss Evolution - {config_name}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_evolution.png", dpi=150)
    plt.close()

def plot_meanfield_density(weights_epoch, epoch, output_dir, config_name, bins='auto'):
    """
    Plot mean-field density (histogram) of weights for a given epoch.
    weights_epoch: tensor of shape [n_neurons, r_channels] or flattened
    """
    # Flatten weights to 1D
    weights_flat = weights_epoch.flatten().cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create histogram with auto bins
    counts, bin_edges, patches = ax.hist(weights_flat, bins=bins, density=True, 
                                         alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Weight Value', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'Mean-Field Weight Distribution - Epoch {epoch}\n{config_name}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = np.mean(weights_flat)
    std_val = np.std(weights_flat)
    min_val = np.min(weights_flat)
    max_val = np.max(weights_flat)
    textstr = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save PNG
    png_path = output_dir / f"meanfield_density_epoch{epoch:04d}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    return fig

def train_one_config(config):
    """Train one configuration and save weights + generate mean-field plots"""
    factor = config['factor']
    hidden_rank = config['rank']
    hidden_width = config['width']
    num_layers = 2
    lr_init = config['lr_init']
    momentum = config['momentum']
    num_epochs = 250  # Only first 250 epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    
    # Create output directory
    config_name = f"factor{factor}_rank{hidden_rank}_mom{momentum}_lr{lr_init}"
    output_dir = Path("meanfield_analysis") / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training: {config_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    # Setup model
    input_rank = 1
    output_rank = 1
    ranks = [input_rank] + [hidden_rank] * num_layers + [output_rank]
    widths = [hidden_width] * (num_layers + 1)
    
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=False,
        fixWb=True  # NTK parameterization
    )
    
    # Create training data
    interval = [-1, 1]
    n_train = max(1, int(factor * hidden_width))
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_function(x_train, factor)
    
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # Verify which layers are trainable (for debugging)
    print(f"\n🔍 Checking trainable layers:")
    for i, fc in enumerate(model.fcs):
        is_trainable = any(p.requires_grad for p in fc.parameters())
        print(f"   fcs[{i}]: trainable={is_trainable}, shape={fc.weight.shape}")
    
    # Setup optimizer: Start with Adam, switch to SGD when loss < 1e-3
    # Only train parameters that require grad (fixWb=True means fcs[0], fcs[2], fcs[4] are frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"   Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Start with Adam (matching tune_lr_decay_L2.py)
    use_adam_first = True
    switched_to_sgd = False
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=lr_init,
        betas=(0.9, 0.999)
    )
    optimizer_type = 'Adam'
    
    # Adaptive stagnation scheduler (same as main script)
    adaptive_scheduler = {
        'lr_sequence': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'current_lr_index': 0,
        'window_size': 10,
        'min_epochs_before_reduce': 20,
        'last_reduction_epoch': -1
    }
    
    # Storage for weights: [num_epochs+1, n_neurons, r_channels]
    # First trainable layer is fcs[1] (width→rank, index 1)
    # fcs[0] is rank→width (frozen), fcs[1] is width→rank (trainable)
    # fcs[1].weight shape: [hidden_rank, hidden_width]
    n_neurons = hidden_width
    r_channels = hidden_rank
    weights_storage = torch.zeros(num_epochs + 1, n_neurons, r_channels, device=device)
    
    # Store initial weights (epoch 0)
    with torch.no_grad():
        w1_layer = model.fcs[1].weight.data  # [hidden_rank, hidden_width]
        weights_storage[0] = w1_layer.t()  # Transpose to [hidden_width, hidden_rank] for easier analysis
    
    all_losses = []
    all_lrs = []
    
    # Training loop
    batch_size = 4 * factor * 10
    dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\n📊 Starting training for {num_epochs} epochs...")
    pbar = tqdm(range(num_epochs), desc=f"{config_name}")
    
    for epoch in pbar:
        epoch_losses = []
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = nn.functional.mse_loss(y_pred, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
        
        epoch_loss = np.mean(epoch_losses)
        all_losses.append(epoch_loss)
        
        # Switch from Adam to SGD when loss < 1e-3
        if use_adam_first and not switched_to_sgd and epoch_loss < 1e-3:
            print(f"\n🔄 Switching from Adam to SGD at epoch {epoch} (loss={epoch_loss:.6e} < 1e-3)")
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            # Create new SGD optimizer with same LR and specified momentum
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=current_lr,
                momentum=momentum
            )
            switched_to_sgd = True
            optimizer_type = 'SGD'
        
        # Handle adaptive stagnation scheduler
        current_lr_index = adaptive_scheduler['current_lr_index']
        lr_sequence = adaptive_scheduler['lr_sequence']
        window_size = adaptive_scheduler['window_size']
        min_epochs = adaptive_scheduler['min_epochs_before_reduce']
        last_reduction = adaptive_scheduler['last_reduction_epoch']
        
        # Check if we can reduce LR (enough epochs passed and enough data)
        if (epoch >= min_epochs and 
            epoch - last_reduction >= min_epochs and
            len(all_losses) >= 2 * window_size and
            current_lr_index < len(lr_sequence) - 1):
            
            # Compare mean of last window_size losses vs previous window_size losses
            recent_mean = np.mean(all_losses[-window_size:])
            previous_mean = np.mean(all_losses[-2*window_size:-window_size])
            
            # If loss is stagnating (recent mean >= previous mean), reduce LR
            if recent_mean >= previous_mean:
                current_lr_index += 1
                new_lr = lr_sequence[current_lr_index]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                adaptive_scheduler['current_lr_index'] = current_lr_index
                adaptive_scheduler['last_reduction_epoch'] = epoch
                print(f"\n   📉 Loss stagnating at epoch {epoch}: reducing LR to {new_lr:.2e}")
        
        all_lrs.append(optimizer.param_groups[0]['lr'])
        
        # Store weights for this epoch
        with torch.no_grad():
            w1_layer = model.fcs[1].weight.data  # [hidden_rank, hidden_width]
            weights_storage[epoch + 1] = w1_layer.t()  # Transpose to [hidden_width, hidden_rank]
        
        # Keep LR fixed at 0.01 (no adaptive reduction)
        # all_losses_window.append(epoch_loss)
        # if len(all_losses_window) >= 20:
        #     last_10_mean = np.mean(all_losses_window[-10:])
        #     prev_10_mean = np.mean(all_losses_window[-20:-10])
        #     if last_10_mean >= prev_10_mean and current_lr_idx < len(lr_sequence) - 1:
        #         current_lr_idx += 1
        #         new_lr = lr_sequence[current_lr_idx]
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = new_lr
        #         print(f"\n   📉 LR reduced to {new_lr:.2e} at epoch {epoch}")
        
        opt_name = 'SGD' if switched_to_sgd else 'Adam'
        pbar.set_postfix({'loss': f'{epoch_loss:.6e}', 'lr': f'{all_lrs[-1]:.2e}', 'opt': opt_name})
    
    # Save weights tensor
    weights_path = output_dir / "weights_first_layer.pt"
    torch.save(weights_storage.cpu(), weights_path)
    print(f"\n💾 Saved weights tensor: {weights_path} (shape: {weights_storage.shape})")
    
    # Save config and results
    config_save = {
        'factor': factor,
        'rank': hidden_rank,
        'width': hidden_width,
        'num_layers': num_layers,
        'lr_init': lr_init,
        'momentum': momentum,
        'optimizer_type': optimizer_type,  # Final optimizer type (Adam or SGD)
        'switched_to_sgd': switched_to_sgd,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'n_train': n_train,
        'weights_shape': list(weights_storage.shape),
        'all_losses': all_losses,
        'all_lrs': all_lrs
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Plot loss evolution
    plot_loss_evolution(all_losses, output_dir, config_name)
    
    # Generate mean-field density plots and GIF
    print(f"\n📊 Generating mean-field density plots...")
    frames = []
    bins = 'auto'  # Auto bins for all epochs
    
    # Determine key epochs for PNGs (0, 50, 100, 150, 200, 250)
    key_epochs = [0, 50, 100, 150, 200, 250]
    
    for epoch in range(num_epochs + 1):
        weights_epoch = weights_storage[epoch]  # [hidden_width, hidden_rank]
        
        # Create figure for GIF frame
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        weights_flat = weights_epoch.flatten().cpu().numpy()
        
        counts, bin_edges, patches = ax.hist(weights_flat, bins=bins, density=True,
                                             alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Weight Value', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'Mean-Field Weight Distribution - Epoch {epoch}\n{config_name}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Statistics
        mean_val = np.mean(weights_flat)
        std_val = np.std(weights_flat)
        textstr = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save frame for GIF
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append((epoch, imageio.imread(buf)))
        plt.close(fig)
        
        # Save PNG for key epochs
        if epoch in key_epochs:
            png_path = output_dir / f"meanfield_density_epoch{epoch:04d}.png"
            fig_png, ax_png = plt.subplots(1, 1, figsize=(10, 6))
            ax_png.hist(weights_flat, bins=bins, density=True,
                       alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
            ax_png.set_xlabel('Weight Value', fontsize=14)
            ax_png.set_ylabel('Density', fontsize=14)
            ax_png.set_title(f'Mean-Field Weight Distribution - Epoch {epoch}\n{config_name}', fontsize=12)
            ax_png.grid(True, alpha=0.3)
            ax_png.text(0.02, 0.98, textstr, transform=ax_png.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close(fig_png)
            print(f"   💾 Saved PNG: {png_path.name}")
    
    # Create GIF
    print(f"\n🎬 Creating mean-field density GIF ({len(frames)} frames)...")
    frames_sorted = sorted(frames, key=lambda x: x[0])
    gif_frames = [frame for _, frame in frames_sorted]
    gif_path = output_dir / "meanfield_density_epochs_0_250.gif"
    imageio.mimsave(str(gif_path), gif_frames, duration=0.1, loop=0)
    print(f"   ✅ Saved GIF: {gif_path.name}")
    
    # Plot weight variation distribution
    print(f"\n📊 Computing and plotting weight variations...")
    # Compute variations: difference between consecutive epochs
    weight_variations = []
    for epoch in range(1, num_epochs + 1):
        variation = (weights_storage[epoch] - weights_storage[epoch-1]).flatten().cpu().numpy()
        weight_variations.append(variation)
    
    # Flatten all variations
    all_variations = np.concatenate(weight_variations)
    
    # Plot distribution of weight variations
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Histogram of all variations
    ax1 = axes[0]
    ax1.hist(all_variations, bins='auto', density=True, alpha=0.7, color='red', 
             edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Weight Variation (Δw)', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title(f'Distribution of Weight Variations (All Epochs)\n{config_name}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Statistics
    mean_var = np.mean(all_variations)
    std_var = np.std(all_variations)
    textstr1 = f'Mean: {mean_var:.6f}\nStd: {std_var:.6f}\nMin: {np.min(all_variations):.6f}\nMax: {np.max(all_variations):.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot 2: Evolution of variation statistics over epochs
    ax2 = axes[1]
    epoch_means = [np.mean(v) for v in weight_variations]
    epoch_stds = [np.std(v) for v in weight_variations]
    epochs_range = range(1, num_epochs + 1)
    
    ax2.plot(epochs_range, epoch_means, 'b-', linewidth=1.5, alpha=0.7, label='Mean variation')
    ax2.fill_between(epochs_range, 
                     [m - s for m, s in zip(epoch_means, epoch_stds)],
                     [m + s for m, s in zip(epoch_means, epoch_stds)],
                     alpha=0.3, color='blue', label='±1 Std')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Weight Variation', fontsize=14)
    ax2.set_title(f'Evolution of Weight Variation Statistics\n{config_name}', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    variation_path = output_dir / "weight_variation_distribution.png"
    plt.savefig(variation_path, dpi=150)
    plt.close()
    print(f"   ✅ Saved weight variation plot: {variation_path.name}")
    
    # Also create GIF of variation distributions over epochs
    print(f"\n🎬 Creating weight variation GIF...")
    variation_frames = []
    bins_var = 'auto'
    
    for epoch in range(1, num_epochs + 1):
        variation_epoch = weight_variations[epoch - 1]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(variation_epoch, bins=bins_var, density=True, alpha=0.7, 
               color='red', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Weight Variation (Δw)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'Weight Variation Distribution - Epoch {epoch}\n{config_name}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        mean_var_epoch = np.mean(variation_epoch)
        std_var_epoch = np.std(variation_epoch)
        textstr = f'Mean: {mean_var_epoch:.6f}\nStd: {std_var_epoch:.6f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save frame for GIF
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        variation_frames.append((epoch, imageio.imread(buf)))
        plt.close(fig)
    
    # Create variation GIF
    variation_frames_sorted = sorted(variation_frames, key=lambda x: x[0])
    gif_var_frames = [frame for _, frame in variation_frames_sorted]
    gif_var_path = output_dir / "weight_variation_epochs_1_250.gif"
    imageio.mimsave(str(gif_var_path), gif_var_frames, duration=0.1, loop=0)
    print(f"   ✅ Saved variation GIF: {gif_var_path.name}")
    
    print(f"\n✅ Completed: {config_name}")
    return config_save

def main():
    """Main function: train all configurations"""
    # Configurations extracted from tofill.tex figures
    configs = [
        {
            'factor': 4,  # cos(8πx)
            'rank': 50,
            'width': 1024,
            'lr_init': 0.01,
            'momentum': 0.0
        },
        {
            'factor': 4,  # cos(8πx)
            'rank': 50,
            'width': 1024,
            'lr_init': 0.01,
            'momentum': 0.3
        },
        {
            'factor': 4,  # cos(8πx)
            'rank': 15,
            'width': 1024,
            'lr_init': 0.01,
            'momentum': 0.0
        }
    ]
    
    print("="*80)
    print("MEAN-FIELD ANALYSIS: Weight Distribution Through Training")
    print("="*80)
    print(f"Number of configurations: {len(configs)}")
    print(f"Epochs per config: 250 (first 250 epochs only)")
    print(f"Output directory: meanfield_analysis/")
    print("="*80)
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing configuration...")
        result = train_one_config(config)
        results.append(result)
    
    # Save summary
    summary = {
        'experiment': 'Mean-field weight distribution analysis',
        'configs': results,
        'description': 'First 250 epochs, saving weights of first trainable layer (fcs[1]) for each epoch'
    }
    
    summary_path = Path("meanfield_analysis") / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ All configurations completed!")
    print(f"   Summary saved: {summary_path}")
    print("="*80)

if __name__ == "__main__":
    main()
