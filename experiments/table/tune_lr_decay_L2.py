#!/usr/bin/env python3
"""
Tune learning rate decay for L=2 with cos(2*factor*pi*x)
Test over 10000 epochs to find optimal LR decay schedule
Batch size = 4*factor*10 (linear in frequency for Fourier approximation)
Runs configurations in parallel on GPU
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LinearLR
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import json
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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

def target_function(x, factor):
    """Simple cosine function: cos(2*factor*pi*x)"""
    return np.cos(2 * factor * np.pi * x)

def plot_prediction_vs_baseline(model, x_test_plot, y_test_plot, x_train_plot, y_train_plot, 
                                 epoch, new_lr, output_dir, factor, current_optimizer_type, lr_config, sgd_momentum=None):
    """Plot prediction vs baseline at LR reduction moment"""
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test_plot.reshape([-1, 1]), device=next(model.parameters()).device, dtype=torch.float32)
        y_pred_test = model(x_test_tensor).cpu().numpy().flatten()
        
        x_train_tensor = torch.tensor(x_train_plot.reshape([-1, 1]), device=next(model.parameters()).device, dtype=torch.float32)
        y_pred_train = model(x_train_tensor).cpu().numpy().flatten()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot baseline (ground truth)
    ax.plot(x_test_plot, y_test_plot, 'b-', linewidth=2, alpha=0.7, label='Baseline (ground truth)')
    
    # Plot prediction
    ax.plot(x_test_plot, y_pred_test, 'r--', linewidth=2, alpha=0.7, label='Prediction')
    
    # Plot training points
    ax.scatter(x_train_plot, y_train_plot, c='green', s=20, alpha=0.5, label='Training points', zorder=5)
    ax.scatter(x_train_plot, y_pred_train, c='orange', s=20, alpha=0.5, label='Prediction at train points', zorder=5)
    
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    title = f'Prediction vs Baseline - Epoch {epoch}, LR={new_lr:.2e}\n'
    title += f'factor={factor}, {current_optimizer_type}'
    if current_optimizer_type == 'SGD':
        momentum = sgd_momentum if sgd_momentum is not None else lr_config.get("momentum", 0.9)
        title += f', Momentum={momentum}'
    elif current_optimizer_type == 'Adam':
        betas = lr_config.get('betas', (0.9, 0.999))
        title += f', Betas=({betas[0]}, {betas[1]})'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"prediction_vs_baseline_epoch{epoch}_lr{new_lr:.2e}.png"
    plt.savefig(output_dir / plot_filename, dpi=150)
    plt.close()
    print(f"   💾 Saved prediction plot: {plot_filename}")

def train_one_config_wrapper(args):
    """Wrapper for multiprocessing - unpacks arguments"""
    factor, lr_config, output_dir = args
    return train_one_config(factor, lr_config, Path(output_dir))

def train_one_config(factor, lr_config, output_dir):
    """we train one configuration with different LR decay schedules"""
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    
    lr_init = lr_config['lr_init']
    scheduler_type = lr_config.get('scheduler_type', None)
    scheduler_params = lr_config.get('scheduler_params', {})
    optimizer_type = lr_config.get('optimizer_type', 'Adam')
    
    print(f"\n{'='*80}")
    print(f"Training: factor={factor}, Optimizer={optimizer_type}, LR={lr_init}")
    if scheduler_type:
        print(f"  Scheduler={scheduler_type}, params={scheduler_params}")
    if optimizer_type == 'SGD':
        print(f"  Momentum={lr_config.get('momentum', 0.9)}")
    elif optimizer_type == 'Adam':
        print(f"  Betas={lr_config.get('betas', (0.9, 0.999))}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    # we set up model: L=2 layers
    hidden_width = 1024
    hidden_rank = lr_config.get('hidden_rank', 15)  # Get rank from config
    num_layers = 2  # Fixed L=2
    input_rank = 1
    output_rank = 1
    
    ranks = [input_rank] + [hidden_rank] * num_layers + [output_rank]
    widths = [hidden_width] * (num_layers + 1)
    
    # NTK parameterization (fixWb=True)
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=False,
        fixWb=True  # NTK parameterization
    )
    
    # we create training data
    interval = [-1, 1]
    n_train = max(1, int(factor * hidden_width))  # factor * width training points
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_function(x_train, factor)
    
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we create test data
    n_test = 1000
    x_test = np.linspace(interval[0], interval[1], n_test)
    y_test = target_function(x_test, factor)
    
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we keep numpy arrays for plotting
    x_train_plot = x_train.copy()
    y_train_plot = y_train.copy()
    x_test_plot = x_test.copy()
    y_test_plot = y_test.copy()
    
    # we set up optimizer based on config
    # Start with Adam, switch to SGD when loss < 1e-3
    optimizer_type = lr_config.get('optimizer_type', 'Adam')
    use_adam_first = (optimizer_type == 'Adam')
    switched_to_sgd = False
    sgd_momentum = lr_config.get('momentum', 0.9)  # Default momentum for SGD switch
    
    if optimizer_type == 'SGD':
        momentum = lr_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
        use_adam_first = False
    elif optimizer_type == 'Adam':
        betas = lr_config.get('betas', (0.9, 0.999))
        optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=betas)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_init)
    
    # we set up scheduler based on type
    scheduler = None
    adaptive_scheduler = None
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])
    elif scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=scheduler_params['gamma'])
    elif scheduler_type == 'LinearLR':
        # LinearLR: lr = lr_init * (1 - start_factor + start_factor * (1 - epoch / total_epochs))
        # By default, start_factor=1.0, end_factor=0.0 (decay from lr_init to 0)
        start_factor = scheduler_params.get('start_factor', 1.0)
        end_factor = scheduler_params.get('end_factor', 0.0)
        total_iters = scheduler_params.get('total_iters', num_epochs)
        scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    elif scheduler_type == 'AdaptiveStagnation':
        # Custom adaptive scheduler: reduces LR when loss stagnates
        adaptive_scheduler = {
            'lr_sequence': scheduler_params.get('lr_sequence', [0.01, 0.005, 0.001, 0.0005, 0.0001]),
            'current_lr_index': 0,
            'window_size': scheduler_params.get('window_size', 10),
            'min_epochs_before_reduce': scheduler_params.get('min_epochs_before_reduce', 20),
            'last_reduction_epoch': -1
        }
    
    # training parameters
    batch_size = max(1, int(4 * factor * 10))  # batch_size = 4*factor*10
    num_epochs = 10000  # 10k epochs
    
    # we track training
    all_losses = []
    all_lrs = []
    errors_train = []
    errors_test = []
    errors_test_max = []
    min_loss = float('inf')
    min_loss_epoch = 0
    
    start_time = time.time()
    
    # Plot network at initialization (epoch 0)
    print(f"\n📊 Plotting network at initialization...")
    plot_prediction_vs_baseline(
        model, x_test_plot, y_test_plot, x_train_plot, y_train_plot,
        0, lr_init, output_dir, factor, 'Adam' if use_adam_first else 'SGD', lr_config, sgd_momentum
    )
    
    # we use tqdm for progress bar
    desc = f"factor={factor}, {optimizer_type}"
    if scheduler_type:
        desc += f", {scheduler_type}"
    pbar = tqdm(range(num_epochs), desc=desc, unit="epoch")
    
    # Store model state before each epoch to capture "before" state for large drops
    model_state_before = None
    
    for epoch in pbar:
        # Save model state before training this epoch (for plotting before large drops)
        if epoch > 0:  # Don't save at epoch 0 (already plotted initialization)
            model_state_before = {k: v.clone() for k, v in model.state_dict().items()}
        
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
            
            # check for NaN before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  NaN/Inf loss detected at epoch {epoch}, batch {i}")
                print(f"   y_pred stats: min={y_pred.min().item():.6e}, max={y_pred.max().item():.6e}, mean={y_pred.mean().item():.6e}")
                print(f"   y_batch stats: min={y_batch.min().item():.6e}, max={y_batch.max().item():.6e}, mean={y_batch.mean().item():.6e}")
                # check gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"   NaN/Inf gradient in {name}, norm={grad_norm}")
                break
            
            loss.backward()
            
            # gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss /= (n_train // batch_size + 1)
        
        # check for NaN in epoch loss
        if np.isnan(epoch_loss) or np.isinf(epoch_loss):
            print(f"\n⚠️  NaN/Inf epoch loss at epoch {epoch}, stopping training")
            break
        
        all_losses.append(epoch_loss)
        
        # Switch from Adam to SGD when loss < 1e-3
        if use_adam_first and not switched_to_sgd and epoch_loss < 1e-3:
            print(f"\n🔄 Switching from Adam to SGD at epoch {epoch} (loss={epoch_loss:.6e} < 1e-3)")
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            # Create new SGD optimizer with same LR
            optimizer = optim.SGD(model.parameters(), lr=current_lr, momentum=sgd_momentum)
            switched_to_sgd = True
            # Update scheduler if it exists (need to recreate with new optimizer)
            if scheduler is not None:
                if scheduler_type == 'StepLR':
                    scheduler = StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])
                elif scheduler_type == 'ExponentialLR':
                    scheduler = ExponentialLR(optimizer, gamma=scheduler_params['gamma'])
                elif scheduler_type == 'LinearLR':
                    start_factor = scheduler_params.get('start_factor', 1.0)
                    end_factor = scheduler_params.get('end_factor', 0.0)
                    total_iters = scheduler_params.get('total_iters', num_epochs)
                    scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
            # Update optimizer_type for logging
            optimizer_type = 'SGD'
        
        # Handle adaptive stagnation scheduler
        if adaptive_scheduler is not None:
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
                    # Store LR reduction moments
                    if 'lr_reduction_epochs' not in adaptive_scheduler:
                        adaptive_scheduler['lr_reduction_epochs'] = []
                    adaptive_scheduler['lr_reduction_epochs'].append(epoch)
                    print(f"\n📉 Loss stagnating at epoch {epoch}: reducing LR to {new_lr:.2e}")
                    
                    # Plot prediction vs baseline at LR reduction moment
                    current_opt_type = 'SGD' if switched_to_sgd else 'Adam'
                    plot_prediction_vs_baseline(
                        model, x_test_plot, y_test_plot, x_train_plot, y_train_plot,
                        epoch, new_lr, output_dir, factor, current_opt_type, lr_config, sgd_momentum
                    )
        
        all_lrs.append(optimizer.param_groups[0]['lr'])
        
        # we update tqdm with current loss and optimizer type
        opt_name = 'SGD' if switched_to_sgd else 'Adam'
        pbar.set_postfix({'loss': f'{epoch_loss:.6e}', 'lr': f'{all_lrs[-1]:.2e}', 'opt': opt_name})
        
        # plot loss curve every 200 epochs
        if (epoch + 1) % 200 == 0 or epoch == 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
            
            # Add red vertical bars at LR reduction moments
            if adaptive_scheduler is not None and 'lr_reduction_epochs' in adaptive_scheduler:
                for reduction_epoch in adaptive_scheduler['lr_reduction_epochs']:
                    if reduction_epoch < len(all_losses):
                        ax.axvline(x=reduction_epoch, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Loss', fontsize=14)
            current_opt = 'SGD' if switched_to_sgd else 'Adam'
            title = f'Loss Evolution - factor={factor}, {current_opt}, LR={lr_init}'
            if switched_to_sgd:
                title += f', Momentum={sgd_momentum}'
            elif not switched_to_sgd:
                betas = lr_config.get('betas', (0.9, 0.999))
                title += f', Betas=({betas[0]}, {betas[1]})'
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "loss_evolution.png", dpi=150)
            plt.close()
        
        # we compute errors every 200 epochs
        if (epoch + 1) % 200 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                y_pred_train = model(x_train_tensor)
                y_pred_test = model(x_test_tensor)
            
            error_train = torch.mean((y_pred_train - y_train_tensor)**2).item()
            error_test = torch.mean((y_pred_test - y_test_tensor)**2).item()
            error_test_max = torch.max(torch.abs(y_pred_test - y_test_tensor)).item()
            
            errors_train.append(error_train)
            errors_test.append(error_test)
            errors_test_max.append(error_test_max)
            
            # track minimum loss
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                min_loss_epoch = epoch
    
    training_time = time.time() - start_time
    
    # we plot loss evolution and LR schedule
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # loss evolution
    ax1 = axes[0]
    ax1.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
    
    # Add red vertical bars at LR reduction moments
    if adaptive_scheduler is not None and 'lr_reduction_epochs' in adaptive_scheduler:
        for reduction_epoch in adaptive_scheduler['lr_reduction_epochs']:
            if reduction_epoch < len(all_losses):
                ax1.axvline(x=reduction_epoch, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='LR reduction' if reduction_epoch == adaptive_scheduler['lr_reduction_epochs'][0] else '')
    
    ax1.set_xlabel('Epoch', fontsize=18)
    ax1.set_ylabel('Loss', fontsize=18)
    title = f'Loss Evolution - factor={factor}, {final_optimizer_type}'
    if switched_to_sgd:
        title += ' (Adam→SGD)'
    if scheduler_type:
        title += f', {scheduler_type}'
    title += f'\nBatch size={batch_size}, LR init={lr_init}'
    if final_optimizer_type == 'SGD':
        title += f', Momentum={sgd_momentum if switched_to_sgd else lr_config.get("momentum", 0.9)}'
    elif not switched_to_sgd:
        betas = lr_config.get('betas', (0.9, 0.999))
        title += f', Betas=({betas[0]}, {betas[1]})'
    ax1.set_title(title, fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # LR schedule
    ax2 = axes[1]
    ax2.semilogy(all_lrs, 'r-', linewidth=1.5, alpha=0.7, label='Learning Rate')
    ax2.set_xlabel('Epoch', fontsize=18)
    ax2.set_ylabel('Learning Rate', fontsize=18)
    ax2.set_title(f'LR Schedule - {scheduler_type}', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_and_lr_evolution.png", dpi=150)
    plt.close()
    
    # we compute oscillation metrics
    losses_array = np.array(all_losses)
    # compute relative oscillations (coefficient of variation)
    if len(losses_array) > 10:
        # use last 200 epochs for stability
        recent_losses = losses_array[-200:] if len(losses_array) > 200 else losses_array
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        cv = std_loss / mean_loss if mean_loss > 0 else float('inf')
        
        # compute range of oscillations (max/min ratio in recent epochs)
        max_loss = np.max(recent_losses)
        min_loss_recent = np.min(recent_losses)
        oscillation_ratio = max_loss / min_loss_recent if min_loss_recent > 0 else float('inf')
    else:
        cv = float('inf')
        oscillation_ratio = float('inf')
    
    # Store LR reduction epochs if adaptive scheduler was used
    lr_reduction_epochs = []
    if adaptive_scheduler is not None and 'lr_reduction_epochs' in adaptive_scheduler:
        lr_reduction_epochs = adaptive_scheduler['lr_reduction_epochs']
    
    # Determine final optimizer type (Adam or switched to SGD)
    final_optimizer_type = 'SGD' if switched_to_sgd else 'Adam'
    
    # we save results
    results = {
        'factor': factor,
        'num_layers': num_layers,
        'hidden_rank': hidden_rank,
        'hidden_width': hidden_width,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr_init': lr_init,
        'optimizer_type': final_optimizer_type,
        'switched_to_sgd': switched_to_sgd,
        'sgd_momentum': sgd_momentum if switched_to_sgd else None,
        'scheduler_type': scheduler_type,
        'scheduler_params': scheduler_params,
        'momentum': lr_config.get('momentum', None),
        'betas': lr_config.get('betas', None),
        'min_loss': min_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_loss': all_losses[-1],
        'final_train_error': errors_train[-1] if errors_train else None,
        'final_test_error': errors_test[-1] if errors_test else None,
        'final_test_error_max': errors_test_max[-1] if errors_test_max else None,
        'training_time_seconds': training_time,
        'oscillation_cv': cv,
        'oscillation_ratio': oscillation_ratio,
        'all_losses': all_losses,  # Store all losses
        'all_lrs': all_lrs,
        'lr_reduction_epochs': lr_reduction_epochs,  # Store LR reduction moments
        'errors_train': errors_train,
        'errors_test': errors_test,
        'errors_test_max': errors_test_max,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # we save config
    config = {
        'factor': factor,
        'num_layers': num_layers,
        'hidden_rank': hidden_rank,
        'hidden_width': hidden_width,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr_init': lr_init,
        'optimizer_type': final_optimizer_type,
        'switched_to_sgd': switched_to_sgd,
        'sgd_momentum': sgd_momentum if switched_to_sgd else None,
        'scheduler_type': scheduler_type,
        'scheduler_params': scheduler_params,
        'momentum': lr_config.get('momentum', None),
        'betas': lr_config.get('betas', None),
        'function': f'cos(2*{factor}*pi*x)',
        'parameterization': 'NTK',  # fixWb=True
        'initialization': 'uniform[-1,1]/sqrt(n)',
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    print(f"   Min loss: {min_loss:.6e} at epoch {min_loss_epoch}")
    print(f"   Final loss: {all_losses[-1]:.6e}")
    print(f"   Final optimizer: {final_optimizer_type}")
    if switched_to_sgd:
        print(f"   Switched to SGD at loss < 1e-3")
    print(f"   Oscillation CV: {cv:.4f}, Ratio: {oscillation_ratio:.2f}")
    
    return results

def main():
    """we test different LR decay schedules"""
    # we create output directory
    output_base = Path("experiments/table/results_tune_lr_decay_L2")
    output_base.mkdir(parents=True, exist_ok=True)
    
    factors = [4]  # factor=4
    
    # Test different ranks
    ranks_to_test = [10, 15, 20, 25, 50]
    
    # we define different optimizer and LR configurations to test
    # SGD only as preferred, test multiple learning rates
    lr_configs = []
    
    # Test with adaptive LR scheduler: starts at 1e-2, reduces when loss stagnates
    # Use Adam optimizer
    lr_configs.append({
        'optimizer_type': 'Adam',
        'lr_init': 0.01,  # Start at 1e-2
        'betas': (0.9, 0.999),  # Default Adam betas
        'scheduler_type': 'AdaptiveStagnation',  # Custom adaptive scheduler
        'scheduler_params': {
            'lr_sequence': [0.01, 0.005, 0.001, 0.0005, 0.0001],  # 1e-2, 5e-3, 1e-3, 5e-4, 1e-4
            'window_size': 10,  # Compare last 10 vs previous 10
            'min_epochs_before_reduce': 20  # Minimum epochs before checking stagnation
        }
    })
    
    print("="*80)
    print("OPTIMIZER & LR DECAY TUNING: L=2, cos(2*factor*pi*x)")
    print("="*80)
    print(f"Testing factors: {factors}")
    print(f"Testing ranks: {ranks_to_test}")
    print(f"Epochs per config: 10000")
    print(f"Batch size: 4*factor*10 (linear in frequency)")
    print(f"Parameterization: NTK (fixWb=True)")
    print(f"Initialization: Uniform[-1,1] / sqrt(n)")
    print(f"Optimizer: Adam")
    print(f"Initial LR: 0.01 (1e-2), adaptive reduction on stagnation")
    print(f"LR sequence: [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]")
    print(f"Betas: (0.9, 0.999)")
    print(f"Adaptive scheduler: reduces LR when loss stagnates (mean of last 10 >= mean of previous 10)")
    print(f"Total configs to test: {len(lr_configs) * len(factors) * len(ranks_to_test)}")
    print(f"Output directory: {output_base}")
    print("="*80)
    
    # Prepare all configurations for parallel execution
    all_configs_to_run = []
    for factor in factors:
        for rank in ranks_to_test:
            for i, lr_config in enumerate(lr_configs):
                optimizer_name = lr_config['optimizer_type']
                if optimizer_name == 'SGD':
                    optimizer_name += f"_mom{lr_config.get('momentum', 0.9)}"
                elif optimizer_name == 'Adam':
                    beta1 = lr_config.get('betas', (0.9, 0.999))[0]
                    beta2 = lr_config.get('betas', (0.9, 0.999))[1]
                    optimizer_name += f"_beta1{beta1}_beta2{beta2}"
                
                if lr_config.get('scheduler_type') == 'StepLR':
                    scheduler_name = f"StepLR_step{lr_config['scheduler_params'].get('step_size', 'N/A')}_gamma{lr_config['scheduler_params'].get('gamma', 'N/A')}"
                elif lr_config.get('scheduler_type') == 'ExponentialLR':
                    scheduler_name = f"ExpLR_gamma{lr_config['scheduler_params'].get('gamma', 'N/A')}"
                elif lr_config.get('scheduler_type') == 'LinearLR':
                    end_factor = lr_config['scheduler_params'].get('end_factor', 0.0)
                    scheduler_name = f"LinearLR_end{end_factor}"
                elif lr_config.get('scheduler_type') == 'AdaptiveStagnation':
                    scheduler_name = "AdaptiveStagnation"
                else:
                    scheduler_name = "NoScheduler"
                
                output_dir = output_base / f"factor{factor}_rank{rank}_{optimizer_name}_lr{lr_config['lr_init']}_{scheduler_name}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Add rank to config
                lr_config_with_rank = lr_config.copy()
                lr_config_with_rank['hidden_rank'] = rank
                
                all_configs_to_run.append((factor, lr_config_with_rank, str(output_dir)))
    
    # Run configurations in parallel
    print(f"\n🚀 Running {len(all_configs_to_run)} configurations in parallel on GPU...")
    print(f"   Using {min(8, len(all_configs_to_run))} parallel workers\n")
    
    all_results = []
    num_workers = min(8, len(all_configs_to_run))  # Limit to 8 parallel processes to avoid GPU saturation
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks (pass as tuple for easier pickling)
        future_to_config = {
            executor.submit(train_one_config_wrapper, (factor, lr_config, output_dir)): (factor, lr_config, output_dir)
            for factor, lr_config, output_dir in all_configs_to_run
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(all_configs_to_run), desc="Training configs", unit="config") as pbar:
            for future in as_completed(future_to_config):
                try:
                    results = future.result()
                    all_results.append(results)
                    pbar.update(1)
                except Exception as e:
                    config_info = future_to_config[future]
                    print(f"\n❌ Error in config {config_info[2]}: {e}")
                    pbar.update(1)
    
    # we save summary
    summary = {
        'experiment': 'LR decay tuning: L=2, cos(2*factor*pi*x)',
        'factors_tested': factors,
        'num_layers': 2,
        'num_epochs': 10000,
        'batch_size_formula': '4*factor*10',
        'lr_configs_tested': lr_configs,
        'results': all_results
    }
    
    with open(output_base / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # we create a summary table
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLE")
    print("="*80)
    
    # Create DataFrame for table
    table_data = []
    for r in all_results:
        table_data.append({
            'Factor': r['factor'],
            'Rank': r['hidden_rank'],
            'Optimizer': r['optimizer_type'],
            'Momentum': r.get('momentum') if r['optimizer_type'] == 'SGD' else None,
            'Betas': str(r.get('betas')) if r['optimizer_type'] == 'Adam' else None,
            'LR Init': r['lr_init'],
            'Scheduler': r.get('scheduler_type', 'None'),
            'Scheduler Params': str(r.get('scheduler_params', {})),
            'Batch Size': r['batch_size'],
            'Final Loss': f"{r['final_loss']:.6e}",
            'Min Loss': f"{r['min_loss']:.6e}",
            'Oscillation Ratio': f"{r['oscillation_ratio']:.2f}",
            'Oscillation CV': f"{r['oscillation_cv']:.4f}",
            'Epochs': r['num_epochs']
        })
    
    df = pd.DataFrame(table_data)
    
    # Save table to CSV
    csv_path = output_base / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Summary table saved to: {csv_path}")
    
    # Print table sorted by oscillation ratio
    print("\n" + "="*80)
    print("SUMMARY TABLE (sorted by oscillation ratio)")
    print("="*80)
    df_sorted = df.sort_values('Oscillation Ratio', key=lambda x: pd.to_numeric(x, errors='coerce'))
    print(df_sorted.to_string(index=False))
    
    # we analyze results by factor and optimizer
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS BY RANK")
    print("="*80)
    for rank in ranks_to_test:
        rank_results = [r for r in all_results if r['hidden_rank'] == rank]
        if rank_results:
            best = min(rank_results, key=lambda x: x['min_loss'])
            opt_info = f"{best['optimizer_type']}"
            if best['optimizer_type'] == 'SGD':
                opt_info += f", momentum={best.get('momentum', 'N/A')}"
            elif best['optimizer_type'] == 'Adam':
                betas = best.get('betas', (0.9, 0.999))
                opt_info += f", betas=({betas[0]}, {betas[1]})"
            print(f"Rank {rank}: min_loss={best['min_loss']:.6e} with {opt_info}, {r.get('scheduler_type', 'No scheduler')}")

if __name__ == "__main__":
    main()
