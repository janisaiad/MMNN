#!/usr/bin/env python3
"""
Simple test: L=2 with cos(2*factor*pi*x)
Test factor from 1 to 5 at 10k epochs
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
import json
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm

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

def train_one_config(factor, output_dir):
    """we train one configuration with L=2 and cos(2*factor*pi*x)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mydtype = torch.float32
    
    print(f"\n{'='*80}")
    print(f"Training: L=2, factor={factor}, function=cos(2*{factor}*pi*x)")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    # we set up model: L=2 layers
    hidden_width = 777
    hidden_rank = 15
    num_layers = 2  # Fixed L=2
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
    
    # we set up optimizer
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    
    # training parameters
    batch_size = 100
    num_epochs = 10000
    
    # we track training
    all_losses = []
    errors_train = []
    errors_test = []
    errors_test_max = []
    min_loss = float('inf')
    min_loss_epoch = 0
    
    # early stopping threshold
    loss_threshold = 2e-5
    
    start_time = time.time()
    
    # we use tqdm for progress bar
    pbar = tqdm(range(num_epochs), desc=f"factor={factor}", unit="epoch")
    
    for epoch in pbar:
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
        
        # we update tqdm with current loss
        pbar.set_postfix({'loss': f'{epoch_loss:.6e}'})
        
        # early stopping: stop when loss < 2e-5
        if epoch_loss < loss_threshold:
            print(f"\n✅ Early stopping: loss {epoch_loss:.6e} < {loss_threshold:.0e} at epoch {epoch+1}")
            # we plot loss evolution and prediction before stopping
            model.eval()
            with torch.no_grad():
                y_pred_test = model(x_test_tensor)
                y_pred_train = model(x_train_tensor)
            
            y_pred_test_np = y_pred_test.cpu().numpy().flatten()
            y_pred_train_np = y_pred_train.cpu().numpy().flatten()
            
            # plot prediction vs target
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # train set
            ax1 = axes[0]
            ax1.plot(x_train_plot, y_train_plot, 'b-', linewidth=2, label='Target', alpha=0.7)
            ax1.plot(x_train_plot, y_pred_train_np, 'r--', linewidth=2, label='Prediction', alpha=0.7)
            ax1.set_xlabel('x', fontsize=18)
            ax1.set_ylabel('y', fontsize=18)
            ax1.set_title(f'Train Set - Epoch {epoch+1} (Early Stop)\nLoss: {epoch_loss:.6e}', fontsize=16)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # test set
            ax2 = axes[1]
            ax2.plot(x_test_plot, y_test_plot, 'b-', linewidth=2, label='Target', alpha=0.7)
            ax2.plot(x_test_plot, y_pred_test_np, 'r--', linewidth=2, label='Prediction', alpha=0.7)
            ax2.set_xlabel('x', fontsize=18)
            ax2.set_ylabel('y', fontsize=18)
            ax2.set_title(f'Test Set - Epoch {epoch+1} (Early Stop)\nLoss: {epoch_loss:.6e}', fontsize=16)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"prediction_epoch{epoch+1}.png", dpi=150)
            plt.close()
            
            # plot loss evolution
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7)
            ax.axhline(y=loss_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({loss_threshold:.0e})')
            ax.set_xlabel('Epoch', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)
            ax.set_title(f'Loss Evolution - factor={factor}, L=2 (Early Stop at epoch {epoch+1})', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "loss_evolution.png", dpi=150)
            plt.close()
            
            break
        
        # we compute errors every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
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
        
        # we plot every 2000 epochs
        if (epoch + 1) % 2000 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                y_pred_test = model(x_test_tensor)
                y_pred_train = model(x_train_tensor)
            
            y_pred_test_np = y_pred_test.cpu().numpy().flatten()
            y_pred_train_np = y_pred_train.cpu().numpy().flatten()
            
            # plot prediction vs target
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # train set
            ax1 = axes[0]
            ax1.plot(x_train_plot, y_train_plot, 'b-', linewidth=2, label='Target', alpha=0.7)
            ax1.plot(x_train_plot, y_pred_train_np, 'r--', linewidth=2, label='Prediction', alpha=0.7)
            ax1.set_xlabel('x', fontsize=18)
            ax1.set_ylabel('y', fontsize=18)
            ax1.set_title(f'Train Set - Epoch {epoch+1}\nLoss: {epoch_loss:.6e}', fontsize=16)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # test set
            ax2 = axes[1]
            ax2.plot(x_test_plot, y_test_plot, 'b-', linewidth=2, label='Target', alpha=0.7)
            ax2.plot(x_test_plot, y_pred_test_np, 'r--', linewidth=2, label='Prediction', alpha=0.7)
            ax2.set_xlabel('x', fontsize=18)
            ax2.set_ylabel('y', fontsize=18)
            ax2.set_title(f'Test Set - Epoch {epoch+1}\nLoss: {epoch_loss:.6e}', fontsize=16)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"prediction_epoch{epoch+1}.png", dpi=150)
            plt.close()
            
            # plot loss evolution
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7)
            ax.axhline(y=loss_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({loss_threshold:.0e})')
            ax.set_xlabel('Epoch', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)
            ax.set_title(f'Loss Evolution - factor={factor}, L=2', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "loss_evolution.png", dpi=150)
            plt.close()
    
    training_time = time.time() - start_time
    
    # we check if early stopping occurred
    epochs_run = len(all_losses)
    early_stopped = epochs_run < num_epochs
    
    # we ensure loss evolution is plotted at the end (in case it wasn't plotted recently)
    if len(all_losses) > 0:
        # check if loss_evolution.png exists and is recent (within last 2000 epochs)
        loss_plot_path = output_dir / "loss_evolution.png"
        if not loss_plot_path.exists() or (epochs_run % 2000 != 0):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7)
            ax.axhline(y=loss_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({loss_threshold:.0e})')
            ax.set_xlabel('Epoch', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)
            title = f'Loss Evolution - factor={factor}, L=2'
            if early_stopped:
                title += f' (Stopped at epoch {epochs_run})'
            ax.set_title(title, fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(loss_plot_path, dpi=150)
            plt.close()
    
    # we save final results
    results = {
        'factor': factor,
        'num_layers': num_layers,
        'hidden_rank': hidden_rank,
        'hidden_width': hidden_width,
        'num_epochs': num_epochs,
        'epochs_run': epochs_run,
        'early_stopped': early_stopped,
        'loss_threshold': loss_threshold,
        'batch_size': batch_size,
        'lr': lr,
        'min_loss': min_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_loss': all_losses[-1],
        'final_train_error': errors_train[-1] if errors_train else None,
        'final_test_error': errors_test[-1] if errors_test else None,
        'final_test_error_max': errors_test_max[-1] if errors_test_max else None,
        'training_time_seconds': training_time,
        'all_losses': all_losses,
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
        'lr': lr,
        'function': f'cos(2*{factor}*pi*x)',
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # we save model parameters
    torch.save(model.state_dict(), output_dir / "model_parameters.pth")
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    if early_stopped:
        print(f"   Early stopped at epoch {epochs_run} (loss < {loss_threshold})")
    else:
        print(f"   Completed all {num_epochs} epochs")
    print(f"   Min loss: {min_loss:.6e} at epoch {min_loss_epoch}")
    print(f"   Final loss: {all_losses[-1]:.6e}")
    
    return results

def main():
    """we run training for factors 1 to 5"""
    # we create output directory
    output_base = Path("experiments/table/results_simple_cosine_L2")
    output_base.mkdir(parents=True, exist_ok=True)
    
    factors = [1, 2, 3, 4, 5]
    
    print("="*80)
    print("SIMPLE COSINE TEST: L=2, cos(2*factor*pi*x)")
    print("="*80)
    print(f"Testing factors: {factors}")
    print(f"Max epochs per config: 10000")
    print(f"Early stopping: loss < 2e-5")
    print(f"Output directory: {output_base}")
    print("="*80)
    
    all_results = []
    
    for factor in factors:
        output_dir = output_base / f"factor{factor}_L2"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = train_one_config(factor, output_dir)
        all_results.append(results)
    
    # we save summary
    summary = {
        'experiment': 'Simple cosine test: L=2, cos(2*factor*pi*x)',
        'factors_tested': factors,
        'num_layers': 2,
        'num_epochs': 10000,
        'results': all_results
    }
    
    with open(output_base / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETED")
    print("="*80)
    print(f"\nSummary:")
    for r in all_results:
        print(f"  factor={r['factor']}: min_loss={r['min_loss']:.6e} at epoch {r['min_loss_epoch']}")
    print(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()
