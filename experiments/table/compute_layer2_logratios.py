#!/usr/bin/env python3
"""
Script to re-run trainings and compute log ratios for layer 2 partial functions at x=0.
Also replots loss evolution.
Skips GIF generation and other PNG plots.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LinearLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from pathlib import Path
import sys
from tqdm import tqdm

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN

# we configure matplotlib for LaTeX
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
    """Cosine function: cos(2*factor*pi*x) + cos(2*pi*x)"""
    return np.cos(2 * factor * np.pi * x) + np.cos(2 * np.pi * x)

def compute_log_ratios_layer2(model, x_location=0.0, device="cpu", mydtype=torch.float32):
    """
    we compute log ratios for layer 2 partial functions at x=x_location
    layer 2 corresponds to layer_idx=3 (output after fcs[3], which is the second low-rank layer)
    returns: log_ratio_matrix [r, r] where R[i,j] = log(|f_i|) - log(|f_j|)
    """
    model.eval()
    with torch.no_grad():
        # we use small epsilon near 0 to avoid exact zero (like in mean-field)
        if abs(x_location) < 1e-6:
            x_location = 1e-6
        
        # we create input tensor at x_location
        x_tensor = torch.tensor([[x_location]], device=device, dtype=mydtype)
        
        # we forward pass to layer 2 (layer_idx=3)
        # for L=2: fcs[0] rank→width, fcs[1] width→rank (layer 1), fcs[2] rank→width, fcs[3] width→rank (layer 2)
        current = x_tensor
        for i in range(4):  # we go up to fcs[3] (need to include fcs[3])
            current = model.fcs[i](current)
            if i % 2 == 0:  # we apply ReLU after rank→width
                current = torch.relu(current)
        
        # we extract layer 2 output (partial functions)
        f_k = current.cpu().numpy().flatten()  # [r] - one value per channel at x=x_location
        
        # we compute log ratios R[i,j] = log(|f_i|) - log(|f_j|)
        r = len(f_k)
        epsilon = 1e-10
        log_f_k = np.log(np.abs(f_k) + epsilon)
        R = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                R[i, j] = log_f_k[i] - log_f_k[j]
        
        return R, f_k

def plot_loss_evolution(all_losses, lr_reduction_epochs, output_dir, factor, final_optimizer_type, 
                        switched_to_sgd, scheduler_type, lr_init, batch_size, sgd_momentum, lr_config):
    """we replot loss evolution"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
    
    # we add red vertical bars at LR reduction moments
    if lr_reduction_epochs:
        for reduction_epoch in lr_reduction_epochs:
            if reduction_epoch < len(all_losses):
                ax.axvline(x=reduction_epoch, color='r', linestyle='--', linewidth=1.5, alpha=0.7, 
                          label='LR reduction' if reduction_epoch == lr_reduction_epochs[0] else '')
    
    # we add horizontal line for early stopping threshold
    ax.axhline(y=2e-5, color='g', linestyle=':', linewidth=1, alpha=0.5, label='Early stop threshold')
    
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    title = f'Loss Evolution - factor={factor}, {final_optimizer_type}'
    if switched_to_sgd:
        title += ' (Adam→SGD)'
    if scheduler_type:
        title += f', {scheduler_type}'
    title += f'\nBatch size={batch_size}, LR init={lr_init}'
    if final_optimizer_type == 'SGD':
        title += f', Momentum={sgd_momentum if switched_to_sgd else lr_config.get("momentum", 0.9)}'
    elif not switched_to_sgd:
        title += f', Betas={lr_config.get("betas", (0.9, 0.999))}'
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Replotted loss evolution: {output_dir / 'loss_evolution.png'}")

def train_one_config_with_logratios(factor, lr_config, output_dir):
    """we train one configuration and compute log ratios for layer 2"""
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
    hidden_rank = lr_config.get('hidden_rank', 15)
    num_layers = 2
    input_rank = 1
    output_rank = 1
    
    ranks = [input_rank] + [hidden_rank] * num_layers + [output_rank]
    widths = [hidden_width] * (num_layers + 1)
    
    model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
    
    # we create training data
    interval = [-1, 1]
    n_train = max(1, int(factor * hidden_width))
    x_train = np.linspace(interval[0], interval[1], n_train)
    y_train = target_function(x_train, factor)
    
    x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
    y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we set up optimizer
    optimizer_type = lr_config.get('optimizer_type', 'Adam')
    switched_to_sgd = False
    sgd_momentum = lr_config.get('momentum', 0.9)
    
    if optimizer_type == 'SGD':
        momentum = lr_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
    elif optimizer_type == 'Adam':
        betas = lr_config.get('betas', (0.9, 0.999))
        optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=betas)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_init)
    
    # we set up scheduler
    scheduler = None
    adaptive_scheduler = None
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])
    elif scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=scheduler_params['gamma'])
    elif scheduler_type == 'LinearLR':
        start_factor = scheduler_params.get('start_factor', 1.0)
        end_factor = scheduler_params.get('end_factor', 0.0)
        total_iters = scheduler_params.get('total_iters', 10000)
        scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    elif scheduler_type == 'AdaptiveStagnation':
        adaptive_scheduler = {
            'lr_sequence': scheduler_params.get('lr_sequence', [0.01, 0.005, 0.001, 0.0005, 0.0001]),
            'patience': scheduler_params.get('patience', 500),
            'min_delta': scheduler_params.get('min_delta', 1e-6),
            'lr_reduction_epochs': []
        }
    
    # we train
    num_epochs = 10000
    batch_size = max(1, int(4 * factor * 10))
    all_losses = []
    all_lrs = []
    min_loss = float('inf')
    min_loss_epoch = 0
    
    criterion = nn.MSELoss()
    
    print(f"\nTraining for {num_epochs} epochs, batch_size={batch_size}...")
    for epoch in tqdm(range(num_epochs), desc=f"Training factor={factor}"):
        # we switch to SGD when loss < 1e-3 (if using Adam first)
        if optimizer_type == 'Adam' and not switched_to_sgd:
            current_loss = criterion(model(x_train_tensor), y_train_tensor).item()
            if current_loss < 1e-3:
                print(f"\n   Switching to SGD at epoch {epoch} (loss={current_loss:.6e})")
                optimizer = optim.SGD(model.parameters(), lr=optimizer.param_groups[0]['lr'], momentum=sgd_momentum)
                switched_to_sgd = True
        
        # we train one epoch
        model.train()
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            batch_indices = indices[batch_start:batch_end]
            x_batch = x_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        all_losses.append(avg_loss)
        all_lrs.append(optimizer.param_groups[0]['lr'])
        
        if avg_loss < min_loss:
            min_loss = avg_loss
            min_loss_epoch = epoch
        
        # we early stop if loss < 2e-5
        if avg_loss < 2e-5:
            print(f"\n   Early stopping at epoch {epoch} (loss={avg_loss:.6e})")
            break
        
        # we update scheduler
        if scheduler is not None:
            scheduler.step()
        elif adaptive_scheduler is not None:
            # we check for stagnation
            if len(all_losses) >= adaptive_scheduler['patience'] + 1:
                recent_losses = all_losses[-adaptive_scheduler['patience']:]
                if len(recent_losses) > 1:
                    loss_improvement = recent_losses[0] - recent_losses[-1]
                    if loss_improvement < adaptive_scheduler['min_delta']:
                        # we reduce LR
                        current_lr = optimizer.param_groups[0]['lr']
                        lr_sequence = adaptive_scheduler['lr_sequence']
                        if current_lr in lr_sequence:
                            current_idx = lr_sequence.index(current_lr)
                            if current_idx < len(lr_sequence) - 1:
                                new_lr = lr_sequence[current_idx + 1]
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = new_lr
                                adaptive_scheduler['lr_reduction_epochs'].append(epoch)
                                print(f"   LR reduced to {new_lr} at epoch {epoch}")
    
    # we compute log ratios at x=0 for layer 2
    print(f"\n📊 Computing log ratios for layer 2 at x=0...")
    log_ratio_matrix, f_k_values = compute_log_ratios_layer2(model, x_location=0.0, device=device, mydtype=mydtype)
    
    # we save matrices to .npy files (compact binary format)
    matrix_file = output_dir / 'layer2_logratio_matrix_x0.npy'
    np.save(matrix_file, log_ratio_matrix)
    fk_file = output_dir / 'layer2_fk_values_x0.npy'
    np.save(fk_file, f_k_values)
    
    # we compute statistics
    R_clean = log_ratio_matrix[np.isfinite(log_ratio_matrix)]
    R_positive = R_clean[R_clean > 0]
    
    # we save only statistics in JSON (not full matrices to avoid huge files)
    log_ratios_data = {
        'x_location': 0.0,
        'layer': 2,
        'rank': hidden_rank,
        'matrix_file': 'layer2_logratio_matrix_x0.npy',
        'fk_file': 'layer2_fk_values_x0.npy',
        'statistics': {
            'mean': float(np.mean(R_clean)) if len(R_clean) > 0 else None,
            'std': float(np.std(R_clean)) if len(R_clean) > 0 else None,
            'min': float(np.min(R_clean)) if len(R_clean) > 0 else None,
            'max': float(np.max(R_clean)) if len(R_clean) > 0 else None,
            'n_total': int(len(R_clean)),
            'n_positive': int(len(R_positive)),
            'mean_positive': float(np.mean(R_positive)) if len(R_positive) > 0 else None,
            'std_positive': float(np.std(R_positive)) if len(R_positive) > 0 else None,
            'min_positive': float(np.min(R_positive)) if len(R_positive) > 0 else None,
            'max_positive': float(np.max(R_positive)) if len(R_positive) > 0 else None
        }
    }
    with open(output_dir / 'layer2_logratios_x0.json', 'w') as f:
        json.dump(log_ratios_data, f, indent=2)
    print(f"   ✅ Saved log ratios (statistics only) to {output_dir / 'layer2_logratios_x0.json'}")
    print(f"   ✅ Saved matrices to {matrix_file.name} and {fk_file.name}")
    
    # we replot loss evolution
    print(f"\n📈 Replotting loss evolution...")
    lr_reduction_epochs = adaptive_scheduler['lr_reduction_epochs'] if adaptive_scheduler else []
    final_optimizer_type = 'SGD' if switched_to_sgd else optimizer_type
    plot_loss_evolution(all_losses, lr_reduction_epochs, output_dir, factor, final_optimizer_type,
                       switched_to_sgd, scheduler_type, lr_init, batch_size, sgd_momentum, lr_config)
    
    # we update results.json with log ratios info
    results_file = output_dir / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    results['layer2_logratios_x0'] = {
        'log_ratio_matrix': log_ratio_matrix.tolist(),
        'f_k_values': f_k_values.tolist(),
        'mean_log_ratio': float(np.mean(log_ratio_matrix)),
        'max_log_ratio': float(np.max(log_ratio_matrix)),
        'min_log_ratio': float(np.min(log_ratio_matrix)),
        'std_log_ratio': float(np.std(log_ratio_matrix))
    }
    results['all_losses'] = all_losses
    results['all_lrs'] = all_lrs
    results['lr_reduction_epochs'] = lr_reduction_epochs
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Updated results.json")
    
    return results

if __name__ == "__main__":
    # we find all config directories
    results_base = Path("/Data/janis.aiad/MMNN/experiments/table/experiments/table/results_tune_lr_decay_L2")
    
    config_dirs = [d for d in results_base.iterdir() if d.is_dir() and (d / 'config.json').exists()]
    
    print(f"Found {len(config_dirs)} configurations to process")
    
    for config_dir in tqdm(config_dirs, desc="Processing configs"):
        config_file = config_dir / 'config.json'
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            factor = config['factor']
            lr_config = {
                'lr_init': config['lr_init'],
                'optimizer_type': config.get('optimizer_type', 'Adam'),
                'scheduler_type': config.get('scheduler_type', None),
                'scheduler_params': config.get('scheduler_params', {}),
                'momentum': config.get('momentum', None),
                'betas': config.get('betas', None),
                'hidden_rank': config.get('hidden_rank', 15)
            }
            
            train_one_config_with_logratios(factor, lr_config, config_dir)
            
        except Exception as e:
            print(f"   ❌ Error processing {config_dir}: {e}")
            continue
    
    print(f"\n✅ Done! Processed {len(config_dirs)} configurations")
