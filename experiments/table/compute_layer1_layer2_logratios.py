#!/usr/bin/env python3
"""
Script to compute log ratios for layer 1 and layer 2 at multiple x values.
Only for factor=4, rank=15.
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
plt.rcParams['figure.figsize'] = [12, 10]
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

def compute_log_ratios_layer(model, layer_idx, x_location=0.0, device="cpu", mydtype=torch.float32, epsilon=1e-6):
    """
    we compute log ratios for layer partial functions at x=x_location
    layer_idx=1: fcs[1] (first low-rank layer)
    layer_idx=2: fcs[3] (second low-rank layer)
    returns: log_ratio_matrix [r, r] where R[i,j] = log(|f_i|) - log(|f_j|)
    """
    model.eval()
    with torch.no_grad():
        # we use small epsilon near 0 to avoid exact zero
        if abs(x_location) < 1e-6:
            x_location = epsilon
        
        # we create input tensor at x_location
        x_tensor = torch.tensor([[x_location]], device=device, dtype=mydtype)
        
        # we forward pass to the specified layer
        # for L=2: fcs[0] rank→width, fcs[1] width→rank (layer 1), fcs[2] rank→width, fcs[3] width→rank (layer 2)
        current = x_tensor
        if layer_idx == 1:
            # we go up to fcs[1] (first low-rank layer)
            for i in range(2):
                current = model.fcs[i](current)
                if i % 2 == 0:  # we apply ReLU after rank→width
                    current = torch.relu(current)
        elif layer_idx == 2:
            # we go up to fcs[3] (second low-rank layer)
            for i in range(4):
                current = model.fcs[i](current)
                if i % 2 == 0:  # we apply ReLU after rank→width
                    current = torch.relu(current)
        
        # we extract layer output (partial functions)
        f_k = current.cpu().numpy().flatten()  # [r] - one value per channel at x=x_location
        
        # we compute log ratios R[i,j] = log(|f_i|) - log(|f_j|)
        r = len(f_k)
        eps = 1e-10
        log_f_k = np.log(np.abs(f_k) + eps)
        R = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                R[i, j] = log_f_k[i] - log_f_k[j]
        
        return R, f_k

def plot_log_ratio_statistics_positive(log_ratio_matrix, output_path, config_name, x_value, epsilon, layer_name):
    """we plot statistics of log ratios (only positive values)"""
    R = np.array(log_ratio_matrix)
    R_clean = R[np.isfinite(R)]
    R_positive = R_clean[R_clean > 0]  # we keep only positive values
    
    if len(R_positive) == 0:
        print(f"   ⚠️  No positive log ratios for {config_name} at x={x_value}, skipping")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # we plot histogram (only positive values)
    ax = axes[0]
    ax.hist(R_positive, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')
    ax.set_xlabel('$R_{i,j}$ (positive values only)', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    title = f'Distribution of Positive Log Ratios at $x={x_value}$\nLayer: {layer_name}'
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='$R=0$')
    ax.legend(fontsize=14)
    
    # we plot statistics text
    ax = axes[1]
    ax.axis('off')
    stats_text = f"""
    Statistics of Positive Log Ratios $R_{{i,j}} = \\log(|f_i|) - \\log(|f_j|)$ at $x={x_value}$:
    
    Layer: {layer_name}
    Epsilon: $\\epsilon = {epsilon}$
    
    Mean: {np.mean(R_positive):.4f}
    Std:  {np.std(R_positive):.4f}
    Min:  {np.min(R_positive):.4f}
    Max:  {np.max(R_positive):.4f}
    
    Number of positive pairs: {len(R_positive)} / {len(R_clean)}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=16, verticalalignment='center', 
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # we add config name at the bottom
    fig.text(0.5, 0.02, f'{config_name}', ha='center', fontsize=14, wrap=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved statistics: {output_path}")

if __name__ == "__main__":
    results_base = Path("/Data/janis.aiad/MMNN/experiments/table/experiments/table/results_tune_lr_decay_L2")
    
    # we only process factor=4, rank=15
    target_factor = 4
    target_rank = 15
    
    # we find matching config directories
    config_dirs = []
    for config_dir in results_base.iterdir():
        if not config_dir.is_dir():
            continue
        config_file = config_dir / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                if config.get('factor') == target_factor and config.get('hidden_rank') == target_rank:
                    config_dirs.append(config_dir)
            except:
                continue
    
    print(f"Found {len(config_dirs)} configurations with factor={target_factor}, rank={target_rank}")
    
    # we define x values to compute
    x_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    epsilon = 1e-6
    
    for config_dir in config_dirs:
        config_name = config_dir.name
        print(f"\n{'='*80}")
        print(f"Processing: {config_name}")
        print(f"{'='*80}")
        
        config_file = config_dir / 'config.json'
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            factor = config['factor']
            hidden_rank = config['hidden_rank']
            hidden_width = config.get('hidden_width', 1024)
            num_layers = config.get('num_layers', 2)
            
            # we need to re-train the model (models are not saved, only results)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mydtype = torch.float32
            
            print(f"   Re-training model (this may take a while)...")
            
            # we set up model
            ranks = [1] + [hidden_rank] * num_layers + [1]
            widths = [hidden_width] * (num_layers + 1)
            
            model = MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True)
            
            # we create training data
            interval = [-1, 1]
            n_train = max(1, int(factor * hidden_width))
            x_train = np.linspace(interval[0], interval[1], n_train)
            y_train = target_function(x_train, factor)
            
            x_train_tensor = torch.tensor(x_train.reshape([-1, 1]), device=device, dtype=mydtype)
            y_train_tensor = torch.tensor(y_train.reshape([-1, 1]), device=device, dtype=mydtype)
            
            # we set up optimizer from config
            lr_init = config.get('lr_init', 0.01)
            optimizer_type = config.get('optimizer_type', 'Adam')
            scheduler_type = config.get('scheduler_type', None)
            scheduler_params = config.get('scheduler_params', {})
            
            if optimizer_type == 'SGD':
                momentum = config.get('momentum', 0.9)
                optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
            elif optimizer_type == 'Adam':
                betas = config.get('betas', (0.9, 0.999))
                optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=betas)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr_init)
            
            # we train (simplified - just enough to get trained model)
            num_epochs = 10000
            batch_size = max(1, int(4 * factor * 10))
            criterion = nn.MSELoss()
            
            print(f"   Training for up to {num_epochs} epochs (early stop at loss < 2e-5)...")
            for epoch in tqdm(range(num_epochs), desc=f"Training {config_name}"):
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
                
                # we early stop if loss < 2e-5
                if avg_loss < 2e-5:
                    print(f"   Early stopping at epoch {epoch} (loss={avg_loss:.6e})")
                    break
            
            print(f"   ✅ Model trained (final loss: {avg_loss:.6e})")
            
            # we compute log ratios for both layers at all x values
            all_results = {}
            
            for layer_idx in [1, 2]:
                layer_name = f"Layer {layer_idx}"
                print(f"\n   Computing log ratios for {layer_name}...")
                
                layer_results = {}
                for x_val in x_values:
                    print(f"      x = {x_val}...")
                    try:
                        R, f_k = compute_log_ratios_layer(model, layer_idx, x_val, device, mydtype, epsilon)
                        
                        # we compute statistics without storing full matrix
                        R_clean = R[np.isfinite(R)]
                        R_positive = R_clean[R_clean > 0]
                        
                        # we save matrix to .npy file (compact binary format)
                        matrix_file = config_dir / f'layer{layer_idx}_logratio_matrix_x{x_val}.npy'
                        np.save(matrix_file, R)
                        
                        # we save f_k values to .npy as well
                        fk_file = config_dir / f'layer{layer_idx}_fk_values_x{x_val}.npy'
                        np.save(fk_file, f_k)
                        
                        # we store only statistics in JSON (not full matrices)
                        layer_results[f'x_{x_val}'] = {
                            'x': float(x_val),
                            'epsilon': epsilon,
                            'matrix_file': str(matrix_file.name),
                            'fk_file': str(fk_file.name),
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
                        
                        # we plot statistics (only positive values)
                        plot_path = config_dir / f'layer{layer_idx}_logratio_statistics_x{x_val}_positive.png'
                        plot_log_ratio_statistics_positive(R, plot_path, config_name, x_val, epsilon, layer_name)
                        
                    except Exception as e:
                        print(f"      ❌ Error at x={x_val}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                all_results[f'layer_{layer_idx}'] = layer_results
            
            # we save all results
            results_file = config_dir / f'layer1_layer2_logratios_all_x.json'
            with open(results_file, 'w') as f:
                json.dump({
                    'factor': factor,
                    'rank': hidden_rank,
                    'x_values': x_values,
                    'epsilon': epsilon,
                    'results': all_results
                }, f, indent=2)
            
            print(f"   ✅ Saved all results to {results_file}")
            
        except Exception as e:
            print(f"   ❌ Error processing {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Done!")
