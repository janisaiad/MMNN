#!/usr/bin/env python3
"""
we implement channel specialization metrics for cos(12πx) using MMNN architecture
we compute log ratios at the maximum of the function for 2nd, 4th, and 6th low-rank layers
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN

# we configure matplotlib
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12


class CosineFunction:
    """we create a cosine function cos(12πx)"""
    def __init__(self, frequency=12):
        """
        we create cosine function: y(x) = cos(12πx)
        frequency: frequency parameter (default 12)
        """
        self.frequency = frequency
    
    def __call__(self, x):
        """we evaluate the cosine function"""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        result = np.cos(self.frequency * np.pi * x_np)
        
        if isinstance(x, torch.Tensor):
            return torch.tensor(result, dtype=x.dtype, device=x.device)
        return result
    
    def get_maximum_location(self):
        """we find the location of maximum (cos(12πx) = 1 when 12πx = 2kπ, so x = k/6)"""
        # we use x=0 as the maximum (cos(0) = 1)
        return 0.0
    
    def get_training_dataset(self, n_samples=1000, x_range=(-1, 1)):
        """we create training dataset on [-1, 1]"""
        x_points = np.linspace(x_range[0], x_range[1], n_samples)
        y_points = self(x_points)
        return x_points, y_points


class ChannelSpecializationMetrics:
    """we compute channel specialization metrics for MMNN low-rank layers"""
    def __init__(self, epsilon=1e-8):
        """
        we initialize metrics
        epsilon: small constant to avoid division by zero
        """
        self.epsilon = epsilon
    
    def compute_channel_shares(self, f_k):
        """
        we compute channel shares s_k = |f_k| / (sum_j |f_j| + epsilon)
        f_k: [r, batch_size] - partial functions for each channel
        """
        abs_f = torch.abs(f_k)  # [r, batch_size]
        sum_abs = torch.sum(abs_f, dim=0, keepdim=True)  # [1, batch_size]
        shares = abs_f / (sum_abs + self.epsilon)  # [r, batch_size]
        return shares  # [r, batch_size]
    
    def compute_log_ratios(self, f_k):
        """
        we compute log ratios R_{k,ell} = log(|f_k| + epsilon) - log(|f_ell| + epsilon)
        f_k: [r, batch_size] - partial functions for each channel
        """
        r = f_k.shape[0]
        abs_f = torch.abs(f_k) + self.epsilon  # [r, batch_size]
        log_f = torch.log(abs_f)  # [r, batch_size]
        
        log_ratios = torch.zeros(r, r, f_k.shape[1], device=f_k.device)  # [r, r, batch_size]
        for k in range(r):
            for ell in range(r):
                log_ratios[k, ell] = log_f[k] - log_f[ell]
        
        return log_ratios  # [r, r, batch_size]
    
    def compute_dominance_metrics(self, f_k, x_location):
        """
        we compute dominance metrics at a specific location
        f_k: [r, batch_size] - partial functions
        x_location: location to evaluate (scalar)
        """
        # we assume f_k is already evaluated at x_location, so batch_size=1
        shares = self.compute_channel_shares(f_k)  # [r, 1]
        log_ratios = self.compute_log_ratios(f_k)  # [r, r, 1]
        
        metrics = {
            'location': x_location,
            'shares': shares.squeeze(1).detach().cpu().numpy(),  # [r]
            'log_ratios': log_ratios.squeeze(2).detach().cpu().numpy()  # [r, r]
        }
        
        return metrics


def extract_partial_functions_mmnn(model, x, layer_indices=[3, 7, 11]):
    """
    we extract partial functions from MMNN at specified low-rank layers
    layer_indices: list of indices for low-rank layers (fcs[3], fcs[7], fcs[11] for 2nd, 4th, 6th)
    returns: dict mapping layer_idx -> [r, batch_size] partial functions
    """
    model.eval()
    with torch.no_grad():
        # we handle input shape
        if x.dim() == 1:
            x = x.unsqueeze(1)
        batch_size = x.shape[0]
        
        # we store outputs at each low-rank layer
        partial_functions = {}
        current_x = x
        
        # we forward pass and extract outputs at low-rank layers
        depth = model.depth
        for j in range(depth):
            # we compute random features (frozen)
            current_x = model.fcs[2*j](current_x)  # rank→width
            current_x = torch.relu(current_x)
            
            # we compute low-rank mixing (trainable)
            low_rank_idx = 2*j + 1
            current_x = model.fcs[low_rank_idx](current_x)  # width→rank, output shape: [batch_size, rank]
            
            # we extract partial functions if this is a layer we're interested in
            if low_rank_idx in layer_indices:
                # the output of the low-rank layer is [batch_size, rank]
                # each column k is the partial function f_k
                # we transpose to get [rank, batch_size]
                partial_functions[low_rank_idx] = current_x.t()  # [rank, batch_size]
        
        return partial_functions


def train_mmnn_model(X_train, y_train, ranks, widths, device="cpu", 
                     num_epochs=5000, batch_size=100, lr=0.001, fixWb=True):
    """
    we train MMNN model
    """
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=False,
        fixWb=fixWb
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # we create dataset and dataloader
    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    dataset = SimpleDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            if batch_X.dim() == 1:
                batch_X = batch_X.unsqueeze(1)
            if batch_y.dim() == 0:
                batch_y = batch_y.unsqueeze(0)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses


def run_experiment():
    """we run the cosine experiment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # we create cosine function
    cosine_func = CosineFunction(frequency=12)
    x_max = cosine_func.get_maximum_location()  # x=0 where cos(0)=1
    
    # we create training dataset
    x_train, y_train = cosine_func.get_training_dataset(n_samples=1000, x_range=(-1, 1))
    print(f"\nTraining dataset: {len(x_train)} points on [-1, 1]")
    print(f"Target function: cos(12πx)")
    print(f"Maximum location: x={x_max}, value={cosine_func(x_max):.6f}")
    
    # we set up MMNN architecture: L=6, width=1024, rank=2
    L = 6  # depth
    width = 1024  # width
    rank = 2  # rank
    ranks = [1] + [rank] * L + [1]  # [1, 2, 2, 2, 2, 2, 2, 1]
    widths = [width] * (L + 1)  # [1024, 1024, 1024, 1024, 1024, 1024, 1024]
    
    print(f"\nMMNN Architecture:")
    print(f"  Depth: L={L}")
    print(f"  Width: {width}")
    print(f"  Rank: {rank}")
    print(f"  Ranks: {ranks}")
    print(f"  Widths: {widths}")
    
    # we train model
    print("\n" + "="*80)
    print("Training MMNN Model...")
    print("="*80)
    model, losses = train_mmnn_model(
        x_train, y_train, ranks, widths, device=device,
        num_epochs=5000, batch_size=100, lr=0.001, fixWb=True
    )
    
    # we compute partial functions at maximum location
    print("\n" + "="*80)
    print("Computing Channel Specialization Metrics...")
    print("="*80)
    
    x_max_tensor = torch.tensor([[x_max]], dtype=torch.float32, device=device)
    layer_indices = [3, 7, 11]  # 2nd, 4th, 6th low-rank layers
    layer_names = {3: "2nd", 7: "4th", 11: "6th"}
    
    partial_functions = extract_partial_functions_mmnn(model, x_max_tensor, layer_indices)
    
    spec_metrics = ChannelSpecializationMetrics()
    results = {}
    
    for layer_idx in layer_indices:
        if layer_idx in partial_functions:
            f_k = partial_functions[layer_idx]  # [rank, 1]
            metrics = spec_metrics.compute_dominance_metrics(f_k, x_max)
            
            layer_name = layer_names[layer_idx]
            results[f'layer_{layer_name}'] = {
                'layer_idx': layer_idx,
                'metrics': metrics
            }
            
            print(f"\n{layer_name} low-rank layer (fcs[{layer_idx}]):")
            print(f"  Location: x={x_max:.6f}")
            print(f"  Channel shares: {metrics['shares']}")
            print(f"  Log ratios R_{0,1}: {metrics['log_ratios'][0, 1]:.6f}")
            print(f"  Log ratios R_{1,0}: {metrics['log_ratios'][1, 0]:.6f}")
    
    # we create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations...")
    print("="*80)
    
    output_dir = Path(__file__).parent / "meanfield_cosine_results"
    output_dir.mkdir(exist_ok=True)
    
    # we plot 1: predictions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # we plot predictions
    ax = axes[0, 0]
    x_fine = np.linspace(-1, 1, 200)
    y_fine = cosine_func(x_fine)
    x_fine_tensor = torch.tensor(x_fine, dtype=torch.float32, device=device).unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_fine_tensor).squeeze(1).cpu().numpy()
    
    ax.plot(x_fine, y_fine, 'k-', label='Target cos(12πx)', linewidth=2)
    ax.plot(x_fine, y_pred, 'b--', label='MMNN Prediction', linewidth=2)
    ax.axvline(x_max, color='red', linestyle=':', linewidth=2, label=f'Maximum at x={x_max}')
    ax.scatter(x_train[::50], y_train[::50], color='gray', s=10, alpha=0.5, label='Training points (sampled)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('MMNN Prediction vs Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # we plot loss evolution
    ax = axes[0, 1]
    ax.semilogy(range(1, len(losses)+1), losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss Evolution')
    ax.grid(True, alpha=0.3)
    
    # we plot channel shares at maximum for different layers
    ax = axes[1, 0]
    layer_names_list = []
    shares_ch0 = []
    shares_ch1 = []
    for layer_idx in layer_indices:
        if f'layer_{layer_names[layer_idx]}' in results:
            layer_name = layer_names[layer_idx]
            layer_names_list.append(layer_name)
            shares = results[f'layer_{layer_name}']['metrics']['shares']
            shares_ch0.append(shares[0])
            shares_ch1.append(shares[1])
    
    x_pos = np.arange(len(layer_names_list))
    width = 0.35
    ax.bar(x_pos - width/2, shares_ch0, width, label='Channel 0', alpha=0.8)
    ax.bar(x_pos + width/2, shares_ch1, width, label='Channel 1', alpha=0.8)
    ax.set_xlabel('Low-Rank Layer')
    ax.set_ylabel('Channel Share')
    ax.set_title('Channel Shares at Maximum (x=0)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{name} (fcs[{layer_indices[i]}])' for i, name in enumerate(layer_names_list)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # we plot log ratios at maximum for different layers
    ax = axes[1, 1]
    log_ratios_01 = []
    log_ratios_10 = []
    for layer_idx in layer_indices:
        if f'layer_{layer_names[layer_idx]}' in results:
            log_ratios = results[f'layer_{layer_names[layer_idx]}']['metrics']['log_ratios']
            log_ratios_01.append(log_ratios[0, 1])
            log_ratios_10.append(log_ratios[1, 0])
    
    ax.plot(layer_names_list, log_ratios_01, 'b-o', label='R_{0,1} = log(|f_0|) - log(|f_1|)', linewidth=2)
    ax.plot(layer_names_list, log_ratios_10, 'r-s', label='R_{1,0} = log(|f_1|) - log(|f_0|)', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Low-Rank Layer')
    ax.set_ylabel('Log-Ratio')
    ax.set_title('Log-Ratios at Maximum (x=0)')
    ax.set_xticklabels([f'{name}' for name in layer_names_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_channel_specialization.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'cosine_channel_specialization.png'}")
    
    # we save results
    def convert_to_serializable(obj):
        """we convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    save_results = {
        'target_function': 'cos(12πx)',
        'maximum_location': float(x_max),
        'maximum_value': float(cosine_func(x_max)),
        'architecture': {
            'depth': L,
            'width': width,
            'rank': rank,
            'ranks': ranks,
            'widths': widths
        },
        'training': {
            'num_epochs': 5000,
            'batch_size': 100,
            'lr': 0.001,
            'final_loss': float(losses[-1]) if len(losses) > 0 else None
        },
        'results': convert_to_serializable(results)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=4)
    print(f"Saved results to {output_dir / 'results.json'}")
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)


if __name__ == "__main__":
    run_experiment()
