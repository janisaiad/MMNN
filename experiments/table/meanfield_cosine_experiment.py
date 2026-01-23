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
                     num_epochs=5000, batch_size=100, lr=0.001, fixWb=True,
                     save_dir=None, rank=None):
    """
    we train MMNN model and save checkpoints every 100 epochs
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
    weight_snapshots = []  # we store weight snapshots at different epochs
    snapshot_epochs = [0, 1000, 2000, 3000, 4000, 5000]  # we snapshot at these epochs
    
    # we create checkpoint directory if provided
    if save_dir is not None and rank is not None:
        checkpoint_dir = Path(save_dir) / f"checkpoints_rank{rank}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # we save initial checkpoint (epoch 0)
        checkpoint_path = checkpoint_dir / "model_epoch_0.pth"
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': None,
            'ranks': ranks,
            'widths': widths
        }, checkpoint_path)
    
    # we snapshot initial weights (epoch 0) for visualization
    snapshot = {}
    for idx, fc in enumerate(model.fcs):
        if idx % 2 == 1:  # we only track trainable layers (odd indices: low-rank mixing)
            layer_name = f"fcs[{idx}]"
            snapshot[layer_name] = {
                'weight': fc.weight.data.clone().cpu().numpy(),
                'bias': fc.bias.data.clone().cpu().numpy() if fc.bias is not None else None
            }
    weight_snapshots.append({
        'epoch': 0,
        'weights': snapshot
    })
    
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
        
        # we save model checkpoint every 100 epochs
        if save_dir is not None and rank is not None and epoch % 100 == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'ranks': ranks,
                'widths': widths
            }, checkpoint_path)
        
        # we snapshot weights at specified epochs (for visualization)
        if epoch in snapshot_epochs:
            snapshot = {}
            for idx, fc in enumerate(model.fcs):
                if idx % 2 == 1:  # we only track trainable layers (odd indices: low-rank mixing)
                    layer_name = f"fcs[{idx}]"
                    snapshot[layer_name] = {
                        'weight': fc.weight.data.clone().cpu().numpy(),
                        'bias': fc.bias.data.clone().cpu().numpy() if fc.bias is not None else None
                    }
            weight_snapshots.append({
                'epoch': epoch,
                'weights': snapshot
            })
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses, weight_snapshots


def run_experiment():
    """we run the cosine experiment with multiple ranks and locations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # we create cosine function
    cosine_func = CosineFunction(frequency=12)
    
    # we create training dataset
    x_train, y_train = cosine_func.get_training_dataset(n_samples=300, x_range=(-1, 1))
    print(f"\nTraining dataset: {len(x_train)} points on [-1, 1]")
    print(f"Target function: cos(12πx)")
    
    # we set up MMNN architecture: L=6, width=1024, multiple ranks
    L = 6  # depth
    width = 1024  # width
    rank_list = [2, 5, 10, 15, 20, 25]  # we test multiple ranks
    
    # we define locations to analyze
    x_locations = [0.0, 0.5, -0.5]
    location_names = {0.0: "x0", 0.5: "x05", -0.5: "x_05"}
    
    # we define layers to analyze (2nd and 4th only)
    layer_indices = [3, 7]  # 2nd and 4th low-rank layers
    layer_names = {3: "2nd", 7: "4th"}
    
    # we set up output directory
    output_dir = Path(__file__).parent / "meanfield_cosine_results"
    output_dir.mkdir(exist_ok=True)
    
    # we train models for each rank and compute metrics
    all_results = {}  # {rank: {location: {layer: metrics}}}
    all_weight_snapshots = {}  # {rank: weight_snapshots}
    
    for rank in rank_list:
        print(f"\n{'='*80}")
        print(f"Training MMNN with rank={rank}")
        print(f"{'='*80}")
        
        ranks = [1] + [rank] * L + [1]
        widths = [width] * (L + 1)
        
        print(f"  Architecture: L={L}, width={width}, rank={rank}")
        print(f"  Saving checkpoints every 100 epochs to {output_dir / f'checkpoints_rank{rank}'}")
        
        # we train model
        model, losses, weight_snapshots = train_mmnn_model(
            x_train, y_train, ranks, widths, device=device,
            num_epochs=5000, batch_size=100, lr=0.001, fixWb=True,
            save_dir=output_dir, rank=rank
        )
        
        # we store weight snapshots
        all_weight_snapshots[rank] = weight_snapshots
        
        # we compute metrics at each location
        all_results[rank] = {}
        spec_metrics = ChannelSpecializationMetrics()
        
        for x_loc in x_locations:
            print(f"\n  Computing metrics at x={x_loc:.1f}...")
            x_tensor = torch.tensor([[x_loc]], dtype=torch.float32, device=device)
            partial_functions = extract_partial_functions_mmnn(model, x_tensor, layer_indices)
            
            all_results[rank][x_loc] = {}
            
            for layer_idx in layer_indices:
                if layer_idx in partial_functions:
                    f_k = partial_functions[layer_idx]  # [rank, 1]
                    metrics = spec_metrics.compute_dominance_metrics(f_k, x_loc)
                    
                    layer_name = layer_names[layer_idx]
                    all_results[rank][x_loc][layer_name] = {
                        'layer_idx': layer_idx,
                        'metrics': metrics
                    }
                    
                    # we compute statistics across all log ratios
                    log_ratios = metrics['log_ratios']
                    triu_indices = np.triu_indices_from(log_ratios, k=1)
                    all_log_ratios = log_ratios[triu_indices]
                    
                    print(f"    {layer_name} layer: mean_log_ratio={np.mean(all_log_ratios):.4f}, max={np.max(all_log_ratios):.4f}")
    
    # we create 3 separate PNG files, one for each location
    output_dir = Path(__file__).parent / "meanfield_cosine_results"
    output_dir.mkdir(exist_ok=True)
    
    for x_loc in x_locations:
        print(f"\n{'='*80}")
        print(f"Creating plots for x={x_loc:.1f}...")
        print(f"{'='*80}")
        
        # we create comprehensive figure for this location
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # we plot 1: target function with location marked
        ax = fig.add_subplot(gs[0, :])
        x_fine = np.linspace(-1, 1, 200)
        y_fine = cosine_func(x_fine)
        ax.plot(x_fine, y_fine, 'k-', label='Target cos(12πx)', linewidth=2)
        ax.axvline(x_loc, color='red', linestyle='--', linewidth=2, label=f'Analysis location x={x_loc:.1f}')
        ax.scatter([x_loc], [cosine_func(x_loc)], color='red', s=100, zorder=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Target Function: cos(12πx) - Analysis at x={x_loc:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # we plot 2-3: log ratio statistics for layer 2 and layer 4
        for layer_idx, layer_name in [(3, "2nd"), (7, "4th")]:
            ax = fig.add_subplot(gs[1, 0 if layer_name == "2nd" else 1])
            
            mean_log_ratios = []
            max_log_ratios = []
            min_log_ratios = []
            std_log_ratios = []
            
            for rank in rank_list:
                if rank in all_results and x_loc in all_results[rank] and layer_name in all_results[rank][x_loc]:
                    log_ratios = all_results[rank][x_loc][layer_name]['metrics']['log_ratios']
                    triu_indices = np.triu_indices_from(log_ratios, k=1)
                    all_log_ratios = log_ratios[triu_indices]
                    mean_log_ratios.append(np.mean(all_log_ratios))
                    max_log_ratios.append(np.max(all_log_ratios))
                    min_log_ratios.append(np.min(all_log_ratios))
                    std_log_ratios.append(np.std(all_log_ratios))
                else:
                    mean_log_ratios.append(np.nan)
                    max_log_ratios.append(np.nan)
                    min_log_ratios.append(np.nan)
                    std_log_ratios.append(np.nan)
            
            ax.plot(rank_list, mean_log_ratios, 'b-o', label='Mean', linewidth=2, markersize=8)
            ax.plot(rank_list, max_log_ratios, 'r-s', label='Max', linewidth=2, markersize=8)
            ax.plot(rank_list, min_log_ratios, 'g-^', label='Min', linewidth=2, markersize=8)
            ax.fill_between(rank_list, 
                           np.array(mean_log_ratios) - np.array(std_log_ratios),
                           np.array(mean_log_ratios) + np.array(std_log_ratios),
                           alpha=0.2, label='±1 std')
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Rank')
            ax.set_ylabel('Log-Ratio')
            ax.set_title(f'{layer_name} Layer: Log-Ratio Statistics\n(at x={x_loc:.1f}, all channel pairs)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # we plot 4: channel share distribution across ranks for layer 2
        ax = fig.add_subplot(gs[1, 2])
        top_shares_by_rank = {rank: [] for rank in rank_list}
        for rank in rank_list:
            if rank in all_results and x_loc in all_results[rank] and "2nd" in all_results[rank][x_loc]:
                shares = all_results[rank][x_loc]["2nd"]['metrics']['shares']
                top_5_values = np.sort(shares)[-5:][::-1]
                top_shares_by_rank[rank] = top_5_values
        
        x_pos = np.arange(len(rank_list))
        width = 0.12
        colors = plt.cm.viridis(np.linspace(0, 1, 5))
        for i in range(5):
            values = [top_shares_by_rank[rank][i] if len(top_shares_by_rank[rank]) > i else 0 for rank in rank_list]
            ax.bar(x_pos + (i-2)*width, values, width, label=f'Top {i+1}', alpha=0.8, color=colors[i])
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('Channel Share')
        ax.set_title(f'Top 5 Channel Shares: 2nd Layer\n(at x={x_loc:.1f})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rank_list)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # we plot 5-6: log ratio heatmaps for different ranks (layer 2)
        for idx, rank in enumerate([rank_list[0], rank_list[-1]]):
            ax = fig.add_subplot(gs[2, idx])
            
            if rank in all_results and x_loc in all_results[rank] and "2nd" in all_results[rank][x_loc]:
                log_ratios = all_results[rank][x_loc]["2nd"]['metrics']['log_ratios']
                im = ax.imshow(log_ratios, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
                ax.set_xlabel('Channel j')
                ax.set_ylabel('Channel i')
                ax.set_title(f'Log-Ratio Matrix: 2nd Layer, rank={rank}\n(at x={x_loc:.1f})')
                plt.colorbar(im, ax=ax, label='R_{i,j} = log(|f_i|) - log(|f_j|)')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        # we plot 7: log ratio distribution across ranks (layer 2)
        ax = fig.add_subplot(gs[2, 2])
        for rank in rank_list:
            if rank in all_results and x_loc in all_results[rank] and "2nd" in all_results[rank][x_loc]:
                log_ratios = all_results[rank][x_loc]["2nd"]['metrics']['log_ratios']
                triu_indices = np.triu_indices_from(log_ratios, k=1)
                all_log_ratios = log_ratios[triu_indices]
                ax.hist(all_log_ratios, bins=30, alpha=0.5, label=f'rank={rank}', density=True)
        
        ax.set_xlabel('Log-Ratio')
        ax.set_ylabel('Density')
        ax.set_title(f'Log-Ratio Distribution: 2nd Layer\n(at x={x_loc:.1f}, all pairs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # we plot 8-9: log ratio heatmaps for different ranks (layer 4)
        for idx, rank in enumerate([rank_list[0], rank_list[-1]]):
            ax = fig.add_subplot(gs[3, idx])
            
            if rank in all_results and x_loc in all_results[rank] and "4th" in all_results[rank][x_loc]:
                log_ratios = all_results[rank][x_loc]["4th"]['metrics']['log_ratios']
                im = ax.imshow(log_ratios, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
                ax.set_xlabel('Channel j')
                ax.set_ylabel('Channel i')
                ax.set_title(f'Log-Ratio Matrix: 4th Layer, rank={rank}\n(at x={x_loc:.1f})')
                plt.colorbar(im, ax=ax, label='R_{i,j} = log(|f_i|) - log(|f_j|)')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        # we plot 10: log ratio distribution across ranks (layer 4)
        ax = fig.add_subplot(gs[3, 2])
        for rank in rank_list:
            if rank in all_results and x_loc in all_results[rank] and "4th" in all_results[rank][x_loc]:
                log_ratios = all_results[rank][x_loc]["4th"]['metrics']['log_ratios']
                triu_indices = np.triu_indices_from(log_ratios, k=1)
                all_log_ratios = log_ratios[triu_indices]
                ax.hist(all_log_ratios, bins=30, alpha=0.5, label=f'rank={rank}', density=True)
        
        ax.set_xlabel('Log-Ratio')
        ax.set_ylabel('Density')
        ax.set_title(f'Log-Ratio Distribution: 4th Layer\n(at x={x_loc:.1f}, all pairs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # we save figure
        loc_name = location_names[x_loc]
        filename = f'cosine_log_ratios_{loc_name}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_dir / filename}")
        plt.close()
    
    # we create weight distribution plots
    print("\n" + "="*80)
    print("Creating Weight Distribution Plots...")
    print("="*80)
    
    # output_dir is already defined above
    # we create comprehensive weight distribution figure
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)
    
    # we get all trainable layer names (odd indices: fcs[1], fcs[3], ..., fcs[11])
    trainable_layers = [f"fcs[{i}]" for i in range(1, 13, 2)]  # [fcs[1], fcs[3], fcs[5], fcs[7], fcs[9], fcs[11]]
    layer_display_names = {f"fcs[{i}]": f"Layer {i//2 + 1} (fcs[{i}])" for i in range(1, 13, 2)}
    
    # we plot weight distributions for each trainable layer across ranks
    for layer_idx, layer_name in enumerate(trainable_layers):
        row = layer_idx // 3
        col = layer_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # we plot distributions for different ranks at final epoch
        for rank in rank_list:
            if rank in all_weight_snapshots and len(all_weight_snapshots[rank]) > 0:
                final_snapshot = all_weight_snapshots[rank][-1]  # last snapshot (epoch 5000)
                if layer_name in final_snapshot['weights']:
                    weights = final_snapshot['weights'][layer_name]['weight'].flatten()
                    ax.hist(weights, bins='auto', alpha=0.5, label=f'rank={rank}', density=True)
        
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{layer_display_names[layer_name]}\nWeight Distribution (Final Epoch)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # we add one more plot showing weight statistics over time for a specific layer and rank
    ax = fig.add_subplot(gs[3, :])
    
    # we choose layer fcs[3] (2nd layer) and rank=25 as example
    example_layer = "fcs[3]"
    example_rank = 25
    
    if example_rank in all_weight_snapshots:
        epochs = []
        means = []
        stds = []
        for snapshot in all_weight_snapshots[example_rank]:
            if example_layer in snapshot['weights']:
                weights = snapshot['weights'][example_layer]['weight'].flatten()
                epochs.append(snapshot['epoch'])
                means.append(np.mean(weights))
                stds.append(np.std(weights))
        
        ax.plot(epochs, means, 'b-o', label='Mean weight', linewidth=2, markersize=8)
        ax.fill_between(epochs, 
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.3, label='±1 std')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Weight Statistics Over Time: {layer_display_names[example_layer]}, rank={example_rank}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weight Distributions Across All Trainable Layers', fontsize=16, y=0.995)
    plt.savefig(output_dir / 'weight_distributions_all_layers.png', dpi=300, bbox_inches='tight')
    print(f"Saved weight distribution figure to {output_dir / 'weight_distributions_all_layers.png'}")
    plt.close()
    
    # we create another figure showing weight evolution for all layers and ranks
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # we plot weight evolution for each trainable layer
    for layer_idx, layer_name in enumerate(trainable_layers):
        row = layer_idx // 2
        col = layer_idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        for rank in rank_list:
            if rank in all_weight_snapshots:
                epochs = []
                means = []
                for snapshot in all_weight_snapshots[rank]:
                    if layer_name in snapshot['weights']:
                        weights = snapshot['weights'][layer_name]['weight'].flatten()
                        epochs.append(snapshot['epoch'])
                        means.append(np.mean(weights))
                
                if len(epochs) > 0:
                    ax.plot(epochs, means, 'o-', label=f'rank={rank}', linewidth=2, markersize=6)
        
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Weight')
        ax.set_title(f'{layer_display_names[layer_name]}\nMean Weight Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weight Evolution Over Time: All Trainable Layers', fontsize=16, y=0.995)
    plt.savefig(output_dir / 'weight_evolution_all_layers.png', dpi=300, bbox_inches='tight')
    print(f"Saved weight evolution figure to {output_dir / 'weight_evolution_all_layers.png'}")
    plt.close()
    
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
        'locations': [float(x) for x in x_locations],
        'architecture': {
            'depth': L,
            'width': width,
            'ranks_tested': rank_list
        },
        'training': {
            'num_epochs': 5000,
            'batch_size': 100,
            'lr': 0.001
        },
        'results': convert_to_serializable(all_results)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=4)
    print(f"\nSaved results to {output_dir / 'results.json'}")
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)


if __name__ == "__main__":
    run_experiment()
