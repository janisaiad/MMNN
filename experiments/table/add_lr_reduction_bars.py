#!/usr/bin/env python3
"""
Add red vertical bars to existing loss evolution plots at LR reduction moments
and update results.json with lr_reduction_epochs if missing
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def update_plots_and_results(results_dir):
    """Update all plots and results.json files with LR reduction bars"""
    results_dir = Path(results_dir)
    
    # Find all AdaptiveStagnation directories
    config_dirs = list(results_dir.glob("factor*_rank*_*_AdaptiveStagnation"))
    if not config_dirs:
        # Try nested path
        results_dir = results_dir / "experiments" / "table" / "results_tune_lr_decay_L2"
        config_dirs = list(results_dir.glob("factor*_rank*_*_AdaptiveStagnation"))
    
    print(f"Found {len(config_dirs)} AdaptiveStagnation configs")
    
    for config_dir in config_dirs:
        results_file = config_dir / "results.json"
        plot_file = config_dir / "loss_evolution.png"
        
        if not results_file.exists():
            print(f"  ⚠️  {config_dir.name}: results.json not found, skipping")
            continue
        
        print(f"Processing: {config_dir.name}")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Get LR reduction epochs
        lr_reduction_epochs = results.get('lr_reduction_epochs', [])
        all_losses = results.get('all_losses', [])
        
        if not all_losses:
            print(f"  ⚠️  No losses found, skipping")
            continue
        
        # Update results.json if lr_reduction_epochs is missing but we can infer from all_lrs
        if not lr_reduction_epochs and 'all_lrs' in results:
            all_lrs = results['all_lrs']
            lr_sequence = [0.01, 0.005, 0.001, 0.0005, 0.0001]
            lr_reduction_epochs = []
            current_lr_idx = 0
            
            for epoch, lr in enumerate(all_lrs):
                if current_lr_idx < len(lr_sequence) - 1:
                    if lr < lr_sequence[current_lr_idx] - 1e-6:  # LR decreased
                        current_lr_idx += 1
                        lr_reduction_epochs.append(epoch)
            
            if lr_reduction_epochs:
                results['lr_reduction_epochs'] = lr_reduction_epochs
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  ✅ Updated results.json with {len(lr_reduction_epochs)} LR reductions")
        
        # Recreate plot with red bars
        if plot_file.exists() and lr_reduction_epochs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.semilogy(all_losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
            
            # Add red vertical bars
            for i, reduction_epoch in enumerate(lr_reduction_epochs):
                if reduction_epoch < len(all_losses):
                    ax.axvline(x=reduction_epoch, color='r', linestyle='--', linewidth=1.5, 
                              alpha=0.7, label='LR reduction' if i == 0 else '')
            
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Loss', fontsize=14)
            
            # Extract info from directory name
            parts = config_dir.name.split('_')
            factor = parts[0].replace('factor', '')
            rank = parts[1].replace('rank', '')
            momentum = parts[3].replace('mom', '')
            
            title = f'Loss Evolution - factor={factor}, SGD, Momentum={momentum}, LR init=0.01'
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"  ✅ Updated plot with {len(lr_reduction_epochs)} red bars")
        elif not lr_reduction_epochs:
            print(f"  ⚠️  No LR reductions found")

if __name__ == "__main__":
    # Try both possible paths
    results_dir1 = Path("experiments/table/results_tune_lr_decay_L2")
    results_dir2 = Path("experiments/table/experiments/table/results_tune_lr_decay_L2")
    
    if results_dir2.exists():
        results_dir = results_dir2
    elif results_dir1.exists():
        results_dir = results_dir1
    else:
        print("❌ Results directory not found!")
        exit(1)
    
    print(f"Updating plots and results in: {results_dir}")
    print("="*80)
    update_plots_and_results(results_dir)
    print("="*80)
    print("✅ Done!")
