#!/usr/bin/env python3
"""
Script to plot log ratios for layer 1 and layer 2 using existing .npy files.
For factor=4, rank=15 only.
Plots for x = 0, 0.2, 0.4, 0.6, 0.8, 1.0
Uses existing matrices (from x=0) for all x values if specific x files don't exist.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

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

def plot_log_ratio_statistics_positive_improved(log_ratio_matrix, output_path, config_name, x_value, epsilon, layer_name):
    """we plot statistics of log ratios (only positive values) with improved LaTeX"""
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
    
    # we add config name at the bottom (not in title)
    fig.text(0.5, 0.02, f'{config_name}', ha='center', fontsize=14, wrap=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_path.name}")

if __name__ == "__main__":
    results_base = Path("/Data/janis.aiad/MMNN/experiments/table/experiments/table/results_tune_lr_decay_L2")
    
    # we only process factor=4, rank=15
    target_factor = 4
    target_rank = 15
    
    # we define x values to plot
    x_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    epsilon = 1e-6
    
    # we find matching config directories
    config_dirs = []
    for config_dir in results_base.iterdir():
        if not config_dir.is_dir():
            continue
        if 'factor4' in config_dir.name and 'rank15' in config_dir.name:
            config_dirs.append(config_dir)
    
    print(f"Found {len(config_dirs)} configurations with factor=4, rank=15")
    
    for config_dir in config_dirs:
        config_name = config_dir.name
        print(f"\n{'='*80}")
        print(f"Processing: {config_name}")
        print(f"{'='*80}")
        
        # we process both layers
        for layer_idx in [1, 2]:
            layer_name = f"Layer {layer_idx}"
            print(f"\n   Plotting log ratios for {layer_name}...")
            
            # we plot for all x values
            # we prefer old format (x0.npy) which has width×width matrices with valid values
            # over new format (x0.0.npy) which has rank×rank matrices that may be NaN
            # for all x, we use the old x0.npy matrix (which has valid values)
            old_format_file = config_dir / f'layer{layer_idx}_logratio_matrix_x0.npy'
            
            if not old_format_file.exists():
                print(f"   ⚠️  No old format matrix file found for {layer_name}, trying new format...")
                # we try new format as fallback
                old_format_file = config_dir / f'layer{layer_idx}_logratio_matrix_x0.0.npy'
                if not old_format_file.exists():
                    print(f"   ⚠️  No matrix file found for {layer_name}, skipping")
                    continue
            
            # we load the base matrix (from old format, has valid values)
            R_base = np.load(old_format_file)
            R_base_clean = R_base[np.isfinite(R_base)]
            R_base_positive = R_base_clean[R_base_clean > 0]
            print(f"   ✅ Loaded base matrix from {old_format_file.name} (shape: {R_base.shape}, positive: {len(R_base_positive)}/{len(R_base_clean)})")
            
            # we use the same matrix for all x values (as requested: "former values")
            for x_val in x_values:
                R = R_base  # we use the same matrix for all x
                print(f"      Using base matrix for x={x_val}")
                
                plot_path = config_dir / f'layer{layer_idx}_logratio_statistics_x{x_val}_positive_improved.png'
                plot_log_ratio_statistics_positive_improved(R, plot_path, config_name, x_val, epsilon, layer_name)
        
    print(f"\n✅ Done! Generated all plots")
