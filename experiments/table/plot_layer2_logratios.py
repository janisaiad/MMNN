#!/usr/bin/env python3
"""
Script to plot log ratios for layer 2 from existing JSON files.
Creates heatmaps and statistics plots like in mean-field methodology.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from pathlib import Path
import sys

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

def plot_log_ratio_heatmap(log_ratio_matrix, output_path, config_name):
    """we plot heatmap of log ratios"""
    R = np.array(log_ratio_matrix)
    r = R.shape[0]
    
    # we filter out NaN and Inf
    R_clean = np.where(np.isfinite(R), R, 0.0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # we use actual min/max of data for colorbar
    vmin = np.min(R_clean[R_clean != 0]) if np.any(R_clean != 0) else -1
    vmax = np.max(R_clean[R_clean != 0]) if np.any(R_clean != 0) else 1
    
    im = ax.imshow(R_clean, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Channel $j$', fontsize=24)
    ax.set_ylabel('Channel $i$', fontsize=24)
    ax.set_title(f'Log Ratio Matrix $R_{{i,j}} = \\log(|f_i|) - \\log(|f_j|)$ at $x=0$\n{config_name}', fontsize=20)
    
    # we set ticks (limit to avoid too many ticks for large ranks)
    if r <= 50:
        ax.set_xticks(range(r))
        ax.set_yticks(range(r))
        ax.set_xticklabels([f'{i+1}' for i in range(r)], fontsize=14)
        ax.set_yticklabels([f'{i+1}' for i in range(r)], fontsize=14)
    else:
        # we space ticks for large ranks
        step = max(1, r // 20)
        tick_positions = list(range(0, r, step))
        if tick_positions[-1] != r - 1:
            tick_positions.append(r - 1)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([f'{i+1}' for i in tick_positions], fontsize=12)
        ax.set_yticklabels([f'{i+1}' for i in tick_positions], fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$R_{i,j}$', fontsize=20, rotation=0)
    cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved heatmap: {output_path}")

def plot_log_ratio_statistics(log_ratio_matrix, output_path, config_name):
    """we plot statistics of log ratios"""
    R = np.array(log_ratio_matrix)
    R_clean = R[np.isfinite(R)]
    
    if len(R_clean) == 0:
        print(f"   ⚠️  No valid log ratios for {config_name}, skipping statistics plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # we plot histogram
    ax = axes[0]
    ax.hist(R_clean, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('$R_{i,j}$', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.set_title(f'Distribution of Log Ratios at $x=0$\n{config_name}', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='$R=0$')
    ax.legend(fontsize=14)
    
    # we plot statistics text
    ax = axes[1]
    ax.axis('off')
    stats_text = f"""
    Statistics of Log Ratios $R_{{i,j}} = \\log(|f_i|) - \\log(|f_j|)$ at $x=0$:
    
    Mean: {np.mean(R_clean):.4f}
    Std:  {np.std(R_clean):.4f}
    Min:  {np.min(R_clean):.4f}
    Max:  {np.max(R_clean):.4f}
    
    Number of pairs: {len(R_clean)}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=16, verticalalignment='center', 
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved statistics: {output_path}")

if __name__ == "__main__":
    results_base = Path("/Data/janis.aiad/MMNN/experiments/table/experiments/table/results_tune_lr_decay_L2")
    
    # we find all log ratio JSON files
    logratio_files = list(results_base.glob("**/layer2_logratios_x0.json"))
    
    print(f"Found {len(logratio_files)} log ratio files to plot")
    
    for logratio_file in logratio_files:
        config_dir = logratio_file.parent
        config_name = config_dir.name
        
        try:
            with open(logratio_file, 'r') as f:
                data = json.load(f)
            
            log_ratio_matrix = data['log_ratio_matrix']
            
            # we check if there are valid values
            R = np.array(log_ratio_matrix)
            if np.all(~np.isfinite(R)):
                print(f"   ⚠️  All values are NaN/Inf for {config_name}, skipping")
                continue
            
            print(f"\n📊 Plotting log ratios for {config_name}...")
            
            # we plot heatmap
            heatmap_path = config_dir / 'layer2_logratio_heatmap_x0.png'
            plot_log_ratio_heatmap(log_ratio_matrix, heatmap_path, config_name)
            
            # we plot statistics
            stats_path = config_dir / 'layer2_logratio_statistics_x0.png'
            plot_log_ratio_statistics(log_ratio_matrix, stats_path, config_name)
            
        except Exception as e:
            print(f"   ❌ Error processing {config_name}: {e}")
            continue
    
    print(f"\n✅ Done! Plotted log ratios for {len(logratio_files)} configurations")
