#!/usr/bin/env python3
"""
we analyze scaling law: loss = g(L/freq)
we use minimum loss during training (not final loss) to avoid instability
"""
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# we configure matplotlib
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 18
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

results_dir_original = Path("experiments/table/results_frequency_layer_scaling")
results_dir_extended = Path("experiments/table/results_frequency_layer_scaling_extended")

def load_all_results():
    """we load all completed training results, using minimum loss"""
    all_results = []
    
    for results_dir in [results_dir_original, results_dir_extended]:
        if not results_dir.exists():
            continue
            
        for config_dir in sorted(results_dir.iterdir()):
            if not config_dir.is_dir():
                continue
                
            results_file = config_dir / "results.json"
            config_file = config_dir / "config.json"
            checkpoint_file = config_dir / "checkpoint.pth"
            
            if not results_file.exists() or not config_file.exists():
                continue
                
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # we check if training completed
                if checkpoint_file.exists():
                    ckpt = torch.load(checkpoint_file, map_location='cpu')
                    epoch = ckpt.get('epoch', 0)
                    target_epochs = config.get('num_epochs', 0)
                    if epoch < target_epochs * 0.9:  # allow 90% completion
                        continue
                else:
                    continue
                
                # we extract minimum loss (not final loss)
                all_losses = results.get('all_losses', [])
                if not all_losses or len(all_losses) == 0:
                    continue
                
                min_loss = min(all_losses)
                min_loss_idx = all_losses.index(min_loss)
                
                # we also get minimum test error
                errors_test = results.get('errors_test', [])
                min_test_error = min(errors_test) if errors_test and len(errors_test) > 0 else None
                
                entry = {
                    'config_name': config_dir.name,
                    'freq_multiplier': config.get('freq_multiplier', 0),
                    'rank': config.get('hidden_rank', 0),
                    'layers': config.get('num_layers', 0),
                    'min_loss': min_loss,
                    'min_loss_epoch': min_loss_idx,
                    'min_test_error': min_test_error,
                    'final_loss': all_losses[-1] if all_losses else None,
                    'final_test_error': results.get('final_test_error'),
                    'L_over_freq': config.get('num_layers', 0) / config.get('freq_multiplier', 1),
                }
                
                # we skip NaN or extreme outliers
                if entry['min_loss'] is None or np.isnan(entry['min_loss']):
                    continue
                if entry['min_loss'] > 1e10:  # skip training failures
                    continue
                    
                all_results.append(entry)
                
            except Exception as e:
                continue
    
    return all_results

def analyze_loss_vs_L_over_freq(df):
    """we analyze loss as function of L/freq ratio"""
    
    print("="*80)
    print("SCALING LAW ANALYSIS: loss = g(L/freq)")
    print("="*80)
    
    # we compute L/freq ratio
    df['L_over_freq'] = df['layers'] / df['freq_multiplier']
    
    # we group by L/freq ratio (rounded) and compute mean of min loss
    print("\n" + "="*80)
    print("LOSS vs L/freq RATIO - MEAN OF MIN LOSS PER RATIO")
    print("="*80)
    
    # we round L/freq to reasonable precision for grouping
    df['L_over_freq_rounded'] = np.round(df['L_over_freq'], 2)
    
    # we group by rounded ratio and compute mean of min loss
    ratio_groups = df.groupby('L_over_freq_rounded').agg({
        'min_loss': ['mean', 'std', 'count', 'min', 'max'],
        'freq_multiplier': 'nunique',
        'layers': lambda x: sorted(x.unique().tolist())
    }).reset_index()
    
    ratio_groups.columns = ['ratio', 'mean_loss', 'std_loss', 'count', 'min_loss_val', 'max_loss_val', 'num_freqs', 'layers_list']
    
    # we sort by ratio
    ratio_groups = ratio_groups.sort_values('ratio')
    
    print(f"\n{'L/freq ratio':<15} {'Count':<8} {'Mean min loss':<18} {'Std':<15} {'Freqs':<8}")
    print("-" * 80)
    for _, row in ratio_groups.iterrows():
        print(f"{row['ratio']:>14.2f}  {int(row['count']):>7d}  "
              f"{row['mean_loss']:>17.6e}  {row['std_loss']:>14.6e}  {int(row['num_freqs']):>7d}")
    
    # we also create binned version for smoother curve
    ratio_bins = np.logspace(np.log10(df['L_over_freq'].min()), 
                             np.log10(df['L_over_freq'].max()), 
                             25)
    
    binned_groups = []
    for i in range(len(ratio_bins)-1):
        mask = (df['L_over_freq'] >= ratio_bins[i]) & (df['L_over_freq'] < ratio_bins[i+1])
        group_data = df[mask]
        if len(group_data) > 0:
            binned_groups.append({
                'ratio_center': (ratio_bins[i] + ratio_bins[i+1]) / 2,
                'ratio_min': ratio_bins[i],
                'ratio_max': ratio_bins[i+1],
                'mean_loss': group_data['min_loss'].mean(),
                'median_loss': group_data['min_loss'].median(),
                'std_loss': group_data['min_loss'].std(),
                'count': len(group_data)
            })
    
    return ratio_groups, binned_groups, df

def fit_scaling_function(ratio_groups):
    """we fit function g such that loss = g(L/freq)"""
    
    print("\n" + "="*80)
    print("FITTING SCALING FUNCTION: loss = g(L/freq)")
    print("="*80)
    
    # we extract data from ratio_groups (DataFrame)
    # filter by count
    ratio_groups_filtered = ratio_groups[ratio_groups['count'] >= 2].copy()
    
    if len(ratio_groups_filtered) < 3:
        print("Not enough data points for fitting")
        return None, None, None, None
    
    ratios = ratio_groups_filtered['ratio'].values
    losses = ratio_groups_filtered['mean_loss'].values
    counts = ratio_groups_filtered['count'].values
    
    # ratios and losses are already extracted above
    
    results = {}
    
    # Model 1: Power law loss = a * (L/freq)^(-b)
    def power_model(r, a, b):
        return a * (r ** (-b))
    
    try:
        popt_power, _ = curve_fit(power_model, ratios, losses, p0=[1.0, 1.0], maxfev=5000)
        pred_power = power_model(ratios, *popt_power)
        r2_power = 1 - np.sum((losses - pred_power)**2) / np.sum((losses - np.mean(losses))**2)
        results['power'] = {
            'params': popt_power,
            'formula': f"loss = {popt_power[0]:.6e} × (L/freq)^{{-{popt_power[1]:.3f}}}",
            'r2': r2_power,
            'predictions': pred_power
        }
    except:
        results['power'] = None
    
    # Model 2: Exponential decay loss = a * exp(-b * L/freq)
    def exp_model(r, a, b):
        return a * np.exp(-b * r)
    
    try:
        popt_exp, _ = curve_fit(exp_model, ratios, losses, p0=[1.0, 0.1], maxfev=5000)
        pred_exp = exp_model(ratios, *popt_exp)
        r2_exp = 1 - np.sum((losses - pred_exp)**2) / np.sum((losses - np.mean(losses))**2)
        results['exponential'] = {
            'params': popt_exp,
            'formula': f"loss = {popt_exp[0]:.6e} × exp(-{popt_exp[1]:.4f} × L/freq)",
            'r2': r2_exp,
            'predictions': pred_exp
        }
    except:
        results['exponential'] = None
    
    # Model 3: Inverse power loss = a / (L/freq)^b
    def inv_power_model(r, a, b):
        return a / (r ** b)
    
    try:
        popt_inv, _ = curve_fit(inv_power_model, ratios, losses, p0=[1.0, 1.0], maxfev=5000)
        pred_inv = inv_power_model(ratios, *popt_inv)
        r2_inv = 1 - np.sum((losses - pred_inv)**2) / np.sum((losses - np.mean(losses))**2)
        results['inverse_power'] = {
            'params': popt_inv,
            'formula': f"loss = {popt_inv[0]:.6e} / (L/freq)^{popt_inv[1]:.3f}",
            'r2': r2_inv,
            'predictions': pred_inv
        }
    except:
        results['inverse_power'] = None
    
    # Model 4: Logarithmic loss = a - b * log(L/freq)
    def log_model(r, a, b):
        return a - b * np.log(r)
    
    try:
        popt_log, _ = curve_fit(log_model, ratios, losses, p0=[1.0, 0.1], maxfev=5000)
        pred_log = log_model(ratios, *popt_log)
        r2_log = 1 - np.sum((losses - pred_log)**2) / np.sum((losses - np.mean(losses))**2)
        results['logarithmic'] = {
            'params': popt_log,
            'formula': f"loss = {popt_log[0]:.6e} - {popt_log[1]:.6e} × log(L/freq)",
            'r2': r2_log,
            'predictions': pred_log
        }
    except:
        results['logarithmic'] = None
    
    # we print results
    print("\nFitted Models:")
    for model_name, result in results.items():
        if result:
            print(f"\n{model_name.upper()}:")
            print(f"  {result['formula']}")
            print(f"  R² = {result['r2']:.4f}")
    
    # we find best model
    best_model = None
    best_r2 = -np.inf
    for model_name, result in results.items():
        if result and result['r2'] > best_r2:
            best_r2 = result['r2']
            best_model = (model_name, result)
    
    if best_model:
        print(f"\n{'='*80}")
        print(f"BEST FITTING MODEL: {best_model[0].upper()}")
        print(f"{'='*80}")
        print(f"Formula: {best_model[1]['formula']}")
        print(f"R² = {best_model[1]['r2']:.4f}")
    
    return results, best_model, ratios, losses

def plot_scaling_law(df, ratio_groups, binned_groups, fit_results, best_model, ratios, losses):
    """we plot the scaling law analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean loss per L/freq ratio (THE KEY PLOT)
    ax1 = axes[0, 0]
    
    # we use ratio_groups - mean of min loss for each ratio
    ratio_groups_filtered = ratio_groups[ratio_groups['count'] >= 2].copy()
    
    if len(ratio_groups_filtered) > 0:
        # we plot the mean curve
        ax1.plot(ratio_groups_filtered['ratio'], 
                ratio_groups_filtered['mean_loss'],
                'o-', linewidth=3, markersize=10, 
                label='Mean of min loss per L/freq ratio',
                color='blue', zorder=5)
        
        # we add error bars
        ax1.errorbar(ratio_groups_filtered['ratio'], 
                    ratio_groups_filtered['mean_loss'],
                    yerr=ratio_groups_filtered['std_loss'],
                    fmt='none', capsize=4, capthick=2, 
                    color='blue', alpha=0.5, zorder=4)
        
        # we plot best fit
        if best_model and fit_results:
            ratio_plot = np.logspace(np.log10(ratio_groups_filtered['ratio'].min()), 
                                    np.log10(ratio_groups_filtered['ratio'].max()), 
                                    100)
            result = best_model[1]
            if 'power' in best_model[0]:
                loss_plot = result['params'][0] * (ratio_plot ** (-result['params'][1]))
            elif 'exponential' in best_model[0]:
                loss_plot = result['params'][0] * np.exp(-result['params'][1] * ratio_plot)
            elif 'inverse_power' in best_model[0]:
                loss_plot = result['params'][0] / (ratio_plot ** result['params'][1])
            elif 'logarithmic' in best_model[0]:
                loss_plot = result['params'][0] - result['params'][1] * np.log(ratio_plot)
            else:
                loss_plot = None
            
            if loss_plot is not None:
                ax1.plot(ratio_plot, loss_plot, 'r--', linewidth=3, 
                       label=f"Fit: {result['formula']}\nR²={result['r2']:.3f}",
                       zorder=6)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('L / freq (ratio)', fontsize=18)
    ax1.set_ylabel('Mean of Minimum Loss', fontsize=18)
    ax1.set_title('Scaling Law: loss = g(L/freq)\nMean of min loss per ratio', fontsize=20)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')
    
    # Plot 2: Mean loss per L/freq ratio (the key plot)
    ax2 = axes[0, 1]
    
    # we use ratio_groups (DataFrame) - mean of min loss for each ratio
    ratio_groups_filtered = ratio_groups[ratio_groups['count'] >= 2].copy()
    
    if len(ratio_groups_filtered) > 0:
        ax2.errorbar(ratio_groups_filtered['ratio'], 
                    ratio_groups_filtered['mean_loss'],
                    yerr=ratio_groups_filtered['std_loss'],
                    fmt='o-', capsize=5, capthick=2, markersize=8, 
                    linewidth=2, label='Mean of min loss per L/freq ratio')
        
        # we plot best fit if available
        if best_model and fit_results:
            ratio_plot = np.logspace(np.log10(ratio_groups_filtered['ratio'].min()), 
                                    np.log10(ratio_groups_filtered['ratio'].max()), 
                                    100)
            result = best_model[1]
            if 'power' in best_model[0]:
                loss_plot = result['params'][0] * (ratio_plot ** (-result['params'][1]))
            elif 'exponential' in best_model[0]:
                loss_plot = result['params'][0] * np.exp(-result['params'][1] * ratio_plot)
            elif 'inverse_power' in best_model[0]:
                loss_plot = result['params'][0] / (ratio_plot ** result['params'][1])
            elif 'logarithmic' in best_model[0]:
                loss_plot = result['params'][0] - result['params'][1] * np.log(ratio_plot)
            else:
                loss_plot = None
            
            if loss_plot is not None:
                ax2.plot(ratio_plot, loss_plot, 'r--', linewidth=3, 
                       label=f"Fit: {result['formula']}\nR²={result['r2']:.3f}")
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('L / freq (ratio)', fontsize=18)
    ax2.set_ylabel('Mean of Minimum Loss', fontsize=18)
    ax2.set_title('Scaling Law: loss = g(L/freq)\nMean of min loss per ratio', fontsize=20)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    
    # Plot 3: Loss vs L/freq for different frequency ranges
    ax3 = axes[1, 0]
    freq_ranges = [
        (0, 0.5, 'Very Low (0-0.5)', 'blue'),
        (0.5, 2, 'Low (0.5-2)', 'green'),
        (2, 10, 'Medium (2-10)', 'orange'),
        (10, 100, 'High (10-100)', 'red'),
        (100, 1000, 'Very High (100+)', 'purple')
    ]
    
    for freq_min, freq_max, label, color in freq_ranges:
        mask = (df['freq_multiplier'] >= freq_min) & (df['freq_multiplier'] < freq_max)
        subset = df[mask]
        if len(subset) > 0:
            ax3.scatter(subset['L_over_freq'], subset['min_loss'], 
                       s=50, alpha=0.6, label=label, color=color, edgecolors='black', linewidth=0.5)
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('L / freq (ratio)', fontsize=18)
    ax3.set_ylabel('Minimum Loss', fontsize=18)
    ax3.set_title('Loss vs L/freq by Frequency Range', fontsize=20)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=10, ncol=2)
    
    # Plot 4: Contour/heatmap of loss vs L/freq and freq
    ax4 = axes[1, 1]
    
    # we create 2D histogram
    x_bins = np.logspace(np.log10(df['L_over_freq'].min()), 
                        np.log10(df['L_over_freq'].max()), 20)
    y_bins = np.logspace(np.log10(df['freq_multiplier'].min()), 
                        np.log10(df['freq_multiplier'].max()), 20)
    
    # we compute mean loss in each bin
    loss_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
    count_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
    
    for i in range(len(y_bins)-1):
        for j in range(len(x_bins)-1):
            mask = ((df['freq_multiplier'] >= y_bins[i]) & 
                   (df['freq_multiplier'] < y_bins[i+1]) &
                   (df['L_over_freq'] >= x_bins[j]) & 
                   (df['L_over_freq'] < x_bins[j+1]))
            subset = df[mask]
            if len(subset) > 0:
                loss_grid[i, j] = subset['min_loss'].mean()
                count_grid[i, j] = len(subset)
    
    # we mask empty bins
    loss_grid[count_grid == 0] = np.nan
    
    im = ax4.contourf(x_bins[:-1], y_bins[:-1], loss_grid, 
                     levels=20, cmap='viridis', extend='both')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('L / freq (ratio)', fontsize=18)
    ax4.set_ylabel('Frequency Multiplier', fontsize=18)
    ax4.set_title('Loss Heatmap: L/freq vs freq', fontsize=20)
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Minimum Loss', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('experiments/table/scaling_law_loss_vs_L_over_freq.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: experiments/table/scaling_law_loss_vs_L_over_freq.png")
    plt.close()

def verify_universal_scaling(df):
    """we verify if same L/freq ratio gives same loss regardless of freq"""
    
    print("\n" + "="*80)
    print("VERIFICATION: Universal Scaling Law")
    print("="*80)
    print("\nTesting if same L/freq ratio gives similar loss across different frequencies...")
    
    # we group by L/freq ratio (with some tolerance)
    df['L_over_freq_rounded'] = np.round(df['L_over_freq'], 1)
    
    ratio_groups = df.groupby('L_over_freq_rounded')
    
    print(f"\n{'L/freq ratio':<15} {'Count':<8} {'Mean loss':<15} {'Std loss':<15} {'CV (std/mean)':<15}")
    print("-" * 80)
    
    universal_ratios = []
    for ratio, group in ratio_groups:
        if len(group) >= 3:  # need at least 3 samples
            mean_loss = group['min_loss'].mean()
            std_loss = group['min_loss'].std()
            cv = std_loss / mean_loss if mean_loss > 0 else np.inf
            
            # we check if this ratio appears across multiple frequencies
            unique_freqs = group['freq_multiplier'].nunique()
            
            print(f"{ratio:>14.1f}  {len(group):>7d}  {mean_loss:>14.6e}  "
                  f"{std_loss:>14.6e}  {cv:>14.4f}  (freqs: {unique_freqs})")
            
            if cv < 0.5 and unique_freqs >= 2:  # low coefficient of variation
                universal_ratios.append({
                    'ratio': ratio,
                    'mean_loss': mean_loss,
                    'cv': cv,
                    'num_freqs': unique_freqs
                })
    
    if universal_ratios:
        print(f"\n✓ Found {len(universal_ratios)} L/freq ratios with universal scaling (CV < 0.5)")
        print("  These ratios give similar loss across different frequencies!")
    else:
        print("\n⚠ No clear universal scaling found (may need more data)")

def main():
    """we run the scaling law analysis"""
    print("Loading all training results (using minimum loss)...")
    all_results = load_all_results()
    
    if len(all_results) == 0:
        print("No completed results found!")
        return
    
    print(f"Loaded {len(all_results)} completed configurations")
    
    df = pd.DataFrame(all_results)
    
    # we analyze
    ratio_groups, binned_groups, df = analyze_loss_vs_L_over_freq(df)
    
    # we fit scaling function using mean loss per ratio
    fit_results, best_model, ratios, losses = fit_scaling_function(ratio_groups)
    
    # we verify universal scaling
    verify_universal_scaling(df)
    
    # we plot
    plot_scaling_law(df, ratio_groups, binned_groups, fit_results, best_model, ratios, losses)
    
    # we save results
    summary = {
        'total_configs': len(df),
        'best_model': best_model[0] if best_model else None,
        'best_formula': best_model[1]['formula'] if best_model else None,
        'best_r2': float(best_model[1]['r2']) if best_model else None,
    }
    
    import json
    with open('experiments/table/scaling_law_loss_vs_L_over_freq_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ Results saved to: experiments/table/scaling_law_loss_vs_L_over_freq_summary.json")
    
    # we save data
    df.to_csv('experiments/table/scaling_law_loss_vs_L_over_freq_data.csv', index=False)
    print(f"✓ Data saved to: experiments/table/scaling_law_loss_vs_L_over_freq_data.csv")

if __name__ == "__main__":
    import torch
    main()
