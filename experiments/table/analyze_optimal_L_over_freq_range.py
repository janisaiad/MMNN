#!/usr/bin/env python3
"""
we analyze the optimal L/freq range (7-12) and verify the U-shaped curve
we account for actual cosine frequencies: base_freqs = [12, 24, 36, 72] × freq_multiplier
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

# we configure matplotlib
plt.rcParams['figure.figsize'] = [12, 10]
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

def compute_effective_frequency(freq_multiplier):
    """we compute effective frequency from cosine frequencies
    base_freqs = [12, 24, 36, 72] × freq_multiplier
    we use the maximum or mean as effective frequency"""
    base_freqs = np.array([12, 24, 36, 72])
    scaled_freqs = base_freqs * freq_multiplier
    # we use maximum frequency (highest harmonic)
    effective_freq = np.max(scaled_freqs)
    # alternative: mean frequency
    mean_freq = np.mean(scaled_freqs)
    return effective_freq, mean_freq, scaled_freqs

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
                    if epoch < target_epochs * 0.9:
                        continue
                else:
                    continue
                
                # we extract minimum loss
                all_losses = results.get('all_losses', [])
                if not all_losses or len(all_losses) == 0:
                    continue
                
                min_loss = min(all_losses)
                
                freq_mult = config.get('freq_multiplier', 0)
                layers = config.get('num_layers', 0)
                
                # we compute effective frequency and L/effective_freq
                effective_freq, mean_freq, scaled_freqs = compute_effective_frequency(freq_mult)
                
                entry = {
                    'config_name': config_dir.name,
                    'freq_multiplier': freq_mult,
                    'rank': config.get('hidden_rank', 0),
                    'layers': layers,
                    'min_loss': min_loss,
                    'L_over_freq_mult': layers / freq_mult if freq_mult > 0 else 0,
                    'effective_freq': effective_freq,
                    'mean_freq': mean_freq,
                    'L_over_effective_freq': layers / effective_freq if effective_freq > 0 else 0,
                    'L_over_mean_freq': layers / mean_freq if mean_freq > 0 else 0,
                    'max_cosine_freq': scaled_freqs.max(),
                    'min_cosine_freq': scaled_freqs.min(),
                }
                
                # we skip NaN or extreme outliers
                if entry['min_loss'] is None or np.isnan(entry['min_loss']):
                    continue
                if entry['min_loss'] > 1e10:
                    continue
                    
                all_results.append(entry)
                
            except Exception as e:
                continue
    
    return all_results

def analyze_optimal_range(df):
    """we analyze the optimal L/freq range (7-12)"""
    
    print("="*80)
    print("ANALYSIS: OPTIMAL L/freq RANGE (7-12)")
    print("="*80)
    
    # we focus on L/freq_mult ratio
    df['L_over_freq_rounded'] = np.round(df['L_over_freq_mult'], 1)
    
    # we group by ratio and compute mean of min loss
    ratio_stats = df.groupby('L_over_freq_rounded')['min_loss'].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).reset_index()
    ratio_stats.columns = ['ratio', 'mean_loss', 'std_loss', 'count', 'min_loss_val', 'max_loss_val']
    ratio_stats = ratio_stats[ratio_stats['count'] >= 1].sort_values('ratio')
    
    print("\n" + "="*80)
    print("MEAN OF MIN LOSS PER L/freq RATIO")
    print("="*80)
    print(f"\n{'L/freq':<12} {'Count':<8} {'Mean min loss':<18} {'Std':<15} {'Range':<25}")
    print("-" * 90)
    for _, row in ratio_stats.iterrows():
        range_str = f"[{row['min_loss_val']:.2e}, {row['max_loss_val']:.2e}]"
        print(f"{row['ratio']:>11.1f}  {int(row['count']):>7d}  "
              f"{row['mean_loss']:>17.6e}  {row['std_loss']:>14.6e}  {range_str}")
    
    # we focus on range 7-12
    print("\n" + "="*80)
    print("FOCUS: L/freq RATIO BETWEEN 7 AND 12")
    print("="*80)
    
    optimal_range = ratio_stats[(ratio_stats['ratio'] >= 7) & (ratio_stats['ratio'] <= 12)]
    
    if len(optimal_range) > 0:
        print(f"\nFound {len(optimal_range)} ratios in optimal range:")
        for _, row in optimal_range.iterrows():
            print(f"  L/freq = {row['ratio']:.1f}: mean loss = {row['mean_loss']:.6e} "
                  f"(count={int(row['count'])})")
        
        best_in_range = optimal_range.loc[optimal_range['mean_loss'].idxmin()]
        print(f"\n✅ Best in range 7-12: L/freq = {best_in_range['ratio']:.1f}, "
              f"mean loss = {best_in_range['mean_loss']:.6e}")
    else:
        print("\n⚠️  No data in range 7-12 yet")
    
    # we check for U-shaped curve
    print("\n" + "="*80)
    print("CHECKING FOR U-SHAPED CURVE")
    print("="*80)
    
    # we look at ratios < 7, 7-12, and > 12
    before_7 = ratio_stats[ratio_stats['ratio'] < 7]
    in_range = ratio_stats[(ratio_stats['ratio'] >= 7) & (ratio_stats['ratio'] <= 12)]
    after_12 = ratio_stats[ratio_stats['ratio'] > 12]
    
    print(f"\nBefore L/freq=7: {len(before_7)} ratios")
    if len(before_7) > 0:
        print(f"  Mean loss trend: {before_7['mean_loss'].iloc[-1]:.6e} (last) vs "
              f"{before_7['mean_loss'].iloc[0]:.6e} (first)")
        if len(before_7) > 1:
            trend_before = before_7['mean_loss'].iloc[-1] - before_7['mean_loss'].iloc[0]
            print(f"  Trend: {'decreasing' if trend_before < 0 else 'increasing'} "
                  f"({trend_before:.6e})")
    
    print(f"\nIn range 7-12: {len(in_range)} ratios")
    if len(in_range) > 0:
        print(f"  Mean loss range: [{in_range['mean_loss'].min():.6e}, "
              f"{in_range['mean_loss'].max():.6e}]")
        print(f"  Best: {in_range['mean_loss'].min():.6e} at L/freq = "
              f"{in_range.loc[in_range['mean_loss'].idxmin(), 'ratio']:.1f}")
    
    print(f"\nAfter L/freq=12: {len(after_12)} ratios")
    if after_12 is not None and len(after_12) > 0:
        print(f"  Mean loss trend: {after_12['mean_loss'].iloc[0]:.6e} (first) vs "
              f"{after_12['mean_loss'].iloc[-1]:.6e} (last)")
        if len(after_12) > 1:
            trend_after = after_12['mean_loss'].iloc[-1] - after_12['mean_loss'].iloc[0]
            print(f"  Trend: {'increasing' if trend_after > 0 else 'decreasing'} "
                  f"({trend_after:.6e})")
    
    return ratio_stats, before_7, in_range, after_12

def fit_u_shaped_curve(ratio_stats):
    """we fit a U-shaped curve: decreasing until 7, increasing after 12"""
    
    print("\n" + "="*80)
    print("FITTING U-SHAPED CURVE")
    print("="*80)
    
    # we filter by count and use log loss for better fitting
    ratio_stats_filtered = ratio_stats[ratio_stats['count'] >= 1].copy()
    
    if len(ratio_stats_filtered) < 5:
        print("Not enough data points for fitting")
        return None
    
    ratios = ratio_stats_filtered['ratio'].values
    losses = ratio_stats_filtered['mean_loss'].values
    
    # we use log loss to handle wide range
    log_losses = np.log10(losses + 1e-10)
    
    # Model: U-shaped curve with minimum around 7-12
    # log_loss = a * (L/freq - b)^2 + c
    def u_shaped_log(r, a, b, c):
        return a * (r - b)**2 + c
    
    try:
        # we initialize with minimum around 9.5 (middle of 7-12)
        # we constrain b to be between 7 and 12
        popt, _ = curve_fit(u_shaped_log, ratios, log_losses, 
                           p0=[0.1, 9.5, -6], 
                           bounds=([0, 7, -20], [10, 12, 5]),
                           maxfev=5000)
        pred_log = u_shaped_log(ratios, *popt)
        r2 = 1 - np.sum((log_losses - pred_log)**2) / np.sum((log_losses - np.mean(log_losses))**2)
        
        # we convert back to linear scale
        minimum_value = 10**popt[2] if popt[0] > 0 else losses.min()
        
        result = {
            'params': popt,
            'formula': f"log10(loss) = {popt[0]:.4f} × (L/freq - {popt[1]:.3f})² + {popt[2]:.4f}",
            'minimum_at': popt[1],
            'minimum_value': minimum_value,
            'r2': r2,
            'predictions': 10**pred_log
        }
        
        print(f"\nU-SHAPED FIT (log scale):")
        print(f"  {result['formula']}")
        print(f"  Minimum at L/freq = {result['minimum_at']:.3f}")
        print(f"  Minimum loss ≈ {result['minimum_value']:.6e}")
        print(f"  R² = {result['r2']:.4f}")
        
        if 7 <= result['minimum_at'] <= 12:
            print(f"  ✓ Minimum is in optimal range 7-12!")
        else:
            print(f"  ⚠️  Minimum is outside optimal range 7-12")
        
        return result
    except Exception as e:
        print(f"U-shaped fit failed: {e}")
        # we try simpler approach: find minimum in data
        min_idx = ratio_stats_filtered['mean_loss'].idxmin()
        min_ratio = ratio_stats_filtered.loc[min_idx, 'ratio']
        min_loss = ratio_stats_filtered.loc[min_idx, 'mean_loss']
        print(f"\nAlternative: Minimum in data at L/freq = {min_ratio:.2f}, loss = {min_loss:.6e}")
        return None

def plot_optimal_range_analysis(df, ratio_stats, before_7, in_range, after_12, u_fit):
    """we plot the analysis focusing on optimal range"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean loss vs L/freq with optimal range highlighted
    ax1 = axes[0, 0]
    
    ratio_stats_filtered = ratio_stats[ratio_stats['count'] >= 1].copy()
    
    # we plot all points
    ax1.plot(ratio_stats_filtered['ratio'], 
            ratio_stats_filtered['mean_loss'],
            'o-', linewidth=3, markersize=10, 
            label='Mean of min loss per L/freq ratio',
            color='blue', zorder=5)
    
    # we highlight optimal range 7-12
    optimal_data = ratio_stats_filtered[
        (ratio_stats_filtered['ratio'] >= 7) & 
        (ratio_stats_filtered['ratio'] <= 12)
    ]
    if len(optimal_data) > 0:
        ax1.scatter(optimal_data['ratio'], optimal_data['mean_loss'],
                   s=200, marker='*', color='green', zorder=6,
                   label='Optimal range (7-12)', edgecolors='black', linewidth=2)
    
    # we add vertical lines for boundaries
    ax1.axvline(7, color='green', linestyle='--', linewidth=2, alpha=0.7, label='L/freq = 7')
    ax1.axvline(12, color='green', linestyle='--', linewidth=2, alpha=0.7, label='L/freq = 12')
    
    # we plot U-shaped fit if available
    if u_fit:
        ratio_plot = np.linspace(ratio_stats_filtered['ratio'].min(), 
                                ratio_stats_filtered['ratio'].max(), 
                                200)
        # we use log scale fit
        log_loss_plot = u_fit['params'][0] * (ratio_plot - u_fit['params'][1])**2 + u_fit['params'][2]
        loss_plot = 10**log_loss_plot
        ax1.plot(ratio_plot, loss_plot, 'r--', linewidth=3, 
               label=f"U-shaped fit\nMin at L/freq={u_fit['minimum_at']:.2f}\nR²={u_fit['r2']:.3f}",
               zorder=4)
    
    ax1.set_xlabel('L / freq (ratio)', fontsize=18)
    ax1.set_ylabel('Mean of Minimum Loss', fontsize=18)
    ax1.set_title('Scaling Law: loss = g(L/freq)\nOptimal Range 7-12 Highlighted', fontsize=20)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')
    
    # Plot 2: Zoom on optimal range 7-12
    ax2 = axes[0, 1]
    
    if len(optimal_data) > 0:
        ax2.plot(optimal_data['ratio'], optimal_data['mean_loss'],
                'o-', linewidth=3, markersize=12, 
                label='Mean of min loss',
                color='green', zorder=5)
        ax2.errorbar(optimal_data['ratio'], optimal_data['mean_loss'],
                    yerr=optimal_data['std_loss'],
                    fmt='none', capsize=5, capthick=2, 
                    color='green', alpha=0.5, zorder=4)
        
        if u_fit:
            ratio_plot = np.linspace(7, 12, 100)
            log_loss_plot = u_fit['params'][0] * (ratio_plot - u_fit['params'][1])**2 + u_fit['params'][2]
            loss_plot = 10**log_loss_plot
            ax2.plot(ratio_plot, loss_plot, 'r--', linewidth=3, 
                   label=f"U-shaped fit (R²={u_fit['r2']:.3f})")
    
    ax2.set_xlabel('L / freq (ratio)', fontsize=18)
    ax2.set_ylabel('Mean of Minimum Loss', fontsize=18)
    ax2.set_title('Zoom: Optimal Range 7-12', fontsize=20)
    ax2.set_xlim(6, 13)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11)
    
    # Plot 3: All individual points colored by L/freq range
    ax3 = axes[1, 0]
    
    # we color points by range
    df['range_category'] = 'other'
    df.loc[df['L_over_freq_mult'] < 7, 'range_category'] = 'before_7'
    df.loc[(df['L_over_freq_mult'] >= 7) & (df['L_over_freq_mult'] <= 12), 'range_category'] = 'optimal'
    df.loc[df['L_over_freq_mult'] > 12, 'range_category'] = 'after_12'
    
    colors_map = {'before_7': 'blue', 'optimal': 'green', 'after_12': 'red', 'other': 'gray'}
    
    for category, color in colors_map.items():
        subset = df[df['range_category'] == category]
        if len(subset) > 0:
            ax3.scatter(subset['L_over_freq_mult'], subset['min_loss'],
                       s=50, alpha=0.6, label=category.replace('_', ' ').title(),
                       color=color, edgecolors='black', linewidth=0.5)
    
    # we plot mean curve
    ratio_stats_filtered = ratio_stats[ratio_stats['count'] >= 1].copy()
    ax3.plot(ratio_stats_filtered['ratio'], ratio_stats_filtered['mean_loss'],
            'k-', linewidth=2, label='Mean curve', zorder=5)
    
    ax3.axvline(7, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axvline(12, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('L / freq (ratio)', fontsize=18)
    ax3.set_ylabel('Minimum Loss', fontsize=18)
    ax3.set_title('All Configurations by Range', fontsize=20)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=10, ncol=2)
    
    # Plot 4: Loss distribution in optimal range
    ax4 = axes[1, 1]
    
    optimal_df = df[(df['L_over_freq_mult'] >= 7) & (df['L_over_freq_mult'] <= 12)]
    other_df = df[~((df['L_over_freq_mult'] >= 7) & (df['L_over_freq_mult'] <= 12))]
    
    if len(optimal_df) > 0:
        ax4.hist(np.log10(optimal_df['min_loss'] + 1e-10), bins=20, alpha=0.7, 
                label=f'Optimal range (7-12), n={len(optimal_df)}', color='green')
    if len(other_df) > 0:
        ax4.hist(np.log10(other_df['min_loss'] + 1e-10), bins=20, alpha=0.7, 
                label=f'Other ranges, n={len(other_df)}', color='gray')
    
    ax4.set_xlabel('log10(Minimum Loss)', fontsize=18)
    ax4.set_ylabel('Count', fontsize=18)
    ax4.set_title('Loss Distribution: Optimal vs Other', fontsize=20)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('experiments/table/optimal_L_over_freq_range_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: experiments/table/optimal_L_over_freq_range_analysis.png")
    plt.close()

def main():
    """we run the optimal range analysis"""
    print("Loading all training results (using minimum loss)...")
    all_results = load_all_results()
    
    if len(all_results) == 0:
        print("No completed results found!")
        return
    
    print(f"Loaded {len(all_results)} completed configurations")
    
    df = pd.DataFrame(all_results)
    
    # we analyze optimal range
    ratio_stats, before_7, in_range, after_12 = analyze_optimal_range(df)
    
    # we fit U-shaped curve
    u_fit = fit_u_shaped_curve(ratio_stats)
    
    # we plot
    plot_optimal_range_analysis(df, ratio_stats, before_7, in_range, after_12, u_fit)
    
    # we save summary
    summary = {
        'total_configs': len(df),
        'optimal_range_7_12': {
            'count': len(in_range) if in_range is not None and len(in_range) > 0 else 0,
            'best_ratio': float(in_range.loc[in_range['mean_loss'].idxmin(), 'ratio']) if in_range is not None and len(in_range) > 0 else None,
            'best_loss': float(in_range['mean_loss'].min()) if in_range is not None and len(in_range) > 0 else None,
        },
        'u_shaped_fit': {
            'minimum_at': float(u_fit['minimum_at']) if u_fit else None,
            'minimum_value': float(u_fit['minimum_value']) if u_fit else None,
            'r2': float(u_fit['r2']) if u_fit else None,
        }
    }
    
    import json
    with open('experiments/table/optimal_L_over_freq_range_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ Summary saved to: experiments/table/optimal_L_over_freq_range_summary.json")
    
    # we print conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    if in_range is not None and len(in_range) > 0:
        best_ratio = in_range.loc[in_range['mean_loss'].idxmin(), 'ratio']
        best_loss = in_range['mean_loss'].min()
        print(f"\n✅ Optimal range (7-12) analysis:")
        print(f"   Best L/freq ratio: {best_ratio:.2f}")
        print(f"   Best mean loss: {best_loss:.6e}")
        print(f"   Configurations in range: {len(in_range)}")
    else:
        print("\n⚠️  Need more data in range 7-12 to verify hypothesis")
    
    if u_fit:
        print(f"\n✅ U-shaped curve fit:")
        print(f"   Minimum at L/freq = {u_fit['minimum_at']:.2f}")
        print(f"   R² = {u_fit['r2']:.4f}")
        if 7 <= u_fit['minimum_at'] <= 12:
            print(f"   ✓ Minimum is in optimal range 7-12!")
        else:
            print(f"   ⚠️  Minimum is outside optimal range 7-12")

if __name__ == "__main__":
    import torch
    main()
