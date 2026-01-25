#!/usr/bin/env python3
"""
we analyze scaling law for layers (L) vs frequency multiplier
considering Toeplitz matrix structure
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
from scipy.stats import pearsonr

# we configure matplotlib
plt.rcParams['figure.figsize'] = [8, 6]
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

results_dir = Path("experiments/table/results_frequency_layer_scaling")

def load_all_results():
    """we load all completed training results"""
    all_results = []
    
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
            
            if checkpoint_file.exists():
                ckpt = torch.load(checkpoint_file, map_location='cpu')
                epoch = ckpt.get('epoch', 0)
                target_epochs = config.get('num_epochs', 0)
                if epoch < target_epochs:
                    continue
            
            entry = {
                'freq_multiplier': config.get('freq_multiplier', 0),
                'rank': config.get('hidden_rank', 0),
                'layers': config.get('num_layers', 0),
                'final_test_error': results.get('final_test_error'),
            }
            
            # we skip NaN or extreme outliers (training failures)
            if entry['final_test_error'] is None or np.isnan(entry['final_test_error']):
                continue
            if entry['final_test_error'] > 1e10:  # skip training failures
                continue
                
            all_results.append(entry)
            
        except Exception as e:
            continue
    
    return all_results

def find_optimal_layers_per_freq(df):
    """we find optimal layer count for each frequency"""
    optimal = {}
    
    for freq in sorted(df['freq_multiplier'].unique()):
        freq_df = df[df['freq_multiplier'] == freq]
        if len(freq_df) > 0:
            # we find best layer count (lowest error)
            best_idx = freq_df['final_test_error'].idxmin()
            best = freq_df.loc[best_idx]
            optimal[freq] = {
                'best_layers': best['layers'],
                'best_error': best['final_test_error'],
                'all_layers': sorted(freq_df['layers'].unique()),
                'all_errors': {layers: freq_df[freq_df['layers'] == layers]['final_test_error'].mean() 
                               for layers in freq_df['layers'].unique()}
            }
    
    return optimal

def fit_scaling_laws(freqs, layers):
    """we fit various scaling law models"""
    freqs = np.array(freqs)
    layers = np.array(layers)
    
    # we remove any invalid values
    mask = np.isfinite(freqs) & np.isfinite(layers) & (freqs > 0) & (layers > 0)
    freqs = freqs[mask]
    layers = layers[mask]
    
    results = {}
    
    # Model 1: Linear scaling L = a * freq + b
    def linear_model(f, a, b):
        return a * f + b
    
    try:
        popt_linear, _ = curve_fit(linear_model, freqs, layers)
        pred_linear = linear_model(freqs, *popt_linear)
        r2_linear = 1 - np.sum((layers - pred_linear)**2) / np.sum((layers - np.mean(layers))**2)
        results['linear'] = {
            'params': popt_linear,
            'formula': f"L = {popt_linear[0]:.3f} * freq + {popt_linear[1]:.3f}",
            'r2': r2_linear,
            'predictions': pred_linear
        }
    except:
        results['linear'] = None
    
    # Model 2: Power law L = a * freq^alpha
    def power_model(f, a, alpha):
        return a * (f ** alpha)
    
    try:
        popt_power, _ = curve_fit(power_model, freqs, layers, p0=[8, 1.0])
        pred_power = power_model(freqs, *popt_power)
        r2_power = 1 - np.sum((layers - pred_power)**2) / np.sum((layers - np.mean(layers))**2)
        results['power'] = {
            'params': popt_power,
            'formula': f"L = {popt_power[0]:.3f} * freq^{popt_power[1]:.3f}",
            'r2': r2_power,
            'predictions': pred_power
        }
    except:
        results['power'] = None
    
    # Model 3: Logarithmic L = a * log(freq) + b
    def log_model(f, a, b):
        return a * np.log(f) + b
    
    try:
        popt_log, _ = curve_fit(log_model, freqs, layers)
        pred_log = log_model(freqs, *popt_log)
        r2_log = 1 - np.sum((layers - pred_log)**2) / np.sum((layers - np.mean(layers))**2)
        results['log'] = {
            'params': popt_log,
            'formula': f"L = {popt_log[0]:.3f} * log(freq) + {popt_log[1]:.3f}",
            'r2': r2_log,
            'predictions': pred_log
        }
    except:
        results['log'] = None
    
    # Model 4: Toeplitz-inspired: L = round(freq * base_layers)
    # where base_layers = 8 (baseline)
    base_layers = 8
    def toeplitz_model(f):
        return np.round(f * base_layers)
    
    pred_toeplitz = toeplitz_model(freqs)
    r2_toeplitz = 1 - np.sum((layers - pred_toeplitz)**2) / np.sum((layers - np.mean(layers))**2)
    results['toeplitz'] = {
        'params': [base_layers],
        'formula': f"L = round(freq * {base_layers})",
        'r2': r2_toeplitz,
        'predictions': pred_toeplitz
    }
    
    # Model 5: Modified Toeplitz with offset: L = round(freq * base_layers + offset)
    def toeplitz_offset_model(f, base, offset):
        return np.round(f * base + offset)
    
    try:
        popt_toeplitz, _ = curve_fit(toeplitz_offset_model, freqs, layers, p0=[8, 0])
        pred_toeplitz_offset = toeplitz_offset_model(freqs, *popt_toeplitz)
        r2_toeplitz_offset = 1 - np.sum((layers - pred_toeplitz_offset)**2) / np.sum((layers - np.mean(layers))**2)
        results['toeplitz_offset'] = {
            'params': popt_toeplitz,
            'formula': f"L = round({popt_toeplitz[0]:.3f} * freq + {popt_toeplitz[1]:.3f})",
            'r2': r2_toeplitz_offset,
            'predictions': pred_toeplitz_offset
        }
    except:
        results['toeplitz_offset'] = None
    
    return results, freqs, layers

def plot_scaling_laws(freqs, layers, optimal_data, fit_results):
    """we plot the scaling law analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Linear scale - all data points
    ax1 = axes[0, 0]
    for freq in sorted(optimal_data.keys()):
        data = optimal_data[freq]
        ax1.scatter([freq] * len(data['all_layers']), 
                   data['all_layers'],
                   alpha=0.3, s=50, color='gray')
        ax1.scatter(freq, data['best_layers'], 
                   s=200, marker='*', color='red', zorder=5,
                   label='Optimal' if freq == sorted(optimal_data.keys())[0] else '')
    
    # we plot fitted models
    if fit_results['linear']:
        ax1.plot(freqs, fit_results['linear']['predictions'], 
                'b--', linewidth=2, label=f"Linear: {fit_results['linear']['formula']} (R²={fit_results['linear']['r2']:.3f})")
    
    if fit_results['power']:
        freq_plot = np.linspace(freqs.min(), freqs.max(), 100)
        ax1.plot(freq_plot, fit_results['power']['params'][0] * (freq_plot ** fit_results['power']['params'][1]),
                'g--', linewidth=2, label=f"Power: {fit_results['power']['formula']} (R²={fit_results['power']['r2']:.3f})")
    
    if fit_results['toeplitz']:
        freq_plot = np.linspace(freqs.min(), freqs.max(), 100)
        ax1.plot(freq_plot, np.round(freq_plot * fit_results['toeplitz']['params'][0]),
                'm--', linewidth=2, label=f"Toeplitz: {fit_results['toeplitz']['formula']} (R²={fit_results['toeplitz']['r2']:.3f})")
    
    ax1.set_xlabel('Frequency Multiplier', fontsize=18)
    ax1.set_ylabel('Number of Layers (L)', fontsize=18)
    ax1.set_title('Scaling Law: Layers vs Frequency (Linear Scale)', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc='upper left')
    
    # Plot 2: Log-log scale
    ax2 = axes[0, 1]
    for freq in sorted(optimal_data.keys()):
        data = optimal_data[freq]
        ax2.scatter([freq] * len(data['all_layers']), 
                   data['all_layers'],
                   alpha=0.3, s=50, color='gray')
        ax2.scatter(freq, data['best_layers'], 
                   s=200, marker='*', color='red', zorder=5)
    
    if fit_results['power']:
        freq_plot = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 100)
        ax2.loglog(freq_plot, fit_results['power']['params'][0] * (freq_plot ** fit_results['power']['params'][1]),
                  'g--', linewidth=2, label=f"Power: {fit_results['power']['formula']}")
    
    ax2.set_xlabel('Frequency Multiplier', fontsize=18)
    ax2.set_ylabel('Number of Layers (L)', fontsize=18)
    ax2.set_title('Scaling Law: Layers vs Frequency (Log-Log Scale)', fontsize=20)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=12)
    
    # Plot 3: Error vs Layers for each frequency
    ax3 = axes[1, 0]
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(optimal_data)))
    for i, (freq, color) in enumerate(zip(sorted(optimal_data.keys()), colors_map)):
        data = optimal_data[freq]
        layers_sorted = sorted(data['all_errors'].keys())
        errors_sorted = [data['all_errors'][l] for l in layers_sorted]
        ax3.plot(layers_sorted, errors_sorted, 'o-', color=color, 
                linewidth=2, markersize=8, label=f'freq×{freq}')
        # we mark optimal
        best_l = data['best_layers']
        best_e = data['best_error']
        ax3.scatter(best_l, best_e, s=300, marker='*', 
                   color=color, edgecolor='black', linewidth=1.5, zorder=5)
    
    ax3.set_xlabel('Number of Layers (L)', fontsize=18)
    ax3.set_ylabel('Test Error', fontsize=18)
    ax3.set_title('Error vs Layers for Each Frequency', fontsize=20)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, ncol=2)
    
    # Plot 4: Residuals analysis
    ax4 = axes[1, 1]
    if fit_results['toeplitz']:
        pred = fit_results['toeplitz']['predictions']
        residuals = layers - pred
        ax4.scatter(freqs, residuals, s=100, alpha=0.7)
        ax4.axhline(0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Frequency Multiplier', fontsize=18)
        ax4.set_ylabel('Residuals (L_observed - L_predicted)', fontsize=18)
        ax4.set_title('Toeplitz Model Residuals', fontsize=20)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/table/scaling_law_L_vs_freq.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: experiments/table/scaling_law_L_vs_freq.png")
    plt.close()

def main():
    """we run the scaling law analysis"""
    print("="*80)
    print("SCALING LAW ANALYSIS: LAYERS (L) vs FREQUENCY MULTIPLIER")
    print("="*80)
    
    # we load data
    all_results = load_all_results()
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("No data found!")
        return
    
    print(f"\nLoaded {len(df)} completed configurations")
    
    # we find optimal layers for each frequency
    optimal_data = find_optimal_layers_per_freq(df)
    
    print("\n" + "="*80)
    print("OPTIMAL LAYER COUNTS BY FREQUENCY")
    print("="*80)
    
    freqs = []
    optimal_layers = []
    
    for freq in sorted(optimal_data.keys()):
        data = optimal_data[freq]
        freqs.append(freq)
        optimal_layers.append(data['best_layers'])
        
        print(f"\nfreq×{freq:4.1f}:")
        print(f"  Optimal layers: {data['best_layers']} (error={data['best_error']:.6e})")
        print(f"  Tested layers: {data['all_layers']}")
        print(f"  Errors by layer:")
        for layers in sorted(data['all_errors'].keys()):
            marker = " ← BEST" if layers == data['best_layers'] else ""
            print(f"    L={layers:2d}: {data['all_errors'][layers]:.6e}{marker}")
    
    # we fit scaling laws
    print("\n" + "="*80)
    print("SCALING LAW FITS")
    print("="*80)
    
    fit_results, freqs_arr, layers_arr = fit_scaling_laws(freqs, optimal_layers)
    
    print("\nFitted Models:")
    for model_name, result in fit_results.items():
        if result:
            print(f"\n{model_name.upper()}:")
            print(f"  Formula: {result['formula']}")
            print(f"  R² = {result['r2']:.4f}")
            print(f"  Parameters: {result['params']}")
    
    # we find best model
    best_model = None
    best_r2 = -np.inf
    for model_name, result in fit_results.items():
        if result and result['r2'] > best_r2:
            best_r2 = result['r2']
            best_model = (model_name, result)
    
    if best_model:
        print(f"\n{'='*80}")
        print(f"BEST FITTING MODEL: {best_model[0].upper()}")
        print(f"{'='*80}")
        print(f"Formula: {best_model[1]['formula']}")
        print(f"R² = {best_model[1]['r2']:.4f}")
    
    # we analyze Toeplitz structure
    print("\n" + "="*80)
    print("TOEPLITZ STRUCTURE ANALYSIS")
    print("="*80)
    
    base_layers = 8  # baseline
    print(f"\nBaseline: {base_layers} layers (freq×1.0)")
    print("\nToeplitz pattern: L ≈ round(freq × base_layers)")
    print("\nComparison:")
    print(f"{'Frequency':<12} {'Optimal L':<12} {'Toeplitz L':<12} {'Match':<8}")
    print("-" * 50)
    matches = 0
    for freq, opt_l in zip(freqs, optimal_layers):
        opt_l_int = int(round(opt_l))
        toeplitz_l = int(round(freq * base_layers))
        match = "✓" if opt_l_int == toeplitz_l else "✗"
        if opt_l_int == toeplitz_l:
            matches += 1
        print(f"{freq:>11.1f}  {opt_l_int:>11d}  {toeplitz_l:>11d}  {match:>7}")
    
    match_rate = matches / len(freqs) * 100
    print(f"\nMatch rate: {matches}/{len(freqs)} ({match_rate:.1f}%)")
    
    # we plot
    plot_scaling_laws(freqs_arr, layers_arr, optimal_data, fit_results)
    
    # we save results
    results_summary = {
        'optimal_layers': {freq: optimal_data[freq]['best_layers'] for freq in freqs},
        'best_model': best_model[0] if best_model else None,
        'best_formula': best_model[1]['formula'] if best_model else None,
        'best_r2': best_model[1]['r2'] if best_model else None,
        'toeplitz_match_rate': match_rate,
    }
    
    import json
    with open('experiments/table/scaling_law_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n✓ Results saved to: experiments/table/scaling_law_results.json")

if __name__ == "__main__":
    main()
