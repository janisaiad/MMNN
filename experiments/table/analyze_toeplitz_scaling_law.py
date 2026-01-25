#!/usr/bin/env python3
"""
we analyze scaling law considering Toeplitz matrix structure
The idea: if we have a Toeplitz structure, layers might scale differently
"""
import json
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
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 18
mpl.rcParams['mathtext.fontset'] = 'cm'
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

def load_optimal_layers():
    """we load optimal layer counts"""
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
                'layers': config.get('num_layers', 0),
                'final_test_error': results.get('final_test_error'),
            }
            
            if entry['final_test_error'] is None or np.isnan(entry['final_test_error']):
                continue
            if entry['final_test_error'] > 1e10:
                continue
                
            all_results.append(entry)
            
        except:
            continue
    
    df = pd.DataFrame(all_results)
    
    # we find optimal layers per frequency
    optimal = {}
    for freq in sorted(df['freq_multiplier'].unique()):
        freq_df = df[df['freq_multiplier'] == freq]
        if len(freq_df) > 0:
            best_idx = freq_df['final_test_error'].idxmin()
            optimal[freq] = int(round(freq_df.loc[best_idx, 'layers']))
    
    return optimal

def toeplitz_scaling_models():
    """we propose various Toeplitz-inspired scaling laws"""
    
    optimal = load_optimal_layers()
    freqs = np.array(sorted(optimal.keys()))
    layers = np.array([optimal[f] for f in freqs])
    
    print("="*80)
    print("TOEPLITZ-BASED SCALING LAW ANALYSIS")
    print("="*80)
    print(f"\nData points: {len(freqs)}")
    print(f"Frequencies: {freqs}")
    print(f"Optimal layers: {layers}")
    
    # we define baseline
    baseline_layers = 8  # at freq=1.0
    print(f"\nBaseline: {baseline_layers} layers at freq×1.0")
    
    results = {}
    
    # Model 1: Simple Toeplitz L = round(freq × base)
    def toeplitz_simple(f, base):
        return np.round(f * base)
    
    base_opt, _ = curve_fit(lambda f, b: toeplitz_simple(f, b), freqs, layers, p0=[8])
    pred_simple = toeplitz_simple(freqs, base_opt[0])
    r2_simple = 1 - np.sum((layers - pred_simple)**2) / np.sum((layers - np.mean(layers))**2)
    
    results['toeplitz_simple'] = {
        'formula': f"L = round(freq × {base_opt[0]:.3f})",
        'base': base_opt[0],
        'r2': r2_simple,
        'predictions': pred_simple
    }
    
    # Model 2: Toeplitz with offset L = round(freq × base + offset)
    def toeplitz_offset(f, base, offset):
        return np.round(f * base + offset)
    
    popt, _ = curve_fit(lambda f, b, o: toeplitz_offset(f, b, o), freqs, layers, p0=[8, 0])
    pred_offset = toeplitz_offset(freqs, *popt)
    r2_offset = 1 - np.sum((layers - pred_offset)**2) / np.sum((layers - np.mean(layers))**2)
    
    results['toeplitz_offset'] = {
        'formula': f"L = round({popt[0]:.3f} × freq + {popt[1]:.3f})",
        'base': popt[0],
        'offset': popt[1],
        'r2': r2_offset,
        'predictions': pred_offset
    }
    
    # Model 3: Toeplitz with power L = round(base × freq^alpha)
    def toeplitz_power(f, base, alpha):
        return np.round(base * (f ** alpha))
    
    popt_power, _ = curve_fit(lambda f, b, a: toeplitz_power(f, b, a), freqs, layers, p0=[8, 1.0])
    pred_power = toeplitz_power(freqs, *popt_power)
    r2_power = 1 - np.sum((layers - pred_power)**2) / np.sum((layers - np.mean(layers))**2)
    
    results['toeplitz_power'] = {
        'formula': f"L = round({popt_power[0]:.3f} × freq^{popt_power[1]:.3f})",
        'base': popt_power[0],
        'alpha': popt_power[1],
        'r2': r2_power,
        'predictions': pred_power
    }
    
    # Model 4: Theoretical - if Toeplitz structure requires L ~ freq for bandwidth
    # This comes from signal processing: to represent freq, need ~freq samples
    def bandwidth_scaling(f, c):
        return np.round(c * f)
    
    popt_bw, _ = curve_fit(bandwidth_scaling, freqs, layers, p0=[8])
    pred_bw = bandwidth_scaling(freqs, popt_bw[0])
    r2_bw = 1 - np.sum((layers - pred_bw)**2) / np.sum((layers - np.mean(layers))**2)
    
    results['bandwidth'] = {
        'formula': f"L = round({popt_bw[0]:.3f} × freq)",
        'constant': popt_bw[0],
        'r2': r2_bw,
        'predictions': pred_bw
    }
    
    # Model 5: Modified Toeplitz - L = round(base × freq) but with minimum
    def toeplitz_min(f, base, min_layers):
        return np.maximum(np.round(f * base), min_layers)
    
    popt_min, _ = curve_fit(lambda f, b, m: toeplitz_min(f, b, m), freqs, layers, p0=[8, 5])
    pred_min = toeplitz_min(freqs, *popt_min)
    r2_min = 1 - np.sum((layers - pred_min)**2) / np.sum((layers - np.mean(layers))**2)
    
    results['toeplitz_min'] = {
        'formula': f"L = max(round({popt_min[0]:.3f} × freq), {popt_min[1]:.1f})",
        'base': popt_min[0],
        'min_layers': popt_min[1],
        'r2': r2_min,
        'predictions': pred_min
    }
    
    # we print results
    print("\n" + "="*80)
    print("TOEPLITZ-BASED MODELS")
    print("="*80)
    
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  {result['formula']}")
        print(f"  R² = {result['r2']:.4f}")
        print(f"  Predictions vs Actual:")
        for f, l_actual, l_pred in zip(freqs, layers, result['predictions']):
            match = "✓" if abs(l_actual - l_pred) < 0.5 else "✗"
            print(f"    freq×{f:4.1f}: L={l_actual:2d} (pred={l_pred:2.0f}) {match}")
    
    # we find best
    best = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\n{'='*80}")
    print(f"BEST TOEPLITZ MODEL: {best[0].upper()}")
    print(f"{'='*80}")
    print(f"Formula: {best[1]['formula']}")
    print(f"R² = {best[1]['r2']:.4f}")
    
    # we plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: All models comparison
    ax1 = axes[0, 0]
    ax1.scatter(freqs, layers, s=200, marker='*', color='red', zorder=5, label='Optimal L')
    
    freq_plot = np.linspace(freqs.min(), freqs.max(), 100)
    
    for name, result in results.items():
        if 'power' in name:
            pred_plot = result['base'] * (freq_plot ** result['alpha'])
        elif 'min' in name:
            pred_plot = np.maximum(np.round(freq_plot * result['base']), result['min_layers'])
        elif 'offset' in name:
            pred_plot = np.round(freq_plot * result['base'] + result['offset'])
        elif 'bandwidth' in name:
            pred_plot = np.round(freq_plot * result['constant'])
        else:
            pred_plot = np.round(freq_plot * result['base'])
        
        ax1.plot(freq_plot, pred_plot, '--', linewidth=2, alpha=0.7, 
                label=f"{name}: R²={result['r2']:.3f}")
    
    ax1.set_xlabel('Frequency Multiplier', fontsize=18)
    ax1.set_ylabel('Number of Layers (L)', fontsize=18)
    ax1.set_title('Toeplitz-Based Scaling Laws', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper left')
    
    # Plot 2: Best model residuals
    ax2 = axes[0, 1]
    best_pred = best[1]['predictions']
    residuals = layers - best_pred
    ax2.scatter(freqs, residuals, s=100, alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Frequency Multiplier', fontsize=18)
    ax2.set_ylabel('Residuals', fontsize=18)
    ax2.set_title(f'Residuals: {best[0]}', fontsize=20)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: L/freq ratio (should be constant for pure Toeplitz)
    ax3 = axes[1, 0]
    ratio = layers / freqs
    ax3.scatter(freqs, ratio, s=100, alpha=0.7)
    ax3.axhline(ratio.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {ratio.mean():.2f}')
    ax3.set_xlabel('Frequency Multiplier', fontsize=18)
    ax3.set_ylabel('L / freq (should be constant for Toeplitz)', fontsize=18)
    ax3.set_title('Toeplitz Constant Check', fontsize=20)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Log-log to see power law
    ax4 = axes[1, 1]
    ax4.loglog(freqs, layers, 'o', markersize=10, label='Data')
    
    # we plot best power law
    if 'power' in results:
        freq_log = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 100)
        pred_log = results['toeplitz_power']['base'] * (freq_log ** results['toeplitz_power']['alpha'])
        ax4.loglog(freq_log, pred_log, '--', linewidth=2, 
                  label=f"Power: {results['toeplitz_power']['formula']}")
    
    ax4.set_xlabel('Frequency Multiplier', fontsize=18)
    ax4.set_ylabel('Number of Layers (L)', fontsize=18)
    ax4.set_title('Log-Log Scale (Power Law)', fontsize=20)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('experiments/table/toeplitz_scaling_law.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: experiments/table/toeplitz_scaling_law.png")
    plt.close()
    
    return results, best, freqs, layers

def theoretical_analysis():
    """we provide theoretical justification"""
    
    print("\n" + "="*80)
    print("THEORETICAL CONSIDERATIONS")
    print("="*80)
    
    print("""
1. TOEPLITZ MATRIX STRUCTURE:
   - Toeplitz matrices have constant diagonals: T[i,j] = T[i-j]
   - In signal processing, Toeplitz structure relates to convolution
   - For frequency representation, need ~freq samples per period
   
2. BANDWIDTH ARGUMENT:
   - To represent a signal with frequency f, need sampling rate > 2f (Nyquist)
   - In neural networks: to learn freq, need sufficient depth
   - If each layer processes ~1 unit of frequency, then L ~ freq
   
3. MODIFIED TOEPLITZ:
   - Simple L = round(freq × base) assumes perfect scaling
   - But networks have minimum depth requirements
   - Also, very high frequencies may saturate (diminishing returns)
   
4. PROPOSED SCALING LAW:
   Based on the data, the best Toeplitz-inspired model is:
   L = round(base × freq + offset)
   
   This accounts for:
   - Base scaling with frequency (Toeplitz structure)
   - Minimum depth requirement (offset)
   - Practical rounding for discrete layers
""")

if __name__ == "__main__":
    import torch
    results, best, freqs, layers = toeplitz_scaling_models()
    theoretical_analysis()
    
    # we save
    summary = {
        'best_model': best[0],
        'best_formula': best[1]['formula'],
        'best_r2': float(best[1]['r2']),
        'all_models': {k: {'formula': v['formula'], 'r2': float(v['r2'])} 
                      for k, v in results.items()}
    }
    
    with open('experiments/table/toeplitz_scaling_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ Summary saved to: experiments/table/toeplitz_scaling_summary.json")
