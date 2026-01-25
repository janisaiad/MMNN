#!/usr/bin/env python3
"""
we extract all training results and create comprehensive tables with conclusions
"""
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
            
            # we check if training completed
            if checkpoint_file.exists():
                ckpt = torch.load(checkpoint_file, map_location='cpu')
                epoch = ckpt.get('epoch', 0)
                target_epochs = config.get('num_epochs', 0)
                if epoch < target_epochs:
                    continue  # skip incomplete
            
            # we extract key metrics
            entry = {
                'config_name': config_dir.name,
                'freq_multiplier': config.get('freq_multiplier', 0),
                'rank': config.get('hidden_rank', 0),
                'layers': config.get('num_layers', 0),
                'epochs': config.get('num_epochs', 0),
                'final_train_error': results.get('final_train_error'),
                'final_test_error': results.get('final_test_error'),
                'final_test_error_max': results.get('final_test_error_max'),
                'training_time_seconds': results.get('training_time_seconds'),
                'total_parameters': results.get('total_parameters'),
                'epochs_run': results.get('epochs_run', 0),
                'thresholds_reached': len(results.get('thresholds_reached', [])),
            }
            
            # we extract loss trajectory info
            all_losses = results.get('all_losses', [])
            if all_losses:
                entry['initial_loss'] = all_losses[0] if len(all_losses) > 0 else None
                entry['final_loss'] = all_losses[-1] if len(all_losses) > 0 else None
                entry['loss_reduction_factor'] = all_losses[0] / all_losses[-1] if len(all_losses) > 0 and all_losses[-1] > 0 else None
                entry['loss_at_50pct'] = all_losses[len(all_losses)//2] if len(all_losses) > 0 else None
            
            all_results.append(entry)
            
        except Exception as e:
            print(f"Error loading {config_dir.name}: {e}")
            continue
    
    return all_results

def create_summary_tables(all_results):
    """we create comprehensive summary tables"""
    
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("No completed results found!")
        return
    
    print("="*80)
    print("FREQUENCY AND LAYER SCALING EXPERIMENTS - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"\nTotal completed configurations: {len(df)}")
    
    # we create main results table
    print("\n" + "="*80)
    print("TABLE 1: MAIN RESULTS BY CONFIGURATION")
    print("="*80)
    
    display_cols = [
        'config_name', 'freq_multiplier', 'rank', 'layers', 'epochs',
        'final_train_error', 'final_test_error', 'final_test_error_max',
        'thresholds_reached', 'training_time_seconds'
    ]
    
    display_df = df[display_cols].copy()
    display_df = display_df.sort_values(['freq_multiplier', 'rank', 'layers'])
    
    # we format for display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(display_df.to_string(index=False))
    
    # we create summary by frequency multiplier
    print("\n" + "="*80)
    print("TABLE 2: SUMMARY BY FREQUENCY MULTIPLIER")
    print("="*80)
    
    freq_summary = df.groupby('freq_multiplier').agg({
        'final_test_error': ['mean', 'std', 'min', 'max'],
        'final_test_error_max': ['mean', 'std', 'min', 'max'],
        'training_time_seconds': ['mean', 'sum'],
        'thresholds_reached': 'mean',
        'config_name': 'count'
    }).round(6)
    
    freq_summary.columns = ['_'.join(col).strip() for col in freq_summary.columns.values]
    freq_summary = freq_summary.rename(columns={'config_name_count': 'num_configs'})
    print(freq_summary.to_string())
    
    # we create summary by rank
    print("\n" + "="*80)
    print("TABLE 3: SUMMARY BY RANK")
    print("="*80)
    
    rank_summary = df.groupby('rank').agg({
        'final_test_error': ['mean', 'std', 'min', 'max'],
        'final_test_error_max': ['mean', 'std', 'min', 'max'],
        'training_time_seconds': ['mean', 'sum'],
        'thresholds_reached': 'mean',
        'config_name': 'count'
    }).round(6)
    
    rank_summary.columns = ['_'.join(col).strip() for col in rank_summary.columns.values]
    rank_summary = rank_summary.rename(columns={'config_name_count': 'num_configs'})
    print(rank_summary.to_string())
    
    # we create summary by layers
    print("\n" + "="*80)
    print("TABLE 4: SUMMARY BY NUMBER OF LAYERS")
    print("="*80)
    
    layer_summary = df.groupby('layers').agg({
        'final_test_error': ['mean', 'std', 'min', 'max'],
        'final_test_error_max': ['mean', 'std', 'min', 'max'],
        'training_time_seconds': ['mean', 'sum'],
        'thresholds_reached': 'mean',
        'config_name': 'count'
    }).round(6)
    
    layer_summary.columns = ['_'.join(col).strip() for col in layer_summary.columns.values]
    layer_summary = layer_summary.rename(columns={'config_name_count': 'num_configs'})
    print(layer_summary.to_string())
    
    # we create cross-analysis: frequency × rank
    print("\n" + "="*80)
    print("TABLE 5: CROSS-ANALYSIS - FREQUENCY × RANK")
    print("="*80)
    
    cross_freq_rank = df.groupby(['freq_multiplier', 'rank']).agg({
        'final_test_error': 'mean',
        'final_test_error_max': 'mean',
        'config_name': 'count'
    }).round(6)
    cross_freq_rank = cross_freq_rank.rename(columns={'config_name': 'num_configs'})
    print(cross_freq_rank.to_string())
    
    # we create cross-analysis: frequency × layers
    print("\n" + "="*80)
    print("TABLE 6: CROSS-ANALYSIS - FREQUENCY × LAYERS")
    print("="*80)
    
    cross_freq_layers = df.groupby(['freq_multiplier', 'layers']).agg({
        'final_test_error': 'mean',
        'final_test_error_max': 'mean',
        'config_name': 'count'
    }).round(6)
    cross_freq_layers = cross_freq_layers.rename(columns={'config_name': 'num_configs'})
    print(cross_freq_layers.to_string())
    
    return df, freq_summary, rank_summary, layer_summary

def draw_conclusions(df, freq_summary, rank_summary, layer_summary):
    """we draw conclusions from the data"""
    
    print("\n" + "="*80)
    print("CONCLUSIONS AND INSIGHTS")
    print("="*80)
    
    # we find best and worst configurations
    best_test_error = df.loc[df['final_test_error'].idxmin()]
    worst_test_error = df.loc[df['final_test_error'].idxmax()]
    
    best_max_error = df.loc[df['final_test_error_max'].idxmin()]
    worst_max_error = df.loc[df['final_test_error_max'].idxmax()]
    
    print("\n1. BEST AND WORST PERFORMING CONFIGURATIONS:")
    print(f"   Best test error: {best_test_error['config_name']}")
    print(f"     - freq_mult={best_test_error['freq_multiplier']}, rank={best_test_error['rank']}, layers={best_test_error['layers']}")
    print(f"     - test_error={best_test_error['final_test_error']:.6f}, max_error={best_test_error['final_test_error_max']:.6f}")
    
    print(f"\n   Worst test error: {worst_test_error['config_name']}")
    print(f"     - freq_mult={worst_test_error['freq_multiplier']}, rank={worst_test_error['rank']}, layers={worst_test_error['layers']}")
    print(f"     - test_error={worst_test_error['final_test_error']:.6f}, max_error={worst_test_error['final_test_error_max']:.6f}")
    
    print(f"\n   Best max error: {best_max_error['config_name']}")
    print(f"     - freq_mult={best_max_error['freq_multiplier']}, rank={best_max_error['rank']}, layers={best_max_error['layers']}")
    print(f"     - test_error={best_max_error['final_test_error']:.6f}, max_error={best_max_error['final_test_error_max']:.6f}")
    
    # we analyze frequency scaling
    print("\n2. FREQUENCY SCALING ANALYSIS:")
    freq_means = df.groupby('freq_multiplier')['final_test_error'].mean().sort_index()
    print("   Average test error by frequency multiplier:")
    for freq, err in freq_means.items():
        print(f"     freq×{freq:4.1f}: {err:.6f}")
    
    # we check if error increases with frequency
    freq_trend = np.corrcoef(freq_means.index.values, freq_means.values)[0,1]
    if freq_trend > 0.3:
        print(f"   → Error tends to INCREASE with frequency (correlation: {freq_trend:.3f})")
    elif freq_trend < -0.3:
        print(f"   → Error tends to DECREASE with frequency (correlation: {freq_trend:.3f})")
    else:
        print(f"   → No clear frequency trend (correlation: {freq_trend:.3f})")
    
    # we analyze rank effect
    print("\n3. RANK EFFECT ANALYSIS:")
    rank_means = df.groupby('rank')['final_test_error'].mean().sort_index()
    print("   Average test error by rank:")
    for rank, err in rank_means.items():
        print(f"     rank={rank:2d}: {err:.6f}")
    
    rank_trend = np.corrcoef(rank_means.index.values, rank_means.values)[0,1]
    if rank_trend < -0.3:
        print(f"   → Higher rank IMPROVES performance (correlation: {rank_trend:.3f})")
    elif rank_trend > 0.3:
        print(f"   → Higher rank WORSENS performance (correlation: {rank_trend:.3f})")
    else:
        print(f"   → Rank has minimal effect (correlation: {rank_trend:.3f})")
    
    # we analyze layer effect
    print("\n4. LAYER COUNT EFFECT ANALYSIS:")
    layer_means = df.groupby('layers')['final_test_error'].mean().sort_index()
    print("   Average test error by number of layers:")
    for layers, err in layer_means.items():
        print(f"     L={layers:2d}: {err:.6f}")
    
    layer_trend = np.corrcoef(layer_means.index.values, layer_means.values)[0,1]
    if layer_trend < -0.3:
        print(f"   → More layers IMPROVE performance (correlation: {layer_trend:.3f})")
    elif layer_trend > 0.3:
        print(f"   → More layers WORSEN performance (correlation: {layer_trend:.3f})")
    else:
        print(f"   → Layer count has minimal effect (correlation: {layer_trend:.3f})")
    
    # we analyze optimal layer scaling for each frequency
    print("\n5. OPTIMAL LAYER SCALING BY FREQUENCY:")
    for freq in sorted(df['freq_multiplier'].unique()):
        freq_df = df[df['freq_multiplier'] == freq]
        if len(freq_df) > 0:
            best_layer = freq_df.loc[freq_df['final_test_error'].idxmin(), 'layers']
            avg_error = freq_df['final_test_error'].mean()
            print(f"   freq×{freq:4.1f}: best layers={best_layer:2d}, avg_error={avg_error:.6f}")
    
    # we analyze training efficiency
    print("\n6. TRAINING EFFICIENCY:")
    avg_time = df['training_time_seconds'].mean()
    total_time = df['training_time_seconds'].sum()
    print(f"   Average training time per config: {avg_time/3600:.2f} hours")
    print(f"   Total training time: {total_time/3600:.2f} hours")
    
    # we analyze threshold achievement
    print("\n7. THRESHOLD ACHIEVEMENT:")
    avg_thresholds = df['thresholds_reached'].mean()
    max_thresholds = df['thresholds_reached'].max()
    print(f"   Average thresholds reached: {avg_thresholds:.1f}/25")
    print(f"   Maximum thresholds reached: {max_thresholds}/25")
    best_threshold_config = df.loc[df['thresholds_reached'].idxmax(), 'config_name']
    print(f"   Best: {best_threshold_config}")
    
    # we create recommendations
    print("\n8. RECOMMENDATIONS:")
    
    # we find best configuration overall
    best_overall = df.loc[df['final_test_error'].idxmin()]
    print(f"   Best overall configuration:")
    print(f"     - freq_mult={best_overall['freq_multiplier']}, rank={best_overall['rank']}, layers={best_overall['layers']}")
    print(f"     - test_error={best_overall['final_test_error']:.6f}")
    
    # we find best for each frequency
    print(f"\n   Recommended layer count for each frequency:")
    for freq in sorted(df['freq_multiplier'].unique()):
        freq_df = df[df['freq_multiplier'] == freq]
        if len(freq_df) > 0:
            best = freq_df.loc[freq_df['final_test_error'].idxmin()]
            print(f"     freq×{freq:4.1f} → L={best['layers']:2d} (error={best['final_test_error']:.6f})")
    
    # we analyze rank recommendations
    print(f"\n   Recommended rank:")
    best_rank = df.loc[df['final_test_error'].idxmin(), 'rank']
    print(f"     - Best performing rank: {best_rank}")
    rank_avg = df.groupby('rank')['final_test_error'].mean()
    best_rank_avg = rank_avg.idxmin()
    print(f"     - Best average rank: {best_rank_avg} (avg_error={rank_avg[best_rank_avg]:.6f})")

def main():
    """we run the analysis"""
    print("Loading all training results...")
    all_results = load_all_results()
    
    if len(all_results) == 0:
        print("No completed results found!")
        return
    
    print(f"Loaded {len(all_results)} completed configurations")
    
    # we create tables
    df, freq_summary, rank_summary, layer_summary = create_summary_tables(all_results)
    
    # we draw conclusions
    draw_conclusions(df, freq_summary, rank_summary, layer_summary)
    
    # we save to CSV
    output_file = Path("experiments/table/frequency_layer_scaling_analysis.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # we save summary
    summary_file = Path("experiments/table/frequency_layer_scaling_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("FREQUENCY AND LAYER SCALING EXPERIMENTS - SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total completed configurations: {len(df)}\n\n")
        f.write("FREQUENCY SUMMARY:\n")
        f.write(str(freq_summary) + "\n\n")
        f.write("RANK SUMMARY:\n")
        f.write(str(rank_summary) + "\n\n")
        f.write("LAYER SUMMARY:\n")
        f.write(str(layer_summary) + "\n")
    
    print(f"✓ Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
