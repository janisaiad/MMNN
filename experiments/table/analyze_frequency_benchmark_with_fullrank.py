#!/usr/bin/env python3
"""
we analyze frequency benchmark results including full rank (777) comparisons
creates comprehensive ablation tables comparing low-rank vs full-rank
"""
import json
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import re

print("="*120)
print("FREQUENCY BENCHMARK ANALYSIS - INCLUDING FULL RANK (777) COMPARISONS")
print("="*120)
print()

results = []

# we collect from multiple directories
directories = [
    Path("experiments/table/results_frequency_benchmark"),
    Path("experiments/table/results_frequency_benchmark_comprehensive"),
    Path("experiments/table/results_frequency_benchmark_full_rank"),
]

for base_dir in directories:
    if not base_dir.exists():
        continue
    
    print(f"Scanning {base_dir}...")
    
    for config_dir in sorted(base_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        
        config_name = config_dir.name
        checkpoint_file = config_dir / "checkpoint.pth"
        config_file = config_dir / "config.json"
        results_file = config_dir / "results.json"
        
        if not checkpoint_file.exists() or not config_file.exists():
            continue
        
        try:
            # we load config
            with open(config_file) as f:
                config = json.load(f)
            
            rank = config.get("hidden_rank")
            fixWb = config.get("fixWb", False)
            freq1 = config.get("freq1")
            freq2 = config.get("freq2")
            batch_size = config.get("batch_size", 100)
            
            # we extract frequencies from function string if not in config
            if freq1 is None:
                func_str = config.get("function", "")
                if func_str:
                    matches = re.findall(r'cos\((\d+)\*pi\*x\^2\)', func_str)
                    if len(matches) >= 2:
                        freq1, freq2 = int(matches[0]), int(matches[1])
                    elif len(matches) == 1:
                        freq1 = int(matches[0])
            
            # we determine if full rank
            is_full_rank = (rank == config.get("hidden_width", 777)) or (rank == 777) or ("FULL_RANK" in config_name)
            if is_full_rank:
                rank_label = "FULL_RANK (777)"
                rank_display = 777
            else:
                rank_label = f"rank={rank}"
                rank_display = rank
            
            # we load checkpoint
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            final_epoch = ckpt.get('epoch', 0)
            
            # we get final loss from all_losses (training loss)
            final_train_loss = None
            if 'all_losses' in ckpt and len(ckpt['all_losses']) > 0:
                final_train_loss = float(ckpt['all_losses'][-1])
            
            # we load results.json for test errors
            final_test_error = None
            final_train_error = None
            if results_file.exists():
                with open(results_file) as f:
                    res_data = json.load(f)
                    final_test_error = res_data.get('final_test_error')
                    final_train_error = res_data.get('final_train_error')
            
            # we use test error as primary metric
            final_loss = final_test_error if final_test_error is not None else final_train_loss
            
            results.append({
                "config_name": config_name,
                "rank": rank_display,
                "rank_label": rank_label,
                "is_full_rank": is_full_rank,
                "fixWb": fixWb,
                "freq1": freq1,
                "freq2": freq2,
                "batch_size": batch_size,
                "final_epoch": final_epoch,
                "final_test_error": final_test_error,
                "final_train_error": final_train_error,
                "final_train_loss": final_train_loss,
                "final_loss": final_loss,
                "num_layers": config.get("num_layers", "N/A"),
                "hidden_width": config.get("hidden_width", "N/A"),
                "n_train": config.get("num_training_samples", "N/A"),
            })
        except Exception as e:
            print(f"Error processing {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

# we create dataframe
df = pd.DataFrame(results)

# we filter to completed runs (>= 8000 epochs to include the 8500 one)
df = df[df['final_epoch'] >= 8000]

# we sort by frequency, rank, fixWb, batch_size
df = df.sort_values(['freq1', 'rank', 'fixWb', 'batch_size'])

print(f"Collected {len(df)} results")
print(f"Runs with valid loss: {df['final_loss'].notna().sum()}\n")

# we save to CSV
df.to_csv("frequency_benchmark_results_with_fullrank.csv", index=False)
print("✓ Saved to: frequency_benchmark_results_with_fullrank.csv\n")

# we create comprehensive comparison tables
print("="*120)
print("COMPARISON TABLES: LOW-RANK vs FULL-RANK (777)")
print("="*120)

# we group by frequency and create comparison tables
for freq1 in sorted(df['freq1'].unique()):
    freq_df = df[df['freq1'] == freq1]
    freq2_val = freq_df['freq2'].iloc[0] if freq_df['freq2'].notna().any() else None
    
    print(f"\n{'='*120}")
    print(f"FREQUENCY: f1={freq1}, f2={freq2_val} | Function: cos({freq1}*π*x²) - 0.8*cos({freq2_val}*π*x²)")
    print(f"{'='*120}")
    
    # we create comparison table: rank vs fixWb, including full rank
    # we use batch_size=100 for comparison (baseline)
    baseline_df = freq_df[freq_df['batch_size'] == 100]
    
    if len(baseline_df) > 0:
        print(f"\nCOMPARISON TABLE (Batch Size = 100):")
        print(f"{'Rank':<20} {'fixWb':<8} {'Test Error':<15} {'Train Error':<15} {'Epoch':<8} {'N_train':<10}")
        print("-"*120)
        
        for _, row in baseline_df.iterrows():
            test_err = f"{row['final_test_error']:.6e}" if row['final_test_error'] is not None and not pd.isna(row['final_test_error']) else "N/A"
            train_err = f"{row['final_train_error']:.6e}" if row['final_train_error'] is not None and not pd.isna(row['final_train_error']) else "N/A"
            rank_str = row['rank_label'] if row['is_full_rank'] else f"rank={int(row['rank'])}"
            print(f"{rank_str:<20} {str(row['fixWb']):<8} {test_err:<15} {train_err:<15} {int(row['final_epoch']):<8} {row['n_train']:<10}")
    
    # we also show batch size comparison for non-full-rank
    print(f"\nBATCH SIZE COMPARISON (Low-Rank Only):")
    lowrank_df = freq_df[~freq_df['is_full_rank']]
    if len(lowrank_df) > 0:
        print(f"{'Rank':<8} {'fixWb':<8} {'Batch':<8} {'Test Error':<15} {'Train Error':<15}")
        print("-"*80)
        for _, row in lowrank_df.iterrows():
            test_err = f"{row['final_test_error']:.6e}" if row['final_test_error'] is not None and not pd.isna(row['final_test_error']) else "N/A"
            train_err = f"{row['final_train_error']:.6e}" if row['final_train_error'] is not None and not pd.isna(row['final_train_error']) else "N/A"
            print(f"{int(row['rank']):<8} {str(row['fixWb']):<8} {int(row['batch_size']):<8} {test_err:<15} {train_err:<15}")

# we create summary statistics including full rank
print("\n" + "="*120)
print("SUMMARY STATISTICS - BY RANK (INCLUDING FULL RANK 777)")
print("="*120)
print(f"{'Rank':<20} {'Mean Test Err':<18} {'Std Test Err':<18} {'Min Test Err':<18} {'Max Test Err':<18} {'Count':<8}")
print("-"*120)

# we group by rank (including full rank)
for rank_val in sorted(df['rank'].unique()):
    rank_df = df[df['rank'] == rank_val]
    test_errors = rank_df['final_test_error'].dropna()
    if len(test_errors) > 0:
        rank_label = "FULL_RANK (777)" if rank_val == 777 else f"rank={int(rank_val)}"
        print(f"{rank_label:<20} {test_errors.mean():<18.6e} {test_errors.std():<18.6e} {test_errors.min():<18.6e} {test_errors.max():<18.6e} {len(test_errors):<8}")

# we create direct comparison: best low-rank vs full-rank
print("\n" + "="*120)
print("DIRECT COMPARISON: BEST LOW-RANK vs FULL-RANK (777)")
print("="*120)

for freq1 in sorted(df['freq1'].unique()):
    freq_df = df[df['freq1'] == freq1]
    freq2_val = freq_df['freq2'].iloc[0] if freq_df['freq2'].notna().any() else None
    
    # we get best low-rank (batch=100)
    lowrank_batch100 = freq_df[(~freq_df['is_full_rank']) & (freq_df['batch_size'] == 100)]
    fullrank_batch100 = freq_df[(freq_df['is_full_rank']) & (freq_df['batch_size'] == 100)]
    
    print(f"\nFrequency {freq1} (f2={freq2_val}):")
    print("-"*80)
    
    if len(lowrank_batch100) > 0:
        best_lowrank = lowrank_batch100.loc[lowrank_batch100['final_test_error'].idxmin()]
        print(f"Best Low-Rank: rank={int(best_lowrank['rank'])}, fixWb={best_lowrank['fixWb']}, "
              f"test_error={best_lowrank['final_test_error']:.6e}")
    
    if len(fullrank_batch100) > 0:
        for _, row in fullrank_batch100.iterrows():
            print(f"Full-Rank (777): fixWb={row['fixWb']}, "
                  f"test_error={row['final_test_error']:.6e if row['final_test_error'] is not None else 'N/A'}")
    
    # we compute ratio if both exist
    if len(lowrank_batch100) > 0 and len(fullrank_batch100) > 0:
        best_lowrank_err = best_lowrank['final_test_error']
        for _, full_row in fullrank_batch100.iterrows():
            if full_row['final_test_error'] is not None and not pd.isna(full_row['final_test_error']):
                ratio = best_lowrank_err / full_row['final_test_error']
                print(f"  Ratio (best_lowrank / fullrank): {ratio:.3f}x")

# we create pivot table: rank vs fixWb including full rank
print("\n" + "="*120)
print("PIVOT TABLES - TEST ERROR BY RANK AND fixWb (INCLUDING FULL RANK)")
print("="*120)

for freq1 in sorted(df['freq1'].unique()):
    freq_df = df[df['freq1'] == freq1]
    freq2_val = freq_df['freq2'].iloc[0] if freq_df['freq2'].notna().any() else None
    
    # we use batch_size=100 for fair comparison
    freq_df_batch100 = freq_df[freq_df['batch_size'] == 100]
    
    print(f"\nFrequency {freq1} (f2={freq2_val}, batch=100):")
    print("-"*80)
    
    # we create pivot with rank labels
    pivot_data = []
    for _, row in freq_df_batch100.iterrows():
        rank_label = "FULL_RANK" if row['is_full_rank'] else f"rank{int(row['rank'])}"
        pivot_data.append({
            'rank_label': rank_label,
            'rank_val': row['rank'],
            'fixWb': row['fixWb'],
            'test_error': row['final_test_error']
        })
    
    if pivot_data:
        pivot_df = pd.DataFrame(pivot_data)
        pivot = pivot_df.pivot_table(
            values='test_error',
            index='rank_label',
            columns='fixWb',
            aggfunc='mean'
        )
        # we sort by rank value
        rank_order = ['rank10', 'rank15', 'rank20', 'rank25', 'rank50', 'rank100', 'FULL_RANK']
        pivot = pivot.reindex([r for r in rank_order if r in pivot.index])
        print(pivot.to_string())
        print()

print("\n✓ Analysis complete with full rank comparisons!")
