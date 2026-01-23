#!/usr/bin/env python3
"""
we analyze frequency benchmark results and create comprehensive ablation tables
"""
import json
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import re

base_dir = Path("experiments/table/results_frequency_benchmark")

print("="*120)
print("FREQUENCY BENCHMARK ANALYSIS - FINAL LOSS vs RANK ABLATION")
print("="*120)
print()

results = []

for config_dir in sorted(base_dir.iterdir()):
    if not config_dir.is_dir():
        continue
    
    config_name = config_dir.name
    
    # we only want LOW RANK runs (not FULL_RANK)
    if "FULL_RANK" in config_name or "FULL_MLP" in config_name:
        continue
    
    checkpoint_file = config_dir / "checkpoint.pth"
    config_file = config_dir / "config.json"
    results_file = config_dir / "results.json"
    
    if not checkpoint_file.exists():
        continue
    
    try:
        # we parse config name
        parts = config_name.split("_")
        freq_part = parts[0]  # e.g., "freq144"
        rank = None
        fixWb = None
        
        for part in parts:
            if part.startswith("rank"):
                rank = int(part.replace("rank", ""))
            elif "fixWbTrue" in part:
                fixWb = True
            elif "fixWbFalse" in part:
                fixWb = False
        
        # we load config to get function string and frequencies
        config = {}
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        
        # we extract frequencies from function string
        func_str = config.get("function", "")
        freq1, freq2 = None, None
        if func_str:
            matches = re.findall(r'cos\((\d+)\*pi\*x\^2\)', func_str)
            if len(matches) >= 2:
                freq1, freq2 = int(matches[0]), int(matches[1])
            elif len(matches) == 1:
                freq1 = int(matches[0])
        
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
        
        # we use test error as primary metric (or train loss if test error not available)
        final_loss = final_test_error if final_test_error is not None else final_train_loss
        
        results.append({
            "config_name": config_name,
            "rank": rank,
            "fixWb": fixWb,
            "freq1": freq1,
            "freq2": freq2,
            "final_epoch": final_epoch,
            "final_test_error": final_test_error,
            "final_train_error": final_train_error,
            "final_train_loss": final_train_loss,
            "final_loss": final_loss,  # we use test error as primary
            "num_layers": config.get("num_layers", "N/A"),
            "hidden_width": config.get("hidden_width", "N/A"),
            "batch_size": config.get("batch_size", "N/A"),
            "n_train": config.get("num_training_samples", "N/A"),
        })
    except Exception as e:
        print(f"Error processing {config_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

# we create dataframe
df = pd.DataFrame(results)

# we sort by frequency, rank, fixWb
df = df.sort_values(['freq1', 'rank', 'fixWb'])

print(f"Collected {len(df)} results")
print(f"Runs with valid loss: {df['final_loss'].notna().sum()}\n")

# we save to CSV
df.to_csv("frequency_benchmark_results.csv", index=False)
print("✓ Saved to: frequency_benchmark_results.csv\n")

# we create comprehensive tables
print("="*120)
print("DETAILED RESULTS BY FREQUENCY")
print("="*120)

# we group by frequency and create tables
for freq1 in sorted(df['freq1'].unique()):
    freq_df = df[df['freq1'] == freq1]
    freq2_val = freq_df['freq2'].iloc[0] if freq_df['freq2'].notna().any() else None
    
    print(f"\n{'='*120}")
    print(f"FREQUENCY: f1={freq1}, f2={freq2_val} | Function: cos({freq1}*π*x²) - 0.8*cos({freq2_val}*π*x²)")
    print(f"{'='*120}")
    print(f"{'Rank':<8} {'fixWb':<8} {'Test Error':<15} {'Train Error':<15} {'Train Loss':<15} {'Epoch':<8} {'N_train':<10}")
    print("-"*120)
    
    for _, row in freq_df.iterrows():
        test_err = f"{row['final_test_error']:.6e}" if row['final_test_error'] is not None and not pd.isna(row['final_test_error']) else "N/A"
        train_err = f"{row['final_train_error']:.6e}" if row['final_train_error'] is not None and not pd.isna(row['final_train_error']) else "N/A"
        train_loss = f"{row['final_train_loss']:.6e}" if row['final_train_loss'] is not None and not pd.isna(row['final_train_loss']) else "N/A"
        print(f"{row['rank']:<8} {str(row['fixWb']):<8} {test_err:<15} {train_err:<15} {train_loss:<15} {int(row['final_epoch']):<8} {row['n_train']:<10}")

# we create summary statistics
print("\n" + "="*120)
print("SUMMARY STATISTICS - BY RANK")
print("="*120)
print(f"{'Rank':<8} {'Mean Test Err':<18} {'Std Test Err':<18} {'Min Test Err':<18} {'Max Test Err':<18} {'Count':<8}")
print("-"*120)
for rank in sorted(df['rank'].unique()):
    rank_df = df[df['rank'] == rank]
    test_errors = rank_df['final_test_error'].dropna()
    if len(test_errors) > 0:
        print(f"{rank:<8} {test_errors.mean():<18.6e} {test_errors.std():<18.6e} {test_errors.min():<18.6e} {test_errors.max():<18.6e} {len(test_errors):<8}")

print("\n" + "="*120)
print("SUMMARY STATISTICS - BY fixWb")
print("="*120)
print(f"{'fixWb':<8} {'Mean Test Err':<18} {'Std Test Err':<18} {'Min Test Err':<18} {'Max Test Err':<18} {'Count':<8}")
print("-"*120)
for fixWb in [False, True]:
    fixWb_df = df[df['fixWb'] == fixWb]
    test_errors = fixWb_df['final_test_error'].dropna()
    if len(test_errors) > 0:
        print(f"{str(fixWb):<8} {test_errors.mean():<18.6e} {test_errors.std():<18.6e} {test_errors.min():<18.6e} {test_errors.max():<18.6e} {len(test_errors):<8}")

print("\n" + "="*120)
print("SUMMARY STATISTICS - BY FREQUENCY")
print("="*120)
print(f"{'Freq':<8} {'Mean Test Err':<18} {'Std Test Err':<18} {'Min Test Err':<18} {'Max Test Err':<18} {'Count':<8}")
print("-"*120)
for freq1 in sorted(df['freq1'].unique()):
    freq_df = df[df['freq1'] == freq1]
    test_errors = freq_df['final_test_error'].dropna()
    if len(test_errors) > 0:
        print(f"{freq1:<8} {test_errors.mean():<18.6e} {test_errors.std():<18.6e} {test_errors.min():<18.6e} {test_errors.max():<18.6e} {len(test_errors):<8}")

# we create pivot table: rank vs fixWb for each frequency
print("\n" + "="*120)
print("PIVOT TABLES - TEST ERROR BY RANK AND fixWb")
print("="*120)

for freq1 in sorted(df['freq1'].unique()):
    freq_df = df[df['freq1'] == freq1]
    freq2_val = freq_df['freq2'].iloc[0] if freq_df['freq2'].notna().any() else None
    
    print(f"\nFrequency {freq1} (f2={freq2_val}):")
    print("-"*80)
    pivot = freq_df.pivot_table(
        values='final_test_error',
        index='rank',
        columns='fixWb',
        aggfunc='mean'
    )
    print(pivot.to_string())
    print()

print("\n✓ Analysis complete!")
