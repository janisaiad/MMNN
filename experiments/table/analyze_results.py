#!/usr/bin/env python3
"""we analyze the comprehensive training results"""
import json
from pathlib import Path
import numpy as np

summary_file = Path('experiments/table/results_1d_comprehensive/comprehensive_summary.json')
with open(summary_file) as f:
    data = json.load(f)

print("="*80)
print("COMPREHENSIVE RESULTS ANALYSIS")
print("="*80)

print(f"\nTotal networks trained: {len(data)}")

print("\nComparison: fixWb=False vs fixWb=True by Rank")
print("="*80)
print(f"{'Rank':<8} {'fixWb=False':<20} {'fixWb=True':<20} {'Difference':<15} {'% Change':<10}")
print("-"*80)

for rank in [3, 6, 10, 15, 25, 50, 1024]:
    false_configs = [r for r in data if r.get('config', {}).get('fixWb') == False and r.get('config', {}).get('rank') == rank]
    true_configs = [r for r in data if r.get('config', {}).get('fixWb') == True and r.get('config', {}).get('rank') == rank]
    
    false_errors = [r.get('final_test_error') for r in false_configs if r.get('final_test_error') is not None]
    true_errors = [r.get('final_test_error') for r in true_configs if r.get('final_test_error') is not None]
    
    if false_errors and true_errors:
        false_mean = np.mean(false_errors)
        true_mean = np.mean(true_errors)
        diff = true_mean - false_mean
        diff_pct = (diff / false_mean) * 100
        print(f"{rank:<8} {false_mean:.4e}      {true_mean:.4e}      {diff:+.4e}    {diff_pct:+.1f}%")

print("\nParameter Count Comparison")
print("="*80)
print(f"{'Rank':<8} {'Total Params':<15} {'Trainable (fixWb=False)':<25} {'Trainable (fixWb=True)':<25}")
print("-"*80)

input_dim = 1
output_dim = 1
num_layers = 5
hidden_width = 1024

for rank in [3, 6, 10, 15, 25, 50, 1024]:
    # we get actual parameter counts from results
    false_config = next((r for r in data if r.get('config', {}).get('fixWb') == False and r.get('config', {}).get('rank') == rank), None)
    true_config = next((r for r in data if r.get('config', {}).get('fixWb') == True and r.get('config', {}).get('rank') == rank), None)
    
    if false_config and true_config:
        total_params = false_config.get('total_parameters', 0)
        trainable_false = false_config.get('trainable_parameters', 0)
        trainable_true = true_config.get('trainable_parameters', 0)
        print(f"{rank:<8} {total_params:<15,} {trainable_false:<25,} {trainable_true:<25,}")

print("\nBest Configurations by Category")
print("="*80)

categories = [
    ("Best Overall", lambda r: r.get('final_test_error', float('inf'))),
    ("Best fixWb=False", lambda r: r.get('final_test_error', float('inf')) if r.get('config', {}).get('fixWb') == False else float('inf')),
    ("Best fixWb=True", lambda r: r.get('final_test_error', float('inf')) if r.get('config', {}).get('fixWb') == True else float('inf')),
    ("Best Low Rank (3-10)", lambda r: r.get('final_test_error', float('inf')) if r.get('config', {}).get('rank') in [3, 6, 10] else float('inf')),
    ("Best Medium Rank (15-25)", lambda r: r.get('final_test_error', float('inf')) if r.get('config', {}).get('rank') in [15, 25] else float('inf')),
]

for cat_name, key_func in categories:
    best = min([r for r in data if r.get('final_test_error') is not None], key=key_func)
    cfg = best.get('config', {})
    print(f"\n{cat_name}:")
    print(f"  Benchmark: {cfg.get('benchmark_name')}")
    print(f"  fixWb: {cfg.get('fixWb')}")
    print(f"  rank: {cfg.get('rank')}")
    print(f"  seed: {cfg.get('seed')}")
    print(f"  Test error: {best.get('final_test_error'):.4e}")
    print(f"  Parameters: {best.get('total_parameters', 0):,} total, {best.get('trainable_parameters', 0):,} trainable")
