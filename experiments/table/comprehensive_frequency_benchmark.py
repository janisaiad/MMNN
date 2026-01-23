#!/usr/bin/env python3
"""
we run comprehensive frequency benchmark with:
1. Rank 100 (new rank to test)
2. 2x batch size (batch=200) for existing configs
3. 0.5x batch size (batch=50) for existing configs
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN
from test_frequency_benchmark import train_one_config, func_from_string


def generate_comprehensive_configs():
    """we generate all comprehensive configurations"""
    configs = []
    
    # we define frequency pairs
    frequency_pairs = [
        (36, 12),   # base frequency
        (72, 24),   # 2x frequency
        (144, 48),  # 4x frequency
    ]
    
    # we base config
    base_config = {
        "num_layers": 8,
        "hidden_width": 777,
        "input_rank": 1,
        "output_rank": 1,
        "use_resnet": False,
        "num_epochs": 10000,
        "lr_init": 0.001,
        "lr_gamma": 0.9,
        "lr_step_size": 100,
        "interval": [-1, 1],
        "show_plot": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": "torch.float32"
    }
    
    # PART 1: Rank 100 runs (batch_size=100, default)
    print("="*80)
    print("PART 1: Generating Rank 100 configurations")
    print("="*80)
    ranks_100 = [100]
    fixWb_options = [False, True]
    
    for freq1, freq2 in frequency_pairs:
        scale_factor = max(freq1, freq2) / 36.0
        num_training_samples = int(1000 * scale_factor)
        num_test_samples = int(1234 * scale_factor)
        
        for rank in ranks_100:
            for fixWb in fixWb_options:
                config = base_config.copy()
                config.update({
                    "hidden_rank": rank,
                    "batch_size": 100,  # we use default batch size
                    "num_training_samples": num_training_samples,
                    "num_test_samples": num_test_samples,
                    "function": f"cos({freq1}*pi*x^2) - 0.8*cos({freq2}*pi*x^2)",
                    "freq1": freq1,
                    "freq2": freq2,
                    "fixWb": fixWb,
                    "config_type": "rank100",  # we label this config type
                })
                configs.append(config)
    
    print(f"Generated {len([c for c in configs if c['config_type'] == 'rank100'])} rank 100 configs")
    
    # PART 2: 2x batch size runs (batch_size=200) for existing ranks
    print("\n" + "="*80)
    print("PART 2: Generating 2x batch size configurations (batch=200)")
    print("="*80)
    existing_ranks = [10, 15, 20, 25, 50]  # we use existing ranks
    
    for freq1, freq2 in frequency_pairs:
        scale_factor = max(freq1, freq2) / 36.0
        num_training_samples = int(1000 * scale_factor)
        num_test_samples = int(1234 * scale_factor)
        
        # we skip rank 50 for frequency 144 (it wasn't in original)
        ranks_to_use = existing_ranks if freq1 != 144 else [10, 15, 20, 25]
        
        for rank in ranks_to_use:
            for fixWb in fixWb_options:
                config = base_config.copy()
                config.update({
                    "hidden_rank": rank,
                    "batch_size": 200,  # we use 2x batch size
                    "num_training_samples": num_training_samples,
                    "num_test_samples": num_test_samples,
                    "function": f"cos({freq1}*pi*x^2) - 0.8*cos({freq2}*pi*x^2)",
                    "freq1": freq1,
                    "freq2": freq2,
                    "fixWb": fixWb,
                    "config_type": "batch2x",  # we label this config type
                })
                configs.append(config)
    
    print(f"Generated {len([c for c in configs if c['config_type'] == 'batch2x'])} 2x batch size configs")
    
    # PART 3: 0.5x batch size runs (batch_size=50) for existing ranks
    print("\n" + "="*80)
    print("PART 3: Generating 0.5x batch size configurations (batch=50)")
    print("="*80)
    
    for freq1, freq2 in frequency_pairs:
        scale_factor = max(freq1, freq2) / 36.0
        num_training_samples = int(1000 * scale_factor)
        num_test_samples = int(1234 * scale_factor)
        
        # we skip rank 50 for frequency 144
        ranks_to_use = existing_ranks if freq1 != 144 else [10, 15, 20, 25]
        
        for rank in ranks_to_use:
            for fixWb in fixWb_options:
                config = base_config.copy()
                config.update({
                    "hidden_rank": rank,
                    "batch_size": 50,  # we use 0.5x batch size
                    "num_training_samples": num_training_samples,
                    "num_test_samples": num_test_samples,
                    "function": f"cos({freq1}*pi*x^2) - 0.8*cos({freq2}*pi*x^2)",
                    "freq1": freq1,
                    "freq2": freq2,
                    "fixWb": fixWb,
                    "config_type": "batch0_5x",  # we label this config type
                })
                configs.append(config)
    
    print(f"Generated {len([c for c in configs if c['config_type'] == 'batch0_5x'])} 0.5x batch size configs")
    
    print("\n" + "="*80)
    print(f"TOTAL CONFIGURATIONS: {len(configs)}")
    print("="*80)
    print(f"  - Rank 100: {len([c for c in configs if c['config_type'] == 'rank100'])}")
    print(f"  - 2x batch (200): {len([c for c in configs if c['config_type'] == 'batch2x'])}")
    print(f"  - 0.5x batch (50): {len([c for c in configs if c['config_type'] == 'batch0_5x'])}")
    print("="*80)
    
    return configs


def get_output_dir_name(config):
    """we generate output directory name based on config"""
    freq_str = f"freq{config['freq1']}_{config['freq2']}"
    rank = config['hidden_rank']
    fixWb_str = "fixWbTrue" if config['fixWb'] else "fixWbFalse"
    batch_size = config['batch_size']
    
    # we create descriptive name
    if config['config_type'] == 'rank100':
        name = f"{freq_str}_rank{rank}_{fixWb_str}_batch{batch_size}"
    elif config['config_type'] == 'batch2x':
        name = f"{freq_str}_rank{rank}_{fixWb_str}_batch{batch_size}"
    elif config['config_type'] == 'batch0_5x':
        name = f"{freq_str}_rank{rank}_{fixWb_str}_batch{batch_size}"
    else:
        name = f"{freq_str}_rank{rank}_{fixWb_str}_batch{batch_size}"
    
    return name


def main():
    """we run comprehensive frequency benchmark"""
    print("="*80)
    print("COMPREHENSIVE FREQUENCY BENCHMARK")
    print("="*80)
    print("This will run:")
    print("  1. Rank 100 configurations (6 runs)")
    print("  2. 2x batch size configurations (28 runs)")
    print("  3. 0.5x batch size configurations (28 runs)")
    print("  TOTAL: 62 runs, each for 10,000 epochs")
    print("="*80)
    
    # we generate all configs
    configs = generate_comprehensive_configs()
    
    # we set base output directory
    base_output_dir = Path("experiments/table/results_frequency_benchmark_comprehensive")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {base_output_dir}")
    print(f"Total configurations: {len(configs)}\n")
    
    # we run each configuration
    for i, config in enumerate(configs, 1):
        config_type = config['config_type']
        output_dir_name = get_output_dir_name(config)
        output_dir = base_output_dir / output_dir_name
        
        print(f"\n{'='*80}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"Type: {config_type}")
        print(f"Rank: {config['hidden_rank']}, fixWb: {config['fixWb']}, batch: {config['batch_size']}")
        print(f"Frequency: ({config['freq1']}, {config['freq2']})")
        print(f"Output: {output_dir}")
        print(f"{'='*80}")
        
        # we check if already completed
        checkpoint_file = output_dir / "checkpoint.pth"
        if checkpoint_file.exists():
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            epoch = ckpt.get('epoch', 0)
            if epoch >= 10000:
                print(f"✓ Already completed (epoch {epoch}/10000), skipping...")
                continue
        
        # we create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # we train
        try:
            results = train_one_config(config, output_dir)
            print(f"✓ Completed: {output_dir_name}")
        except Exception as e:
            print(f"✗ Error in {output_dir_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
