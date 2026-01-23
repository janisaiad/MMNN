#!/usr/bin/env python3
"""
we run comprehensive 1D training experiments
comparing fixWb (True/False) and different ranks (low rank vs full rank)
depth=10, 2000 samples, batch_size=500, lr=0.001
designed to run for ~4 hours
"""
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import (
    AblationConfig, 
    train_one_config, 
    generate_ablation_configs
)
import torch


def generate_1d_configs():
    """we generate comprehensive 1D configurations"""
    configs = []
    
    # we define 1D benchmarks - we add more to fill 4 hours
    benchmarks = ["flowbench", "pinnacle"]  # we use flowbench as generic 1D, pinnacle for PINN
    
    # we define rank options: low ranks and full rank
    rank_options = [3, 6, 10, 15, 25, 50, 1024]  # we test from very low to full rank
    
    # we test both fixWb options
    fixWb_options = [False, True]
    
    # we also test different seeds for robustness (3 seeds per config)
    seeds = [42, 123, 456]
    
    # we set common parameters
    num_layers = 10  # we use depth 10 as requested
    hidden_width = 1024
    n_train_samples = 2000  # we use 2000 training samples as requested
    n_test_samples = 500
    batch_size = 500
    num_epochs = 8000  # we use more epochs to fill ~4 hours of training
    input_dim = 1  # we focus on 1D
    output_dim = 1
    
    # we generate all combinations with multiple seeds
    for benchmark in benchmarks:
        use_pinn = (benchmark.lower() == "pinnacle")
        
        for fixWb in fixWb_options:
            for rank in rank_options:
                for seed in seeds:
                    config = AblationConfig(
                        benchmark_name=benchmark,
                        fixWb=fixWb,
                        rank=rank,
                        num_layers=num_layers,
                        hidden_width=hidden_width,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        n_train_samples=n_train_samples,
                        n_test_samples=n_test_samples,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        use_pinn=use_pinn,
                        n_collocation=1000 if use_pinn else 0,
                        n_boundary=100 if use_pinn else 0,
                        n_initial=100 if use_pinn else 0,
                        log_every=100,
                        save_every=1000,
                        seed=seed,  # we set seed for reproducibility
                    )
                    configs.append(config)
    
    return configs


def estimate_time_per_config():
    """we estimate time per configuration (rough estimate)"""
    # we estimate based on: epochs * samples / batch_size * time_per_batch
    # rough estimate: ~0.01 seconds per batch
    samples = 2000  # we updated to 2000 samples
    batch_size = 500
    epochs = 8000  # we use 8000 epochs
    batches_per_epoch = (samples + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    time_per_batch = 0.01  # we estimate 10ms per batch (will vary)
    estimated_seconds = total_batches * time_per_batch
    return estimated_seconds


def main():
    """we run comprehensive 1D training"""
    print("="*80)
    print("Comprehensive 1D Training Experiments")
    print("Comparing fixWb and different ranks")
    print("="*80)
    
    # we generate configurations
    configs = generate_1d_configs()
    print(f"\ngenerated {len(configs)} configurations")
    print(f"  benchmarks: flowbench, pinnacle")
    print(f"  fixWb options: False, True")
    print(f"  ranks: {[3, 6, 10, 15, 25, 50, 1024]}")
    print(f"  seeds: 3 per configuration (for robustness)")
    print(f"  depth: 10 layers")
    print(f"  width: 1024")
    print(f"  samples: 2000 train, 500 test")
    print(f"  batch size: 500")
    print(f"  learning rate: 0.001")
    print(f"  epochs: 5000")
    
    # we estimate total time
    time_per_config = estimate_time_per_config()
    total_estimated_hours = (time_per_config * len(configs)) / 3600
    print(f"\nestimated time per config: {time_per_config/60:.1f} minutes")
    print(f"estimated total time: {total_estimated_hours:.1f} hours")
    print(f"target: ~4 hours")
    
    if total_estimated_hours > 5:
        print(f"\n⚠ warning: estimated time ({total_estimated_hours:.1f}h) exceeds 4 hours")
        print("  consider reducing epochs or number of configs")
    
    # we set output directory
    base_output_dir = Path("experiments/table/results_1d_comprehensive")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # we setup logging
    log_file = base_output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    class Tee:
        def __init__(self, file_path):
            self.file = open(file_path, 'w')
            self.stdout = sys.stdout
        
        def write(self, text):
            self.file.write(text)
            self.file.flush()
            self.stdout.write(text)
        
        def flush(self):
            self.file.flush()
            self.stdout.flush()
    
    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nusing device: {device}")
        print(f"started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"log file: {log_file}")
        
        all_results = []
        start_time = time.time()
        
        for idx, config in enumerate(configs):
            config.output_dir = base_output_dir / f"{config.benchmark_name}_fixWb{config.fixWb}_rank{config.rank}_seed{config.seed}_run{idx}"
            config.output_dir.mkdir(parents=True, exist_ok=True)
            config.device = str(device)
            
            elapsed = time.time() - start_time
            remaining_configs = len(configs) - idx
            avg_time = elapsed / (idx + 1) if idx > 0 else time_per_config
            estimated_remaining = avg_time * remaining_configs
            
            print(f"\n{'='*80}")
            print(f"CONFIG {idx+1}/{len(configs)}")
            print(f"  benchmark: {config.benchmark_name}")
            print(f"  fixWb: {config.fixWb}")
            print(f"  rank: {config.rank}")
            print(f"  elapsed: {elapsed/3600:.2f} hours")
            print(f"  estimated remaining: {estimated_remaining/3600:.2f} hours")
            print(f"{'='*80}")
            
            try:
                results = train_one_config(config, config.output_dir)
                all_results.append(results)
                print(f"✓ Config {idx+1} completed successfully")
            except Exception as e:
                print(f"✗ Config {idx+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        
        # we save summary
        summary_path = base_output_dir / "comprehensive_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"completed: {len(all_results)}/{len(configs)} configurations")
        print(f"total time: {total_time/3600:.2f} hours")
        print(f"results saved to: {base_output_dir}")
        print(f"summary saved to: {summary_path}")
        print(f"completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stdout
        tee.file.close()


if __name__ == "__main__":
    main()
