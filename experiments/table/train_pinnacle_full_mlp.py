#!/usr/bin/env python3
"""
we train PINNacle benchmark with FULL MLP (rank=width=1024)
this is NOT using low-rank structure - it's a standard fully-connected MLP
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
    train_one_config
)
import torch


def generate_full_mlp_configs():
    """we generate configurations for FULL MLP (rank=width)"""
    configs = []
    
    # we only use pinnacle benchmark
    benchmarks = ["pinnacle"]
    
    # we use FULL MLP: rank = width = 1024 (no low-rank structure)
    rank = 1024  # we set rank equal to width for full MLP
    hidden_width = 1024
    
    # we test both fixWb options
    fixWb_options = [False, True]
    
    # we use multiple seeds for robustness
    seeds = [42, 123, 456]
    
    # we set common parameters
    num_layers = 10
    n_train_samples = 2000
    n_test_samples = 500
    batch_size = 500
    num_epochs = 8000
    input_dim = 1
    output_dim = 1
    
    # we generate all combinations
    for benchmark in benchmarks:
        use_pinn = (benchmark.lower() == "pinnacle")
        
        for fixWb in fixWb_options:
            for seed in seeds:
                config = AblationConfig(
                    benchmark_name=benchmark,
                    fixWb=fixWb,
                    rank=rank,  # we use rank=1024 (FULL MLP)
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
                    seed=seed,
                )
                configs.append(config)
    
    return configs


def train_one_config_full_mlp(config: AblationConfig, output_dir: Path) -> dict:
    """we train one configuration with FULL MLP and update titles/names"""
    # we modify the output directory name to clearly indicate FULL MLP
    original_output_dir = output_dir
    parent_dir = output_dir.parent
    # we extract the base name and modify it
    base_name = output_dir.name
    # we replace rank info with FULL_MLP
    if "rank" in base_name:
        parts = base_name.split("_")
        new_parts = []
        for part in parts:
            if part.startswith("rank"):
                new_parts.append("FULL_MLP_rank1024")  # we make it very clear
            else:
                new_parts.append(part)
        new_name = "_".join(new_parts)
    else:
        new_name = base_name + "_FULL_MLP_rank1024"
    
    output_dir = parent_dir / new_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # we call the original training function
    results = train_one_config(config, output_dir)
    
    # we update plot titles and folder names in results
    if "config" in results:
        results["config"]["architecture"] = "FULL_MLP_rank1024"  # we add architecture info
        results["config"]["is_full_mlp"] = True  # we mark as full MLP
    
    return results


def main():
    """we run FULL MLP training on PINNacle"""
    print("="*80)
    print("PINNacle Training with FULL MLP (rank=width=1024)")
    print("NOT using low-rank structure - this is a standard fully-connected MLP")
    print("="*80)
    
    # we generate configurations
    configs = generate_full_mlp_configs()
    print(f"\ngenerated {len(configs)} configurations")
    print(f"  benchmark: pinnacle only")
    print(f"  architecture: FULL MLP (rank=1024, width=1024)")
    print(f"  fixWb options: False, True")
    print(f"  seeds: 3 per configuration")
    print(f"  depth: 10 layers")
    print(f"  samples: 2000 train, 500 test")
    print(f"  batch size: 500")
    print(f"  epochs: 8000")
    
    # we set output directory
    base_output_dir = Path("experiments/table/results_pinnacle_FULL_MLP")
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
            # we create output directory with FULL_MLP in the name
            config_name = (f"pinnacle_FULL_MLP_rank1024_"
                          f"fixWb{config.fixWb}_"
                          f"seed{config.seed}_run{idx}")
            config.output_dir = base_output_dir / config_name
            config.output_dir.mkdir(parents=True, exist_ok=True)
            config.device = str(device)
            
            elapsed = time.time() - start_time
            remaining_configs = len(configs) - idx
            avg_time = elapsed / (idx + 1) if idx > 0 else 0
            estimated_remaining = avg_time * remaining_configs
            
            print(f"\n{'='*80}")
            print(f"CONFIG {idx+1}/{len(configs)} - FULL MLP (rank=1024)")
            print(f"  benchmark: {config.benchmark_name}")
            print(f"  architecture: FULL MLP (rank=width=1024)")
            print(f"  fixWb: {config.fixWb}")
            print(f"  seed: {config.seed}")
            print(f"  elapsed: {elapsed/3600:.2f} hours")
            print(f"  estimated remaining: {estimated_remaining/3600:.2f} hours")
            print(f"{'='*80}")
            
            try:
                results = train_one_config(config, config.output_dir)
                # we update results to indicate FULL MLP
                results["config"]["architecture"] = "FULL_MLP_rank1024"
                results["config"]["is_full_mlp"] = True
                all_results.append(results)
                print(f"✓ Config {idx+1} completed successfully (FULL MLP)")
            except Exception as e:
                print(f"✗ Config {idx+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        
        # we save summary
        summary_path = base_output_dir / "FULL_MLP_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE - FULL MLP (rank=1024)")
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
