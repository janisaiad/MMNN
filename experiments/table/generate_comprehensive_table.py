#!/usr/bin/env python3
"""we generate comprehensive table of all training runs"""
import json
from pathlib import Path
from collections import defaultdict

# we find all result directories
result_dirs = {
    "1D Comprehensive": Path("experiments/table/results_1d_comprehensive"),
    "Frequency Benchmark (Low Rank)": Path("experiments/table/results_frequency_benchmark"),
    "Frequency Benchmark (Full Rank)": Path("experiments/table/results_frequency_benchmark_full_rank"),
    "Test Results": Path("test_results"),
    "Results": Path("results"),
}

all_configs = []

for experiment_name, base_dir in result_dirs.items():
    if not base_dir.exists():
        continue
    
    for config_dir in sorted(base_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        
        config_name = config_dir.name
        results_file = config_dir / "results.json"
        config_file = config_dir / "config.json"
        checkpoint_file = config_dir / "checkpoint.pth"
        model_file = config_dir / "model_parameters.pth"
        
        # we parse config
        config_data = {}
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
            except:
                pass
        
        # we get training info
        status = "NOT_STARTED"
        epochs = 0
        final_loss = None
        final_error = None
        benchmark = "unknown"
        rank = "unknown"
        fixWb = "unknown"
        width = "unknown"
        layers = "unknown"
        seed = "unknown"
        freq = "N/A"
        target_epochs = 8000
        
        # we extract from config
        if config_data:
            benchmark = config_data.get("benchmark_name", config_data.get("benchmark", "unknown"))
            rank = config_data.get("rank", config_data.get("hidden_rank", "unknown"))
            fixWb = config_data.get("fixWb", "unknown")
            width = config_data.get("hidden_width", config_data.get("width", "unknown"))
            layers = config_data.get("num_layers", config_data.get("layers", "unknown"))
            seed = config_data.get("seed", "unknown")
            target_epochs = config_data.get("num_epochs", 8000)
            if "freq1" in config_data:
                freq = f"({config_data['freq1']},{config_data['freq2']})"
        
        # we try to parse from folder name
        if benchmark == "unknown":
            if "flowbench" in config_name.lower():
                benchmark = "flowbench"
            elif "pinnacle" in config_name.lower():
                benchmark = "pinnacle"
            elif "freq" in config_name.lower():
                benchmark = "frequency"
                # we extract frequency
                parts = config_name.split("_")
                for i, p in enumerate(parts):
                    if p.startswith("freq") and i+1 < len(parts):
                        try:
                            f1 = int(p.replace("freq", ""))
                            f2 = int(parts[i+1])
                            freq = f"({f1},{f2})"
                        except:
                            pass
        
        if rank == "unknown":
            if "rank" in config_name:
                for part in config_name.split("_"):
                    if part.startswith("rank"):
                        try:
                            rank = int(part.replace("rank", ""))
                        except:
                            pass
                    elif "FULL_RANK" in part or "FULL_MLP" in part:
                        rank = "FULL"
                        if "rank777" in config_name:
                            rank = 777
                        elif "rank1024" in config_name:
                            rank = 1024
        
        if fixWb == "unknown":
            if "fixWbTrue" in config_name:
                fixWb = True
            elif "fixWbFalse" in config_name:
                fixWb = False
        
        if seed == "unknown":
            if "seed" in config_name:
                for part in config_name.split("_"):
                    if part.startswith("seed"):
                        try:
                            seed = int(part.replace("seed", ""))
                        except:
                            pass
        
        # we check training status
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    losses = data.get("losses", [])
                    errors_test = data.get("errors_test", [])
                    
                    if losses:
                        epochs = len(losses)
                        final_loss = losses[-1] if losses else None
                    if errors_test:
                        final_error = errors_test[-1] if errors_test else None
                    
                    # we check target epochs
                    if epochs >= target_epochs:
                        status = "✓ COMPLETED"
                    elif epochs > 0:
                        status = f"IN_PROGRESS ({epochs}/{target_epochs})"
                    else:
                        status = "FINISHED"
            except:
                status = "ERROR"
        elif checkpoint_file.exists():
            try:
                import torch
                ckpt = torch.load(checkpoint_file, map_location='cpu')
                if 'epoch' in ckpt:
                    epochs = ckpt['epoch']
                    status = f"IN_PROGRESS ({epochs}/{target_epochs})"
                else:
                    status = "IN_PROGRESS"
            except:
                status = "IN_PROGRESS"
        elif model_file.exists():
            status = "FINISHED"
        
        all_configs.append({
            "experiment": experiment_name,
            "benchmark": benchmark,
            "frequency": freq,
            "rank": rank,
            "width": width,
            "layers": layers,
            "fixWb": fixWb,
            "seed": seed,
            "status": status,
            "epochs": epochs,
            "target_epochs": target_epochs,
            "final_loss": final_loss,
            "final_error": final_error,
            "config_name": config_name
        })

# we save to JSON
output_file = Path("comprehensive_training_status.json")
with open(output_file, "w") as f:
    json.dump(all_configs, f, indent=2, default=str)
print(f"✓ Saved detailed results to: {output_file}")

# we print concise table
print("\n" + "="*120)
print("COMPREHENSIVE TRAINING STATUS - ALL EXPERIMENTS")
print("="*120)
print(f"{'Experiment':<28} {'Benchmark':<12} {'Freq':<10} {'Rank':<6} {'Width':<6} {'Layers':<6} {'fixWb':<6} {'Seed':<6} {'Status':<22} {'Epochs':<10}")
print("-"*120)

for c in sorted(all_configs, key=lambda x: (x['experiment'], x['benchmark'], x['rank'] if isinstance(x['rank'], int) else 9999, x['fixWb'], x['seed'] if isinstance(x['seed'], int) else 9999)):
    rank_str = str(c['rank']) if c['rank'] != "unknown" else "N/A"
    width_str = str(c['width']) if c['width'] != "unknown" else "N/A"
    layers_str = str(c['layers']) if c['layers'] != "unknown" else "N/A"
    fixWb_str = str(c['fixWb']) if c['fixWb'] != "unknown" else "N/A"
    seed_str = str(c['seed']) if c['seed'] != "unknown" else "N/A"
    epoch_str = f"{c['epochs']}/{c['target_epochs']}" if c['epochs'] > 0 else "0"
    
    print(f"{c['experiment']:<28} {c['benchmark']:<12} {c['frequency']:<10} {rank_str:<6} {width_str:<6} {layers_str:<6} {fixWb_str:<6} {seed_str:<6} {c['status']:<22} {epoch_str:<10}")

print("="*120)

# we print summary by experiment
print("\n" + "="*80)
print("SUMMARY BY EXPERIMENT")
print("="*80)

for exp_name in sorted(set(c['experiment'] for c in all_configs)):
    exp_configs = [c for c in all_configs if c['experiment'] == exp_name]
    completed = sum(1 for c in exp_configs if "COMPLETED" in c['status'])
    in_progress = sum(1 for c in exp_configs if "IN_PROGRESS" in c['status'])
    finished = sum(1 for c in exp_configs if "FINISHED" in c['status'] and "COMPLETED" not in c['status'])
    not_started = sum(1 for c in exp_configs if "NOT_STARTED" in c['status'])
    
    print(f"\n{exp_name}:")
    print(f"  Total: {len(exp_configs)}")
    print(f"  ✓ Completed: {completed}")
    print(f"  → In progress: {in_progress}")
    print(f"  → Finished: {finished}")
    print(f"  → Not started: {not_started}")

print("\n" + "="*80)
print(f"GRAND TOTAL: {len(all_configs)} configurations")
print("="*80)
