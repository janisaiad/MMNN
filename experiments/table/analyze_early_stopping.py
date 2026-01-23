#!/usr/bin/env python3
"""we analyze why some frequency benchmark runs stopped early"""
import json
import torch
from pathlib import Path

base_dir = Path("experiments/table/results_frequency_benchmark")

print("="*100)
print("ANALYZING EARLY STOPPING IN FREQUENCY BENCHMARK")
print("="*100)
print(f"{'Config':<50} {'Target':<8} {'Actual':<8} {'Checkpoint':<10} {'Early Stop':<12} {'Final Loss':<12}")
print("-"*100)

early_stopped = []
completed = []
incomplete = []

for config_dir in sorted(base_dir.iterdir()):
    if not config_dir.is_dir():
        continue
    
    config_name = config_dir.name
    results_file = config_dir / "results.json"
    checkpoint_file = config_dir / "checkpoint.pth"
    config_file = config_dir / "config.json"
    
    # we parse config name
    parts = config_name.split("_")
    freq1 = parts[0].replace("freq", "")
    freq2 = parts[1]
    rank = "unknown"
    fixWb = "unknown"
    
    for part in parts:
        if part.startswith("rank"):
            rank = part.replace("rank", "")
        elif "fixWbTrue" in part:
            fixWb = "True"
        elif "fixWbFalse" in part:
            fixWb = "False"
    
    target_epochs = 10000
    actual_epochs = 0
    checkpoint_epoch = 0
    early_stop = False
    final_loss = None
    
    # we get target from config
    if config_file.exists():
        try:
            with open(config_file) as f:
                cfg = json.load(f)
                target_epochs = cfg.get("num_epochs", 10000)
        except:
            pass
    
    # we check results.json
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
                losses = data.get("losses", [])
                actual_epochs = len(losses)
                if losses:
                    final_loss = losses[-1]
        except:
            pass
    
    # we check checkpoint
    if checkpoint_file.exists():
        try:
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            checkpoint_epoch = ckpt.get("epoch", 0)
        except:
            pass
    
    # we determine if early stopped
    if actual_epochs > 0 and actual_epochs < target_epochs:
        if final_loss is not None and final_loss < 5e-4:
            early_stop = True
        incomplete.append({
            "name": config_name,
            "target": target_epochs,
            "actual": actual_epochs,
            "checkpoint": checkpoint_epoch,
            "early_stop": early_stop,
            "final_loss": final_loss,
            "rank": rank,
            "fixWb": fixWb
        })
    elif actual_epochs >= target_epochs:
        completed.append(config_name)
    
    if actual_epochs > 0:
        early_stop_str = "YES" if early_stop else "NO"
        loss_str = f"{final_loss:.4e}" if final_loss else "N/A"
        print(f"{config_name:<50} {target_epochs:<8} {actual_epochs:<8} {checkpoint_epoch:<10} {early_stop_str:<12} {loss_str:<12}")

print("="*100)
print(f"\nSUMMARY:")
print(f"  Completed (>=10000 epochs): {len(completed)}")
print(f"  Incomplete (<10000 epochs): {len(incomplete)}")
print(f"  Early stopped (loss < 5e-4): {sum(1 for x in incomplete if x['early_stop'])}")

print(f"\n{'='*100}")
print("INCOMPLETE RUNS ANALYSIS (by fixWb and rank):")
print("="*100)
print(f"{'Rank':<8} {'fixWb':<8} {'Count':<8} {'Avg Epochs':<12} {'Early Stop':<12}")
print("-"*100)

# we group by rank and fixWb
from collections import defaultdict
groups = defaultdict(list)

for item in incomplete:
    key = (item['rank'], item['fixWb'])
    groups[key].append(item)

for (rank, fixWb), items in sorted(groups.items()):
    avg_epochs = sum(x['actual'] for x in items) / len(items)
    early_stop_count = sum(1 for x in items if x['early_stop'])
    print(f"{rank:<8} {fixWb:<8} {len(items):<8} {avg_epochs:<12.0f} {early_stop_count}/{len(items)}")

print(f"\n{'='*100}")
print("DETAILED BREAKDOWN OF INCOMPLETE RUNS:")
print("="*100)

for item in sorted(incomplete, key=lambda x: (x['rank'], x['fixWb'], x['actual'])):
    reason = "Early stop (loss < 5e-4)" if item['early_stop'] else f"Stopped at {item['actual']} epochs (unknown reason)"
    print(f"\n{item['name']}:")
    print(f"  Target: {item['target']} epochs")
    print(f"  Actual: {item['actual']} epochs")
    print(f"  Checkpoint: {item['checkpoint']} epochs")
    print(f"  Final loss: {item['final_loss']:.6e}" if item['final_loss'] else "  Final loss: N/A")
    print(f"  Reason: {reason}")
