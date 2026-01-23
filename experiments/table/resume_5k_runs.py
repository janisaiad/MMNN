#!/usr/bin/env python3
"""we resume runs that stopped at ~5k epochs and continue to 10k"""
import json
import torch
from pathlib import Path
import sys

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN
from test_frequency_benchmark import train_one_config

# we load LOW RANK runs to resume (NOT full rank)
runs_to_resume = []
if Path("runs_to_resume_low_rank.json").exists():
    with open("runs_to_resume_low_rank.json") as f:
        runs_to_resume.extend(json.load(f))
print(f"Loaded {len(runs_to_resume)} LOW RANK runs to resume")

print("="*80)
print(f"RESUMING {len(runs_to_resume)} RUNS FROM ~5K TO 10K EPOCHS")
print("="*80)

for run_info in runs_to_resume:
    config_dir = Path(run_info["path"])
    config_file = config_dir / "config.json"
    checkpoint_file = config_dir / "checkpoint.pth"
    
    print(f"\n{'='*80}")
    print(f"Resuming: {run_info['name']}")
    print(f"Current: {run_info['current_epoch']} epochs")
    print(f"Target: {run_info['target_epochs']} epochs")
    print(f"{'='*80}")
    
    # we load config and update num_epochs to 10000
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        print(f"⚠️  Config file not found, skipping...")
        continue
    
    # we ensure target is 10000
    config["num_epochs"] = 10000
    
    # we save updated config
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Updated config: num_epochs = 10000")
    print(f"✓ Checkpoint exists: {checkpoint_file.exists()}")
    print(f"  Training will resume from epoch {run_info['current_epoch'] + 1}")
    
    # we call train_one_config which will automatically resume from checkpoint
    try:
        results = train_one_config(config, config_dir)
        print(f"✓ Completed: {run_info['name']}")
    except Exception as e:
        print(f"✗ Error resuming {run_info['name']}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print("RESUME COMPLETE")
print(f"{'='*80}")
