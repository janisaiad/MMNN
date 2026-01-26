#!/usr/bin/env python3
"""
Script to clean up large JSON files by removing full matrices and keeping only statistics.
Saves matrices to .npy files instead.
"""
import json
import numpy as np
from pathlib import Path

results_base = Path("/Data/janis.aiad/MMNN/experiments/table/experiments/table/results_tune_lr_decay_L2")

# we find all layer2_logratios_x0.json files
logratio_files = list(results_base.glob("**/layer2_logratios_x0.json"))

print(f"Found {len(logratio_files)} log ratio JSON files to clean up")

for logratio_file in logratio_files:
    config_dir = logratio_file.parent
    print(f"\nProcessing: {config_dir.name}")
    
    try:
        with open(logratio_file, 'r') as f:
            data = json.load(f)
        
        # we check if it has large matrices
        if 'log_ratio_matrix' in data:
            R = np.array(data['log_ratio_matrix'])
            f_k = np.array(data.get('f_k_values', []))
            
            # we compute statistics
            R_clean = R[np.isfinite(R)]
            R_positive = R_clean[R_clean > 0]
            
            # we save matrices to .npy files
            matrix_file = config_dir / 'layer2_logratio_matrix_x0.npy'
            np.save(matrix_file, R)
            print(f"   ✅ Saved matrix to {matrix_file.name}")
            
            if len(f_k) > 0:
                fk_file = config_dir / 'layer2_fk_values_x0.npy'
                np.save(fk_file, f_k)
                print(f"   ✅ Saved f_k values to {fk_file.name}")
            
            # we replace with statistics only
            data_clean = {
                'x_location': data.get('x_location', 0.0),
                'layer': data.get('layer', 2),
                'rank': data.get('rank', R.shape[0]),
                'matrix_file': 'layer2_logratio_matrix_x0.npy',
                'fk_file': 'layer2_fk_values_x0.npy' if len(f_k) > 0 else None,
                'statistics': {
                    'mean': float(np.mean(R_clean)) if len(R_clean) > 0 else None,
                    'std': float(np.std(R_clean)) if len(R_clean) > 0 else None,
                    'min': float(np.min(R_clean)) if len(R_clean) > 0 else None,
                    'max': float(np.max(R_clean)) if len(R_clean) > 0 else None,
                    'n_total': int(len(R_clean)),
                    'n_positive': int(len(R_positive)),
                    'mean_positive': float(np.mean(R_positive)) if len(R_positive) > 0 else None,
                    'std_positive': float(np.std(R_positive)) if len(R_positive) > 0 else None,
                    'min_positive': float(np.min(R_positive)) if len(R_positive) > 0 else None,
                    'max_positive': float(np.max(R_positive)) if len(R_positive) > 0 else None
                }
            }
            
            # we save cleaned JSON
            with open(logratio_file, 'w') as f:
                json.dump(data_clean, f, indent=2)
            
            print(f"   ✅ Cleaned JSON (removed {R.size} matrix values, kept statistics)")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        continue

print(f"\n✅ Done! Cleaned {len(logratio_files)} files")
