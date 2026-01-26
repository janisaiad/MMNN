#!/usr/bin/env python3
"""
Script to fix log ratio JSON files by recalculating statistics from .npy files,
filtering out NaN values properly.
"""
import json
import numpy as np
from pathlib import Path

results_base = Path("/Data/janis.aiad/MMNN/experiments/table/experiments/table/results_tune_lr_decay_L2")

# we find all layer1_layer2_logratios_all_x.json files
logratio_files = list(results_base.glob("**/layer1_layer2_logratios_all_x.json"))

print(f"Found {len(logratio_files)} log ratio JSON files to fix")

for logratio_file in logratio_files:
    config_dir = logratio_file.parent
    print(f"\nProcessing: {config_dir.name}")
    
    try:
        with open(logratio_file, 'r') as f:
            data = json.load(f)
        
        # we fix statistics for each layer and x value
        for layer_key in ['layer_1', 'layer_2']:
            if layer_key not in data['results']:
                continue
            
            layer_data = data['results'][layer_key]
            for x_key in layer_data.keys():
                x_data = layer_data[x_key]
                matrix_file = config_dir / x_data.get('matrix_file', '')
                
                if not matrix_file.exists():
                    print(f"   ⚠️  Matrix file not found: {matrix_file.name}")
                    continue
                
                # we load matrix and recalculate statistics
                R = np.load(matrix_file)
                R_clean = R[np.isfinite(R)]  # we remove NaN and Inf
                R_positive = R_clean[R_clean > 0]  # we keep only positive values
                
                # we also check f_k
                fk_file = config_dir / x_data.get('fk_file', '')
                if fk_file.exists():
                    f_k = np.load(fk_file)
                    f_k_clean = f_k[np.isfinite(f_k)]
                else:
                    f_k_clean = np.array([])
                
                # we recalculate statistics
                if len(R_clean) > 0:
                    stats = {
                        'mean': float(np.mean(R_clean)),
                        'std': float(np.std(R_clean)),
                        'min': float(np.min(R_clean)),
                        'max': float(np.max(R_clean)),
                        'n_total': int(len(R_clean)),
                        'n_positive': int(len(R_positive)),
                    }
                    if len(R_positive) > 0:
                        stats.update({
                            'mean_positive': float(np.mean(R_positive)),
                            'std_positive': float(np.std(R_positive)),
                            'min_positive': float(np.min(R_positive)),
                            'max_positive': float(np.max(R_positive))
                        })
                    else:
                        stats.update({
                            'mean_positive': None,
                            'std_positive': None,
                            'min_positive': None,
                            'max_positive': None
                        })
                else:
                    stats = {
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'n_total': 0,
                        'n_positive': 0,
                        'mean_positive': None,
                        'std_positive': None,
                        'min_positive': None,
                        'max_positive': None
                    }
                
                # we update statistics
                x_data['statistics'] = stats
                x_data['f_k_valid'] = int(len(f_k_clean))
                x_data['f_k_total'] = int(len(f_k)) if fk_file.exists() else 0
                
                print(f"   ✅ Fixed {layer_key} {x_key}: n_total={stats['n_total']}, n_positive={stats['n_positive']}")
        
        # we save fixed JSON
        with open(logratio_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   ✅ Saved fixed JSON")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n✅ Done! Fixed {len(logratio_files)} files")
