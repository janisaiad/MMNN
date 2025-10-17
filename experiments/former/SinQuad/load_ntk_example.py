"""
example script to load and analyze saved ntk matrices and model parameters

this script shows how to load ntk matrices and parameters saved during training
and perform basic analysis on them
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# we specify the experiment folder
experiment_folder = "figures/mmnn_training/largescaletraining/mmnn_L6_W666_R15_E1500_lr0.001_bs100_ntr1000"

print("="*60)
print("LOADING NTK MATRICES")
print("="*60)

# we load the ntk matrices
ntk_data = np.load(os.path.join(experiment_folder, "ntk_matrices.npz"))

# we get all stored epochs
all_epochs = sorted([int(key.split('_')[1]) for key in ntk_data.keys()])
print(f"NTK matrices available at epochs: {all_epochs}")

# we load a specific ntk matrix
epoch = all_epochs[0]
ntk_matrix = ntk_data[f'epoch_{epoch}']
print(f"\nNTK matrix at epoch {epoch}:")
print(f"Shape: {ntk_matrix.shape}")
print(f"Min value: {ntk_matrix.min():.3e}")
print(f"Max value: {ntk_matrix.max():.3e}")
print(f"Mean value: {ntk_matrix.mean():.3e}")

# we compute eigenvalues
eigenvalues = np.linalg.eigvalsh(ntk_matrix)
print(f"\nEigenvalues:")
print(f"Min: {eigenvalues[0]:.3e}")
print(f"Max: {eigenvalues[-1]:.3e}")
print(f"Number of negative eigenvalues: {(eigenvalues < 0).sum()}")
print(f"Number of positive eigenvalues: {(eigenvalues > 0).sum()}")

# we plot the ntk matrix as a heatmap
fig, axes = plt.subplots(1, len(all_epochs), figsize=(5*len(all_epochs), 4))
if len(all_epochs) == 1:
    axes = [axes]

for idx, epoch in enumerate(all_epochs):
    ntk = ntk_data[f'epoch_{epoch}']
    im = axes[idx].imshow(ntk, cmap='RdBu_r', aspect='auto')
    axes[idx].set_title(f'NTK at epoch {epoch}')
    axes[idx].set_xlabel('Training sample i')
    axes[idx].set_ylabel('Training sample j')
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig(os.path.join(experiment_folder, 'ntk_matrices_heatmaps.png'), dpi=100)
print(f"\nHeatmap saved to {os.path.join(experiment_folder, 'ntk_matrices_heatmaps.png')}")

# we compare ntk evolution
if len(all_epochs) >= 2:
    print(f"\nNTK evolution from epoch {all_epochs[0]} to {all_epochs[-1]}:")
    ntk_0 = ntk_data[f'epoch_{all_epochs[0]}']
    ntk_final = ntk_data[f'epoch_{all_epochs[-1]}']
    
    diff = ntk_final - ntk_0
    print(f"Matrix difference norm (Frobenius): {np.linalg.norm(diff):.3e}")
    print(f"Relative change: {np.linalg.norm(diff) / np.linalg.norm(ntk_0) * 100:.2f}%")

print("\n" + "="*60)
print("LOADING MODEL PARAMETERS")
print("="*60)

# we load the parameter evolution
params_data = np.load(os.path.join(experiment_folder, "parameters_evolution.npz"))

# we get all stored epochs for parameters
params_epochs = sorted([int(key.split('_')[1]) for key in params_data.keys()])
print(f"Parameters available at epochs: {params_epochs}")

# we load parameters at a specific epoch
epoch = params_epochs[0]
params_vector = params_data[f'epoch_{epoch}']
print(f"\nParameters at epoch {epoch}:")
print(f"Total number of parameters: {len(params_vector)}")
print(f"Parameter mean: {params_vector.mean():.6e}")
print(f"Parameter std: {params_vector.std():.6e}")
print(f"Parameter min: {params_vector.min():.6e}")
print(f"Parameter max: {params_vector.max():.6e}")

# we compute parameter evolution
if len(params_epochs) >= 2:
    print(f"\nParameter evolution from epoch {params_epochs[0]} to {params_epochs[-1]}:")
    params_0 = params_data[f'epoch_{params_epochs[0]}']
    params_final = params_data[f'epoch_{params_epochs[-1]}']
    
    # we compute distance traveled in parameter space
    param_diff = params_final - params_0
    distance = np.linalg.norm(param_diff)
    print(f"L2 distance in parameter space: {distance:.3e}")
    print(f"Relative change: {distance / np.linalg.norm(params_0) * 100:.2f}%")
    
    # we plot parameter evolution
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # we plot parameter norms over time
    param_norms = [np.linalg.norm(params_data[f'epoch_{ep}']) for ep in params_epochs]
    axes[0].plot(params_epochs, param_norms, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Parameter L2 Norm')
    axes[0].set_title('Parameter Norm Evolution')
    axes[0].grid(True, alpha=0.3)
    
    # we plot cumulative distance traveled
    cumulative_distance = [0]
    for i in range(1, len(params_epochs)):
        prev_params = params_data[f'epoch_{params_epochs[i-1]}']
        curr_params = params_data[f'epoch_{params_epochs[i]}']
        dist = np.linalg.norm(curr_params - prev_params)
        cumulative_distance.append(cumulative_distance[-1] + dist)
    
    axes[1].plot(params_epochs, cumulative_distance, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cumulative Distance')
    axes[1].set_title('Cumulative Distance Traveled in Parameter Space')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_folder, 'parameter_evolution.png'), dpi=100)
    print(f"\nParameter evolution plot saved to {os.path.join(experiment_folder, 'parameter_evolution.png')}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

