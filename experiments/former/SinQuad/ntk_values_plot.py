# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def load_and_plot_ntk_eigenvalues(results_folder):
    """we load all .pt files and create matshow plots of ntk eigenvalues"""
    
    pt_files = glob.glob(os.path.join(results_folder, "*.pt"))
    
    if len(pt_files) == 0:
        print(f"no .pt files found in {results_folder}")
        return
    
    print(f"found {len(pt_files)} result files")
    
    for pt_file in pt_files:
        print(f"\nprocessing: {os.path.basename(pt_file)}")
        
        try:
            results = torch.load(pt_file)
            
            config_name = results["config_name"]
            ntk_eigenvalues = results["ntk_eigenvalues"]
            
            if len(ntk_eigenvalues) == 0:
                print(f"no ntk eigenvalues stored for {config_name}")
                continue
            
            epochs_list = sorted(ntk_eigenvalues.keys())
            n_epochs = len(epochs_list)
            n_eigenvalues = ntk_eigenvalues[epochs_list[0]].shape[0]
            
            eigenvalues_matrix = np.zeros((n_epochs, n_eigenvalues))
            
            for i, epoch in enumerate(epochs_list):
                eigenvalues_matrix[i, :] = ntk_eigenvalues[epoch].numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            im1 = axes[0].matshow(eigenvalues_matrix.T, aspect='auto', cmap='viridis')
            axes[0].set_xlabel('epoch index', fontsize=12)
            axes[0].set_ylabel('eigenvalue index', fontsize=12)
            axes[0].set_title(f'ntk eigenvalues matrix - {config_name}', fontsize=14)
            axes[0].set_xticks(np.arange(0, n_epochs, max(1, n_epochs//10)))
            axes[0].set_xticklabels([epochs_list[i] for i in np.arange(0, n_epochs, max(1, n_epochs//10))])
            cbar1 = plt.colorbar(im1, ax=axes[0])
            cbar1.set_label('eigenvalue magnitude', fontsize=10)
            
            log_eigenvalues_matrix = np.log10(eigenvalues_matrix + 1e-15)
            im2 = axes[1].matshow(log_eigenvalues_matrix.T, aspect='auto', cmap='viridis')
            axes[1].set_xlabel('epoch index', fontsize=12)
            axes[1].set_ylabel('eigenvalue index', fontsize=12)
            axes[1].set_title(f'log10(ntk eigenvalues) - {config_name}', fontsize=14)
            axes[1].set_xticks(np.arange(0, n_epochs, max(1, n_epochs//10)))
            axes[1].set_xticklabels([epochs_list[i] for i in np.arange(0, n_epochs, max(1, n_epochs//10))])
            cbar2 = plt.colorbar(im2, ax=axes[1])
            cbar2.set_label('log10(eigenvalue)', fontsize=10)
            
            plt.tight_layout()
            
            plot_path = os.path.join(results_folder, f"{config_name}_eigenvalues_matshow.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"saved matshow to: {plot_path}")
            
            max_eigs = eigenvalues_matrix[:, -1]
            min_eigs = eigenvalues_matrix[:, 0]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            axes[0].plot(epochs_list, max_eigs, 'b-', linewidth=2, label='max eigenvalue')
            axes[0].plot(epochs_list, min_eigs, 'r-', linewidth=2, label='min eigenvalue')
            axes[0].set_xlabel('epoch', fontsize=12)
            axes[0].set_ylabel('eigenvalue', fontsize=12)
            axes[0].set_title(f'extreme eigenvalues evolution - {config_name}', fontsize=14)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].semilogy(epochs_list, max_eigs, 'b-', linewidth=2, label='max eigenvalue')
            axes[1].semilogy(epochs_list, min_eigs + 1e-15, 'r-', linewidth=2, label='min eigenvalue')
            axes[1].set_xlabel('epoch', fontsize=12)
            axes[1].set_ylabel('eigenvalue (log scale)', fontsize=12)
            axes[1].set_title(f'extreme eigenvalues evolution (log) - {config_name}', fontsize=14)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path_evolution = os.path.join(results_folder, f"{config_name}_eigenvalues_evolution.png")
            plt.savefig(plot_path_evolution, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"saved evolution plot to: {plot_path_evolution}")
            
            if n_epochs > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for i in [0, n_epochs//4, n_epochs//2, 3*n_epochs//4, n_epochs-1]:
                    if i < n_epochs:
                        ax.plot(eigenvalues_matrix[i, :], label=f'epoch {epochs_list[i]}', linewidth=2)
                
                ax.set_xlabel('eigenvalue index', fontsize=12)
                ax.set_ylabel('eigenvalue magnitude', fontsize=12)
                ax.set_title(f'ntk spectrum evolution - {config_name}', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plot_path_spectrum = os.path.join(results_folder, f"{config_name}_spectrum_snapshots.png")
                plt.savefig(plot_path_spectrum, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"saved spectrum snapshots to: {plot_path_spectrum}")
            
        except Exception as e:
            print(f"error processing {pt_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    """we load and plot all results from the latest results folder"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    storage_dir = os.path.join(script_dir, "../../data/storage/mmnn_ntk_values")
    storage_dir = os.path.normpath(storage_dir)
    
    if not os.path.exists(storage_dir):
        print(f"storage directory not found: {storage_dir}")
        return
    
    results_folders = glob.glob(os.path.join(storage_dir, "results_ntk_*"))
    
    if len(results_folders) == 0:
        print(f"no results folders found in {storage_dir}")
        return
    
    results_folders.sort()
    latest_folder = results_folders[-1]
    
    print(f"processing latest results folder: {latest_folder}")
    print(f"found {len(results_folders)} total results folders")
    
    choice = input(f"\nprocess latest folder ({os.path.basename(latest_folder)})? [y/n/all]: ").lower()
    
    if choice == 'all':
        for folder in results_folders:
            print(f"\n{'='*80}")
            print(f"processing folder: {os.path.basename(folder)}")
            print(f"{'='*80}")
            load_and_plot_ntk_eigenvalues(folder)
    elif choice == 'y' or choice == '':
        load_and_plot_ntk_eigenvalues(latest_folder)
    else:
        print("cancelled")
        return
    
    print("\n" + "="*80)
    print("all plots created successfully")
    print("="*80)

if __name__ == "__main__":
    main()

