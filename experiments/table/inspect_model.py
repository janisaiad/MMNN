#!/usr/bin/env python3
"""
Script to inspect MMNN model architecture from saved results.
Can reconstruct model from config.json and show detailed layer information.
"""

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import MMNN
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.table.mmnn_vs import MMNN


def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def inspect_model_from_config(config_path, device='cpu'):
    """
    Reconstruct and inspect model from config.json file
    
    Args:
        config_path: Path to config.json file OR directory containing config.json
        device: Device to load model on ('cpu' or 'cuda')
    """
    config_path = Path(config_path)
    
    # If directory provided, look for config.json inside
    if config_path.is_dir():
        config_path = config_path / "config.json"
    
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        print(f"   (Tried: {config_path})")
        return None
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("MODEL INSPECTION")
    print("="*80)
    print(f"\n📁 Config file: {config_path}")
    print(f"📁 Results directory: {config_path.parent}")
    
    # Extract model parameters from config
    num_layers = config.get('num_layers', 2)
    hidden_rank = config.get('hidden_rank', 15)
    hidden_width = config.get('hidden_width', 1024)
    input_rank = 1
    output_rank = 1
    fixWb = config.get('parameterization', 'NTK') == 'NTK'
    
    # Reconstruct ranks and widths
    ranks = [input_rank] + [hidden_rank] * num_layers + [output_rank]
    widths = [hidden_width] * (num_layers + 1)
    
    print(f"\n📊 Model Configuration:")
    print(f"   Parameterization: {config.get('parameterization', 'NTK')} (fixWb={fixWb})")
    print(f"   Number of layers (L): {num_layers}")
    print(f"   Hidden rank: {hidden_rank}")
    print(f"   Hidden width: {hidden_width}")
    print(f"   Ranks: {ranks}")
    print(f"   Widths: {widths}")
    
    # Create model
    device_obj = torch.device(device)
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device_obj,
        ResNet=False,
        fixWb=fixWb
    )
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Device: {device_obj}")
    print(f"   Total depth (blocks): {model.depth}")
    print(f"   Total Linear layers: {len(model.fcs)}")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📈 Parameter Counts:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Detailed layer information
    print(f"\n🔍 Layer Details:")
    print(f"{'Layer':<8} {'Type':<15} {'Input Dim':<12} {'Output Dim':<12} {'Params':<12} {'Trainable':<10}")
    print("-" * 80)
    
    layer_idx = 0
    for block_idx in range(model.depth):
        # rank → width layer (even index)
        fc_rank_to_width = model.fcs[2 * block_idx]
        in_dim = fc_rank_to_width.in_features
        out_dim = fc_rank_to_width.out_features
        params = fc_rank_to_width.weight.numel() + fc_rank_to_width.bias.numel()
        trainable = any(p.requires_grad for p in fc_rank_to_width.parameters())
        
        print(f"L{layer_idx:<7} {'rank→width':<15} {in_dim:<12} {out_dim:<12} {params:<12} {str(trainable):<10}")
        layer_idx += 1
        
        # width → rank layer (odd index)
        fc_width_to_rank = model.fcs[2 * block_idx + 1]
        in_dim = fc_width_to_rank.in_features
        out_dim = fc_width_to_rank.out_features
        params = fc_width_to_rank.weight.numel() + fc_width_to_rank.bias.numel()
        trainable = any(p.requires_grad for p in fc_width_to_rank.parameters())
        
        print(f"L{layer_idx:<7} {'width→rank':<15} {in_dim:<12} {out_dim:<12} {params:<12} {str(trainable):<10}")
        layer_idx += 1
    
    # Show weight statistics
    print(f"\n📊 Weight Statistics:")
    for block_idx in range(model.depth):
        fc_rank_to_width = model.fcs[2 * block_idx]
        fc_width_to_rank = model.fcs[2 * block_idx + 1]
        
        w1 = fc_rank_to_width.weight.data.cpu().numpy()
        w2 = fc_width_to_rank.weight.data.cpu().numpy()
        
        print(f"\n   Block {block_idx + 1}:")
        print(f"      rank→width weights: shape={w1.shape}, mean={w1.mean():.6f}, std={w1.std():.6f}, "
              f"min={w1.min():.6f}, max={w1.max():.6f}")
        print(f"      width→rank weights: shape={w2.shape}, mean={w2.mean():.6f}, std={w2.std():.6f}, "
              f"min={w2.min():.6f}, max={w2.max():.6f}")
    
    # Load training results if available
    results_path = config_path.parent / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print(f"\n📈 Training Results:")
        print(f"   Factor: {results.get('factor', 'N/A')}")
        print(f"   Min loss: {results.get('min_loss', 'N/A'):.6e} at epoch {results.get('min_loss_epoch', 'N/A')}")
        print(f"   Final loss: {results.get('final_loss', 'N/A'):.6e}")
        print(f"   Final optimizer: {results.get('optimizer_type', 'N/A')}")
        print(f"   Training time: {results.get('training_time_seconds', 0):.2f} seconds")
    
    print("\n" + "="*80)
    
    return model


def inspect_model_direct(model, config_info=None):
    """
    Inspect a model object directly (if already loaded)
    
    Args:
        model: MMNN model instance
        config_info: Optional dict with config info for display
    """
    print("="*80)
    print("MODEL INSPECTION (Direct)")
    print("="*80)
    
    if config_info:
        print(f"\n📊 Configuration Info:")
        for key, value in config_info.items():
            print(f"   {key}: {value}")
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Ranks: {model.ranks}")
    print(f"   Widths: {model.widths}")
    print(f"   Depth (blocks): {model.depth}")
    print(f"   ResNet: {model.ResNet}")
    print(f"   fixWb: {any(not p.requires_grad for p in model.fcs[0].parameters())}")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n📈 Parameter Counts:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Detailed layer information
    print(f"\n🔍 Layer Details:")
    print(f"{'Layer':<8} {'Type':<15} {'Input Dim':<12} {'Output Dim':<12} {'Params':<12} {'Trainable':<10}")
    print("-" * 80)
    
    layer_idx = 0
    for block_idx in range(model.depth):
        # rank → width layer
        fc_rank_to_width = model.fcs[2 * block_idx]
        in_dim = fc_rank_to_width.in_features
        out_dim = fc_rank_to_width.out_features
        params = fc_rank_to_width.weight.numel() + fc_rank_to_width.bias.numel()
        trainable = any(p.requires_grad for p in fc_rank_to_width.parameters())
        
        print(f"L{layer_idx:<7} {'rank→width':<15} {in_dim:<12} {out_dim:<12} {params:<12} {str(trainable):<10}")
        layer_idx += 1
        
        # width → rank layer
        fc_width_to_rank = model.fcs[2 * block_idx + 1]
        in_dim = fc_width_to_rank.in_features
        out_dim = fc_width_to_rank.out_features
        params = fc_width_to_rank.weight.numel() + fc_width_to_rank.bias.numel()
        trainable = any(p.requires_grad for p in fc_width_to_rank.parameters())
        
        print(f"L{layer_idx:<7} {'width→rank':<15} {in_dim:<12} {out_dim:<12} {params:<12} {str(trainable):<10}")
        layer_idx += 1
    
    print("\n" + "="*80)
    
    return model


def main():
    """Main function - can be called from command line"""
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_config.json_or_directory> [device]")
        print("\nExamples:")
        print("  # Using config.json file:")
        print("  python inspect_model.py results_tune_lr_decay_L2/factor4_rank15_Adam_.../config.json")
        print("  # Using directory (will look for config.json inside):")
        print("  python inspect_model.py results_tune_lr_decay_L2/factor4_rank15_Adam_.../")
        print("  # With CUDA:")
        print("  python inspect_model.py results_tune_lr_decay_L2/factor4_rank15_Adam_.../ cuda")
        sys.exit(1)
    
    config_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
    
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        device = 'cpu'
    
    inspect_model_from_config(config_path, device)


if __name__ == "__main__":
    main()
