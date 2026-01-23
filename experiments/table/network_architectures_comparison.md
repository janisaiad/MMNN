# Network Architectures Comparison: MMNN vs PDE Benchmarks

## Overview
This document compares the network architectures and sizes used in `mmnn_vs.py` with those used in the PDE benchmark datasets (PINNacle, PDEArena, PDEGym/Poseidon, PDEBench).

---

## MMNN Architecture (from mmnn_vs.py)

### Default Architecture
- **Type**: Matrix Multiplication Neural Network (MMNN) with low-rank structure
- **Default ranks**: `[1] + [16]*5 + [1]` = `[1, 16, 16, 16, 16, 16, 1]`
- **Default widths**: `[366]*6` = `[366, 366, 366, 366, 366, 366]`
- **Number of layers**: 6 hidden layers
- **Architecture**: Alternating rank→width and width→rank layers with ReLU activation
- **Total parameters**: Varies based on rank and width configuration

### Ablation Study Configurations
From `AblationConfig`:
- **Number of layers**: 6 (fixed)
- **Hidden width**: 1024 (fixed)
- **Rank options tested**: `[3, 6, 10, 15, 25, 50, None]` where `None` means rank = width (1024)
- **Input/Output dimensions**: Variable (1D, 2D, etc. depending on benchmark)
- **ResNet**: False (not used in ablation)
- **fixWb**: Tested both True and False

### Architecture Details
- Uses mu-parameterization initialization
- Weights scaled by `1/sqrt(width)` for width→rank layers
- Weights scaled by `1/sqrt(rank)` for rank→width layers
- Optional fixWb mode: freezes rank→width weights during training

---

## PINNacle Benchmark

### Default Architecture (DeepXDE-based)
- **Type**: Fully Connected Network (FNN)
- **Default hidden layers**: `"100*5"` = **5 layers of 100 neurons each**
- **Architecture**: `[input_dim, 100, 100, 100, 100, 100, output_dim]`
- **Activation**: Tanh
- **Initialization**: Glorot normal (Xavier)
- **Total parameters**: ~(input_dim × 100) + 4×(100 × 100) + (100 × output_dim) + biases

### VPINN Architecture (varies by problem)
From `default_arg.json`:
- **Burgers1D**: `[2, 15, 15, 15, 1]` = 3 hidden layers of 15 neurons
- **Burgers2D**: `[3, 30, 30, 30, 2]` = 3 hidden layers of 30 neurons
- **Poisson2D**: `[2, 15, 15, 15, 1]` = 3 hidden layers of 15 neurons
- **Poisson3D**: `[3, 20, 20, 20, 1]` = 3 hidden layers of 20 neurons
- **Heat equations**: `[3, 20, 20, 20, 1]` = 3 hidden layers of 20 neurons
- **NS equations**: `[3, 20, 20, 20, 3]` = 3 hidden layers of 20 neurons
- **Activation**: Tanh
- **Typical range**: 15-30 neurons per hidden layer, 3 hidden layers

### FBPINNs Architecture
- **Type**: Fully Connected Network (FCN)
- **Parameters**: `N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS`
- **Architecture**: `[N_INPUT, N_HIDDEN, ..., N_HIDDEN (N_LAYERS-1 times), N_OUTPUT]`
- **Activation**: Tanh
- **Typical values**: Varies by subdomain

---

## PDEArena Benchmark

### Architecture Types
PDEArena uses various architectures, not just fully connected networks:

#### FNO (Fourier Neural Operator)
- **FNO-128-8m**: hidden_channels=128, modes=[8,8], num_blocks=[1,1,1,1]
- **FNO-128-16m**: hidden_channels=128, modes=[16,16], num_blocks=[1,1,1,1]
- **FNOs-128-32m**: hidden_channels=128, modes=[32,32], num_blocks=[1,1]
- **FNOs-64-32m**: hidden_channels=64, modes=[32,32], num_blocks=[1,1]
- **FNOs-96-32m**: hidden_channels=96, modes=[32,32], num_blocks=[1,1]

#### UNet Variants
- **Unet2015-64**: hidden_channels=64
- **Unet2015-128**: hidden_channels=128
- **Unetbase-64**: hidden_channels=64
- **Unetbase-128**: hidden_channels=128
- **Unetmod-64**: hidden_channels=64, norm=True
- **FourierUnet**: hidden_channels=64, modes=[16,16], n_fourier_layers=2-3

#### ResNet Variants
- **ResNet-128**: hidden_channels=128, num_blocks=[1,1,1,1]
- **ResNet-256**: hidden_channels=256, num_blocks=[1,1,1,1]
- **DilResNet-128**: hidden_channels=128, dilated blocks

#### UNO (U-Net Operator)
- **UNO-64**: hidden_channels=64
- **UNO-128**: hidden_channels=128

**Note**: PDEArena architectures are convolutional/operator-based, not fully connected, making direct parameter count comparison difficult.

---

## PDEGym / Poseidon Benchmark

### Architecture Type
- **Type**: Transformer-based (SwinV2 architecture)
- **Not a fully connected network** - uses vision transformer architecture

### Model Sizes
From `train.py` MODEL_MAP:

#### Tiny (T)
- **embed_dim**: 48
- **depths**: [4, 4, 4, 4] (4 stages, 4 blocks each = 16 transformer blocks)
- **num_heads**: [3, 6, 12, 24] (per stage)
- **mlp_ratio**: 4.0
- **window_size**: 16
- **patch_size**: 4

#### Small (S)
- **embed_dim**: 48
- **depths**: [8, 8, 8, 8] (4 stages, 8 blocks each = 32 transformer blocks)
- **num_heads**: [3, 6, 12, 24]
- **mlp_ratio**: 4.0
- **window_size**: 16
- **patch_size**: 4

#### Base (B) - Default
- **embed_dim**: 96
- **depths**: [8, 8, 8, 8] (4 stages, 8 blocks each = 32 transformer blocks)
- **num_heads**: [3, 6, 12, 24]
- **mlp_ratio**: 4.0
- **window_size**: 16
- **patch_size**: 4

#### Large (L)
- **embed_dim**: 192
- **depths**: [8, 8, 8, 8] (4 stages, 8 blocks each = 32 transformer blocks)
- **num_heads**: [3, 6, 12, 24]
- **mlp_ratio**: 4.0
- **window_size**: 16
- **patch_size**: 4

**Note**: Poseidon uses a completely different architecture (vision transformer) compared to fully connected networks.

---

## PDEBench

### Architecture
- **Type**: Various architectures depending on implementation
- **Common**: Fully connected networks similar to PINNacle
- **Typical sizes**: Similar to PINNacle defaults (100-200 neurons, 3-5 layers)
- **Note**: PDEBench is more of a dataset collection; specific architectures depend on the implementation used

---

## Comparison Summary

| Benchmark | Architecture Type | Hidden Layers | Hidden Size | Total Params (approx) | Notes |
|-----------|------------------|---------------|------------|----------------------|-------|
| **MMNN (default)** | Low-rank FCN | 6 | 366 width, 16 rank | ~2.1M | Matrix multiplication structure |
| **MMNN (ablation)** | Low-rank FCN | 6 | 1024 width, 3-1024 rank | ~6-12M | Rank ablation study |
| **PINNacle (default)** | FCN | 5 | 100 | ~50-100K | Standard PINN architecture |
| **PINNacle (VPINN)** | FCN | 3 | 15-30 | ~1-10K | Problem-specific |
| **PDEArena** | Conv/Operator | N/A | 64-256 channels | Varies widely | Not FCN, uses convolutions |
| **Poseidon** | Transformer | 16-32 blocks | 48-192 embed | ~1-50M | Vision transformer, not FCN |

---

## Key Differences

1. **MMNN** uses a **low-rank matrix structure** with alternating rank→width and width→rank layers, which is fundamentally different from standard fully connected networks.

2. **PINNacle** uses **standard fully connected networks** with relatively small hidden sizes (15-100 neurons) and shallow depths (3-5 layers).

3. **PDEArena** uses **convolutional/operator architectures** (FNO, UNet, ResNet) which are not directly comparable to FCNs in terms of parameter count.

4. **Poseidon** uses a **vision transformer architecture** which is completely different from FCNs.

5. **MMNN ablation study** tests much larger networks (1024 width) compared to PINNacle defaults (100 width), but with low-rank constraints that reduce effective capacity.

---

## Recommendations for Fair Comparison

To fairly compare MMNN with benchmarks:

1. **For PINNacle**: Compare MMNN with rank=width (full rank) to match standard FCN capacity, or use similar parameter budgets.

2. **For PDEArena**: Direct comparison is difficult due to architectural differences. Consider comparing at similar parameter budgets or computational costs.

3. **For Poseidon**: Not directly comparable due to transformer architecture. Consider comparing at similar parameter budgets.

4. **Parameter budget matching**: Consider matching total parameter counts rather than architecture details for fair comparison.
