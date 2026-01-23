# Benchmark Functions for MMNN Architecture Tuning

This folder contains plots of **actual benchmark functions** (1D and 2D) from PDE benchmarks to help tune your **MMNN (Matrix Multiplication Neural Network)** architecture.

## Why These Benchmarks?

### ✅ Included (Suitable for MLP/MMNN)
- **PINNacle**: Uses fully connected networks (FCN) with 1D problems like Burgers1D, Wave1D
- **Synthetic Functions**: Standard 1D test functions for neural network comparison
- **PDEBench**: Dataset collection (if 1D slices are available)

### ❌ Excluded (Not Suitable for MLP/MMNN)
- **PDEArena**: Uses **convolutional architectures** (FNO, UNet, ResNet) - not FCN
- **Poseidon/PDEGym**: Uses **transformer architectures** (SwinV2) - not FCN

## Why Transformers Are Not Included

**Poseidon/PDEGym** uses a **Vision Transformer (SwinV2)** architecture, which is fundamentally different from fully connected networks:

- **Architecture**: Transformer with attention mechanisms, patch embeddings, window-based self-attention
- **Input Processing**: Processes 2D/3D spatial grids as image patches
- **Not Comparable**: Cannot be directly compared to MLP/MMNN which operate on 1D vectors

For a fair comparison between **MLP** and **MMNN**, we need:
1. **Fully connected network architectures** (not convolutional/transformer)
2. **1D or vectorized inputs** (not 2D/3D spatial grids)
3. **Similar parameter budgets** for fair comparison

## Generated Plots

The script `plot_benchmark_functions.py` generates plots from **actual benchmark datasets**:

### 1D Functions:
- `pinnacle_burgers1d.png`: Burgers 1D equation time evolution
- `pinnacle_burgers1d_ic.png`: Initial condition for Burgers 1D
- `pinnacle_wave1d.png`: Wave 1D equation (x-t slice)

### 2D Functions (Contour Plots):
- `pinnacle_poisson2d_classic.png`: Poisson 2D Classic (contour)
- `pinnacle_poisson2d_classic_surface.png`: Poisson 2D Classic (3D surface)
- `pinnacle_poisson_boltzmann2d.png`: Poisson-Boltzmann 2D
- `pinnacle_ns2d_lid_u.png`: Navier-Stokes 2D - u velocity component
- `pinnacle_ns2d_lid_v.png`: Navier-Stokes 2D - v velocity component
- `pinnacle_ns2d_lid_p.png`: Navier-Stokes 2D - pressure

### 2D Time-Dependent Functions (Time Slices):
- `pinnacle_heat2d_varying.png`: Heat 2D with varying coefficient (4 time slices)
- `pinnacle_heat2d_multiscale.png`: Heat 2D multiscale (4 time slices)
- `pinnacle_wave2d_heterogeneous.png`: Wave 2D heterogeneous (4 time slices)

## Usage

Run the plotting script:
```bash
cd /Data/janis.aiad/MMNN/experiments/table
python plot_benchmark_functions.py
```

Plots will be saved to `plots_to_fit/` directory.

**Note**: This script plots **actual benchmark functions** from the datasets, not synthetic functions. All plots are generated from real reference data in the PINNacle benchmark.

## For MMNN Architecture Tuning

These plots help you:
1. **Understand function complexity**: See the actual shapes, scales, and features of benchmark functions
2. **Tune MMNN architecture**: Use visual inspection to determine appropriate:
   - Rank values (for low-rank structure)
   - Width values (for hidden layer sizes)
   - Number of layers
   - Input/output dimensions
3. **Compare 1D vs 2D**: See how function complexity changes with dimensionality
4. **Time-dependent behavior**: Understand how functions evolve over time (for time-dependent PDEs)

### Key Observations:
- **1D functions**: Simpler, good for initial architecture testing
- **2D functions**: More complex, require larger networks
- **Time-dependent**: Need to handle temporal evolution
- **Multi-component**: NS equations have u, v, p components - consider separate outputs or vector outputs
