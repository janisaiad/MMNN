# Dataset Installation Guide for PDE Benchmarks

## Overview

This guide provides installation instructions for the 5 PDE benchmarks used in the ablation study:
1. **Flowbench**
2. **PDEArena** (Gupta & Brandstetter, 2022)
3. **PDEGym** (Herde et al., 2024)
4. **PDEBench**
5. **PINNacle**

---

## 1. Flowbench

### Installation

**Option 1: From Hugging Face (Recommended)**
```bash
pip install datasets
```

Then in Python:
```python
from datasets import load_dataset
dataset = load_dataset("BGLab/FlowBench")
```

**Option 2: Direct Download**
- **Hugging Face**: https://huggingface.co/datasets/BGLab/FlowBench
- **Paper**: https://arxiv.org/abs/2409.18032

### Dataset Details
- **Samples**: 10,000+ flow simulation samples
- **Formats**: FPO (Flow Past Object), LDC (Lid Driven Cavity)
- **Resolutions**: 1024x256, 512x512, 256x256, 128x128x128
- **Data**: Velocity, pressure, temperature fields
- **Input**: 2D/3D spatial coordinates
- **Output**: Flow field variables

### Implementation Notes
- Load from Hugging Face datasets
- Extract train/test splits
- Format as (x, y) → (u, v, p) for 2D Navier-Stokes

---

## 2. PDEArena

### Installation

**From GitHub:**
```bash
git clone https://github.com/pdearena/pdearena.git
cd pdearena
pip install -e .
```

**Or via pip (if available):**
```bash
pip install pdearena
```

### Resources
- **GitHub**: https://github.com/pdearena/pdearena
- **Documentation**: https://pdearena.github.io/pdearena/
- **Paper**: Gupta & Brandstetter (2022) - "Towards Multi-spatiotemporal-scale Generalized PDE Modeling"

### Dataset Details
- **Framework**: PyTorch Lightning-based
- **Features**: Multiple PDE types, distributed training support
- **Input**: 2D spatial coordinates
- **Output**: Solution fields

### Implementation Notes
- Use PDEArena's dataset loaders
- Follow their data format conventions
- Check their examples for proper usage

---

## 3. PDEGym

### Installation

**From GitHub (Poseidon project):**
```bash
git clone https://github.com/camlab-ethz/poseidon.git
cd poseidon
pip install -e .
```

**Datasets on Hugging Face:**
```bash
pip install datasets
# Datasets available at: https://huggingface.co/camlab-ethz
```

### Resources
- **GitHub**: https://github.com/camlab-ethz/poseidon
- **Hugging Face**: https://huggingface.co/camlab-ethz
- **Paper**: Herde et al. (2024) - "Poseidon: Efficient Foundation Models for PDEs"

### Dataset Details
- **Collection**: Multiple PDEs and operators
- **Format**: Custom dataset classes (BaseDataset/BaseTimeDataset)
- **Input**: 2D spatial coordinates
- **Output**: Solution fields

### Implementation Notes
- Datasets may need custom loading from Hugging Face
- Check Poseidon repository for dataset structure
- May require implementing custom dataset class

---

## 4. PDEBench

### Installation

**From PyPI (Recommended):**
```bash
pip install pdebench
```

**With data generation (if needed):**
```bash
pip install "pdebench[datagen310]"  # Python 3.10
# or
pip install "pdebench[datagen39]"   # Python 3.9
```

**From GitHub:**
```bash
git clone https://github.com/pdebench/PDEBench.git
cd PDEBench
pip install --upgrade pip wheel
pip install .
```

### Resources
- **GitHub**: https://github.com/pdebench/PDEBench
- **DaRUS Repository**: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
- **Paper**: NeurIPS 2022 Datasets and Benchmarks track

### Dataset Details
- **Format**: HDF5 files
- **PDE Types**: Advection, Burgers, diffusion-reaction, Navier-Stokes
- **Pre-generated datasets**: Available for download
- **Input**: 2D spatial coordinates
- **Output**: Solution fields

### Implementation Notes
- Use `pdebench` package's data loaders
- Pre-generated HDF5 files can be downloaded
- Check their examples for loading specific PDE types

---

## 5. PINNacle

### Installation

**Note**: PINNacle is a benchmark toolbox, not just a dataset. You may need to:

1. **Download the repository:**
```bash
# Check for official repository (may be on GitHub or other platform)
# Look for: "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks"
```

2. **Install dependencies:**
```bash
pip install torch numpy scipy matplotlib
# Additional dependencies as specified in their repository
```

### Resources
- **Paper**: NeurIPS 2024 Datasets and Benchmarks track
- **ArXiv**: https://arxiv.org/abs/2306.08827
- **OpenReview**: https://openreview.net/forum?id=ApjY32f3Xr

### Dataset Details
- **PDEs**: 20+ distinct PDEs from various domains
- **Domains**: Heat conduction, fluid dynamics, biology, electromagnetics, geophysics
- **Challenges**: Complex geometry, multi-scale, nonlinearity, high dimensionality
- **Backend**: PyTorch
- **Input**: 2D spatial (+ time for time-dependent)
- **Output**: Solution fields

### Implementation Notes
- PINNacle provides PDE definitions, not just datasets
- You'll need to implement the PDE residual computation in `compute_pinn_loss()`
- Check their toolbox for PDE equation definitions
- May need to generate collocation points yourself

---

## Quick Installation Summary

### All at Once (if using pip):
```bash
# Core dependencies
pip install datasets pdebench

# PDEArena
git clone https://github.com/pdearena/pdearena.git
cd pdearena && pip install -e . && cd ..

# PDEGym (Poseidon)
git clone https://github.com/camlab-ethz/poseidon.git
cd poseidon && pip install -e . && cd ..

# PINNacle - check for official repository
# (May need manual installation)
```

### Using uv (your package manager):
```bash
# Add to pyproject.toml dependencies:
# - datasets
# - pdebench

# Then install:
uv sync

# For git-based packages, add them separately
```

---

## Implementation Checklist

After installing datasets, you need to:

### 1. Update `load_pde_dataset()` function
**Location**: `experiments/table/mmnn_vs.py` (line ~160)

Replace placeholder loaders with actual dataset loading:
- [ ] Flowbench: Load from Hugging Face datasets
- [ ] PDEArena: Use PDEArena's data loaders
- [ ] PDEGym: Load from Hugging Face or Poseidon format
- [ ] PDEBench: Use pdebench package loaders
- [ ] PINNacle: Implement PINN-specific data loading

### 2. Update `compute_pinn_loss()` function
**Location**: `experiments/table/mmnn_vs.py` (line ~328)

Implement actual PDE residual computation:
- [ ] Replace placeholder PDE residual with actual equations
- [ ] Implement boundary condition enforcement
- [ ] Implement initial condition handling
- [ ] Add proper automatic differentiation for derivatives

### 3. Test Each Dataset
- [ ] Test Flowbench loading
- [ ] Test PDEArena loading
- [ ] Test PDEGym loading
- [ ] Test PDEBench loading
- [ ] Test PINNacle PINN loss computation

---

## Alternative: Start with One Dataset

If installing all datasets is overwhelming, you can:

1. **Start with PDEBench** (easiest - pip install)
   ```bash
   pip install pdebench
   ```

2. **Test with one benchmark first**
   - Modify `generate_ablation_configs()` to only generate PDEBench configs
   - Implement PDEBench loader
   - Run small test
   - Then add other benchmarks one by one

3. **Use placeholder data for testing**
   - The script already works with placeholders
   - You can test the full pipeline first
   - Then replace placeholders with real data gradually

---

## Current Status

✅ **Script structure**: Ready  
✅ **Training pipeline**: Ready  
✅ **Saving/logging**: Ready  
⚠️ **Dataset loaders**: Need implementation  
⚠️ **PINN residuals**: Need implementation  

The script will work with placeholder data, but for meaningful results, you need to implement the actual dataset loaders and PINN residual computations.
