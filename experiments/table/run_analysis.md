# Ablation Study: Run Analysis and Requirements

## Total Number of Runs

### Configuration Breakdown
- **5 benchmarks**: Flowbench, PDEArena, PDEGym, PDEBench, PINNacle
- **2 fixWb options**: False, True
- **7 rank options**: 3, 6, 10, 15, 25, 50, 1024

**Total: 5 × 2 × 7 = 70 configurations**

### Per Configuration
- **Epochs**: 5000 per configuration
- **Training samples**: 1000
- **Test samples**: 500
- **Batch size**: 100
- **Batches per epoch**: 1000 / 100 = 10 batches

**Total epochs across all configs: 70 × 5000 = 350,000 epochs**

## Time Estimates

### Per Epoch Estimate
Based on typical MMNN training with width=1024:
- **Forward pass**: ~0.01-0.05 seconds per batch
- **Backward pass**: ~0.01-0.05 seconds per batch
- **Total per batch**: ~0.02-0.10 seconds
- **Per epoch** (10 batches): ~0.2-1.0 seconds
- **With evaluation/logging**: ~0.5-2.0 seconds per epoch

### Per Configuration Estimate
- **Best case** (fast GPU, simple model): 5000 epochs × 0.5s = **~42 minutes**
- **Typical case** (standard GPU): 5000 epochs × 1.0s = **~83 minutes (1.4 hours)**
- **Worst case** (slower GPU, complex PINN): 5000 epochs × 2.0s = **~167 minutes (2.8 hours)**

### Total Time Estimates

#### Sequential Execution (1 GPU)
- **Best case**: 70 × 42 min = **49 hours (~2 days)**
- **Typical case**: 70 × 83 min = **97 hours (~4 days)**
- **Worst case**: 70 × 167 min = **195 hours (~8 days)**

#### Parallel Execution (Multiple GPUs)
With **N GPUs**, time is approximately divided by N:

| GPUs | Best Case | Typical Case | Worst Case |
|------|-----------|--------------|------------|
| 1    | ~2 days   | ~4 days      | ~8 days    |
| 2    | ~1 day    | ~2 days      | ~4 days    |
| 4    | ~12 hours | ~1 day       | ~2 days    |
| 8    | ~6 hours  | ~12 hours    | ~1 day     |

**Note**: These are estimates. Actual time depends on:
- GPU model and speed
- Dataset complexity
- PINN residual computation overhead
- System I/O performance

## Storage Requirements

### Per Configuration
- **Model checkpoint**: ~50-200 MB (depending on rank)
- **All tensors** (`.pt` + `.npz`): ~100-500 MB
- **Plots and JSON**: ~10-50 MB
- **Total per config**: ~200-800 MB

### Total Storage
- **70 configurations**: 70 × 800 MB = **~56 GB** (worst case)
- **Typical**: 70 × 400 MB = **~28 GB**
- **Best case**: 70 × 200 MB = **~14 GB**

**Recommended**: At least **60-100 GB free space** for safety

## Installation and Setup Requirements

### ✅ Already Installed (from pyproject.toml)
- PyTorch (≥2.7.1)
- NumPy (≥2.1.3)
- Matplotlib (≥3.10.3)
- tqdm (≥4.67.1)
- All standard dependencies

### ❌ Need to Install/Download

#### 1. PDE Benchmark Datasets

**Flowbench**
```bash
# TODO: Install Flowbench dataset
# Check: https://github.com/.../flowbench
# pip install flowbench  # or similar
```

**PDEArena** (Gupta & Brandstetter, 2022)
```bash
# TODO: Install PDEArena
# pip install pdearena  # or download from repository
```

**PDEGym** (Herde et al., 2024)
```bash
# TODO: Install PDEGym
# pip install pdegym  # or download from repository
```

**PDEBench**
```bash
# TODO: Install PDEBench
# pip install pdebench  # or download from repository
```

**PINNacle**
```bash
# TODO: Install PINNacle
# pip install pinnacle  # or download from repository
# Check: https://github.com/.../pinnacle
```

#### 2. Dataset Loading Implementation

The script currently uses **placeholder datasets**. You need to implement actual dataset loaders in the `load_pde_dataset()` function:

**Location**: `experiments/table/mmnn_vs.py`, function `load_pde_dataset()` (around line 155)

**What to implement**:
```python
def load_pde_dataset(benchmark_name: str, split: str, use_pinn: bool = False, **kwargs) -> Dataset:
    # Replace placeholder with actual dataset loading
    if benchmark_name == "flowbench":
        # Load Flowbench dataset
        return FlowbenchDataset(split=split, ...)
    elif benchmark_name == "pdearena":
        # Load PDEArena dataset
        return PDEArenaDataset(split=split, ...)
    # ... etc
```

#### 3. PINN Residual Computation

For PINNacle, implement actual PDE residual computation in `compute_pinn_loss()`:

**Location**: `experiments/table/mmnn_vs.py`, function `compute_pinn_loss()` (around line 328)

**What to implement**:
- Actual PDE equations (Poisson, Burgers, Navier-Stokes, etc.)
- Proper boundary condition enforcement
- Initial condition handling
- Physics residual computation with automatic differentiation

## Current Status: Ready to Test (with Placeholders)

### ✅ What Works Now
- Script structure and configuration generation
- Training loop and optimization
- Model architecture (MMNN with fixWb)
- Parallel execution framework
- Saving and logging infrastructure
- Placeholder datasets (for testing script structure)

### ⚠️ What Needs Implementation
1. **Real dataset loaders** (currently using random dummy data)
2. **Actual PDE residual computation** for PINN (currently placeholder)
3. **Dataset installation** (download/install benchmark packages)

## Recommended Workflow

### Step 1: Test with Placeholders (Optional)
```bash
# Test that script runs (will use dummy data)
uv run python experiments/table/mmnn_vs.py
```
This verifies the script structure works, but results won't be meaningful.

### Step 2: Install Datasets
Install each benchmark dataset according to their documentation.

### Step 3: Implement Dataset Loaders
Update `load_pde_dataset()` to load real data.

### Step 4: Implement PINN Residuals (for PINNacle)
Update `compute_pinn_loss()` with actual PDE equations.

### Step 5: Run Full Ablation Study
```bash
# Run on GPU cluster
uv run python experiments/table/mmnn_vs.py
```

## GPU Cluster Recommendations

### Minimum Requirements
- **GPUs**: 2-4 GPUs recommended for reasonable completion time
- **GPU Memory**: At least 8GB per GPU (for width=1024)
- **Storage**: 60-100 GB free space
- **RAM**: 32GB+ recommended

### Optimal Setup
- **GPUs**: 8+ GPUs for fastest completion (~6-12 hours)
- **GPU Memory**: 16GB+ per GPU
- **Storage**: 100GB+ SSD for fast I/O
- **RAM**: 64GB+

## Monitoring Progress

The script provides:
- **Progress bars** (tqdm) for each configuration
- **Epoch-level logging** every 50 epochs
- **Summary JSON** updated as configurations complete
- **Individual result files** per configuration

Check progress:
```bash
# Count completed configurations
ls experiments/table/results/*/results.json | wc -l

# View latest summary
cat experiments/table/results/ablation_summary.json
```

## Next Steps

1. **Install benchmark datasets** (Flowbench, PDEArena, PDEGym, PDEBench, PINNacle)
2. **Implement dataset loaders** in `load_pde_dataset()`
3. **Implement PINN residuals** in `compute_pinn_loss()` for PINNacle
4. **Test on small subset** (e.g., 1 benchmark, 1 rank) before full run
5. **Run full ablation study** on GPU cluster
