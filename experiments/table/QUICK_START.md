# Quick Start Guide

## 1. Install Datasets (One-Liner)

**Note**: PDEBench has a torch version conflict. Use this workaround:

```bash
uv pip install datasets && git clone https://github.com/pdearena/pdearena.git && cd pdearena && uv pip install -e . && cd .. && git clone https://github.com/camlab-ethz/poseidon.git && cd poseidon && uv pip install -e . && cd .. && echo "done (PDEBench: download datasets manually from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)"
```

## 2. Test Script (Small Test)

```bash
uv run python experiments/table/test_small.py
```

**What it tests:**
- 3 configurations (Flowbench fixWb=False/True, PINNacle fixWb=True)
- 10 epochs each (quick test)
- Uses **placeholder/dummy data** (random Gaussian noise)
- Verifies script structure works, not real PDE solving

**Output:** `experiments/table/test_results/`

## 3. Run Full Ablation Study

```bash
uv run python experiments/table/mmnn_vs.py
```

**What it runs:**
- 70 configurations total
- 5 benchmarks × 2 fixWb × 7 ranks (3,6,10,15,25,50,1024)
- 5000 epochs each
- **All output logged to files** (see below)

## 4. Logging

### Per-Configuration Logs
Each configuration saves:
- `training.log` - Full training log (stdout/stderr)
- `results.json` - Final metrics
- `config.json` - Configuration
- `all_tensors.pt` / `all_tensors.npz` - All predictions/targets for plotting
- `errors.npz` - Training/test errors
- `model_parameters.pth` - Model checkpoint
- Plots: `loss_evolution.png`, `error_evolution.png`, `prediction_epoch*.png`

### Main Log
- `experiments/table/results/main_log_YYYYMMDD_HHMMSS.log` - Overall run log

### Summary
- `experiments/table/results/ablation_summary.json` - All results combined

## 5. Test Data Used

The small test uses **placeholder data**:
- Random Gaussian noise (`np.random.randn`)
- Not real PDE solutions
- Just verifies code structure works
- Results are meaningless but proves script runs

For real results, implement actual dataset loaders in `load_pde_dataset()`.

## 6. PDEBench Dependency Issue

PDEBench requires `torch>=1.13.0,<1.14.dev0` but you have `torch>=2.7.1`.

**Solutions:**
1. **Download datasets manually**: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
2. **Clone repository**: `git clone https://github.com/pdebench/PDEBench.git` (use data loaders without installing package)
3. **Skip PDEBench** for now, use other 4 benchmarks

## 7. What's Ready vs What Needs Work

### ✅ Ready
- Script structure
- Training loop
- Logging to files
- Saving all tensors
- Parallel execution
- Test script

### ⚠️ Needs Implementation
- Real dataset loaders (currently placeholders)
- PINN residual computation (currently placeholder)
- PDEBench dataset loading (dependency conflict)

## 8. Expected Runtime

- **Sequential (1 GPU)**: ~4 days (70 configs × ~83 min each)
- **Parallel (4 GPUs)**: ~1 day
- **Parallel (8 GPUs)**: ~12 hours

## 9. Storage

- **Estimated**: 28-56 GB total
- **Recommended**: 60-100 GB free space
