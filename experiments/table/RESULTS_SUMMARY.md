# Comprehensive 1D Training Results Summary

## Training Completion
✅ **All 84 networks trained successfully!**
- **Total time**: 1.80 hours
- **Completed at**: 2026-01-22 20:15:53
- **All configurations**: 84/84 completed

## Network Statistics

### Total Networks Trained: **84**

### Breakdown by Benchmark
- **flowbench**: 42 networks
- **pinnacle**: 42 networks (PINN-based)

### Breakdown by fixWb
- **fixWb=False**: 42 networks (all weights trainable)
- **fixWb=True**: 42 networks (rank→width weights frozen)

### Breakdown by Rank
- **rank=3**: 12 networks (very low rank)
- **rank=6**: 12 networks
- **rank=10**: 12 networks
- **rank=15**: 12 networks
- **rank=25**: 12 networks
- **rank=50**: 12 networks
- **rank=1024**: 12 networks (full rank, equivalent to standard MLP)

### Breakdown by Seed
- Each configuration tested with 3 seeds (42, 123, 456) for robustness
- Total: 84 configurations = 2 benchmarks × 2 fixWb × 7 ranks × 3 seeds

## Performance Results

### Overall Performance
- **Mean test error**: 1.0332e+00
- **Min test error**: 8.2427e-01 (best configuration)
- **Max test error**: 4.7404e+00
- **Median test error**: 9.5569e-01

### Best 5 Configurations (Lowest Test Error)

1. **flowbench | fixWb=False | rank=15 | seed=123**
   - Test error: **8.2427e-01**

2. **flowbench | fixWb=False | rank=25 | seed=456**
   - Test error: **8.2691e-01**

3. **flowbench | fixWb=False | rank=15 | seed=456**
   - Test error: **8.3325e-01**

4. **flowbench | fixWb=False | rank=25 | seed=42**
   - Test error: **8.3626e-01**

5. **flowbench | fixWb=False | rank=50 | seed=42**
   - Test error: **8.4535e-01**

## Key Findings

### fixWb Comparison

#### fixWb=False (All weights trainable)
- **Best ranks**: 15, 25, 50 (mean error ~0.90)
- **Worst rank**: 1024 (full rank, mean error ~0.97)
- **Low ranks (3-10)**: Perform reasonably well (~0.94-0.94)
- **Optimal range**: rank 15-50 shows best performance

#### fixWb=True (Rank→width weights frozen)
- **Best ranks**: 1024, 50, 15 (mean error ~0.96-0.98)
- **Problematic ranks**: 6, 10 (high variance, mean error ~1.4-1.6)
- **Full rank (1024)**: Performs best with fixWb=True (~0.95)
- **Low ranks (3-10)**: High variance, inconsistent performance

### Rank Analysis

**For fixWb=False:**
- Rank 15-25: **Optimal** (mean error ~0.90)
- Rank 50: Good performance (~0.92)
- Rank 3-10: Moderate performance (~0.94)
- Rank 1024: Slightly worse (~0.97)

**For fixWb=True:**
- Rank 1024: **Best** (mean error ~0.95)
- Rank 3, 15, 50: Good performance (~0.96-0.98)
- Rank 6, 10: **Problematic** (high variance, poor performance)

### Benchmark Comparison
- **flowbench**: Generally performs better (best error: 0.82)
- **pinnacle**: Slightly higher errors (PINN loss adds complexity)

## Architecture Insights

1. **Low-rank structure works**: Ranks 15-50 perform as well or better than full rank (1024)
2. **fixWb=False is generally better**: Allows all weights to learn, giving more flexibility
3. **Optimal rank range**: 15-25 appears to be the sweet spot for 1D problems
4. **fixWb=True with low ranks (6, 10) is problematic**: High variance suggests instability
5. **Full rank (1024) benefits from fixWb=True**: When rank=width, freezing helps

## Training Details

### Per Configuration
- **Epochs**: 8000
- **Training samples**: 1000
- **Test samples**: 500
- **Batch size**: 500
- **Depth**: 5 layers
- **Width**: 1024
- **Average training time**: ~1.3 minutes per configuration

### Model Sizes
- **Low rank (3)**: ~39K parameters
- **Medium rank (15-25)**: ~200K-300K parameters
- **Full rank (1024)**: ~10M+ parameters

## Results Location

All results saved to:
- **Directory**: `experiments/table/results_1d_comprehensive/`
- **Summary file**: `comprehensive_summary.json`
- **Individual results**: Each configuration has its own directory with:
  - `results.json`: Final metrics
  - `config.json`: Configuration parameters
  - `errors.npz`: Error evolution
  - `model_parameters.pth`: Trained weights
  - `all_tensors.pt` / `all_tensors.npz`: Predictions
  - `loss_evolution.png`: Loss curves
  - `error_evolution.png`: Error curves
  - `prediction_epoch*.png`: Prediction plots

## Recommendations for MMNN Architecture Tuning

Based on these results:

1. **For 1D problems**: Use **rank 15-25** with **fixWb=False**
2. **If using fixWb=True**: Prefer **rank 1024** (full rank) or rank 50+
3. **Avoid**: fixWb=True with very low ranks (6, 10) - high variance
4. **Parameter efficiency**: Rank 15-25 provides good performance with ~10x fewer parameters than full rank
