# Comprehensive 1D Training Run

## Overview
Running comprehensive 1D training experiments to compare MMNN architectures with different configurations.

## Configuration Details

### Parameters
- **Depth**: 5 layers
- **Width**: 1024
- **Training samples**: 1000
- **Test samples**: 500
- **Batch size**: 500
- **Epochs**: 8000 per configuration

### Comparisons
1. **fixWb**: Testing both `False` and `True`
   - `False`: All weights trainable
   - `True`: Rank→width weights frozen (only width→rank weights trainable)

2. **Rank values**: Testing from very low to full rank
   - Low ranks: 3, 6, 10, 15, 25, 50
   - Full rank: 1024 (rank = width)

3. **Benchmarks**: 
   - `flowbench`: Generic 1D PDE benchmark
   - `pinnacle`: PINN-based 1D problems (uses physics-informed loss)

4. **Seeds**: 3 different seeds per configuration (42, 123, 456)
   - For statistical robustness and reproducibility

### Total Configurations
- **84 configurations** total
- 2 benchmarks × 2 fixWb options × 7 ranks × 3 seeds = 84

## Expected Runtime
- Estimated: ~3-4 hours total
- Per configuration: ~2-3 minutes (with 8000 epochs)

## Output Location
- **Results directory**: `experiments/table/results_1d_comprehensive/`
- **Log file**: `experiments/table/train_1d_comprehensive.log`
- **Summary file**: `results_1d_comprehensive/comprehensive_summary.json`

## Monitoring
Run the status check script:
```bash
./check_training_status.sh
```

Or check the log directly:
```bash
tail -f train_1d_comprehensive.log
```

## What's Being Compared

### fixWb Comparison
- **fixWb=False**: Standard training, all weights learnable
- **fixWb=True**: Only width→rank weights learnable (rank→width frozen)
  - Tests if low-rank structure benefits from this constraint

### Rank Comparison
- **Low ranks (3-50)**: Tests if very low-rank MMNN can still learn effectively
- **Full rank (1024)**: Baseline comparison (equivalent to standard MLP)

### Expected Insights
1. How does rank affect learning for 1D problems?
2. Does fixWb help or hurt performance?
3. What's the optimal rank for 1D PDE problems?
4. How do results vary with different random seeds?

## Results Structure
Each configuration generates:
- `config.json`: Configuration parameters
- `results.json`: Final results and metrics
- `errors.npz`: Training/test error evolution
- `model_parameters.pth`: Trained model weights
- `all_tensors.pt` / `all_tensors.npz`: Predictions and data
- `training.log`: Detailed training log
- `loss_evolution.png`: Loss curves
- `error_evolution.png`: Error curves

## Completion
Training will complete automatically. Check `comprehensive_summary.json` for all results once finished.
