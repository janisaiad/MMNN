# Training Run: 10 Layers, 2000 Samples

## Configuration

**Started**: 2026-01-23 02:52:38

### Updated Parameters
- **Layers**: 10 (was 5)
- **Training samples**: 2000 (was 1000)
- **Test samples**: 500 (unchanged)
- **Batch size**: 500 (unchanged)
- **Width**: 1024 (unchanged)
- **Learning rate**: 0.001 (unchanged)
- **Epochs**: 8000 per configuration

### Experiment Setup
- **Total configurations**: 84
- **Benchmarks**: flowbench (42), pinnacle (42)
- **fixWb options**: False (42), True (42)
- **Ranks**: [3, 6, 10, 15, 25, 50, 1024] × 12 each
- **Seeds**: 3 per configuration (42, 123, 456)

### Expected Runtime
- **Estimated time per config**: ~5.3 minutes
- **Total estimated time**: ~7.5 hours
- **Note**: With 10 layers and 2000 samples, training will take longer than the previous 4-hour target

## Process Status

**Process ID**: 1314088
**Status**: Running
**Current config**: 1/84 (flowbench, fixWb=False, rank=3, seed=42)

## Output Locations

- **Log file**: `train_1d_comprehensive_10layers_2000samples.log`
- **Results directory**: `experiments/table/results_1d_comprehensive/`
- **Summary file**: `experiments/table/results_1d_comprehensive/comprehensive_summary.json`

## Monitoring

To check progress:
```bash
# we check if process is running
ps aux | grep train_1d_comprehensive.py | grep -v grep

# we view recent log output
tail -f train_1d_comprehensive_10layers_2000samples.log

# we count completed configurations
find experiments/table/results_1d_comprehensive -name "results.json" | wc -l
```

## Differences from Previous Run

1. **Depth**: 5 → 10 layers (2x deeper)
2. **Training samples**: 1000 → 2000 (2x more data)
3. **Expected runtime**: ~1.8 hours → ~7.5 hours (longer due to more layers and data)

## Model Size Comparison

With 10 layers (vs 5):
- **Rank 3**: ~75K params (was ~39K)
- **Rank 15**: ~324K params (was ~162K)
- **Rank 1024**: ~21M params (was ~10.5M)
