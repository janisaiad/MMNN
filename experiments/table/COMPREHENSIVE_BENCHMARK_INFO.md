# Comprehensive Frequency Benchmark - Run Information

## Summary

**Total Runs: 62**

### Breakdown:

1. **Rank 100 Configurations**: 6 runs
   - 3 frequencies (36, 72, 144) × 2 fixWb options (True, False)
   - Batch size: 100 (default)
   - Epochs: 10,000

2. **2x Batch Size Configurations**: 28 runs
   - Existing ranks: [10, 15, 20, 25, 50]
   - 3 frequencies × existing ranks × 2 fixWb options
   - Note: Rank 50 not tested at frequency 144 (as in original)
   - Batch size: 200 (2× default)
   - Epochs: 10,000

3. **0.5x Batch Size Configurations**: 28 runs
   - Existing ranks: [10, 15, 20, 25, 50]
   - 3 frequencies × existing ranks × 2 fixWb options
   - Note: Rank 50 not tested at frequency 144 (as in original)
   - Batch size: 50 (0.5× default)
   - Epochs: 10,000

## Configuration Details

### Frequency Scaling
- **Frequency 36**: 1000 training samples, 1234 test samples
- **Frequency 72**: 2000 training samples, 2468 test samples
- **Frequency 144**: 4000 training samples, 4936 test samples

### Architecture
- Layers: 8
- Hidden width: 777
- Learning rate: 0.001 (step decay: gamma=0.9, step_size=100)
- Device: CUDA (if available)

### Function
$f(x) = \cos(f_1 \pi x^2) - 0.8 \cos(f_2 \pi x^2)$

## Output Directory

All results saved to: `experiments/table/results_frequency_benchmark_comprehensive/`

## Naming Convention

Configurations are named as:
- `freq{freq1}_{freq2}_rank{rank}_{fixWb}_{batch}{batch_size}`

Examples:
- `freq36_12_rank100_fixWbTrue_batch100`
- `freq72_24_rank20_fixWbFalse_batch200`
- `freq144_48_rank15_fixWbTrue_batch50`

## Expected Duration

Each run trains for 10,000 epochs. Estimated time per run depends on:
- Rank (higher rank = more parameters = slower)
- Batch size (larger batch = fewer iterations per epoch = faster)
- Frequency (higher frequency = more samples = slower)

**Total estimated time**: Several days (depending on GPU availability and parallelization)

## Monitoring

Check progress:
```bash
tail -f comprehensive_benchmark.log
```

Check running processes:
```bash
ps aux | grep comprehensive_frequency_benchmark
```

Check completed runs:
```bash
ls -d results_frequency_benchmark_comprehensive/*/ | wc -l
```

## Analysis

After completion, run:
```bash
python3 analyze_frequency_benchmark.py
```

This will generate comprehensive tables comparing:
- Rank ablation (including rank 100)
- Batch size ablation (50, 100, 200)
- fixWb effects
- Frequency scaling

---

*Started: 2024*
*Total configurations: 62*
*Epochs per configuration: 10,000*
