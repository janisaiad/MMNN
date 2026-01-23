# Frequency Benchmark Training

## Configuration

**Started**: Background process

### Test Setup
- **Functions**: 3 frequency pairs
  - Base: `cos(36*pi*x^2) - 0.8*cos(12*pi*x^2)`
  - 2x: `cos(72*pi*x^2) - 0.8*cos(24*pi*x^2)`
  - 4x: `cos(144*pi*x^2) - 0.8*cos(48*pi*x^2)`

- **Ranks**: [10, 15, 20, 25, 50]
- **fixWb**: [False, True]
- **Total configurations**: 3 frequencies × 5 ranks × 2 fixWb = **30 configs**

### Base Config (from benchmark.py)
- **Layers**: 8
- **Width**: 777
- **Base training samples**: 1000 (scales with frequency)
- **Base test samples**: 1234 (scales with frequency)
- **Batch size**: 100
- **Epochs**: 3000
- **Learning rate**: 0.001
- **LR scheduler**: StepLR (gamma=0.9, step_size=100)
- **Interval**: [-1, 1]

### Sample Scaling
Training/test samples scale with frequency:
- **freq (36,12)**: 1000 train, 1234 test (base)
- **freq (72,24)**: 2000 train, 2468 test (2x)
- **freq (144,48)**: 4000 train, 4936 test (4x)

## Output Locations

- **Log file**: `test_frequency_benchmark.log`
- **Results directory**: `experiments/table/results_frequency_benchmark/`
- **Summary file**: `experiments/table/results_frequency_benchmark/frequency_benchmark_summary.json`

## Monitoring

```bash
# we check if process is running
ps aux | grep test_frequency_benchmark.py | grep -v grep

# we view live log
tail -f test_frequency_benchmark.log

# we count completed configs
find experiments/table/results_frequency_benchmark -name "results.json" | wc -l
```

## Expected Results

This benchmark will test:
1. **Frequency scaling**: How well MMNN handles higher frequency functions
2. **fixWb effect**: Whether freezing rank→width layers affects high-frequency learning
3. **Rank selection**: Optimal rank for different frequency regimes
