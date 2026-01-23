# Complete MMNN Frequency Benchmark Experiment Documentation

## Table of Contents

1. [What is Batch Size?](#what-is-batch-size)
2. [Experiment Overview](#experiment-overview)
3. [Phase 1: Initial Frequency Benchmark](#phase-1-initial-frequency-benchmark)
4. [Phase 2: Comprehensive Benchmark](#phase-2-comprehensive-benchmark)
5. [Configuration Details](#configuration-details)
6. [Results Summary](#results-summary)
7. [Analysis and Insights](#analysis-and-insights)

---

## What is Batch Size?

**Batch size** is a fundamental hyperparameter in neural network training that determines how many training samples are processed together before updating the model's weights.

### Key Concepts:

1. **Definition**: The number of training examples used in one forward/backward pass through the network.

2. **How it works**:
   - During training, the dataset is divided into batches
   - The model processes one batch at a time
   - After processing a batch, gradients are computed and weights are updated
   - This repeats until all batches (one "epoch") are processed

3. **Effects on Training**:

   **Small Batch Size (e.g., 50)**:
   - ✅ More frequent weight updates (more gradient steps per epoch)
   - ✅ Better generalization (noisier gradients act as regularization)
   - ✅ Lower memory requirements
   - ❌ Slower training (more iterations per epoch)
   - ❌ More noisy gradients (less stable training)

   **Large Batch Size (e.g., 200)**:
   - ✅ Faster training (fewer iterations per epoch)
   - ✅ More stable gradients (smoother optimization)
   - ✅ Better GPU utilization
   - ❌ Higher memory requirements
   - ❌ May lead to worse generalization (sharp minima)
   - ❌ Fewer gradient updates per epoch

   **Medium Batch Size (e.g., 100)**:
   - Balanced trade-off between speed and generalization
   - Common default choice

4. **In Our Experiments**:
   - **Batch size 50** (0.5×): More gradient updates, potentially better generalization
   - **Batch size 100** (1×): Baseline/default configuration
   - **Batch size 200** (2×): Faster training, fewer updates per epoch

5. **Mathematical Relationship**:
   - Number of iterations per epoch = Total training samples / Batch size
   - Example: 1000 samples, batch size 100 → 10 iterations per epoch
   - Example: 1000 samples, batch size 50 → 20 iterations per epoch
   - Example: 1000 samples, batch size 200 → 5 iterations per epoch

---

## Experiment Overview

This document describes a comprehensive ablation study of **Matrix Multiplication Neural Networks (MMNN)** on frequency-dependent function approximation tasks. The experiments systematically vary:

1. **Rank** (network capacity): 10, 15, 20, 25, 50, 100
2. **fixWb** (architectural regularization): True, False
3. **Frequency** (task difficulty): (36,12), (72,24), (144,48)
4. **Batch Size** (optimization hyperparameter): 50, 100, 200

### Target Function

All experiments approximate the function:
$$f(x) = \cos(f_1 \pi x^2) - 0.8 \cos(f_2 \pi x^2)$$

where $x \in [-1, 1]$ and $(f_1, f_2)$ are frequency pairs:
- **Low frequency**: (36, 12) - 1000 training samples
- **Medium frequency**: (72, 24) - 2000 training samples
- **High frequency**: (144, 48) - 4000 training samples

### Architecture

- **Layers**: 8
- **Hidden width**: 777
- **Input/Output dimensions**: 1
- **ResNet**: False
- **Training epochs**: 10,000 (all runs)
- **Learning rate**: 0.001 (step decay: γ=0.9, step_size=100)
- **Device**: CUDA (GPU)

---

## Phase 1: Initial Frequency Benchmark

### Objective
Establish baseline performance across different ranks and fixWb settings.

### Configurations Tested

**Ranks**: 10, 15, 20, 25, 50  
**fixWb**: True, False  
**Frequencies**: (36,12), (72,24), (144,48)  
**Batch size**: 100 (fixed)

### Total Runs
- Frequency 36: 5 ranks × 2 fixWb = **10 runs**
- Frequency 72: 5 ranks × 2 fixWb = **10 runs**
- Frequency 144: 4 ranks × 2 fixWb = **8 runs** (rank 50 not tested)
- **Total: 28 runs**

### Results Location
`experiments/table/results_frequency_benchmark/`

### Key Findings (from analysis)

1. **Rank 20-25 optimal**: Best balance between expressivity and regularization
2. **fixWb=True critical**: 3.7× better mean error, 2.5× lower variance
3. **Frequency scaling exponential**: Error increases ~3× → 39× per frequency doubling
4. **Rank 50 excellent** for low/medium frequencies but only tested on those

---

## Phase 2: Comprehensive Benchmark

### Objective
Complete ablation study including:
1. Higher rank (100) to test capacity limits
2. Batch size ablation (50, 100, 200) to study optimization effects

### Part 1: Rank 100 Configurations

**Purpose**: Test if higher rank (100) improves performance, especially at high frequencies.

**Configurations**:
- Rank: 100
- fixWb: True, False
- Frequencies: (36,12), (72,24), (144,48)
- Batch size: 100 (default)

**Total runs**: 3 frequencies × 2 fixWb = **6 runs**

**Results location**: `experiments/table/results_frequency_benchmark_comprehensive/`  
**Naming**: `freq{freq1}_{freq2}_rank100_fixWb{True/False}_batch100`

### Part 2: 2× Batch Size Configurations

**Purpose**: Test if larger batch size (200) improves training speed and stability.

**Configurations**:
- Ranks: 10, 15, 20, 25, 50 (existing ranks)
- fixWb: True, False
- Frequencies: (36,12), (72,24), (144,48)
- Batch size: **200** (2× default)

**Total runs**: 
- Frequency 36: 5 ranks × 2 fixWb = 10
- Frequency 72: 5 ranks × 2 fixWb = 10
- Frequency 144: 4 ranks × 2 fixWb = 8
- **Total: 28 runs**

**Results location**: `experiments/table/results_frequency_benchmark_comprehensive/`  
**Naming**: `freq{freq1}_{freq2}_rank{rank}_fixWb{True/False}_batch200`

### Part 3: 0.5× Batch Size Configurations

**Purpose**: Test if smaller batch size (50) improves generalization through more frequent updates.

**Configurations**:
- Ranks: 10, 15, 20, 25, 50 (existing ranks)
- fixWb: True, False
- Frequencies: (36,12), (72,24), (144,48)
- Batch size: **50** (0.5× default)

**Total runs**: 
- Frequency 36: 5 ranks × 2 fixWb = 10
- Frequency 72: 5 ranks × 2 fixWb = 10
- Frequency 144: 4 ranks × 2 fixWb = 8
- **Total: 28 runs**

**Results location**: `experiments/table/results_frequency_benchmark_comprehensive/`  
**Naming**: `freq{freq1}_{freq2}_rank{rank}_fixWb{True/False}_batch50`

### Comprehensive Benchmark Summary

**Total configurations**: 6 + 28 + 28 = **62 runs**

**All runs**: 10,000 epochs each (early stopping removed)

**Script**: `comprehensive_frequency_benchmark.py`

---

## Configuration Details

### Complete Parameter Space

| Parameter | Values Tested | Total Combinations |
|-----------|---------------|-------------------|
| **Rank** | 10, 15, 20, 25, 50, 100 | 6 |
| **fixWb** | True, False | 2 |
| **Frequency** | (36,12), (72,24), (144,48) | 3 |
| **Batch Size** | 50, 100, 200 | 3 |

**Theoretical maximum**: 6 × 2 × 3 × 3 = 108 configurations

**Actually tested**: 62 configurations (systematic subset)

### Training Configuration

```python
{
    "num_layers": 8,
    "hidden_width": 777,
    "input_rank": 1,
    "output_rank": 1,
    "use_resnet": False,
    "num_epochs": 10000,
    "lr_init": 0.001,
    "lr_gamma": 0.9,
    "lr_step_size": 100,
    "interval": [-1, 1],
    "device": "cuda:0",
    "dtype": "torch.float32"
}
```

### Data Configuration

| Frequency | Training Samples | Test Samples | Scale Factor |
|-----------|------------------|--------------|--------------|
| (36, 12)  | 1000             | 1234         | 1.0×         |
| (72, 24)  | 2000             | 2468         | 2.0×         |
| (144, 48) | 4000             | 4936         | 4.0×         |

**Rationale**: Higher frequencies require more samples (Nyquist sampling theorem: need ~2× frequency samples).

### Iterations per Epoch

| Batch Size | Frequency 36 (1000 samples) | Frequency 72 (2000 samples) | Frequency 144 (4000 samples) |
|------------|----------------------------|-----------------------------|------------------------------|
| 50         | 20 iterations/epoch         | 40 iterations/epoch         | 80 iterations/epoch          |
| 100        | 10 iterations/epoch         | 20 iterations/epoch          | 40 iterations/epoch           |
| 200        | 5 iterations/epoch          | 10 iterations/epoch          | 20 iterations/epoch          |

**Total iterations over 10k epochs**:
- Batch 50: 200k, 400k, 800k iterations (for freq 36, 72, 144)
- Batch 100: 100k, 200k, 400k iterations
- Batch 200: 50k, 100k, 200k iterations

---

## Results Summary

### Phase 1 Results (Baseline: Batch Size 100)

**Best configurations by frequency**:
- **Frequency 36**: Rank 25, fixWb=True → 6.683e-04 error
- **Frequency 72**: Rank 20, fixWb=False → 5.654e-04 error
- **Frequency 144**: Rank 20, fixWb=True → 6.423e-04 error

**Key statistics**:
- **By Rank**: Rank 50 best average (1.66e-3), but rank 20-25 more consistent
- **By fixWb**: fixWb=True 3.7× better mean error, 2.5× lower variance
- **By Frequency**: Error scales exponentially (3× → 39× per doubling)

### Phase 2 Results (Comprehensive)

**Status**: Currently running (62 configurations, 10k epochs each)

**Expected insights**:
1. **Rank 100**: Will reveal if higher capacity helps at high frequencies
2. **Batch size 200**: Will test if larger batches improve speed/stability
3. **Batch size 50**: Will test if smaller batches improve generalization

---

## Analysis and Insights

### Rank Ablation

**Observations**:
- Rank 10-15: Too low for high frequencies (insufficient capacity)
- Rank 20-25: Optimal balance (best performance across frequencies)
- Rank 50: Excellent for low/medium frequencies
- Rank 100: Testing in progress (may help high frequencies)

**Interpretation**: 
- Low-rank structure provides implicit regularization
- Higher rank needed for higher frequencies (more complex functions)
- Sweet spot around rank 20-25 for general use

### fixWb Effect

**Observations**:
- fixWb=True: Freezes rank→width expansion matrices
- Dramatically improves stability (3.7× better mean error)
- Critical for high frequencies (prevents catastrophic failures)

**Interpretation**:
- fixWb=True acts as architectural regularization
- Prevents gradient flow issues in low-rank layers
- Essential for frequency-dependent problems

### Frequency Scaling

**Observations**:
- Error scales exponentially with frequency
- Frequency 144 is extremely challenging
- Some configurations fail catastrophically at high frequencies

**Interpretation**:
- High frequencies require exponentially more capacity
- Training samples scaled linearly (may be insufficient)
- Need adaptive strategies for high-frequency problems

### Batch Size (Expected Effects)

**Hypotheses**:
- **Batch 200**: Faster training, potentially worse generalization (sharp minima)
- **Batch 50**: Slower training, potentially better generalization (flat minima)
- **Batch 100**: Balanced baseline

**Testing**: Results pending from comprehensive benchmark

---

## File Structure

```
experiments/table/
├── test_frequency_benchmark.py          # Original frequency benchmark script
├── comprehensive_frequency_benchmark.py  # Comprehensive benchmark script
├── analyze_frequency_benchmark.py       # Analysis script
├── FREQUENCY_BENCHMARK_ANALYSIS.md      # Phase 1 analysis
├── COMPREHENSIVE_BENCHMARK_INFO.md      # Phase 2 info
├── COMPLETE_EXPERIMENT_DOCUMENTATION.md # This file
│
├── results_frequency_benchmark/         # Phase 1 results (28 runs)
│   ├── freq36_12_rank10_fixWbFalse_run0/
│   ├── freq36_12_rank10_fixWbTrue_run1/
│   └── ...
│
└── results_frequency_benchmark_comprehensive/  # Phase 2 results (62 runs)
    ├── freq36_12_rank100_fixWbTrue_batch100/
    ├── freq72_24_rank20_fixWbFalse_batch200/
    ├── freq144_48_rank15_fixWbTrue_batch50/
    └── ...
```

---

## Methodology

### Training Procedure

1. **Data Generation**: 
   - Training: Uniform grid on [-1, 1]
   - Test: Random samples (different distribution)

2. **Model Initialization**: 
   - Random weights (seed=42)
   - MMNN architecture with specified rank/width

3. **Optimization**:
   - Adam optimizer (lr=0.001)
   - StepLR scheduler (γ=0.9, step=100)
   - Gradient clipping (max_norm=1.0)

4. **Evaluation**:
   - Test error computed every 50 epochs
   - Final metrics: train error, test error, max error

5. **Checkpointing**:
   - Checkpoint saved every 500 epochs
   - Automatic resume from checkpoint if interrupted

### Metrics

- **Training Loss**: MSE on training set
- **Test Error**: MSE on test set (primary metric)
- **Max Error**: Maximum pointwise error on test set
- **Training Time**: Wall-clock time per configuration

---

## Future Directions

1. **Complete Rank 100 Analysis**: Once comprehensive benchmark finishes
2. **Batch Size Analysis**: Compare batch 50, 100, 200 effects
3. **Full Rank Comparison**: Compare with rank=777 (full MLP) results
4. **Adaptive Strategies**: Test frequency-adaptive architectures
5. **Extended Frequencies**: Test even higher frequencies (288, 576)

---

## Key Takeaways

1. **Batch Size**: Controls optimization dynamics - smaller = more updates, larger = faster but potentially worse generalization

2. **Rank Selection**: Rank 20-25 optimal for general frequency problems

3. **fixWb Critical**: Always use fixWb=True for frequency-dependent tasks

4. **Frequency Scaling**: Exponential difficulty increase requires adaptive strategies

5. **Comprehensive Study**: 62 configurations provide complete ablation across rank, fixWb, frequency, and batch size

---

*Documentation created: 2024*  
*Total experiments: 90 configurations (28 Phase 1 + 62 Phase 2)*  
*Total training epochs: 900,000+ (90 configs × 10k epochs)*  
*Status: Phase 1 complete, Phase 2 in progress*
