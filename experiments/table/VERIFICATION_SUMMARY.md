# Hypothesis Verification Plan: loss = g(L/freq) with Optimal Range 7-12

## Hypothesis

The scaling law follows: **loss = g(L/freq)** where:
- **Decreasing** until L/freq = 7
- **Optimal** in range L/freq = 7-12 (perfect training)
- **Increasing** after L/freq = 12

## Current Data Analysis

From 60 completed configurations:
- **L/freq = 10.0**: Excellent performance (loss = 4.4e-07) ✓
- **L/freq = 8.0**: High loss (~1.7) ⚠️ (but tested on different frequencies)
- **L/freq = 12.0**: High loss (~1.7) ⚠️ (but tested on different frequencies)

**Issue**: Current data is sparse - need more points to verify the curve shape.

## Important Consideration

The frequency multiplier is **not directly** the frequency in the cosine:
- Cosine frequencies: [12, 24, 36, 72] × freq_multiplier
- Maximum frequency: 72 × freq_multiplier
- Mean frequency: 36 × freq_multiplier

We may need to use **L / max_cosine_freq** or **L / mean_cosine_freq** instead of **L / freq_multiplier**.

## Large Scale Verification Plan

### Configuration
- **Frequencies**: [0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
- **Ranks**: [10, 15, 25]
- **L values**: Densely sampled to cover L/freq range 4-20
- **Total**: 378 configurations

### Strategy
For each frequency:
1. Compute L range to cover L/freq from 4 to 20
2. Test many L values (every 1-2 layers depending on frequency)
3. Ensure dense coverage of optimal range 7-12 (test every 0.5 in ratio)
4. This will give smooth curve: loss = g(L/freq)

### Expected Outcome
After running:
- Smooth curve showing loss = g(L/freq)
- Clear U-shape with minimum in range 7-12
- Verification that L/freq = 7-12 gives best training

## Scripts Ready

1. **`train_large_scale_verification.py`**: Main training script
2. **`analyze_optimal_L_over_freq_range.py`**: Analysis script (updated to use mean of min loss)
3. **`plan_large_scale_verification.py`**: Planning script

## Next Steps

1. Review the plan
2. Run `train_large_scale_verification.py` to generate data
3. Run `analyze_optimal_L_over_freq_range.py` to verify hypothesis
4. Check if using L/max_cosine_freq gives better scaling law
