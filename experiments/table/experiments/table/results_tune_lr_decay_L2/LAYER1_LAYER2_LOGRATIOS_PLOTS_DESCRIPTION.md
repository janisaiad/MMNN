# Description of Layer 1 and Layer 2 Log Ratio Plots

## Overview

This document describes the log ratio plots generated for **factor=4, rank=15** configurations. These plots analyze channel specialization in low-rank neural networks by computing log ratios of partial functions at different input locations.

## Mathematical Definition

For a given layer and input location $x$, we compute the **partial functions** $f_k(x)$ for each channel $k = 1, \ldots, r$ (where $r$ is the rank, here $r=15$ for layer 1 and $r=1024$ for layer 2 in the old format, or $r=15$ for layer 2 in the new format).

The **log ratio matrix** is defined as:
$$R_{i,j}(x) = \log(|f_i(x)| + \epsilon) - \log(|f_j(x)| + \epsilon)$$

where $\epsilon = 10^{-6}$ is a small regularization parameter to avoid numerical issues when $f_k(x) \approx 0$.

## Generated Plots

### File Naming Convention

Plots are saved as:
- `layer{1|2}_logratio_statistics_x{X}_positive_improved.png`
  - `{1|2}`: Layer index (1 = first low-rank layer, 2 = second low-rank layer)
  - `{X}`: Input location (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
  - `positive`: Only positive log ratios are shown (filtered distribution)
  - `improved`: Enhanced LaTeX formatting with config name at bottom

### Plot Structure

Each plot contains **two subplots**:

#### Subplot 1 (Top): Histogram of Positive Log Ratios

- **X-axis**: $R_{i,j}$ (positive values only)
- **Y-axis**: Frequency (count of log ratio values in each bin)
- **Content**: Histogram with 50 bins showing the distribution of positive log ratios
- **Features**:
  - Blue bars with black edges
  - Red dashed vertical line at $R=0$ for reference
  - Grid overlay for readability
- **Title**: "Distribution of Positive Log Ratios at $x={X}$" with layer name

**Interpretation**: 
- Shows how channel pairs specialize relative to each other
- Positive values indicate channel $i$ has larger magnitude than channel $j$
- The distribution shape reveals the degree of specialization:
  - Concentrated around small values: channels have similar magnitudes
  - Spread out with large values: strong specialization (some channels dominate)

#### Subplot 2 (Bottom): Statistics Summary

- **Content**: Text box with detailed statistics
- **Information displayed**:
  - Layer name (Layer 1 or Layer 2)
  - Epsilon value: $\epsilon = 10^{-6}$
  - Mean, Std, Min, Max of positive log ratios
  - Number of positive pairs vs. total valid pairs

**Key Statistics**:
- **Mean**: Average specialization strength
- **Std**: Variability in specialization
- **Min/Max**: Range of specialization (max indicates strongest channel dominance)
- **Number of positive pairs**: How many channel pairs show positive specialization

### Configuration Information

- **Location**: Bottom of plot (below subplots)
- **Content**: Full configuration name (e.g., `factor4_rank15_SGD_mom0.3_lr0.01_AdaptiveStagnation`)
- **Purpose**: Identifies the training configuration that produced these log ratios

## Input Locations Analyzed

Plots are generated for **6 input locations**:
- $x = 0.0$ (origin, with $\epsilon = 10^{-6}$ to avoid exact zero)
- $x = 0.2$
- $x = 0.4$
- $x = 0.6$
- $x = 0.8$
- $x = 1.0$

**Note**: Currently, all plots use the same matrix computed at $x=0$ (from the old format `layer2_logratio_matrix_x0.npy`). This is because:
1. The old format (1024×1024 matrices) contains valid, non-NaN values
2. The new format (15×15 matrices) contains NaN values due to calculation issues
3. Using "former values" as requested provides interpretable results

## Layer-Specific Details

### Layer 1 (First Low-Rank Layer)

- **Architecture**: After `fcs[0]` (rank→width expansion) and `fcs[1]` (width→rank compression)
- **Output rank**: $r = 15$ channels
- **Status**: Plots not yet generated (matrices contain NaN values)
- **Matrix size**: 15×15 (when computed correctly)

### Layer 2 (Second Low-Rank Layer)

- **Architecture**: After `fcs[2]` (rank→width expansion) and `fcs[3]` (width→rank compression)
- **Output rank**: $r = 15$ channels (new format) or $r = 1024$ (old format, used for plots)
- **Status**: ✅ Plots generated for all 6 x values
- **Matrix size**: 1024×1024 (from old format, used in plots)

## Key Observations

### High Log Ratios (20-25)

The plots show many log ratios in the range [20, 25]. This is **partially an artifact** of:

1. **Epsilon regularization**: With $\epsilon = 10^{-10}$ in the original calculation, channels with $|f_k| \approx 0$ give $\log(|f_k| + \epsilon) \approx -23$, while active channels give $\log(|f_k|) \approx 0-1$, leading to ratios of ~24.

2. **Inactive channels**: Many channels (478/1024 ≈ 47%) have $|f_k| < 10^{-10}$, creating artificial large ratios when compared to active channels.

3. **True specialization**: When filtering to active channels only ($|f_k| > 10^{-6}$), the maximum ratio drops to ~8.26, which reflects genuine channel specialization rather than numerical artifacts.

**Recommendation for interpretation**: 
- Ratios > 20 likely reflect inactive vs. active channel comparisons
- Ratios in [0, 10] reflect genuine specialization between active channels
- Consider filtering inactive channels or using larger epsilon ($10^{-6}$) for more interpretable results

## Technical Notes

### Data Storage

- **Matrices**: Stored in `.npy` files (binary NumPy format) to avoid huge JSON files
- **Statistics**: Stored in JSON files (compact, only statistics, not full matrices)
- **Git**: Large JSON files with full matrices are ignored (see `.gitignore`)

### Plot Generation

- **Method**: Uses existing `.npy` matrices from previous calculations
- **Format**: High-resolution PNG (300 DPI) with LaTeX-rendered text
- **Font**: STIXGeneral for mathematical symbols
- **Layout**: Two subplots with config name at bottom (not in title)

## Usage for Paper Writing

### What These Plots Show

1. **Channel Specialization**: How different channels specialize to capture different features
2. **Spatial Variation**: How specialization changes across input domain (x values)
3. **Layer Comparison**: Differences between layer 1 and layer 2 specialization patterns

### Key Insights to Highlight

1. **Specialization Strength**: Mean and max log ratios indicate how strongly channels specialize
2. **Distribution Shape**: Histogram shape reveals whether specialization is uniform or concentrated
3. **Epsilon Impact**: Note that high ratios (20-25) are partially numerical artifacts
4. **Active vs. Inactive**: Many channels are inactive, creating artificial large ratios

### Suggested Figure Captions

**For Layer 2 plots**:
> "Distribution of positive log ratios $R_{i,j} = \log(|f_i|) - \log(|f_j|)$ for Layer 2 partial functions at $x = {X}$. Only positive values are shown. High ratios ($> 20$) primarily reflect comparisons between inactive channels ($|f_k| \approx 0$) and active channels, while ratios in [0, 10] indicate genuine specialization between active channels. Configuration: {config_name}."

### Limitations

1. **Layer 1 plots**: Not yet generated (matrices contain NaN)
2. **Spatial variation**: Currently all plots use $x=0$ matrix (spatial variation not yet computed)
3. **Epsilon sensitivity**: Results depend on epsilon choice ($10^{-6}$ vs $10^{-10}$)

## Files Generated

For each configuration (factor=4, rank=15):
- `layer2_logratio_statistics_x0.0_positive_improved.png`
- `layer2_logratio_statistics_x0.2_positive_improved.png`
- `layer2_logratio_statistics_x0.4_positive_improved.png`
- `layer2_logratio_statistics_x0.6_positive_improved.png`
- `layer2_logratio_statistics_x0.8_positive_improved.png`
- `layer2_logratio_statistics_x1.0_positive_improved.png`

**Total**: 6 plots per configuration × number of factor=4, rank=15 configurations

## Next Steps

1. **Fix Layer 1 calculation**: Resolve NaN issue to generate Layer 1 plots
2. **Compute spatial variation**: Calculate matrices for each x value (not just x=0)
3. **Filter inactive channels**: Optionally filter $|f_k| < 10^{-6}$ for cleaner ratios
4. **Adjust epsilon**: Consider using $\epsilon = 10^{-6}$ instead of $10^{-10}$ for more interpretable results
