# Mean-Field Analysis Plots Description

## Experimental Setup

**Target Function:** Multi-frequency cosine function
$$f(x) = \cos(12\pi x) + \cos(24\pi x + 0.5) + \cos(36\pi x) + \cos(72\pi x + 0.5)$$

**Architecture:** 2-layer mean-field neural network (approximating 8-layer network from config)
- Width: $n = 777$ neurons per layer
- Rank: $r = 15$ low-rank channels
- Training samples: $N = 5000$ points on $[-1, 1]$
- Time span: $t \in [0, 1000]$ (mean-field ODE evolution)

**Analysis Location:** $x \approx 0$ (using $x = 10^{-6}$ because $\mathrm{ReLU}(0) = 0$ makes partial functions zero at exact $x = 0$)

**Key Concept:** Channel specialization is measured via log-ratios $R_{i,j} = \log(|f_i|) - \log(|f_j|)$, where $f_k$ are the low-rank partial functions (before mixing). Large log-ratios indicate that different channels specialize to capture different features of the target function.

---

## Plot Descriptions

### 1. `meanfield_log_ratio_heatmap.png`
**What it shows:** Log-ratio matrix $R_{i,j}$ at final time $t = 1000$ for all channel pairs $(i, j)$.

**Interpretation:**
- Each entry $(i, j)$ shows $\log(|f_i|) - \log(|f_j|)$ at $x \approx 0$
- Red values indicate channel $i$ dominates over channel $j$ (positive log-ratio)
- Blue values indicate channel $j$ dominates over channel $i$ (negative log-ratio)
- White values indicate similar magnitudes
- The diagonal is zero by definition ($R_{i,i} = 0$)
- Strong off-diagonal patterns indicate channel specialization: different channels have different magnitudes at this location

**Key insight:** This heatmap reveals the specialization structure. If channels were identical, all off-diagonal entries would be near zero. Non-zero patterns show that channels have differentiated during training.

---

### 2. `meanfield_log_ratio_statistics_time.png`
**What it shows:** Evolution of log-ratio statistics over time: mean, max, min, and ±1 standard deviation.

**Interpretation:**
- **Mean log-ratio:** Average specialization across all channel pairs. Near zero indicates balanced channels; non-zero indicates systematic specialization.
- **Max log-ratio:** Maximum specialization between any two channels. Growing max indicates increasing differentiation.
- **Min log-ratio:** Minimum specialization (most negative). Large negative values indicate strong dominance of one channel over another.
- **Standard deviation:** Spread of log-ratios. Large std indicates heterogeneous specialization (some channels much stronger than others).

**Key insight:** The evolution shows how channel specialization develops over training time. If mean stays near zero but max/min diverge, it indicates that channels specialize in different ways rather than all becoming similar.

---

### 3. `meanfield_log_ratio_distribution.png`
**What it shows:** Histogram of all log-ratio values $R_{i,j}$ (for $i \neq j$) at final time $t = 1000$.

**Interpretation:**
- Distribution centered near zero: channels are balanced
- Distribution with wide spread: strong specialization (some channels much stronger/weaker than others)
- Skewed distribution: asymmetric specialization (more channels dominating than dominated, or vice versa)
- Bimodal distribution: clear separation into dominant and dominated channels

**Key insight:** The shape of this distribution reveals the specialization pattern. A narrow distribution indicates uniform channels, while a wide or multi-modal distribution indicates heterogeneous specialization.

---

### 4. `meanfield_log_ratio_distribution_time_evolution.png`
**What it shows:** Overlaid histograms of log-ratio distributions at different time points ($t = 0, 250, 500, 1000$).

**Interpretation:**
- Compare distributions across time to see how specialization evolves
- If distributions widen over time: channels are differentiating
- If distributions narrow: channels are converging
- Shifts in distribution center: systematic changes in channel balance

**Key insight:** This plot shows the dynamics of specialization. The evolution from initial to final distribution reveals whether specialization emerges gradually or suddenly, and whether it stabilizes or continues evolving.

---

### 5. `meanfield_channel_shares.png`
**What it shows:** Bar plot of channel shares $s_k = |f_k| / \sum_j |f_j|$ at final time, showing the relative contribution of each channel.

**Interpretation:**
- Each bar shows the share of channel $k$ in the total magnitude
- Uniform bars (all $\approx 1/r$): balanced channels, no specialization
- Uneven bars: specialization - some channels contribute more than others
- Dominant channels (high bars): channels that have specialized to capture specific features
- Weak channels (low bars): channels that have become less active

**Key insight:** This directly shows which channels are dominant. If all channels had equal shares ($1/15 \approx 0.067$), there would be no specialization. Uneven shares indicate that certain channels have specialized to be more active at this location.

---

### 6. `meanfield_weight_density_normal.png`
**What it shows:** Histogram of all weight values $w$ (from both $w_1$ and $w_2$ layers) at final time, in normal scale.

**Interpretation:**
- Distribution shape reveals weight statistics
- Centered at zero: weights initialized and evolved around zero
- Gaussian-like: typical for mean-field initialization
- Skewed: asymmetric weight evolution
- Multi-modal: different weight populations (e.g., active vs inactive neurons)

**Key insight:** Weight distributions reflect the learned representation. The shape and spread indicate how weights have evolved from their initial Gaussian distribution during training.

---

### 7. `meanfield_weight_density_loglog.png`
**What it shows:** Weight density in log-log scale, showing the power-law behavior of absolute weights $|w|$.

**Interpretation:**
- Linear relationship in log-log: power-law distribution $P(|w|) \propto |w|^{-\alpha}$
- Slope indicates the exponent $\alpha$
- Deviations from linearity: deviations from pure power-law (e.g., cutoffs, exponential tails)
- Power-law behavior: characteristic of scale-invariant systems, often observed in trained neural networks

**Key insight:** Power-law weight distributions are a signature of criticality and scale-invariance in neural networks. This plot reveals whether the mean-field dynamics lead to such critical behavior.

---

### 8. `meanfield_weight_density_time_evolution.png`
**What it shows:** Overlaid histograms of weight distributions at different time points ($t = 0, 250, 500, 1000$).

**Interpretation:**
- Compare weight distributions across time to see evolution
- Shifts in distribution: systematic changes in weight magnitudes
- Widening/narrowing: increasing/decreasing weight diversity
- Shape changes: transitions in weight statistics (e.g., from Gaussian to heavy-tailed)

**Key insight:** This shows how the weight distribution evolves during training. The mean-field ODE dynamics determine how weights change, and this plot reveals whether the distribution stabilizes, spreads, or develops structure over time.

---

## Summary of Key Findings

1. **Channel Specialization:** The log-ratio plots (heatmap, statistics, distributions) show that channels do specialize during mean-field training, with non-zero log-ratios indicating differentiation between channels.

2. **Temporal Evolution:** The time-evolution plots reveal that specialization develops gradually over time, with log-ratios and weight distributions evolving from initial conditions toward specialized states.

3. **Heterogeneous Specialization:** The distribution plots show that specialization is heterogeneous - not all channels specialize equally, with some becoming dominant and others remaining weak.

4. **Weight Statistics:** The weight density plots show how the weight distribution evolves, potentially revealing power-law behavior or other statistical signatures of the mean-field dynamics.

5. **Location-Specific Analysis:** All analysis is performed at $x \approx 0$, revealing how channels specialize at this specific location. Different locations may show different specialization patterns.

---

## How to Use These Plots in the Paper

1. **Introduction/Motivation:** Use the heatmap and channel shares to illustrate what channel specialization looks like.

2. **Main Results:** Use the statistics over time and distribution evolution to show how specialization develops during training.

3. **Theoretical Analysis:** Use the weight density plots to connect with theoretical predictions about weight statistics in mean-field dynamics.

4. **Comparison with Theory:** Compare the observed log-ratio distributions and weight statistics with theoretical predictions from mean-field theory.

5. **Discussion:** Use the time-evolution plots to discuss the dynamics of specialization and how it relates to the target function structure.

---

## Technical Notes

- All plots use LaTeX formatting for mathematical notation
- Architecture details are provided in the bottom text of each plot
- The analysis location $x \approx 0$ is used because exact $x = 0$ gives zero partial functions due to $\mathrm{ReLU}(0) = 0$
- The 2-layer mean-field approximation is used to analyze an 8-layer network configuration
- All plots are saved at 300 DPI for publication quality
