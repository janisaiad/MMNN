# Exhaustive resume: partial-function oscillatory complexity (quantitative and qualitative)

Summary of all results from the partial-function analysis discussion: MMNN 1D regression, strictly-positive local minima per component across layers, with single-config and multi-config (L6, W128, varying R) experiments.

---

## 1. Experimental setup (fixed across analyses)

| Parameter | Value |
|-----------|--------|
| **Grid** | \(x \in [-1,1]\), \(N = 1000\) points |
| **Positivity threshold** \(\varepsilon\) | \(10^{-4}\) |
| **Definition of minimum** | Strict local minimum: \(y_i < y_{i-1}\), \(y_i < y_{i+1}\), \(y_i > \varepsilon\) |
| **Statistic** | \(\bar{m}_\ell = \frac{1}{d_\ell}\sum_j m_{\ell,j}\) (mean minima per component at layer \(\ell\)) |
| **Architecture** | MMNN: \(L=6\), \(W=128\), \(R \in \{5, 10, 15, 20, 30, 36, 50\}\) |
| **Layers** | 14 “leap” layers (linear blocks; last layer index 14 has \(d_{14}=1\)) |
| **Runs per group** | 1 run per \((L,R)\) (no seed replication) |
| **Aggregate** | 7 runs (one per R) used for global mean curve and pooled histograms |

---

## 2. Quantitative results

### 2.1 Mean minima per component by group and layer (full table)

From `group_layer_stats.csv`. Each cell: **mean** (std in parentheses where useful). \(n\) = number of components at that layer (128 for most hidden layers, then rank dimension for “bottleneck” layers, 1 for layer 14).

**Early layers (1–4):** mean \(\bar{m}_\ell\) is 0 or very small (\(\leq 0.35\)) for all groups.

| group   | layer 1 | layer 2 | layer 3   | layer 4   | layer 5   | layer 6   | layer 7   |
|---------|---------|---------|-----------|-----------|-----------|-----------|-----------|
| L6_R5   | 0.00    | 0.00    | 0.039     | 0.00      | 0.344     | 0.80      | 0.672     |
| L6_R10  | 0.00    | 0.00    | 0.094     | 0.10      | 0.273     | 0.60      | 0.734     |
| L6_R15  | 0.00    | 0.133    | 0.117     | 0.067     | 0.336     | 0.20      | 0.406     |
| L6_R20  | 0.00    | 0.10     | 0.117     | 0.35      | 0.312     | 1.10      | 0.523     |
| L6_R30  | 0.00    | 0.133    | 0.125     | 0.20      | 0.320     | 0.40      | 0.297     |
| L6_R36  | 0.00    | 0.083     | 0.281     | 0.139     | 0.320     | 0.361     | 0.25      |
| L6_R50  | 0.00    | 0.06      | 0.055     | 0.24      | 0.219     | 0.54      | 0.297     |

| group   | layer 8   | layer 9   | layer 10  | layer 11  | layer 12  | layer 13  | layer 14 (output) |
|---------|-----------|-----------|------------|------------|------------|------------|-------------------|
| L6_R5   | 2.40      | 2.08      | 2.40       | 4.22       | **8.60**   | 6.51       | **11.0**          |
| L6_R10  | 1.50      | 1.45      | 1.60       | 1.97       | **3.10**   | 3.03       | **7.0**           |
| L6_R15  | 1.27      | 1.17      | 1.93       | 2.36       | **2.33**   | 3.80       | **5.0**           |
| L6_R20  | 0.85      | 0.70      | 1.20       | 1.70       | **3.40**   | 3.15       | **3.0**           |
| L6_R30  | 0.87      | 0.98      | 2.03       | 2.40       | **5.17**   | 4.02       | **3.0**           |
| L6_R36  | 0.42      | 0.55      | 1.64       | 1.38       | **3.08**   | 3.34       | **2.0**           |
| L6_R50  | 0.74      | 0.66      | 1.48       | 1.38       | **4.44**   | 3.59       | **8.0**           |

All values above are **unrounded** (floats) as in the CSV.

### 2.2 Rank effect at a fixed late layer (layer 12)

Mean minima per component at **layer 12** (representative late layer):

| \(R\) | \(\bar{m}_{12}\) | std (approx) |
|-------|-------------------|---------------|
| 5     | **8.60**          | 6.05          |
| 10    | 3.10             | 3.70          |
| 15    | 2.33             | 3.16          |
| 20    | 3.40             | 3.85          |
| 30    | 5.17             | 6.11          |
| 36    | 3.08             | 3.78          |
| 50    | 4.44             | 4.37          |

**Quantitative conclusion:** At layer 12, \(\bar{m}_\ell\) ranges from **2.33** (R=15) to **8.60** (R=5). The dependence on \(R\) is **not monotonic**: it does not increase or decrease consistently with \(R\).

### 2.3 Final layer (layer 14, scalar output)

Layer 14 has a single component (\(n=1\)), so the “mean” is just that component’s count:

| \(R\) | minima count (layer 14) |
|-------|--------------------------|
| 5     | 11                       |
| 10    | 7                        |
| 15    | 5                        |
| 20    | 3                        |
| 30    | 3                        |
| 36    | 2                        |
| 50    | 8                        |

Again, **no monotonic trend in \(R\)**; R=5 and R=50 give the highest final-layer counts in this single-run sample.

### 2.4 Spread across components (quantiles at layer 12)

From the same CSV (q25, q50, q75 at layer 12):

| group   | q25 | q50 | q75 |
|---------|-----|-----|-----|
| L6_R5   | 2.0 | 10.0| 11.0|
| L6_R10  | 0.25| 1.5 | 3.75|
| L6_R15  | 0.0 | 1.0 | 2.5 |
| L6_R20  | 0.0 | 2.5 | 5.0 |
| L6_R30  | 0.0 | 3.5 | 6.75|
| L6_R36  | 0.0 | 2.0 | 4.0 |
| L6_R50  | 1.0 | 3.0 | 6.0 |

So at layer 12 the distribution of minima counts across components is **right-skewed** and **heterogeneous** (e.g. L6_R5: median 10, q75 11; L6_R20: median 2.5, q75 5).

### 2.5 Aggregate over the seven runs

- **Number of runs:** 7 (one per \(R \in \{5,10,15,20,30,36,50\}\)).
- **Common layers:** 14.
- **Global mean curve:** Average of \(\bar{m}_\ell\) across these 7 runs at each \(\ell\): low in early layers, **sharp rise in the second half** (layers ~7–13), matching the per-group depth effect.

No single scalar “aggregate mean at layer 12” is stored in the JSON; the aggregate is used for the **mean curve vs layer** plot and the **pooled histogram** over all runs/layers/components.

---

## 3. Qualitative results

### 3.1 Depth effect (strong, consistent)

- **Observation:** In **every** run and every group, \(\bar{m}_\ell\) is **near zero in early layers** (1–4) and **increases in later layers** (roughly 7–13).
- **Interpretation:** Deeper layers exhibit **more oscillatory** partial functions in the sense of this proxy (more strictly-positive local minima on the grid).
- **Robustness:** Holds for all seven \(R\) values; the **aggregate curve** (average over the 7 runs) has the same shape.
- **Spread:** Boxplots show **widening and right-skewed** distributions in late layers: many components with few minima, some with many.

### 3.2 Rank effect (present, not monotonic)

- **Observation:** At fixed \(L=6\), \(W=128\), the curve \(\bar{m}_\ell\) vs \(\ell\) **depends on \(R\)**, but there is **no monotonic relation** (e.g. “higher \(R\) ⇒ higher \(\bar{m}_\ell\)” or the reverse) in the current data.
- **Example:** At layer 12, \(\bar{m}_\ell\) is highest for R=5 (8.6) and lowest for R=15 (2.33), with R=20, R=36, R=10 in between and R=30, R=50 again higher.
- **Interpretation:** **Rank influences internal oscillatory complexity**, but with **one run per \(R\)** the pattern may reflect optimization/training noise; no clear functional form (e.g. linear or unimodal in \(R\)) can be claimed.

### 3.3 Heterogeneity across components

- **Observation:** At a given layer, the number of minima **varies strongly across components** (large std, q25 \(\ll\) q75 in late layers).
- **Interpretation:** Some channels develop highly oscillatory 1D partial functions, others stay flatter; the **mean** \(\bar{m}_\ell\) summarizes the layer but does not capture the full distribution (hence boxplots and histograms are reported).

### 3.4 Last layer (layer 14)

- **Observation:** Layer 14 has a single output dimension; the “mean” is just one scalar count (between 2 and 11 in the seven runs).
- **Interpretation:** This layer should be **interpreted separately** from the rest; it is the final map to the 1D target, not a multi-component representation layer.

---

## 4. Limitations (qualitative caveats)

1. **No seed replication:** Each \((L,R)\) has one run; rank effects could be confounded by optimization noise.
2. **Grid resolution:** \(N=1000\); faster oscillations can lead to **under-counted** minima.
3. **Last layer:** Scalar output; “mean over components” is not comparable to earlier layers.
4. **Task and scale:** Conclusions apply to the **specific** 1D regression setup and \(L,W,R\) range; generalization would need further experiments.

---

## 5. Outputs and paper figure

- **Figure chosen for the paper:** Mean minima per component vs layer for **L6_R5** (\(W=128\)).  
  Path: `experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped/grouped/L6_R5/mean_minima_per_component_vs_layer.png`  
  Referred to as **Figure 1** in the paper subsection.
- **Other key outputs:** Single-config (e.g. L6_W128_R20) mean/boxplot; aggregate mean curve and pooled histogram; per-group mean curves and histograms (all under `output_leap_all_L6_W128_grouped`); machine-readable `group_layer_stats.csv` and `group_summary.json`.

---

## 6. Suggested follow-ups (qualitative)

- **Multiple seeds** per \((L,R)\) and report means ± std or confidence intervals for rank effects.
- **Fourier-based metrics** (e.g. spectral centroid, 95% rolloff) to complement minima counts.
- **Broader sweep** over \(L\) and \(W\) to map the \((L,W,R)\) landscape.

---

## 7. One-sentence quantitative vs qualitative summary

- **Quantitative:** For MMNN with \(L=6\), \(W=128\), \(R \in \{5,10,15,20,30,36,50\}\), the mean number of strictly-positive local minima per component \(\bar{m}_\ell\) is \(\leq 0.35\) for layers 1–4, rises to roughly **2–9 at layer 12** and **2–11 at layer 14** depending on \(R\), with no monotonic trend in \(R\) and substantial spread across components (right-skewed, layer-dependent).
- **Qualitative:** **Depth effect:** deeper layers have higher oscillatory complexity (higher \(\bar{m}_\ell\)) in all runs; **rank effect:** \(R\) changes the curves but not in a monotonic way with one run per \(R\); **heterogeneity:** strong component-to-component variation in late layers; **caveats:** no seed replication, grid resolution, last layer scalar, task-specific scope.
