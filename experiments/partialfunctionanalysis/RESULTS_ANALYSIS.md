## What we computed

We analyze 1D **partial functions** extracted exactly like `experiments/former/SinQuad/leap.py`: for each saved model (`model_parameters.pth`) we evaluate the network on a uniform grid \(x \in [-1,1]\) and record the intermediate outputs after each `fcs[i]` (with the same ReLU placement as in `leap.py`).

For any chosen layer output \(h(x)\in\mathbb{R}^d\), we compute, **per component** \(h_j(x)\), the number of **strict local minima** on the grid subject to a positivity threshold:

- **strict local minimum**: \(h_j(x_i) < h_j(x_{i-1})\) and \(h_j(x_i) < h_j(x_{i+1})\)
- **positivity filter**: \(h_j(x_i) > 10^{-4}\) (to avoid ReLU zero plateaus)

The main statistic plotted is:

- **mean minima per component** at each layer index: \(\frac{1}{d}\sum_{j=1}^d \#\text{minima}(h_j)\)

The distributions are shown via:

- **boxplots** across components at each layer index
- **histograms** of minima counts across components (per layer, and pooled)

Important: **means are not rounded** (stored/aggregated as floats).


## Key outputs (high-signal)

### Single-config (paper-ready) example

This is the figure you pointed to as the main paper plot (L=6, W=128, R=20):

- `experiments/partialfunctionanalysis/output_leap_all_L6/leap_all_layers/mmnn_L6_W128_R20_E3000_lr0.001_bs100_ntr1000/mean_minima_per_component_vs_layer.png`

Companion “spread across components” plot:

- `experiments/partialfunctionanalysis/output_leap_all_L6/leap_all_layers/mmnn_L6_W128_R20_E3000_lr0.001_bs100_ntr1000/boxplot_minima_per_component_vs_layer.png`

### Multi-config aggregate (global “across runs”)

Aggregated across the 7 configs with fixed \(L=6, W=128\) and varying \(R \in \{5,10,15,20,30,36,50\}\):

- `experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped/leap_all_layers_aggregate/mean_minima_per_component_vs_layer_across_runs.png`
- `experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped/leap_all_layers_aggregate/hist_minima_pooled_all_runs.png`

### Comparative study by rank \(R\)

Per-group plots and pooled/per-layer histograms are in:

- `experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped/grouped/`

The main machine-readable summary for tables/plots:

- `experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped/grouped/group_layer_stats.csv`
- `experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped/grouped/group_summary.json`


## What the results suggest

### Depth effect (strong)

Across layers, the **mean minima per component increases substantially in late layers**, and the **boxplots widen**. This is consistent with the qualitative claim:

- **deeper layers learn more oscillatory (higher-frequency) partial functions** (in this discrete “minima count” proxy).

### Rank \(R\) effect (currently not clean / not monotone)

At fixed \(L=6, W=128\), different \(R\) values produce noticeably different curves, but **the dependence is not monotone in \(R\)** in the current sweep.

Example from `group_layer_stats.csv` at **layer 12** (mean minima per component):

- \(R=5\): ~8.6  
- \(R=10\): ~3.1  
- \(R=15\): ~2.33  
- \(R=20\): ~3.4  
- \(R=30\): ~5.17  
- \(R=36\): ~3.08  
- \(R=50\): ~4.44  

So: **\(R\) matters**, but the pattern is “messy” with one run per \(R\).


## Caveats (important before writing a strong claim)

- **No replication over seeds**: each \((L,W,R)\) group currently has **one run**, so rank effects can reflect training randomness/optimization path.
- **Grid resolution**: counts depend on `grid_size` (default used here: 1000). If components oscillate faster than the grid, minima can be under-counted.
- **Last leap layer has small output dimension**: in this architecture, the final `fcs` outputs can have dimension 1, so “mean over components” becomes a single curve’s count (treat this separately).


## How to reproduce

### Single-config (example)

```bash
python /Data/janis.aiad/MMNN/experiments/partialfunctionanalysis/positive_local_minima_analysis.py \
  --run_dir /Data/janis.aiad/mmnn_training/mmnn_L6_W128_R20_E3000_lr0.001_bs100_ntr1000 \
  --out_dir /Data/janis.aiad/MMNN/experiments/partialfunctionanalysis/output_leap_all_L6 \
  --grid_size 1000 --value_threshold 1e-4 \
  --leap_all_layers --hist_bins 50 --device cpu
```

### Multi-config + grouping by rank (example sweep)

```bash
python /Data/janis.aiad/MMNN/experiments/partialfunctionanalysis/positive_local_minima_analysis.py \
  --runs_root /Data/janis.aiad/mmnn_training \
  --out_dir /Data/janis.aiad/MMNN/experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped \
  --grid_size 1000 --value_threshold 1e-4 \
  --only_num_layers 6 --only_hidden_width 128 \
  --max_hidden_rank 0 --max_hidden_width 0 \
  --leap_all_layers --hist_bins 60 \
  --group_by L,R --group_hist_mode pooled \
  --device cpu
```


## Suggested next steps (to strengthen the paper claim)

- **Add multiple seeds** per \(R\) (and per \(L\)) and re-run grouping; then report error bars (std/quantiles) for rank effects.
- Optionally complement minima counts with a **Fourier-based** metric (spectral centroid / rolloff) if you want a more direct “high frequency” proxy.

