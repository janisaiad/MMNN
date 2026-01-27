# Full-rank network component plotting

## Where component plotting lives

**Script:** `experiments/table/plot_frequency_partials.py`

- Plots layer-wise partial functions (components 1 and 2, i.e. \(h_1(x)\), \(h_2(x)\)) for frequency-benchmark models.
- Writes `partials/layer_{N}_components.png` inside each config directory.
- Requires `model_parameters.pth` or `checkpoint.pth` in each config dir to load the model.

## Current setup: full-rank **excluded**

`plot_frequency_partials.py` only searches:

- `experiments/table/results_frequency_benchmark_comprehensive`
- `experiments/table/results_frequency_benchmark`
- `results_frequency_benchmark_comprehensive` (relative to script)
- `results_frequency_benchmark` (relative to script)

It **does not** search `results_frequency_benchmark_full_rank`.

## Full-rank results location

- **Path:** `experiments/table/experiments/table/results_frequency_benchmark_full_rank/`
- **Contents:** `config.json`, `results.json`, `loss_evolution.png`, `final_prediction.png`, `error_evolution.png` per config.
- **Missing:** No `model_parameters.pth`, no `checkpoint.pth`, and no `partials/` subdirs.

So **full-rank component plots do not exist** in the current repo: the plotting script never looks at full-rank runs, and those runs have no saved model state to plot from.

## How to generate full-rank component plots

1. **Add full-rank to the plot script**  
   Include `results_frequency_benchmark_full_rank` (and the nested `experiments/table/experiments/table/results_frequency_benchmark_full_rank` if you use that) in the `candidates` list in `plot_frequency_partials.py` (see `main()`).

2. **Ensure full-rank runs save model state**  
   `test_frequency_benchmark.py` saves `checkpoint.pth` every 500 epochs. Use runs that actually wrote checkpoints, or re-run full-rank training with checkpointing.

3. **Run the plot script**  
   From project root:
   ```bash
   uv run python experiments/table/plot_frequency_partials.py
   ```

4. **Output location**  
   Plots will appear under  
   `.../results_frequency_benchmark_full_rank/<config_name>/partials/layer_{N}_components.png`.

## Summary

| Item | Status |
|------|--------|
| Component plotting script | `plot_frequency_partials.py` |
| Includes full-rank dirs? | **No** |
| Full-rank model checkpoints | **None** in current full-rank result dirs |
| Full-rank component plots on disk | **None** |

To get full-rank component plots: add full-rank result dirs to the script, run full-rank training with checkpointing (if needed), then run `plot_frequency_partials.py`.
