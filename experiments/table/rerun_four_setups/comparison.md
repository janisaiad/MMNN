# Low-rank vs full-rank loss comparison

**Setup.** 1D regression: *y* = cos(2π*x*) on *x* ∈ [-1, 1]. **Samples:** *n* = 5000 (train). **Batch size:** 4. **Model:** MMNN, L = 2 layers, width *W* = 1024. **Optimizer:** SGD lr = 0.01, AdaptiveStagnation. **Epochs:** low-rank 250, full-rank 1000. **Plot:** 0–800 epochs.

**Low-rank (R=10).** Four runs differing only by **η** (momentum): 0.0, 0.3, 0.6, 0.7. Red bars: LR reductions for η = 0.0.

**Full-rank (R=1024).** One run, η = 0.0, same optimizer and scheduler.

**Figure.** `four_setups_loss_with_lr_bars.png` overlays the four low-rank curves and the full-rank curve. Full rank reaches much lower loss sooner; low-rank models need more epochs and their convergence depends on momentum.
