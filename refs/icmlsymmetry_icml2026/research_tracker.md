# ICML Symmetry Workshop Research Tracker

## Goal

Build a workshop paper around the claim that MMNN/RF-LR models learn symmetric partial functions for even 1D targets, and that this symmetry is visible in weight space through mirror-pair structure. The immediate starting point is the ICLR landscape figure:

`refs/ICLR/landscape.tex`, Figure 5: partial functions before/after LR divides for `f3`, `N = 768`, `bs = 8`, depth `L = 3`.

## Files Already Identified

- `refs/ICLR/merged.tex`: merged ICLR draft with the complete script/config index.
- `refs/ICLR/landscape.tex`: source text for the loss-landscape paper and Figure 5 reference.
- `experiments/table/run_selected_sumcos_configs.py`: reruns selected sumcos configs and saves LR-divide checkpoints.
- `experiments/table/plot_selected_sumcos_configs.py`: generates loss/fit before-after LR divide figures.
- `experiments/table/plot_partial_f3_N768_bs8_L3.py`: generates partial-function before-after LR divide figures.
- `experiments/table/results_sumcos_selected_rerun/f3_N768_bs8_L3/plot_partial_before_after_lr_divides.png`: current Figure 5 source image.
- `experiments/table/results_sumcos_selected_rerun/f3_N768_bs8_L3/plot_loss_and_fit_before_after_lr_divides.png`: paired loss/fit image.
- `experiments/table/plateau/run_plateau_escape.py`: plateau escape and log-ratio experiments.
- `experiments/posticml/symmetry_rank_sweep.py`: existing output-level symmetry-defect sweep for MMNN.

## Current Observation

The existing `results_sumcos_selected_rerun` directories contain `config.json`, `losses.json`, and the generated PNG files. The LR-divide checkpoint folders used by the plotting scripts are not present in the current local copy, so the original Figure 5 can be cited and reused, but regenerating it exactly requires rerunning `run_selected_sumcos_configs.py`.

## Core Hypotheses

1. Even sumcos targets induce symmetric output functions, but symmetry of internal partial functions is architecture-dependent.
2. MMNN/RF-LR should show lower partial-function symmetry defect than a full-rank MLP trained on the same even target.
3. For MMNN/RF-LR, symmetry should be encoded in the first frozen feature layer plus trainable outgoing weights: ReLU atoms with mirrored slopes should receive similar outgoing coefficients.
4. LR divides should correspond to sharper basin entry, and partial functions should become more symmetric and more oscillatory after each divide.
5. Symmetry breaking may happen in landscape valleys for full-rank networks: output can remain approximately even while hidden features become asymmetric.

## Metrics

- Output symmetry defect: $E_x[(f(x)-f(-x))^2] / E_x[f(x)^2]$.
- Partial symmetry defect: same metric per bottleneck channel or hidden feature layer.
- Mirror-pair weight mismatch: for first-layer atoms $(a_j,b_j)$ and mirror atoms $(-a_j,b_j)$, compare outgoing coefficients.
- Oscillatory complexity: mean number of strict positive local minima per channel.
- Plateau event metrics: loss drop, partial symmetry defect change, minima-count change before and after LR divide.

## First Automatic Experiment

Script:

`refs/icmlsymmetry/symmetry_weightspace_experiment.py`

Outputs:

- `refs/icmlsymmetry/results/symmetry_weightspace/metrics.json`
- `refs/icmlsymmetry/results/symmetry_weightspace/metrics.csv`
- `refs/icmlsymmetry/results/symmetry_weightspace/symmetry_defects_bar.png`
- `refs/icmlsymmetry/results/symmetry_weightspace/mirror_pair_weightspace.png`
- `refs/icmlsymmetry/results/symmetry_weightspace/partial_symmetry_heatmap_mmnn_seed*.png`
- `refs/icmlsymmetry/results/symmetry_weightspace/partial_symmetry_heatmap_mlp_seed*.png`

Quick run:

```bash
python refs/icmlsymmetry/symmetry_weightspace_experiment.py --quick
```

Fuller run:

```bash
python refs/icmlsymmetry/symmetry_weightspace_experiment.py --seeds 42 43 44 --epochs 1200 --width 512 --rank 10 --depth 3 --n-train 768 --batch-size 8
```

## First Results Generated

Command run:

```bash
python refs/icmlsymmetry/symmetry_weightspace_experiment.py --seeds 42 43 --epochs 600 --width 256 --rank 10 --depth 3 --n-train 384 --batch-size 8
```

Results are in `refs/icmlsymmetry/results/symmetry_weightspace/`.

Key observations:

- Seed 42: MMNN final train MSE $9.52e-2$, output defect $2.00e-3$, mean partial defect $4.97e-1$; MLP final train MSE $1.09e-1$, output defect $4.53e-4$, mean partial defect $1.07$.
- Seed 43: MMNN final train MSE $1.07e-1$, output defect $4.21e-3$, mean partial defect $8.51e-1$; MLP final train MSE $1.05e-1$, output defect $3.14e-3$, mean partial defect $1.29$.
- The strongest signal is layerwise: MMNN last bottleneck defects are $3.28e-3$ and $5.76e-3$, while MLP last hidden-layer defects are $1.26$ and $1.37$.
- Interpretation: output symmetry alone is not enough because both models can produce nearly even outputs. The paper should emphasize partial-function symmetry and weight-space diagnostics.
- Mirror-pair weight mismatch is lower for MMNN than MLP in this quick run, but the gap is moderate. This metric should be strengthened by explicit paired random-feature initialization or larger width.

Immediate paper claim supported by this quick run:

The low-rank architecture can drive late bottleneck partials toward symmetry even when early random-feature layers are not individually symmetric. Full-rank MLPs can fit the even target while keeping asymmetric hidden features.

## Paper Structure

1. Introduction: low-rank weight-space symmetry as a mechanism for symmetric partial functions.
2. Setup: even 1D targets, MMNN/RF-LR, full-rank MLP baseline, partial functions.
3. Figure 1: before/after LR divides from ICLR Figure 5.
4. Figure 2: partial symmetry defect across MMNN vs full-rank.
5. Figure 3: mirror-pair weight-space encoding.
6. Figure 4: symmetry defects around LR divides after rerunning checkpointed experiments.
7. Discussion: symmetry preservation, symmetry breaking in valleys, link to plateau-sharp-plateau landscape.

## Next Experiments

- Rerun `f3_N768_bs8_L3` with checkpoints to recompute Figure 5 and add symmetry metrics before/after each LR divide.
- Add rank sweep `r in {5,10,20,32}` for MMNN and compare mirror mismatch.
- Add full-rank MLP seeds with Adam and SGD to test whether valleys produce asymmetric hidden features.
- Add layerwise minima counts before/after LR divides and correlate with symmetry defect.
- Add RF-LR-specific experiment where frozen random features are explicitly paired, then compare paired versus unpaired initialization.

## Long Grid Run: Function-Space and Weight-Space Symmetry

Scripts:

- `refs/icmlsymmetry/symmetry_grid_long.py`
- `refs/icmlsymmetry/analyze_symmetry_grid.py`

Command run:

```bash
python refs/icmlsymmetry/symmetry_grid_long.py --seeds 42,43 --widths 384 --ranks 5,10,20 --depths 2,3 --n-train 512 --batch-size 8 --epochs 2000 --models mmnn mlp
python refs/icmlsymmetry/analyze_symmetry_grid.py
```

Outputs:

- `refs/icmlsymmetry/results/symmetry_grid_long/summary.csv`
- `refs/icmlsymmetry/results/symmetry_grid_long/active_summary.csv`
- `refs/icmlsymmetry/results/symmetry_grid_long/active_output_vs_partial_symmetry.png`
- `refs/icmlsymmetry/results/symmetry_grid_long/layerwise_partial_symmetry_heatmap.png`
- `refs/icmlsymmetry/results/symmetry_grid_long/active_weightspace_vs_partial_symmetry.png`
- `refs/icmlsymmetry/results/symmetry_grid_long/active_last_layer_even_distribution.png`
- Per-run: `loss_curve.png`, `fit_target_prediction.png`, `partial_even_defect_distribution.png`, `weightspace_distribution.png`, `layerwise_partial_even_defect.png`, `layerwise_minima.png`, and raw `distributions.npz`.

Why active-channel metrics are needed:

Some MLP hidden channels have very small energy, so a raw normalized defect can explode due to a tiny denominator. The robust post-processing keeps channels above the median energy in each layer and reports active-channel symmetry defects. This preserves the important signal while avoiding dead-channel artifacts.

Main findings from 16 configs, each trained for 2000 epochs:

- Output-space symmetry is not the discriminating metric. MLP outputs can be very even, e.g. output defects around $1.9e-4$ to $7.1e-3$.
- Internal partial functions separate the models. Active last-layer MLP defects range from about $2.9$ to $15.4$.
- MMNN depth 3 gives much more symmetric late bottleneck partials. Active last-layer defects include $1.16e-3$, $5.70e-3$, $8.91e-3$, $3.94e-3$, $4.09e-3$, and $1.96e-2$ across seeds/ranks.
- Depth matters: depth 3 MMNNs have substantially lower last-layer partial defects than depth 2 in most configs.
- Weight-space mirror metrics show moderate but real structure: MMNN best-20% mirror mismatch often lies around $0.70$ to $0.95$, while MLP is around $0.95$ to $1.01$. The gap is weaker than the function-space gap, suggesting that the current nearest-mirror metric is useful but not yet the whole story.
- Oscillatory-complexity metrics are saved per channel; current mean minima values remain small in this width/rank/epoch regime but are available for layerwise correlation.

Current interpretation:

The cleanest paper claim is now: full-rank MLPs can learn an even output while using asymmetric active hidden features, whereas MMNNs, especially at depth 3, learn late bottleneck partial functions that are much closer to even. Weight-space mirror-pair statistics move in the expected direction but need a stronger paired-feature experiment to make a sharper claim.

## MNIST Practical Batch-Size Ablation

Script:

- `refs/icmlsymmetry/mnist_batch_symmetry_ablation.py`

Command run:

```bash
python refs/icmlsymmetry/mnist_batch_symmetry_ablation.py --models mlp mmnn --seeds 42 --batch-sizes 1,8,64,512,full --epochs 20 --lr 0.05 --width 128 --rank 10,25 --depth 2 --train-subset 2000 --test-subset 1000 --overwrite
```

Outputs:

- `refs/icmlsymmetry/results/mnist_batch_symmetry/summary.csv`
- `refs/icmlsymmetry/results/mnist_batch_symmetry/mnist_accuracy_vs_batch.png`
- `refs/icmlsymmetry/results/mnist_batch_symmetry/mnist_partial_defect_vs_batch.png`
- `refs/icmlsymmetry/results/mnist_batch_symmetry/mnist_output_vs_partial_defect.png`
- `refs/icmlsymmetry/results/mnist_batch_symmetry/mnist_weightspace_vs_partial_defect.png`
- Per run: `loss_curve.png`, `input_transform_defects.png`, `metrics.json`, `history.json`.

Setup:

- MNIST subset: 2000 train, 1000 test.
- SGD, lr $0.05$, momentum $0$, 20 epochs.
- MLP: width 128, depth 2, 118282 trainable params.
- MMNN: width 128, depth 2, ranks 10 and 25, fixWb=True with 2580 and 4515 trainable params.
- Batch sizes: 1, 8, 64, 512, full batch.
- Symmetry/proxy transformations: horizontal flip, vertical flip, 180-degree rotation, one-pixel right shift, one-pixel down shift.

Main findings:

- Batch size is the dominant practical factor. Small batches learn; large/full batches stall, especially for MMNN.
- MLP test accuracy: batch 1 $90.3\%$, batch 8 $89.9\%$, batch 64 $89.3\%$, batch 512 $84.3\%$, full batch $64.2\%$.
- MMNN rank 10: batch 1 $83.9\%$, batch 8 $83.5\%$, batch 64 $72.0\%$, batch 512 $21.3\%$, full batch $10.5\%$.
- MMNN rank 25: batch 1 $84.2\%$, batch 8 $83.1\%$, batch 64 $71.8\%$, batch 512 $29.7\%$, full batch $19.7\%$.
- Internal transform-defect metrics are not direct symmetry claims for MNIST because labels are not invariant under all tested transforms. They are useful as representation-stability diagnostics.
- Full-batch MMNN has low logit transform defect partly because it barely learns and produces weak/near-constant logits. Accuracy must always be interpreted alongside symmetry metrics.
- First-layer image-filter symmetry metrics are weak in this unstructured MNIST setting. The weight-space symmetry claim remains strongest for controlled even targets, not raw MNIST.

Paper interpretation:

MNIST supports the optimizer/landscape part of the story more than the geometric symmetry part. It shows that RF-LR/MMNN models are highly batch-size sensitive: stochasticity is important for learning useful representations. This is consistent with the plateau/basin picture and motivates reporting symmetry metrics only together with task performance.

### Adam Follow-Up For Failed Large-Batch MNIST

Concern: some SGD losses were not low enough, especially for large/full batch. We therefore reran Adam on the failing regimes.

Scripts:

- `refs/icmlsymmetry/mnist_batch_symmetry_ablation.py` now supports `--optimizer sgd|adam`.
- `refs/icmlsymmetry/analyze_mnist_optimizer_ablation.py` merges SGD and Adam runs.

Command run:

```bash
python refs/icmlsymmetry/mnist_batch_symmetry_ablation.py --models mlp mmnn --optimizer adam --seeds 42 --batch-sizes 512,full --epochs 20 --lr 0.001 --width 128 --rank 10,25 --depth 2 --train-subset 2000 --test-subset 1000 --overwrite
python refs/icmlsymmetry/analyze_mnist_optimizer_ablation.py
```

Outputs:

- `refs/icmlsymmetry/results/mnist_batch_symmetry/merged_optimizer_summary.csv`
- `refs/icmlsymmetry/results/mnist_batch_symmetry/optimizer_accuracy_vs_batch.png`
- `refs/icmlsymmetry/results/mnist_batch_symmetry/optimizer_loss_vs_batch.png`
- `refs/icmlsymmetry/results/mnist_batch_symmetry/optimizer_loss_vs_partial_defect.png`

Key comparisons:

- MLP batch 512: SGD acc $0.843$, loss $0.585$; Adam acc $0.890$, loss $0.355$.
- MLP full batch: SGD acc $0.642$, loss $1.968$; Adam acc $0.842$, loss $0.521$.
- MMNN rank 10 batch 512: SGD acc $0.213$, loss $2.255$; Adam acc $0.505$, loss $1.658$.
- MMNN rank 25 batch 512: SGD acc $0.297$, loss $2.256$; Adam acc $0.671$, loss $1.525$.
- MMNN rank 10 full batch: SGD acc $0.105$, loss $2.295$; Adam acc $0.258$, loss $2.227$.
- MMNN rank 25 full batch: SGD acc $0.197$, loss $2.288$; Adam acc $0.330$, loss $2.194$.

Conclusion:

Adam fixes large-batch MLP and partially rescues large-batch MMNN, but full-batch MMNN remains near chance in this 20-epoch setup. Therefore the previous low logit-defect values in failed full-batch MMNN are not evidence of useful symmetry; they mostly reflect weak or undertrained logits. For MNIST, the reliable claim is optimization sensitivity: small-batch SGD works best for MMNN, Adam helps when batch is large, and full-batch remains a bad regime unless we change more factors such as learning rate schedule, epochs, rank, or unfreezing.

## Low-Loss ICLR Configs Re-Run And Analyzed

Motivation:

The MNIST ablations do not reach the very small losses of the ICLR sumcos configs, so they should not be the core evidence for symmetry. We therefore reran the actual low-loss sumcos configs from `refs/ICLR/merged.tex` and the baseline sweep tables, with checkpoints saved.

Script:

- `refs/icmlsymmetry/good_sumcos_low_loss.py`

Command run:

```bash
python refs/icmlsymmetry/good_sumcos_low_loss.py --epochs 10000 --overwrite
```

Outputs:

- `refs/icmlsymmetry/results/good_sumcos_low_loss/summary.csv`
- `refs/icmlsymmetry/results/good_sumcos_low_loss/low_loss_test_error_vs_partial_symmetry.png`
- `refs/icmlsymmetry/results/good_sumcos_low_loss/low_loss_rank_vs_partial_symmetry.png`
- Per config: `model_parameters.pth`, `losses.json`, `low_loss_curve.png`, `low_loss_layerwise_symmetry.png`, `low_loss_layerwise_minima.png`, `low_loss_partial_defect_distribution.png`, plus many `params_at_div_1.2_*` checkpoints.

Configs and achieved losses:

- `rank5_f3_N768_bs8_L3`: test MSE $1.599e-3$, train MSE $1.527e-3$.
- `rank5_f4_N1024_bs4_L3`: test MSE $1.590e-3$, train MSE $1.419e-3$.
- `rank5_f5_N1280_bs4_L3`: test MSE $3.262e-3$, train MSE $2.845e-3$.
- `rank10_f2_N512_bs2_L3`: test MSE $1.938e-5$, train MSE $1.926e-5$.
- `rank10_f3_N768_bs4_L3`: test MSE $1.340e-3$, train MSE $1.254e-3$.
- `rank10_f3_N768_bs8_L3`: test MSE $2.477e-3$, train MSE $2.371e-3$.

Main symmetry results:

- Output even defects are tiny: from $4.1e-6$ to $8.0e-4$.
- Last-layer active partial even defects are also tiny: from $6.24e-5$ to $5.22e-4$.
- This is the strongest current evidence for the paper: in regimes where the model truly fits the even target, MMNN low-rank bottleneck partials are also strongly even.
- Mirror-pair weight-space mismatch is consistently below 1 in this panel, around $0.56$ to $0.71$ for the best 20% nearest mirror pairs, with positive mirror correlations around $0.30$ to $0.46$. This supports but does not fully prove the weight-space encoding claim.

Interpretation:

The core paper should prioritize these low-loss ICLR configs over MNIST. MNIST is useful for optimizer sensitivity, but the clean symmetry claim belongs to controlled low-loss even-target sumcos runs.

## Exhaustive ICLR Worked-Config Rerun

Motivation:

After the focused low-loss panel, we started a complete rerun of every `worked=True` config available in the ICLR sweep CSVs, rather than only the manually selected six examples.

Script:

- `refs/icmlsymmetry/all_iclr_sumcos_rerun.py`

Command launched:

```bash
python refs/icmlsymmetry/all_iclr_sumcos_rerun.py --epochs 10000
```

Inputs:

- `experiments/table/baseline_sweep_sumcos_rank5_results.csv`
- `experiments/table/baseline_sweep_sumcos_results.csv`

Outputs:

- `refs/icmlsymmetry/results/all_iclr_sumcos_rerun/summary.csv`
- `refs/icmlsymmetry/results/all_iclr_sumcos_rerun/summary.json`
- `refs/icmlsymmetry/results/all_iclr_sumcos_rerun/all_test_error_vs_partial_symmetry.png`
- `refs/icmlsymmetry/results/all_iclr_sumcos_rerun/all_factor_vs_partial_symmetry.png`
- Per config: `model_parameters.pth`, `losses.json`, `low_loss_curve.png`, `low_loss_layerwise_symmetry.png`, `low_loss_layerwise_minima.png`, and `low_loss_partial_defect_distribution.png`.

Current status:

- The script selected 149 configs and is running in the background.
- At the latest check, at least 64 configs had completed, and the summary file is updated after every completed config.
- The early `f1` configs show an important caveat: low output loss does not always imply symmetric active partials in shallow settings.
- The `f2` and `f3` depth-3/depth-4 configs already reproduce the cleaner signal: low-loss runs often have last-layer active partial even defects around $10^{-4}$ to $10^{-3}$.
- Some rows that were `worked=True` in the historical CSV do not reproduce the same low test MSE under this fresh run; these should be separated from the main low-loss evidence rather than averaged blindly.

## Weight-Space Distribution Analysis

Motivation:

To make the workshop paper stronger, we need more than scalar mirror statistics. We need visual evidence of the trained weight-space organization, plus counterexamples showing when output symmetry does not imply internal symmetry.

Script:

- `refs/icmlsymmetry/weightspace_distribution_analysis.py`

Command run:

```bash
python refs/icmlsymmetry/weightspace_distribution_analysis.py
```

Outputs:

- `refs/icmlsymmetry/results/weightspace_distributions/weightspace_classification.csv`
- `refs/icmlsymmetry/results/weightspace_distributions/mirror_distance_distribution.png`
- `refs/icmlsymmetry/results/weightspace_distributions/outgoing_mismatch_distribution.png`
- `refs/icmlsymmetry/results/weightspace_distributions/outgoing_correlation_distribution.png`
- `refs/icmlsymmetry/results/weightspace_distributions/outgoing_weight_distribution.png`
- `refs/icmlsymmetry/results/weightspace_distributions/loss_partial_vs_mirror_mismatch.png`
- `refs/icmlsymmetry/results/weightspace_distributions/loss_partial_vs_mirror_correlation.png`
- `refs/icmlsymmetry/results/weightspace_distributions/example_partial-symmetric_mirror_pairs.png`
- `refs/icmlsymmetry/results/weightspace_distributions/example_output-only_asymmetric_mirror_pairs.png`
- `refs/icmlsymmetry/results/weightspace_distributions/example_underfit_mirror_pairs.png`

Classification from the currently completed checkpoints:

- 141 checkpoints analyzed.
- 12 partial-symmetric runs: low loss and active last-layer partial defect below $10^{-3}$.
- 28 output-only/asymmetric runs: low loss but active partial defect above $10^{-2}$.
- 92 intermediate runs.
- 9 underfit runs.

Automatically selected examples:

- Symmetric: `rank5_f2_N512_bs1_L3`, test MSE $9.53e-4$, active last-layer partial defect $2.62e-5$.
- Output-only/asymmetric: `rank10_f1_N256_bs2_L1`, test MSE $1.17e-4$, active last-layer partial defect $2.17$.
- Underfit: `rank5_f2_N128_bs1_L2`, test MSE $4.11e-1$.

Interpretation:

This is a strong new framing for the paper. The key claim should become conditional and sharper: MMNNs can encode target symmetry as symmetric partial functions and mirror-organized first-layer atoms, but this is not automatic from low output loss alone. The asymmetric counterexamples are useful, not bad news: they show that output-space symmetry and internal partial symmetry are distinct phenomena, and therefore the positive low-loss depth-3/depth-4 cases are nontrivial.
