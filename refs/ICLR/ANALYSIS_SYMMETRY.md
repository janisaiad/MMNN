# Complete analysis: symmetry preservation (low-rank vs full-rank)

## 1. Setup and motivation

- **Target**: $f(x) = \cos(f_1\pi x^2) - 0.8\cos(f_2\pi x^2)$ on $x \in [-1,1]$, with $(f_1,f_2) = (144,48)$ for the symmetry comparison.
- **Symmetry**: Both terms are **even in $x$**, so the target is **symmetric about $x=0$**.
- **Question**: Do learned internal representations (channel partials, activations) respect this symmetry under batched optimization?

## 2. Experimental design (from `tofill.tex` and table)

| Setting | Symmetry (low-rank) | Asymmetry (full-rank) |
|--------|----------------------|------------------------|
| Figure | channel-partials, spike-in-H | mlp_asymmetry (layers 7, 16) |
| $L$ | 8 | 8 |
| $n$ | 1024 | 1024 |
| $r$ | 15 (channel-partials) / 8 (spike) | 20 |
| Target | (144,48) | (144,48) |
| $N_{\mathrm{train}}$ | 4k | 4k |
| Epochs | 10k | 10k |
| Batch | 100 | 100 |
| Optimizer | Adam, lr 0.001, $\gamma=0.9$ every 100 | same |
| **RF (low-rank structure)** | **True** | **False** |

So the **only** difference in the symmetry comparison is: **RF = True** (low-rank, fix first layer / RF-LR) vs **RF = False** (full-rank MLP, no RF constraint). Same depth, width, target, data size, and optimizer.

## 3. Observations from the figures

- **`mlp_asymmetry_layer7.png` and `mlp_asymmetry_layer16.png`** (full-rank, RF=False): channel partials at layers 7 and 16 are **visibly asymmetric** about $x=0$. Peaks and valleys do not mirror at $\pm x$; the learned features break the target’s even symmetry.
- **Symmetric runs** (e.g. channel_partials_layer5/9/16 with RF=True): channel partials and ReLU($H$) activations remain **symmetric about $x=0$**, with “bigger spikes” but mirror structure.

So: **low-rank (RF-LR) preserves symmetry; full-rank (RF removed) does not**, under the same training setup.

## 4. Interpretation

1. **Target symmetry**: The task only has even symmetry; there is no incentive for the optimizer to prefer asymmetric representations.
2. **Batched optimization**: With mini-batch updates, the loss landscape has many basins; some minimizers are symmetric and some are asymmetric while still fitting the data.
3. **Full-rank MLP**: The extra degrees of freedom allow the trajectory to converge to **asymmetric** internal representations that still achieve low loss. So “fitting the target” does not imply “symmetric features.”
4. **Low-rank (RF-LR / MMNN)**: The **low-rank constraint** (and, where used, the frozen first layer / RF) **restricts the hypothesis set**. In practice, under the same optimizer and data, the learned channel partials and activations **preserve symmetry**. So the constraint acts as an **implicit regularizer** that favors symmetric minimizers.
5. **Landscape link**: Symmetric targets plausibly have both symmetric and asymmetric basins. Low-rank nets are steered into symmetric basins; full-rank nets can fall into asymmetric ones. So symmetry preservation is another facet of the **relation between feature learning and the loss landscape**: the geometry (here, symmetry) of the learned representation is tied to the constraint and the basin selected by optimization.

## 5. Connection to landscape paper

In **landscape.tex** we argue that (i) plateaus/saddles correspond to frequencies and (ii) the landscape has sharp, deep basins that require small LR to enter. The **symmetry comparison** adds:

- The **same** target and optimizer can lead to **symmetric** (low-rank) or **asymmetric** (full-rank) internal features.
- So the **architecture/constraint** (low-rank vs full-rank) influences **which basin** is reached and hence the **geometric structure** (here, symmetry) of the learned representation.
- This supports the claim that feature learning and loss landscape geometry are tightly linked: not only frequency-by-frequency learning and plateau escape, but also preservation of global properties (e.g. even symmetry) when the constraint favors it.

## 6. References in the repo

- **Source text**: `refs/ICLR/icml_sgdadamlandscapedynamical/tofill.tex`, Sec. 5.3 (Channel and activations learn symmetric spikes), Table (hyperparameters), and Fig. mlp_asymmetry (layers 7 and 16).
- **Figures**: `refs/ICLR/figures/mlp_asymmetry_layer7.png`, `mlp_asymmetry_layer16.png` (copied from icml_sgdadamlandscapedynamical/figures/).
- **Landscape paper**: subsection “Symmetry preservation: low-rank vs full-rank” and appendix “Symmetry comparison” in `landscape.tex`.

## 7. Summary

- **Empirical fact**: For a symmetric target, low-rank (RF-LR) training yields symmetric channel partials; full-rank (RF=False) training yields asymmetric ones under the same hyperparameters.
- **Mechanism**: The low-rank constraint implicitly regularizes toward symmetric representations; full-rank nets have the capacity to fit the target with asymmetric features.
- **Role in the narrative**: Symmetry preservation is a concrete example of how the **constraint** and **landscape geometry** together determine the **structure of learned features**, alongside plateau–frequency correspondence and LR-dependent basin entry.
