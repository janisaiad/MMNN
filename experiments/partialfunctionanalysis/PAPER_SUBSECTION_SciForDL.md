# Empirical analysis of oscillatory complexity across layers in low-rank neural networks

*Subsection for the Workshop on Scientific Methods for Understanding Deep Learning (SciForDL).*

---

## Motivation and question

A recurring hypothesis in the theory of deep learning is that **deeper layers** build increasingly complex, high-frequency internal representations—e.g. in the context of frequency bias, spectral bias, or the compositional structure of learned functions. Testing such claims in a controlled way requires **observable proxies** for “oscillatory complexity” that can be computed from trained models without relying on simplifying limits (e.g. infinite width or linearized dynamics).

We focus on **matrix-rank neural networks (MMNNs)** trained on 1D regression tasks. For each layer we have a vector of **partial functions** \(h_\ell(x) \in \mathbb{R}^{d_\ell}\) (one scalar function per channel), defined on the input domain \(x \in [-1,1]\). A natural, interpretable proxy for “how oscillatory” a scalar function is along \(x\) is the **number of strict local minima** (with a positivity threshold to ignore ReLU zero plateaus). We ask:

- **Does the mean number of strictly-positive local minima per component increase with layer index \(\ell\)?**
- **How does this depend on architectural parameters such as depth \(L\) and hidden rank \(R\)?**

This subsection reports a **controlled experimental study** designed to answer these questions: we define the statistic, describe the protocol, summarize the results, and state limitations and follow-ups.

---

## Setup: models and partial functions

**Architecture.** We consider MMNNs with depth \(L\), hidden width \(W\), and hidden rank \(R\). Each layer consists of a rank-to-width linear map, ReLU, and a width-to-rank linear map. Input and output are 1D (\(x \in \mathbb{R}\), \(y \in \mathbb{R}\)). We use saved checkpoints (`model_parameters.pth`) from training on a 1D target (e.g. a sum of cosines) with MSE loss.

**Partial functions.** For a given trained model, we evaluate the network on a uniform grid \(x_1,\ldots,x_N \in [-1,1]\) (\(N = 1000\) in our runs) and record the **intermediate activations after each linear block** (including after ReLU where applicable), in the same order as in the code that produced the training runs. That yields, for each “layer index” \(\ell\) (indexing the sequence of linear blocks), a matrix of values \(h_{\ell,j}(x_i)\) with \(j = 1,\ldots,d_\ell\). Each column \(j\) is the **partial function** of the \(j\)-th channel at layer \(\ell\).

**Positivity threshold.** Because ReLU sets many pre-activations to zero, the resulting scalar functions often have long flat segments at zero. To count only “meaningful” minima we restrict to points where the function is **strictly above a small threshold** \(\varepsilon = 10^{-4}\): we count a strict local minimum only if \(h_{\ell,j}(x_i) > \varepsilon\) in addition to the usual inequality conditions on neighbors.

---

## Method: statistic and aggregation

**Per-component count.** For each scalar signal \(y_i = h_{\ell,j}(x_i)\), \(i=1,\ldots,N\), we define a **strict local minimum** at index \(i\) (with \(1 < i < N\)) by \(y_i < y_{i-1}\), \(y_i < y_{i+1}\), and \(y_i > \varepsilon\). The total count for that component is \(m_{\ell,j}\).

**Layer-level statistic.** For layer \(\ell\) we report the **mean number of minima per component**:
\[
\bar{m}_\ell = \frac{1}{d_\ell} \sum_{j=1}^{d_\ell} m_{\ell,j}.
\]
We also record the full distribution across \(j\) (e.g. boxplots and histograms) to assess heterogeneity.

**Multi-config aggregation.** When several runs are available (e.g. same \(L,W\) but different \(R\)), we either (i) average \(\bar{m}_\ell\) across runs to get a single “global” curve vs \(\ell\), or (ii) group runs by \((L,R)\) (or \((L,W,R)\)) and report per-group means and, where applicable, standard deviations and quantiles. All reported means are **unrounded** (floats); we store and plot them without rounding to integers.

---

## Experiments

**Figure chosen for the paper.** We select the **mean minima per component vs layer** plot for the grouped configuration \(L=6\), \(R=5\) (with \(W=128\)), as the main figure illustrating the depth effect. The file is: `experiments/partialfunctionanalysis/output_leap_all_L6_W128_grouped/grouped/L6_R5/mean_minima_per_component_vs_layer.png`. In the paper we refer to it as **Figure 1**.

*Suggested caption for Figure 1:* Mean number of strictly-positive local minima per component vs layer index \(\ell\), for MMNN with \(L=6\), \(W=128\), \(R=5\). Error bands show ±1 standard deviation across components (when aggregated over the run).

**Single-config (additional).** One other representative configuration: \(L=6\), \(W=128\), \(R=20\), trained for 3000 epochs on 1000 samples. We also compute \(\bar{m}_\ell\) for every layer index \(\ell\) and plot (a) mean minima per component vs layer index, and (b) boxplots of the per-component counts at each \(\ell\) to show spread and skewness.

**Multi-config sweep.** We fix \(L=6\), \(W=128\) and vary \(R \in \{5,10,15,20,30,36,50\}\) (one trained run per \(R\)). For each run we compute the same layer-wise statistics, then:
- **Aggregate across all runs:** mean curve \(\bar{m}_\ell\) and pooled histogram of minima counts over all components and layers.
- **Group by \(R\):** for each value of \(R\), we plot the mean curve vs \(\ell\) and (optionally) per-layer or pooled histograms. Summary statistics are exported to CSV/JSON for tables and further plotting.

**Reproducibility.** Script and options are documented in the repository; the analysis script reads `config.json` and `model_parameters.pth` from each run directory and writes figures and CSV under a specified output directory.

---

## Results

**Depth effect.** In all runs we observe a **clear increase in \(\bar{m}_\ell\) with layer index \(\ell\)** in the second half of the network. Early layers have mean close to zero (most components have no strictly-positive local minima); later layers show mean counts of order 1–3 and higher, with **substantial spread** across components (boxplots show widening distributions and right-skewed counts). This is consistent with the hypothesis that **deeper layers learn more oscillatory partial functions** in this discrete, grid-based proxy.

**Rank effect.** At fixed \(L=6\), \(W=128\), the curves \(\bar{m}_\ell\) vs \(\ell\) **differ across \(R\)**, but the dependence on \(R\) is **not monotonic** in the current data. For example, at a fixed late layer (e.g. layer index 12), the mean minima per component takes values around 2–9 depending on \(R\), without a simple increasing or decreasing trend in \(R\). Thus **rank influences the internal complexity**, but a single run per \(R\) does not allow a clean claim about the functional form of this dependence; replication over seeds is needed.

**Aggregate curve.** The mean curve computed by averaging \(\bar{m}_\ell\) across the seven configs (varying \(R\)) preserves the same qualitative shape: low in early layers, then a sharp rise in the second half of the network, supporting the robustness of the depth effect across the sampled rank values.

---

## Discussion

**Interpretation.** The increase in mean minima per component with depth can be read as an **empirical regularity**: under the present training setup and architecture, later layers tend to produce partial functions with more oscillations (more local minima above the positivity threshold). This aligns with intuition from approximation theory and frequency bias (deeper compositions supporting higher frequencies) but is here **measured directly** on the learned 1D partial functions rather than in a simplified theoretical model.

**Limitations.** (1) **No seed replication:** each \((L,W,R)\) configuration corresponds to one training run, so differences across \(R\) may reflect optimization noise. (2) **Grid resolution:** the count is a lower bound on the number of minima; components that oscillate faster than the sampling grid may have under-counted minima. (3) **Last layer:** the final layer output is scalar in our setup, so “mean over components” at that layer is a single curve’s count and should be interpreted separately. (4) **Task and scale:** conclusions are for the specific 1D regression task and the chosen \(L,W,R\) range; generalization to other tasks or architectures would require further experiments.

**Suggested follow-ups.** To strengthen claims about the effect of rank \(R\), we recommend (i) **multiple seeds** per configuration and reporting means with standard errors or confidence intervals; (ii) optionally combining the minima count with a **Fourier-based** metric (e.g. spectral centroid or 95% rolloff frequency) for a more direct “high frequency” proxy; (iii) extending the same protocol to other depths \(L\) and widths \(W\) to map the \((L,W,R)\) landscape more systematically.

---

## Conclusion

We defined a simple, reproducible statistic—the mean number of strictly-positive local minima per component of the layer-wise partial functions—and applied it to trained MMNNs on 1D regression. The experiments provide **empirical evidence that deeper layers exhibit higher oscillatory complexity** in this proxy, while the effect of hidden rank \(R\) is present but not yet characterized in a monotonic way with the current single-run-per-config design. The pipeline (partial function extraction, minima counting, aggregation, and export to CSV/figures) is implemented and documented so that the scientific claim can be refined or falsified with additional runs and metrics. This work fits the workshop’s aim of using **controlled experiments** to formulate and test precise hypotheses about deep learning, with results that can inform both theory (e.g. depth vs frequency in low-rank nets) and practice (e.g. choice of depth and rank for oscillatory targets).
