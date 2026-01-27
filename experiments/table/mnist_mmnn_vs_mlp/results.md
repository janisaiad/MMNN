# MNIST: MLP baseline vs MMNN

*For paper-writing: all training parameters and consolidated results.*

---

## 1. Training parameters

| Parameter | Value |
|-----------|-------|
| **Dataset** | MNIST; 60k train, 10k test; images flattened 28×28→784, 10 classes |
| **Input** | 784; normalisation mean 0.1307, std 0.3081 |
| **Output** | 10 (logits); loss CrossEntropy |
| **Optimizer** | Adam |
| **Learning rate** | 1e-3 |
| **Epochs** | 30 |
| **Batch size** | 128 |
| **Gradient clipping** | max_norm=1.0 |
| **Seed** | 42 |

**MLP:** ReLU MLP, 784→512→512→10 (2 hidden layers, hidden 512). All parameters trained.

**MMNN:** 2 blocks; `ranks = [784, R, 10]`, `widths = [512, 512]`. Each block: Linear(rank→512) → ReLU → Linear(512→rank). Mu-parameterisation init (uniform rescaled by 1/√fan).  
- **fixWb=True (random/frozen features):** rank→512 (and bias) frozen; only 512→rank trained.  
- **fixWb=False:** all parameters trained.

---

## 2. Consolidated results

| Model | R | fixWb | Params | Trainable | Test acc (%) | Test loss |
|-------|---|-------|--------|-----------|--------------|-----------|
| **MLP** | — | — | 669,706 | 669,706 | **98.39** | 0.1177 |
| **MMNN** | 5  | True  | 412,687 | 7,695   | **93.72** | 0.2116 |
| **MMNN** | 10 | True  | 417,812 | 10,260  | **96.24** | 0.1348 |
| **MMNN** | 15 | True  | 422,937 | 12,825  | **96.97** | 0.1194 |
| **MMNN** | 25 | True  | 433,187 | 17,955  | **97.00** | 0.1139 |
| **MMNN** | 50 | True  | 458,812 | 30,780  | **97.01** | 0.1254 |
| **MMNN** | 32 | False | 440,362 | 440,362 | **98.30** | 0.112  |

R=5, R=10 from a separate run with the same training parameters. R=15, 25, 50 (fixWb) and R=32 (fixWb=False) from runs stored in `results.json`.

---

## 3. Plot: test accuracy vs log₁₀(trainable parameters)

**File:** `mnist_perf_vs_params.png`  
**Script:** `plot_mnist_perf_vs_params.py` (matplotlib config from `meanfield_cosine_multifreq_experiment.py` lines 24–41).  
**Data:** `plot_data.json` (log₁₀ trainable params and test acc per model).

- **X-axis:** log₁₀(trainable parameters)
- **Y-axis:** test accuracy (%)
- **MMNN fixWb (random features):** curve for R=5, 10, 15, 25, 50; only width→rank matrices trained.
- **MMNN R=32 fixWb=False:** single point; all parameters trained.
- **MLP:** horizontal line at 98.39%.

**Caption (for paper):** *MNIST test accuracy vs trainable parameters. MLP (dashed, 98.39%) uses 670k parameters. Low-rank MMNN with frozen random features (fixWb) reaches ~97% with 13k–31k trainable parameters; MMNN with all parameters trained (R=32, fixWb=False) matches MLP at ~98.3% with ~440k parameters.*

---

## 4. Run summary

- **Run 1:** MMNN R=32 fixWb=False vs MLP → both ~98.3%; MMNN uses ~34% fewer parameters.
- **Run 2 (fixWb):** R=5,10,15,25,50 with fixWb=True. R=15–50 reach ~97% with 13k–31k trainable params; R=5 reaches 93.72% with 7.7k.

---

## 5. Interpretation (random features vs learned features)

This experiment is an **example where random (frozen) features are competitive**: RF-LR reaches ~97% with far fewer trainable parameters than the full MLP. Random features are known to **help for highly oscillating functions** (e.g. high-frequency cosines, as in the meanfield/cosine benchmarks in this repo), where a random basis can span the right harmonics.

**Caveat:** For **extracting manifold or task-relevant features**, **learning is provably better**: random features do worse in that setting, and learned representations are required to capture the data geometry. The MNIST setup here does not target manifold structure; it illustrates the RF-LR efficiency regime, not the regime where learned features are necessary.

---

## 6. Files (for paper / reproducibility)

| File | Description |
|------|-------------|
| `results.md` | This document (training params, tables, captions) |
| `mnist_perf_vs_params.png` | Figure: test accuracy vs log₁₀(trainable parameters) |
| `plot_mnist_perf_vs_params.py` | Script to regenerate the figure |
| `plot_data.json` | Numerical data for the figure |
| `results.json` | Last run: MLP + MMNN R=15,25,50 (fixWb) |
| `histories.json` | Train/test loss and acc per epoch |
| `mlp.pt`, `mmnn_r*.pt` | Saved model state dicts + config |
