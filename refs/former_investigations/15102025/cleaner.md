

# MMNN Training Analysis Script

## 1. Context and Setup

I have extensively trained MMNNs while monitoring the training process and manually optimizing hyperparameters to establish a reliable baseline. Here I present the training setup, qualitative results, and suggested research directions.

### Baseline Architecture
- 120k parameters (666365 + 666), with half being trained
- 1000 target functions
- MLP width 175, depth 4
- Extensive hyperparameter optimization and monitoring

### Key Questions
- Power law scaling: Is it due to convex descent or Adam's learning rate decay?
  - Answer: Due to Adam convergence in highly convex landscape during descent phase
- Can we perfectly approximate highly oscillating 1D functions (freq ~100)?
  - Yes, to MSE 1e-3, with super-convergence starting at 1e-2
- Are MSE and max correlated? Interpolation vs kernel regime?
- Does solution recover all Fourier modes?
- Is there an inductive bias toward learning small absolute values first?

## 2. Observations and Results

### Training Dynamics
- 2-3 layer MMNNs are hard to train, show strong gradient descent bias
- Layer 1-2 behavior differs greatly from deeper layers
- Only 5-7 dictionary functions learned, some highly redundant
- Large width shows localized spike functions, combined by low ranks
- Inductive bias produces positive ReLU spikes selecting input intervals at specific frequencies

### NTK Behavior
- Search phase: Erratic NTK movement post-initialization
- Large width: More PSD NTK, convex descent even at low rank
- Width 1000 (1000 samples): Very convex, overparametrized regime works well
- 5x1000 config: Feature learning with convex descent
- Global minimum sometimes missed due to long training time (10 1024 50 config)

### Key Findings
- NTK poor metric for search phase but stable in descent phase
- Spectrum becomes PSD (convex) even at small width
- NTK predicts PSD-ness and convergence well
- Training usually converges but 5x longer for 10x more width/depth
- SGD fails completely
- Fourier-based loss more relevant for training/monitoring

## 3. Future Directions

### Theoretical Work
- Study landscape convexity near MMNN decomposition
- Evaluate dictionary complexity
- Develop Sobolev loss theory

### Experimental Work
- Plot parameter budgets, epochs, loss, convexity scaling laws
- Test Sobolev-Fourier and PINN losses
- Study finite-width NTK evolution with recursive formula













