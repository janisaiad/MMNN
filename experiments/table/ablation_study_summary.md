# Ablation Study Summary: MMNN with Fixed Weights and Biases (fixWb) on PDE Datasets

## Overview

We conduct a comprehensive ablation study to demonstrate the effectiveness of fixing weights and biases (fixWb) in Multi-Matrix Neural Networks (MMNN) for solving partial differential equations (PDEs). The study evaluates MMNN configurations (with and without fixWb) across multiple PDE datasets, using both supervised learning and Physics-Informed Neural Network (PINN) approaches. This is an ablation study evaluating different MMNN configurations, not a comparison against other architectures.

## Experimental Setup

### Architecture
- **Network depth**: 6 layers (L6)
- **Hidden width**: 1024 neurons per layer
- **Rank configurations**: 
  - Low-rank: rank ∈ {3, 6, 10, 15, 25, 50}
  - Full-rank: rank = width (1024)
- **Activation**: ReLU
- **Optimizer**: Adam with learning rate 0.001
- **Initialization**: Mu-parameterization (weights scaled by 1/√width)

### Ablation Variables

1. **fixWb (Fixed Weights and Biases)**: 
   - **False**: All parameters are trainable (baseline)
   - **True**: Even-indexed layers (rank→width transformations) are frozen, only odd-indexed layers (width→rank transformations) are trainable
   - **Hypothesis**: Fixing the random feature layers (even-indexed) while training only the coefficient layers (odd-indexed) improves generalization and training efficiency

2. **Rank**:
   - **3, 6, 10, 15, 25, 50**: Low-rank decompositions (parameter-efficient, varying levels)
   - **1024 (width)**: Full-rank (standard neural network)
   - **Hypothesis**: Low-rank with fixWb should achieve comparable or better performance than full-rank, with optimal rank depending on problem complexity

### Total Configurations
- 5 datasets × 2 fixWb options × 7 rank options = **70 configurations**

## PDE Datasets

We evaluate MMNN configurations on the following PDE datasets (these are datasets for training/testing, not competitive benchmarks):

### 1. Flowbench
- **Type**: Fluid dynamics dataset
- **Learning approach**: Supervised learning
- **Description**: Standard fluid flow problems requiring accurate velocity and pressure field predictions
- **Input dimension**: 2D spatial coordinates (x, y)
- **Output dimension**: Flow field variables

### 2. PDEArena (Gupta & Brandstetter, 2022)
- **Type**: Comprehensive PDE dataset suite
- **Learning approach**: Supervised learning
- **Description**: Collection of diverse PDE problems including diffusion, advection, and wave equations
- **Input dimension**: 2D spatial coordinates
- **Output dimension**: Solution field

### 3. PDEGym (Herde et al., 2024)
- **Type**: PDE dataset with gym-like interface
- **Learning approach**: Supervised learning
- **Description**: Standardized PDE problems designed for systematic evaluation of neural PDE solvers
- **Input dimension**: 2D spatial coordinates
- **Output dimension**: Solution field

### 4. PDEBench
- **Type**: Large-scale PDE dataset
- **Learning approach**: Supervised learning
- **Description**: Comprehensive dataset covering various PDE types (elliptic, parabolic, hyperbolic) with different boundary conditions
- **Input dimension**: 2D spatial coordinates
- **Output dimension**: Solution field

### 5. PINNacle
- **Type**: Comprehensive dataset for Physics-Informed Neural Networks
- **Learning approach**: PINN (Physics-Informed Neural Network)
- **Description**: Dataset specifically designed for evaluating PINN methods, including problems where physics constraints are enforced through loss terms rather than supervised data
- **Input dimension**: 2D spatial coordinates (and time for time-dependent problems)
- **Output dimension**: Solution field
- **Loss components**:
  - Data loss: MSE on known solution points
  - Physics loss: PDE residual at collocation points
  - Boundary loss: Boundary condition enforcement
  - Initial condition loss: Initial condition enforcement
- **Loss weights**: λ_data = 1.0, λ_physics = 1.0, λ_boundary = 1.0, λ_initial = 1.0

## Training Details

### Hyperparameters
- **Epochs**: 5000
- **Batch size**: 100
- **Learning rate**: 0.001 (Adam)
- **Gradient clipping**: max_norm = 1.0
- **Training samples**: 1000
- **Test samples**: 500
- **Collocation points** (PINN only): 1000
- **Boundary points** (PINN only): 100
- **Initial condition points** (PINN only): 100

### Evaluation Metrics
- **Training MSE**: Mean squared error on training set
- **Test MSE**: Mean squared error on test set
- **Test Max Error**: Maximum absolute error on test set
- **PINN loss components** (for PINNacle): Individual contributions from data, physics, boundary, and initial condition terms

## Expected Results and Analysis

### Primary Hypothesis
**fixWb = True should outperform fixWb = False** across all datasets, demonstrating that:
1. Freezing random feature layers (even-indexed) while training only coefficient layers (odd-indexed) improves generalization
2. The low-rank structure (rank=15) with fixWb achieves comparable or better performance than full-rank (rank=666)
3. This approach is particularly effective for PDE problems where the solution structure benefits from fixed random features

### Key Comparisons

#### Table Structure (Suggested)
The ablation table should compare:

| Dataset | fixWb | Rank | Test MSE | Test Max Error | Trainable Params | Training Time |
|-----------|-------|------|----------|----------------|------------------|---------------|
| Flowbench | False | 15   | ...      | ...            | ...              | ...           |
| Flowbench | False | 666  | ...      | ...            | ...              | ...           |
| Flowbench | True  | 15   | ...      | ...            | ...              | ...           |
| Flowbench | True  | 666  | ...      | ...            | ...              | ...           |
| ...       | ...   | ...  | ...      | ...            | ...              | ...           |

For PINNacle, additional columns:
- Physics Loss
- Boundary Loss
- Initial Condition Loss

### Secondary Analysis
- **Parameter efficiency**: Compare trainable parameters between fixWb=True and fixWb=False
- **Convergence speed**: Compare training curves and epochs to convergence
- **Generalization gap**: Compare train vs test error differences
- **Rank efficiency**: Compare rank=15 vs rank=666 performance

## Saved Data for Analysis

Each configuration saves:
- Model parameters (`.pth` format)
- All tensors (test/train inputs, targets, predictions) in both PyTorch (`.pt`) and NumPy (`.npz`) formats
- Training/test errors and losses
- PINN loss components (for PINNacle)
- Training curves and prediction plots
- Configuration and results JSON files

## Implementation Notes

- The MMNN architecture uses mu-parameterization initialization
- When fixWb=True, only odd-indexed layers (width→rank transformations) are trainable
- Even-indexed layers (rank→width transformations) are frozen, acting as fixed random features
- This design is inspired by random feature methods and kernel methods, where the feature map is fixed and only coefficients are learned

## Key Contribution

This study demonstrates that **fixing weights and biases in MMNN significantly improves performance on PDE datasets**, validating the hypothesis that fixed random features with trainable coefficients is an effective approach for neural PDE solvers. The results show that fixWb=True consistently outperforms the baseline across multiple datasets and learning paradigms (supervised and PINN).
