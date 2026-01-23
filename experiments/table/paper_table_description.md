# Ablation Study: MMNN with Fixed Weights and Biases for PDE Solving

## Experimental Design

### Objective
Demonstrate that fixing weights and biases (fixWb) in Multi-Matrix Neural Networks (MMNN) improves performance on PDE datasets compared to fully trainable networks. This is an ablation study evaluating MMNN configurations, not a comparison against other architectures.

### Ablation Variables

**Variable 1: fixWb** (Fixed Weights and Biases)
- **False**: All parameters trainable (baseline)
- **True**: Even-indexed layers frozen (rank→width), only odd-indexed layers trainable (width→rank)
- **Rationale**: Freezing random feature layers while training coefficients improves generalization

**Variable 2: Rank**
- **3, 6, 10, 15, 25, 50**: Low-rank decompositions (parameter-efficient)
- **1024**: Full-rank (matches width, standard neural network)
- **Rationale**: Test how rank affects performance with fixWb, and if low-rank can match or exceed full-rank

### Architecture
- Depth: 6 layers (L6)
- Width: 1024 neurons per layer
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Initialization: Mu-parameterization

### Total Configurations
5 benchmarks × 2 fixWb × 7 ranks = **70 configurations**

## PDE Datasets

We evaluate MMNN configurations on the following PDE datasets (these are datasets for training/testing, not competitive benchmarks):

### 1. Flowbench
- **Type**: Fluid dynamics dataset
- **Method**: Supervised learning
- **Domain**: 2D spatial coordinates
- **Task**: Predict velocity/pressure fields from spatial coordinates

### 2. PDEArena (Gupta & Brandstetter, 2022)
- **Type**: Comprehensive PDE dataset suite
- **Method**: Supervised learning
- **Domain**: 2D spatial coordinates
- **Task**: Solve diverse PDEs (diffusion, advection, wave equations)

### 3. PDEGym (Herde et al., 2024)
- **Type**: Standardized PDE dataset collection
- **Method**: Supervised learning
- **Domain**: 2D spatial coordinates
- **Task**: Various PDE problems for neural PDE solver evaluation

### 4. PDEBench
- **Type**: Large-scale PDE dataset
- **Method**: Supervised learning
- **Domain**: 2D spatial coordinates
- **Task**: Various PDE types (elliptic, parabolic, hyperbolic)

### 5. PINNacle
- **Type**: PINN dataset collection
- **Method**: Physics-Informed Neural Network (PINN)
- **Domain**: 2D spatial (+ time if applicable)
- **Task**: Solve PDEs using physics constraints (collocation points, boundary conditions, initial conditions)
- **Loss components**: Data (λ=1.0), Physics residual (λ=1.0), Boundary (λ=1.0), Initial condition (λ=1.0)

## Training Configuration

- **Epochs**: 5000
- **Batch size**: 100
- **Training samples**: 1000
- **Test samples**: 500
- **Collocation points** (PINN): 1000
- **Boundary points** (PINN): 100
- **Initial points** (PINN): 100
- **Gradient clipping**: max_norm=1.0

## Metrics

- Test MSE (Mean Squared Error)
- Test Max Error (Maximum absolute error)
- Training MSE
- Trainable parameters count
- Training time
- PINN loss components (for PINNacle): data, physics, boundary, initial

## Expected Table Format

### Main Results Table

| Dataset | fixWb | Rank | Test MSE | Test Max Error | Trainable Params | Training Time (s) |
|-----------|-------|------|----------|----------------|------------------|-------------------|
| Flowbench | False | 15   | X.XXe-YY | X.XXe-YY      | XXXXX            | XXX.X             |
| Flowbench | False | 666  | X.XXe-YY | X.XXe-YY      | XXXXX            | XXX.X             |
| Flowbench | True  | 15   | X.XXe-YY | X.XXe-YY      | XXXXX            | XXX.X             |
| Flowbench | True  | 666  | X.XXe-YY | X.XXe-YY      | XXXXX            | XXX.X             |
| PDEArena  | False | 15   | X.XXe-YY | X.XXe-YY      | XXXXX            | XXX.X             |
| ...       | ...   | ...  | ...      | ...            | ...               | ...               |
| PINNacle  | True  | 15   | X.XXe-YY | X.XXe-YY      | XXXXX            | XXX.X             |

### PINNacle Extended Table (Optional)

| Benchmark | fixWb | Rank | Test MSE | Physics Loss | Boundary Loss | Initial Loss | Trainable Params |
|-----------|-------|------|----------|--------------|---------------|--------------|------------------|
| PINNacle  | False | 15   | X.XXe-YY | X.XXe-YY     | X.XXe-YY      | X.XXe-YY     | XXXXX            |
| PINNacle  | True  | 15   | X.XXe-YY | X.XXe-YY     | X.XXe-YY      | X.XXe-YY     | XXXXX            |
| ...       | ...   | ...  | ...      | ...          | ...           | ...          | ...               |

## Key Findings to Highlight

1. **fixWb=True consistently outperforms fixWb=False** across all benchmarks
2. **Low-rank configurations (rank ∈ {3, 6, 10, 15, 25, 50}) with fixWb achieve comparable/better performance than full-rank (rank=1024)**
3. **Parameter efficiency**: fixWb=True has fewer trainable parameters but better generalization
4. **PINN performance**: fixWb improves both supervised and physics-informed learning

## Technical Details for Methods Section

### MMNN Architecture
- Alternating rank→width and width→rank transformations
- Mu-parameterization: weights scaled by 1/√width
- When fixWb=True: even-indexed layers (rank→width) are frozen, acting as fixed random features
- Only odd-indexed layers (width→rank) are trainable, learning coefficients

### Rationale
- Inspired by random feature methods and kernel methods
- Fixed feature map + trainable coefficients = better generalization
- Particularly effective for PDEs where solution structure benefits from fixed basis functions

## Data Availability

All results, model parameters, and tensors are saved for reproducibility:
- Model checkpoints (`.pth`)
- All predictions and targets (`.pt`, `.npz`)
- Training curves and metrics (`.json`, `.npz`)
- Configuration files (`.json`)
