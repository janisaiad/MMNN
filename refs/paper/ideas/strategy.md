# Research Strategy: Multi-Component Multi-layer Neural Networks (MMNN)

## General Vision

Our objective is to demonstrate that MMNN architectures provide a fundamental advantage for deep learning due to their low-rank structure and mean-field training regime. This architecture drastically reduces the number of parameters while maintaining (or even improving) universal approximation capabilities.

## Central Hypothesis

MMNNs operate in a **mean-field regime** (widths $n \sim 500$-$5000$) where the training dynamics can be described by a deterministic ODE. This regime is fundamentally different from the NTK regime (lazy training) and enables genuine feature learning, rather than merely selecting among a fixed function space.

## Key Contributions to Establish

### 1. Parametric Advantage via Low-Rank Structure

**Claim:** For 3+ layer networks, low-rank factorization $W = A_1 A_2$ with intermediate dimension $r \ll \min(n_1, n_2)$ reduces the complexity from $O(n^2)$ to $O(n \cdot r)$ while preserving expressiveness.

**Approach:**
- Demonstrate that the 2-layer regime corresponds to the classical random features model (no benefit)
- Prove that the advantage emerges starting from 3 layers, via finite-time universal approximation
- Analyze the "extensive rank" regime: $r \asymp d^\beta$ for $\beta \in (0,1)$

### 2. Dictionary Learning and Wavelet Learning

**Empirical observation:** MMNNs learn basis functions (wavelets) which are stretched and placed at specific intervals, creating high-frequency spikes at precise locations.

**Mechanism:**
- The first layer learns functions via universal approximation
- These functions are recursively combined to build higher and higher frequencies
- The process remains highly symmetric even with non-symmetric batches

**Key parameters:**
- $\beta/w$: spike position
- $w$: frequency
- Singular values matter less than the **direction** of singular vectors

### 3. Global Convergence in the Non-Convex Regime

**Main theoretical result:** The mean-field limit converges to the global optimum under suitable regularity assumptions, **even when the loss is non-convex**.

**Crucial ingredient:** The universal approximation property holds at **any finite training time** (not necessarily at convergence). This is enabled by:
- The full support of the particle distribution (ensured by random features)
- The factorization of the intermediate weight matrix
- Taking the limit $n_1, n_2 \to \infty$, leading to a linear regime in the number of parameters

### 4. Random Features in High-Dimensional Space

**Perspective:** Each data point is projected into a high-dimensional embedding space, where the network learns to use this structure to build dictionary functions.

**Advantages:**
- Agnostic to the input data dimension
- Avoids the high-dimensional statistical difficulties of the NTK regime
- Random features $\{\sigma(w_j^\top h^{(\ell-1)} + \beta b_j)\}$ remain fixed, preserving full support
- Training is performed on the vector $A$ in this high-dimensional feature space (analogous to the quadratic problem in learningquadratic)

### 5. Partial Training: Computational Benefits

**Strategy:** Train only the output weights $\{A_j^{(\ell)}\}$ and biases $\{c^{(\ell)}\}$ while keeping the input weights $\{w_j^{(\ell)}, b_j^{(\ell)}\}$ fixed.

**Benefits:**
- Halves the number of trainable parameters
- Avoids Riemannian gradient descent
- Avoids the $\alpha^2$ term in the factorized update: $W_{t+1} = W_t - \alpha \nabla_t + \alpha^2 \nabla W_t W_t \nabla W_t^\top$
- Transforms the problem into **linear programming** with uniform distributions

### 6. Emergent Behavior and Scaling Laws

**Inspiration:** Multi-index models and scaling laws (Ben Arous et al., OpenAI scaling laws).

**Observations:**
- Emergent behavior (plateaus followed by abrupt drops) during training
- Each plateau marks the learning of a term in the additive model
- Gradual transition from step-like curves ($\alpha=0$) to smooth curves (increasing $\alpha$)
- In the light-tailed regime ($\alpha > 1/2$): $\mathcal{R} \sim 1/(\text{Data size})^a + 1/(\text{Model size})^b$

**Stiefel connection:** The weights live on a Stiefel manifold (orthogonal in high dimensions, number of weights < $d$), allowing control of $\beta$ in $r \asymp d^\beta$.

### 7. Operator Switching: Adam → SGD

**Major empirical finding:** Switching from Adam to SGD during training stabilizes phases where Adam becomes unstable (the edge of stability).

**Mechanism:** The curse of adaptive learning rates disappears with SGD at critical moments.

**Note:** This phenomenon will be the subject of a separate, more detailed paper.

## Incremental Publication Plan

### ArXiv Paper (Complete, Mathematical)

**Contents:**
- All developed mathematical results (NTK, NNGP, recursions)
- Mean-field ODE framework for 3 layers (simplified adaptation)
- Non-quantitative convergence results
- Spectral NTK analysis through depth (Terjék-like approach)
- Unifying theory using tensor programs + $\mu$P for hyperparameter transfer

**Theoretical frameworks involved:**
- NTK and RKHS with randomness (high dimension)
- Partial then random training
- Spherical harmonics (multi-index models)
- Inductive bias toward low frequencies (explained by NTK)
- PINN loss and Sobolev training

### Conference Paper (Empirical, Benchmark)

**Contents:**
- Empirical demonstration that MMNN outperforms MLP for 3-layer architectures
- Extensive comparative benchmarks
- Ablation study: what makes MMNN optimization effective?
- Experimental results illustrating universal convergence (globalconvergencesthreelayers)

### Positioning: Random Features Framework

**Strategic justification:** Presenting MMNNs under the random features framework inherits the whole existing literature on:
- Generalization (VC-dimension, Rademacher complexity)
- Approximation (universal convergence rates)
- High-dimensional analysis
- RKHS properties

**Progressive extension:** Build the preprint by iteratively adding results in directions such as:
- Tensor programs
- Central flow point of view
- SETOL criteria for SVD
- Hermite expansion (chaos expansion for random features)

## Open Questions and Future Directions

1. **Lyapunov Analysis:** Study scaling with depth $L$ via products of independent random matrices
2. **Taylor Expansion:** Develop the NTK close to $\rho = 1$ for exact spectral bounds
3. **NTK Fluctuations:** Exploit the fast decay rate $O(1/r)$ for trivial eigenvalue bounds
4. **LP Formulation:** Exact optimization via linear programming with uniform distributions
5. **Multiple minima:** Glass-like structure for multiple local minima at finite $N$ corresponding to the same minimizer $\rho_*$ as $N \to \infty$
6. **Hermite vs ReLU:** Understand if the Hermite expansion can characterize MMNN optimization despite the ReLU nonlinearity (added difficulty vs quadratic case)

## Timeline and Priorities

**Short-term (3 months):**
1. Robust numerical experiments for the mean-field regime (tracking particle distribution + ODE satisfied)
2. 3-layer MMNN vs MLP benchmarks
3. Complete ablation study

**Medium-term (6 months):**
1. Non-quantitative convergence proof via mean-field ODE
2. Characterization of learning dynamics (wavelets/dictionary)
3. Write conference paper

**Long-term (12 months):**
1. Complete NTK spectral results with depth
2. Integration of tensor programs + $\mu$P
3. Complete mathematical ArXiv paper
4. Proof of Prof. Haizhao Yang's conjecture (global minima stability)

## Conclusion

MMNNs represent an architecture fundamentally different from standard MLPs, operating in a regime where genuine feature learning is possible with a linear parameter budget. The key lies in the low-rank structure which, starting from three layers, unlocks universal approximation properties maintained at finite time, enabling global convergence even in the non-convex regime. The random features framework provides the ideal theoretical context to analyze these properties, while benefiting from an extensive existing literature.