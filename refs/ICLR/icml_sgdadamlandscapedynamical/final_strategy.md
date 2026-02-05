# title propositions
Low rank and random features neural network is enough for high frequency feature learning
Low rank and random features neural network is enough for fourier features
Low rank and random features neural network learns fourier features
Low rank neural networks is enough for mean-field global convergence


3 layers Low rank random features neural networks for mean-field global convergence



(workshop)
Low rank networks learns fourier features
low rank networks revert MLP spectral bias




TOWARDS APPLICATIONS

plan : 
mean field equations derivation
existence unicity of ODE equations 
global convergence (proof to write and state)
rate of convergence

fourier feature learning, approximation
table of convergence comparisons with flops
weight distirbution plot

then adam sgd algorithm for hierarchical feature learning
PDEBench, PDEGym
(we then expose leap complexity in the future, lee/krz)







'then i add all the remarks done previously'



bien expliquer les assumptions car c'est le coeur applicatif sinon on se fait rejeter
NON convex loss === PINS !!!!!! and this done, this is how we gonna sell the idea
the wholte strat is non convex loss global convergence is allowed by low rank  ! 




# Revised Paper Strategy (ICML 2026) - EXPERIMENTAL FOCUS

## Title:
**"Global Convergence of Mean-Field Dynamics for Low-Rank Random Feature Networks"**

## Core Contribution (ONE sentence):
We develop mean-field theory for low-rank random feature networks, outline proof strategies for global convergence, and provide comprehensive numerical experiments validating the theoretical predictions.

## Paper Structure (Simple):
1. **Introduction** (1 page)
2. **Related Work** (1 page)
3. **Mean-Field Theory and Proof Ideas** (2.5 pages)
4. **Numerical Experiments** (3 pages) ← **ALL PLOTS HERE**
5. **Conclusion** (0.5 pages)

---

## Paper Structure (8 pages main body)

### 1. Introduction (1 page)
- **Motivation**: 
  - Mean-field theory for neural networks
  - Computational efficiency of low-rank structure (O(rN) vs O(N²))
  - Need for convergence guarantees
- **Main contributions**: 
  - Mean-field ODE derivation for low-rank RF networks
  - Proof ideas for well-posedness and global convergence
  - Comprehensive numerical validation on multiple datasets
- **Paper organization**: Brief roadmap

### 2. Related Work (1 page)
- **Mean-field theory for neural networks**:
  - Classical results (Mei et al., Rotskoff & Vanden-Eijnden, etc.)
  - Well-posedness and convergence results
  - Limitations of existing work
- **Low-rank neural networks**:
  - Computational efficiency
  - Expressivity questions
  - Training dynamics
- **Random features and NTK**:
  - Kernel methods connection
  - Infinite-width limits
- **Gap**: Our work combines low-rank structure with mean-field theory

### 3. Mean-Field Theory and Proof Ideas (2.5 pages)
- **3.1 Setup and Architecture**:
  - 3-layer low-rank random feature network
  - First layer: random features (frozen)
  - Second layer: low-rank mixing matrix L (rank r)
  - Third layer: trainable weights
  - Mean-field limit: N→∞ limit, measure-valued ODE system
  
- **3.2 Mean-Field ODE Derivation**:
  - Derivation of the ODE system
  - Key structure: Low-rank structure multiplies constants by r
  
- **3.3 Well-Posedness (Proof Ideas)**:
  - **Theorem 3.1 (Existence and Uniqueness)**: 
    - Statement: Under conditions [X, Y, Z], solutions exist and are unique
    - **Proof strategy**:
      - Picard iteration approach
      - A priori bounds using ψ₂ norms
      - Contraction mapping argument
    - Key lemma: Low-rank structure only multiplies constants by r
  
- **3.4 Global Convergence (Proof Ideas)**:
  - **Theorem 3.2 (Global Convergence)**: 
    - Statement: Under conditions [initialization, learning rate, data], dynamics converge to global minimizer
    - **Proof strategy options**:
      - **Option A (Lyapunov)**: Construct energy function E(t) = ||W(t) - W*||², show dE/dt ≤ -cE
      - **Option B (Convexity)**: Show loss is convex in Wasserstein metric, use gradient flow theory
      - **Option C (ODE stability)**: Use LaSalle's invariance principle
      - **Option D (Monotonicity)**: Show loss decreases monotonically, use compactness
    - Conditions needed:
      - Initialization: small enough or specific distribution
      - Learning rate: constant/decaying schedule
      - Data: bounded support, regularity
  
- **3.5 Rate of Convergence (Proof Ideas)**:
  - **Theorem 3.3 (Convergence Rate)**:
    - Statement: ||W(t) - W*|| ≤ C exp(-λt) or C t^{-α}
    - **Proof strategy**:
      - From Lyapunov analysis: extract decay constant
      - Or: From gradient flow: use condition number
    - Dependence on: condition number, learning rate, rank r, data distribution
    - **Conjecture**: Rate depends on rank r as [formula]

### 4. Numerical Experiments (3 pages) 📊 **ALL PLOTS HERE**
- **4.1 Experimental Setup**:
  - Implementation details
  - Datasets: [list]
  - Hyperparameters: learning rates, initialization schemes, ranks r tested
  
- **4.2 Convergence Rate Verification**:
  - **Figure 1**: Loss vs time (log scale) for different ranks r
    - Multiple datasets
    - Show exponential/polynomial decay
  - **Figure 2**: Fitted convergence rates (λ or α) vs rank r
    - Compare theoretical prediction vs empirical
  - **Table 1**: Empirical convergence rates across datasets
    - Exponential decay constant λ (or polynomial exponent α)
    - Dependence on rank r
    - Dependence on learning rate
  
- **4.3 Hypothesis Testing**:
  - **Figure 3**: Effect of initialization
    - Different initialization schemes
    - Verify theoretical conditions
  - **Figure 4**: Effect of learning rate schedule
    - Constant vs decaying
    - Match theoretical requirements
  - **Figure 5**: Effect of data distribution
    - Different datasets
    - Which satisfy theoretical assumptions?
  - **Table 2**: Summary of hypothesis validation
    - Which hypotheses hold on which datasets?
  
- **4.4 Rank Dependence**:
  - **Figure 6**: Convergence speed vs rank r
    - Does larger r help or hurt?
    - Compare to theoretical prediction
  - **Figure 7**: Final loss vs rank r
    - Expressivity vs efficiency trade-off
  
- **4.5 Additional Experiments** (if space allows):
  - Weight distribution plots
  - Computational efficiency (flops comparison)
  - Comparison with full-width networks

### 5. Conclusion (0.5 pages)
- Summary of contributions:
  - Mean-field theory for low-rank RF networks
  - Proof ideas for convergence
  - Experimental validation
- Limitations:
  - Proofs are ideas, not complete (if applicable)
  - Assumptions on data/initialization
- Future directions:
  - Complete proofs
  - Extensions to deeper networks
  - Applications to PDEs, etc.

---

## Appendix (unlimited pages)
- Detailed proof sketches (if space allows in main text, move here)
- Additional experimental details:
  - Dataset descriptions
  - Hyperparameter settings
  - Additional plots
  - Computational efficiency tables
- Technical lemmas
- Implementation details

---

## Critical Components

### 1. Proof Ideas (MUST HAVE) ⚠️
**Need clear proof strategies (even if not complete):**
- **Well-posedness**: 
  - Picard iteration approach
  - A priori bounds
  - Contraction mapping
- **Global convergence**: 
  - Choose one approach (Lyapunov, convexity, ODE stability, monotonicity)
  - Outline the key steps
  - Identify what conditions are needed
- **Rate of convergence**:
  - How to extract rate from proof strategy
  - Conjecture on rank r dependence

**What you need:**
- Clear statement of theorems (even if proofs are incomplete)
- Precise conditions: initialization, learning rate, data distribution
- Proof strategy that is plausible and well-motivated
- Connection between proof ideas and experimental observations

### 2. Comprehensive Numerical Experiments (MUST HAVE) ⚠️
**This is the main contribution - all plots here:**
- **Convergence rate verification**:
  - [ ] Loss vs time plots (log scale) for different ranks r
  - [ ] Fit exponential/polynomial rates
  - [ ] Compare across multiple datasets
  - [ ] Show rank r dependence
  
- **Hypothesis testing**:
  - [ ] Initialization: Test different schemes, show which work
  - [ ] Learning rate: Constant vs decaying, show effect
  - [ ] Data assumptions: Test on different datasets, show which satisfy assumptions
  - [ ] Rank dependence: Comprehensive study of r effect
  
- **Additional plots**:
  - [ ] Weight distribution plots
  - [ ] Computational efficiency (flops, memory)
  - [ ] Comparison with baselines (full-width, other methods)

- **Datasets**:
  - [ ] Synthetic: Controlled experiments
  - [ ] Real-world: Multiple regression/classification tasks
  - [ ] Varying sizes: Test robustness

### 3. Connection Between Theory and Experiments (MUST HAVE) ⚠️
- **Validate proof ideas**:
  - Do experiments support the proof strategy?
  - Do conditions match what's needed?
- **Inform theory**:
  - What do experiments suggest about convergence?
  - What conditions seem necessary?
- **Bridge gap**:
  - Mean-field limit vs finite-width
  - Theory vs practice

---

## Red Flags to Address

1. ❌ "proof to write" → Must have clear proof ideas/strategy (even if not complete)
2. ❌ "we then expose leap complexity in the future" → Remove from main paper
3. ❌ Vague titles → Pick one clear contribution (mean-field convergence + experiments)
4. ❌ Missing experimental validation → Must have comprehensive plots validating theory
5. ❌ Unclear connection theory-experiments → Must clearly connect proof ideas to numerical results
6. ❌ Incomplete experiments → All plots must be in numerical experiments section

---

## Strengths to Emphasize

1. ✅ Rigorous mean-field derivation (ODE system is well-defined)
2. ✅ Clear proof strategies for convergence (even if not complete)
3. ✅ Comprehensive numerical validation (all plots in one section)
4. ✅ Connection between theory and experiments
5. ✅ Computational efficiency (O(rN) vs O(N²) for full-width)

---

## Timeline Recommendation

**Week 1**: Develop proof ideas/strategies:
  - Well-posedness: Outline Picard iteration approach
  - Global convergence: Choose and develop one strategy (Lyapunov/convexity/etc.)
  - Rate: Outline how to extract rate from proof
  
**Week 2-3**: Design and run comprehensive experiments:
  - Convergence rate verification (fit rates, multiple datasets)
  - Hypothesis testing (initialization, learning rate, data assumptions, rank r)
  - All plots: loss curves, rate comparisons, rank dependence, etc.
  - Additional: weight distributions, efficiency comparisons
  
**Week 4**: Write full paper:
  - Introduction, Related Work
  - Mean-field theory + proof ideas
  - Numerical experiments (all plots)
  - Conclusion
  - Connect theory to experiments
  
**Week 5**: Revise based on feedback, polish

**DO NOT SUBMIT** until:
1. Proof ideas are clear and well-motivated
2. All numerical experiments are complete (all plots ready)
3. Connection between theory and experiments is clear
4. All sections are well-written and integrated

---

## Numerical Experiments Checklist

### All Plots Must Be in Section 4 (Numerical Experiments):

#### 4.2 Convergence Rate Verification:
- [ ] **Figure 1**: Loss vs time (log scale) for different ranks r
  - Multiple datasets
  - Show exponential/polynomial decay clearly
- [ ] **Figure 2**: Fitted convergence rates (λ or α) vs rank r
  - Compare to theoretical prediction (if available)
- [ ] **Table 1**: Empirical convergence rates
  - Across datasets
  - Dependence on rank r
  - Dependence on learning rate

#### 4.3 Hypothesis Testing:
- [ ] **Figure 3**: Effect of initialization
  - Different schemes
  - Which satisfy theoretical conditions?
- [ ] **Figure 4**: Effect of learning rate
  - Constant vs decaying
  - Match theoretical requirements
- [ ] **Figure 5**: Effect of data distribution
  - Different datasets
  - Which satisfy assumptions?
- [ ] **Table 2**: Summary of hypothesis validation

#### 4.4 Rank Dependence:
- [ ] **Figure 6**: Convergence speed vs rank r
  - Does larger r help or hurt?
- [ ] **Figure 7**: Final loss vs rank r
  - Expressivity vs efficiency trade-off

#### 4.5 Additional (if space):
- [ ] Weight distribution plots
- [ ] Computational efficiency (flops, memory)
- [ ] Comparison with baselines

### Datasets:
- [ ] Synthetic: Controlled experiments
- [ ] Real-world: Multiple regression/classification tasks
- [ ] Varying sizes: Test robustness

### Connection to Theory:
- [ ] Do experiments support proof ideas?
- [ ] Do conditions match what's needed?
- [ ] What do experiments suggest about convergence?

