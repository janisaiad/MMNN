
### TODO - EXPERIMENTAL (12 items):

**NEW SECTION ADDED (2025-01-11): Section 6.6 "Comparison with MLPs: NTK spectrum and stepwise dynamics"**
- Comprehensive comparison framework between MMNN and MLP architectures
- 4 new experiments (10-13) covering NTK spectrum, stepwise dynamics, landscape geometry, and performance
- All marked as TO BE COMPLETED with clear theoretical predictions and experimental setup
- 3 figure placeholders added (TO BE GENERATED)
- quantifying sharpness with central flows


**Original items (1-7):**
1. Experiments confirming NTK training behavior for practical tasks
2. Empirical validation of $r \in [5,50]$ for good concentration bounds
3. Comparison with/without finite width corrections
4. Spherical 2d/3d experiments for cosine distance
5. Test growing $r$ makes landscape have exponentially more minima
6. Validate low rank allows good minima exponentially in $r$
7. General empirical validations across different tasks/dimensions

**New MLP comparison experiments (8-11):**
8. **Experiment 10 - NTK spectrum comparison MMNN vs MLP**: Compare NTK eigenvalue distributions at initialization, verify $O(1/r^2)$ vs $O(1/N)$ concentration, analyze outlier structure and bulk spectrum (Marchenko-Pastur), plots TO BE GENERATED
9. **Experiment 11 - Stepwise loss dynamics MMNN vs MLP**: Compare training loss curves on multi-index targets, verify MMNN shows clear staircase behavior with $O(r)$ drops while MLP smoother, analyze convergence speed differences, side-by-side plots TO BE ADDED
10. **Experiment 12 - Landscape geometry comparison MMNN vs MLP**: Compare Hessian spectrum evolution, $\lambda_{\max}(H_t)$ through training, landscape convexification rates, sharpness of final minima, condition number analysis, Hessian eigenvalue plots TO BE ADDED
11. **Experiment 13 - Performance at matched parameter budgets**: Test accuracy on MNIST/CIFAR-10 with matched total parameters, verify MMNN competitive/superior at $O(rn)$, efficiency metric showing MMNN with $r \approx 20$-$30$ matches MLP with $10\times$-$50\times$ more parameters, performance curves TO BE ADDED

**NEW (2025-01-11): Two-layer width scheme investigation:**
12. **Experiment 3b - Two-layer width scheme investigation (ADDED TO SECTION 6.1)**: For 2-layer MMNN ($L=2$), systematically vary width configurations $(n_1, n_2)$ and rank $r$ to test finite-width NTK predictions. Compare: (i) theoretical NTK with finite-width corrections $O(1/n) + O(1/r)$, (ii) empirical NTK via finite differences at initialization, (iii) training dynamics deviation from kernel regression predictions. Width schemes: $n_1, n_2 \in \{128, 256, 512, 1024, 2048\}$ with $r \in \{5, 10, 20, 30, 50\}$. Measure: (a) relative error $\|\Kop_{\text{empirical}} - \Kop_{\text{theory}}\|_F / \|\Kop_{\text{theory}}\|_F$, (b) deviation from kernel regime $\|f_t - f_{\text{NTK}}(t)\|_{L^2}$, (c) variance of NTK entries across random initializations to verify $O(1/r^2)$ concentration bound. Identifies minimal $(n, r)$ for accurate kernel predictions. **Placeholder added, TO BE COMPLETED**

### TODO - THEORETICAL (15 items):
1. **Formalize factor-of-2 NTK reduction from partial training (NEW - ADDED TO TEXT)**: By training only $\bm{A}^{(\ell)}$ (output weights) and freezing $\bm{w}^{(\ell)}$ (input weights), the effective NTK magnitude is approximately half: $\Kop_{\text{MMNN}} \approx \frac{1}{2} \Kop_{\text{full}}$. This halves kernel eigenvalues in kernel regression regime, doubling effective learning rate. Need explicit computation showing this factor-of-2 for comparable architectures. **Remark added to Section 2.1, marked TO BE FORMALIZED**
2. Non-gaussian process propagation analysis
3. Explicit NTK bulk formula in normalized $\times r^2$ manner
4. Curse of dimensionality impact on norm (hypercube vs spherical)
5. NTK std impact, which NTK useful for which case (sphere $Tr=1$)
6. Full analysis on 2 mmnn layers with infinite layer NTK
7. Concentration bound $2 \times NTK_{MMNN} - NTK_{MLP}$ explicit formula
8. NTK for DSRN (Deep Structured Random Networks)
9. Full spectrum of kernel random matrices analysis (beyond citation)
10. Complete Theorem 3.2 (concentration bounds marked TO BE FILLED)
11. Complete NTK outlier analysis (ben arous style)
12. Complete finite width NTK parts marked TO BE FILLED
13. Stiefel manifold assumption and implications
14. Trainable $A$ as vector in high-dim feature space (learningquadratic connection)
15. Landscape autosimilarity formalization
16. **NTK derivation via Feynman diagrams and tensor programs (NEW - 2025-01-11)**: Develop systematic NTK computation rules using Feynman diagram techniques combined with tensor program formalism for $\mu$-parameterization. Specific goals: (i) derive diagrammatic expansion rules for low-rank random feature architectures, (ii) establish tensor program framework for MMNN with rank-$r$ bottlenecks under $\mu$-parameterization scaling, (iii) compute global NTK theory connecting diagram orders to width/rank scalings, (iv) compare Feynman vs. traditional recursive NTK derivations for computational efficiency, (v) extend to arbitrary depth $L$ and heterogeneous width schemes. This provides a unified computational framework for NTK analysis across parameterizations and architectures, particularly powerful for analyzing finite-width corrections systematically.

### TODO - EXTENSIONS & FUTURE (3 items):
1. Finite width corrections scaling with PMISOF comparison
2. Combine high-dim stats and Feynman for scaling laws
3. Finite width corrections to explain mean field training

### MOVED TO FUTURE CONTRIBUTIONS:
- Mean field theory (comprehensive treatment)
- Global convergence theory (quantitative results)
- Central flow dynamics
- Dynamical stability analysis
- Glass theory for multiple minima
- Wasserstein gradient flows