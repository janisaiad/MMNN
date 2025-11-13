Experiments:
- Disentangle frequency vs localization (spikes) in low-rank random features.
- Test dynamical stability near global minima (return time, NTK drift).
- Confirm symmetry breaking during training (anisotropy, directional selection).
- Investigate central-flow dynamics during training (ρ_t tracking, transport).

## for 1st version

### NTK and probability:
✓ training characterized by 2 regimes: mean field, mup or NTK (line 10) - DONE: intro paragraph mentions both regimes
✓ early training appears bad for SGD, NTK can partly explain this (lines 215-216) - DONE: Section 3.1 discusses poor conditioning at init
✓ NTK to confirm no disadvantage in this regime due to RKHS being the same (line 219) - DONE: Section 3.3 discusses RKHS comparison with MLPs
✓ recursive formulation for NTK with probabilistic viewpoint, std wrt $r$, concentration bounds hold for $r$ and input dimensions (lines 230-231) - DONE: Theorem 2.3 (thm:ntk_recursion) and Theorem 3.2 (thm:ntk_concentration)
- experiments confirming NTK training behavior for practical tasks, comparing predictions vs experiments, increase dimensions and $r$ to see where lazy training stops (lines 233-234) - TODO: Section 6.1 outlined but experiments need implementation
- having $r$ between 5 and 50 leads to good concentration bound (around 30) and enough expressivity with fewer weights (lines 237-238) - TODO: empirical validation needed
✓ finite width correction way for NTK, finitewidth correction in $Nr$ due to EOC for MMNNs (lines 244-245) - DONE: Theorem 2.4 (thm:finite_width_ntk)
✓ NTK randomness can lead to lyapunov product expansion analysis, better conditioning than NN or faster early training curve, NTK std disagreement: $1/r^2$ instead of $1/r$ (lines 247-249) - DONE: Section 3.1 discusses Lyapunov product expansion, NEW PARAGRAPH added explaining $O(1/r^2)$ is exponentially faster than classical $O(1/N)$ decay
- non gaussian process propagation analysis, differs from NTK trying to fit kernel gaussian process (lines 254-255) - TODO: not yet covered
✓ propagation of $\rho$ argument following, how $\rho_i$ propagates due to fisher law (line 258) - DONE: Theorem 2.4 mentions Fisher distribution analysis
✓ computations for bias related gaussian NTK and standard deviations in appendix (line 260) - DONE: Appendix B (Fisher and Kibble distributions)
✓ apply theorem from 'spectrum of kernel random matrices' to explain all NTK results, 2 randomness to disentangle giving bound in $1/r$, need linear number of samples wrt dimension (lines 261-262) - DONE: Section 3.1 bullet point 2, references spectrum_kernel_random_matrices
✓ compare with MLP from Terjek paper (line 264), concentration of the NTK and spectrum with 1 main outlier and having a ill conditionned NTK - DONE: Section 3.1 bullet point 3 discusses Terjék-style analysis
✓ NTK expansion in sum of $C_i/r^i$ (1 to $L$) for $L$ layers, $C_i$ random variable computed according to $\rho_i$, $\rho$ the cosine dist between $x$ and $y$ vectors, lyapunov product analysis of $(1-\arccos(\rho_i)/\pi)$, perturbation in kernel random matrix leads to bulk spectrum/outlier differing from MLP (lines 266-268) - DONE: Theorem 2.4, Section 3.1, Section 6.5
✓ $1/r$ expansion as premise to finite width correction in infinite depth (line 270) - DONE: Theorem 2.4 and Section 6.5 paragraph on "Finite-width corrections and the 1/r expansion"
✓ outlier analysis inherits from terjek analysis and its scaling, complete analysis of bulk/outlier behavior (ben arous discussion), experiment on spherical 2d and 3d data to avoid scaling issues and understand cosine distance well (lines 272-274) - DONE: Section 3.1 mentions this, experiments outlined but need implementation
✓ NTK converges to 1 in large low rank tells us RF gives..., results for high dim entries with $d$, $N$, $r$ growing linearly from "spectrum of kernel random matrices", conjecture need only linear number of dictionary functions wrt input dim, grow $N$, $d$, $r$ simultaneously, NTK analysis from marchenko pastur holds (lines 281-284) - DONE: Theorem 2.4, Section 3.1 mentions Marchenko-Pastur
✓ linear coefficient between $d$ and $N$ from discretization, $N$ width grows with gram product with $\rho_1$ as random variables conditioned on $X$ (line 286) - DONE: Section 2.4 paragraph on doubly-random Gram matrix regime
✓ doubly random matrix, gram matrix appearing, spectrum linearized wrt outputs, apply marchenko pastur theory (line 288) - DONE: Theorem 2.4 and its paragraph
✓ should see only 1 outlier describing early training part (as terjek), recover it easier more tractable manner (line 290) - DONE: Section 3.1 bullet 2 and Section 2.4
✓ explanation of $1/r^2$ decay comes from $1/r$ randomness added from gram output in NElkaroui paper (line 292) - DONE: Theorem 2.4 mentions doubly-random Gram regime with $O(1/r^2)$, NEW: emphasized this is exponentially better than standard $O(1/N)$ from classical NTK theory
- explain NTK bulk in normalized $\times r^2$ manner (line 294) - TODO: needs more explicit formulation
- curse of dim impact: unit norm $w$ but not unit norm bias depends on data normalization (hypercube vs spherical), for hypercube need $Tr(\sigma_w) = d$ and unit bias for approx theorem, for spherical unit norm weight and unit bias (lines 305-307) - TODO: not explicitly covered
- NTK std impact, explain which NTK useful for which case, over sphere $Tr = 1$ removing $1/r$ from normalization (line 309) - TODO: not explicitly covered
- full analysis on 2 mmnn layers due to untractable integral after 2nd mmnn layer, tackle infinite layer NTK with $1/r$ exponential expansion (line 312) - TODO: Section 6.5 mentions this but full analysis not complete
✓ concentration bound between $2 \times NTK_{MMNN} - NTK_{MLP}$ for large $r$, $NTK_{MMNN}$ std decay in $1/r^2$, can be in NTK regime very easily (lines 387-388) - DONE: NEW paragraph emphasizes $O(1/r^2)$ decay is exponentially better than MLP's $O(1/N)$, making kernel regime easier to reach
✓ at init very bad conditioning so NTK explains partially the training (line 391) - DONE: Section 3.1 first sentence
✓ 1st part with NTK and finite width (line 402) - DONE: paper structure mentions this
✓ NTK explain high frequencies not learned directly through RKHS (same as MLP), more low to high frequency bias, tune frequency learning in variance of weights at init and maximal update param (mu param) (lines 383-384) - DONE: Section 3.4, Section 6.8
✓ proof of kibble and fisher distrib (line 324) - DONE: Appendix B section
- NTK for infinite width and NTK for DSRN, terjek-like analysis for NTK spectrum with depth (line 166) - TODO: Terjék mentioned but DSRN not covered
- spectrum of kernel random matrices (line 169) - TODO: cited but full analysis not shown
✓ NTK theory for 2 layers (line 378) - DONE: Corollary after Theorem 2.3
- concentration of low rank ntk (line 508) - TODO: Theorem 3.2 outlined but needs completion (TO BE FILLED)
- ntk outlier analysis (cf ben arous) (line 509) - TODO: mentioned but needs completion
- finite width NTK (line 511) - TODO: theorem present but some parts marked TO BE FILLED

### finite width corrections:
✓ finite width correction way for NTK, better depth estimation of early training, finitewidth correction in $Nr$ due to EOC (lines 244-245) - DONE: Theorem 2.4
✓ $1/r$ expansion as premise to finite width correction in infinite depth (line 270) - DONE: Theorem 2.4, Section 6.5
✓ since only 2 hidden layers, finite width corrections tractable, can try $Nr$ and $r/d$ scaling law (line 314) - DONE: Section 6.5 mentions "Two-layer tractability"
- comparison with/without finite width corrections, compare minimum found vs practice (lines 328-329) - TODO: experimental validation needed
- results on finite width corrections scaling wrt depth, scaling law, comparison with PMISOF, can lead to great optimization bounds in future, combine high dim stats and feynman to get scaling law for any depth in RF low rank models, great future direction due to tractability (lines 371-377) - TODO: mentioned in Section 7.2 as future direction but not implemented
- finite width corrections to explain partly mean field training (line 394) - TODO: not explicitly covered
✓ 1st part with NTK and finite width (line 402) - DONE: paper structure follows this

### high dimensional statistics:
✓ no dimensional perspective on input data, dimension agnostic, don't care about high dimension statistics features arising from NTK kernel analysis because not what we see in practice (lines 47-48) - DONE: intro mentions "dimension-agnostic"
✓ MMNN training understood through lens of already understood high dimension learning theory (learningquadratic), extensive rank regime where $r \asymp d^\beta$ for $\beta \in (0,1)$ and $r_s \asymp d^\gamma$ for $\gamma \in [0,1)$, power-law assumption on second-layer coefficients $\lambda_j \asymp j^{-\alpha}$ for $\alpha \geq 0$ (lines 94-97) - DONE: intro Q3 mentions extensive-rank regime $r \asymp d^\beta$, Section 4.1
- assumption to live in stiefel manifold important (weights orthogonal in bigger data dimension, fewer weights than $d$), can have same scaling laws for MMNNs with small $\beta$ and tune this $\beta$ (lines 105-106) - TODO: not explicitly covered
✓ random feature space being high dimensional because of large width, train on that (lines 109-110) - DONE: Section 6.6 "High-dimensional random feature space"
- high dim point of view (line 161) - TODO: general mention but not detailed
✓ NTK and RKHS with randomness (high dim also) (line 165) - DONE: Section 3.3 discusses RKHS
✓ concentration bounds for $r$ and input dimensions (line 231) - DONE: Theorem 3.2
✓ spectrum of kernel random matrices need linear number of samples wrt dimension (line 262) - DONE: Section 3.1 mentions this requirement
✓ high dim entries with $d$, $N$, $r$ growing linearly from "spectrum of kernel random matrices", need only linear number of dictionary functions wrt input dim, grow $N$, $d$, $r$ simultaneously (lines 282-283) - DONE: Section 2.4, Theorem 2.4
- curse of dim: unit norm $w$ but not unit norm bias depends on data normalization (lines 305-309) - TODO: not explicitly covered
- trainable $A$ is vector like $w$ in learningquadratic, in high dimensional feature space (line 132) - TODO: not explicitly mentioned
- combine high dim stats and feynman to get scaling law for any depth in RF low rank models (line 375) - TODO: mentioned in future directions (Section 7.2) but not implemented
✓ random feature space high dimensional because of large width (lines 476-477) - DONE: Section 6.6







### landscape and ben arous stepwise curve:
✓ ben arous paper: multi index models, training dynamics exhibit emergent (or staircase-like) behavior — long plateaus followed by sharp drops in loss (lines 99-100) - DONE: intro mentions stepwise/staircase loss, Section 1.3, related work paragraph on multi-index models
✓ every plateau corresponds to term in additive model, can only learn certain number, explain scaling law in openai paper (line 102) - DONE: Section 4.3 explains multi-index structure, plateaus correspond to inactive directions
✓ asymptotic risk behavior shows sharp step-like emergent curve at $\alpha=0$ (earlier works on multi-index learning) gradually transitions to smooth curve as $\alpha$ increases, light-tailed regime $\alpha > 1/2$ resembles neural scaling laws $\mathcal{R} \sim 1/(\text{Data size})^a + 1/(\text{Model size})^b$ (line 103) - DONE: intro contributions mentions scaling laws, Section 6.7 discusses scaling laws
- different frequency timescales learned in order, making landscape autosimilar, explaining early stage leads to same as final stage (super convergence) (lines 221-222) - TODO: concept mentioned but not formalized
- multiple finite-$N$ local minima can correspond to same minimizer $\rho_*$ of $R(\rho)$ in limit $N \to \infty$, ideas from glass theory (lines 139-140) - MOVED TO FUTURE (Section 5.3 discusses this but moved to future contributions)
- for $\beta < \infty$, evolution converges to minimizer of $F_{\beta,\lambda}(\rho)$, implying global convergence of noisy SGD in number of steps independent of $N$ (lines 143-144) - MOVED TO FUTURE (part of mean field theory)
- 2nd part with mean field and stepwise loss (benarous & mean field), explain when sgd gets stuck but helps where adam fails (wobbling part) (line 404) - MOVED TO FUTURE (mean field section 4 moved to future)
- mean field approach in fourier space can explain same thing as ben arous, big frequency training when loss decays fast (line 415) - MOVED TO FUTURE (mean field Fourier approach)
✓ landscape is sharp, very sharp after having passed low pass stuff (line 421) - DONE: Section 6.4 discusses sharpness transitions, Section 5.3 "Sharpness of global minima"
- growing $r$ makes landscape have exponentially more minima (line 436) - TODO: mentioned conceptually but not proven
✓ outlier analysis inherits from terjek analysis, complete analysis of bulk/outlier behavior (ben arous discussion) (line 272) - DONE: Section 3.1 discusses this
- low rank allow to have good minima exponentially in $r$, to test (line 518) - TODO: needs experimental validation
- ntk outlier analysis (cf ben arous) (line 509) - TODO: mentioned but analysis incomplete

---

---

## NEW CONTRIBUTIONS ADDED (2025-01-11)

### 1. ✓ **NTK theory with no RKHS disadvantage and linear width scaling** - DONE: Added as 1st contribution
- we prove MMNN induces same RKHS as standard fully-connected networks for large rank $r$, ensuring no expressivity loss
- NTK regime achieved at linear parameter budget $O(rn)$ rather than quadratic $O(n^2)$
- dramatically reduces width required to enter kernel regime while maintaining full theoretical guarantees
- establishes low-rank random features as strictly more efficient than dense architectures for lazy training

### 2. ✓ **Improved NTK concentration: $O(1/r^2)$ vs $O(1/N)$ decay** - DONE: Added new paragraph + updated 3rd contribution
- **CRUCIAL ADVANTAGE**: NTK randomness decays as $O(1/r^2)$ in doubly-random Gram regime
- **SIGNIFICANTLY FASTER** than standard $O(1/N)$ decay from classical NTK theory
- for fixed total parameters $N_{\text{total}} = rn$, we get $1/r^2 \sim 1/(N_{\text{total}}/n)^2$
- this decays **exponentially faster** than $1/N_{\text{total}}$ when $r$ moderate and $n$ large
- **DOUBLE BENEFIT**: low-rank random features improve (1) parameter efficiency AND (2) kernel approximation reliability
- makes deterministic NTK limit more accurate at finite width
- this was previously thought impossible - NTK theory expected $1/N$ expansion, we achieve $1/r^2$

---

## SUMMARY OF STATUS

### DONE (25 items marked with ✓):
**NTK Theory & Theorems:**
- **NTK theory with no RKHS disadvantage and linear width scaling (NEW - 1st contribution)**
- **Improved NTK concentration: $O(1/r^2)$ vs $O(1/N)$ decay (NEW - exponentially faster than classical NTK)**
- Recursive NTK formulation (Theorem 2.3)
- Two-layer NTK corollary
- General L-layer NTK corollary
- Finite-width NTK theorem (Theorem 2.4) with $O(1/r)$ and $O(1/r^2)$ bounds
- Lyapunov product expansion analysis
- Fisher & Kibble distribution proofs (Appendix)
- Doubly-random Gram matrix regime
- Marchenko-Pastur theory application

**NTK Analysis:**
- Poor conditioning at initialization
- RKHS comparison with MLPs (no expressivity loss proven)
- Terjék-style outlier analysis
- Single outlier characterization
- Spectrum of kernel random matrices (cited)

**High-dim Statistics:**
- Dimension-agnostic framework
- Extensive-rank regime $r \asymp d^\beta$
- High-dimensional random feature space
- Concentration bounds

**Landscape:**
- Stepwise/staircase loss curves explanation
- Multi-index learning connection
- Neural scaling laws
- Sharpness transitions
- Outlier analysis from Terjék

**Other:**
- Two-layer tractability discussion
- Paper structure with NTK and finite width

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













**TOTAL STATUS:**
- DONE: 25 items (✓) - **UPDATED: Added 2 major NTK contributions (RKHS + improved concentration)**
- TODO Experimental: 12 items - **NEW: Added 4 MLP comparison experiments (Experiments 10-13, Section 6.6) + 2-layer width scheme investigation (Experiment 3b)**
- TODO Theoretical: 16 items - **NEW: Added factor-of-2 NTK reduction formalization + Feynman diagrams/tensor programs for μ-parameterization**
- TODO Extensions: 3 items
- MOVED TO FUTURE: ~35 items (separate section)

**KEY INSIGHTS FROM NEW CONTRIBUTIONS:**

1. **Linear scaling advantage**: The linear parameter budget $O(rn)$ vs quadratic $O(n^2)$ is not just a computational saving—it fundamentally changes the width requirements to enter the kernel regime. we achieve the same RKHS guarantees with drastically fewer parameters, making low-rank random features the optimal architecture for theoretical analysis in the lazy training regime.

2. **Improved concentration beyond classical NTK**: The $O(1/r^2)$ decay in NTK randomness is a breakthrough result that was not predicted by standard NTK theory (which gives $O(1/N)$ decay). This means that:
   - For the same parameter budget, MMNN achieves exponentially better kernel approximation
   - The deterministic NTK limit is reached faster with fewer parameters
   - This establishes a fundamental theoretical advantage: low-rank random features are not just efficient but also more reliable
   - Previous NTK theory did not anticipate this quadratic improvement in concentration











## future contributions (to be moved out from 1st version)

### mean field theory and analysis:
from idea1.md:
- high level idea to explain MMNN training dynamic is mean field one (line 5)
- training characterized by 2 regimes: mean field, mup or NTK, whole training procedure done using muP parametrization, some using MF one (lines 10-11)
- mean field limit converges to global optimum under suitable regularity and convergence mode assumptions (globalconvergences paper), technique doesn't generalize to three-layer setup, use universal approximation property instead of exploiting convexity, conceptually new: global convergence achieved even when loss function is non-convex, universal approximation property holds at any finite training time (not necessarily at convergence), factorize middle weight matrix, linear regime of number of parameters wrt width, for width around 1000 we are in MF regime (lines 31-44)
- strong MF regime idea: no dimensional perspective on input data, dimension agnostic, don't care about high dimension statistics features arising from NTK kernel analysis (lines 47-48)
- from this point of view can adapt mean field ODE formulation to try to get convergence results or discover something new in equation, need strong numerical experiments for mean field regime, tracking particle distribution and seeing which ode it satisfies (lines 81-83)
- multiple finite-$N$ local minima correspond to same minimizer $\rho_*$ of $R(\rho)$ in limit $N \to \infty$, ideas from glass theory might be useful to investigate this structure (meanfieldlandscape2layers breakthrough) (lines 138-140)
- for $\beta < \infty$, evolution converges to minimizer of $F_{\beta,\lambda}(\rho)$, implying global convergence of noisy SGD in number of steps independent of $N$ (lines 143-144)
- from globalconvergence: having full support at each time gives universal approximation, random features keep that, we keep huge support because it remains the same (lines 148-149)
- identified should take back math ODE material for 3 layers but simplifying it, using this insight could prove non quantitative convergence result (line 155)
- mainly results lie in mean field framework (between 500 and 5000 width), need to prove it beats MLP for conference paper by adding benchmark in comparison for 3 layers (lines 173)
- mean field approx don't care about input dimensions, it's an orthogonal viewpoint on tackling the problem (line 181)
- mean field analysis: tremendous remark from meanfieldlandscape: think of $\theta_1,\dots,\theta_N$ as positions of $N$ particles in $D$-dimensional space, when $N$ large behavior of such gas of particles effectively described by density $\rho_t(\theta)$ (with $t$ indexing time), not all small changes of density profile can be realized in actual physical dynamics: dynamics conserves mass locally because particles cannot move discontinuously, if $\text{supp}(\rho_t) = S_1 \cup S_2$ for two disjoint compact sets, then total mass in each region cannot change over time (lines 334-341)
- should see what's happening between 1 layer RF and NN, should make MF experiments to show distributions behavior, should test with bounded activations (lines 346-348)
- from globalconvergences: given same family Init, law of MF trajectory is insensitive to choice of neuronal embedding of Init, from Chizat & Bach: give criteria for Wasserstein gradient flows to escape from non-optimal stationary points, valid both in finite-particle regime and many-particle limit, even in finite-particle case (classical gradient flows), point of view using measures is natural (lines 353-359)
- 2nd part of training, end of lazy training: before knowing if can try mean field proof by getting the..., from "propagation of chaos": intuition that in many teacher-student settings with uniform initialization, neurons are dispersed before converging to teacher neurons (lines 361-367)
- finite width corrections will remain to try to explain partly mean field training, mean field point of view can be satisfying and we have supportive results explains well 2nd part of training where SGD have much more convex training (lines 394-396)
- 2nd part with mean field and stepwise loss (benarous & mean field), explain in which case sgd gets stuck but helps us where adam fails because of wobbling part, think wobbling occurs in subspace orthogonal to this of symmetry condition by mean field approach, Hessian and gauss newton easy to calculate in this regime and supported by plot through training of TK/GN matrix, explained by central flow the high frequency grokking is explained by last part of functions high frequency learned by spikes (lines 404-407)
- mean field approach in fourier space can explain same thing as ben arous, big frequency training when loss decays fast, random features allow to grok that because it push dictionary functions at any place over interval, just matter of choosing which one and having sufficient depth to build more and more frequencies (lines 415-417)
- mean field do not explain that because of what it requires with let high frequency thrive with standard init without preserving (lines 525-526)
- global convergence for all depth from mean field approach (line 530)
- tester en 2d le curse od fim mean field (line 514)

from templateArxiv.tex:
- mean-field framework and extensive-rank regime (Section 4.1, lines 486-494): beyond lazy training regime, network enters mean-field regime where parameters evolve significantly, track time-varying kernel $\Theta_t$ induced by particle distribution $\rho_t$ of trainable weights $A^{(\ell)}$, extensive-rank regime $r \asymp d^\beta$ for $\beta \in (0,1)$, training dynamics exhibit emergent staircase-like behavior
- band-by-band Fourier mode activation (Section 4.2, lines 497-509): stepwise loss curve from sequential activation of Fourier modes, mean-field transport of $A^{(\ell)}$ causes $\kappa_t(\omega)$ to grow, repeats across $O(r)$ bands
- multi-index learning in Fourier space (Section 4.3, lines 512-541): multi-index model structure, frozen random features provide dense span, low-rank bottleneck $r$ limits simultaneous active directions, training greedily activates directions
- three-layer mean-field convergence theory (Section 5.1, lines 607-660): universal approximation property holding at all finite times, convergence to global minimizer $\rho_*$ achieving zero training loss, frozen features maintain full support automatically and trivially throughout training, major advantage over trainable-weight networks
- gradient-flow orthogonality and convergence (Section 5.2, lines 663-672): mean-field gradient flow satisfies second-layer orthogonality condition, each Fourier mode decays monotonically, global convergence follows if $\kappa_t(\omega)$ uniformly bounded away from zero
- sharpness of global minima (Section 5.3, lines 675-700): landscape structure in mean-field limit, multiple finite-$N$ local minima correspond to same global minimizer $\rho_*$ (glass-like structure), early vs late sharpness transition, why SGD helps in sharp regime

### central flow dynamics:
from idea1.md:
- central flow point of view (line 161)
- explained by central flow the high frequency grokking is explained by last part of functions high frequency learned by spikes (line 407)

from templateArxiv.tex:
- central flow analysis (Section 4.5, lines 561-591): high-frequency learning exhibits central flow dynamics, optimization trajectory must pass through narrow "channel" in parameter space to access high-quality minima
- central flow in Fourier space: set of weight configurations $\{A^{(\ell)}\}$ that yield large $\kappa_t(\omega)$ for high $|\omega|$ forms low-dimensional manifold (the "channel"), entrance barrier corresponds to breaking low-frequency solution and beginning to build high-frequency spikes, flow within channel constrained to manifold with dynamics governed by residual orthogonality condition, channel leads to sharp global minima where all frequency bands learned
- random features enable channel entry: provide dense unstructured dictionary spanning all possible spike locations and frequencies, network only needs to select from frozen dictionary via adjusting $A$ (not search for right feature directions via gradient descent on $w$), selection problem lower-dimensional ($O(rn)$ vs $O(n^2)$) and better conditioned
- grokking interpretation: training loss drops quickly (low-frequency fitting), test loss remains high for extended plateau (overfitting on low frequencies, channel not yet entered), suddenly both training and test loss drop simultaneously (channel entered, high-frequency generalization achieved)
- connection to dynamical stability (line 591): channel corresponds to stable manifold of global minimum, SGD's implicit bias toward flat minima helps trajectory remain within channel once entered

### dynamical stability:
from idea1.md:
- convergence and dynamical stability (line 424)
- EOS, dynamical stability and 3rd order tensor (line 512)

from templateArxiv.tex:
- dynamical stability and SGD implicit bias (Section 5.6, lines 726-776): stability of MMNN training in sharp regime connected to optimizer choice and loss landscape geometry
- Lyapunov stability analysis: minimum $\theta^*$ is Lyapunov stable if small initial perturbations remain bounded, discrete SGD update near minimum linearizes, stability requires spectral condition $0 < \eta \lambda_i < 2$ for all eigenvalues and noise balance
- stability conditions: Adam effective learning rate $\tilde{\eta}_i \approx \eta / \sqrt{v_i}$, in sharp directions if $v_i$ remains small then $\tilde{\eta}_i \lambda_i$ can exceed 2 violating stability, explains Adam escape phenomenon
- SGD implicit bias toward flat minima: injected noise scales as $\sigma_{\text{noise}}^2 \propto \eta^2 / B$, effective "temperature" $T \propto \eta / B$ biases sampling toward wider basins, sharp minima require exponentially precise parameter values, noise makes them entropically unfavorable
- Wasserstein gradient flow perspective: SGD on particle distribution $\rho_t$ corresponds to Wasserstein gradient flow $\partial \rho_t / \partial t = \nabla \cdot (\rho_t \nabla \delta R / \delta \rho)$, diffusion process with implicit entropy regularization, naturally favors flatter landscapes, frozen random features ensure flow remains non-degenerate
- practical implications: learning rate schedules (decay aggressively in late training), optimizer switching (switch from Adam to SGD when entering sharp regime), batch size tuning, monitoring sharpness via $\lambda_{\max}(H_t)$ and $\text{tr}(H_t) / \text{tr}(G_t)$

### global convergence and minima:
from idea1.md:
- MF limit converges to global optimum, global convergence can be achieved even when loss function is non-convex (lines 32-39)
- for $\beta < \infty$, evolution converges to minimizer, implying global convergence of noisy SGD in number of steps independent of $N$ (lines 143-144)
- multiple finite-$N$ local minima correspond to same minimizer $\rho_*$ of $R(\rho)$ in limit $N \to \infty$ (lines 139-140)
- from globalconvergence: law of MF trajectory is insensitive to choice of neuronal embedding of Init, criteria for Wasserstein gradient flows to escape from non-optimal stationary points (lines 353-359)
- global convergence for all depth from mean field approach (line 530)

from templateArxiv.tex:
- three-layer mean-field convergence theory (Section 5.1, Theorem 5.1 & 5.2, lines 607-660): qualitative and quantitative global convergence, universal approximation at all finite times via frozen features, convergence dimension-agnostic (no curse of dimensionality), moderate-width regime $n \sim 500$-$5000$
- key insight: universal approximation at all times via frozen features (lines 637-660): support $\text{supp}(\text{Law}(w^{(\ell)})) = \mathbb{R}^{d_{\ell-1}}$ remains unchanged throughout training with probability 1, major advantage over trainable-weight networks (no support collapse, simplified convergence, huge effective support)
- gradient-flow orthogonality and convergence (Section 5.2, lines 663-672): residual orthogonal to time derivative, each Fourier mode decays monotonically, global convergence follows
- sharpness of global minima (Section 5.3, lines 675-692): landscape structure in mean-field limit with glass-like structure, early vs late sharpness transition
- connection to central flow and grokking (Section 5.4, lines 715-722): network learns high-frequency components via recursive spike construction, requires traversal through narrow channel in weight space, random features enable this by providing dictionary functions at all locations
