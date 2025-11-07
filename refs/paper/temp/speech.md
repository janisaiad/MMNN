
# in the order
i state



- **core thesis**
  - MMNN training dynamics are best explained via the mean-field perspective  # we state the central lens
  - universal approximation appears at finite training times, enabling global convergence despite non-convexity  # we highlight key result
  - factorizing the middle weight in 3-layer models yields a linear-in-width parameter regime and practical MF behavior around width ≈ 1000  # we note the parameter scaling


- **mean-field theory and convergence**
  - global convergence with MF flows under regularity and convergence assumptions; universal approximation holds at any finite time  # we state the key theorem-level insight
  - for \(\beta < \infty\): noisy SGD converges in steps independent of \(N\); multiple finite-\(N\) minima can correspond to the same limit minimizer \( \rho_* \)  # we list consequences
  - full-support trajectories in MF underpin universal approximation; RF models maintain large support consistently  # we connect to RF
  - Wasserstein gradient flows can escape non-optimal stationary points; MF trajectory law is insensitive to neuronal embedding given the same init family  # we add structural insights


ntk theory will not make me able to explain well the superconvergence, because it's also a matter to show how easy it is for sgd after escaping a minima
there is litterature on that , under the low rank random feature perspective ;

I discovered that I am fundamentally in the mean field regime with 1000 neurons, and it is dimension independent
with 2 hidden layer we have the POC of benefits of low ranks

i can have a ton of different research lines but mostly we observe those kind of step hierarchical training curves
I believe by testing other functions that this comes from a frequency inductive bias
to run very large experiments i'm at the limit of having time to tune hyperparameters and many experimental details, the 



- **architecture and factorization**
  - 3-layer MMNNs benefit from low-rank factorization of the middle matrix; 2-layer models reduce to random features with no factorization advantage  # we delineate regimes
  - train only one factor (e.g., \(U\)) to halve trainable parameters and avoid Riemannian optimization  # we emphasize pragmatic training
  - difference between unfactorized GD and factorized updates: cross-term structure and an extra second-order term \( \alpha^2 \nabla W_t W_t \nabla W_t^\top \); practically, focus on training \(U\)  # we summarize the update contrast

- **learning behavior and inductive bias**
  - MMNNs empirically perform dictionary/wavelet-like learning: base functions are stretched and placed at specific intervals  # we note observed behavior
  - learned partials look like localized high-frequency spikes; parameters \(w\) (frequency) and \(\beta/w\) (location) succinctly describe these features  # we capture parameterization
  - randomization choice (RF vs not) does not break universal approximation due to combinations of ReLUs  # we clarify robustness

- **optimization and training protocol**
  - use Adam as the main optimizer; switch to SGD when Adam destabilizes (e.g., EOS) to recover stability and progress  # we record optimizer switching
  - evidence of three phases where the problem “convexifies”; switching helps quantify the onset  # we describe training phases
  - aim for ablation studies to isolate what drives MMNNs’ strong optimization properties  # we propose evaluation

- **extensive-rank regime and scaling**
  - focus on “extensive-rank” with \( r \asymp d^\beta \), \( \beta \in (0,1) \); \( r_s \asymp d^\gamma \), \( \gamma \in [0,1) \); second-layer coefficients \( \lambda_j \asymp j^{-\alpha} \), \( \alpha \ge 0 \)  # we state the regime
  - Ben Arous multi-index models show emergent staircase dynamics; as \(\alpha\) increases, steps smooth into scaling-law-like curves  # we link to phenomena
  - in light-tailed regime \(\alpha > 1/2\): risk \( \mathcal{R} \sim 1/(\text{Data})^a + 1/(\text{Model})^b \) resembling neural scaling laws  # we relate to scaling
  - Stiefel manifold assumption (orthogonality) is important; MMNNs can realize similar scaling with small tunable \(\beta\)  # we note geometry

- **random features viewpoint**
  - MMNNs ≈ low-rank random features networks; training occurs in a high-dimensional RF space  # we frame the model
  - dimension-agnostic MF lens: avoids reliance on high-dim NTK artifacts that don’t match practice  # we justify MF choice
  - data embedded into larger feature spaces; the network composes dictionary functions from this embedding  # we capture mechanism
  - open: impact of training only the first layer versus both on expressivity and optimization  # we flag question

- **NTK analysis (lazy to end-of-lazy)**
  - recursive NTK formulation with probabilistic view supports concentration in \(r\) and dimension  # we cite structure
  - early training is poorly conditioned for SGD; NTK partially explains this, with RKHS equivalence mitigating disadvantages  # we provide interpretation
  - finite-width corrections (EOC-driven) enter at order \(N r\); NTK admits expansion \( \sum_{i=1}^L C_i / r^i \) with \(C_i\) depending on layerwise correlations \( \rho_i \)  # we specify expansions
  - kernel random matrix perspective: bulk/outlier structure differs from MLP; “Terjek-like” depth scaling of the outlier  # we state spectral story
  - practical observations: variance behaves like \(1/r^2\) vs expected \(1/r\); explain via doubly random Gram (features × data) and normalization by \(r^2\)  # we reconcile experiments
  - consider spherical 2D/3D datasets for clean cosine-geometry and spectral analysis  # we suggest datasets
  - in large low rank, NTK tends to 1; analyze finite-width corrections and depth scaling (towards infinite depth via \(1/r\) expansion)  # we chart the limit
  - relation to tensor programs for depth and hyperparameter transfer (via \(\mu\)P); plan NTK for DSRN as well  # we connect frameworks


- **normalization and data geometry**
  - hypercube data: use \(\mathrm{Tr}(\Sigma_w) = d\) and unit bias for approximation theorems  # we define scaling
  - spherical data: unit-norm weights and unit bias; \(\mathrm{Tr} = 1\) removes the \(1/r\) from normalization  # we contrast cases
  - these choices materially affect NTK standard deviation and which NTK model is appropriate  # we underscore impact
  - bias can be removed by augmenting inputs with 1  # we mention a trick

- **mean-field experiments and ODEs**
  - track particle distributions to identify governing ODEs; test bounded activations; compare MF minima with empirical training minima  # we outline experiments
  - local mass conservation across disconnected supports constrains feasible density evolution  # we include physics analogy
  - consider Gronwall-based arguments; check alternatives for sharper bounds  # we guide analysis
  - propagation of chaos in teacher–student with uniform init: neurons disperse before concentrating to teacher  # we add dynamics
  - investigate whether targets lie in Barron space  # we add a check

- **empirical program**
  - show super-convergence and the three convexifying phases; analyze symmetry breaking and recovery  # we define phenomena
  - frequency timescales appear ordered; landscape may be autosimilar across training stages  # we hypothesize structure
  - use \( r \in [5,50] \) for good concentration and expressivity with small parameter budgets; include TFLOPs accounting  # we give practical ranges
  - compare MMNN vs MLP across dimensions; include cosine regression probe; locate where lazy training ceases  # we specify benchmarks
  - ablations: factorization, partial training, optimizer switching, normalization, width/depth, initialization  # we list factors

- **theory: kernel random matrices and MP regime**
  - apply spectrum-of-kernel-random-matrices results where \(N, d, r\) grow linearly; use Marchenko–Pastur tools  # we anchor the regime
  - linear coefficient between \(d\) and \(N\) comes from discretization; Gram entries driven by \(\rho_1\)  # we provide rationale
  - expect a single outlier capturing early training; tie to “Terjek” analyses  # we predict spectrum
  - caution: the kernel spectrum results need linear samples wrt dimension  # we note assumptions

- **paper plan and deliverables**
  - incremental paper: disentangle contributions (RF, low rank, optimizer switching, MF/NTK, generalization/approximation, tensor programs, PINN/Sobolev)  # we map scope
  - for arXiv: full math (MF convergence, NTK finite-width corrections, spectral analysis); for conference: benchmarks showing MMNN > MLP with 3 layers  # we split goals
  - structure: lit review (incl. Montúfar), theory (MF/NTK/spectral), experiments (dictionary learning, scaling, spectra), ablations, discussion  # we sketch sections
  - code release: JAX first, then PyTorch  # we state tooling
  - supporting notes: GoodNotes `https://web.goodnotes.com/s/F0IFxLELb1470d6AGRdbxH#page-2`  # we cite notes

- **open questions / to-dos**
  - clarify benefits/drawbacks of training only the first layer vs both in MMNN  # we pose a design question
  - finalize MF ODE for 3 layers and prove non-quantitative convergence  # we set a theory task
  - write proofs related to Kibble and Fisher distributions; derive Fisher–propagation of \(\rho_i\)  # we list proof tasks
  - extend finite-width corrections: scaling with depth; compare with PMISOF; explore Feynman-style expansions for low-rank RF  # we propose extensions
  - quantify RKHS of MMNN-induced RFs (Bach, Misiakiewicz) for generalization guarantees  # we call for bounds
  - reconcile empirical \(1/r^2\) vs \(1/r\) NTK variance via doubly random Gram and normalization  # we mark a variance puzzle
  - determine normalization for hypercube vs sphere in each experiment; document its effect on NTK and training  # we enforce protocol