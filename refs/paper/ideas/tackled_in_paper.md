# Ideas tackled in the paper

- Low‑rank random feature (ex‑MMNN) model and training setup  
  – `NTK and lazy training analysis / Model`: train A,c; freeze w,b; NTK scaling (σ_A/√n), bias β; low‑rank bottleneck (r ≪ n).  
  – Architecture figure included.

- Base kernels and NTK recursion (lazy regime)  
  – `Base NNGP and Derivative Kernels` (Σ^(ℓ), ẊΣ^(ℓ));  
  – `NTK Recursion` theorem (sequential infinite‑width) + `Two‑layer NTK (β=0)` corollary.

- Low‑rank fluctuation decay and extensive‑rank viewpoint  
  – `Low‑Rank Bottlenecks and O(1/r) Fluctuations` (variance ↓ ∼ O(1/r));  
  – `Scaling Laws and Extensive‑Rank` (skeleton and discussion).

- Distributional tools in appendix  
  – `Fisher` law for sample correlation; `Kibble` bivariate χ² (U=‖x₁‖², V=‖y₁‖²) with key moments.

- Notation + plan d’expériences (placeholder) + Discussion/Outlook  
  – `Numerical experiments` (emplacement prêt) ;  
  – `Notation`, `Acknowledgments`, `Proofs` (appendix).


# Ideas not yet tackled (planned / to add)

- Mean‑field (MF) analysis & ODE / particle (density ρ_t) viewpoint; MF–NTK bridge; three‑layer MF global convergence links.

- Edge‑of‑Chaos (EOC) conditions in low‑rank two‑layer setting (σ_A²·Tr(Σ_w)=2/r), scaling of Θ̇^(1), normalization by √r, explicit role of β.

- Frequency vs localization disentanglement (w magnitude/orientation vs β/offset → spike placement); Hermite / Gegenbauer spectral decomposition; RKHS characterization.

- Dynamical stability near global minima: NTK/Hessian smallest eigenvalues, Lyapunov rates, NTK drift; symmetry breaking and recovery during training (anisotropy, directional selection).

- Empirical scaling laws (extensive‑rank r≈d^β): stepwise (“multi‑index”) learning curves; early‑training lazy regime vs transition out of laziness.

- Finite‑width corrections: Nr and r/d scaling; Lyapunov product analysis across depth; 1/r vs 1/r² variance debate.

- Non‑Gaussian deep GP propagation (ℓ>1): explicit analysis of Σ^(ℓ), ẊΣ^(ℓ) randomness; ρ→ρ₁ propagation via Fisher/Kibble in the main text.

- Outlier/bulk spectrum via random‑matrix theory (Marchenko–Pastur, double randomness of Gram Θ); Terjék‑style outlier scaling; comparison vs MLP baselines.

- Optimizer‑switching (Adam→SGD) and “convexification” checkpoints; ablations; benchmarks (cosine regression, spherical 2D/3D, high‑d).

- Code release plan (JAX then PyTorch); experimental protocols for reproducibility.

- Function class assumptions: investigate whether targets live in Barron space (and implications for approximation/generalization).