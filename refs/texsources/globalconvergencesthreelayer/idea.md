### Goal

- Prove two extensions using the mean-field framework already developed in `iclr2021_v4.tex`.
- Settings:
  - Random features: freeze first layer, i.e., set $\xi_{1}(\cdot)\equiv 0$.
  - Low-rank second layer: $w_{2}(t,c_{1},c_{2})=\sum_{r=1}^{R} a_{r}(t,c_{1})\, b_{r}(c_{2})$, train $\{a_{r}\}$, freeze $\{b_{r}\}$.

### Theorem (random features: frozen first layer)

- Assumptions: Keep Assumptions Regularity, Regularity-init, and Assumption 4.1 (support and universal approximation) from `iclr2021_v4.tex`. Set $\xi_{1}(\cdot)\equiv 0$. For the third layer, use either clause in Theorem 4.2 (untrained or trained).
- Claim: The conclusions of Theorem 4.2 (global convergence for convex losses; zero risk in the realizable nonnegative-loss case) hold for the mean-field dynamics $W(t)$ with frozen $w_{1}$.

Sketch.
- With $\xi_{1}\equiv 0$, $w_{1}(t,\cdot)\equiv w_{1}(0,\cdot)$. Since ${\rm supp}(\rho^{1})=\mathbb{R}^{d}$ and $\{\varphi_{1}(\langle u,\cdot\rangle):u\in\mathbb{R}^{d}\}$ is dense in $L^{2}(\mathcal{P}_{X})$, density holds at all finite times without the topology argument used in Lemma 4.3.  # we avoid the support-evolution step
- At any limit point $W(\infty)$, stationarity of $w_{2}$ gives $\Delta_{2}(c_{1},c_{2};W(\infty))=\mathbb{E}_{Z}[\Delta_{2}^{H}(Z,c_{2};W(\infty))\varphi_{1}(\langle w_{1}(0,c_{1}),X\rangle)]=0$ for $P$-a.e. $(c_{1},c_{2})$.  # we use the same second-layer argument
- Density of the ridge family implies $\mathbb{E}[\Delta_{2}^{H}(Z,c_{2};W(\infty))\mid X=x]=0$, hence $\mathbb{E}[\partial_{2}\mathcal{L}(Y,\hat y(X;W(\infty)))\mid X=x]=0$.  # we get conditional optimality
- The last step matches Theorem 4.2: convexity yields global optimality; the realizable nonnegative-loss case yields zero risk.  # we conclude as in the original theorem

Remarks (finite width).
- The quantitative coupling bound (Theorem 3.1) and its corollaries remain valid; if anything, freezing $w_{1}$ simplifies Lipschitz interdependencies.  # we inherit approximation bounds
- Practically, approximation depends on the random feature width $n_{1}$; larger $n_{1}$ may be needed to match learned $w_{1}$ at finite width.  # we note the bottleneck

### Theorem (low-rank $w_{2}$, train left factor only)

- Parameterization: $w_{2}(t,c_{1},c_{2})=\sum_{r=1}^{R} a_{r}(t,c_{1})\, b_{r}(c_{2})$ with fixed measurable $\{b_{r}\}_{r=1}^{R}$ and trainable $\{a_{r}\}_{r=1}^{R}$. Keep Assumptions Regularity, Regularity-init, and Assumption 4.1; maintain the same clause on $\xi_{3}$ as in Theorem 4.2.
- Claim (i): If $\overline{\mathrm{span}}\{b_{r}\}_{r=1}^{R}$ is dense in $L^{2}(P_{C_{2}})$ (e.g., $R\to\infty$ with a universal dictionary), then the conclusions of Theorem 4.2 hold unchanged.  # universality restores full expressivity along $c_2$
- Claim (ii): For finite $R$ without density, the MF dynamics converges to a global minimizer of $\mathscr{L}$ restricted to $\{w_{2}:\; w_{2}=\sum_{r=1}^{R} a_{r}(\cdot)\, b_{r}(\cdot)\}$; the asymptotic excess risk equals the $L^{2}(P_{C_{2}})$-projection error of the Bayes residual onto $\overline{\mathrm{span}}\{b_{r}\}$.  # we quantify the restricted optimum

Sketch.
- Stationarity with the low-rank structure yields, for each $r$, $\mathbb{E}_{Z}\big[\Delta_{2}^{H}(Z,\cdot;W(\infty))\, b_{r}(\cdot)\, \varphi_{1}(\langle w_{1}(\infty,c_{1}),X\rangle)\big]=0$ for $P$-a.e. $c_{1}$.  # orthogonality conditions per basis function
- If $\{b_{r}\}$ is dense in $L^{2}(P_{C_{2}})$, this forces $\mathbb{E}[\Delta_{2}^{H}(Z,c_{2};W(\infty))\mid X=x]=0$, and the proof proceeds identically to Theorem 4.2.  # recover the same conclusion
- Otherwise, only the projection of $\Delta_{2}^{H}$ onto $\mathrm{span}\{b_{r}\}$ vanishes, yielding optimality within the restricted class and an irreducible approximation gap given by the orthogonal complement.  # restricted convergence

Remarks (conditioning and practice).
- Choose $\{b_{r}\}$ from a bounded universal dictionary over $\Omega_{2}$ (e.g., orthogonal polynomials, Fourier, wavelets); increase $R$ until the approximation gap stabilizes.  # we suggest a practical basis
- Monitor the Gram $G_{rs}(t,x)=\mathbb{E}_{C_{2}}[w_{3}(t,C_{2})\varphi_{2}'(H_{2})\, b_{r}(C_{2})b_{s}(C_{2})]$; add ridge regularization on $a_{r}$ if ill-conditioned.  # we ensure numerical stability

### Interaction with random features

- The two modifications compose: freeze $w_{1}$ and train only the left factor of low-rank $w_{2}$.  # we can combine them
- Theory: Claim (i) for $\{b_{r}\}$ density plus the random-features theorem above still delivers the original global convergence conclusions at MF.  # the MF result persists
- Practice: finite-width errors now depend on both the random feature width $n_{1}$ and the rank $R$; both may need to scale to match fully trainable baselines.  # we highlight the finite-width tradeoff




### frequency building

#### Setup for stacked low-rank modules (frozen right dictionaries)

- For layer $\ell=1,\dots,L$, parameterize $w_{2}^{(\ell)}(t,c_{1},c_{2})=\sum_{r=1}^{R_{\ell}} a_{r}^{(\ell)}(t,c_{1})\, b_{r}^{(\ell)}(c_{2})$ with fixed measurable $\{b_{r}^{(\ell)}\}_{r=1}^{R_{\ell}}$.  # we define the architecture per layer
- Keep random features in the first layer ($\xi_{1}\equiv 0$) or trainable $w_{1}$; both cases below use only boundedness and universal approximation of the $\varphi_{1}$-ridge family.  # we keep the same input mechanism
- Let $\mathsf{B}^{(\ell)}=\mathrm{span}\{b_{r}^{(\ell)}\}_{r=1}^{R_{\ell}}\subset L^{2}(P_{C_{2}})$.  # we define per-layer right-span

#### Concentration at initialization (orthogonality of the right bases)

Claim (Gram near identity).
- Suppose for each $\ell$, the dictionary $\{b_{r}^{(\ell)}\}_{r=1}^{R_{\ell}}$ is orthonormal in $L^{2}(P_{C_{2}})$ and bounded by $|b_{r}^{(\ell)}(c_{2})|\leq B$. Let $\widehat{G}^{(\ell)}\in\mathbb{R}^{R_{\ell}\times R_{\ell}}$ be the empirical Gram $\widehat{G}^{(\ell)}_{rs}=\frac{1}{n_{2}}\sum_{j_{2}=1}^{n_{2}}b_{r}^{(\ell)}(C_{2}(j_{2}))\,b_{s}^{(\ell)}(C_{2}(j_{2}))$. Then, for any $\varepsilon\in(0,1)$, with probability at least $1-2\exp\{-c\,n_{2}\varepsilon^{2}+c'R_{\ell}\log R_{\ell}\}$,
$$\big\|\widehat{G}^{(\ell)}-I\big\|_{\mathrm{op}}\le\varepsilon,$$
for absolute constants $c,c'$ depending only on $B$.  # we apply matrix Bernstein via bounded entries

Sketch.
- Each rank-one matrix $X_{j_{2}}=b(C_{2}(j_{2}))\,b(C_{2}(j_{2}))^{\top}-I$ is zero-mean, $\|X_{j_{2}}\|_{\mathrm{op}}\le B^{2}R_{\ell}$ and has variance proxy $\sigma^{2}\lesssim R_{\ell}B^{4}$. Matrix Bernstein yields the bound after a union/epsilon-net refinement.  # we state the standard route

Weighted Gram at small time.
- Define the effective inner product $\langle f,g\rangle_{t}=\mathbb{E}_{C_{2}}\!\left[w_{3}(t,C_{2})^{2}\,\varphi_{2}'(H_{2}(t,C_{2}))^{2}\,f(C_{2})g(C_{2})\right]$. If $w_{3}$ and $H_{2}$ are $K$-bounded (Assumption Regularity), then $\kappa^{-1}\|f\|_{2}^{2}\le \|f\|_{t}^{2}\le \kappa\|f\|_{2}^{2}$ for some $\kappa=\kappa(K)$. Hence the weighted Gram remains well-conditioned uniformly over time, and at initialization it remains $\varepsilon$-close to diagonal in operator norm with high probability as above (up to $\kappa$).  # we pass to weighted geometry

#### Frequency multiplication by nonlinearity

Notation.
- For a bounded nonlinearity $\varphi_{2}$ with nontrivial polynomial/Hermite expansion $\varphi_{2}(u)=\sum_{k\ge0}\alpha_{k}\,H_{k}(u)$, consider
$$S^{(\ell)}(t,c_{1},c_{2})=\sum_{r=1}^{R_{\ell}} a_{r}^{(\ell)}(t,c_{1})\, b_{r}^{(\ell)}(c_{2}),\qquad U^{(\ell)}=\varphi_{2}\!\left(S^{(\ell)}\right).$$

Key lemma (spectral expansion over $C_{2}$).
- For each fixed $(t,c_{1})$,
$$U^{(\ell)}(t,c_{1},\cdot)=\sum_{k\ge0}\alpha_{k}\!\!\sum_{r_{1},\dots,r_{k}\in[R_{\ell}]}\!\!\Big(\prod_{i=1}^{k}a_{r_{i}}^{(\ell)}(t,c_{1})\Big)\,\big(b_{r_{1}}^{(\ell)}\cdots b_{r_{k}}^{(\ell)}\big)(\cdot),$$
with convergence in $L^{2}(P_{C_{2}})$ under boundedness of $a_{r}^{(\ell)}$ and $\sum_{k}\alpha_{k}^{2}\,\mathbb{E}[S^{(\ell)}]^{2k}<\infty$. Thus $U^{(\ell)}$ lives in the degree-$\ge1$ polynomial closure of $\mathsf{B}^{(\ell)}$.  # nonlinearity generates higher-degree products of basis functions

Consequence (dictionary growth across layers).
- If the next layer’s right dictionary contains or is well-approximated by products of previous layer’s right dictionary (e.g., take $b_{r}^{(\ell+1)}$ to be an orthonormalization of $\{b_{r_{1}}^{(\ell)}\cdots b_{r_{k}}^{(\ell)}:1\le k\le K_{\ell}\}$), then depth-$L$ compositions expand the accessible right-span from $\mathsf{B}^{(1)}$ to the polynomial hull of degree up to $\sum_{\ell=1}^{L}K_{\ell}$.  # we formalize the frequency-building mechanism

Depth–universality corollary.
- Suppose: (i) for each $\ell$, $\{b_{r}^{(\ell)}\}$ is bounded and orthonormal in $L^{2}(P_{C_{2}})$; (ii) the layerwise right dictionaries are chosen so that the union of polynomial products $\bigcup_{\ell\le L}\mathrm{poly}_{\le K_{\ell}}(\mathsf{B}^{(\ell)})$ is dense in $L^{2}(P_{C_{2}})$ as $L\to\infty$; (iii) $\varphi_{2}$ has nonzero coefficients $\alpha_{k}$ for infinitely many $k$. Then, for any $\varepsilon>0$ and any $g\in L^{2}(P_{C_{2}})$, there exist $L$ and ranks $\{R_{\ell}\}$ such that the depth-$L$ stacked low-rank architecture (training only left factors) expresses a function $\tilde g$ with $\|g-\tilde g\|_{2}\le\varepsilon$.  # depth grows the right-span to universality

Remarks.
- Fourier dictionaries on $C_{2}$: products of sines/cosines generate sum/difference frequencies, so $K_{\ell}$ controls the maximum frequency reachable by layer $\ell$.  # classical harmonic growth
- Orthogonal polynomials on $C_{2}$: products expand polynomial degree; nonlinearities with rich Hermite spectra accelerate degree growth.  # polynomial-frequency growth

#### Training dynamics with stacked low-rank and frozen right factors

Proposition (projected stationarity per layer).
- In the MF limit with the above parameterization and Regularity assumptions, at any limit point $W(\infty)$ the projected stationarity holds layerwise:
$$\mathbb{E}_{Z}\Big[\Delta_{2}^{H,(\ell)}(Z,\cdot;W(\infty))\, b_{r}^{(\ell)}(\cdot)\,\varphi_{1}(\langle w_{1}(\infty,c_{1}),X\rangle)\Big]=0\quad\text{for all }r,\ \text{a.e. }c_{1}.$$
If the right-span at depth $L$ is dense in $L^{2}(P_{C_{2}})$ (by the frequency-building mechanism), then $\mathbb{E}[\partial_{2}\mathcal{L}(Y,\hat y(X;W(\infty)))\mid X]=0$ and the same global convergence conclusions as Theorem 4.2 follow.  # we lift the single-layer proof to depth via density

Quantitative note (finite width).
- Concentration of the (weighted) Gram and standard coupling (Theorem 3.1) give finite-width error scales like $e^{K_{T}}\big(\sqrt{\epsilon}+\min_{\ell}n_{2}^{(\ell)-1/2}\big)$ up to logs. The rank–depth tradeoff appears in approximation error via the polynomial truncation degree.  # we connect to rates and approximations

Overall message.
- Even if right factors are frozen and orthogonality is only guaranteed at initialization by concentration, stacking low-rank modules with a nonlinearity that mixes coefficients multiplies and redistributes frequency content across $C_{2}$. As depth increases (and with a suitable choice of right dictionaries), the accessible right-span grows to universality, while the MF stationarity argument delivers optimization to the best function in that span.  # we summarize the mechanism