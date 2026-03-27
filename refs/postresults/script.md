Presentation script for `beamer_global_picture.tex`

Use this file as the **full speaker script** (exhaustive). Slide text in `beamer_global_picture.tex` is kept **short** on purpose.

Slide 1. Title

Today I want to present one coherent picture across the papers rather than treating them as separate projects.
The guiding question is simple: why does deep learning optimization become hard as soon as architecture really matters?
My main claim is that low-rank structure gives us a clean way to study that difficulty.
It lets us connect kernel theory, mean-field dynamics, landscape diagnostics, and feature learning inside one framework.

Slide 2. Main thesis

The first thing to emphasize is that low rank is not just a compression trick.
In this work it acts as an organizing principle for optimization, conditioning, spectra, and representation learning.
Across the papers, the same control variables keep reappearing: depth, rank, geometry, and training dynamics.
So the presentation is built to show how those variables line up theoretically and empirically.

Slide 3. Why architecture makes optimization hard

The difficulty is that architecture changes several objects at the same time.
Depth changes correlation propagation and pushes kernels toward collapse or saturation.
Rank changes information flow, parameter sharing, and competition between channels.
Then the optimizer sees a different loss landscape, so the actual training behavior changes as well.
This is why architecture-aware optimization theory is harder than just analyzing one fixed model class.

Slide 4. Architectural core

The main object here is the RF-LR or MMNN viewpoint.
We keep a large random feature bank, but we only train through a small rank-$r$ bottleneck.
That makes the architecture simple enough to analyze and still rich enough to learn nontrivial features.
It also creates visible channels, which is crucial later when we talk about specialization and hierarchy.

Slide 5. Parameter efficiency in practice

Before going into proofs, I want to show that this is not only a theoretical toy.
The empirical table says that a small bottleneck can keep competitive performance while using far fewer trainable parameters.
So rank is not just a mathematical convenience; it is also a practical design variable.
This is one reason the theoretical story is worth developing carefully.

Slide 6. Analytical tools

The project combines several tools because no single viewpoint is enough.
NTK recursion tells us about local optimization, spectra, and how rank enters the kernel.
Mean-field analysis tells us how representations actually move during training.
High-dimensional asymptotics and RKHS arguments help separate expressivity from conditioning and dynamics.
So each theorem should be read as one part of a larger toolkit.

Slide 7. Theorem 1: Infinite-width NTK recursion

This is the first foundational theorem.
In the sequential infinite-width limit, the RF-LR NTK satisfies an explicit recursion, and every bottleneck contributes a visible factor $1/r$.
That matters because the architecture is directly encoded in the kernel formula.
Rank is not hidden in constants or black-box bounds; it appears algebraically.
Once this theorem is proved, the later scaling and conditioning results become structurally natural.

Slide 8. Theorem 2: Depth scaling of the proxy kernel

The second theorem studies what depth does to the proxy kernel.
Correlations align polynomially with depth, but the optimization-relevant gap shrinks like $1/(rk)$.
So depth pushes the model toward a flatter and less informative kernel geometry.
Rank does not remove that effect, but it slows it down in a controllable way.
This is the first precise statement showing why depth and rank must be studied together.

Slide 9. Theorem 3: Conditioning regimes

This is the most optimization-oriented theorem in the first part of the talk.
In generic non-equicorrelated settings, the proxy condition number grows at least like $\Omega(rL)$.
But in equicorrelated and high-dimensional spherical regimes, the effective conditioning can become essentially ideal.
So the same architecture can be easy or hard depending on the geometry of the data.
The message is that architecture and data geometry have to be analyzed jointly.

Slide 10. Kernel-regime summary

This table summarizes what becomes clearer in the low-rank kernel regime.
The question is the same as in fully trained networks, but the relevant scales are exposed much more explicitly.
Depth, rank, kernel size, and centered gaps can all be tracked in closed-form scaling laws.
That is why the RF-LR model is such a useful theoretical lens for the broader optimization problem.

Slide 11. Theorem 4: Low rank is enough for the RKHS

This theorem concerns the three-layer mean RF-LR kernel with isotropic random features.
The key point is that the randomness changes the coefficient in the endpoint expansion, but it does not change the leading exponent.
Because the exponent remains the same as for the shallow ReLU kernel, the induced RKHS is the same.
So the low-rank random bottleneck does not enlarge the function class in the kernel regime.
The advantage must therefore come from conditioning, spectra, and training geometry rather than RKHS size.

Slide 12. Theorem 5: Mean-field well-posedness and global minimizers

Now we move from initialization to training dynamics.
The theorem says that the mean-field flow is well posed, and when the dynamics converges the limit is globally optimal.
The important improvement is that this works with arbitrary depth under standard i.i.d. initialization.
Frozen random features keep enough richness in the first layer to avoid a major obstruction present in deeper full-rank theories.
So this gives us a dynamic backbone for the second half of the presentation.

Slide 13. Theorem 6: Log-ratio growth and channel dominance

This theorem is about feature specialization.
The log-ratio between two channels tells us which one dominates at a given point.
Under explicit sign and stability conditions, that log-ratio grows, meaning dominance persists and amplifies over time.
So feature learning is not only visible in experiments; it can be turned into a rigorous dynamical statement.
This is one of the clearest bridges between theorem and observed training behavior.

Slide 14. Main contributions

At this point I want to highlight what has genuinely improved.
The program moved from heuristic pictures to explicit scaling laws and better-controlled assumptions.
It also became much cleaner conceptually by separating expressivity questions from conditioning and dynamic questions.
Finally, it introduced observables such as log-ratios, partial functions, and symmetry diagnostics that let theory speak to experiments.

Slide 15. Depth and rank in kernel geometry

Now I switch from theorem statements to figures that visualize the same claims.
These panels show that depth does align correlations and that rank controls the remaining useful gap.
So the theory is not merely asymptotic decoration; it is visible numerically in the actual curves.
This slide is the first place where the statements about depth and rank become visually concrete.

Slide 16. Further evidence for depth scaling

I use this slide to show that the same scaling story appears from several angles.
Gap decay, variance decay, and spectral behavior all move in the direction predicted by the theorem.
The point is not one perfect fit, but the consistency of the picture across several observables.
That consistency is what makes the depth-and-rank mechanism believable.

Slide 17. Conditioning is the optimization bottleneck

These figures say that the deterministic proxy is not just formal bookkeeping.
It tracks the empirical condition number well and already distinguishes benign from difficult regimes.
So conditioning really is the right quantity to watch if we want to understand optimization difficulty here.
This is where the theoretical proxy starts to become an actual diagnostic.

Slide 18. Smallest eigenvalue and difficult regimes

This slide focuses on the bottom of the spectrum, which is often where training becomes fragile.
If the smallest eigenvalue gets too small, convergence slows down and instability becomes more likely.
The plots show that this is not a marginal effect; it is one of the main geometric bottlenecks.
So difficult data geometry survives even when we control the architecture through rank.

Slide 19. Rank helps, but not uniformly

This slide is important because it prevents us from overselling the method.
Low rank is not a universal cure and it does not magically fix every dataset.
On genuinely hard non-equicorrelated geometries, the condition number can still be bad.
The right message is more subtle: rank gives a principled and analyzable tradeoff, not a miracle.

Slide 20. Expressivity is not the whole story

This figure supports the RKHS theorem from a more intuitive angle.
Even when the underlying RKHS is the same, the optimization picture can be very different.
So architecture matters because it changes geometry, conditioning, and access to features during training.
That is why pure expressivity comparisons miss most of the real phenomenon.

Slide 21. Early empirical NTK evidence

These were the first experiments that suggested the larger program.
Even before the theory was fully in place, rank was already visibly affecting kernel magnitude and spectral spread.
So these plots were not just exploratory; they were early signs that the optimization problem was being reshaped in a structured way.
The later theory explains why those first observations were pointing in the right direction.

Slide 22. Spectrum comparison

This slide compares the MMNN spectrum with a more classical benchmark.
As rank grows, the MMNN spectrum becomes more concentrated and more comparable to the benchmark spectrum.
That suggests a deeper spectral relation rather than an isolated empirical coincidence.
It also reinforces the idea that rank is controlling concentration in a measurable way.

Slide 23. Kernel statistics beyond the mean

Here I want to insist that the mean kernel is only part of the story.
Across several statistics and several values of $\beta$, we see that rank changes concentration in a robust way.
The distribution becomes more regular as rank increases, and the dependence on parameters remains visible.
So the kernel story is richer than a single average curve.

Slide 24. Distributional view of the kernel

This slide pushes that point even further.
Heavy tails and distributional shape matter because they reveal instability that mean values can hide.
Initialization and data geometry change where the probability mass sits, and that changes practical behavior.
So if we want to understand sensitivity, we need the whole distribution, not only the expectation.

Slide 25. Feature learning and landscape dynamics

This is where the second branch of the project becomes visually obvious.
Training is not only smooth kernel descent around initialization.
We see plateaus, sudden drops, and specialization of channels as learning progresses.
The log-ratio observable gives us a principled way to connect those visual phenomena to theorem-level statements.

Slide 26. Representation learning through training

These panels connect three views of the same process: observables, learned outputs, and mean-field densities.
The key point is that the representation is really moving during training rather than staying frozen.
That motion improves the final prediction and is visible in the density evolution as well.
So this slide is evidence that the model leaves the lazy picture in an organized way.

Slide 27. Internal features are structured, not arbitrary

The internal channels are not learning random shapes.
They become localized, sharper, and more oscillatory as depth increases.
This is the most concrete visual evidence for hierarchical feature learning in the architecture.
It also explains why the low-rank bottleneck is not merely restricting the model; it is organizing it.

Slide 28. Low rank also regularizes geometry

This slide asks what happens if we remove the low-rank structure.
The answer is that the internal geometry becomes much less stable and symmetry breaks more easily.
So rank is not only saving parameters; it is also regularizing the learned representation.
That is one of the reasons the architecture matters beyond simple compression.

Slide 29. Broader perspective

The broader point is that these results open a path toward architecture-aware optimization theory.
Instead of treating architecture as a black box, we identify control parameters and follow how they shape kernels, landscapes, and features.
That creates a framework that can potentially extend beyond this specific model family.
So the long-term value is not only these theorems, but the style of analysis they make possible.

Slide 30. Conclusion

The main conclusion is that deep optimization is hard because architecture acts on several levels at once.
Low rank is the lens that makes those interactions analyzable without making the model trivial.
Several important statements are now proved rather than only observed numerically.
And most importantly, the proofs, figures, and dynamic diagnostics now tell one coherent story.

Appendix slides

If there are questions, I can use the appendix to go into extra scaling plots, extra kernel comparisons, or more feature-learning figures.
The appendix is there as backup material, not as part of the main arc.




we can try our best 