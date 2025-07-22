
\begin{itemize}
    \item Soft Structure, TP like, Sparsity $\beta$ (probabilistic def).
    \item $Y_{bridge}$
    \item \textit{ILLEGIBLE} (same as Kernel?)
    \item If there is no norm or if this norm is identical to \textit{ILLEGIBLE}?
    \item Narrow residual? Usual Structure $\rightarrow$ NTK. Can we find the Tangent Angle \textit{ILLEGIBLE}?
    \item ``something should come out here, I think still with $1/p_n$''
    \item ``to verify with $1/p_n$ or 1''
    \item ``not to be calculated with $\nabla\theta$ !! apparently !!''
    \item ``put recursion! No need \textit{ILLEGIBLE}, always \textbf{Always}''
    \item ``Watch out for Q, I''
    \item ``If we let n tend to infinity, we hope that... so we have a recursive relationship''
    \item Possible literary references: ``Jac Silverstein and others, and \textit{ILLEGIBLE} to calculate the exact limit (\textit{TejoK}?) -- random scaling laws''
\end{itemize}


make a small explanation in the introduction to elaborate on why it's so useful (with MMNNS theorems) 
but also that training on low rank manifolds is hard, and not interpretable that much, without a lot of non linearities involved
for same number of parameters, but useful for compression and distillation (without SETOL setup)

1 thing to remark is that low rank makes ntk to explode and grow more, when ranks grow, ntk is diminished, which 
can be counter intuitive since we don't normalize w in the relu (we should because of curse of dim, or to approximate along
a direction and separate well at many scale)

for uniform weights and b to see, because of no gaussianity but we can compute many things with integrals (or symbolic stuffs)
also weights can be sampled non isotropically for Q, b,, or by changing data by Q only, with spectral radius 1 (or other with
sigma_A to respect the EOC)

We can try to compute the scaling wrt L to see how the optimization process works
and also compare with the hessian (for the surmrise)

also the jax code for computations can be improved a lot

surmrise to see on MMNNs

so the idea is to use a block structure (like on the anatomy of attention) with internal dim going to infty
and that can be useful without a lot of params, for instance nlog(n) attentions , or transformers (but very costly) to see under the
ntk's pespective that the optmization goes well, or has gaussian local structure (big conjecture for any TP)

in fact the formalism is to use TP and to choose what u train