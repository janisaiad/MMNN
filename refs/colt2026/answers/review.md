Skip to main content
Submissions
Contact Chairs
Help Center
Select Your Role : 
Author 
COLT2026 
JANIS AIAD 
View Reviews
Paper ID
527
Paper Title
Low rank is enough for the neural tangent kernel: towards stable training of deep networks
Reviewer #1
Questions
1. Rating (scale of 1-7)
3: Marginally below acceptance threshold
3. Review
The paper considers deep networks with low rank bottlenecks and consider the neural tangent kernel for such networks. The paper gives an explicit recursive formula for the NTK, and proves condition number bounds. The paper also shows that for shallow networks the kernel for RF-LR model is equivalent to a shallow ReLU kernel.

Strength:
- clear analysis for kernels of RF0LR model, reasonable bounds on the condition numbers
- the kernel equivalence result seems interesting.

Weakness/comments
- One interesting result of the paper is that one can increase the depth in RF-LR setting without worrying too much about the conditioning or largest eigenvalue of the kernel. However, the paper doesn't highlight what are benefits of deep kernels. A deep neural network that is not in the kernel regime is likely to have stronger representation power, but one needs to ask the same question for the kernel setup and it's not completely clear to me what is the benefit of having a deep kernel (either some prior work or some empirical results could solve this issue).

- The A^{(l)} and c^{(l)}'s are trained while feature directions are fixed to be random, why is this choice made? What would happen if w^{(l)} is also trained?

- the writing of the paper can be improved. Many lemmas/theorems just have proofs pointing directly to appendix without giving out much intuition. Some important concepts are defined only in appendix (such as K_{proxy}).
Reviewer #2
Questions
1. Rating (scale of 1-7)
4: Marginally above acceptance threshold
3. Review
The submission analyzes low-rank networks in an NTK regime, deriving an explicit form for their NTK as well as a variety of results related to their rank, condition number, and expressivity of the RKHS. While some of the results require strong assumptions, the limitations are clearly stated even in the abstract and there is clear explanation of why they are difficult to extend in the present approach. The results are interesting, and while I unfortunately did not have time to thoroughly verify them, the sketches and a skim are sufficient for me to believe that they are probably correct. The numerical evaluations are also solid and convincing. The only thing preventing me from a higher score is that the results are themselves somewhat limited, but in numerical terms my opinion of the paper is between a 4 and a 5. Sorry that this is not a more interesting review!

A few minor points of presentation: while 1^\perp and k_\perp are fairly intuitive, they don't seem to be defined until Appendix B and are used regularly enough in the main paper that they merit a quick definition at first use. Also, in author-year styles like this one, you should use \citet{} (or perhaps \textcite{} depending on your setup) for citations that play a grammatical role in the sentence and \citep{} (or \parencite{}) for ones that do not, rather than \cite{} in all instances as you seem to here. The main body of the paper would also benefit from at least a reference to Appendix E, if not finding space for one of the main verification results in the main body.
Reviewer #3
Questions
1. Rating (scale of 1-7)
3: Marginally below acceptance threshold
3. Review
Summary: This paper characterizes the Neural Tangent Kernel of the "RF-LR" architecture, a fully-connected neural network with bottleneck layers of size r. The main contributions include a recursion for computing the infinite-width NTK (Theorem 3), a characterization of the entries and spectrum for a limiting "deterministic proxy kernel" (Theorem 5, Proposition 6), and a result showing that the 3 layer RF-LR RKHS is equivalent to the shallow ReLU RKHS (Corollary 13)

Review:
1. My primary concern with the paper is that its clarity is lacking. Many terms are referred to in the main text, without ever being properly defined. It is thus difficult to understand the main results of the paper without having to refer heavily to the appendices. Some examples are:
- On page 2 of the introduction, the terms "edge of chaos," "Fisher-Kibble" decoupling, "Puiseux analysis," and the quantity $\rho$ are referred to without definition.
- Theorem 3 refers to $\kappa_\perp$, but this is not defined until much later in Corollary 8.
- Section 3.1 claims to provide a summary of the "probabilistic recursion and exponential depth suppression." No such summary is provided; it is not even clear to the reader what these terms mean.
- Section 4 refers to the quantity $\varrho$, but this is only defined in Appendix D.1 (finding this definition was difficult for the reader as well). The "deterministic proxy kernel" is also not defined in the main text and instead is deferred to (84) in the appendix.
- In Proposition 10, what is Fisher(\rho, r)? I assume that this is the distribution defined in Lemma 9, but this needs to be stated explicitly.

2. My next concern is that it is not explained at all why the NTK of the RF-LR architecture is an interesting object to study.
- One motivation presented in this paper is that the RF-LR maintains the expressivity of MLPs with fewer parameters. However in practice, neural networks trained via SGD are in a "feature learning" regime where they are not approximated by their NTK. The results presented in this paper thus do not provide any explanation for when one should even consider the RF-LR architecture beyond the NTK regime.
- The next justification is that for the RF-LR NTK, the maximum stable learning rate stays constant as depth increases. I don't see why it's necessarily an issue to scale the learning rate inversely with depth.
I thus find the significance and relevance of the presented results to be low.

3. I am confused by the claim that "depth and rank commute". Is it preferable to increase depth $L$ or rank $r$? Corollary 13 seems to say that the RKHS's are equivalent, so it doesn't seem like the NTK perspective can distinguish between trading off one versus the other.

4. (minor) A more accurate statement of Theorem 2 should be that the ith entries $(h^{(\ell)}_i(x_1), \dots, h^{(\ell)}_i(x_m))$ converge in distribution to a multivariate gaussian.
© 2026 Microsoft Corporation
About CMT | Docs | Terms of Use | Privacy & Cookies | Consumer Health Privacy | Request Free Site
