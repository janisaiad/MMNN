OpenReview.net
Search OpenReview...
Notifications15
Activity
Tasks
Janis Aiad 
back arrowGo to ICML 2026 Conference homepage
Low-Rank Structure Suffices for Global Convergence of Neural Networks in the Mean-Field Limit
Download PDF
Janis Aiad, Haizhao Yang, Shijun Zhang 
23 Jan 2026 (modified: 09 Feb 2026)
ICML 2026 Conference Submission
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
CC BY 4.0
Verify Author List: I have double-checked the author list and understand that additions and removals will not be allowed after the abstract submission deadline.
TL;DR: We prove mean-field convergence to global minimizers for low-rank neural networks with frozen random features at any depth under standard initialization, avoiding neural collapse while using 99% fewer parameters in practice.
Abstract:
This work studies the training dynamics of low-rank neural networks with frozen random features in the mean-field regime. When the mean-field dynamics converges, the limit is shown to be a global minimizer of the population loss; this holds for gradient-based training under standard independent and identically distributed initialization, despite low-rank constraints and nonconvex loss functions. By constraining low-rank structure into the network architecture, a tractable mean-field evolution system is derived without relying on Riemannian gradient descent. Its well-posedness is established, and it is shown that with frozen random features, neural collapse (the main bottleneck in prior work) is avoided while the learning dynamics are simplified. The analysis identifies a low-rank feature learning mechanism, in which different low-rank channels specialize to distinct spatial locations and progressively capture higher-frequency components. This mechanism explains both the persistence of global convergence and the emergence of interpretable features. Numerical experiments demonstrate that low-rank networks achieve faster convergence and higher accuracy on highly oscillatory targets and match baselines on MNIST, while using 95--99% fewer parameters than full-rank networks.

Primary Area: Theory->Deep Learning
Keywords: mean-field theory, low-rank neural networks, random features, global convergence, non-convex optimization
Ethics Agreement: I certify that all co-authors of this work have read and are committed to adhering to the Call for Papers, Author Instructions, Research Ethics, and Peer-review Ethics.
LLM Policy: This submission requires Policy A.
Proceedings-only Option: If this paper is accepted, the authors tentatively plan to present it in person at the conference (as a poster and, if selected, as an oral).
Reciprocal Reviewing Status: This submission is NOT exempt from the Reciprocal Reviewing requirement. (We expect most submissions to fall in this category.)
Reciprocal Reviewing Author:  Janis Aiad
Submission Number: 25189
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
15 / 15 replies shown
Add:
Official Review of Submission25189 by Reviewer VCqo
Official Reviewby Reviewer VCqo16 Mar 2026, 18:27 (modified: 01 Apr 2026, 18:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer VCqoRevisions
Summary:
The authors study the mean field dynamics of low rank multi-layer neural networks. The authors show that rewriting the weight matrices as low rank factorizations M = WA^T and then training only the channel part, A, while defining the mixing part as frozen random features makes it possible to avoid neural collapse. The authors also show that whenever the dynamics converges, it always converges to global minimizers and that each low rank factorization learns a distinct feature.

Strengths And Weaknesses:
Some of the figures could be cleaner and clearer. As an example of this, in Figure 2, it is not clear. Why not keep a two color code. Show the high rank curve in one color (e.g. blue) and the low rank curves in another (or shades of another) color. Also if you have a low rank and a high rank curve at the same momentum, wouldn't it make sense to only keep those two curves? Does it make sense to compare curves with different momentums ?

Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
Section 2,

you should keep the same citation style throughout the paper. Before line 126 you cite works by mentioning the authors and then providing the reference in the bibliography but then on line 130 (left column), you only provide the citation in the bibliography
line 139, frozen random features ensure supp(W^0(C_1)) —> this is not clear
line 110, second column, what do you mean by the mean-field width limit ? Do you mean the limit of infinite width networks?
Section 3.1.

I understand that the low rank factorization appears from the stacking of W_j^{(\ell)} and h^{(\ell-1)} but wouldn’t it be clearer to write x’ = \varphi(W^T A x + b) ?
Section 3.2.

Line 166, your notations with the lower and uppercase case c_1 in W^0(c_1) is not really clear. Why put that C_i ? Why not just keep the notation of Eq. 1, W_0^{(i)} ?
Your expectation is taken with respect to the neurons indices ? which is strange to me ? You also say that in the infinite width limit, the neurons indices are continuous. Isn’t the norm to work with a measure on the weight vectors associated to each neuron ? I.e whose support would be non zero at the points W(C_i) ?
Section 3.3.

lines 187-188, I strongly recommend writing down the expression of the loss at least and recall that the system corresponds to a simple gradient descent on this loss? Because the way you introduce (2) is a little rough
lines 187-188: What are the learning rate schedules?
Section 3.4.

On lines 166 - 168, you say that the “Better than null” assumption requires the initial loss to be better than the null function.. What does that mean ?
Line 178, second column, you say that for Relu, the same holds with high probability in r. I would recall the meaning of r here
I don’t know how easily this can be done but I would recommend laying out the assumptions cleary on lines 215 - 216. Some of them are not very clear. E.g. what do you mean by bounded activation and MIXING (i.e the mixing part)?
Section 3.5 - 3.6

lines 207 - 210, second column, you have |H_\ell(W’) - H_\ell(W’’)|< K||W^{\ell}||_{\infty, 1}. What is W^{\ell} in this setting? Does that mean all the weights from layer ell ? but then how does this relate to W’ and W’’? That notation does not make sense to me.
lines 218 - 219, what is the solution operator F?
lines 220 - 223: “After defining norms, the solution operator F,… and the contraction operator …” —> “After defining norms, F and bounds, and the contraction, the argument proceeds… ” (I think you want to add a coma here)
Section 3.7

In the statement of Theorem 3.2., when you talk about the mean field limit, you are referring to n_1, n_2 —> infinity right? A depth 2 network as in Theorem 3.3. Or does it hold for an arbitrary number of layers?
lines 269 -, What you call “High level proof idea” is pretty opaque. I think I would remove it. Or I would remove lines 227 - 243, second column and expand a little bit on the proof.
lines 272 - 273, “this yields \mathbb{E}_Z[upstream\times local]” —> what does that mean? I understand you are referring to the expressions in (2) but what do the terms upstream and local refer to?
lines 220 - 223: What does \partial_2 \mathcal{L} mean here? Does that mean the partial derivative with respect to the second argument of \mathcal{L}?
lines 239 - 241, second column, it is not clear to me why the frozen random features yield heterogeneous shifts across neurons. In fact, even before this, what do you mean by “shifts” in this setting?
Section 3.8

In the statement of Theorem 3.3. Here as well, it is not clear why/how you initialize over the indices.. you have a tuple {n1, n2} that is in the index set of Init ? What does that mean concretely ? Why just a two tuple? What do n1 and n2 represent ? do they refer to the number of neurons in the first and second layers ? Does the result only hold for depth 2 then?
Same Theorem, what do you mean by the “coupling procedure”?
Same, What does 
 refer to here ? If I’m not wrong, you don’t introduce it. I understand it measures the deviation between the mean field solution and the finite solution but what measure do you use?
Limitations:
see above

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Review Concerns:
NA

Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Rebuttal by Authors
Rebuttalby Authors (Haizhao Yang, Shijun Zhang, Janis Aiad)31 Mar 2026, 13:28 (modified: 31 Mar 2026, 16:13)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for the detailed and constructive comments. Below we try to answer under the 5k characters limit.

Sec. 2 (citation style). Agreed, we will make the style consistent throughout.

Sec. 3.1 (frozen RF; richness / support phrasing). The informal “
” wording was imprecise. What we use is a standard random-feature richness assumption: the linear span of 
 is dense in a suitable target class (e.g. 
). This richness is only required at the frozen first layer (
). For trained layers 
 we do not require a “full support in 
” statement for the trained objects.

Mean-field width limit. Yes: this means the infinite-width limit under mean-field scaling (neuronal embedding), i.e. widths 
 so empirical neuron averages converge to expectations under limiting neuron measures. This is not the NTK/lazy scaling and not maximal update parameterization (Hayou & Yang et al).

Sec. 3.1 (matrix form). We can rewrite 
 with 
 and 
, alongside the channel form 
. But this lacks low rank channel interpretration.

Sec. 3.2 (notation and expectations). 
 is a random neuron label with law 
; 
 is integration under 
. At finite width, labels are i.i.d. samples 
, so empirical averages converge to 
 under mean-field scaling. This matches the usual pushforward-measure picture on Euclidean weights. We will remove the misleading phrase “indices become continuous” and harmonize 
 notation.

Sec. 3.3 (loss; gradient flow; schedules). The population loss is 
. The mean-field ODE is 
 with layer-wise schedules 
, and Euler / SGD reads 
.

Sec. 3.4 (“better than null”; ReLU; mixing bounds). “Better than null” means 
 (strict improvement over the trivial constant predictor). Here 
 is the **channel rank** (number of channels). The formal global-optimality step uses 
, so plain ReLU is excluded in that theorem; for ReLU the obstruction is the dead-gate regime (
 with nonzero residual). “Mixing” means the (frozen) channel recombination matrices 
. A convenient sufficient bound is 
 and 
.

Sec. 3.5–3.8 (Lipschitz display; Picard map; wording). The Lipschitz display around 
 was confusing: 
 denotes frozen mixing, and differences in trajectories enter through channel features. A typical corrected form is 
. We define the solution operator 
 as the Picard map 
. We replace “upstream 
 local” by “backpropagated signal 
 local derivative,” and 
 means the partial derivative w.r.t. the prediction argument. “Shifts” refer to the neuron-independent translation-collapse mode in deep i.i.d. full-rank limits (see Nguyen et al section 5.1.2); RF-LR breaks that symmetry via frozen mixing and yields heterogeneous updates across neurons.

Theorem 3.2 vs 3.3 (hidden widths; coupling; symbol D). In Theorem 3.3, 
 are exactly the hidden widths in the depth-2 quantitative bound—hence a pair only; for depth 
 one would use 
.

“Coupling procedure” is not an extra algorithm: it is the neuronal-embedding viewpoint (i.i.d. labels 
, compare empirical neuron averages to mean-field expectations).

The informal symbol 
 denotes a concrete sup-norm trajectory discrepancy 
 on 
. For depth 2, writing 
 and 
, one may set 
.

Add:
 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer VCqo
Rebuttal Acknowledgementby Reviewer VCqo03 Apr 2026, 09:42Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:
I acknowledge the rebuttal and appreciate the clarifications provided. My assessment remains unchanged. I continue to believe that the work is of great quality and therefore maintain my original vote.

Add:
Official Review of Submission25189 by Reviewer RZVa
Official Reviewby Reviewer RZVa13 Mar 2026, 05:08 (modified: 01 Apr 2026, 18:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer RZVaRevisions
Summary:
This paper establishes a rigorous theoretical framework proving that multi-layer neural networks with a low-rank random feature (LR-RF) architecture can reach the global minimum, provided the mean-field dynamics converge. By introducing this low-rank bottleneck, the authors overcome the neural collapse issue found in standard full-rank networks. Empirically, the paper demonstrates on the MNIST dataset that LR-RF models match the accuracy of full-rank networks despite a massive reduction in parameters. Furthermore, the authors highlight that, unlike full-rank models, the LR-RF setting inherently preserves the geometric symmetry of the target functions.

Strengths And Weaknesses:
Strengths:

The paper presents a rigorous mathematical framework proving that multilayer networks with low-rank random features converge to a global minimum under mean-field dynamics, cleverly mitigating the neural collapse phenomenon.

The experimental design is clear and highly reproducible; it can be successfully verified the theoretical claims by reproducing the 1D high-frequency fitting experiment using the comprehensive details provided in the appendix.

Weaknesses

The theoretical novelty is somewhat incremental, primarily extending the framework of Nguyen & Pham (2023) by relaxing their initialization to a "frozen initial weights with low-rank features" setting.

The global convergence guarantee relies critically on freezing the mixing matrices 
 as random features. This architectural restriction significantly limits the theory's applicability to modern, fully end-to-end trained deep neural networks.

Theorem 4.1 provides an elegant explanation for feature learning, but it is rigorously proved only for a highly simplified two-point, two-channel toy model. Scaling this claim to continuous, high-dimensional real-world data remains largely empirical.

More critically, the core assumption maintaining the network's symmetric structure is overly restrictive and sensitive to hyperparameters; in my extended reproduction (using a 1D function with frequencies 
 and 
, depth 
, and width 
), the symmetry was largely imperceptible at rank 
 and only clear at rank 
, suggesting that this strict low-rank requirement significantly limits the theory's robustness and applicability to broader network scales.

Soundness: 3: good
Presentation: 3: good
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:
While symmetry is preserved at 
, it becomes largely imperceptible at 
. Since 
 remains an extremely low-rank constraint relative to a width of 
, this sensitivity challenges the practical robustness of the feature learning mechanism .

Theorem 3.3 bounds the finite-width approximation error by 
. Does the symmetry loss at 
 occur because the exponential factor 
 dominates 
, causing divergence from idealized mean-field dynamics? If so, is "symmetry preservation" a genuine benefit of low-rank architectures, or merely a fragile artifact of keeping 
 artificially small to prevent mean-field breakdown?

If 
 must be kept strictly minimal to preserve mean-field behaviors, how does this framework scale to datasets with higher intrinsic dimensionality 
? If evaluated on a synthetic dataset with controlled 
, is there a theoretical or empirical phase transition when 
? It would be insightful to clarify if the network loses its theoretical structural benefits once the required rank 
 exceeds what the practical width 
 can exponentially suppress.

Limitations:
Freezing the mixing matrices 
 as random features---this architectural restriction significantly limits the theory's applicability to modern, fully end-to-end trained deep neural networks.

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Rebuttal by Authors
Rebuttalby Authors (Haizhao Yang, Shijun Zhang, Janis Aiad)31 Mar 2026, 12:42 (modified: 31 Mar 2026, 16:13)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer. We will sharpen scope/novelty in the introduction and discussion and trim minor repetition.

Q1 (Novelty; freezing; applicability; Theorem 4.1). RF-LR / channelwise technical novelty vs. Nguyen & Pham (2023): see reviewer Z9GA Q1 (technical novelty); not repeated here. The revision will state more explicitly that our proof builds on their template.

Frozen first-layer features: same phrasing as Sec. 3.1 (reviewer VCqo); not repeated here.

For Theorem 4.1 we keep the two-point/two-channel proof as anchor; the revision adds a brief remark on positive feedback (details in Q4 below). Higher-dimensional/overlapping features stay empirical/open, not in the theorem’s scope.

Applicability: extensions/benchmarks are in reviewer ByWV Q1 (Additional experiments)

Q2 (Symmetry sensitivity; exponential factor). We agree this is important. The Gronwall exponential in finite-width bounds is a worst-case stability artifact; it should not be read as a causal explanation of any particular symmetry diagnostic. To avoid anecdotal conclusions, we ran a focused post-review sweep (width 1024, 5 seeds) and observed that the even-function symmetry diagnostic can improve when increasing rank. In this protocol, with 
 evaluated on a positive grid in 
, values decreased from 
 at 
 to 
 at 
, with comparable test MSE; the relative version dividing by output energy behaved similarly. In the revision we will report mean and std and tone down any universal claim that a specific rank is "necessary" for symmetry.

For transparency, we will also describe the exact protocol (target, evaluation grid, seed list) and, if space is tight, move the full table to the supplement while keeping a concise summary in the main text.

Q3 (Scaling to higher intrinsic dimension; phase transition). We agree this is one of the most important "beyond the theorem" questions. Our current mean-field guarantees are formulated for fixed rank 
 in the infinite-width limit; they are not, by themselves, a phase diagram in 
. We will state that limitation plainly in the discussion.

That said, we can still articulate a research roadmap. On the NTK / kernel side, how rank interacts with input dimension depends on controlling the induced kernel ensemble; related random-matrix analyses become delicate for low-rank structured weights, and we are building on recent frameworks for structured kernels (e.g. arxiv:2508.20036) to connect scaling of 
 with high-dimensional concentration. On the mean-field / Chizat side, recent scaling-law discussions suggest a "sweet spot" for faithful ODE descriptions when rank grows sublinearly with width (e.g. regimes such as 
 are often discussed as practically relevant; see also this scaling note). Separately, in LoRA-style NTK analyses, rank thresholds of the form 
 with 
 are sometimes discussed as regimes where kernel descriptions remain predictive; we treat such statements as indicators, not as theorems for our exact architecture.

Q4 (Mechanism behind the channel spike; beyond the two-point toy). The explanatory content is not the simplified two-point, two-channel statement itself, but the positive feedback it isolates. Fix a second-layer neuron index 
 and recall the low-rank pre-activation

 

For ReLU, gating enters mean-field gradients through 
. The backward signal for channel 
 (cf. the ODE for 
) is built from expectations over 
 of terms that couple (i) the mixing weight 
, (ii) the gate 
, and (iii) upstream factors (output error and later-layer weights). When channel 
 dominates, neurons 
 that contribute to the drift along channel 
 are precisely those with the gate ``on.'' Across 
, the effective weights in the expectation are then positively aligned with 
 and with 
, producing a large contribution to the drift of the 
th channel, which in turn pushes 
 further in the same direction: a reinforcing loop. The ODE-level mechanism is dimension-agnostic; the two-point theorem is a minimal tractable instance. We will add this explanation in the main text.

Q5 (Limitations: frozen mixing). Frozen mixing is not a universal model of all deep nets, but RF/factorized weights are standard in SciML/structured learning—and our results show global-optimality-type statements can still align with strong low rank under mean-field scaling (pushing back on “low rank always hurts”). We will rewrite limitations as a clear scope statement (assumptions 
 claims).

Add:
 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer RZVa
Rebuttal Acknowledgementby Reviewer RZVa03 Apr 2026, 13:43Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (b) Partially resolved - I have follow-up questions for the authors.
Reasons:
Thank you for the detailed rebuttal. While I appreciate the additional clarifications, some concerns remain.

First, the highly simplified setting limits the paper's technical contribution and novelty compared to Nguyen & Pham (2023). More critically, the frozen mixing matrices constraint makes it difficult to scale or extend this theoretical framework to modern, fully end-to-end trained deep neural networks.

Given these limitations in technical depth and broader applicability, I am inclined to keep my original score.

Add:
Official Review of Submission25189 by Reviewer Z9GA
Official Reviewby Reviewer Z9GA12 Mar 2026, 22:14 (modified: 01 Apr 2026, 18:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Z9GARevisions
Summary:
This paper characterize the mean-field limit for deep network with low-rank layer. The paper compute the mean-field limit, compute the approximation error and prove a conditional convergence to global optimum. Then the authors show that the low-rank network has a reverse spectral bias compare to full-rank setting.

Strengths And Weaknesses:
Strengths
This paper studied the spectral bias between low-rank and full-rank training, which seems to be interesting.
Weaknesses
The low-rank model this paper consider have a constant rank w.r.t to the width that tends to infinity, and only one matrix of the desomposed weight is trained, this seems to be far from practical settings. In particular, if we write the network architecture ( equation (1)) in matrix form, the efficient weight for each layer 
 is 
 where 
 At the mean-field limit, 
 and 
 remain constant.

Technically speaking, the proof of the Theorem 3.1-3.3 follows exactly the same route as in (Nguyen & Pham, 2023), and does not provide any new insight in my opinion. Could the authors elaborate more on the technical difference compare to (Nguyen & Pham, 2023)?

The authors claim that this formulation avoid the nuronal collapse problem in (Nguyen & Pham, 2023), which is correct, but it is due to the fundamentally simplified structure of the low-rank network. In particular, the nuronal collapse in (Nguyen & Pham, 2023) is precisely due to that the trainable weights in the middle layer is of size 
 and when 
 the weight matrix converge in a complicated way, which need to be characterized using neuronal embedding. In the setting of this paper, the trainable weights in all layers are of size 
 which means the architecture in this paper is effectively a stacking of multiple two-layer networks in the mean-field regime (in equation (1), given 
 
 is basically the output of a two-layer networks in the mean-field regime). Thus, technically speaking, the architecture studied in this paper is fundamentally easier than (Nguyen & Pham, 2023).

While I acknowledge the spectral bias for low-rank network could be interesting, the discussions in Section 4.1 seems to be too qualitative, and the results in Section 4.2 seems to be on too simple setting (only two point in the datasets). Could the authors elaborate more on why the low-rank network lead to high-frequency bias, and fully trained network lead to low-frequency bias?

Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
As I mentioned in the Strengths and Weaknesses part, the architecture considered in this paper is a stacking of two-layer network, I don't see the difficulty of proving the mean-field limit when both 
 are trained. Could the authors elaborate more on the technical difficulty?

I wonder how universal is spectral bias discovered in Section 4 for low-rank training? For example, if we do not consider network in the mean-field regime, do we still have such spectral bias?

Limitations:
Yes.

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Rebuttal by Authors
Rebuttalby Authors (Haizhao Yang, Shijun Zhang, Janis Aiad)31 Mar 2026, 12:39 (modified: 31 Mar 2026, 16:13)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for the careful reading and thoughtful questions.

Q1 (Technical novelty). We apologize for not being precise enough in the coupling argument presentation: one intermediate step was missing in the draft wording.

More precisely, relative to Nguyen et al., we do not claim technical novelty for the existence/uniqueness line of argument; we include those steps for completeness and to check that the 
 coupled mean-field equations are well-posed. That block is ancillary to the paper’s main message.

The substantive point is conceptual. Under convergence-type assumptions, the 
 channel trajectories could a priori lack spatial expressivity—independent paths might learn maps that poorly target localized input structure. Empirically, such effects can appear without mechanisms that diversify features across channels (see Fig. 8, fourth panel). We argue this is less of a limitation in our RF-LR setting because the frozen random first-layer features 
 provide the spatial diversity needed for channel-resolved adaptation.

Technical novelty (coupling). Our modification is to adapt their coupling template to the channel-wise sum (their three-layer assumption / coupling display). After introducing the coupling 
 and using bounded first-layer activations, one obtains a bound of the form


and one then controls the right-hand side by weighted coupled gaps (same logic as Nguyen & Pham), with low-rank-specific channel aggregation.

Our low-rank translation of this missing step is that the main coupling term is controlled through a channel sum under 
:

 

with notation matched to our neuronal embedding variables in Eq. (1). This is exactly the place where the RF-LR structure replaces the fully connected contraction chain by channelwise control plus mixing bounds.

We will make this explicit in the appendix by adding: (i) the post-translates-to display; (ii) the time-regularity condition for each channel,


(iii) the bi-Lipschitz/bounded-mixing constants used to close the backward estimate.

For ReLU, the closure step uses the sharp inequality

 
 
 
 

rather than any extra slack of order 
. We will state this explicitly (with compact-support / boundedness assumptions for pre-activations) to clarify why coupling across 
 channels closes.

Q2 (Spectral bias and Section 4). On universality and training regimes: we will revise Section 4 to separate statements cleanly. In the NTK / lazy training regime, the effective bias is governed by the kernel spectrum and one expects the classical low-frequency-first behavior emphasized in that line of work. The empirical tilt toward comparatively higher-frequency structure that we highlight is observed in a richer feature-learning setting for our low-rank RF architecture; we do not claim that this behavior is universal across all scalings (including finite width outside mean-field parameterizations or lazy limits).

Regarding Rahaman et al., "On the Spectral Bias of Neural Networks" (2019), arXiv:1806.08734, https://arxiv.org/abs/1806.08734, is an apt reference for the standard Fourier / low-frequency-first picture in sufficiently wide, kernel-like regimes.

On why low-rank maps might carry more high-frequency content (strictly heuristic, under review): in ReLU networks of depth 
, one can view NN outputs through "sum over paths" (per-region linear parametrizations); shrinking the effective rank reduces how many independent hinge directions can co-activate across depth, which loosely relaxes constraints on how linear regions can tile input space---informally, a less constrained CPWL can allow relatively more high-dimensional faces versus low-dimensional ones and thus alter how energy is distributed across spatial frequencies in a Fourier lens. This complements the shorter "fewer effectively independent hinge directions / boundary geometry" phrasing we will keep in the main text and will be marked explicitly as exploratory reasoning and ongoing review, not as a formal implication.

We agree Section 4 is currently qualitative and that the two-point analytic anchor is minimal. We will tone down claims, acknowledge the simplicity of the two-point model.

Add:
 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer Z9GA
Rebuttal Acknowledgementby Reviewer Z9GA04 Apr 2026, 09:34Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:
I thank the reviewer for addressing my questions and concerns.

While I appreciate the authors' detailed discussions on the technical novelty compare to previous work, it is still an adaption of the techiques in (Nguyen & Pham, 2023) to a simpler case, and the authors agree that the Section 4 is currently qualitative.

While I think it is fine to adapt the technical results to a slightly different setting to get new theoretical insights, which means the major contribution should be on providing new interesting insights, the currect new insights ( spectral bias in Section 4) are not strong enough in my opinion.

I believe the spectral bias for low rank network is an interesting question. I suggest the authors to study the problem in a more rigorous way under a more general setting than a two-point setting.

Thus, I would like to keep my original evaluation.

Add:
New Author AC Confidential Comment
* denotes a required field
Title
(Optional) Brief summary of your comment.

Comment*
Confidential comments (max 5000 characters).

Write
Preview
At the moment, the two reviewers who gave us 4 do not seem to raise major concerns. The main issue comes from the other three reviewers — the ones who gave 2, 3, and 3. Their main criticism is that our work is too similar to Nguyen & Pham. I believe it is very important for us to address this point clearly.

I think we should highlight the following points:

We do build on the method of Nguyen & Pham, but our key contribution is to apply it to a new low-rank structure, which has been shown to be numerically effective.

Our proposed low-rank structure freezes roughly half of the parameters, and as a result, it helps avoid neural collapse, which is the main bottleneck in prior work.

Our structure can be easily extended to end-to-end training: one only needs to freeze parameters in intermediate layers.

Although from a mathematical point of view our model may look like a “simpler case” of a standard network, training behavior is a completely different story. Following the same logic, one could argue that a CNN is just a special case of an MLP with a particular parameter-sharing mechanism — for example, by placing the filters into one large matrix — and therefore CNNs would have no independent value. Clearly, that conclusion would be unreasonable. The same applies here: even if the model appears structurally simpler, the training dynamics and practical implications are fundamentally different and meaningful.
TeX is supported
Characters remaining: 3572
Readers*
ICML 2026 Conference Program ChairsICML 2026 Conference Submission25189 Senior Area ChairsICML 2026 Conference Submission25189 Area ChairsICML 2026 Conference Submission25189 Authors
Signatures*
signatures
Edit History
Readers*
ICML 2026 Conference Program ChairsICML 2026 Conference Submission25189 Senior Area ChairsICML 2026 Conference Submission25189 Area ChairsICML 2026 Conference Submission25189 Authors
Signatures*
ICML 2026 Conference Submission25189 Authors
Official Review of Submission25189 by Reviewer ByWV
Official Reviewby Reviewer ByWV12 Mar 2026, 02:01 (modified: 01 Apr 2026, 18:00)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer ByWVRevisions
Summary:
Nguyen & Pham (2023) showed that in the mean-field limit, learning dynamics of a multilayer neural network with at least four layers with degenerate i.i.d. initializations, i.e., "at an intermediate layer, each weight evolves as a function of only time, its own initialization and the initial biases associated with its connected neurons". This means that if all biases are initialized to the same constant, then the weights in the intermediate layers remain mutually independent at all times. This paper proves that when the intermediate layers of an arbitrarily deep neural network constrained into a low-dimensional space, and then trained with frozen random features, the degeneracy that comes from i.i.d. initialization is avoided in the mean-field limit. Furthermore, they show that under this configuration and i.i.d. initialization, the learning dynamics converge to a global minimizer. They also provide a characterization of a feature learning mechanism in the mean-field limit.

Strengths And Weaknesses:
Strengths: Leveraging random feature maps to avoid neural collapse in the mean-field regime is an interesting idea.

Weaknesses:

Lack of additional experimental results on the performance of low-rank random feature neural networks. If they avoid neural collapse in the mean-field regime, we would expect these architectures to outperform the multilayer neural networks when gradient-based methods are initialized i.i.d..

There is no convergence guarantee provided for the learning dynamics of low-rank random feature networks under i.i.d. initialization (also see the question below).

Derivations heavily rely on a previous work of Nguyen & Pham, 2023. It is very unclear what is actually new in this paper.

I would strongly recommend mentioning the Assumptions in the main paper thoroughly with discussion about them, rather than hiding them in the Appendix since they are quite important for the analysis.

Soundness: 2: fair
Presentation: 2: fair
Significance: 1: poor
Originality: 2: fair
Key Questions For Authors:
Typo at (040) "...is low rank all we need for global convergence ? Low-rank networks...": There is an extra space between the question mark and the first word of the second sentence. Typo at (042): "...with 
. substantially...".

At (044), it is posed that "can gradient-based training still converge to a global minimizer, or is full rank essential?". I didn't understand why improvement in the training performance, as shown in Fig. 
, implies that the low-rank structure affects the dynamics of the gradient-based optimization algorithms such that they converge to a global minimizer. First, the low-rank factorization might change the loss landscape completely if some global minimizers are not feasible for the specific factorization determined by 
. Note that the low-rank problem is different than the full-rank problem. Therefore, the functions represented by the global minimizers of the full-rank problem can be different than the low-rank problem.

For the Fig. 
, how are different momentum rates selected for the corresponding level of rank constraints?

Do low-rank random feature networks differ from the multi-component multilayer neural networks proposed by Zhang et al. (2023) only in their low-rank structure?

Typo at (137), 
 should be denoted by vector notation.

Could you clarify the convergence assumptions made by Nguyen & Pham (2023) for multilayer and two layer neural networks?

Nguyen & Pham, 2023 established the global convergence of multilayer neural networks trained under stochastic gradient descent (SGD). Is your convergence analysis also based on SGD?

Limitations:
yes

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Rebuttal by Authors
Rebuttalby Authors (Haizhao Yang, Shijun Zhang, Janis Aiad)31 Mar 2026, 12:48 (modified: 31 Mar 2026, 16:13)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for the careful reading and constructive feedback. We respond below in the same order as the review.

Q1 (Additional experiments). We agree. We will extend to higher-dimensional 2D/3D highly oscillating targets, CIFAR, and PDE/operator-learning setups anchored to standard benchmarks: PINNacle https://arxiv.org/abs/2306.08827, PDEBench https://arxiv.org/abs/2210.07182, SPDEBench https://arxiv.org/abs/2505.18511, PDEArena https://arxiv.org/abs/2209.15616 (benchmark code with that paper). We will report variance over seeds (and frozen-feature draws) and include a matched full-rank i.i.d. baseline under the same optimizer/schedule as RF-LR.

Q2 (Convergence guarantee under i.i.d. initialization). Our main theoretical statement is conditional: if the mean-field dynamics converges to a limit, that limit achieves global optimality within the RF-LR parametrized class under the stated assumptions. We do not claim a general unconditional convergence rate or finite-time guarantee for the training dynamics themselves. We will foreground this in the camera-ready and avoid wording readable as an unconditional convergence theorem.

Q3 (What is new relative to Nguyen & Pham, 2023). We refer to our answer to reviewer Z9GA, Q1 (technical novelty), for coupling/channelwise RF-LR adaptation relative to Nguyen & Pham (2023). We will add a compact "What is new" paragraph and short roadmap so the main paper stays self-contained at a high level.

Q4 (Assumptions in main text vs. appendix). We agree. We will move a compact assumption block into the main text with brief notes per item (structural richness vs. technical regularity/couplings).

Q5 (Typos and notation). We agree and will correct the reported issues in the camera-ready, including the extra space after the question mark, the broken fragment around "substantially," and the vector notation typo.

Q6 (Global minimizer versus low-rank feasible set). We agree with the reviewer's logic. Low-rank and full-rank parametrizations generally induce different feasible sets in function space, so one must not equate global optimality in one problem with global optimality in the other. Our statements concern global optimality within the RF-LR class, conditional on mean-field convergence. Empirical curves should therefore be read as evidence about training behavior and landscape effects under that architecture, not as a claim that every full-rank global minimizer is matched or attainable under rank constraints. We will clarify so figures are not read as full-rank global-optimality claims.

Q7 (Momentum selection across rank in the figures). Momentum and learning-rate values in the paper were obtained from optimizer sweeps over a prescribed grid; we will report the exact ranges and selection rule in the camera-ready. In the main plots we will compare ranks only under matched momentum and matched auxiliary hyperparameters so that rank is not confounded with per-rank tuning. Extra momentum-vs-rank sweeps go to appendix/supplement.

Q8 (Relation to Zhang et al., 2023). We agree that, at the architectural level, the model is essentially equivalent to the multi-component template of Zhang et al. (2023). Our contribution is mean-field parametrization and training analysis: we derive the RF-LR mean-field ODE for the 
 channel bundles, prove conditional global optimality when dynamics converge, and formalize channel specialization. Zhang et al. do not provide this mean-field training analysis.

Q9 (Convergence assumptions). We will add a concise boxed summary in the manuscript. In one line: their global-optimality limit uses weighted coupled-
 conditions under 
 (backward-aligned weights 
 coupled gaps—not bare marginal convergence) plus time-regularity (ess-sup decay of selected velocities) to commute limits along the flow. Our appendix keeps the same template for RF-LR with channelwise aggregation across 
 bundles; the box restates the two-layer schematic and a time-regularity line. We will note that 
/transport control is sufficient (Hölder/moments) for those weighted 
 quantities, not definitional.

Q10 (SGD versus mean-field ODE). Yes. Following Nguyen & Pham (2023), the mean-field ODE is the rigorous continuous-time scaling limit of SGD when step size 
 under mean-field scaling; see their Prop. 23 (discrete-to-continuous). Our finite-width bound tracks 
 sampling and 
 discretization with the usual Grönwall factors.

Add:
 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer ByWV
Rebuttal Acknowledgementby Reviewer ByWV02 Apr 2026, 21:33Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (b) Partially resolved - I have follow-up questions for the authors.
Reasons:
I would like to thank the authors for their responses. Even with their responses (both to me and the other reviewers), I am still not sure about the novelty from Nguyen & Pham. Therefore, I keep my score.

Add:
Official Review of Submission25189 by Reviewer A6kZ
Official Reviewby Reviewer A6kZ11 Mar 2026, 20:08 (modified: 04 Apr 2026, 01:43)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer A6kZRevisions
Summary:
This paper studies low-rank networks with random frozen features in the mean field regime. The authors introduce a low-rank feedforward architecture, deriving mean field ODEs and proving their well-posedness. They prove that if the training dynamics converge, it converges to a global minimizer of the loss. A mechanism for spike learning is described and then numerical experiments are presented.

Strengths And Weaknesses:
Soundness.

The proofs are generally thorough and assumptions made clear. Both Section 3.8 and appendix F.6 claim that in practice, the channel specialization avoids the exponentially large approximation bound, however this is unsupported and is not returned to in the discussion. Similarly, the claim of empirically validated minimum network size in appendix F.6 is missing any citation or justification. The toy model in Section 4 is well designed to highlight spike learning mechanism, however is limited in scope and doesn't address whether such a mechanism plays a role for more than two data points and two channels. The numerical results would benefit from variance analysis over different realisations of the frozen random features.

Presentation.

Sections 1 and 2 are well presented, clearly placing the work in the wider literature. Section 3 is dense with mathematical notation, although the logic flow is good. In Section 4.1, Theorem 4.1 is referenced before it is presented, making the logic difficult to follow. Some figure captions are unclear/uninformative, especially figures 8 and 9.

Significance.

The work extends previous mean-field analysis to the low rank regime. The choice of architecture to avoid neural collapse is novel and interesting. Theorem 4.1 is limited to a two-point data distribution, and although an informal argument is made for when its conditions hold, it could benefit from a more developed treatment.

Originality.

To the best of my knowledge, the application of mean-field theory to this particular model is novel.

Soundness: 3: good
Presentation: 3: good
Significance: 4: excellent
Originality: 3: good
Key Questions For Authors:
1. Convergence guarentees. Are there conditions under which convergence is guaranteed for RF-LR models (those mentioned in section 3.5 or otherwise)?

2. High-dimension extensions. Does the spike learning mechanism extend to higher dimensions with overlapping spikes in feature space?

3. Sensititivity of randomness of fixed feature maps. How sensitive are the convergence guarantees and empirical performance to the initial draw of the frozen feature maps for finite width networks?

Limitations:
yes

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Rebuttal by Authors
Rebuttalby Authors (Haizhao Yang, Shijun Zhang, Janis Aiad)31 Mar 2026, 12:36 (modified: 31 Mar 2026, 16:13)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for these additional points. They help us sharpen what is proved, what is empirical, and what remains open. Several of these points overlap with other reviews, and we will make the overall paper wording consistent across all responses.

Q1 (Convergence guarantees). Our main guarantee is conditional on convergence of the mean-field dynamics: given convergence, we can conclude global optimality within the RF-LR class under the stated assumptions. We will revise the abstract/introduction/theorem statements so this reads as "conditional global optimality" rather than as an unconditional convergence-rate claim.

We do not currently prove an unconditional convergence rate for deep RF-LR, and we agree this remains largely open. In the revision we will be explicit about which assumptions are used for the global-optimality implication (non-degeneracy of the initialization relative to the trivial predictor, and the regularity condition used to pass from a vanishing conditional gradient to a vanishing residual), versus which statements are empirical/heuristic.

Q2 (High-dimensional spike mechanism). The ODE-level feedback behind channel specialization is dimension-agnostic at the algebraic level: it is driven by the same identity

 

together with gating/backprop terms. To make the mechanism more transparent, we will add a short explanation in the main text. For ReLU-type gates, the derivative term is an indicator


so the drift and backprop signal for channel 
 is built from expectations over 
 of terms that couple (i) the mixing weight 
, (ii) the gate 
, and (iii) upstream factors (output error and later-layer weights). When one channel becomes slightly dominant on part of the input space, it increases 
 there, turns on more gates for neurons aligned with that channel, and thereby increases the drift along channel 
. This in turn pushes 
 further in the same direction: a reinforcing feedback loop. The two-point theorem is a minimal tractable instance where this loop can be proved cleanly.

What we currently prove is a minimal two-point anchor; for general 
, overlapping channel contributions complicate the rigorous isolation of "well-separated spikes". We will state this distinction clearly, keep discussion in dimension 
 honest as primarily empirical, and add higher-
 experiments/diagnostics as space permits. We will also note that taking 
 is mainly for visualization clarity; the algebraic structure above does not rely on 
.

For ReLU specifically, we will also clarify the main degenerate failure mode: convergence to a stationary point where gates are off on a large set (so 
 despite nonzero residual). In RF-LR,

 

is a sum over 
 channels, so the event that all channels are simultaneously dead is a joint sign-alignment event across channels. Under symmetric random mixing and non-degenerate channel features, one can heuristically view this as exponentially unlikely in 
 (e.g. on the order of 
 pointwise), which motivates the "high probability in 
" intuition. We will keep this strictly as intuition (not as a proof step) and will be careful about how it is stated.

Q3 (Sensitivity to the frozen draw). The analytical conditions are not tied to one sampling recipe: once mixing matrices are frozen, the arguments use bounded mixing and a richness/diversity assumption for the frozen first-layer features, not Gaussianity. Concretely, a convenient sufficient form is: for all 
 and 
,


and hence

 
 

almost surely. This can be satisfied by any bounded-support mixing distribution (and is also compatible with nonnegative/NMF-style mixing); there is no Gaussian-specific requirement in the theory.

At finite width, performance can vary with the frozen draw. In our current experiments we often use Gaussian/Xavier initializations for convenience, even though those are unbounded; in the camera-ready we will clearly separate the theorem assumptions from this engineering choice, and where feasible we will include multi-draw variability (multiple frozen draws, multiple seeds) in addition to the usual seed variability. We will also clarify that any ReLU dead-gate-across-all-channels probability discussion is heuristic: in a symmetric mixing picture it can be exponentially small in 
, but we treat this as intuition, not as a proof step.

We have not yet run a systematic post-review sweep over independent frozen draws with optimization held fixed; in the revision we will either add a compact version (if space permits) or state this explicitly as a limitation/future-work item.

Add:
 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer A6kZ
Rebuttal Acknowledgementby Reviewer A6kZ04 Apr 2026, 01:19Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.
Reasons:
I thank the authors for their response.

Their rebuttal has addressed my questions regarding convergence guarantees and spike mechanisms.

The treatment of variability across different frozen mixing weights remains insufficient. Furthermore, I believe that such variability is generally more relevant to practical applications than variability within frozen mixing weights. Although the authors indicate that such analyses will be included in the camera-ready version, it is currently difficult to assess whether and how their methods apply in this scenario without seeing those results.

Thus I will keep my current score.

Add:
About OpenReview
Hosting a Venue
All Venues
Contact
Sponsors
Donate
FAQ
Terms of Use / Privacy Policy
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

Low-Rank Structure Suffices for Global Convergence of Neural Networks in the Mean-Field Limit | OpenReview



