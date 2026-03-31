OpenReview.net
Search OpenReview...
Notifications10
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
5 / 5 replies shown
Add:
Official Review of Submission25189 by Reviewer VCqo
Official Reviewby Reviewer VCqo16 Mar 2026, 18:27 (modified: 24 Mar 2026, 15:15)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer VCqoRevisions
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
Official Review of Submission25189 by Reviewer RZVa
Official Reviewby Reviewer RZVa13 Mar 2026, 05:08 (modified: 24 Mar 2026, 15:16)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer RZVaRevisions
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
Official Review of Submission25189 by Reviewer Z9GA
Official Reviewby Reviewer Z9GA12 Mar 2026, 22:14 (modified: 24 Mar 2026, 15:16)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Z9GARevisions
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
Official Review of Submission25189 by Reviewer ByWV
Official Reviewby Reviewer ByWV12 Mar 2026, 02:01 (modified: 24 Mar 2026, 15:16)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer ByWVRevisions
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
Official Review of Submission25189 by Reviewer A6kZ
Official Reviewby Reviewer A6kZ11 Mar 2026, 20:08 (modified: 24 Mar 2026, 15:16)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer A6kZRevisions
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

