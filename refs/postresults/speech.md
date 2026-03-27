

meaningful research now is more like papers management and high level ideas
too trendy research like muon is shitty, we need to stay out of the bulk

i wanted to run experiments then papers out etc ..


ICML reviews coming this week so friday

i still think i've not read as many papers as needed because i began to work on smth then 2 papers appeared


The whole idea to promote is that low rank answer many difficult questions 


high dim / low dim 

to keep track of our idea our question is how optimization is difficult for depth/width 
under ntk or statistical perspectives

is attention better at the same parameter budget and why 
for sobolev training, which optimizers
resnet analysis (neural ode) & network analysis
frequency and hierarchical feature learning under low rank descriptions & optim wrt r, sobolev fourier
low rank resnets ?
central flow and landscape description
sgd/adam/optimizer switching
muon , 
KST kolmogorov representation sqrt(n)


ntk for dsrn, attention hermite etc .. mathematica
tensor programs for full framework, mup and hyperparam transfer








stat / high-dim : this is what I'll be doing next month under stat phy perspectives






focus : ntk/attention, (rank lm head / gradient)



right now we can't do research alone
feynman : send a mail to misof x2
terjek x2



in this moment where many papers are out every day or weeks i should make connections
litterature review





low rank spectrum : benigni paquette


low rank resnets : features learned inside, and highly fundamental (why no one talked about this)



depth/width : feynman misof, depth width 

transformers : low rank & networks lm gradient, expert routing



central flow : 3rd order taylor necessary, implicit preconditionning
optimizer switching when central flow + variance too big (not only plateau) (3rd order principled+1,2 order optim)
this begins strong theory for principled switching




sobolev training : training in fourier space against frequency bias, theoretical feature learning analysis


landscape description : topological very long but holes, valleys in high dim has no vlaue (tda)
but statistical desc (scaling exponents, lr, distribution of stopping times,)


mmnn feature learning : characterizing frequencies through depth/width and spectral analysis through layers
refine my result on optim and unify it


hyperparam transfer : learning rate / sgd

arxiv : waiting for reviews, better experiments and refinments





experiments : low rank resnets, transformers


ask questions by mail
 
make a short presentation about this project, assign something to work on that 













CHIZAT
https://arxiv.org/pdf/2603.18168 https://arxiv.org/pdf/2509.10167 chizat : backward pass with entrywise scale 1/D
completeP with residual scale 1/(Lsqrt(D)) under M linear in D
'it was clear from MFODE literature that residual scale as 1/ML when M,L grows and D fixed leads to local feature upadtes === now MLU D linear in LM and residual sqrt(D)/LM
2LP block corresponds to MMNN, gradient clipping
DMFT = tensor programs  (its algorithmic/programmatic counterpart Tensor Programs), but limit are sequential
1/sqrt(d) mean ode prob bound
their formulation is general enough to include transformers and many resnets type like analysis (i propose to use attention residuals)
'4It is infactnotclearwhetherscaling-updk isbeneficial'
"
ForAdam, from[OrvietoandGower,2025,Eq.(9)],theLipschitzpropertyholdsuniformlywhenthesequence
ofbatchgradientshasuniformlylower-boundedempiricalvariance"
for transformers the analysis is weird, before and after attention/mlp nothing matter and 1/2 factor mean them, very weird and not coherent with what saw in practice, depth regime don't correspond to what we see
d model and finite head count act as implicit dimensions D


resnets formulation remove the optimal transport tool from measures and dmft/mckean formulation is better
assumptions : phi lipschitz, jacobian too ;;; losses same ; input bounded
"WefocusonGDonlytofixideas;our
techniquewouldapplytoanyupdaterulethatisaLipschitzfunctionofthesamplegradients
suchasGD,SGD,clipSGD,Adam5,etc."


"Clearly, the forward pass after k steps of GD in a ResNet is very similar to (17), but there
is an important difference: in the ResNet, the ˆZj,ℓ
k are not sampled from the limit dynamics
and are not independent, except at k = 0. The core of the proof of Theorem 1 consists then
in an argument by recursion over k to jointly control ∆Z
k, ∆h
k and ∆b
k. This argument shows
that the only new source of error at each iteration if the approximation error of (17), and
its analog for the backward pass."" it's a propagation of chaos argument

"There is a direct analogy between the
convergence of (17) to the Mean ODE (16) and the classical result that mini-batch SGD converges
to gradient flow as the LR tends to zero"


neural tangent ode vis the linearization but in mean field regime, kernel replaced by function zeta

my idea = change chizat equation for MMNN that ensure no neural collapse, be sure of scaling to have neural collaps )

' (MLU) when D = O(LM'
gradient clipping 
'it is necessary however to assume that ρ has at most linear growth since otherwise the Mean ODE can explode in finite time.'

maybe resnets are not useful for frequency features but for nlp ? but better for ode features 
L = Θ(P 1/5) and M, D = Θ(P 2/5), which means M,D square of L, L = sqrt(M,D)
"practical architectures where M = Θ(D) ,nno prior work has considered the large D behavior of the Mean ODE dynamics, as we do in the present paper.""



depth and width do not commute in the mean field regime ! 

TODO : carryon H/d analysis, according to them they share same structure (D = d_model, M = H)
they have not proved the minimal noise result, transformer is much less linear

carryon lowrank analysis from central flows ?






https://www.arxiv.org/abs/2509.24914 ZDEBOROVA single head :
the non linearity in transformers comes from a head only, they do matriw sensing problems







transformers lenka sagitova retrieval : https://arxiv.org/pdf/2603.03993
patch retrieval task
" the learning dynamics of the attention can be expressed in a closed-form over m and r"
" However, later in the learning, the misplaced heads can
rearrange, by performing an excursion, so that at the end the split is even and the loss optimized. This shows that the
structure of the multi-head softmax helps SGD to navigate in the loss landscape and not to stay stuck in bad local
minima."
"Rather, thehierarchical learningcomes fromthestructureofthemulti-headattentionitself"

"AssumingthatCovθisdiagonal, the headslearnthefeaturesfsequentially, fromtheoneswiththelargestsignalVarθf totheoneswithsmallersignal. ThishappensbecausethedirectionswithlargerVarθfhavesmallernegativeeigenvalues.Thismakesakeyconnection
withwhat[2,3]observeinpractice,wheretheattentionfirstlearneasytaskssuchthatbigramstatisticsandthen hardersuchthatn-gramsandinduction"

"This proposition gives a prescription on the right number H of attention heads: each point of the support of Pθ
should correspond to a different attention head"


"The most noticeable property of the Bayes-softmax activation is its ability to deactivate some of the heads via
normalization. In our setting, heads that are not aligned with the signal ˆ k introduce noise that cannot be reduced
by other means"

"the number of heads H that can be pruned without significant loss of performance is close to H − F;"
H is for location, d_k for in-context












the landscape of spiked tensor  : 





https://arxiv.org/pdf/2502.20003v2 : 
they prove replica trick works in gaussian linear models
"the validity of these characterizations does not depend on convexity. Instead, it relies on a more subtle
stability condition ; the replicon conditions" 


"The replicon condition, which we
fully characterize, determines when the high-dimensional optimization landscape is sufficiently well-behaved
for our analysis to hold.Notably, this condition can be satisfied even in clearly non-convex settings "



"The Gaussian Min-Max Theorem (GMT) — The GMT" + Approximate Message Passing (AMP) — Originating in spin glass theory as the Thouless-Anderson
Palmer (TAP) equations [Thouless et al., 1977] for the Sherrington-Kirpatrick model  ;; They have become a cornerstone of high
dimensional inference with synthetic data.  

applications : negative regularizations, and general loss functions (like cauchy) ;; proving optimality of a loss function in the case of ε contaminated outlier model 



proof idea : We first establish a lower bound using Gordon’s
Min-Max Theorem [Gordon, 1988, Thrampoulidis et al., 2015]. We then construct a matching upper bound
through an algorithmic realization using a message passing algorithm. 


"This result provides a precise analytical description of the spectral estimator’s
performance without requiring random matrix theory tools.
Our analysis reveals that the algorithmic and statistical aspects of spectral estimation are intimately
connected through the state evolution of AMP." 




https://arxiv.org/pdf/2106.03791 : 

model : mixture of gaussian for data input and labels, regularized 1 layer perceptron

"we prvoe closed form equations characterizing the asymptotic distribution of weights , to compute then trianing and generalisation error"


'AMP has closed form asymptotic at each step", 'we use several refinement with spatial coupling, multilayer approach and matrix valued variable with non separable updaete function"








I want to use taht for high dimensional diffusion






central flows : we analyze optimizers under the lens of central flows, and not full trajectories, then examine some of (rmspro)
rmsprop implictly steer towards lower curvature regions

"Suppose that gradient descent is oscillating along the top Hessian eigenvector, 
. Let denote w the point where we'd be if we were not oscillating."
"each negative gradient step on the loss implicitly takes a negative gradient step on the sharpness of the loss"
we don't care about momentum

"we make the ansatz that the time-averaged dynamics of gradient descent can be captured by a sharpness-penalized gradient flow"
this IS A MODEL to compare optimizers 'Ouranalysistreats the loss function as a black box, and never uses that the optimization
problem at hand involves training a neural network"

for 1 or several eig : projection/3rd order formulation
they derive central flow for several optimizers and can compare them


takeaway "Thus, our work suggests that acceleration via regularization is a vital design principle for adaptive optimizers."
"An exciting direction for future work is to intentionally design first-order adaptive methods with such implicit preconditioners in mind"











flat or sharp https://arxiv.org/pdf/2602.05065 : MORE imbalance = MORE sharpness
linear model like hayou
"For all global minimum of (12), the spectrum of the Hessian is identical"
" the first term of (14) suggests that D−1 layers contribute to the sharpness in the same way, while the last layer (the second term of (14)) contributes differently. "
label Noise covariance imbalance as a source of sharpness : it is the data distribution that changes and determines sharpness





non euclidean gd https://arxiv.org/pdf/2603.05002 : include MUON
"our explanation of this behavior is not fully satisfying, as our theory
only proves that non-Euclidean EoS is divergent under a specific initialization, whereas in practice
we observe that this divergence seems to occur quite generically" TO FUTURE WORK




central flow NTK : https://arxiv.org/pdf/2507.12837
linear network , EOS analysis with 1 outlier (thankks terjek) : i think that their data assumption can be removed for any network from MLP at the EOC paper
they assume a rank1 structure for weights (low rank can be induced by us and constrained)

they prove classical central flow/EOS analysis, 
empirically aligns with central flows
(appendix) with relu K become rank 2 



