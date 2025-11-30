It's been 2 months since i've decided to tackle MMNN optimization under various mathematical perspective to try to explain why it's working so much
on fitting high frequency functions easily

I read around 20 papers a week for my research
I have a tremendous amount of remarks and comments, i've written some small parts on the overleaf but the work is in fact done on (goodnotes)
I'll only talk about a very small fraction of what i've done today nor the next weeks, everything will be written in the paper

i'll talk about all of what i went through to give a rigorous framwork for explaining observations done on MMNNs


what are the observations / experiments : 
in the mean field limit, 1000 width, MMNN fit very highly oscillating functions using adam, and sgd after some pretraining, stepwise plateau training  

G:\Mon Drive\JANIS AIAD Internship - NTK for NN\mmnn_training_shifted\mmnn_training_shifted\L4_W512_R15_E30000_lr0.001_bs100_ratio5\L4_W512_R15_E30000_lr0.001_bs100_ratio5\th0.007lr_decay_steps1000gamma_20.99

along the depth, mmnn can build highly oscillating very smooth functions and not linear by part
low rank functions forms a good dictionary basis for learning repeated patterns 



why hard to explain : 
for approx and generalization This is a matter of spectral bias that MMNN overcome easily 
in terms of optim, this is a question of are global minimizers sharp or flat


explanations : 
core - plateaus corresponds to frequency steps, low to high, this is an observation, this is also in theory to expand, 2 way (ben arous / misia very proficient)
rigorously in the mean field, not training 1st layer is GREAT, it involvees possibility to approximate any function (dict) at any moment, unless it may collapse 






future : 
central flow experiments & dynamical stability of sgd
NTK with reversing bias

For now i know very clearly where to go, the only bottleneck is the high technicality needed to rigorously prove all of those ideas
O've never been so confident in this direction after having read more than 200 papers on the subject i have isolated the key points

To have a breakthrough paper about mitigating NN frequency bias (especially) we have to extend MIsiaK framework


what to do : ICML NeurIPS
.






























## title should reflect the conclusion 

## sell the research for rkhs formulate a , if we have a better layer design, what is the reason for having better performance, 
## for research we eclude the benifit in terms of expressiveness, background story for people curious , to give some new perspective
#


by monday morning you can take the overleaf archive and put it on arxiv,

today i'll present the first preprint that describe how to tackle MMNNs through the NTK perspective, the 2nd preprint will deal with sgd behavior and central flows, the 3rd with landscape through mean field convergence proof and low to high spectral bias proofs i will hope to hold
1st two to icml 2026 by january and 3rd for neurips 


The goal is to throw the bases for NTK analysis, as well as the original NTK paper. 


there are 2 main non trivial results, 1st on the NTK rkhs and statistical properties induced by low ranks,
the second shows for 2 hidden layers, when low rank is comparable to input dimension, NTK gram matrix obeys the same as for 1 hidden layer
with a MP spectrum slightly shifted and scaled, that precondition the NTK matrix from having 0 eigenvalues


NTKs induces 2 randomness, the data one and the kernel value random variable

## insight and guidance from this contribution from practical problem solving to link them


## remove finite-width

## formulate the problem we want to investigate

## then explain how the contribution, in discussion of related wor, distinguish former and new work
## why our work can better answer the question ask


Future directions involves tensor programs like derivations, spectral behavior wrt to depth i conjecture to be quadratic
and finite-width corrections computed with tensor that avoid too much weights correlations (computed with feynmand diagrams)


In the whole paper i use the naming RF-LR instead of MMNNs because statistics people would understand more
this denomination, i think it's the good one because there already is litterature on random features neural networks, same for low ranks
and having framed 'random features and low rank' is great for people to find the paper when reviewing the litterature on the subejct.

## notations MMNNs/ RF-LR redundant, state at a moment that we , inspired by very attractive performance from MMNNs, which use this technique
## and our research investigate this 2 technique in general, and analysis would be applied to other features (add a discussion like this )
## for now we use rf-lr

## low rank and/or random features RF/LR ; in this way we have a unify framework for different architectures, and notation can be consistent
## rf/lr only and both, if we use mmnn together with 
## 

to begin with the paper, after an introduction about MMNNs performance, random feature choice avoiding riemannian projected descent and
low rank for linear computational scaling, I review the litterature, please you can read and modify this part at your convenience
in introduction.tex  ! 


I first define the model network and explain the training setting again to be sure reader do not avoid we don't train a part of the network.
I also disentangle assumptions about the initialization and activation homogeneity
I then derive its EOC intialization.


## add some comment in the definition, LR/RF, it's better to have discussion, 
# our discussion apply automatically, to make the scope consistent, and the architecture consistent with the whole scope
## just define rf and LR, and quick comment, for lR only, and RF only
## if we only use LR, RF

## 10 minutes to review, we'd better using low rank only, and both together and compare

I then define the base and ederivative kernels that are differents from the NNGP one, and I disentangle that
in a paragraph, 

after I state a theorem of composition of gaussian processes conditional to previous outputs and I recall the ntk is a random variable
where generating them have to be done linearly wrt layers thanks to conditional GP




After I state and describe the recursion formula, the main difference lies in the +1 that leads to a quadratic number of term
to be summed in general.  this is the main difference in the NTK recursion and can explain then the scaling wrt depth
## be more rigorous on this point for the stating of the randomness after initialization, because the training is conditional to init


This is a preprint so i'd focus on make it comprehensive for anyone reading it, this is not the conference paper i'll modify it on purpose


Then  I explain what the homogeneity assumption allows for NTK random varaible derivation



After I state the main result


1st the NTK behave the same for the RKHS, despite the randomness the 
taylor series of the kernel in fct of the dot product has the same behavior as arccosine kernels,
and from a former result from francis bach paper, this allows to get the spectral decay, same as for MLPS



This involve a technical challenge to compute the puiseux non integer taylor series under the expectation
with the fisher correlation distribution, that is highly non trivial to deal with (hypergeo functions)

to do that, i managed to do hand calculations for the integral but i was stucked at a moment with the ref given for dealing
with the taylor behavior of hypergeo functions, but with some help i managed to prove that strikingly the mean behavior of the ntk
is the same



## typo before theorem 4.2,,, quantitative

## curse of dimension, eigenvalue fast decay, that means very close to 0, the upperbound is , exponentially fast being worse
## wrt to the dimension if the condition number is fast, curse of dim for optim
## add one more contribution curse of dim in optimization by specifying eigenvlalue distribution
## mention curse of dim don't emphasiez in the discussion
# in general curse of dim, people may wonder with the target function space, not enough 
# for barron space no curse of dim, C(d)/sqrt(n)
## this is some unknown problem, curse of dim in optimization, does not rely on the target function, what is cool ? could it be 
## keep this in mind (this is unknow), the curse of dim is unknow in optimization


this result is very important and shows that low rank gives no rkhs disadvantage, the NTK rkhs is very bad and decay fast, here
we have the same but at a very better computational cost.





the 2nd result comes from computing the spectrum of the NTK gram matrix, as a marchenko pastur spectrum.
The proof holds strong assumptions on data to deals with El karoui theorem. to do that i disentangle ntk between random variable
contribution and data contribution. in a regime where the low rank behave linearly with input dimension, and large width quadratically.

Those strong assumptions gives insight on how to get the spectrum for general data distributions,
and i've identified the future direction from another paper.



In a future version i'll push by december i'll throw the experimental analysis showing all the scalings that appears with pretty plots.



Please read the abstract, conclusion and discussion, I tried to make the conclusion to understand all the contributions
for someone that will only read intro/conclusions,
and the discussion details the purpose to go further for finite width directions, fewer assumptions, scaling behavior, and 
experimental validations to go then


finally i would like to put it on arxiv the next week, i think it'll be ready for monday, i'll tell you by mail

also for both professors i sent you a mail regarding my deadlines for phd applications, i'll enter
your email adress on the portal today for the 1st batch, i'll have around 3 to 4 application each week and this during 4 weeks 
please read it carefully because some department are statistics and some are applied/computational math so the discourse is not the same for each but I trust you on this perspectiv 

29 reminder

## icml 2 colons eight papers

## simple itemzied neat novel results



## numerical example first, then time to do theoretical
experience, conference, then journal,



duke columbia  brown cut, applied math be careful, cs fit more

oden/csem oden is more computational, csem optimization

umd both math and amsc


Efficient Paths to the NTK Regime: Low-Rank and Random
Feature Models with Rigorous Kernel Analysis


orcid


## random features ntk deviation analysis avoided
its about fundamental ntk analysis, not mmnn, if u want to tackle low rank how should you do it, we tacjke mmnna and NN simlicifation at the same time 









we should do the letter talking about the preprint in prepaartion 
done everything alone beside my studies



B.LI : 
EAUF in sobolev space with sobolev norm, function in W2 and norm W1


flops in plot


FlopS / LIGHTER - FASTER §§

NTK is preserved : 
low rank in terms of what

full picture .
On Parameter Reduction through Low-Rank Neural Networks with NTK Preservation Guarantees

new insights, contribution ;;; 1 short sentence to summarize to reframe every paper


transcript, , we prove it
timeline prove it in a short time
significant, important, difficult : famous professor working on that , no results, to show i work on importnat and difficult problem



 : 
 - my transcript
 - my cv
 - paper list
 - all the results i have besides the preprint
 - slides containing all the other results