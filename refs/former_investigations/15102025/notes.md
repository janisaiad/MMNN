
 curse of dim what's his claimed, because for my result I can have some constants depending on the dimensionality, we've talked about this 2 months ago 


ill conditionned problems for bad pdes or bad data when NLA fail
curse of dim
MMNNs ? transformers ?

aaron sintayehu

answers : 

my conclusion for other and experience is that in general we have the curse of dim in optimization, that comes from , in general even though people claime when we restrict to small function space, we still have it
and case by case, large function space, and a target function in C^s, if we use simple like relu tanh sigmoid, curse in NN approx, 
so NN function space far from target function space, we grow NN size exponentially to match 

but if not exponentially large, then approx 

based on this fact we can prove that NN optim can not find the param for high accuracy, NN small exponentially large, if we want optimization to go from this init to target parameters to approximate this is very difficult, the descent is exponentially long, we have exponentially in d number of iteration, that's why

in this way we can prove that for some activation function we have curse of dim, this is the conclusion from "operator learning curse of dim"
number of iteration needed,  this is an open problem (work with student)
his conjecture is that even though , we may still require a large number of iteration
when we try to use NN
C/sqrt(n) with n number of samples, and this constant is target functino dependant
it's markov chain basic fact montecarlo
tipically, defined on a domain like -1,1^d, then the volume is exponential, then in this case this constant is in 2^d, and generalization is 2^d/sqrt(n), some people claim there is no curse of dim, 

his understanding is not precise , if we want to maintain good generalization error
n should be exponentially large in d


in generalization this is the curse, even though the target function is in the function space, the optim error is still exponential, it's his idea


for some very special target space, mmnn can work better
in the worst case scenario mmn should still suffer : because it's a smaller


for operator learning this is a different kjind of perspective, the target operator has a very special structure, it's different from generic function task
he show for certain type of operator, if operator has a compositionnal structure, then we show that general and approx there is no curse of dim because compositionna structure is the same as for MLP composition and that's why there is no curse of dim
optim is still an open problem wether is still an open problem, 
conjecture is , and if compositionnal structuwe don't have curse of dim

for resnets this is not done, the power for resnet in approx
for any resnets we can generate MLP of the same size, of the same order

adding identity is 2 relu and that's why, resnets just partially , it makes vanishing gradients less, it's for optimization convenients oh 
in terms of approximation resnets, we can use relu to generate identity
so approx and generalization

the learning target has a structure , we have the same order of neurons for tansformers

numerically resnets is better , highly non convex with mlp, for resnets it's more 
there is some paper for resnets quantitative 
not still completely solved, experienec for order 



PINN popular for academic to write a paper, because some people get their paper easily published, it's not clear when a pinn is very useful to write a paper, because original version of pinns is poor, you can always have a better method, but in practice pinn can be useful for several reasons, first of all pinn is not mathematical justified (doumeche) the loss function is based on residual, some optinion on numerical PDE people express they won't propose loss function with residual, the solution may be far away from the solution , tipically they trasnform the 
(like deep ritz), so the residual is very naive
secondly iterative solvers to solve the PDE, iterative solver with high condition we discretize, for high accuracy we use larger number of grid points

iterative solvers with high number have high condition !!! very good insight

with a large number of parameters, the expectatction from hzyang say that the number of iteration is larger also for PINNs (see practically)

prreconditionning to achieve high accuracy in general but in deep learning based solvers there is no preconditionning technique, it's difficult to preconditioning in DL, the central idea is try to esssentially assign some operator 

for NN it's to complicate, this kind of problems 



baicheng : the bound of the norms of parameters , people don't care about the bound for parameters, but what's the realtion beteween the acuracy and the bounds of number of parameters
explicitely compute some bounds for sobolev training, all of the construction are very concrete, finding connections is harder to get a bound, it really depends on professors and reason about the problem this is important, for numerical computation we need to pay attention of the bounds of the norms of the parameters, in numerical computation it's difficult to achieve this parameters if bound is big, numerical difficulty to identify what we want, this is a viewpoint : 
we should also investigate the precision required to achieve trarget parameteers (single of double precision) it's important, the problem is that approximation theory assume that all the number on the real line can be computed, in some cases a NN may require some number very large or very low that are far from the capacity a computer can offer !!!!  so there is this numerical limited precision  : open problem it's very difficult to work on the approximation in limited precision not a single paper, for polynomial parts, it may not be very important that's why there is no effort in this direction, there is always other popular, a few year to solve it 

shijun paper, relu to approximate general activation, get some approximation bound
we use others to approximate relu, other activation has least approx power has relu, we approximate relu with others

many other functions has similar maybe better 

some reviewers states that relu and relu square are not differentiable and this is 
and this points is measure zero set, this is a laugh mdrrrr this is a funny argument

smooth general to approximate relu and relu square then DSRN and we solve the issue the relu has proposed, very good direction to remove the non differentiablity
it's doable has long as p is not infty, the intuition we compute derivative we only have 1 point without derivative, so p not infty this is manageable, smooth function to approximate derivative of relu, the support of this jump is 0 so we can control the error,
and we don't need the shijun idea
also the UEAF, for sobolev functions this is not already done to achieve arbitraly small error, it can be mathematically beautiful

there is a kolmogorov inequality , for similar results, this is a very good direction
high dim function with kolmogorov arnold theorem, but there is other idea to reduce the dimension, 
solve 1d case, then generalize in high dim, the keypoint is KA theorem
maybe a larger constant,, fixed network size only matters

1d case before (see with baicheng and shijun)
piecewise network to approximate , in terms of sobolev norms


random linear algebra
















anisotropic noise



BG LI : EAUF for sobolev space, 2 main difficulties
main difficulty is 1d, bramble hilbert
outer layers to 2d +1 and fix inner layers
1d existence we use polynomials and we use polynomial with rationnal coefficients
we can prove that every single polynomial can be approximated by the activation and


EAUF is every explicit and we think our activation should not be 
they tried c_inf activation function to use sobolev 
they focused on Linfty norm for EAUF
get the former papers presented (jason lee), lei wu, denny wu)


to do the approximation for the derivatives, the problem is very challenging in the sens where (hierarchically divide the 1 dimension domain) 

outer functions scattered we cannot derivative, for higher dimensions it's very difficult to maintain the derivative

if we have the 1 dimension Cm approximation, how to generalize it to higher dimensionnal 

kolmogorov arnold, we do not have closedness if we consider smooth functions
the main difficulty, when we arrange high dimensional values to 1 line, keep track of the smoothness is difficult 

existence for 1d case is not difficult (they focus on this part 1st)

smothh KST approximation (outer layers exponential number of layers)


write down the thing and make sure it's correct ! using mathematica



it introduce randomness in the ntk by random basis, more mathematically consistent, more accesible to extend to infinitely deep networks, with resnets also


main story that we want to share,

better experiments
possible title and abstract

test a lot the adam sgd observation !! strong numerical results
1st jump, other jump (test with seed)


global minima do respect symmetry but not the case for local, if we have symetry maybe this is a global minima