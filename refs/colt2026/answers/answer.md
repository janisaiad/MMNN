1)

benefits of deep kernels :
the benefit of deep kernel comes more from a point of view where the NTK helps then to derive PL (polya lojasiewicz) bounds
or smallest eigenvalue bound for the data dependant ntk gram matrix 
(see montufar et al for pyramidal networks, udell et al for the use of polya to derive convergence bounds )

this work kis preliminar to study finite width corrections of the ntk, in a world where low rank and random features
gives same kernel, finite width corrections calculations (Misof et al) are very complicated and their analysis is the base of the correspondance between
nn and field theory caculactions (that are just using isserlis/wick theorem for gaussian perturbations)

with rf lr, having the same kernel allow to get a far easier analysis of those correlations since low rank bottleneck

from misof et al, it appears that some scaling laws are experimentally shown for depth dependant kernel, suggesting that a 
finer analysis of finite width corrections are a very interesting object to study ; for instance, having computed some exact
scaling laws wrt depth of those feynman calculations, one can derive an optimization bound for finite width neural network training

getting this bound for full rank network is a hard path, but our paper throw the basis to show low rank network is a reliable path
with more tractable analysis
getting opt bounds for rflr can then be more tractable than full rank



the benefits of having a deep kernel also comes from the ntk gram matrix analysis (Terjek), its spectra is shifted linearly in depth
which means that convergence guarantees can occur with (at the cost of numerical stability using discrete learning rate that should then be very small
as 1/maxeig), so you're not learning features 
in order to learn features you need a network with some lower width, and hence the compromise betwee, optimization guarantees
can be shown in montufar et al, with only 1 layer going to infty, smallest eigenvalue is bounded away from zero

but convergence can be very slow as woobling effects can occur, this means smallest eigenvalue is not the good object to study, and condition
number remains the good one, this was the object of study ; you can lower r, having bigger L, get optimization guarantees
and get interpretatble partial functions that are lacking in every of former ntk network analysis (because weights are not moving
and full rank matrix have operator norm that have low variation)

with this framework, the low rank bottleneck features are moving  ; at a particular x, f_r(x) is really moving substantially (at a rate depending
on the condition number that we now bound away from 0 !)

the total number of features functions if r*l, which is exactly the condition number bound

finally, very few analysis has been done for the NTK of very deep network (only is some very very simplified settings ernest ryu et al),
this is still an open question to explain why very very deep network are not easily trainable,
neural collapse is one of it , low rank bias in training (jacot et al 2025) could explain this, and the ntk perspective seems to be a workable path
if quantities of interest are tractable to compute, ie finite width corrections,

we think that this analysis of finite width corrections is a major open problem in theoretical understanding of neural network theory that would fit
the COLT community (and not major conferences one where people do not believe in the NTK analysis)





the choice for w and b fixed comes from several facts ; previous analysis has shown that fixing those weights allows to reduce the cost by 2 without
having (shijun zhang et al)

secondly, if we train them, there can happen instability in training and one should prove weights W do not move that much 
finally this instability removed by performing riemanning gradient descent over low rank manifold
3rdly there is an interpretation of approximating the feature function in barron space that occurs when they are not set up, this is not subject of analysis and
it is deferred to futur work, giving mean field wasserstein gradient flow analysis of global convergence in the mean field regime (or in the mu-p (yang hayou et al) / maximum local feature update regime (chaintron et al))

4thly previous analysis shown that in practice low rank networks or lora have a bias towards better training for (see https://arxiv.org/pdf/2402.16842 absract)
s


we would rework our phrasing for the camera ready version 






2)



we will clearly define 1per and kperp 
and also to cite ; sorry for those issues




3)

typo to fix sorry, we will state those explicitly, and explain more the terminology / summary we use

to explain why one should consider this architecture, we refer to previous answer, this is for us a way to show
that previous guarantees from 2019 to 2026 on the NTK for full rank network has a refreshing view under the low rank lens
at the cost of having a probabilistic analysis to perform for the eigenvalues of ntk gram matrix spectrum

having r interpretable features learnt in those partial functions is the feature we want to show
the feature learning regime that people involve comes from THE WEIGHTS, weight variation is interpreted as feature, we believe
to give a refeshing air of what a feature is, not in the weight space, but in the function space, if the partial function 
f_r is moving in a non trivial way by aggregating very small weight variation, this is the 'local feature' regime and not 'feature regime'

the previous simplest way to create small batch or interpretable observable from a very wide network comes from aggregating the distributional shift
in a wasserstein space formulation of training that occurs in the mean field regime ; 
here we show that the basis are thrown to create other observables, those bottleneck functions, that en plus diminuent le nombre de neurones

beyong the ntk regime, zhang et al motivates the use of them showing that very low training error can be obtained from their network when fitting highly
oscillating functions, this is a justification that is experimental but not optimziation related, here we give one under the ntk lens


it's not an issue to scale the learning rate inversely with depth, but under a numerical stability analysis it give very small learning rate
say (1000  Layer Networks for Self-Supervised RL, wang et al) , 
combining those very deep networks with small-mid width is usually not trainable (256 in their paper) and have 2**26 params,
which means no analysis can give you which learning rate to choose 

previous optimal scaling give you 1/sqrt(L) https://arxiv.org/pdf/2310.02244 so you trade having an optimziation guarantee with full rank network longer training time (using smalle larning rate)
this work also obtain 1/sqrt(width) optimal learning rate for the infinite wdith limit for very deep netowrk

here we show bounder learning rate give you still optmiization guarantee but smaller training time !



depth and rank commute should be something, effectively the NTK analysis showing kernels are the same then depth and rank influence on the kernel function
is not in the kernel but in the optimization process or its gram variance, not in the zonal kernel function and eigendecomposition in spherical harmonics


and we agree that further compruatinos of the NTK finite width wrt rank eand depth should lead to  a less confusable use of "commuting" (if future anlysis show it)
having a direct influence on the NTK values and not only its variance or smallest eigenvalue gram shift

thank you for this accurate statement

