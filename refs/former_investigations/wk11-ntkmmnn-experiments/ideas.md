



Monday : analysis of experiments



Setup :
I use relu and relu dot, and I compute NTK for 2 layers MMNNs, 
i'm in 2 dimension, and I take several random points that forms my dataset (n points), I choose dim=2 because NTK only depends in scalar products (zonal kernel) and curse of dim , 2d is a setup that allows big experiments with expressivity in applications, I take vectors on the unit sphere
and sampled uniformly. also output of MMNN is 1d

i chose my beta, before all in former MLP's NTK analysis beta is set up to zero, I choose it nonzero
because the dependance in it is highly non trivial compared to MLP, the key can be in this, I take only 3 values

I also setup different internal ranks, 2 5 10 .. 100 uniformly in log

a config is an uplet (dim,n,beta,rank)

once that is done, I compute NTK using montecarlo. I use montecarlo estimator because the internal rank is also the internal
dim of gaussian weights in W for the 2nd network, so that integration of the exact gaussian measure suffer from curse of dim
(also the measure on the unit circle), but the computationnal issue it represents makes the use of MC the appropriate way to compute
(W has 1000 entries, and MC using 300 random points)
I recall also that the weight matrix is scaled by 1/sqrt(rank)

once I've implemented this, I compute 50 different NTK matrices (over a uniform dataset),
the choice of those values is purely empiric along the computations I've done for various configs before, it remains doable in a night
with sufficient points to trace scaling laws

I recall that since the 1st output follow a gaussian process

once that is done I compute NTK values densities (random variable that is indexed by rho the dot product of entries), NTK spectrum density
since we uniformly sample our points, we can use bootstrapping techniques to enhance our dataset for density viz and estimators (means, std)

since i've remarked that the matrix has a structure of compound matrix, i compute it's L2 loss over this subspace,
I also compute std of diagonal NTK wrt rank for each config, same for off diagonal std, that in log space


for beta = 0 it converges to 0 with a particular slope (that can be computable I guess)
for others it converges towards a value that is also computable by computing (partly done in the overleaf report, it requires integrating
erf, and gaussian cdf in 1d, 2d, special function that is highly non expressible in simple functions, I'll calculate it on python also),
but the dependance in beta appears to be linear or quadratic


remember that for a particular couple, the ntk is random, so that the mean is a function of rho, dot product between 2 entry vectors



I've plotted this, dependance for my points, it appears to be linear with a good R² for rank = 100



then I compute NTK distribution, indexed by rho, for every (rank, beta), with kernel (epach, gaussian, cosine etc ..) and an appropriate bandwith
to match std

(i suspect a power law tail behavior, i'm veryfiing this  using hill estimators, its running now)

I observe that the std decay as a power law (observed also in the other plot)

and I trace the mean wrt rho, I compare also for the corresponding beta with
the NTK function for 1 layer MMNN, it appears that there is a gain, that corresponds to the contributions of 2 terms in the recurrence formula
and the scaling look appropriate, 

also with rank growing, I can observe a behavior that quickly tends towards a gaussian distribution (the tail can remain highly power lawed)
but at the end it remains gausssian for high ranks


I also can see a very low convexity in the rho dependance of the mean ntk, which can be characteristics of MMNNs
(it's also something remarkable when we compare 1 layer MMNN and 2 layers MLP)

This observation is that the NTK for 1 layer MMNN looks as the same as 2 layers MLP's NTK but with a factor 2 (divided by 2)
for every beta values

the main thing i've remarked iis : low ranks enhance NTK spectrum with many equal positive eigenvalue with same order
as the leading one for low rank
and std of the ntk decay exponentially with internal rank, so that the random thing that appears in NTK for MMNNs can be moved out and I
can reason globally with NTK means with a lot of confidence ! that's great because I can condition the intergral very well even
if I have deep gaussian processes.

I'll keep you posted tomorrow for the other things





so with those experiments I conclude that the NTK for MMNs



for relu, converge with 1 when ranks goes to infty with beta 0

in fact it converges toward a value related to beta
out and in giagonal coeffcients lower with rank

std is also divided with rank, and shape of ntk is gaussian for in diagonal (or looks like pyramid also, to link with rmt)

and outdiagonal is tailed, with mode near 1 but become gaussian with ranks going to infty

attention faire attention au w car si on le resample pas alors on a plus d'independance, ce n'est pas la meme variable aleatoire
, en vrai ça fait une dependance en 1/sqrt(1000) en defaut d'independance donc osef

justifier le 2 layer par la simulation gaussienne possible
on a un petit scaling

appliquer directement approximation















Tuesday : 
power law loss for rank scaling with 2 and 4 coeffificents, need to go further to get b etter scaling laws
remark for zero beta
ok pour beta qui grandit la variabilité est portée par b, sinon on a une std qui est quasi nulle et qui decroit exponentiellement
donc on peut se placer dans le regime beta 0 ou petit beta d'abord pour analyser tout ça, mais ça n'a pas grand intéret, donc on va stack les layers sur ce regime !! 
et sur l'autre regime on stack pas les layerset on analyse directement fonctionnenemement en fonction de beta, puis avec un random kernel en grand dim (random kernel matrix resultats par ex de el karoui)


et là on fait les comparaisons dans ce petit régime en stackant les layers avec terjek,
avec l'observation spectre 2x plus petit, c'est vraiment sinquad qu'il faut run alors pour voir des images
(faire un rapport dessus)
ensuite on calculera la hessienne et ntk et on comparera, avec resutlat d'ethan dyer et jacot

ensuite faire un training, check le NTK de manière experimentales et faire des animations pour montrer le NNGP,
comment il varie, ce que fait une descente de gradient stochastique, visuellement quoi
puis en PINN avec la loss ce que ça fait comme NTK,

et là on doit voir ce que l'on trouve, calculer le ntk local avec neural tangents et le ntk théorique et comparer !!!




wednesday : i guess that the sinquad directions are taken for 2 layers with a specific stuff because the big slope is always orthogonal ; 
for fcnn the big slope is always at 0, maybe because the sintu make everything goes bad ? idk 

also put without bias, compute ntk, maybe compute ntk and hessians
over a specific subspace of parameters with log dim (johson lindenstrauss)
to get a better experimentation, with hand computations before to get the theoretical
baseline

orthogonal initialization, and rotation invariance - GOE

sometimes the directions mess up a bit for MMNN, idk why, it just looks like
fcnn but with fewer frequencies, and sometimes not, maybe something to see in the directions

i think the most important ideas can be to show that the optimization landscape
has a radically new shape for MMNNs, and locally the ntk can tackle some other stuff
but here compared to fcnn there is for some reasons a complete different new horizon

investigate also with full relu, and sine at the end
, with this high frequency cos function
we can rescale the optimization landscapebecause of the rank thing and curse of dim
i can get that this matter of frequencies in the functions can be seen in the optimization landscape
because there is something in the weights that can make the opt land wavy, a bit weird

also need to understand the weight vector and its influence for sintu for instance


supprimer bigrunv1
et enlever les html de tout le git et foutre ça sur un wetransfer jsp




Thursday : it appeared that it depends a lot on the initialization when the mountain vs wavy appear, this is very curiious and need to b e investigated systematically

sint1 is not sintu, there is maybe a s inside to disentangle, still did not seen with resnets

now i can see that the landscape is very propice to gradient descnet so it's very interesintg, like the mountain stuff, lot of experiments to show 

for seed 462084 we get that non wavy, it depends a lot on the initialization

the value of the loss is like the same, 1.5 in meaan when mid blue, with this high frequency cos function
we can rescale the optimization landscapebecause of the rank thing and curse of dim


540281 is super less wavy well on y1[-3]

346356 cool
918573
2649 pas trop wavy mais cool*
155544 un mix des 2, vraiment curieux; signe de superposition

65246 vraiment super

-- now homogeneous by same nmuber of activation composed
250889 idem



--- now in the vector spcae 2d 


679175 petite pente

we should not take conclusions by seeing the 2d stuff because by random projection, we can see all of what i describe on a high dimensional optimization landscape, so we cannot have nay conclusion
but there is maybe something to disentangle in the way the optim landscape suddenly has this shape or not, and if it's true globally with a lot of weights involed in, by computing some high dimensional metrics, not only
the hessian spectrum