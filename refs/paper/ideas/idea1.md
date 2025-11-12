



the high level idea to explain MMNN training dynamic is the mean field one 
we can clearly see that the 1st layer correspond to functions learned by nuniversal approximation
but those functions are then used in  arecursive manner to build more and more frequencies  in a highly symetrical manner  even for non symetric batches (symmetric deffect to measrure)


training is characterized by 2 regimes, mean field, mup or NTK, the whole training procedure were done using muP parametrization
some of them using the MF one




the goal of the paper is to prove that MMNNs gives a great advantage given the linear parameterization , the low rank structure gives parameterbudget advantage
but this is something that we should see only for three layers, when a weight matrix factorizes, for 2 layers no benefits arrives and the model lies in the random feature model

we have to be careful, because there is mmnn training 

we can clearly see in our experiments that MMNNs gives dictionary learning or even wavelet learning ! it performs learning
of base functions to be streched and placed at a particular interval 

the learning mechanism for MMNN can differ from what we see in theory, the approximatino comes from selectionning intervals where one or the r-th wavelet, after training
lie in after streching, but when we plot the partial functions learned we can see that what's really happening is more "building high frequency spikes' at particular places





the very main main idea is this from globalconvergences  " 
We prove that the MF limit, given by our framework, converges to the
global optimum under suitable regularity and convergence mode assumptions.
Several elements of our proof are inspired by \cite{chizat2018};
the technique in their work however does not generalize to our three-layer
setup. Unlike previous two-layer analyses, we do not exploit convexity;
instead we make use of a new element: a universal approximation property.
The result turns out to be conceptually new: global convergence can
be achieved even when the loss function is non-convex. An important
crux of the proof is to show that the universal approximation property
holds at \textit{any} finite training time (but not necessarily at
convergence, i.e. at infinite time, since the property may not realistically
hold at convergence). " !! the main breakthrough is that we can factorize the middle weight matrix and put other n1 n2 go to infty, we have a linear regime of number of parameters
wrt width, and for width around 1000 we are in the MF regime


this strong MF regime idea also comes from no dimensional perspective on the input data, we are dimension agnostic, and then we don't care
about high dimension statistics features arising from NTK kernel analysis. because this is not what we see in practice

then we can choose to randomize or not our model, this do not change the universal approximation property because of putting relu functions together in a tranable way

by just training 1 part we avoid riemannian GD, divide by 2 number of parameters training, and 



from exploringlowranktraining We show that spectral initialisation offers equivalent performance when compared to traditional initialisation schemes. Then, we show empirically that the singular values do not play a major role in improving performance and that it is the direction of the singular vectors that matters. This finding is in contrast with prior beliefs \cite{khodak2021initialization} about the role of singular values in retaining the scale of initialisation.  We establish this by setting the singular values to ones in Equation \ref{eqn:svd}.

factorization but it's not what we want to do


there is a difference 
Consider the difference between the vanilla gradient descent update (unfactorised) \(W_{t+1} = W_t - \alpha \nabla W\) and the update performed in the factorised setting:
\begin{align*}
W_{t+1} ={}& U_{t+1} V_{t+1}^\top \nonumber \\ 
\end{align*}
\begin{align}
W_{t+1} ={}& (U_t - \alpha \nabla U)(V_t - \alpha \nabla V)^\top \nonumber \\ 
\begin{split}
W_{t+1} ={}& W_t - \alpha \underbrace{(\nabla W_t V_t V_t^\top + U_t U_t^\top \nabla W_t)}_{\nabla_t} \\
{}& + \alpha^2 \nabla W_t W_t \nabla W_t ^ \top
\label{fullupdate}
\end{split}
\end{align}




we tackle this difference by just training the layer U 


from this point of view i can try to adapt the mean field ODE formulation to try to get convergence results, or discovering something new in the equation
but before i need to do also strong numerical experimentsfor the mean field regime, that means tracking the particle distribution
and seeing which ode it satisfies


a major aspect is that the training procedure is adam
operator switching was a major aspect in our discovery, we discovered again that switching sgd in part of training where adam is unstable due to EOS
is tackled very well by sgd, the curse of adaptative learning rate disappear; this will be tackled in another paper i n a more extensive manner

the goal is to show in an ablation study what makes MMNNs optimization great 



the story i want to tell is that the MMNN training can be understood through the lens of already understood high dimension learning theory see learningquadratic
because this work is done in the "extensive rank"

In this work we focus on the ``extensive-rank'' regime where $r \asymp d^\beta$ for $\beta \in (0, 1)$ and $\rs \asymp d^{\gamma}$ for $\gamma \in [0,1)$, and place a power-law assumption on the second-layer coefficients: $\lambda_j\asymp j^{-\alpha}$ for $\alpha \geq 0$. Our setting is motivated by the following lines of research. 

in the ben arous paper, multi index models are tha latent directions " it is known that the training dynamics typically exhibit emergent (or
 staircase-like) behavior — long plateaus followed by sharp drops in loss"

every plateau corresopnds to a term in the additive model, and at the end you can only learn a certain number of it (r in the paper, not the same r, because in the paper it's 2 layers), btw they explain the scaling law in openai paper 
The asymptotic risk behavior in Corollary \ref{cor:asympriskcont} is visualized in Figure \ref{fig:asymptoticriskgf}  (see also Figure~\ref{fig:intro}(b) for empirical simulation). The figure illustrates how the sharp, step-like emergent curve at $\alpha = 0$ (as observed in earlier works on multi-index learning \cite{benarous2021online,abbe2023sgd}) gradually transitions into a smooth curve as $\alpha$ increases. Notably, in the light-tailed regime $\alpha>1/2$, our risk curve resembles the neural scaling laws in \cite{kaplan2020scaling,hoffmann2022training} which takes the form of $\cR \sim 1/(\text{Data size})^a + 1/(\text{Model size})^{b}$, where the data size can be connected to optimization time under the one-pass discretization, which we analyze in the ensuing section. 

and their assumption to live in the stiefel manifold is important (weights orthogonal in bigger data dimension, less weights number that d) !! this is important
because we can have same scaling laws for MMNNs with pretty small beta in fact, and even tune this beta
also this work should be  understood under the lens of SETOL because of the SVD criteria given in SETOL

also we recall that we're doing this as taking the random feature space being high dimensional ! because of large width
and we train on that, and we do that both

why is this important ? because what we saw int he training dynamics whas that the relu part is just selecting where the pspikes are
that means what matters only is beta/w, w  ; w for the frequency and beta/w for where the spikes live, 
this is a better parameterization of random features, and we don't care about that


every datapoint are moved to a bigger embedding space and the network learn how to use this embedding structure
to construct dictionary functions 

this paper show that we have several scaling law in the training curve that comes from several hermite contributions

using the MMNN we add 2 difficulties, what's coming from 3 layers and the relu (not quadratic) difficulty

I don't think that we can understand MMNN optimization using hermite exapansion, okay the random features natural spaces then comes from 
hermite decomposition with chaos expansion



i think thaT my random features lives in a manifold that ressembles the stiefel one,
using linear combination to combine them, so not quadratic case but linear case ?

at the end the trainable A is a vector like w in learningquadratic, in a high dimensional feature space

and what's happening if we don't train only the 1st layer ?



breakthrough from meanfieldlandscape2layers
$(ii)$~Multiple finite-$N$ local minima can correspond to the same  minimizer $\rho_*$ of $R(\rho)$ in the limit
$N\to\infty$. Ideas from glass theory \cite{mezard1999thermodynamics} might be useful to investigate this structure.

"
We will prove below that, for $\beta<\infty$, the evolution \eqref{eq:GeneralPDE_Temp} generically converges to the minimizer of
$F_{\beta, \lambda}(\rho)$, hence implying global convergence of noisy SGD in a number of steps \emph{independent of $N$}.
"


from the globalconvergence it's having a full supoort at each time that gives universal approximation
but the random features keep that ! we keep a huge support because it remains the same !





i've identified that i should take back the math ODE material for 3 layers but simplifying it, using this insight i could prove non quantitative convergence result



##0611

my idea is to write an incremental paper, because from random features, low ranks, optimizer switching, high dim point of view, central flow point of view
generalization, approximation, adding the PINN loss and sobolev training framework from papers found, the litterature is very big and I need to disentangle every piece
of contributions I add

NTK and RKHS with randomness (high dim also), partial training then random, the spherical harmonics point of view (multi index model also)
inductive bias of low frequencies and lower X (maybe explained by the NTK) and NTK interpolation , and also tfinally the unifying tensorprograms theory added with muP to work well the theory, then the terjek-like analysis for the NTK spectrum with depth (ideally that's cool because we can go in the NTK regime then), from tensorprograms we inherits hyperparam transfer, then the NTK for infinite width and NTK for DSRN
the PINN analysis go then (ernest ryu)

i've not even talked about the transformers analysis, and the hermite expansion i wanna try, also the spectrum of kernel random matrices

and at the whole end, the NTK review book, and then the DL optimization book

mainly my results lie in the mean field framework (between 500 and 5000 width), and I need to prove that it beats MLP for conference paper also by adding the benchmark
in comparison for 3 layers
so for the arxiv i before put all the math i've done to have a great math paper, for the conference i'll work on the benchmark, sincerely all the math i've done so far

i'll also give the insight for 3 layers by convergence theorem and universal approximation as explained in globalconvergencesthreelayers because this is what we see in the experimental results


i'm gonna sell this under the random features framework before to inherits from the generalization/approxiamtion litterature already given
recall that mean field approx don't care about input dimensions, sos it's an orthogonal viewpoint on tackling the problem

more and more we will construct the preprint adding results in this direction like tensor programs

in fact mmnn is Low rank random features networkn


goodnotes : https://web.goodnotes.com/s/F0IFxLELb1470d6AGRdbxH#page-2















## paper whole timeline
The goal of the paper is 1st to provide extensive results explaining numerical experiments related to mmnn, especially the super convergence observed
especially 3 part of training where the problem convexify
we do 1st an extensive litterature review, to be added at the end because of showing up computations
we should mention a lot of people in the review, also the montufar paper with only 1 hidden layer growing

we provide 1qst explanation of why MMNN tends to have great training behavior, we conjecture this kind of architecture forms a good preconditionning technique
to combine RF and low rank is mainly due to  https://arxiv.org/pdf/2209.13569 remarks and a way to make hand computations tractable at a large scale

1)

in fact the early part of training appears to be very bad for SGD and the NTK can partly explain this ?  this is a line of research that is investigated for mmnns
we try to use NTK to explain early training and the apparently bad conditionned starting


we use the NTK to confirm that in this regime there is no disadvantage of using this kind of network due to RKHS being the same

from extensive numerical experiments, we believe different frequency timescales are learned in the order, making the landscape autosimilar, so that 
explaining early stage of training leads to same as final stage of training, that is super convergece


symmetry breaking then recovery during training and after convergence seems to be very 

also sgd switching gives us a way to quantify at which point the training convexify


NTK, we have the recursive formulation for the NTK, with the probabilistic view point and std wrt r so concentraion bounds holds for r and also
for input dimensions; we believe that this NTK regime that appears with very smaller parameter budget can lead to better understanding of 1st part of training behavior

then we show experiments confirming the NTK training behavior for practical tasks and also comparing predictions/exp for a simple NN cosine regression
we increase dimensions, r see how practical NTK behave compare to theory and where lazy training stops, we still want to show the 1st part of lazy training analyze the NTK inductive bias for lower inots absolute values (we show it appears)

it won't be a benchmark paper but a physics theory inspired paper
mathematically, we believe also that having r between 5 and 50 lead very good results in general for many tasks that can lead for around 30 good concentration bound
and enough expressivity with fewer weights (Tflops analysis, maybe scaling law)

about expressivity we describe the RKHS (BE CAREFUL THERE IS EXTENSIVE APPROX THEOREMS FOR RF MODELS), from bach paper, but also misiakewicz
the goal of this RKHS analysis is here to provide statistical guarantees that our intuition for MMNN approximation results will holds
and give quantitative generalization bounds

then we extent in a finite width correction way the NTK (as explained in theory we can lead to better depth estimation of early part of training)
the finitewidth correction is in Nr due to EOC for MMNNs !

we explain that the NTK randomness can lead to a lyapunov product expansion analysis and can explain a better conditionning than NN
or at least a faster early training curve, we have to keep in mind normalization data
we also show what's going on with the NTK std because disagreeing experiments with 1/r² instead of 1/r




also we can show a bit the analysis wrt non gaussian process propagation , the fact it differs from the NTK trying to fit a kernel gaussian process
there is no interpretation like that 


there should be a propagation of rho argument following, how the rho_i propagates due to the fisher law

we give in appendix computations for bias related gaussian NTK and standard deviations
we apply a theorem from 'spectrum of kernel random matrices' that explain all of our NTK results in theory,  there will be 2 randomness to disentangle so we would have a bound in 1/r then ..
we have to be careful because for the paper 'the spectrum of kernel random matrices' need a linear number of samples wrt dimension

and we compare with MLP !!! especially from Terjek paper

for the NTK there is a expansion in sum of C_i/r^i (1 to L) for L layers and C_i a random variable to be computed according to rho_i, from rho the cosine dist between x and y vector
finer analysis of the constant from lyapunov product analysis of those (1-arccos(rho_i)/pi) ; the pertubation in the kernel random matrix leads
to a bulk spectrum/outlier that differs from MLP

with this 1/r expansion maybe this is the premise to the finite width correction in infinite depth !! great idea to explore then

outlier analysis inherits from terjek analysis and its scaling then, there is a complete analysis to give on how the bulk will behave/outlier (ben arous discussion)
since what matters is outliers analysis we need to experiment on spherical 2d and 3d data (circle and sphere) because it avoids scaling issues
and the cosine distance can be very well understood


we can understood that the NTK decay

the code will be released in JAX first but then in torch (because of cuda management for my PC, maybe i'll see then)

the fact NTK converges to 1 in large low rank tells us that RF gives 
we can give results for high dim entries with d N r growing linearly from "the spectrum of kernel random matrices"
because we conjecture we need only a linear number of dictionary functions wrt input dim, in this regime we grow N d r at the same time !! (NTK analysis from marchenko pastur holds !!)
so that we avoid the complexity and get 3 layers analysis in this direction

the linear coefficient between d and N comes from discretization, N width grows and we have gram product with rho_1 as random variables ! conditioned on X

then we have a doubly random matrix, we have gram matrix appearing ! the spectrum is linearized wrt outputs and then we can apply marchenko pastur theory

weh sould see only 1 outlier that describe the early trianing part, as of terjek, and we recover it in a easier manner more tractable

i think the explanation of 1/r² decay comes from the 1/r randomness added from the gram output in NElkaroui paper !!

we can then explain the NTK bulk in a normalized * r² manner 


this is the most interesting regime







A very important aspect is that the curse of dim makes not to have unit norm w but not unit norm bias, it depends on if your data are normalized or in hypercube
if hypoercube then we need Tr(sigma_w) = d and unit bias to be sure approx theorem remains applicable
if spherical, unit norm weight and unit bias

This has an impact on the NTK std  ! ! and explain which NTK is useful for which case, in particular over the sphere Tr = 1 and we remove the 1/r coming from normalization


we're doing our full analysis on 2 mmnn layers because of untractable integral arising after the 2nd mmnn layer, we can tackle infinite layer NTK with the 1/r exponential expansion

since we only have 2 hidden layers, finite width corrections are tractable ! and we can try to have Nr and r/d scaling law









reminder : i should write the proof of kibble and fisher distrib !! 



we hope to explain this part with no finite width corrections, we will try to do a comparison between the minimum found by this and
what we get in practice

then we describe all our experiments showing this dictionary feature learning, behavior accross layers for several dimensional tasks


2) mean field analysis
this is a tremendous remark from meanfieldlandscape The answer to the last question is generally negative, and a physics analogy can explain why.
Think of $\btheta_1,\dots,\btheta_N$ as the positions of $N$ particles in a $D$-dimensional space. 
When $N$ is large, the behavior of such a `gas' of particles is effectively described by a density $\rho_t(\btheta)$ (with $t$ indexing time). However, not all `small' 
changes of this density profile can be realized in the actual physical dynamics:
the dynamics conserves mass locally because particles cannot move discontinuously.
For instance, if $\supp(\rho_t) = S_1\cup S_2$ for two disjoint compact sets $S_1,S_2\subseteq\reals^D$, and all $t\in [t_1,t_2]$,
then the total mass in each of these regions cannot change over time, i.e. $\rho_t(S_1) = 1-\rho_t(S_2)$ does not depend on $t\in [t_1,t_2]$. 



the bias is removed by adding 1 to data
i should see before what's happening betwen 1 layer RF and NN
i should make MF experiments to show distributions behavior
i should test with bounded activations 

they use gronwall insttead of https://arxiv.org/pdf/2504.13110 


from globalconvergences "Another
 corollary of Theorem 3 is that given the same family Init, the law of the MF trajectory is insensitive
 to the choice of the neuronal embedding of Init."  mail Huanh Min
 DM CHIZAT ET bach " Wenowgive a criteria for Wasserstein gradient flows to escape from non-optimal stationary points.
 It is valid both in the finite-particle regime and in the many-particle limit. Such a result supports
 the idea that, even in the finite-particle case (i.e. classical gradient flows), the point of view using
 measures is natural." from their paper

3) the 2nd part of training, end of lazy training
before knowing if I can try a mean field proof by getting the 

 
from "propagation of chaos"  " intuition that in many
 teacher-student settings with uniform initialization, the neurons are dispersed before converging to the
 teacher neurons."

we have to be careful if our function lives in the barron space or not, to invsstigate

4) about finite width corrections

we show then our results on finite width corrections scaling wrt depth, scaling law,
do comparison with what PMISOF get and try to show this can lead to great optimization bounds in the future
especially in RF low rank models where we can combine high dim stats and feynman to get scaling law for any depth
this can lead to a great future direction of research because of tractability for low rank RF models

also we have a NTK theory for 2 layers 




for mmnn we use the ntk to explain  that high frequencies are not learned directly through the RKHS becasue same as MLP , but there is more low to high frequency bias
we can tune the frequency learning in the variance of weights at init and maximal update param (mu param)


we can have a concentration bound between 2*NTK_MMNN-NTK_MLP for large r, ntk_mmnn std edcay in 1/r²
we can be in the NTK regime very easily 

there is still the lr decay anad batch size contribution to disentangle
at init there is a very bad conditionning so NTK explain partially the training


(it will remains finite width corrections to try to explain partly the mean field training)
mean field point of view can be satisfying and we have supportive results explains well the 2nd part of training where SGD have a much more convex training
(And now we have to fill with mean field previsions and reality)


the paper timeline is shaped in 3 part : 'The landscape of random features low rank neural networks'


1st part with NTK and finite width

2nd part with mean field and stepwise loss (benarous & mean field), we explain in which case sgd gets stucked but helps us where adam fails because of the wobbling part
we think that the wobbling  occurs in a subspace orthogonal to this of  the symmetry condition by the mean field approach,
Hessian and gauss newton is easy to calculate in this regime and supported by the plot through training of the TK/GN matrix
explained by central flow the high frequency grokking is explained by the last part of the functions high frequency learned by spikes

SGD is great at learning the final spikes after the global direction is given, ie the high spike given the high frequency variation given see "G:\Mon Drive\JANIS AIAD Internship - NTK for NN\mmnn_training_shifted\mmnn_training_shifted\L4_W512_R15_E30000_lr0.001_bs100_ratio5\L4_W512_R15_E30000_lr0.001_bs100_ratio5\th0.03lr_decay_steps1000gamma_20.99\mmnn_epoch10000_1D.png"
those steps can be explained in the teacher student perspective when seen in the fourier space
in fact  we believe it is only the 1st quadratic part of relu that works the most 

the approx interpretation with the dictionary we want to place somewhere ..

the mean field approach in fourier space can explain the same thing as ben arous ?, this is what we observe, big frequency training when the loss decay fast
and I think that the random features allow to grok that because it push dictionary functions at any place over the interval, it's just a matter of chosing which one
and having sufficient depth to build more and more frequencies

we only can track fourier training in bandlimited with barron space (same as haizhao)

the landscape is sharp, very sharp after having passed the low pass stuff

 "low-pass filter" limitation of shallow networks.
last part with convergence adn dynamical stability
 we still have the wavelet training to do
 Pinn fourier loss to try at the end to see if it mitigates (other section)

 this G:\Mon Drive\JANIS AIAD Internship - NTK for NN\mmnn_training_shifted\mmnn_training_shifted\L4_W512_R15_E30000_lr0.001_bs100_ratio5\L4_W512_R15_E30000_lr0.001_bs100_ratio5\th0.007lr_decay_steps1000gamma_20.99 explain a lot the low to high frequency bias

more depth = more frequency (make a plot depth against high fequency)
scatter loss vs depth/highest frequency


we observe also 2n spikes for 2n layers

we think growing r makes the ladnscape having exponentially more minimia
many dictionary functions learned are the same as PHI in mmnn construction , that select a range of interval, which is true
for what's happening for the 1st part/2nd part, we select this interval range and put a blob on this by thresholding randomly
then those blobs are assembled to form the high frequency function

this is a concurrent point of view from the approximation result but we decompose fèi psièi with this 2 interpretation


contrary to openai paper, shallow is low pass filter
in some numerical experiments adam makes us escape the global minma very fast 

fourier interpreatation in terms of multi index learning, in fourier spac 

we go beyong multi index models and in 1d we learn high frequency localzized part directly (instead of direction, a localization directly)

## fourier



## dictionary learning

- "we can clearly see in our experiments that MMNNs gives dictionary learning or even wavelet learning ! it performs learning of base functions to be streched and placed at a particular interval"

- "then we describe all our experiments showing this dictionary feature learning, behavior accross layers for several dimensional tasks"

- "the approx interpretation with the dictionary we want to place somewhere .."

- "many dictionary functions learned are the same as PHI in mmnn construction , that select a range of interval, which is true for what's happening for the 1st part/2nd part, we select this interval range and put a blob on this by thresholding randomly then those blobs are assembled to form the high frequency function"

- "this is a concurrent point of view from the approximation result but we decompose fèi psièi with this 2 interpretation"




# depth
- "more depth = more frequency (make a plot depth against high fequency)"
- "scatter loss vs depth/highest frequency"
- "with this 1/r expansion maybe this is the premise to the finite width correction in infinite depth !! great idea to explore then"
- "we're doing our full analysis on 2 mmnn layers because of untractable integral arising after the 2nd mmnn layer, we can tackle infinite layer NTK with the 1/r exponential expansion"
- "since we only have 2 hidden layers, finite width corrections are tractable ! and we can try to have Nr and r/d scaling law"
- "also we recall that we're doing this as taking the random feature space being high dimensional ! because of large width
and we train on that, and we do that both"
- "mainly my results lie in the mean field framework (between 500 and 5000 width), and I need to prove that it beats MLP for conference paper also by adding the benchmark
in comparison for 3 layers
so for the arxiv i before put all the math i've done to have a great math paper, for the conference i'll work on the benchmark, sincerely all the math i've done so far

i'll also give the insight for 3 layers by convergence theorem and universal approximation as explained in globalconvergencesthreelayers because this is what we see in the experimental results"

- "The landscape is sharp, very sharp after having passed the low pass stuff"

- "the paper timeline is shaped in 3 part : 'The landscape of random features low rank neural networks'" (context: different parts, including depth/width considerations)

- "for mmnn we use the ntk to explain  that high frequencies are not learned directly through the RKHS becasue same as MLP , but there is more low to high frequency bias
we can tune the frequency learning in the variance of weights at init and maximal update param (mu param)"

- "with this 1/r expansion maybe this is the premise to the finite width correction in infinite depth !! great idea to explore then"

- "reminder : i should write the proof of kibble and fisher distrib !!" (context: exploring topics as depth and width)

- "for the NTK there is a expansion in sum of C_i/r^i (1 to L) for L layers and C_i a random variable to be computed according to rho_i, from rho the cosine dist between x and y vector
finer analysis of the constant from lyapunov product analysis of those (1-arccos(rho_i)/pi) ; the pertubation in the kernel random matrix leads
to a bulk spectrum/outlier that differs from MLP"

- "the approx interpretation with the dictionary we want to place somewhere .." (context: perhaps related to depth vs width in network dictionary learning)

- "scatter loss vs depth/highest frequency"


todo  : 
2 layers analysis only 
littearature of random features, curse of dim for  approx theorem mean file 
recup anciens resultats probants
concentration of the low rank ntk
ntk outlier analysis (cf ben arous)
bias not removable for apporx results; 
finite width NTK
EOS, dynamical stability and 3rd order tensor

tester en 2d le curse od fim mean field

trouver un threshold qui fonctionne parfait et l'utiliser

low rank allow to have good minima exponentially in r, to test

depth wrt high frequency

passer en ntk et mu param

wavelet training ; signel multi index models
mean field do not explain that because of what it requires with
let high frequency thrive with standard init without preserving

randomness in the data just move the bulk but outlier emerges from

global convergence for all depth from mean field approach