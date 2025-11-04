



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



 from meanfieldlandscape2layers

 
$(ii)$~Multiple finite-$N$ local minima can correspond to the same  minimizer $\rho_*$ of $R(\rho)$ in the limit
$N\to\infty$. Ideas from glass theory \cite{mezard1999thermodynamics} might be useful to investigate this structure.
