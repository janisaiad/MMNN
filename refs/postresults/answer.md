



1)


Some of the figures could be cleaner and clearer. As an example of this, in Figure 2, it is not clear. Why not keep a two color code. Show the high rank curve in one color (e.g. blue) and the low rank curves in another (or shades of another) color. Also if you have a low rank and a high rank curve at the same momentum, wouldn't it make sense to only keep those two curves? Does it make sense to compare curves with different momentums ?


sure the plot is not well made








Section 2,
you should keep the same citation style throughout the paper.
Before line 126 you cite works by mentioning the authors and then providing the reference in the bibliography but then on line 130 (left column), you only provide the citation in the bibliography

OK


line 139, frozen random features ensure supp(W^0(C_1)) —> this is not clear*

the first layer output is f(w dot x) but if frozen random features we have universal approximation by integral representation of functions (or just linear approx) in barron space like integral which means dense space for continuous
functions


line 110, second column, what do you mean by the mean-field width limit ? Do you mean the limit of infinite width networks?

typo when writing : infinite width limit instead


Section 3.1.
I understand that the low rank factorization appears from the stacking of W_j^{(\ell)} and h^{(\ell-1)} but wouldn’t it be clearer to write x’ = \varphi(W^T A x + b) ?

it writes (\varphi(W^T (A x +c) + b) = \varphi(R x + d)  where d is shifted and live in low rank affine subspace for each layer, and R low rank)
we did not wanted to make this weird representation in affine subspace
and this way of doing emphasis the r-th channels (which means R partial functions) that do not appear in the compressed representation
those partial functions are the one on which we vizualize features


Section 3.2.
Line 166, your notations with the lower and uppercase case c_1 in W^0(c_1) is not really clear. Why put that C_i ? Why not just keep the notation of Eq. 1, W_0^{(i)} ?
Your expectation is taken with respect to the neurons indices ? which is strange to me ? You also say that in the infinite width limit, the neurons indices are continuous. Isn’t the norm to work with a measure on the weight vectors associated to each neuron ? I.e whose support would be non zero at the points W(C_i) ?

we emphasis that the framework we use is this of neuronal embedding, up to our knowledge this framework has never been used after the initial authors (HTPHAM) ; this framework allows to work over weights distributions that is not necessarily over real numbers. equations are not to be seen
in wasserstein space (even though it's equivalent) the neuronal embedding/ensemble framework 
in fact neuron indices are not continuous but in the mean field limit with a continuum of neurons we agree that it's not right to say 'continuum of neuron indices'



Section 3.3.

lines 187-188, I strongly recommend writing down the expression of the loss at least and recall that the system corresponds to a simple gradient descent on this loss? Because the way you introduce (2) is a little rough

we should write all the expressions for B, losses etc .. at least in appendix (we did it to fit the 8 pages constraint) and we promise a better formulation to come for the camera ready version, we should make it to the appendix and make a link


lines 187-188: What are the learning rate schedules?

this is your learning rate function (say exp(-t)) coming from the ODE formulation, it can be stepwise, cosine schedules
from the discretization of the ODE you perform x_n+1=x_n-ksi(epsilon*n), where epsilon goes to 0 as long as you discretize more the ode (smaller learning rate)




Section 3.4.

On lines 166 - 168, you say that the “Better than null” assumption requires the initial loss to be better than the null function.. What does that mean ?

L(0) bigger than L(f_init) which means we perform better at initialization than the function 0


Line 178, second column, you say that for Relu, the same holds with high probability in r. I would recall the meaning of r here
I don’t know how easily this can be done but I would recommend laying out the assumptions cleary on lines 215 - 216. Some of them are not very clear. E.g. what do you mean by bounded activation and MIXING (i.e the mixing part)?

with high probability in r the output of relu(Wx+b) is strictly positive (in fact I have a confusion to disantengle, i'm not sure of that, in the paper from nguyen they use the non zero assumption at every x to show that A*B=0 with A non zero leads to B = 0, this is very important because if for some points it's not the case lets say with probability p, then we have an integral that is not 0 but if p small the integral is small, which means gradients are small) but for a fixed x with high probability it has a non zero value, so that the set of x where a neuron value 0 is small at initialization
in practice we observe that it is the case , using a function like GeLu allows to get the theoretical ground but ReLU not (it would need an over complication for something that is not relevant in practice)
Goal is to show that intuitively low rank remove the neural collapse, neural collapse is not due to relu but due to no representations being learned
by a continuumm of neurons that act in the same optimal way ; that is why removing relu do not exert our understanding, and we leave the relu proof for future work

proving that an output node remains non zero everywhere requires a propagation of non zero values which is pretty hard  ; bounded mixing means matrix W is bounded (which means its distributions has finite support), which is supported in practice by default torch initialization for instance ; in fact
this is not the case for gaussians but the mean field regime occurs in practice for N = 1000 (I still need to disentangle if the mean field regime is dimension dependent or not) for which gaussians less than 10





Section 3.5 - 3.6

lines 207 - 210, second column, you have |H_\ell(W’) - H_\ell(W’’)|< K||W^{\ell}||_{\infty, 1}. What is W^{\ell} in this setting? Does that mean all the weights from layer ell ? but then how does this relate to W’ and W’’? That notation does not make sense to me.

Hl takes weights and output a real number, this is effectively a sum of relu that is lipschitz, we simplified the proof 
bi lipschitzness is trivial by aggregating lipness for each previous channels ; in this setup we have missed to write H_l 
the constant K is layer dependant by a multiplication, fixing L it's obvious that all the network is lipschitz but the constant matters
here the constant W_l is here because after taking W (as random or fixed arbitrary by someone) the problem is optimization over A, we denote W
as A in the appendix and confused the reader




lines 218 - 219, what is the solution operator F?

this is desribe in appendix, it solved the ode and maps A to the integral, we framed it as "solutioon operator" but it is it's iteration limit that forms it, F^n converge to solution operator by picard argument 


lines 220 - 223: “After defining norms, the solution operator F,… and the contraction operator …” —> “After defining norms, F and bounds, and the contraction, the argument proceeds… ” (I think you want to add a coma here)

I think so, typo




Section 3.7

In the statement of Theorem 3.2., when you talk about the mean field limit, you are referring to n_1, n_2 —> infinity right? A depth 2 network as in Theorem 3.3. Or does it hold for an arbitrary number of layers?

this holds for an arbitrary number of layers ! from nguyen & pham only 3 layers works and 4 layaers lead to neural collapse ; here any layer leads to no neural collapse






lines 269 -, What you call “High level proof idea” is pretty opaque. I think I would remove it. Or I would remove lines 227 - 243, second column and expand a little bit on the proof.

okay i wanted to make it clear but we can remove and write it in appendix, or make it shorter and expand the notational part said before to write loss and all that matters we missed 


lines 272 - 273, “this yields \mathbb{E}_Z[upstream\times local]” —> what does that mean? I understand you are referring to the expressions in (2) but what do the terms upstream and local refer to?

no sorry it's forward * backward ; forward is local, backward is upstream, it's an artifact from a copy paste


lines 220 - 223: What does \partial_2 \mathcal{L} mean here? Does that mean the partial derivative with respect to the second argument of \mathcal{L}?

sorry this is the derivative wrt the 2nd argument


lines 239 - 241, second column, it is not clear to me why the frozen random features yield heterogeneous shifts across neurons. In fact, even before this, what do you mean by “shifts” in this setting?

neuronal collapse tells that each neurons shift the same quantity in the classical mean field analysis of 4 layers networks (proved in nguyen)
here we prove they don't shift the same value due to the mixing matrix, this means they don't learn the same quantity and we avoid neural collapse


Section 3.8

In the statement of Theorem 3.3. Here as well, it is not clear why/how you initialize over the indices.. you have a tuple {n1, n2} that is in the index set of Init ? What does that mean concretely ? Why just a two tuple? What do n1 and n2 represent ? do they refer to the number of neurons in the first and second layers ? Does the result only hold for depth 2 then?


okay we should write it all better from the appendix detailing the result, we did not explained Init from neuronal embedding framework (nguyen)
that led to unclear theorem formulation when Init not stated before


Same Theorem, what do you mean by the “coupling procedure”?

artifact from copy past (i think this has been an llm hallucination) this is not a coupling procedure but just a mixing INIT


Same, What does  D
 refer to here ? If I’m not wrong, you don’t introduce it. I understand it measures the deviation between the mean field solution and the finite solution but what measure do you use?

 D is the wasserstein distance (sorry we missed to write it before), it's defined in nguyen


















2)


The theoretical novelty is somewhat incremental, primarily extending the framework of Nguyen & Pham (2023) by relaxing their initialization to a "frozen initial weights with low-rank features" setting.

The global convergence guarantee relies critically on freezing the mixing matrices 
 as random features. This architectural restriction significantly limits the theory's applicability to modern, fully end-to-end trained deep neural networks.


Yes this is not a restrictive part but an enriching part, this is what we try to sell

Theorem 4.1 provides an elegant explanation for feature learning, but it is rigorously proved only for a highly simplified two-point, two-channel toy model. Scaling this claim to continuous, high-dimensional real-world data remains largely empirical.


(I'll try to do my best for this part)


More critically, the core assumption maintaining the network's symmetric structure is overly restrictive and sensitive to hyperparameters; in my extended reproduction (using a 1D function with frequencies 
 and  , depth  , and width  ), the symmetry was largely imperceptible at rank  and only clear at rank  , suggesting that this strict low-rank requirement significantly limits the theory's robustness and applicability to broader network scales.

we are not sure on how symmetry is maintained 




questions : While symmetry is preserved at rank 10 , it becomes largely imperceptible at rank 20 . Since r=20 remains an extremely low-rank constraint relative to a width of  1024, this sensitivity challenges the practical robustness of the feature learning mechanism .


we are running symmetry experiments to see how it behaves (construct a table where we train and measure symmetry defects)


Theorem 3.3 bounds the finite-width approximation error by 
. Does the symmetry loss at  occur because the exponential factor  dominates  , causing divergence from idealized mean-field dynamics? If so, is "symmet preservation" a genuine benefit of low-rank architectures, or merely a fragile artifact of keeping  artificially small to prevent mean-field breakdown?

No, bound just comes from gronwal lemmas and ensure mathematical convergence but in practice symmetry is learned in parallel, it's just a worst case
analysis from gronwall

If  rmust be kept strictly minimal to preserve mean-field behaviors, how does this framework scale to datasets with higher intrinsic dimensionality 
? If evaluated on a synthetic dataset with controlled 
, is there a theoretical or empirical phase transition when 
? It would be insightful to clarify if the network loses its theoretical structural benefits once the required rank 
 exceeds what the practical width 
 can exponentially suppress


(need to run experiments on d), several recent papers suggests that r = sqrt(M) is ideal (chizat, LoRA jason lee)
we think so also, (try to run experiments)


We want to show that everyone went wrong training full rank matrices, and want to apply it fo rhighly oscillating functions in sciML and PDE learning
that is why the applicative field is not current llms etc .. but is a way to bring a theoretically grounded path towards low rank so that people using that
after are not afraid : no, it does not constrain global convergence !





3) The low-rank model this paper consider have a constant rank w.r.t to the width that tends to infinity, and only one matrix of the desomposed weight is trained, this seems to be far from practical settings. In particular, if we write the network architecture ( equation (1)) in matrix form, the efficient weight for each layer  is  where

same answer



Technically speaking, the proof of the Theorem 3.1-3.3 follows exactly the same route as in (Nguyen & Pham, 2023), and does not provide any new insight in my opinion. Could the authors elaborate more on the technical difference compare to (Nguyen & Pham, 2023)?

we explain the main difficulty (and how low rank resolve it), it's done in a companion Goodnotes, the technical difficulty is just at a particular point of the proof where low rank lipschitzness resolves everything

we are stacking 2 layers networks and merging them, but after each neuron (1024) get a different weighted contribution, those weights are not
trained which means there is no bias or collapse towards selecting a particular function (setting weights to 1 and other to 0)
we are currently working on deciphering the low rank collapse without random features, so that the architecture looks easier technically
but this random contribution make the final result highly not similar ; for more layers we do not have neural collapse but a rich feature learning
mechanism that we can analyze
concerning feature learning in mean field parameterized NN (i need to decipher SOTA) we do better because we interpret the output as a partial function
it's technically easier, easier to analyze features, mathematically the same difficulty (no simplification allowed in any part of the proof)


While I acknowledge the spectral bias for low-rank network could be interesting, the discussions in Section 4.1 seems to be too qualitative, and the results in Section 4.2 seems to be on too simple setting (only two point in the datasets). Could the authors elaborate more on why the low-rank network lead to high-frequency bias, and fully trained network lead to low-frequency bias?

it is effectively qualitative (i need to make it better and understand more this feature learning mechanism and separation)
(i need also to make better log ratio trajectories, disentangle the 0 contribution because everything mess up when sign changes which leads to big spikes)
(does it converge to 0 or it traverse towards negatives ?)
the low rank network high frequency bias is an unsolved problem, in other submission (ICLR) we show experimentally that low rank networks
lean high frequency features at each plateau landscape
low frequency bias for full rank NN is partly explained by the NTK, it's not that they don't fit the same function, it's that the loss landscape
is not guided towards high frequency compared to low rank experimentally
this is work in progress (i can elaborate a lot on that)






As I mentioned in the Strengths and Weaknesses part, the architecture considered in this paper is a stacking of two-layer network, I don't see the difficulty of proving the mean-field limit when both 
 are trained. Could the authors elaborate more on the technical difficulty

 (SAME EXPLANATION GOODNOTES)
 

I wonder how universal is spectral bias discovered in Section 4 for low-rank training? For example, if we do not consider network in the mean-field regime, do we still have such spectral bias?

we have mainly ntk, mu-p, meanfield regimes ; under the NTK regime the spectral bias is the same (ntk is the same, COLT PAPER)
for mu-p we still don't know, this is work in progress ; we point towards this experiments 






3)


Lack of additional experimental results on the performance of low-rank random feature neural networks. If they avoid neural collapse in the mean-field regime, we would expect these architectures to outperform the multilayer neural networks when gradient-based methods are initialized i.i.d..


We are running experiments on CIFAR (and many others ? like central flow ?)


There is no convergence guarantee provided for the learning dynamics of low-rank random feature networks under i.i.d. initialization (also see the question below). 


Derivations heavily rely on a previous work of Nguyen & Pham, 2023. It is very unclear what is actually new in this paper.




we add low rank and random features, derivations nguyen paper throw the basis for the network 
in fact in our workflow we ran a lot of experiments (this comment is valable for any review) and wanted to see if theoretically it was sounded
the answer is yes, but to prove it you have to rely a lot on former framework, up to our knowledge this is the 1st time this framwork used in another context than the author, after explaining all what's needed to understand the nguyen paper, we write the whole proof, in fact the only thing that matters
is just at a moment, every other stuff are somewhat trivial understanding TO BE SURE that everything is well defined mathematically (unless you prove convergence of an object that do not exist)
the only thing that matters the most is the part (GOODNOTES)




I would strongly recommend mentioning the Assumptions in the main paper thoroughly with discussion about them, rather than hiding them in the Appendix since they are quite important for the analysis.

yes





Typo at (040) "...is low rank all we need for global convergence ? Low-rank networks...": There is an extra space between the question mark and the first word of the second sentence. Typo at (042): "...with 
. substantially...".

At (044), it is posed that "can gradient-based training still converge to a global minimizer, or is full rank essential?". I didn't understand why improvement in the training performance, as shown in Fig. 
, implies that the low-rank structure affects the dynamics of the gradient-based optimization algorithms such that they converge to a global minimizer. First, the low-rank factorization might change the loss landscape completely if some global minimizers are not feasible for the specific factorization determined by 
. Note that the low-rank problem is different than the full-rank problem. Therefore, the functions represented by the global minimizers of the full-rank problem can be different than the low-rank problem.


the loss landscape problem is the one we are trying to solve, in every regime (ntk, rich regime, mean field etc ..) we try to understand
how low rank local and global analysis behave to explain our experimental results (i can elaborate a lot on that, it's up to how much i can say)
related to ICLR paper





For the Fig. 
, how are different momentum rates selected for the corresponding level of rank constraints?

we ran sweep (run other experiments)




Do low-rank random feature networks differ from the multi-component multilayer neural networks proposed by Zhang et al. (2023) only in their low-rank structure?

Typo at (137), 
 should be denoted by vector notation.



Could you clarify the convergence assumptions made by Nguyen & Pham (2023) for multilayer and two layer neural networks?

convergence assumption assume that every weights stop moving, which means training is ended
we follow the francis bach paper that threw basis https://francisbach.com/gradient-descent-for-wide-two-layer-neural-networks-implicit-bias/


Nguyen & Pham, 2023 established the global convergence of multilayer neural networks trained under stochastic gradient descent (SGD). Is your convergence analysis also based on SGD

yes, in the mean field sgd converges well provided that discretization is small enough
it's also in the bach wor





5)

. Convergence guarentees. Are there conditions under which convergence is guaranteed for RF-LR models (those mentioned in section 3.5 or otherwise)?

we still do not have convergence guarentees, we are trying to decipher that with bach works


2. High-dimension extensions. Does the spike learning mechanism extend to higher dimensions with overlapping spikes in feature space?

we still don't know (run experiments), 

3. Sensititivity of randomness of fixed feature maps. How sensitive are the convergence guarantees and empirical performance to the initial draw of the frozen feature maps for finite width networks?

we are running variance mean experiements








TODO : 
what to do : extend feature learning theorem, understand more to explain better the shift in the neural collapse originally, trying to prove this story of relu high probability ; need to run experiments on higher d and r to see a phase transition maybe sqrt, explain the goodnotes by hand, see the collapse without frozen weights but keeping low rank, deciphering feature learning for mean field models (mu-p), elaborate a lot on low/high frequency bias for low rank/full rank, run a lot of other experiments showing low rank do better, run other sweep for momentum, see convergence guaranteees with former bach work, trying to run variance mean sweep experiements for the random features, run experiements in high dimension with overlapping spikes, we can take N=10 maybe