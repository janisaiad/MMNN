



1)


Some of the figures could be cleaner and clearer. As an example of this, in Figure 2, it is not clear. Why not keep a two color code. Show the high rank curve in one color (e.g. blue) and the low rank curves in another (or shades of another) color. Also if you have a low rank and a high rank curve at the same momentum, wouldn't it make sense to only keep those two curves? Does it make sense to compare curves with different momentums ?


sure the plot is not well made and we  have runned more sweep experiments to make it better
(experiements are running)







Q
Section 2,
you should keep the same citation style throughout the paper.
Before line 126 you cite works by mentioning the authors and then providing the reference in the bibliography but then on line 130 (left column), you only provide the citation in the bibliography

A
OK

Q
line 139, frozen random features ensure supp(W^0(C_1)) —> this is not clear*
A
the first layer output is f(w dot x) but if frozen random features, we have universal approximation by integral representation of functions (or just linear approx) in barron space like integral which means dense space for continuous functions, 
in fact we should clarify assumption C3 : for other layers, it is not
required that the support is dense, because in fact we require bounded mixing matrices which would
counteract the fact that we have a dense support for the first layer.

So assumption C-C43 should now read : for l bigger than 1 then bounded W
for l = 0 full R^d support for W



Q
line 110, second column, what do you mean by the mean-field width limit ? Do you mean the limit of infinite width networks?
A
typo when writing : infinite width limit instead



Q
Section 3.1.
I understand that the low rank factorization appears from the stacking of W_j^{(\ell)} and h^{(\ell-1)} but wouldn’t it be clearer to write x’ = \varphi(W^T A x + b) ?
A
it writes (\varphi(W^T (A x +c) + b) = \varphi(R x + d)  where d is shifted and live in low rank affine subspace for each layer, and R low rank)
we did not wanted to make this weird representation in affine subspace
and this way of doing emphasis the r-th channels (which means R partial functions) that do not appear in the compressed representation
those partial functions are the one on which we vizualize features



Q
Section 3.2.
Line 166, your notations with the lower and uppercase case c_1 in W^0(c_1) is not really clear. Why put that C_i ? Why not just keep the notation of Eq. 1, W_0^{(i)} ?
Your expectation is taken with respect to the neurons indices ? which is strange to me ? You also say that in the infinite width limit, the neurons indices are continuous. Isn’t the norm to work with a measure on the weight vectors associated to each neuron ? I.e whose support would be non zero at the points W(C_i) ?

A
we emphasis that the framework we use is this of neuronal embedding, up to our knowledge this framework has never been used after the initial authors publication (HTPHAM) ; this framework allows to work over weights distributions that is not necessarily over real numbers. equations are not to be seen
in wasserstein space (even though it's equivalent) the neuronal embedding/ensemble framework 
in fact neuron indices are not continuous but in the mean field limit with a continuum of neurons we agree that it's not right to say 'continuum of neuron indices'



q
Section 3.3.
lines 187-188, I strongly recommend writing down the expression of the loss at least and recall that the system corresponds to a simple gradient descent on this loss? Because the way you introduce (2) is a little rough

a
we should write all the expressions for B, losses etc .. at least in appendix (we did it to fit the 8 pages constraint) and we promise a better formulation to come for the camera ready version, we should make it to the appendix and make a link


q
Lines 187-188: What are the learning rate schedules?
a
this is your learning rate function (let's say exp(-t)) coming from the ODE formulation, it can be stepwise, cosine schedules
from the discretization of the ODE you perform x_n+1=x_n-ksi(epsilon*n), where epsilon goes to 0 as long as you discretize more the ode (smaller learning rate)

in the mean field parameterization, lr = cst/n ; this is exactly how this cst = ksi(t) for n data


Q
Section 3.4.
On lines 166 - 168, you say that the “Better than null” assumption requires the initial loss to be better than the null function.. What does that mean ?

A
L(0) bigger than L(f_init) which means we perform better at initialization than the function 0
Line 178, second column, you say that for Relu, the same holds with high probability in r. I would recall the meaning of r here

Q
I don’t know how easily this can be done but I would recommend laying out the assumptions cleary on lines 215 - 216. Some of them are not very clear. E.g. what do you mean by bounded activation and MIXING (i.e the mixing part)?


A
we promise a better and clearer assumption formulation for the main body to be self contained

regarding the high probabiilty explanation : 
the idea we wanted to transmit  : to have φ′ 2(H2(c2;x, ¯W)) = 0, you can choose any mixing matrix
so that if they are all positive, at initialization the 
if W is drawn under a positive law, and A drawn under symmetric law then the set on which you have negative output in the r channel is 1/2 exponentially small in r which means relu is 0 on a exponentially small set

You can restrict yourself W to be of the same sign, as long as W as changing signs
the low rank factorization will stil be preserved, one way to show it does not change is doing a change of basis
with positive vectors, as long as r is less than W (you can form a basis with W-1 positive vectors)

; that is why removing relu do not exert our understanding, and we leave the relu high probability proof for future work



---

### ReLU vs.\ Assumption C.1 (nonzero $\varphi_2'$) — draft for rebuttal / camera-ready

**Assumption C.1 (regularity)** in the appendix requires $\varphi'_2$ to be **bounded away from zero**, which **strictly excludes plain ReLU**. In the global convergence proof (Appendix F), this is used to show that a vanishing conditional moment $\mathbb{E}[d_L \cdot \varphi'_2(H_2) \mid X=x] = 0$ implies a vanishing residual $\mathbb{E}[d_L \mid X=x] = 0$. If $\varphi'_2$ can be zero while $d_L \neq 0$, one has a **“dead gradient”** at a non-optimal point.

**ReLU dead zone (ensemble).** If $H_2(c_2; x, \bar W) \le 0$ for all neurons $c_2$ in the support, gates vanish. In the **low-rank** architecture, $H_2 = \sum_{k=1}^r L_{c_2,k} f_k(x)$ is a sum of $r$ channel contributions. If $L$ is drawn from a **symmetric** distribution (e.g.\ Gaussian or Xavier), then at a fixed $x$ where the representation is non-degenerate, the probability that **one** neuron’s gate $\varphi'(H_2)$ is zero is often **$\approx 1/2$** (half-space). The probability that **all** $r$ channels conspire so that **no** gradient propagates through the gate is **exponentially small in $r$** (informally $\sim 2^{-r}$; manuscript text can say $e^{-O(r)}$).

**Conceptual link to collapse.** Neural collapse is a failure of **representation diversity** (Corollary 30, Nguyen \& Pham 2023), not “because of ReLU.” Frozen random mixing gives **heterogeneous** modulation of shifts across neurons. **Formal ReLU-only** closure: future work; **GeLU** (or other smooth $\varphi_2$) satisfies the theory directly and is empirically close.

**Line 178 — draft fix (recall $r$ + high probability).**  
> For ReLU, the same global convergence property holds **with high probability in the rank $r$** (the number of independent low-rank channels), as the probability of a **total** gradient vanish at a non-optimal stationary point (all gates dead simultaneously) decays as **$e^{-O(r)}$.**

**Lines 215–216 — clarify “mixing” (draft Assumption C.1 text).**  
Replace a vague “bounded activation and mixing” line with something like:

> **Assumption C.1 (bounded activation and mixing matrix).** The activation $\varphi$ is Lipschitz. The **mixing matrix** $L^{(\ell)}$ is the **frozen** low-rank factor that recombines channel features into neuronal pre-activations ($H_\ell = \sum_k L_{\cdot,k}^{(\ell)} f_k$). Entries satisfy $|L_{c,k}|\le K$, hence $\|L\|_{\infty,1}\le rK$. For the global-convergence appendix we also assume $\varphi_2'$ bounded away from zero (excluding plain ReLU there); GeLU satisfies this.

**Change-of-basis / sign structure (optional response).** If one needs $L$ with **prescribed signs** (e.g.\ to steer away from the ReLU dead zone), one can appeal to **nonnegative / structured factorizations** of rank-$r$ weights (NMF viewpoint) or **Perron–Frobenius**-type positivity on a sub-basis when $r$ is small—**without loss of expressivity in the rank-$r$ class** at fixed $r$. This is design advice, not a proved reduction in the mean-field draft.




"
\paragraph*{Degeneracy of the dynamics.}

By looking closely at $W^{*}$, we observe a simplifying property.
By Theorem \ref{thm:iid dynamics}, under i.i.d. initialization, for
each intermediate layer $i=3,...,L-2$, the weight $w_{i}^{\infty}\left(t,C_{i-1},C_{i}\right)$
is a function of only the time $t$, its own initialization $w_{i}^{0}\left(C_{i-1},C_{i}\right)$
and the initializations of the adjacent biases $b_{i-1}^{0}\left(C_{i-1}\right)$
and $b_{i}^{0}\left(C_{i}\right)$, and the bias $b_{i}^{\infty}\left(t,C_{i}\right)$
is a function of only the time $t$ and its own initialization $b_{i}^{0}\left(C_{i}\right)$.
When we further assume constant initial biases (i.e. $b_{i}^{0}\left(C_{i}\right)=B_{i}$
a constant almost surely for all $i\geq2$), $w_{i}^{\infty}\left(t,C_{i-1},C_{i}\right)$
is a function of only the time $t$ and its own initialization, and
$b_{i}^{\infty}\left(t,C_{i}\right)$ is almost surely only a function
of time $t$, regardless of $C_{i}$. Consequently, in this scenario,
because the initialization is independent across layers, the weights
of intermediate layers remain mutually independent at all time, for
depth $L\geq5$, in the infinite-width limit.

The theorem in fact further asserts that degeneracy can already be
observed for $L\geq4$. In particular, for $2\leq i\leq L-2$, if
the initial bias $b_{i}^{0}\left(\cdot\right)=B_{i}$ is a constant,
then
\[
\mathbb{E}\left[\left|H_{i}\left(X,C_{i};W^{M}(t)\right)-H_{i}^{*}\left(t,X,B_{i}\right)\right|^{2}\right]^{1/2}\leq\frac{K_{T,L}}{M^{0.499}}.
\]
Note that $H_{i}^{*}\left(t,X,B_{i}\right)$ is independent of $C_{i}$.
This suggests that at any training time $t$, the neurons of each
intermediate layer $i$ compute the same function of the data input
$x\mapsto H_{i}^{*}\left(t,x,B_{i}\right)$ in the infinite-width
limit. This is formalized directly for the neural network $\mathbf{W}$
in the following.
\begin{cor}
\label{cor:iid_same_neurons}Consider the same setting as Corollary
\ref{cor:iid_tracking} with $L\geq4$. For $2\leq i\leq L-2$, supposing
that $b_{i}^{0}\left(C_{i}\right)=B_{i}$ a constant almost surely,
then we have for any $t\leq T$, with probability at least $1-3\delta-KLn_{\max}\exp\left(-Kn_{\min}^{c_{2}}\right)$,
\[
\bigg(\frac{1}{n_{i}}\sum_{j_{i}=1}^{n_{i}}\mathbb{E}_{Z}\left[\left|{\bf H}_{i}\left(\left\lfloor t/\epsilon\right\rfloor ,X,j_{i}\right)-H_{i}^{*}\left(t,X,B_{i}\right)\right|^{2}\right]\bigg)^{1/2}=\tilde{O}\left(n_{\min}^{-c_{1}}+\epsilon^{c_{1}}\right).
\]
\end{cor}

Thus, by Markov's inequality, if one is to pick at random a neuron
$j_{i}\in\left[n_{i}\right]$ at layer $i$ from the neural network
$\mathbf{W}$ at the training step $\left\lfloor t/\epsilon\right\rfloor $,
for $2\leq i\leq L-2$, then with high probability, this neuron would
compute the function $x\mapsto H_{i}^{*}\left(t,x,B_{i}\right)$ which
is independent of the index $j_{i}$.


\paragraph*{Collapse to effectively one parameter per layer.}

Further consideration to standard neural network architectures reveals
a stronger simplifying property. The next consequence of Theorem \ref{thm:iid dynamics}
is that with i.i.d. initialization and constant initial biases, for
each intermediate layer $i=3,...,L-2$, the weight $w_{i}^{\infty}\left(t,c_{i-1},c_{i}\right)$
translates by a quantity that is independent of $c_{i-1}$ and $c_{i}$,
provided that $\sigma_{i}^{\mathbf{w}}$ satisfies a certain condition.
This condition holds for unregularized standard fully-connected or
convolutional neural networks (see Examples \ref{exa:fully-connected}
and \ref{exa:conv}). Therefore, for these networks, in the infinite-width
limit, with i.i.d. initialization and constant initial biases, the
dynamics of the weight at each intermediate layer reduces to a single
deterministic translation parameter.
\begin{cor}
\label{cor:iid_standard-network}Under the same setting as Theorem
\ref{thm:iid dynamics} with $L\geq5$, assume that $b_{i}^{0}\left(C_{i}\right)=B_{i}$
a constant almost surely for all $i\geq2$. Further assume that for
each $i\in\left\{ 3,...,L-2\right\} $, there exists a function $\bar{\sigma}_{i}^{\mathbf{w}}$
that satisfies 
\[
\sigma_{i}^{\mathbf{w}}\left(\Delta,w,b,g,h\right)=\bar{\sigma}_{i}^{\mathbf{w}}\left(\Delta,b,g,h\right),
\]
i.e. $\sigma_{i}^{\mathbf{w}}$ does not depend on the second variable.
Then there are differentiable functions $w_{i}^{\#}\left(t\right)$
such that for $3\le i\le L-2$, almost surely, for any $t\geq0$,
\[
w_{i}^{\infty}\left(t,C_{i-1},C_{i}\right)-w_{i}^{\infty}\left(0,C_{i-1},C_{i}\right)=w_{i}^{\#}\left(t\right).
\]
\end{cor}
" from nguyen (just exract the necessary part from that? this explanation is called )

in our case (it is not proved for the sake of concicenes) each neurons has different backward
signal due to the mixing matrix, this means they don't learn the same quantity

we have not proved the theorem or a NON-collapse bound
in fact each neuron do not receive the same backward signal and hence A paramaters do not receive the same

proving that an output node remains non zero everywhere requires a propagation of non zero values which is pretty hard  ; bounded mixing means matrix W is bounded (which means its distributions has finite support), which is supported in practice by default torch initialization for instance ; in fact
this is not the case for gaussians but the mean field regime occurs in practice for N = 1000 (I still need to disentangle if the mean field regime is dimension dependent or not) for which gaussians less than 10

"\item \textbf{Low-rank mixing matrix}: The mixing matrix entries $L_{c_2,k}$ are random variables (e.g., Uniform) with $\sup_{c_2,k}|L_{c_2,k}|\le K$ almost surely, and $k\mapsto L_{c_2,k}$ is measurable for each $c_2$. This implies $\|L\|_{\infty,1}\equiv \sup_{c_2}\sum_{k=1}^r |L_{c_2,k}|\le rK$ almost surely."

(end of the answer)
(THIS EXPLANATION IS CALLED NEURONALCOLLAPSELINK AND COULD BE USED FOR OTHER ANSWERS)




Q

Section 3.5 - 3.6

lines 207 - 210, second column, you have |H_\ell(W’) - H_\ell(W’’)|< K||W^{\ell}||_{\infty, 1}. What is W^{\ell} in this setting? Does that mean all the weights from layer ell ? but then how does this relate to W’ and W’’? That notation does not make sense to me.

A
Hl takes weights and output a real number, this is effectively a sum of relu that is lipschitz, we simplified the proof 
bi lipschitzness is trivial by aggregating lipness for each previous channels ; in this setup we have missed to write H_l 
the constant K is layer dependant by a multiplication, fixing L it's obvious that all the network is lipschitz but the constant matters
here the constant W_l is here because after taking W (as random or fixed arbitrary by someone) the problem is optimization over A, we denote W
as A in the appendix and confused the reader



Q

lines 218 - 219, what is the solution operator F?
A
this is desribe in appendix, it solved the ode and maps A to the integral, we framed it as "solutioon operator" but it is it's iteration limit that forms it, F^n converge to solution operator by picard argument 

Q
lines 220 - 223: “After defining norms, the solution operator F,… and the contraction operator …” —> “After defining norms, F and bounds, and the contraction, the argument proceeds… ” (I think you want to add a coma here)

A
I think so, typo



Q
Section 3.7

In the statement of Theorem 3.2., when you talk about the mean field limit, you are referring to n_1, n_2 —> infinity right? A depth 2 network as in Theorem 3.3. Or does it hold for an arbitrary number of layers?

A
this holds for an arbitrary number of layers ! from nguyen & pham only 3 layers works and 4 layaers lead to neural collapse ; here any layer leads to no neural collapse
for the sake of conciceness we did not write the theorem for arbitrary number of layers
looking at the proof the thing making everything working is line 1452
Step6: Gronwall argument. Combining the bounds and taking a union bound over a discrete time grid t ∈
{0,ξ,2ξ,...,⌊T/ξ⌋ξ}forsomeξ∈(0,1),weobtain:
max max
j2≤n2
∂
∂t
˜A2(t,j2)− ∂
∂tA2(t,C2(j2)) ,
max
j1≤n1,1≤k≤r
∂
∂t
˜A1(t,j1,k)− ∂
∂tA1(t,C1(j1),k)
≤KT(1+rK)(Dt(W, ˜W)+γ1+γ2+ξ),

for any layer this translates to lipschitz constants rK multiplied fo reach layer (rk)^L-2
but qualitatively the same reasoning holds by iterating 1331 and 1344



Q
lines 269 -, What you call “High level proof idea” is pretty opaque. I think I would remove it. Or I would remove lines 227 - 243, second column and expand a little bit on the proof.
A
okay i wanted to make it clear but we can remove and write it in appendix, or make it shorter and expand the notational part said before to write loss and all that matters we missed 

Q
lines 272 - 273, “this yields \mathbb{E}_Z[upstream\times local]” —> what does that mean? I understand you are referring to the expressions in (2) but what do the terms upstream and local refer to?
A
no sorry it's forward * backward ; forward is local, backward is upstream, it's an artifact from a copy paste

Q
lines 220 - 223: What does \partial_2 \mathcal{L} mean here? Does that mean the partial derivative with respect to the second argument of \mathcal{L}?
A
sorry this is the derivative wrt the 2nd argument

Q
lines 239 - 241, second column, it is not clear to me why the frozen random features yield heterogeneous shifts across neurons. In fact, even before this, what do you mean by “shifts” in this setting?

A
neuronal collapse tells that each neurons shift the same quantity in the classical mean field analysis of 4 layers networks (proved in nguyen)
here we prove they don't shift the same value due to the mixing matrix, this means they don't learn the same quantity and we avoid neural collapse
(there we can expand a lot from the neural collapse what i already explained in a part i forgot)
(YOU CAN PUT NEURONALCOLLAPSELINK)


Q
Section 3.8

In the statement of Theorem 3.3. Here as well, it is not clear why/how you initialize over the indices.. you have a tuple {n1, n2} that is in the index set of Init ? What does that mean concretely ? Why just a two tuple? What do n1 and n2 represent ? do they refer to the number of neurons in the first and second layers ? Does the result only hold for depth 2 then?

A
this is a mistake we've done by not clarifying this Init comes from Nguyen paper we should write it all better from the appendix detailing the result, we did not explained Init from neuronal embedding framework (nguyen)
that led to unclear theorem formulation when Init not stated before
the proof is done for 2 layers but is easilyt extendable to any layer (see other answer when i talk about extending for more layers)


Q
Same Theorem, what do you mean by the “coupling procedure”?
A
artifact from copy past (i think this has been an llm hallucination) this is not a coupling procedure but just a mixing INIT

Q
Same, What does  D
 refer to here ? If I’m not wrong, you don’t introduce it. I understand it measures the deviation between the mean field solution and the finite solution but what measure do you use?

A
D is the wasserstein distance (sorry we missed to write it before), it's defined in nguyen


















2)

Q
The theoretical novelty is somewhat incremental, primarily extending the framework of Nguyen & Pham (2023) by relaxing their initialization to a "frozen initial weights with low-rank features" setting.

The global convergence guarantee relies critically on freezing the mixing matrices 
 as random features. This architectural restriction significantly limits the theory's applicability to modern, fully end-to-end trained deep neural networks.


Yes this is not a restrictive part but an enriching part, this is what we try to sell (take some other parts about when we talk about sciml, )

Theorem 4.1 provides an elegant explanation for feature learning, but it is rigorously proved only for a highly simplified two-point, two-channel toy model. Scaling this claim to continuous, high-dimensional real-world data remains largely empirical.

A
**Feature-learning mechanism (for rebuttal / camera-ready; LaTeX below).** The two-point toy theorem is minimal; what we want to convey is a **reinforcing loop**: dominance of channel $k$ aligns the ReLU gate with large mixing entries $L_{c_2,k}$, which **amplifies** the mean-field backward drive on that channel; near the null initialization, $f_k$ grows monotonically toward a plateau—hence the “spike.”

```latex
\begin{remark}[Mechanism behind the channel spike (beyond the two-point toy)]
\label{rem:feature-learning-mechanism}
The explanatory content is not the simplified two-point, two-channel statement itself, but the \emph{positive feedback} it isolates.

Fix a second-layer neuron index $c_2$ and recall the low-rank pre-activation
\[
  H_2(c_2;x,W)=\sum_{j=1}^r L_{c_2,j}\,f_j(x;W).
\]
For ReLU, gating enters mean-field gradients through $\varphi_2'(H_2)=\mathbf{1}\{H_2>0\}$ (under the usual a.e.\ conventions). The channel-$k$ backward signal $B_k^{(2)}(t;x)$ (cf.\ the ODE for $\partial_t w_1(\cdot,k)$) is built from expectations over $C_2$ of terms that couple \textbf{(i)} the mixing weight $L_{C_2,k}$, \textbf{(ii)} the gate $\mathbf{1}\{H_2(C_2;x,W)>0\}$, and \textbf{(iii)} upstream factors (output error and later-layer weights).

When channel $k$ \emph{dominates}---$L_{c_2,k}f_k$ is large relative to other terms and typically has the same sign as $H_2(c_2;\cdot)$ on the active set---those neurons $c_2$ that contribute to $B_k^{(2)}$ are precisely those with the gate ``on.'' Across $c_2$, the effective weights in the expectation are then \textbf{positively aligned} with $L_{c_2,k}$ and with $\mathbf{1}\{H_2(c_2)>0\}$ (magnitude and sign), so the empirical correlation between ``large $|L_{c_2,k}|$'' and ``neuron active'' is high. That alignment produces a \textbf{large positive} contribution to the drift of the $k$-th channel, which in turn pushes $f_k$ further in the same direction: a \emph{reinforcing loop}.

At initialization the predictor is near the null map, so each $f_k$ starts small and increases toward a stabilizing magnitude while this coupling is active; the spike in channel $k$ is the visible signature of that transient, not an artifact of the toy data alone.
\end{remark}
```




Q
More critically, the core assumption maintaining the network's symmetric structure is overly restrictive and sensitive to hyperparameters; in my extended reproduction (using a 1D function with frequencies 
 and  , depth  , and width  ), the symmetry was largely imperceptible at rank  and only clear at rank  , suggesting that this strict low-rank requirement significantly limits the theory's robustness and applicability to broader network scales.

A
we are not sure on how symmetry is maintained, this is an open question tackled in future work/under reveiw

Q
questions : While symmetry is preserved at rank 10 , it becomes largely imperceptible at rank 20 . Since r=20 remains an extremely low-rank constraint relative to a width of  1024, this sensitivity challenges the practical robustness of the feature learning mechanism .


we are running symmetry experiments to see how it behaves (construct a table where we train and measure symmetry defects)
we still don't know and this symmetrical part looks more like something inherent to a highly symmetrical loss landscape for local minimas that are not symmetric for full rank NN
we emphasize we are running very small batch sgd which means symmetry has a lot of reason to be broken
for full rank NN and surprisingly remains present for low rank NN

Q

Theorem 3.3 bounds the finite-width approximation error by 
. Does the symmetry loss at  occur because the exponential factor  dominates  , causing divergence from idealized mean-field dynamics? If so, is "symmet preservation" a genuine benefit of low-rank architectures, or merely a fragile artifact of keeping  artificially small to prevent mean-field breakdown?

A
No, bound just comes from gronwal lemmas and ensure mathematical convergence but in practice symmetry is learned in parallel, it's just a worst case
analysis from gronwall

Q
If  rmust be kept strictly minimal to preserve mean-field behaviors, how does this framework scale to datasets with higher intrinsic dimensionality 
? If evaluated on a synthetic dataset with controlled 
, is there a theoretical or empirical phase transition when 
? It would be insightful to clarify if the network loses its theoretical structural benefits once the required rank 
 exceeds what the practical width 
 can exponentially suppress

A
for the higher intrinsice dimensionality d, random matrix theory coupled to ntk theory (work under review)
show that linear scaling of r with d leads to NTK convergence guarantees ; this is still work in progress but characterizing
the NTK matrix distribution for low rank networks is highly difficult and makes use of the framework developed in https://arxiv.org/abs/2508.20036. for the mean field framework (that is not the ntk one) the scaling should resemble something like
r linear in d to ensure high dimensional probability bounds (like sub gaussianity for gaussian equivalence principle)
remains valid after several output steps (which can be losed if we constrain outputs to live in a smaller space)
this is just our intuition that is currently tackled in work in progress

Q:
Limitations:
Freezing the mixing matrices 
 as random features---this architectural restriction significantly limits the theory's applicability to modern, fully end-to-end trained deep neural networks.



We want to show that everyone went wrong training full rank matrices, and want to apply it fo rhighly oscillating functions in sciML and PDE learning
that is why the applicative field is not current llms that do the converse with orthogonalizing matrices with Muon for instanceetc .. but is a way to bring a theoretically grounded path towards low rank so that people using that
after are not afraid : no, it does not constrain global convergence !

we are working under the optimization lense :
under the chizat perspective, https://arxiv.org/pdf/2509.10167 it seems that rank = sqrt(width) looks the best regime
where the mean field ODEs describe well the training dynamics (and hence global convergence occurs in practice)
so the phase diagram should be : you have in practice good convergence guarantees less than sqrt(width) ; and no convergence guarantee if upper ; but this mathematical limiteation only holds for a particular part of the proof

in practice if we use former ntk theory https://arxiv.org/pdf/2402.11867 r more than M^0.25 allows good lora training
in the ntk regime without spurious local minima, which means you should be between M^0.25 and M^0.5 according to sota results
to have good training in ntk/mean field regime ; even though this regimes are not the same, those bounds
remains indicative and work in progress adresses optimal choice for r

















3) The low-rank model this paper consider have a constant rank w.r.t to the width that tends to infinity, and only one matrix of the desomposed weight is trained, this seems to be far from practical settings. In particular, if we write the network architecture ( equation (1)) in matrix form, the efficient weight for each layer  is  where

same answer, we want to make people use low rank networks with/without random features, or at least try it 
random features is far from practical but avoid 
current trendy optimizers like muon do the converse, they orthogonalize the weights
some work have shown (weightwatchers) heavy tail spectra for llms weights ; we do not want to advocate this use for llms
in particular we are guided towards scientific computing and sciml, where deep learning architecture and spatial features matters
the most, also fourier/frequency features ; we think those networks can be the backbone of principal spatial-input architceture
(like branch networks of deeponets) 



Technically speaking, the proof of the Theorem 3.1-3.3 follows exactly the same route as in (Nguyen & Pham, 2023), and does not provide any new insight in my opinion. Could the authors elaborate more on the technical difference compare to (Nguyen & Pham, 2023)?

we apologize we did not explained the main new mechanism (because after having stated bounds it looked trivila to us to follow the path)
we explain the main difficulty (and how low rank resolve it), it's done in a companion Goodnotes, the technical difficulty is just at a particular point of the proof where low rank lipschitzness resolves everything

we are stacking 2 layers networks and merging them, but after each neuron (1024) get a different weighted contribution, those weights are not
trained which means there is no bias or collapse towards selecting a particular function (setting weights to 1 and other to 0)
we are currently working on deciphering the low rank collapse without random features, so that the architecture looks easier technically
but this random contribution make the final result highly not similar ; for more layers we do not have neural collapse but a rich feature learning
mechanism that we can analyze
concerning feature learning in mean field parameterized NN (i need to decipher SOTA) we do better because we interpret the output as a partial function
it's technically easier, easier to analyze features, mathematically the same difficulty (no simplification allowed in any part of the proof)

"


: for the answers that talk about goodnotes5 thisis the goodnotes explanation) in fact the explanation is that : , in the original nguyen paper 
there is the section 6.2.1 "
\subsubsection{High-level idea of the proof\label{subsec:three-layers-high-level-idea}}

Before we proceed, we give a high-level discussion of the proof of
Theorem \ref{thm:global-optimum-3}. This is meant to provide intuitions
and explain the technical crux, so our discussion may simplify and
deviate from the actual proof. Our first insight is to look at the
second layer's weight $w_{2}^{*}$. Recall that
\[
\frac{\partial}{\partial t}w_{2}^{*}\left(t,u_{1},u_{2},u_{3}\right)=-\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,u_{3};W^{*}\left(t\right)\right)\varphi_{1}\left(\left\langle w_{1}^{*}\left(t,u_{1}\right),X\right\rangle \right)\right].
\]
At convergence time $t=\infty$, we expect to have zero movement and
hence, denoting $\bar{W}=\left\{ \bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right\} $:
\[
\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,u_{3};\bar{W}\right)\varphi_{1}\left(\left\langle \bar{w}_{1}\left(u_{1}\right),X\right\rangle \right)\right]=0
\]
for $u_{1}\in{\rm supp}\left(\rho^{1}\right)$, $u_{3}\in{\rm supp}\left(\rho^{3}\right)$.
Suppose for the moment that we are allowed to make an additional (strong)
assumption on the limit $\bar{w}_{1}$: ${\rm supp}\left(\bar{w}_{1}\left(U_{1}\right)\right)=\mathbb{R}^{d}$
for $U_{1}\sim\rho^{1}$. It implies that the universal approximation
property, described in Assumption \ref{assump:three-layers}.5, holds
at $t=\infty$; more specifically, it implies $\left\{ \varphi_{1}\left(\left\langle \bar{w}_{1}\left(u_{1}\right),\cdot\right\rangle \right):\;u_{1}\in{\rm supp}\left(\rho^{1}\right)\right\} $
has dense span in $L^{2}\left({\cal P}_{X}\right)$. This thus yields
\[
\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,u_{3};\bar{W}\right)\middle|X=x\right]=0
\]
for ${\cal P}$-almost every $x$. Recalling the definition of $\Delta_{2}^{H*}$,
one can then easily show that
\[
\mathbb{E}_{Z}\left[\partial_{2}{\cal L}\left(Y,\hat{y}^{*}\left(x;\bar{W}\right)\right)\middle|X=x\right]=0.
\]
Global convergence follows immediately; for example, in Case 2 of
Theorem \ref{thm:global-optimum-3}, this is equivalent to that $\partial_{2}{\cal L}\left(y\left(x\right),\hat{y}^{*}\left(x;\bar{W}\right)\right)=0$
and hence ${\cal L}\left(y\left(x\right),\hat{y}^{*}\left(x;\bar{W}\right)\right)=0$
for ${\cal P}$-almost every $x$. In short, the gradient flow structure
of the dynamics of $w_{2}^{*}$ provides a seamless way to obtain
global convergence. Furthermore there is no critical reliance on convexity.

However this plan of attack has a potential flaw in the strong assumption
that ${\rm supp}\left(\bar{w}_{1}\left(U_{1}\right)\right)=\mathbb{R}^{d}$,
i.e. the universal approximation property holds at convergence time.
Indeed there are setups where it is desirable that ${\rm supp}\left(\bar{w}_{1}\left(U_{1}\right)\right)\neq\mathbb{R}^{d}$
\cite{mei2018mean,chizat2019sparse}; for instance, it is the case
where the neural network is to learn some ``sparse and spiky'' solution,
and hence the weight distribution at convergence time, if successfully
trained, cannot have full support. On the other hand, one can entirely
expect that if ${\rm supp}\left(w_{1}^{*}\left(0,U_{1}\right)\right)=\mathbb{R}^{d}$
initially at $t=0$, then ${\rm supp}\left(w_{1}^{*}\left(t,U_{1}\right)\right)=\mathbb{R}^{d}$
at \textsl{any} finite $t\geq0$. The crux of our proof is to show
the latter without assuming ${\rm supp}\left(\bar{w}_{1}\left(U_{1}\right)\right)=\mathbb{R}^{d}$.
This is done via an algebraic topology argument, in which the mapping
$\left(t,u\right)\mapsto M\left(t,u\right)$ that maps from $\left(t,w_{1}^{*}\left(0,u_{1}\right)\right)=\left(t,u_{1}\right)$
to $w_{1}^{*}\left(t,u_{1}\right)$ is shown to preserves a homotopic
structure through time."



2 remarks : we avoid the potentiel flaw by not training the 1st layer ; in this perspective and up to our knowledge
for the high level picture we are the first to use this "no training of 1st layer" for "global convergence" ; which means we are the 1st use
this insight they have built thourgh 1 year of work ; but not training what ? only the 1st layers ? and other layers ? should I not train some other complete W matrices inside ? this paper tells that no matter where you're not training, if you go through 4 layers (included) or more neural features are not learned anymore in the mean field framework.
We counter this by training only 1 part of matrices  of their low rank factorization, which 1) remove the neural collapses 2) do not change expressivity that much under approximation bounds 3) has far less parameters 4) has empirically a far better loss landscape

for us the workflow was "low rank was incredibly better to train, better landscape, better interpreation of features, far less parameters" but why ? using nguyen paper we now know why and this is what we want to share
to the community



in this perspective we fitted all the low rank framework in the meanfield one, but we need to ensure
existence, unicity = this is what the 1st part of the appendix do, instead of writing 'left to the reader"

the technical difficulty is not there, this part was trivial

but the most important difficulty comes from what the global convergence requires

in the main argument section 6.3 after step3  "

\begin{proof}[Proof of Theorem \ref{thm:global-optimum-3}]
Let $U_{i}\sim\rho^{i}$, $i=1,2,3$ independently. It is easy to
check that Assumptions \ref{enu:Assump_lrSchedule}-\ref{enu:Assump_backward},
as well as the conditions of Lemma \ref{lem:full-support-3}, hold.
Therefore, by Lemma \ref{lem:full-support-3}, the support of ${\rm Law}\left(w_{1}^{*}\left(t,U_{1}\right)\right)$
is $\mathbb{R}^{d}$ at all $t$. We recall from the convergence assumption
the limits $\bar{w}_{1}$, $\bar{w}_{2}$ and $\bar{w}_{3}$, and
we shall first prove $\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)$
is a global minimizer of $\mathscr{L}$ in Case 1 and $\mathscr{L}\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)=0$
in Case 2.

By the convergence assumption, we have that for any $\epsilon>0$,
there exists $T\left(\epsilon\right)$ such that for all $t\geq T\left(\epsilon\right)$
and almost surely:
\[
\epsilon\geq\left|\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(t,Z,U_{3}\right)\varphi_{1}\left(\left\langle w_{1}^{*}\left(t,U_{1}\right),X\right\rangle \right)\right]\right|=\left|\left\langle \mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(t,Z,U_{3}\right)|X=x\right],\varphi_{1}\left(\left\langle w_{1}^{*}\left(t,U_{1}\right),x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|.
\]
Since ${\rm Law}\left(w_{1}^{*}\left(t,U_{1}\right)\right)$ has full
support, we obtain that for $u$ in a dense subset of $\mathbb{R}^{d}$,
\[
{\rm ess\text{-}sup}\left|\left\langle \mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(t,Z,U_{3}\right)|X=x\right],\varphi_{1}\left(\left\langle u,x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|\leq\epsilon.
\]
By continuity of $u\mapsto\varphi_{1}(\left\langle u,\cdot\right\rangle )$
in $L^{2}({\cal P}_{X})$, we extend the above to all $u\in\mathbb{R}^{d}$.
Recall the couplings $\pi_{t}$ in Assumption \ref{assump:three-layers}.4,
since $\varphi_{1}$ is bounded,
\begin{align*}
 & \mathbb{E}_{(U_{3},U_{3}')\sim\pi_{t}}\left[\left|\left\langle \mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(t,Z,U_{3}\right)-\Delta_{2}^{H*}\left(Z,U_{3}';\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\middle||X=x\right],\varphi_{1}\left(\left\langle u,x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|\right]\\
 & \le K\mathbb{E}_{\pi_{t}}\left[\left|\Delta_{2}^{H*}\left(t,Z,U_{3}\right)-\Delta_{2}^{H*}\left(Z,U_{3}';\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right|\right]\\
 & \le K\mathbb{E}_{\pi_{t}}\Big[\left(1+\left|\bar{w}_{3}(U_{3})\right|\right)\Big(\left|w_{3}^{*}(t,U_{3}')-\bar{w}_{3}(U_{3})\right|+\left|\bar{w}_{3}(U_{3})\right|\left|w_{2}^{*}(t,U_{1}',U_{2}',U_{3}')-\bar{w}_{2}(U_{1},U_{2},U_{3})\right|\\
 & \qquad+\left|\bar{w}_{3}(U_{2})\right|\left|\bar{w}_{2}(U_{1},U_{2},U_{3})\right|\left|w_{1}^{*}(t,U_{1}')-\bar{w}_{1}(U_{1})\right|\Big)\Big],
\end{align*}
where the last step is by the regularity assumption, similar to the
calculation in the proof of Theorem \ref{thm:global-optimum-2}. Recall
that the right-hand side converges to $0$ as $t\to\infty$. We thus
obtain that for all $u\in\mathbb{R}^{d}$,
\[
\mathbb{E}_{U_{3}}\left[\left|\left\langle \mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,U_{3};\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)|X=x\right],\varphi_{1}\left(\left\langle u,x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|\right]=0,
\]
which yields that for all $u\in\mathbb{R}^{d}$ and almost surely,
\[
\left|\left\langle \mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,U_{3};\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)|X=x\right],\varphi_{1}\left(\left\langle u,x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|=0.
\]
Here we note that by the regularity assumption that 
\[
\left|\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,U_{3};\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)|X=x\right]\right|\leq K\left|\bar{w}_{3}\left(U_{3}\right)\right|,
\]
and so $\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,u_{3};\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)|X=x\right]$
is in $L^{2}\left({\cal P}_{X}\right)$ for almost every $u_{3}$.
Since $\left\{ \varphi_{1}\left(\left\langle u,\cdot\right\rangle \right):\;u\in\mathbb{R}^{d}\right\} $
has dense span in $L^{2}\left({\cal P}_{X}\right)$, we have $\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,u_{3};\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)|X=x\right]=0$
for ${\cal P}_{X}$-almost every $x$ and almost every $u_{3}$, and
hence
\[
\mathbb{E}_{Z}\left[\partial_{2}{\cal L}\left(Y,\hat{y}^{*}\left(X;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)\middle|X=x\right]\varphi_{3}'\left(H_{3}^{*}\left(x;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)\bar{w}_{3}\left(u_{3}\right)\varphi_{2}'\left(H_{2}^{*}\left(x,u_{3};\bar{w}_{1},\bar{w}_{2}\right)\right)=0.
\]
We note that our assumptions guarantee that $\mathbb{P}\left(\bar{w}_{3}\left(U_{3}\right)\ne0\right)$
is positive. Indeed:
\begin{itemize}
\item In the case $\int\mathbb{I}\left(u_{3}\neq0\right)\rho^{3}\left(du_{3}\right)>0$
and $\xi_{3}\left(\cdot\right)=0$, it is obvious that $\mathbb{P}\left(\bar{w}_{3}\left(U_{3}\right)\ne0\right)>0$.
\item In the case $\mathscr{L}\left(w_{1}^{0},w_{2}^{0},w_{3}^{0}\right)<\mathbb{E}_{Z}\left[{\cal L}\left(Y,\varphi_{3}\left(0\right)\right)\right]$,
it can be easily checked that
\[
\mathscr{L}\left(w_{1}^{*}\left(t,\cdot\right),w_{2}^{*}\left(t,\cdot,\cdot,\cdot\right),w_{3}^{*}\left(t,\cdot\right)\right)\leq\mathscr{L}\left(w_{1}^{*}\left(t',\cdot\right),w_{2}^{*}\left(t',\cdot,\cdot,\cdot\right),w_{3}^{*}\left(t',\cdot\right)\right),
\]
for $t\geq t'$. This is in fact a standard property of gradient flows.
In particular, setting $t'=0$ and taking $t\to\infty$, it is easy
to see that
\[
\mathscr{L}\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\leq\mathscr{L}\left(w_{1}^{0},w_{2}^{0},w_{3}^{0}\right)<\mathbb{E}_{Z}\left[{\cal L}\left(Y,\varphi_{3}\left(0\right)\right)\right].
\]
If $\mathbb{P}\left(\bar{w}_{3}\left(U_{3}\right)=0\right)=1$ then
$\mathscr{L}\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)=\mathbb{E}_{Z}\left[{\cal L}\left(Y,\varphi_{3}\left(0\right)\right)\right]$,
a contradiction.
\end{itemize}
Then since $\varphi_{2}'$ and $\varphi_{3}'$ are strictly non-zero,
we have $\mathbb{E}_{Z}\left[\partial_{2}{\cal L}\left(Y,\hat{y}^{*}\left(X;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)\middle|X=x\right]=0$
for ${\cal P}_{X}$-almost every $x$.

In Case 1, since ${\cal L}$ convex in the second variable, for any
measurable function $\tilde{y}(x)$, 
\[
{\cal L}\left(y,\tilde{y}\left(x\right)\right)-{\cal L}\left(y,\hat{y}\left(x;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)\ge\partial_{2}{\cal L}\left(y,\hat{y}\left(x;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)\left(\tilde{y}\left(x\right)-\hat{y}\left(x;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right).
\]
Taking expectation, we get $\mathbb{E}_{Z}\left[{\cal L}\left(Y,\tilde{y}\left(X\right)\right)\right]\geq\mathscr{L}\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)$,
i.e. $\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)$ is a global
minimizer of $\mathscr{L}$.

In Case 2, since $y$ is a function of $x$, we obtain $\partial_{2}{\cal L}\left(y,\hat{y}\left(x;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)=0$
and hence ${\cal L}\left(y,\hat{y}\left(x;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)=0$
for ${\cal P}_{X}$-almost every $x$.

Finally to connect $\mathscr{L}\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)$
with $\mathscr{L}\left(W^{*}\left(t\right)\right)$ in the limit $t\to\infty$,
we have:
\begin{align*}
\left|\mathscr{L}\left(W^{*}\left(t\right)\right)-\mathscr{L}\left(\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right| & =\left|\mathbb{E}_{Z}\left[{\cal L}\left(Y,\hat{y}^{*}\left(t,X\right)\right)-{\cal L}\left(Y,\hat{y}^{*}\left(X;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right)\right]\right|\\
 & \leq K\mathbb{E}_{Z}\left[\left|\hat{y}^{*}\left(t,X\right)-\hat{y}^{*}\left(X;\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right|\right]\\
 & \leq K\mathbb{E}_{\pi_{t}}\Big[\left|w_{3}^{*}\left(t,U_{3}'\right)-\bar{w}_{3}\left(U_{3}\right)\right|+\left|\bar{w}_{3}\left(U_{3}\right)\right|\left|w_{2}^{*}\left(t,U_{1}',U_{2}',U_{3}'\right)-\bar{w}_{2}\left(U_{1},U_{2},U_{3}\right)\right|\\
 & \qquad+\left|\bar{w}_{3}\left(U_{3}\right)\right|\left|\bar{w}_{2}\left(U_{1},U_{2},U_{3}\right)\right|\left|w_{1}^{*}\left(t,U_{1}'\right)-\bar{w}_{1}\left(U_{1}\right)\right|\Big]
\end{align*}
which tends to $0$ as $t\to\infty$. This completes the proof.
\end{proof}
"


the most important part is that : 
gle u,x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|\leq\epsilon.
\]
By continuity of $u\mapsto\varphi_{1}(\left\langle u,\cdot\right\rangle )$
in $L^{2}({\cal P}_{X})$, we extend the above to all $u\in\mathbb{R}^{d}$.
Recall the couplings $\pi_{t}$ in Assumption \ref{assump:three-layers}.4,
since $\varphi_{1}$ is bounded,
\begin{align*}
 & \mathbb{E}_{(U_{3},U_{3}')\sim\pi_{t}}\left[\left|\left\langle \mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(t,Z,U_{3}\right)-\Delta_{2}^{H*}\left(Z,U_{3}';\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\middle||X=x\right],\varphi_{1}\left(\left\langle u,x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|\right]\\
 & \le K\mathbb{E}_{\pi_{t}}\left[\left|\Delta_{2}^{H*}\left(t,Z,U_{3}\right)-\Delta_{2}^{H*}\left(Z,U_{3}';\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right|\right]\\
 & \le K\mathbb{E}_{\pi_{t}}\Big[\left(1+\left|\bar{w}_{3}(U_{3})\right|\right)\Big(\left|w_{3}^{*}(t,U_{3}')-\bar{w}_{3}(U_{3})\right|+\left|\bar{w}_{3}(U_{3})\right|\left|w_{2}^{*}(t,U_{1}',U_{2}',U_{3}')-\bar{w}_{2}(U_{1},U_{2},U_{3})\right|\\
 & \qquad+\left|\bar{w}_{3}(U_{2})\right|\left|\bar{w}_{2}(U_{1},U_{2},U_{3})\right|\left|w_{1}^{*}(t,U_{1}')-\bar{w}_{1}(U_{1})\right|\Big)\Big],
\end{align*}

and 
Here we note that by the regularity assumption that 
\[
\left|\mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(Z,U_{3};\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)|X=x\right]\right|\leq K\left|\bar{w}_{3}\left(U_{3}\right)\right|,
\]





that is translated in our paper by : \paragraph{$\mathscr{L}(W(t))\to\mathscr{L}(\bar W)$.}
By Assumption~\ref{assump:convergence}, the couplings $\pi_t$ and the Wasserstein-like integrals \eqref{eq:conv-w1}--\eqref{eq:conv-w2} (and their $L$-layer analogues) tend to 0. The output difference $|\hat{y}(X;W(t))-\hat{y}(X;\bar W)|$ is bounded by a $K$-multiple of those integrals (via the low-rank structure: $H_2=\sum_k L_{c_2,k}f_k$, $B_k^{(\ell)}$, and the regularity of $\varphi_\ell$, $\partial_2\mathcal{L}$). Thus $\mathscr{L}(W(t))-\mathscr{L}(\bar W)=\E_Z[\mathcal{L}(Y,\hat{y}(X;W(t)))-\mathcal{L}(Y,\hat{y}(X;\bar W))]$ is bounded by $K\E_Z[|\hat{y}(X;W(t))-\hat{y}(X;\bar W)|]\to 0$ as $t\to\infty$.



we apologize because in fact we were not precise enough on the coupling argument : there is 1 step lacking

this part from nguyen paper Recall the couplings $\pi_{t}$ in Assumption \ref{assump:three-layers}.4,
since $\varphi_{1}$ is bounded,
\begin{align*}
 & \mathbb{E}_{(U_{3},U_{3}')\sim\pi_{t}}\left[\left|\left\langle \mathbb{E}_{Z}\left[\Delta_{2}^{H*}\left(t,Z,U_{3}\right)-\Delta_{2}^{H*}\left(Z,U_{3}';\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\middle||X=x\right],\varphi_{1}\left(\left\langle u,x\right\rangle \right)\right\rangle _{L^{2}\left({\cal P}_{X}\right)}\right|\right]\\
 & \le K\mathbb{E}_{\pi_{t}}\left[\left|\Delta_{2}^{H*}\left(t,Z,U_{3}\right)-\Delta_{2}^{H*}\left(Z,U_{3}';\bar{w}_{1},\bar{w}_{2},\bar{w}_{3}\right)\right|\right]\\
 & \le K\mathbb{E}_{\pi_{t}}\Big[\left(1+\left|\bar{w}_{3}(U_{3})\right|\right)\Big(\left|w_{3}^{*}(t,U_{3}')-\bar{w}_{3}(U_{3})\right|+\left|\bar{w}_{3}(U_{3})\right|\left|w_{2}^{*}(t,U_{1}',U_{2}',U_{3}')-\bar{w}_{2}(U_{1},U_{2},U_{3})\right|\\
 & \qquad+\left|\bar{w}_{3}(U_{2})\right|\left|\bar{w}_{2}(U_{1},U_{2},U_{3})\right|\left|w_{1}^{*}(t,U_{1}')-\bar{w}_{1}(U_{1})\right|\Big)\Big],
\end{align*}
translates to 



GOODNOTES (tablet sketch $\rightarrow$ LaTeX; notation aligned with the Nguyen--Pham coupling step around lines 478--485 above).

**Setup ($r{+}1$ coupled mean-field equations).** Picard / fixed-point step as on pp.~37--38 of \NP{}; $W_2$ convergence via their Lemma~8-type estimate. Low-rank specialization ($L{=}2$ toy in margin): $f_1$ fixed, $f_2{=}w_1$. Red margin: $W_2(U_1,U_2,U_3)\to W_2(U_2)$ under pushforwards of $\mathrm{Law}(w,c,u)$ --- this is the **new hypothesis** replacing the fully-connected contraction.

**Main coupling term (integral w.r.t.\ $\pi_t$).** Bounded first-layer gates $\varphi^{(1,1)},\ldots,\varphi^{(1,r)}$ and second-layer $\varphi^{(2)}$ (cf.\ Assumption~3 / bounded activations). The low-rank analogue of the chain after ``translates to'' is schematically:
\begin{align*}
& \mathbb{E}_{\pi_t}\Big[
\Big(\prod_{k=1}^{r}\varphi^{(1,k)}(\cdots)\Big)\,\varphi^{(2)}(\cdots)\,
\bigl(1+\lvert \bar w_{2}(U_{2})\rvert\bigr)\,\lvert \bar w_{2}(U_{2})\rvert
\sum_{k=1}^{r}
\bigl\lvert \bar w_{1}(u,v_{k})\bigr\rvert\,
\bigl\lvert \bar w_{1}(h,v,v_{k})-\bar w_{1}(u,v_{k})\bigr\rvert
\Big]
\;\longrightarrow\; 0,
\end{align*}
where $\pi_t$ couples finite-$N$ and mean-field particles (same role as $(U_{3},U_{3}')\sim\pi_{t}$ in the display above). The arguments $(h,u,v,v_k)$ are the neuronal labels from the sketch; in the write-up they should match the explicit $(U_{1},U_{1}',C_k)$ notation of Eq.~(1) / the neuronal embedding section. The blank ``$\displaystyle\int (\cdots)\,d\pi_t\to 0$'' in the pad is this expectation.

**Time regularity (Sect.~7.4.2 style).** For each channel $k=1,\ldots,r$,
\[
\mathrm{ess\,sup}_{t}\Bigl\lvert \frac{\partial}{\partial t}w_{1}(t,U_{k})\Bigr\rvert \;\to\; 0
\]
(``page~42 $\le K\mathbb{E}_{\pi_t}[\cdots]$'' in the notes). The delicate factor is $\varphi_{2}'(H_{2})$; bounds use $\Delta_{2}\le K_{t}(1+\lvert U_{3}\rvert)$, $L_{s}\le K(1+\Delta_{3}^{H^{*}})$, and backward maps Lipschitz/bilinear with constant $K(\lvert a\rvert+\lvert b\rvert)(1+\lvert\Delta\rvert)$.

**Bi-Lipschitz / backward term.** The sum $\sum_{k=1}^{r}(\cdots)$ is controlled under the bi-Lipschitz conditions on backward quantities (blue arrow in notes).

**ReLU trick (boxed).** With $s=\sum_{i=1}^{r}x_{i}$ and $s'=\sum_{i=1}^{r}y_{i}$,
\begin{equation}
\label{eq:goodnotes-relu-sum}
\left\lvert \mathrm{ReLU}(s)-\mathrm{ReLU}(s')\right\rvert
\;\le\; \left\lvert s-s'\right\rvert
\;\le\; \sum_{i=1}^{r}\lvert x_{i}-y_{i}\rvert.
\end{equation}
(Handwritten ``$\le r\sum_i|x_i-y_i|$'' is either a slack bookkeeping bound or a typo; \eqref{eq:goodnotes-relu-sum} is the sharp triangle-inequality step.) Green margin: this is the **structural reason $r$ coupled equations help** --- without ReLU one gets a cleaner $\sum_{k}\mathbb{E}_{c_{k}}[\cdots]\le K\lvert c_{s2}\rvert$-type bound; with ReLU one needs \eqref{eq:goodnotes-relu-sum} plus **compact support / boundedness** of pre-activations to close constants.

**What to paste into `answers.tex` / appendix.** One paragraph + \eqref{eq:goodnotes-relu-sum} + the $\mathbb{E}_{\pi_t}[\cdots]\to 0$ display after fixing $(u,h,v,v_k)$ to official notation.






While I acknowledge the spectral bias for low-rank network could be interesting, the discussions in Section 4.1 seems to be too qualitative, and the results in Section 4.2 seems to be on too simple setting (only two point in the datasets). Could the authors elaborate more on why the low-rank network lead to high-frequency bias, and fully trained network lead to low-frequency bias?



it is effectively qualitative (i need to make it better and understand more this feature learning mechanism and separation)
(i need also to make better log ratio trajectories, disentangle the 0 contribution because everything mess up when sign changes which leads to big spikes)
(does it converge to 0 or it traverse towards negatives ?)
the low rank network high frequency bias is an unsolved problem, in other submission (ICLR) we show experimentally that low rank networks
lean high frequency features at each plateau landscape
low frequency bias for full rank NN is partly explained by the NTK, it's not that they don't fit the same function, it's that the loss landscape
is not guided towards high frequency compared to low rank experimentally

low rank networks do not have high frequency bias but a better bias than full rank 
the first explanation can come from bengio paper : 
the number of regions (polytopes) of same variations is significantly smaller
the decay is low in directions orthogonal of high dimension boundaries, and fast in low dimensional boundaries, but there are far less low dim boundaries, (because so few number of equations), which means
the regions have better fourier spetcrum ; since we drastically diminish the number of path , a boundary
being intersection of  a lot of linear inequalities, there are a lot of them for full rank NN, few f them for low rank NN , which mean its globally decay faster for full rank NN


from iclr paper i'll put Figure 5: Mean number of strictly-positive local minima per component vs. layer `; MMNN L = 6,
W = 128, R = 5

this is work in progress (i can elaborate a lot on that)

concerning the frequency bias — **condensed from `refs/ICLR/` (merged draft, `tex/rkhs.tex`, `tex/archive.tex`, `quantitative.tex`, `tex/introduction.tex`, `finalidea.md`):**

**NTK / lazy training (does *not* explain high-frequency reversal by itself).** For RF-LR with frozen random features, the **RKHS in the NTK regime is the same object as for a comparable MLP** (same “deep equals shallow” ReLU RKHS picture as Bietti et al.); freezing features is a way to make that statement literal. **New low-rank-specific effect in the NTK:** the **radial / rank-$r$ piece of the kernel concentrates exponentially fast in $r$** (Gaussian norm products $\|x_1\|\|y_1\|/r$; Fisher/Kibble structure in the empirical NTK). So at **infinite width + small step** you should **not** expect the *kernel* alone to flip classical low-frequency preference—the interesting story is **feature learning / rich dynamics**, not the lazy kernel spectrum.

**What the ICLR experiments actually claim (landscape ↔ frequency).** Training on **1D sum-of-cosines / frequency-structured targets** shows: **long loss plateaus** then **sharp drops**; each plateau/saddle segment is aligned with **a frequency band not yet fit** → **saddle-to-saddle dynamics** along training. **LR decay** matters: the picture is a **flat plateau pitted with very sharp, deep basins**; trajectories often **wander on the plateau** until **LR is small enough**, then **snap** into a basin (loss drops sharply)—see the planned figures under `experiments/table/results_sumcos_selected_rerun/.../plot_loss_and_fit_before_after_lr_divides.png` (noted in `finalidea.md`). **Plateau-escape time** empirically scales **roughly like $1/\text{lr}$** (e.g. $\mathcal{O}(10)$ epochs vs $\mathcal{O}(10^3)$), with **batch-size** effects (small batch helps escape); there is also a **leap-complexity** note (`leap_complexity_scaling.tex`) modeling diffusion on plateaus toward “holes.”

**Feature-learning diagnostics (beyond loss curves).** **Channel partial functions** $f_k$ and **log-ratios** $R_{i,j}=\log|f_i|-\log|f_j|$ at fixed $x$: used to track **specialization vs. collapse**; **LR/batch** can yield **diverging vs. converging** log-ratio trajectories (e.g. smaller LR → stable specialization). **Oscillatory complexity:** e.g. **mean number of strict local minima per partial-function component** grows with **layer index**—deeper layers look **more oscillatory** after training. Empirical slogan in `merged.tex`: layer $L$ can show on the order of **$\sim 2L$ “spikes”** in partials (distinct from generic MLP scaling). **Scaling sweep** (`SCALING_LAW_DEPTH_WIDTH` family in `merged.tex`): **low** frequency multipliers converge to very small error; **high** frequency + depth can **fail** or **blow up** (NaN at very large $L$ in some configs)—so “linear-in-depth spectrum capability” is **partial / regime-dependent WIP**, not a clean theorem yet.

**Theoretical hooks already written (ICLR tex).** `quantitative.tex` **Remark (deterministic mixing):** choosing structured (even **deterministic**) $L$ with only $\|L\|_{\infty,1}\le rK$ can **steer channels** toward **different frequency components** via $B_k$. `archive.tex` / intro: connect **multi-index staircase** and **band-by-band** Fourier intuition to **plateaus** (low $\kappa(\omega)$ for high $|\omega|$ → slow directions). **Title-level claim** in `finalidea.md`: **“Low-rank networks revert the spectral bias and learn high-frequency features”** — intended meaning: **in the observed rich regime**, **optimization landscape + channel dynamics** route learning through **high-frequency plateaus** that **full-rank / NTK-first training** often **does not prioritize** (not a denial of classical NTK low-frequency bias in the lazy limit).

**What we can honestly say to this reviewer (ICML vs ICLR).** Section 4.1–4.2 here are **intentionally qualitative**; the **sharper quantitative story** (log-ratio protocol, plateau–escape scaling, depth–frequency scaling tables, RKHS/NTK concentration) lives in the **ICLR companion** and referenced experiment scripts. **Open:** make **log-ratio plots** robust to **sign changes / zeros**; separate **“converges to 0”** vs **“crosses negative”** in ratio space; tie **Gram / doubly-stochastic** spikes to a **publishable** statement (currently **speculative** in your scratch).


from the bengio/dimension polytope faces explanation of the low rank bias (put it there) it feels convincing and a 1st formal/algebraic explanation of this bias ; in practice work in review has shown empirically a frequency learning mechanism : layer L seems to fit L spikes well
(linear in depth spectrum)



As I mentioned in the Strengths and Weaknesses part, the architecture considered in this paper is a stacking of two-layer network, I don't see the difficulty of proving the mean-field limit when both 
 are trained. Could the authors elaborate more on the technical difficulty

 the paper from nguyen et al was seminal in the way it bring this new framework, we are up to our knowledge the 1st one to use thigs framework to show that low rank is enough, then the technical difficulty occurs in 1 particular moment (GOODNOTES EXPLANATION°)
 

I wonder how universal is spectral bias discovered in Section 4 for low-rank training? For example, if we do not consider network in the mean-field regime, do we still have such spectral bias?

we have mainly ntk, mu-p, meanfield regimes ; under the NTK regime the spectral bias is the same (ntk is the same, COLT PAPER)
for mu-p we still don't know, this is work in progress ; we point towards this experiments 

































4)


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
in fact this paper is a 1st path towards exploring that landscape for low rank networks,
in the ntk regime this is work under reveiw, in the mean field regime tihis is this one, 
we hold experimental resuls showing the loss landscape behavior for low rank nn
differ a lot than for full rank NN





For the Fig. 
, how are different momentum rates selected for the corresponding level of rank constraints?

we ran sweep (run other experiments)




Do low-rank random feature networks differ from the multi-component multilayer neural networks proposed by Zhang et al. (2023) only in their low-rank structure?

**Response.** The architecture is identical to the multi-component networks of Zhang et al. (2023). Our novel contribution is the mean-field training analysis: we derive tractable MF ODEs for the $r$ channels, prove global convergence under standard i.i.d. initialization, and uncover the channel-specialization mechanism. Zhang et al. do not study optimization.

Typo at (137), 
 should be denoted by vector notation.

Could you clarify the convergence assumptions made by Nguyen & Pham (2023) for multilayer and two layer neural networks?

### Draft answer / notes (for the verbatim reviewer question above): Nguyen \& Pham (2023) convergence assumptions (setting-dependent)

**Note.** The labels below (e.g.\ Assumption numbers / page ranges like 6.4 p.~35--36, 7.4 p.~37--38, 8.4 p.~46--47) are as in a common NP PDF layout—**verify against your local copy** before citing page numbers in the camera-ready.

#### Two-layer mean-field networks
*(NP: two-layer MF assumption block, often numbered in the vein of their two-layer list.)*

- **Coupled, weight-scaled $L^1$ vanishing.** There exist limits $(\bar w_1,\bar w_2)$ and, as $t\to\infty$, a **coupling** $\pi_t$ of the first-layer marginal $P_1$ with itself such that, for $(C_1,C_1')\sim\pi_t$,
  $$
  \mathbb{E}_{\pi_t}\Bigl[\bigl|\bar w_2(C_1)\bigr|\,\bigl|w_1(t,C_1')-\bar w_1(C_1)\bigr|+\bigl|w_2(t,C_1',1)-\bar w_2(C_1)\bigr|\Bigr]\to 0.
  $$
- **Uniform decay of the second-layer velocity** (used to pass finite-$t$ identities to the limit):
  $$
  \mathrm{ess\text{-}sup}_{t}\left|\frac{\partial}{\partial t}w_2(t,C_1,1)\right|\to 0.
  $$

#### Three-layer networks
- **Same logical structure on** $\rho^1\times\rho^2\times\rho^3$: a coupling $\pi_t$ of the **product law with itself** such that **three** weighted integrals of $|w_1^*-\bar w_1|$, $|w_2^*-\bar w_2|$, $|w_3^*-\bar w_3|$ tend to $0$, with prefactors built from **products of (limit) downstream weight magnitudes** $(1+|\bar w_3|)|\bar w_3||\bar w_2|$-type terms (exact display matches their Assumption for the three-layer case).
- **Parallel** $\mathrm{ess\,sup}\,|\partial_t w|$ conditions where their proof needs them.

#### Multilayer ($L$ layers)
- For **arbitrary depth**, there exists $\pi_t$ coupling the full $L$-tuple of laws with itself such that **weighted** gaps between $w_i(t,\cdot)$ and $\bar w_i$ vanish; weights involve **products of** $|\bar w_j|$ **for $j>i$** (downstream amplification along the backward pass).

#### Why this assumption matters (NP)
- They prove **partial converses** (e.g.\ propositions in their global-convergence section—sometimes cited around pp.~38 and ~50 in some PDFs): if the **uniform** $\mathrm{ess\,sup}\,|\partial_t w|$-type conditions fail, **global convergence need not hold** along their argument. So the time-regularity part is **proof-relevant**, not a side remark.
- The **weighted** integrals are **not decorative**: they arise from **chain-rule / Lipschitz** control of backward quantities ($\Delta^{H*}$-type terms)—errors are scaled by **downstream weight magnitudes**.

---

### “Weighted Wasserstein”: what NP’s quantity is (and is not)

**Standard Wasserstein-$p$.** For measures $\mu,\nu$ on $\mathbb{R}^d$,
$$
W_p(\mu,\nu)=\left(\inf_{\pi\in\Pi(\mu,\nu)}\int \|x-y\|^p\,d\pi(x,y)\right)^{1/p},
$$
where $\Pi(\mu,\nu)$ are **couplings** (joint laws with marginals $\mu,\nu$).

**NP’s two-layer expression** is **not** literally $W_p$. It is a **coupled expectation**
$$
\mathbb{E}_{\pi_t}\bigl[|\bar w_2(C_1)|\,|w_1(t,C_1')-\bar w_1(C_1)|+|w_2(t,C_1',1)-\bar w_2(C_1)|\bigr]
$$
with **nonnegative weights** (here $|\bar w_2|$ on the $w_1$ gap). Interpretation: **transport-type coupling** of the finite-$t$ law to the limit, but the “cost” of the $w_1$ discrepancy is **scaled by how much that coordinate matters for the output at the limit** (downstream magnitude).

**Why weight $w_1$ by $|\bar w_2|$?** Heuristically, $\partial \hat y/\partial w_1$ scales with **$w_2$** (and $\sigma'$, etc.). Large $|\bar w_2|$ ⇒ small changes in $w_1$ move $\hat y$ a lot; if $|\bar w_2|$ is tiny, $w_1$ can differ more without moving the loss much. The condition asks convergence **in the sense needed for the global-convergence proof**, not bare $L^2$ convergence of raw weights.

**Deeper layers.** Prefactors become **products** along downstream layers—same **backprop amplification** picture.

**Relation to $W_4$.** NP also give a **sufficient** route: **$W_4$ convergence** of the coupled particle tuple, plus **a priori** moment bounds, can **imply** these weighted $L^1$ integrals (Hölder). The **definition** of their convergence item remains the **weighted coupled integrals**, not “$W_4$” alone.

---

**One-line scratch (keep or delete).**  
Informally: “weights stop moving” $\approx$ **limit object exists** and **velocities vanish in $\mathrm{ess\,sup}$** where NP require it; the **integral** conditions are stronger than weak convergence—they are **proof-tailored weighted transport**.

convergence assumption assume that every weights stop moving, which means training is ended
we follow the francis bach paper that threw basis https://francisbach.com/gradient-descent-for-wide-two-layer-neural-networks-implicit-bias/

"
\begin{assumption}
\label{assump:two-layers}Consider the MF limit corresponding to the
network (\ref{eq:two-layer-nn}), such that they are coupled together
by the coupling procedure in Section \ref{subsec:Neuronal-Embedding}.
We consider the following assumptions:
\begin{enumerate}
\item Initialization: The initialization law $\rho^{0}$ satisfies
\[
\max\left(\sup_{m\geq1}\frac{1}{\sqrt{m}}\mathbb{E}_{C_{1}}\left[\left|w_{1}\left(0,C_{1}\right)\right|^{m}\right]^{1/m},\quad\sup_{m\geq1}\frac{1}{\sqrt{m}}\mathbb{E}_{C_{1}}\left[\left|w_{2}\left(0,C_{1},1\right)\right|^{m}\right]^{1/m}\right)\leq K.
\]
\item Diversity: The support of $\rho^{0}$ contains the graph of a continuous
function $F:\;\mathbb{R}^{d}\to\mathbb{R}$ such that $\left|F\left(u\right)\right|\leq K$
for all $u\in\mathbb{R}^{d}$.
\item Regularity: $\varphi_{1}$ is $K$-bounded, $\varphi_{1}'$ and $\varphi_{2}'$
are $K$-bounded and $K$-Lipschitz, $\varphi_{2}'$ is non-zero everywhere,
$\partial_{2}{\cal L}\left(\cdot,\cdot\right)$ is $K$-Lipschitz
in the second variable and $K$-bounded\footnote{We denote by $\partial_{2}{\cal L}\left(\cdot,\cdot\right)$ the partial
derivative of ${\cal L}$ with respect to the second variable.}, and $\left|X\right|\leq K$ with probability $1$. 
\item Convergence: There exist limits $\bar{w}_{1}$ and $\bar{w}_{2}$
such that as $t\to\infty$, there exists a coupling $\pi_{t}$ of
$P_{1}$ and itself such that 
\[
\mathbb{E}_{\pi_{t}}\left[\left|\bar{w}_{2}(C_{1})\right|\left|w_{1}(t,C_{1}')-\bar{w}_{1}(C_{1})\right|+\left|w_{2}(t,C_{1}',1)-\bar{w}_{2}(C_{1})\right|\right]\to0
\]
for $(C_{1},C_{1}')\sim\pi_{t}$. Furthermore, ${\rm ess\text{-}sup}\left|\frac{\partial}{\partial t}w_{2}\left(t,C_{1},1\right)\right|\to0$.
\item Universal approximation: $\left\{ \varphi_{1}\left(\left\langle u,\cdot\right\rangle \right):\;u\in\mathbb{R}^{d}\right\} $
has dense span in $L^{2}\left({\cal P}_{X}\right)$ (the space of
square integrable functions w.r.t. the measure ${\cal P}_{X}$, which
is the distribution of the input $X$).
\end{enumerate}
\end{assumption}" 
\item Convergence: There exist functions $\bar{w}_{1}$, $\bar{w}_{2}$
and $\bar{w}_{3}$ such that as $t\to\infty$, there exists a coupling
$\pi_{t}$ of $\rho^{1}\times\rho^{2}\times\rho^{3}$ and itself such
that 
\begin{align*}
\int\left(1+\left|\bar{w}_{3}(u_{3})\right|\right)\left|\bar{w}_{3}(u_{3})\right|\left|\bar{w}_{2}(u_{1},u_{2},u_{3})\right|\left|w_{1}^{*}(t,u_{1}')-\bar{w}_{1}(u_{1})\right|d\pi_{t}(u_{1},u_{2},u_{3},u_{1}',u_{2}',u_{3}') & \to0,\\
\int\left(1+\left|\bar{w}_{3}(u_{3})\right|\right)\left|\bar{w}_{3}(u_{3})\right|\left|w_{2}^{*}\left(t,u_{1}',u_{2}',u_{3}'\right)-\bar{w}_{2}\left(u_{1},u_{2},u_{3}\right)\right|d\pi_{t}(u_{1},u_{2},u_{3},u_{1}',u_{2}',u_{3}') & \to0,\\
\int\left(1+\left|\bar{w}_{3}(u_{3})\right|\right)\left|w_{3}^{*}\left(t,u_{3}'\right)-\bar{w}_{3}(u_{3})\right|d\pi_{t}(u_{1},u_{2},u_{3},u_{1}',u_{2}',u_{3}') & \to0.
\end{align*}
*(NP also use an \(ess\,sup\) bound on \(\partial_t w\) to pass from finite-\(t\) identities to the limit in the dense-span / orthogonality step.)*

**Low-rank schematic** (same weighted coupled-\(L^1\) idea, \(r\) channels):
\begin{align}
\label{eq:lr-coupling-schematic}
  \mathbb{E}_{\pi_{t}}\Big[
    \Big(\prod_{j=1}^{r}\psi_{j}(\cdots)\Big)\,\psi_{0}(\cdots)\,
    \bigl(1+|\bar w_{2}|\bigr)\,|\bar w_{2}|
    \sum_{k=1}^{r}
    |\bar w_{1}(\cdot,k)|\,
    \bigl|w_{1}^{*}(t,\cdot,k)-\bar w_{1}(\cdot,k)\bigr|
  \Big]
  \longrightarrow 0,
\end{align}

**Remark (NP: a sufficient route via \(W_4\)).** The three weighted \(\int(\cdots)\,d\pi_t\to 0\) conditions are **implied** if the tuple \((w_{1}^{*}(t,\cdot),w_{2}^{*}(t,\cdot,\cdot,\cdot),w_{3}^{*}(t,\cdot))\) converges to \((\bar{w}_{1},\bar{w}_{2},\bar{w}_{3})\) in **Wasserstein-\(4\)** distance,
\begin{align*}
 & \inf_{\pi}\int\Big(|w_{1}^{*}(t,u_{1}')-\bar{w}_{1}(u_{1})|^{4}+|w_{2}^{*}(t,u_{1}',u_{2}',u_{3}')-\bar{w}_{2}(u_{1},u_{2},u_{3})|^{4}\\
 & \qquad+|w_{3}^{*}(t,u_{3}')-\bar{w}_{3}(u_{3})|^{4}\Big)d\pi(u_{1},u_{2},u_{3},u_{1}',u_{2}',u_{3}')\to0,
\end{align*}
where the infimum is over couplings \(\pi\) of \(\rho^{1}\times\rho^{2}\times\rho^{3}\) with itself, **together with** initialization / regularity and **a priori mean-field bounds** on weights (NP’s bounds on the MF solution — e.g.\ their “bounds MF a priori” lemma). **Hölder** pairs **\(L^4\)** coupling errors with **\(L^{4/3}\)** (or higher-moment) control of the polynomial weights \((1+|\bar w_3|)|\bar w_3|\), \(|\bar w_2|\), so the **weighted \(L^1\)** integrals vanish. Thus **\(W_4\)** is a clean **sufficient** technical input; it is still **not** the same statement as “the fourth moments of \(w\) converge to those of \(\bar w\)” unless you unpack the transport definition.

---

**NEW EXPLANATION (convergence assumption — what it is, what it is not)**

- **Source.** The three \(\int(\cdots)\,d\pi_t\to 0\) lines are Nguyen \& Pham’s *three-layer* convergence-to-limit-point condition (coupling \(\pi_t\) on \(\rho^1\times\rho^2\times\rho^3\) with itself). The prefactor \((1+|\bar w_3|)|\bar w_3|\) and the extra \(|\bar w_2|\) in the first line are exactly the weights that appear when the backward–forward Lipschitz chain for \(\Delta_2^{H*}\) is integrated under \(\pi_t\); they are not chosen for aesthetic reasons.

- **Not a “moment convergence” statement by itself.** A classical moment would be something like \(\mathbb{E}[|U|^p]\to c\) or uniform integrability of a single coordinate. Here you have **limits of integrals of products**: a **nonnegative weight** built from **limit** weights \((\bar w_2,\bar w_3)\) times a **coupled gap** \(|w^*(t,\cdot')-\bar w(\cdot)|\), integrated under \(\pi_t\). The right name is **coupled, weight-scaled \(L^1\) vanishing** (a **proof-tailored transport** condition), not “moments converge.”

- **How it relates to Wasserstein / \(W_4\).** With **uniformly bounded** weights, these integrals are comparable to **\(W_1\)** transport of \((w_1,w_2,w_3)\). With **unbounded** weights, NP use **\(W_4\)** convergence of the coupled triple **plus** **a priori** MF moments to deduce the weighted \(L^1\) integrals (Hölder). So: **\(W_4\)** is **one standard sufficient gate**, not the definition of the displayed assumption.

- **Stronger than weak convergence.** Weak convergence only gives \(\int f\,d\mu_t\to\int f\,d\bar\mu\) for bounded continuous \(f\). This assumption asks for **specific** bilinear-weighted gaps to vanish, matching the **factorization** of constants in the global-convergence proof.

- **`ess\,sup_t|\partial_t w|\to 0\).** Parallel condition in NP: ensures the finite-\(t\) gradient-flow identities **carry** to the limit \(\bar w\) without a leftover time derivative in the orthogonality step.

- **Our low-rank translation.** Replace the single \(\bar w_1\) gap by a **\(\max_k\) or \(\sum_k\)** channel gap; the schematic \(\mathbb{E}_{\pi_t}[(\cdots)\sum_k|\bar w_1(\cdot,k)|\,|w_1^*-\bar w_1|]\to 0\) is the **\(r\)-channel** version of the same weighted coupled-\(L^1\) object (again **deducible** from a \(W_p\) statement + a priori bounds if you choose \(p\) to match Hölder).


Nguyen & Pham, 2023 established the global convergence of multilayer neural networks trained under stochastic gradient descent (SGD). Is your convergence analysis also based on SGD

yes, in the mean field sgd converges toward a mean field ODE solution well provided that discretization is small enough 
"While the global optimality is established for the mean-field ODE (which corresponds to continuous-time gradient descent on the population loss), the framework rigorously connects this limit to the actual discrete-time SGD dynamics through Proposition 23 of Nguyen & Pham (2023). Our quantitative bound (Theorem 4) explicitly accounts for the $O(\sqrt{\epsilon})$ discretization error and $O(1/\sqrt{n})$ sampling error introduced by SGD"

































5)

Theorem 4.1 is limited to a two-point data distribution, and although an informal argument is made for when its conditions hold, it could benefit from a more developed treatment.

**Answer.** We will add a **developed treatment in the camera-ready**: a standalone **Remark (mechanism behind the channel spike)** right after Theorem~4.1, spelling out the **mean-field feedback loop** in closed form (notation aligned with the paper): $H_2(c_2;x,W)=\sum_{j=1}^r L_{c_2,j} f_j(x;W)$, ReLU gating $\varphi_2'(H_2)=\mathbf{1}\{H_2>0\}$, and how the channel backward object $B_k^{(2)}$ couples **mixing weights** $L_{C_2,k}$, **gates**, and **upstream** factors. The point is that when channel $k$ **dominates**, neurons with the gate on are **positively aligned** with large $|L_{c_2,k}|$, which **amplifies** the drift on $w_1(\cdot,k)$ and hence on $f_k$---a **reinforcing loop**. Near the **null** predictor at initialization, each $f_k$ grows toward a stabilizing magnitude; the **spike** is the visible transient signature. Theorem~4.1 remains the **minimal rigorous** instance; this remark is the **general mechanism** (dimension-agnostic at the ODE level). **LaTeX** for the remark is in the \texttt{latex} code block under \textbf{Feature-learning mechanism} earlier in this file (same section as the Theorem~4.1 reviewer thread).

. Convergence guarentees. Are there conditions under which convergence is guaranteed for RF-LR models (those mentioned in section 3.5 or otherwise)?

In the large width setting the convergence guarantees holds , up to a reparameterization mean field and NTK parameterization
are just scaled versions of each other ; which means mean field regime ensure convergence after a long time and features learned
ntk param is just kernel regression

concerning the ntk theory for low rank and/or random features models : the answer is yes, this is work currently in review,
rf-lr are converging in the ntk regime


2. High-dimension extensions. Does the spike learning mechanism extend to higher dimensions with overlapping spikes in feature space?

**Answer.** Yes in substance: the **mechanism** (channel dominance $\leftrightarrow$ amplified backward signal through the mixing matrix and ReLU gates; cf.\ the feature-learning mechanism remark drafted above / the two-point toy) is **dimension-agnostic**---it is the same algebra of $H_2=\sum_k L_{c_2,k} f_k$ and the ODE for each $f_k$. What is **not** fully written in closed form for general $d$ is a **tractable theorem** isolating non-overlapping spikes; high-dimensional inputs typically induce **overlapping** channel contributions in feature space, so the **rigorous** two-point explanation is a deliberately minimal caricature. Empirically and heuristically, the same feedback loop operates in higher $d$; overlapping spikes mainly complicate **bookkeeping and visualization**, not the core optimization picture.

3. Sensititivity of randomness of fixed feature maps. How sensitive are the convergence guarantees and empirical performance to the initial draw of the frozen feature maps for finite width networks?

in fact the convergence guarantees only requires that mixing matrices W are frozen integers, it can be drawn 
from any distribution with finite support ! in practice we use gaussian for simplicity,
for positive values W (which means non negative factorization of weights) neural collapse is avoided and convergence
guaranteed for relu networks wigh high probability in r (which means for a lot of x, gradients of weights at are 0) 











TODO : 
what to do : extend feature learning theorem, understand more to explain better the shift in the neural collapse originally, trying to prove this story of relu high probability ; need to run experiments on higher d and r to see a phase transition maybe sqrt, explain the goodnotes by hand, see the collapse without frozen weights but keeping low rank, deciphering feature learning for mean field models (mu-p), elaborate a lot on low/high frequency bias for low rank/full rank, run a lot of other experiments showing low rank do better, run other sweep for momentum, see convergence guaranteees with former bach work, trying to run variance mean sweep experiements for the random features, run experiements in high dimension with overlapping spikes, we can take N=10 maybe






GOODNOTES (tablet sketch $\rightarrow$ LaTeX; notation aligned with the Nguyen--Pham coupling step around lines 478--485 above).

**Setup ($r{+}1$ coupled mean-field equations).** Picard / fixed-point step as on pp.~37--38 of \NP{}; $W_2$ convergence via their Lemma~8-type estimate. Low-rank specialization ($L{=}2$ toy in margin): $f_1$ fixed, $f_2{=}w_1$. Red margin: $W_2(U_1,U_2,U_3)\to W_2(U_2)$ under pushforwards of $\mathrm{Law}(w,c,u)$ --- this is the **new hypothesis** replacing the fully-connected contraction.

**Main coupling term (integral w.r.t.\ $\pi_t$).** Bounded first-layer gates $\varphi^{(1,1)},\ldots,\varphi^{(1,r)}$ and second-layer $\varphi^{(2)}$ (cf.\ Assumption~3 / bounded activations). The low-rank analogue of the chain after ``translates to'' is schematically:
\begin{align*}
& \mathbb{E}_{\pi_t}\Big[
\Big(\prod_{k=1}^{r}\varphi^{(1,k)}(\cdots)\Big)\,\varphi^{(2)}(\cdots)\,
\bigl(1+\lvert \bar w_{2}(U_{2})\rvert\bigr)\,\lvert \bar w_{2}(U_{2})\rvert
\sum_{k=1}^{r}
\bigl\lvert \bar w_{1}(u,v_{k})\bigr\rvert\,
\bigl\lvert \bar w_{1}(h,v,v_{k})-\bar w_{1}(u,v_{k})\bigr\rvert
\Big]
\;\longrightarrow\; 0,
\end{align*}
where $\pi_t$ couples finite-$N$ and mean-field particles (same role as $(U_{3},U_{3}')\sim\pi_{t}$ in the display above). The arguments $(h,u,v,v_k)$ are the neuronal labels from the sketch; in the write-up they should match the explicit $(U_{1},U_{1}',C_k)$ notation of Eq.~(1) / the neuronal embedding section. The blank ``$\displaystyle\int (\cdots)\,d\pi_t\to 0$'' in the pad is this expectation.

**Time regularity (Sect.~7.4.2 style).** For each channel $k=1,\ldots,r$,
\[
\mathrm{ess\,sup}_{t}\Bigl\lvert \frac{\partial}{\partial t}w_{1}(t,U_{k})\Bigr\rvert \;\to\; 0
\]
(``page~42 $\le K\mathbb{E}_{\pi_t}[\cdots]$'' in the notes). The delicate factor is $\varphi_{2}'(H_{2})$; bounds use $\Delta_{2}\le K_{t}(1+\lvert U_{3}\rvert)$, $L_{s}\le K(1+\Delta_{3}^{H^{*}})$, and backward maps Lipschitz/bilinear with constant $K(\lvert a\rvert+\lvert b\rvert)(1+\lvert\Delta\rvert)$.

**Bi-Lipschitz / backward term.** The sum $\sum_{k=1}^{r}(\cdots)$ is controlled under the bi-Lipschitz conditions on backward quantities (blue arrow in notes).

**ReLU trick (boxed).** With $s=\sum_{i=1}^{r}x_{i}$ and $s'=\sum_{i=1}^{r}y_{i}$,
\begin{equation}
\label{eq:goodnotes-relu-sum}
\left\lvert \mathrm{ReLU}(s)-\mathrm{ReLU}(s')\right\rvert
\;\le\; \left\lvert s-s'\right\rvert
\;\le\; \sum_{i=1}^{r}\lvert x_{i}-y_{i}\rvert.
\end{equation}
(Handwritten ``$\le r\sum_i|x_i-y_i|$'' is either a slack bookkeeping bound or a typo; \eqref{eq:goodnotes-relu-sum} is the sharp triangle-inequality step.) Green margin: this is the **structural reason $r$ coupled equations help** --- without ReLU one gets a cleaner $\sum_{k}\mathbb{E}_{c_{k}}[\cdots]\le K\lvert c_{s2}\rvert$-type bound; with ReLU one needs \eqref{eq:goodnotes-relu-sum} plus **compact support / boundedness** of pre-activations to close constants.

**What to paste into `answers.tex` / appendix.** One paragraph + \eqref{eq:goodnotes-relu-sum} + the $\mathbb{E}_{\pi_t}[\cdots]\to 0$ display after fixing $(u,h,v,v_k)$ to official notation.
