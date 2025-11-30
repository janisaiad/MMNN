


not only trained a lot but monitored training and a lot of careful hand made optimization for training hyperparams to be sure to have a confident baseline and a lot of experiments, so i'll what whas my training setup, the qualitative results we can have a priori and a posteriori, and what reseacrh direction this gives us in theory
and how to further experimentally

my first idea was to understand the scaling law in the original paper, to undersatnd if it was the descent phase that was convex, or the Adam optimizer
learning rate exponential decay that gives this power law, spoiler this is due to Adam convergence in a very convex landscape in the descent phase after searching phase



for usual regression tasks in 1d, it appears that normalized highly oscillating functions (with oscillations of frequency of order 100) can still
be approximated perfectly with a e-3 MSE, so super convergence begin at approx e-2


are MSE and max correlated ? are we in the interpolation regime or the kernel regime ?
do the solution recover all fourier modes ? is there an inductive bias of learning small absolute values entry points before ?

how convex is the landscape in the search phase, the descent phase ? do NTK describe well the traing ? can we identify those phases with the NTK and loss curve ?

are MMNN low rank functions performing dictionary learning ? do depth matter more than width ? is this the same as the approximation result


are fourier based sobolev loss more relevant to monitor and train on those tasks ?


to give a hint, baseline is 666*36*5 + 666 = 120k parameters, only the half trained, 1000 functions, MLP of 175 width and 4 depth


- 2 or 3 layers MMNNs are hard to train and have a lot of inductive bias in the gradient descent

- 1st layer and 2nd layer behavior is highly different from the others, real dichotomy, which means depper means better landscape
- we can see only 5 to 7 dictionary functions learned, some functions learned are very redundant
- we can see spiked localized functions in large width, and low ranks combine them


- MMNNs inductive bias give relu spiked positive functions that select parts of the input interval with particular frequencies


a priori : search phase is erratic so NTK should move quickly after init, for all configs and width
NTK more PSD for large width with convex descent even in low rank
in large width (1000 for 1000 samples) its much more convex so the overparametrized setting works well, and 
very small smallest eig for large width

for 5*1000 we have feature learning and convex descent

the only reason that makes the global minima not attained was too long training time for 10 1024 50, with a very long search phase, ntk not moving

MMNNs


SGD don't work at all



fourier based loss is more relevant, for training and monitoring


what those experiences shown : 
- NTK not a good metric for MMNNs search phase, but spectrum very stable in the descent phase, becomes PSD (convex), generic behavior even in small width
- NTK predicts well PSD-ness and near convergence
- training almost everytime converge even in long time, 5x longer for 10 times more width/depth there is a huge dilema


to do for the math :
-  theoretical study landscape convexity near the MMNN decomposition (because even though there could be many global minimas at the same loss level in the overparameterized regime, they can have same flatness/sharpness due to random basis fixed weight norm W/b (only A c to study), see paper lexing ying)
- evaluate dictionary complexity
- sobolev loss



to do experimentally : 
- plots in parameter budgets, epochs, loss, convexity and conduct large scaling law experiments (like in openai paper)
- test my sobolev fourier loss, pinn type loss also
- NTK evolution in finite width correction with my recursive formula and finite width correction (tractable since linear scaling for budget now)




- ordering them by absolute value for the last layer


describe criteria
deep very idea
2d functions


redundants
symmetry breaking low rank
categorizer similarity, more similarity

describe similarity inner product

criterion for the best low rank

2n th spikes after n-th layer

layer by layer more spikes and higher frequencies


phenomena sgd after adam
summarize some important things