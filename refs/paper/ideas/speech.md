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
