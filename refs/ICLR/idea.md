i want to build scaling law and feature learning for low rank models in scientific computing
and loss landscape analysis :




loss landscape : plateau then sharp then plateau etc .. each corresponding to a frequency
several directions seen as noise at the beginning , a lot of parameters directions
leads to the same result (maybe low rank reduce it)
at the beginning the function is a constant and loss constant and after a lot of epoch that is 
around 10-20 epoch  for lr e-2 , 1000 for lr e-4 , so this plateau analysis is consistent
but the leap analysis gives you a martingale then go deep into the hole


also by doing LR reduction we go deep into a hole everytime, so after plateaus the landscape for grokking
looks much like some holes are hidden and too big lr avoid them
(large scale scientific experiments should confirm that on simple cases)


it seems that Adam training can be proficient in that perspective, having a jump at EOS allows
to get into a sharp but small hole (see picture) but having a shape like 3 local minima and the global in the middle
if doing a jump is possible at EOS we go in the hole (maybe this is explanation of how adam works so much)

adam training is very unstable
the lower part o of the landscape is very not like the other part, much more sensible to N samples ; it's weird
of instable training is symmetrical, very long training is weird


machine precision analysis is different 




scaling law : with sgd/adam
adam analysis for EOS, maybe EOS is less , or hessian analysis better (more tractable at least)
scaling wrt depth/width, N samples and r ; +freq and lr ; especially lr small for large scale experiments
and width at least N_samples to have expectations well computed (can be larger but not necesary)

at least we have empirically a good result, layer L have 2L spike which is not the case for MLPs
this is an incredible feature learning mechanism (and large scale scientific experiments should confirm this fact)


also having leap hierarchical fourier spectrum leads to better learning, how having low then high allow better
learning than high frequency only (impossible nearly) 

we should replicate the scaling law openAI paper, or give premices



feature learning : 
another fact is that partial functions separate their learning mechanism at different localization, we
already have the paper that deals with a small explanation in this perspective, but this is an explanation
in a toy model setting, in practice we need more log ratio analysis


we see experimentally this 2*l spike for layer L
the rank effect should be taken into account, it can gives exponentially better results








scaling law paper openai : 
too general and no microstructure of learning focus
dataset size, param count, training time ; maximum likelihood loss

"Transformerperformancedependsveryweaklyontheshapeparametersnlayer,nheads,anddffwhenwehold
thetotalnon-embeddingparametercountNfixed."





optimizers are a way to analyze the local loss landscape

our scaling law : 
critical batch size
plot  test loss wrt epoch/compute for different number of param
batch size also


we gonna compare central flows and show it's better ?
doing it on classical mmnn tasks


Finality : 
across many pde solving benchmark do low rank performs better ?
this is something that should be runned, especially just with replacing baselines by low rank ones

goal is to write a lot of independent .py files to make those experiments runnable on different computers at scale

training curve typology


what we gonna sell is feature learning , spike learning, landscape

can we superpose cosine trianing curves inside the full training curves ? self similarity of the loss landscape


4 submissions : 1 on the landscape, 1 on central flow, 1 on feature learning, 1 on scaling law for depth/width etc ..
1 on NTK scaling law to ensure trainability, NTK scaling converge to 1
1 on fourier training




plateau escape time (already have a good picture f3N384bs1L2 verify this plateau hypothesis) 2k epoch e-4, 20 e-2
log ratio trajectory and histogram, 
then log ratio for full training



scaling law for losses



1st experiment scaling law in epoch to escape = 1/lr ; this is coherent and sqrt(bs) because lower variance still
but it depends on the hierarchical structure 
we got 0.75 because there is a bigger dependency somewhere ? between 1/N and 1/sqrt(N) bias variance compromise ?

there is an optimal batch size that 

is the landscape autosimilar ? yes 


small batch sizes escape very well for large factor and low rank 5
sgd wobling effect due to sgd not gd