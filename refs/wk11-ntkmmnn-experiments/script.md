to introduce all of what i've done in this i would say before that there MMNNs which stands
for multi componenet neural network. this type of network is supported by 4 ideas, 1st we reduce the number of parameters by using a low rank structure,
not inside
the weight because this will give us a manifold of low rank weight matrices, but instead
also freezing the weights of 1 matrix in the decomposition, that is the W matrix, and that corresponds then to 
a decomposition in a random basis manner. and this is also supported by approximation theory results
for this kind of basis. also there is an inherent interpretation that is given by the
frequency learning in a spatial localization, that is like in wavelets but here we domain decompose our input space into 
batches of same frequencies to learn unit order coefficients and functions on those domains.

so we have less weights by linear weight scaling in width, low rank interpretation for feature learning
and then what PR zhang shown was that the optimization process was also better for this kind of network, that is the main question we ask is 
to understand theoretically optimization landscape results we got empirically that are sometimes astonishing, and undersatnd it globally and locally. My idea was to use the ntk to tackle it locally during the last weeks. 


The ntk for MMMN has to be tackled very differently because there is 2 source of randomness, 1 that will vary through training that is
A and c, and the random bases W b. The whole thing makes, when the width goes to infty, to have a random function that is a gaussian process
or a composition of gaussian processes by concatening low rank MMNNs layers.

This randomness in the function itself gives us a randomness in the NTK, but that is perfectly computable in expectations,
so to analyze the optim landscape locally, i have undersatnd the std scaling empirically to see if i can go further in the
theoretical analysis without making unreliable assumptions.
And it appeared that my intuition was correct, that is the std decay very fastly, so this idea is supported
by this graph, where you see a scaling law in log space of NTK distributions for several configs when i grow the internal rank a bit
this std decay can be computed because this basically comes from curse of dimensionnality from the weights
scalar products with entries, but first I wanted to be sure by experiments that my math investigations path is
correct before tackling integrations over gaussian measures in high dimensionnal spaces that I do by hand computations



another very important thing is the 