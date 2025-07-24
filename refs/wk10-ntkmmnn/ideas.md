
\begin{itemize}
    \item Soft Structure, TP like, Sparsity $\beta$ (probabilistic def).
    \item $Y_{bridge}$
    \item \textit{ILLEGIBLE} (same as Kernel?)
    \item If there is no norm or if this norm is identical to \textit{ILLEGIBLE}?
    \item Narrow residual? Usual Structure $\rightarrow$ NTK. Can we find the Tangent Angle \textit{ILLEGIBLE}?
    \item ``something should come out here, I think still with $1/p_n$''
    \item ``to verify with $1/p_n$ or 1''
    \item ``not to be calculated with $\nabla\theta$ !! apparently !!''
    \item ``put recursion! No need \textit{ILLEGIBLE}, always \textbf{Always}''
    \item ``Watch out for Q, I''
    \item ``If we let n tend to infinity, we hope that... so we have a recursive relationship''
    \item Possible literary references: ``Jac Silverstein and others, and \textit{ILLEGIBLE} to calculate the exact limit (\textit{TejoK}?) -- random scaling laws''
\end{itemize}





make a small explanation in the introduction to elaborate on why it's so useful (with MMNNS theorems) 
but also that training on low rank manifolds is hard, and not interpretable that much, without a lot of non linearities involved
for same number of parameters, but useful for compression and distillation (without SETOL setup)

1 thing to remark is that low rank makes ntk to explode and grow more, when ranks grow, ntk is diminished, which 
can be counter intuitive since we don't normalize w in the relu (we should because of curse of dim, or to approximate along
a direction and separate well at many scale)

for uniform weights and b to see, because of no gaussianity but we can compute many things with integrals (or symbolic stuffs)
also weights can be sampled non isotropically for Q, b,, or by changing data by Q only, with spectral radius 1 (or other with
sigma_A to respect the EOC)

We can try to compute the scaling wrt L to see how the optimization process works
and also compare with the hessian (for the surmrise)

also the jax code for computations can be improved a lot

surmrise to see on MMNNs

so the idea is to use a block structure (like on the anatomy of attention) with internal dim going to infty
and that can be useful without a lot of params, for instance nlog(n) attentions , or transformers (but very costly) to see under the
ntk's pespective that the optmization goes well, or has gaussian local structure (big conjecture for any TP)

in fact the formalism is to use TP and to choose what u train


le fait est que maintenant on peut comparere differentes tailles de MMNNs face à des MLP pour vraiment avoir un tableau de nombres solide 

on peut inverser le training de w et a, on peut comparer dans quel regime on est dans le ntk pour mmnnn car on peut raccrocher la theorie mtn
on peut etuider le ntk local aussi pour voir si le surmrise verifie la meme propriété car maitnenant on est en accord avec le ntk
avec moins de paramètres maintenant, mais beaucoup plus d'aléatoire pour mmmnns

il reste à comparer pour 2 layers de mmnns avec le structure I+J car là on a fait qu'une layer
je pense uassi que c'est le norme(w)² dans l'espereance qui peut changer les choses, selon certaines dimensions internes
et selon comment on veut approximer, on unscale w et on peut faire diverger ça pour obtenir un meilleur NTK
on peut faire un developpement en serie entiere du noyau zonal comme data_ntk ou alors avec funk hecke on calcule les produtits scalaire 
et on trouve le spectre, on essaye comme ça pour voir ce que ça donne (c'est aussi sobolev friendly et à noter)

on peut aussi se lancer dans la RMT ou même dans la preuve de terjek que l'on peut adapter avec notre noyau qui est maintenant une autre somme, on doit faire un DL pour ça et trouver s'il y a un scaling lineaire ou exponentiel pour une preuve  !!




de maniere plus lointaine on peut vraiment se demander ce qui se passer si on gele certains neurones quiioi, et on formalise tout ça 
dans une theorie de tensor programs ave ceratins parametres random, à la TP de greg yang ou de abbott 
faire attention à la parametrisation mup ou pas pour beta*n ou non car ça compte pour le gradient 

on doit aussi comparer le NTK sans biais pour isoler directment la dependance en norme(w²)

je sens qu'il va falloir faire de la rmt déjà pour le NTK MLP pour s'assurer en fonction de sdonnées de commment ça converge, et d'avoir les stats locales à la Baskerville
et de continuer avec les MMNNs
je pense qu'il va falloir discuter assez longtemps par mail aussi avec shijun zhang sur ça, car il doit avoir pleind 'idées, il va falloir lire tout son papier dans les details aussi

il y a aussi l'analyse à la bietti à faire sur le DL sur les bords du noyau en série de puiseux pour obtenir les propriétés approximatives du noyau et de son spectre pour l'opérateur

la conjecture du spectre divisé par 2 est en marche, il va aussi falloir faire une analyse sans biais

on peut asusi déjà analyser le scaling de la recurrence car c'est ça selon la preuve de terjek qui va fonctionner

les transformers, gros truc, à analyser, là ca va etre different et nlog(n) transformers pour pouvoir comparer

faut commencer la redeaction du rapport et des preuves, puis aussi re recuperer le calcul hessienne pour les noyaux d'ordre supérieurs pour le rapport ça va etre important


on doit aussi balancer toutes les experiences en loi uniforme

on peut essayer d'oublier les biais et de les train aussi dans les 2 cas peut etre je sais pas

il y a aussi l'intuition à avoir parce que pas evident que si la loi est pas uniforme, en fait on choisit la direction mais aussi le scaling dessus

pour creer un MMNN il faut 2 activations oppposées, et donc 2 directions opposées avec un biais comme il faut ? je vois ça

faire un DL pour beta petit aussi car on peut se le permettre je pense

plus loin il faut un DSRN de MMNN et on calcule le NTK dessus wow
mais je pense un neural tangents ou on choisit qui on train c'est super parfait en fait, et je crois que c'est déjà fait !!!
bah oui car dans jax on peut preciser qui on train à priori ?

en fait je pense aussi qu'il y a toute une wavelet analysis à faire !!, sauf que les wavelets en high dim ça croit exponentiellement
d'ou le nouveau truc


for sine and relu, low ranks are prefered

the main thing i get is that the ntk is near diagonal when we concatenate layers

PsinTU est à faire dans le futur, un peu complexe à rajouter dans le ntk, sintu et relu c'est un nombre constant de layer à approximer



we also need to implement and see the hessian in jax


avec toutes les observatoins sur la variance et sur l'alea du ntk il y a un compromis
une matrice ntk qui traite tous les elemnts pareils, aps trop random, à quel prix ?
aussi on voit qu'on divise le spectre par 2 (en parler dans le script) car on divise le trianing set par 2
ce qui est deja une grande avancée

on a tojours pas parlé de ce qui se passe si on train tout le monde, et de l'approche mean field en w sur les particules
ou peut etre mean field sur a pour voir penadnt le training