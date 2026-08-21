# Limite continue, polygones, entraînement, noyaux et oscillations

## 1. Quand la somme devient une intégrale

Pour `n` tokens sur la sphère, introduisons leur mesure empirique

\[
\mu_t^n=\frac1n\sum_{i=1}^n\delta_{x_i(t)}.
\]

Pour une matrice symétrique `V`, un noyau positif `g` et une attention
normalisée ligne par ligne, la vitesse d'un token situé en `x` s'écrit

\[
v_g^{\mathrm N}[\mu](x)
=P_x^\perp
\frac{\int g(x^\top Vy)Vy\,d\mu(y)}
     {\int g(x^\top Vy)\,d\mu(y)}.
\]

Sans le dénominateur de l'attention, mais toujours avec projection sur la
sphère,

\[
v_g^{\mathrm U}[\mu](x)
=P_x^\perp\int g(x^\top Vy)Vy\,d\mu(y).
\]

Quand `n` tend vers l'infini, la collection d'ODE devient l'équation de
transport non locale

\[
\partial_t\mu_t+\operatorname{div}_{\mathbb S}
(\mu_t v_g[\mu_t])=0.
\]

Cela ne signifie pas que les tokens disparaissent. La PDE dit simplement que la
masse de tokens est transportée par un champ calculé à partir de toute la
distribution. Pour un noyau suffisamment régulier et un support compact, la
stabilité en distance de Wasserstein donne précisément la convergence de la
mesure empirique vers cette PDE.

Notre audit utilise une densité à trois bosses autour du polygone mixte et
compare la somme aléatoire avec une quadrature à 8192 points. L'erreur sur la
vitesse suit

\[
\operatorname{erreur}(n)\simeq n^{-0.494},
\]

en accord avec la fluctuation Monte-Carlo `n^{-1/2}` :

| tokens | erreur de vitesse |
|---:|---:|
| 16 | 0.289 |
| 64 | 0.146 |
| 256 | 0.071 |
| 1024 | 0.038 |

### Quel rôle joue Wasserstein ?

Wasserstein fournit d'abord une distance entre deux nuages de tokens qui tient
compte du coût de déplacement de leur masse. L'équation de transport ci-dessus
vit naturellement dans cet espace. Mais toute équation de transport n'est pas
automatiquement une descente de gradient Wasserstein ordinaire.

- Pour l'attention non normalisée et symétrique, si `G'=g`, la dynamique est la
  montée du potentiel

  \[
  \mathcal E_G(\mu)=\frac12\iint G(x^\top Vy)\,d\mu(x)d\mu(y)
  \]

  dans la géométrie Wasserstein usuelle, à un choix de signe près.
- Le dénominateur de l'attention multiplie la mobilité de chaque token par un
  facteur dépendant de `x` et de `mu`. On obtient une géométrie Wasserstein
  modifiée, parfois appelée *twisted* ou Hessienne.
- Sinkhorn symétrise autrement les échanges et peut restaurer une structure de
  gradient Wasserstein plus standard.

Ce sont les distinctions faites, sous des conventions légèrement différentes,
par Rigollet et par Castin–Ablin–Carrillo–Peyré.

## 2. Ce que deviennent nos polygones dans la limite continue

Une mesure atomique

\[
\mu=\sum_{a=1}^q p_a\delta_{w_a},\qquad p_a>0,\quad\sum_a p_a=1,
\]

est une solution exacte de la PDE déterministe si les positions `w_a` suivent
le système à `q` particules pondérées. Elle est stationnaire exactement lorsque

\[
P_{w_a}^\perp
\sum_{b=1}^q p_b g(w_a^\top Vw_b)Vw_b=0
\qquad\text{pour tout }a.
\]

Les multiplicités entières de notre classification deviennent donc simplement
des masses réelles `p_a`. Le passage à l'intégrale ne détruit pas le polygone :
il agrandit au contraire la famille, car les rapports de masses ne sont plus
contraints à être rationnels.

Pour notre famille mixte à trois groupes

\[
u,\qquad qu+\sqrt{1-q^2}v,\qquad qu-\sqrt{1-q^2}v,
\]

le rapport entier `m/k` de l'équation finie est remplacé par `p_0/p_+`, avec
`p_+=p_-`. Les polygones réguliers dans un espace propre multiple restent aussi
des équilibres pour tout noyau radial, par symétrie.

Deux réserves sont essentielles :

1. Une distribution continue proche du polygone forme des bosses qui peuvent se
   contracter, fusionner ou se déplacer; elle n'est pas exactement une somme de
   Dirac.
2. Avec le bruit brownien de la PDE de Fokker–Planck, chaque Dirac s'élargit
   immédiatement. À faible bruit, un polygone stable devient un ensemble de
   bosses métastables; à bruit fort, elles peuvent disparaître.

## 3. Comparaison précise avec arXiv:2605.07772

Le papier d'Isobe–Inoue–Imaizumi ne forme pas les matrices d'attention. Il fixe
une matrice `A` symétrique définie positive avec plus grande valeur propre simple,
utilise l'attention symétrique **non normalisée**, ajoute du bruit, et entraîne
seulement un FFN dépendant de la profondeur. Dans ce cadre, l'énergie possède
deux puits principaux près de `+e1` et `-e1`.

Son résultat est un problème de contrôle optimal :

1. l'attention rassemble d'abord les tokens;
2. la distribution reste près d'un puits durant la majorité des couches, car
   employer le FFN partout coûterait cher;
3. le FFN formé agit fortement près de la sortie pour diminuer la perte finale.

Le papier appelle cela clustering, plateau *turnpike*, puis échappement terminal.
Il ne classe donc pas nos polygones indéfinis : l'hypothèse `A>0` exclut le
mécanisme positif/négatif `V=diag(2,-3)` qui les stabilise.

### Une première étape de Muon

Au point initial nul, le papier obtient pour chaque profondeur

\[
G_t=\nabla J(0)_t
=\int \nabla\phi_t(x)\sigma(x)^\top\,d\bar\mu(x),
\qquad W_t^{\rm GD}=-\alpha G_t.
\]

Après discrétisation de la profondeur, si
`G_k=U_k Sigma_k V_k^T`, une étape de Muon idéalisée donne plutôt

\[
W_k^{\rm Muon}=-\eta U_kV_k^\top.
\]

La première étape ne dépend donc pas réellement du momentum : celui-ci ne fait
que multiplier `G_k` par un scalaire, ensuite effacé par le facteur polaire.
Muon conserve les sous-espaces singuliers mais remplace toutes les valeurs
singulières non nulles par une taille comparable. Le gain infinitésimal devient

\[
\delta J_{\rm Muon}=-\eta\sum_k\|G_k\|_*,
\]

au lieu de `-alpha sum_k ||G_k||_F^2` pour GD. La norme nucléaire apparaît
parce que `U_k V_k^T` est la direction la plus descendante sous une contrainte
de norme opérateur.

Dans notre analogue polygonal à 40 tranches de profondeur, la norme du gradient
de la dernière tranche est `7.11e11` fois celle de la première. GD hérite de
cette enveloppe et ne déplace le polygone qu'à la toute fin. Muon normalise
l'enveloppe tranche par tranche : à petit budget, il dépense une grande partie
du pas dans des couches précoces dont l'attracteur efface l'effet; à budget plus
fort, il détruit le plateau et commence l'échappement beaucoup plus tôt.

À norme `L2` totale égale à `0.3`, la perte terminale passe de `0.37839` à
`0.28547` avec GD, contre `0.35385` avec le facteur polaire exact et `0.35323`
avec cinq itérations de Newton--Schulz. Ce n'est pas un classement universel des
optimiseurs : la comparaison favorise naturellement GD puisqu'elle impose la
géométrie `L2` du théorème. Sous une contrainte spectrale par matrice, Muon est au
contraire la direction optimale.

Le seuil numérique de Newton--Schulz est important. Il évite de transformer un
gradient réellement nul en grand déplacement et crée en pratique une frontière
entre des premières couches presque ignorées et les couches dont le gradient est
orthogonalisé. Le minorant d'échappement du papier ne se transfère donc pas tel
quel : son Gramien linéaire `B B*` est remplacé par la réponse non linéaire
`B Polar(B* phi)`.

### Extension entraînée sur notre polygone

Nous avons entraîné un petit contrôle FFN angulaire, linéaire en cinq features de
Fourier, pour déplacer le polygone mixte stable vers une cible tournée de `0.9`
radian. La pénalisation quadratique est la même idée que dans le papier.

- amplitude maximale du contrôle durant la première moitié : `0.0011`;
- amplitude maximale dans les cinq derniers intervalles : `2.19`;
- erreur terminale vers la cible : `0.00133`.

Le contrôle est environ 2000 fois plus grand à la fin qu'au début. Le polygone
absorbe d'abord la perturbation, reste sur un plateau, puis est arraché de son
bassin dans les dernières couches. Le mécanisme *turnpike* survit donc lorsque
le puits n'est plus un consensus mais notre équilibre polygonal.

Ce résultat est une expérience contrôlée, pas encore un théorème général. Une
preuve demanderait une inégalité de coercivité locale autour de chaque branche
polygonale stable et une condition de contrôlabilité du FFN, analogues à celles
du papier.

Si `Q`, `K` et `V` sont eux-mêmes entraînés, la situation est plus radicale : le
paysage n'est plus seulement incliné par un contrôle externe, le noyau qui définit
les puits change. Des polygones peuvent être créés, déplacés, fusionnés ou rendus
instables pendant l'entraînement. La théorie récente de Barboni–de Hoop–Furuya–
Peyré représente alors à la fois les tokens et les paramètres d'attention par des
mesures et entraîne ces dernières dans une géométrie Wasserstein conditionnelle.

## 4. Remplacer l'exponentielle

Nous avons remplacé `exp(1.5 s)` par trois fonctions positives, toutes égales à
un en `s=0`, dans le cas `V=diag(2,-3)` et trois masses égales.

| noyau | racines polygonales | racine stable |
|---|---|---|
| exponentielle | -0.907, 0.024, 0.801 | 0.024 |
| sigmoïde bornée | -0.680 | aucune |
| softplus | -0.787 | aucune |
| polynôme `(1+s/3.01)^4` | -0.842, 0.231, 0.548 | 0.231 |

La forme générale du polygone subsiste, mais son angle et sa stabilité dépendent
fortement de la forme du noyau. L'exponentielle n'est donc pas nécessaire à
l'existence de polygones, mais elle sélectionne ici une branche très différente
de la sigmoïde ou du polynôme.

Théoriquement, il suffit de remplacer chaque exponentielle dans notre équation
spectrale–Gram par `g`. Pour l'attention non normalisée, toute fonction régulière
`g` possède une primitive `G`, donc la structure de gradient survit. Si `g>0`,
la normalisation par la somme conserve les mêmes équilibres et les mêmes signes
de stabilité locale, mais change fortement les vitesses. C'est exactement ce que
montre l'audit : les racines stables sont identiques avec et sans normalisation,
alors que les taux de contraction peuvent différer d'un facteur supérieur à 20.

L'exponentielle possède néanmoins une propriété supplémentaire : sa dérivée est
proportionnelle à elle-même. Pour une sonde dans un environnement gelé, la force
normalisée est donc le gradient d'un `log-sum-exp`. Avec un autre noyau, l'énergie
est donnée par sa primitive et non directement par le noyau lui-même.

## 5. Les trois sens différents de « sans normalisation »

### Retirer seulement le dénominateur softmax

On garde la sphère, mais remplace la moyenne probabiliste par une somme divisée
par `n`. Les équilibres polygonaux ne changent pas. Les horloges, les bassins et
la métastabilité peuvent changer. Cette version possède la structure Wasserstein
la plus simple.

### Remplacer Post-LN par Pre-LN

Les directions restent contrôlées, mais les normes modulent leur vitesse. Dans le
modèle équiangulaire présenté par Rigollet, Post-LN donne une contraction
exponentielle `1-rho ~ exp(-2t)`, tandis que Pre-LN donne une contraction
polynomiale `1-rho ~ 1/t^2`. Les deux finissent par s'effondrer dans le modèle
idéal, mais Pre-LN retarde fortement cet effondrement.

### Retirer la contrainte sphérique

Les composantes radiales ne sont plus supprimées et le paysage change
qualitativement. Pour un seul token sur le vecteur propre positif de valeur `2` :

- avec projection sphérique, il reste exactement de norme `1`;
- avec softmax normalisé mais sans sphère, sa norme atteint `10` à `t=1.151`;
- sans sphère ni dénominateur, avec le noyau exponentiel, la norme diverge en
  temps fini `t=0.003262` pour notre paramétrage.

Les polygones sphériques ne sont alors généralement plus stationnaires : une
force auparavant éliminée parce qu'elle était radiale change maintenant leur
rayon. Cette possibilité de croissance ou d'explosion est également visible dans
l'analyse gaussienne sans LayerNorm de Castin et al.

## 6. Quel est le bon espace observable ?

L'intuition « le bon espace est celui du noyau d'attention » est presque juste,
mais il faut inclure les valeurs. Pour chaque tête, les deux objets exactement
observés par la dynamique sont

\[
Z_h^\mu(x)=\int k_h(x,y)d\mu(y),\qquad
N_h^\mu(x)=\int k_h(x,y)C_hy\,d\mu(y).
\]

La vitesse utilise `N_h/Z_h`. Ainsi :

- `Q_h` et `K_h` définissent **qui voit qui**;
- `C_h=W_h^O V_h` définit **ce qui est transporté**;
- le véritable observable est l'opérateur vectoriel `mu -> (Z_h,N_h)_h`, pas
  seulement la matrice des scores.

Dans notre cas symétrique aligné, `C=B=V`, le Gram spectral
`H=XVX^T` et les Gram partielles par espace propre forment un quotient exact :
ils oublient uniquement les rotations que la dynamique ne peut pas distinguer.

Lorsque `B` est positif et `Q=K`, le noyau exponentiel admet une feature map de
tous les ordres :

\[
e^{\beta x^TBy}
=\sum_{r\ge0}\frac{\beta^r}{r!}
\langle B^{1/2}x,B^{1/2}y\rangle^r.
\]

La représentation observable contient donc les moments tensoriels de tous les
ordres. Sur le cercle, cela revient aux modes de Fourier. Un `q`-gone régulier
peut avoir moyenne et covariance presque banales, mais il apparaît nettement dans
le mode de Fourier `q`. C'est pourquoi moyenne et covariance ne suffisent pas pour
notre taxonomie polygonale.

Pour un noyau positif caractéristique, l'embedding moyen du noyau peut déterminer
toute la distribution. Pour `Q != K`, un masque causal, ou une matrice bilinéaire
indéfinie, le noyau peut être dirigé ou non positif : il faut alors étudier ses
fonctions singulières ou l'opérateur intégral, plutôt que supposer un RKHS
classique.

Enfin, une tâche supervisée observe aussi la position, le label et le readout. La
bonne mesure devient souvent une mesure jointe `mu(position, feature, label)` ou
une famille conditionnelle par séquence. Le noyau d'attention seul est suffisant
pour la dynamique d'attention, pas pour l'ensemble du Transformer entraîné.

## 7. Plusieurs têtes peuvent-elles osciller ?

Oui, mais pas dans notre modèle symétrique aligné à une tête : son énergie est
strictement monotone hors équilibre, donc aucun cycle limite n'est possible.
Une somme de têtes **non normalisées**, chacune symétrique et alignée entre score
et valeur, reste aussi le gradient de la somme de leurs énergies.

Les oscillations deviennent possibles dès que la recombinaison des valeurs
introduit une partie antisymétrique, ou plus généralement lorsqu'il n'existe plus
d'énergie commune. Nous avons testé deux têtes ayant le même score `B=I` :

- tête 1 : sortie `C1=I`, qui rassemble les tokens;
- tête 2 : sortie `C2=omega J`, où `J` tourne tout vecteur de 90 degrés et
  `omega=0.8`.

Avec 64 tokens sur le cercle, le taux de synchronisation passe de `0.471` à
`1.000`, mais le cluster ne s'arrête jamais : sa vitesse angulaire mesurée est
`0.800000`, exactement `omega`. C'est un cluster tournant stable.

Ce cas révèle aussi la limite d'un observable fondé seulement sur le noyau : une
fois les tokens synchronisés, leur matrice de Gram reste constante pendant toute
la rotation. Le noyau de score voit un équilibre, alors que l'espace des valeurs
voit un cycle. Pour détecter les oscillations, il faut donc observer à la fois les
scores, les sorties de têtes et leurs phases temporelles.

Avec plusieurs têtes normalisées séparément, même des champs individuellement
simples sont multipliés par des mobilités différentes. Une énergie commune peut
encore exister sous des conditions d'alignement, mais elle n'est plus automatique.
Les diagnostics décisifs sont : partie antisymétrique de la Jacobienne, circulation
du champ, spectre complexe de la linéarisation, et présence ou absence d'une
fonction monotone commune.

## 8. Données et références

Fichiers reproductibles :

- `experiments/spectral_self_attention/mean_field_extensions.py`
- `experiments/spectral_self_attention/one_step_muon.py`
- `data/spectral_self_attention/mean_field_extensions/continuum_convergence.csv`
- `data/spectral_self_attention/mean_field_extensions/kernel_polygon_roots.csv`
- `data/spectral_self_attention/mean_field_extensions/trained_polygon_turnpike.csv`
- `data/spectral_self_attention/mean_field_extensions/oscillatory_multihead.csv`
- `data/spectral_self_attention/one_step_muon/depth_profiles.csv`
- `data/spectral_self_attention/one_step_muon/summary.json`

Références primaires :

- P. Rigollet, *The Mean-Field Dynamics of Transformers*,
  <https://arxiv.org/abs/2512.01868>
- V. Castin, P. Ablin, J. A. Carrillo, G. Peyré, *A Unified Perspective on the
  Dynamics of Deep Transformers*, <https://arxiv.org/abs/2501.18322>
- N. Isobe, D. Inoue, M. Imaizumi, *Training-Induced Escape from Token
  Clustering in a Mean-Field Formulation of Transformers*,
  <https://arxiv.org/abs/2605.07772>
- R. Barboni, M. de Hoop, T. Furuya, G. Peyré, *Training Infinitely Deep and
  Wide Transformers*, <https://arxiv.org/abs/2605.17660>
