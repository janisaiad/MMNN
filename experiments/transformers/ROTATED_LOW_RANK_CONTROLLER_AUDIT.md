# Audit causal : quel solveur apprend quel sous-espace ?

## Conclusion

La tête apprend bien une covariance inverse de faible rang, mais le choix du
solveur ne doit pas être confondu avec le choix de la géométrie.

- Sans étape spectrale hardcodée, l'objectif Heavy--Ball apprend un bien
  meilleur sous-espace que Richardson : le recouvrement avec les deux
  outliers passe de `0.878` à `0.945` et le conditionnement effectif de
  `320.1` à `115.7`.
- Une seule étape de puissance en bloc, exacte et matrix-free, porte déjà le
  recouvrement à `0.9998` et le conditionnement à environ `6.25`. Les trois
  objectifs d'entraînement deviennent alors presque indistinguables : cette
  relation doit être hardcodée, pas réapprise.
- À profondeur quatre, Heavy--Ball est le meilleur contrôleur stationnaire
  bouclé, mais il ne bat ni Chebyshev ni PCG. La meilleure méthode en erreur
  est la tête suivie de PCG.
- La tête + PCG-4 bat PCG pur à nombre égal de **rounds HVP en bloc** d'un
  facteur `3.43`, mais perd d'un facteur `42.9` si les trois directions sont
  facturées comme des HVP scalaires séquentiels. Le gain est donc un gain de
  latence parallèle GPU, pas un gain universel de FLOPs CPU.

Ces résultats excluent la revendication « Heavy--Ball bat tous les solveurs ».
Ils valident une revendication plus précise : Heavy--Ball fournit le meilleur
compromis de cellule stationnaire exacte et entraîne une géométrie utile,
tandis que PCG reste la meilleure cellule de consommation lorsque l'algèbre de
Krylov explicite est autorisée.

## Protocole apparié

Chaque tâche est un problème inverse latent de dimension `K=12`. Sa covariance
de population possède deux outliers spectraux de force `100`, mais leur
sous-espace est tourné indépendamment par une matrice de Haar à chaque prompt.
Une covariance globale apprise dans les poids est donc isotrope en moyenne et
ne peut pas mémoriser ces directions.

Les trois modèles :

1. commencent avec exactement les mêmes poids de tête ;
2. voient les mêmes minibatches dans le même ordre ;
3. utilisent trois seeds, `1000` pas et la même perte relative dans la norme
   postérieure \(H\) ;
4. diffèrent uniquement par la cellule exacte Richardson, Heavy--Ball ou PCG ;
5. sont ensuite croisés avec toutes les cellules d'évaluation.

La tête est une seule tête softmax équivariante sur les équations. Elle route
un bloc \(Y_\theta\in\mathbb R^{K\times S}\). QR, HVP, Ritz, application du
préconditionneur et transitions du solveur sont entièrement hardcodés. Aucun
spectre exact n'est donné au réseau ; les eigendecompositions complètes ne sont
utilisées qu'après inférence pour les diagnostics et les contrôles oracle.

Le script reproductible est
[`audit_rotated_low_rank_controller_learning.py`](audit_rotated_low_rank_controller_learning.py).

## Pourquoi la nouvelle normalisation est nécessaire

La certification initiale utilisait seulement

\[
  \lambda_{\max}(H)\leq\operatorname{tr}H.
\]

Elle reste vraie après déflation, mais devient très lâche précisément quand la
tête a supprimé les outliers. Elle forçait donc le pas HB à être beaucoup trop
petit.

Posons \(A=H/s_H\), avec \(\operatorname{tr}A=\bar L\), et écrivons dans la
base \([U,U_\perp]\)

\[
 A=\begin{bmatrix}C&R^\top\\R&D\end{bmatrix}.
\]

La correction Ritz \(M=c_\star C^{-1}\) transforme exactement le bloc choisi
en \(c_\star I\). Sans former \(H\), on connaît

\[
 d=\operatorname{tr}A-\operatorname{tr}C,
 \qquad
 \gamma=\|RM^{1/2}\|_F.
\]

Par positivité, \(\lambda_{\max}(D)\le d\), donc

\[
 \lambda_{\max}(\widetilde B^{1/2}A\widetilde B^{1/2})
 \le
 \widehat L
 :=\frac{c_\star+d+
 \sqrt{(c_\star-d)^2+4\gamma^2}}{2}.
\]

Le rescaling final par \(\widehat L/\bar L\) sature une borne déterministe
beaucoup plus serrée. Il ne demande que le Ritz \(S\times S\), le résidu déjà
calculé et une trace. Les tests vérifient simultanément la formule, la
covariance de jauge, l'absence de matrice normale dans le décodeur matrix-free
et la borne finale.

## Résultat 1 : routage appris pur, aucune étape de puissance

Moyennes sur trois seeds et 4096 tâches nouvelles par seed :

| tête entraînée avec | recouvrement outliers | conditionnement | risque avec Richardson | risque avec HB | risque avec PCG |
|---|---:|---:|---:|---:|---:|
| initiale | 0.8783 | 320.09 | 0.2336 | 0.1476 | 0.01598 |
| Richardson | 0.8845 | 297.31 | 0.2151 | 0.1406 | 0.01763 |
| Heavy--Ball | **0.9446** | **115.67** | **0.1989** | **0.1036** | **0.00945** |
| PCG | 0.9412 | 123.25 | 0.2018 | 0.1066 | 0.00978 |

Ici l'objectif HB apprend réellement quelque chose : avec la même cellule HB,
sa tête réduit le risque de `29.8 %` face à la tête initiale. Appliquée ensuite
à PCG, elle réduit aussi le risque de `40.8 %`. Richardson donne un signal de
géométrie beaucoup plus faible.

Ce régime n'est toutefois pas compétitif en coût total : PCG pur à cinq HVP
atteint `6.31e-4`, contre `9.45e-3` pour tête-HB + PCG-4. Une approximation
low-rank imparfaite peut réduire le conditionnement tout en détruisant les
quelques clusters que CG exploitait déjà efficacement. Le conditionnement seul
n'est donc pas une théorie suffisante pour PCG.

![Croisement sans étape de puissance](rotated_low_rank_controller_learning/r0_learned_routing/controller_subspace_cross.png)

## Résultat 2 : une étape de puissance en bloc hardcodée

| tête entraînée avec | recouvrement outliers | conditionnement | risque Richardson-4 | risque HB-4 | risque Cheb-4 | risque PCG-4 |
|---|---:|---:|---:|---:|---:|---:|
| initiale | 0.99974 | 6.272 | 0.06785 | 0.01210 | 0.002014 | 1.092e-4 |
| Richardson | 0.99984 | **6.246** | 0.06745 | 0.01197 | **0.001876** | 1.081e-4 |
| Heavy--Ball | 0.99982 | 6.252 | **0.06732** | **0.01195** | 0.001972 | **1.059e-4** |
| PCG | 0.99981 | 6.299 | 0.06830 | 0.01205 | 0.001954 | 1.078e-4 |

Les écarts de tête sont maintenant minuscules : la bonne décision
first-principles est de conserver cette unique étape linéaire hardcodée.
L'ordre des cellules à profondeur quatre est sans ambiguïté :

\[
  \text{PCG }(1.06\,10^{-4})
  < \text{Chebyshev }(1.97\,10^{-3})
  < \text{HB }(1.19\,10^{-2})
  < \text{Richardson }(6.73\,10^{-2}).
\]

Le coût de tête vaut deux rounds HVP en bloc, soit six HVP scalaires si les
trois slots sont exécutés séquentiellement. Ainsi :

| méthode | comptage | risque (H) |
|---|---|---:|
| tête + HB-4 | 6 rounds en bloc | 1.195e-2 |
| tête + Cheb-4 | 6 rounds en bloc | 1.972e-3 |
| tête + PCG-4 | 6 rounds en bloc | **1.059e-4** |
| PCG pur | 6 HVP / rounds | 3.694e-4 |
| PCG pur | 10 HVP scalaires | **2.515e-6** |

![Croisement avec une étape de puissance](rotated_low_rank_controller_learning/r1_fixed_power/controller_subspace_cross.png)

## Validation de la théorie d'optimisation

Pour Richardson et HB, la prédiction est calculée à partir du polynôme de
résidu exact

\[
 p_{-1}(\lambda)=p_0(\lambda)=1,
 \qquad
 p_{t+1}(\lambda)
 =(1+\beta-\alpha\lambda)p_t(\lambda)-\beta p_{t-1}(\lambda),
\]

et de la mesure spectrale pondérée par le second membre. Dans le régime
principal (r=1), l'écart absolu moyen maximal entre risque prédit et risque
réalisé vaut `3.36e-7`. Cela valide la partie optimisation de la théorie
conjointe sans approximation par MLP.

Le contrôle « HB oracle d'intervalle » montre aussi une subtilité importante :
les paramètres asymptotiques de Polyak peuvent avoir un fort transitoire à
très petite profondeur. Les deux scalaires HB entraînés directement sur le
risque fini sont alors préférables. Chebyshev demeure le meilleur polynôme
minimax d'intervalle à horizon fixé.

## Régression PDE avec la certification serrée

Le benchmark PDE corrélé (K=32,M=128,S=4,r=2,L=8) a été réentraîné avec le
même protocole à trois seeds. C'est un contrôle nécessaire : l'ancienne
normalisation par la seule trace imposait une échelle HB très conservatrice,
même après une excellente déflation.

| méthode | profondeur solveur | risque (H) moyen |
|---|---:|---:|
| HB-8, tête initiale | 8 | 1.893e-4 |
| HB-8, tête et scalaires entraînés | 8 | **7.339e-5** |
| PCG-8 sans tête | 8 | 2.192e-6 |
| PCG-8, tête entraînée | 8 | **1.811e-8** |
| PCG-20 sans tête, travail scalaire égal | 20 | 4.739e-8 |

La correction de certification réduit le risque HB entraîné de `0.6883` à
`7.339e-5`, soit un facteur d'environ `9.38e3`. Cela confirme que le principal
défaut HB précédent était une échelle certifiée trop lâche, pas la cellule
bouclée elle-même. Néanmoins PCG-8 sans tête reste `33.5` fois meilleur que
HB-8, et tête + PCG-8 reste environ `4.05e3` fois meilleur. Heavy--Ball n'est
donc toujours pas quasi-identique à PCG sur cette loi à profondeur huit.

Le résultat positif contre solveur pur est plus précis : tête + PCG-8 utilise
trois rounds HVP en bloc pour le setup, c'est-à-dire douze HVP scalaires. Même
contre PCG pur à vingt HVP, il réduit le risque d'un facteur `2.62`. Ici le
préconditionnement appris apporte donc un gain à la fois au comptage parallèle
et au comptage scalaire égal.

![Audit PDE avec certificat post-déflation](pde_matrix_free_learning_tight_certificate/pde_matrix_free_solver_comparison.png)

## Chebyshev appris avec exactement la même géométrie

Pour ne pas attribuer au solveur une différence de préconditionneur, l'audit
spectral a ensuite gelé les trois têtes entraînées ci-dessus et n'a appris que
la mesure spectrale conditionnelle utilisée par le polynôme de degré huit. Le
budget total est identique : trois rounds HVP en bloc pour la tête, soit douze
HVP scalaires, puis huit HVP de solveur.

| consommateur de la même tête | risque (H) moyen | ratio vs PCG même tête |
|---|---:|---:|
| PCG-8 exact | **1.811e-8** | 1 |
| Heavy--Ball-8 exact | 7.339e-5 | 4.05e3 |
| Chebyshev de mesure appris-8 | 1.509e-4 | 8.33e3 |
| Chebyshev gardé par PCG | 3.404e-7 | 18.8 |

HB est donc `2.06` fois meilleur que le contrôleur Chebyshev appris dans ce
test apparié. Ce n'est pas une limitation de la classe polynomiale : le
polynôme oracle construit avec la mesure spectrale exacte atteint
`1.727e-8`, près du plancher PCG. Le goulot est la prédiction en contexte des
noeuds et masses par le MLP. La formule spectrale reste néanmoins prédictive :
son risque moyen `1.50896e-4` coïncide avec le risque Clenshaw réalisé
`1.50913e-4`. La garde résiduelle déclenche PCG sur `92.6--95.9 %` des prompts
et ne constitue donc pas un gain Chebyshev pratique.

![Chebyshev appris avec tête PDE partagée](pde_moment_chebyshev_tight_shared_head/moment_chebyshev_pde_comparison.png)

## Mesure Ritz--Krylov : le MLP n'est plus le goulot

La tête gelée fournit maintenant les sondes covariantes \(U\). Deux actions
supplémentaires de l'opérateur symétrique
\(A_\theta=B_\theta^{1/2}HB_\theta^{1/2}\), suivies d'un QR en bloc,
construisent exactement

\[
 Q=[Q_0,Q_1],\qquad T=Q^\top A_\theta Q.
\]

Les huit valeurs de Ritz de \(T\) sont les nœuds résolus. La trace exacte de
\(A_\theta\), déjà déterminée par les moments du préconditionneur, donne
l'unique atome moyen du complément. Sous le prior isotrope, les poids d'énergie
sont alors proportionnels à « multiplicité \(\times\) valeur propre ». Le
Gram solve Chebyshev et Clenshaw restent identiques et exacts. Aucun spectre
complet, label spectral ou MLP de poids n'est utilisé.

| contrôleur | coût scalaire équivalent | risque \(H\) moyen |
|---|---:|---:|
| Ritz--Chebyshev, \(q=1\) | 24 HVP | \(2.803\times10^{-3}\) |
| Ritz--Chebyshev, \(q=2\), tête HB gelée | 28 HVP | **\(9.676\times10^{-6}\)** |
| Ritz--Chebyshev, \(q=3\), tête HB gelée | 32 HVP | \(8.868\times10^{-6}\) |
| Ritz--Chebyshev, \(q=4\), tête HB gelée | 36 HVP | \(8.862\times10^{-6}\) |
| Ritz--Chebyshev, \(q=2\), tête réentraînée | 28 HVP | \(9.945\times10^{-6}\) |
| Heavy--Ball-8 | 20 HVP | \(7.339\times10^{-5}\) |
| Chebyshev avec mesure MLP | 20 HVP | \(1.509\times10^{-4}\) |
| PCG-8 pur | 8 HVP | \(2.192\times10^{-6}\) |
| PCG-28 pur | 28 HVP | \(4.739\times10^{-8}\) |
| tête + PCG-8 | 20 HVP | **\(1.811\times10^{-8}\)** |

Le contrôleur structuré \(q=2\) améliore donc HB d'un facteur \(7.58\) et le
MLP spectral d'un facteur \(15.6\). Passer à \(q=3\) ne gagne que \(8.35\%\)
pour quatre HVP scalaires de plus, et \(q=4\) est indiscernable de \(q=3\).
Réentraîner le softmax avec son propre
risque ne l'améliore pas : les étapes block-power hardcodées avaient déjà
extrait la géométrie utile. Le résultat rejette simultanément deux excès :
un MLP libre n'est pas nécessaire pour construire les poids, mais Chebyshev
ne bat toujours pas PCG à travail égal.

![Frontière risque--travail](pde_ritz_moment_shared_head/ritz_moment_selection.png)

## Décision architecturale

La sélection finale dépend du sens exact de « Transformer pur » :

- **Cellule bouclée, stationnaire, algèbre minimale :** Heavy--Ball. Deux
  scalaires partagés, un token mémoire et aucune division adaptative.
- **Meilleur polynôme à horizon fixé :** Ritz--Chebyshev \(q=2\). Le softmax
  apprend seulement les sondes ; la mesure comprimée, les coefficients et
  Clenshaw sont exacts. Il bat HB sur cette PDE mais demande deux blocs de
  setup supplémentaires.
- **Meilleure précision shallow :** tête contextuelle + PCG explicite. C'est
  la seule variante qui bat ici PCG pur au comptage de latence en rounds.

La contribution apprise commune est donc

\[
 \mathcal C\longmapsto
 U_\theta(\mathcal C)\longmapsto
 B_\theta(\mathcal C)\approx H_\mathcal C^{-1},
\]

c'est-à-dire une covariance postérieure low-rank spécifique au prompt. Le
solveur exact qui la consomme peut être changé sans réentraîner la géométrie.

## Relation exacte avec Bordelon--Letey--Pehlevan

Leur modèle réduit écrit la profondeur comme les puissances de
\(I-L^{-1}\widehat\Sigma\Gamma\). Dans le cas à covariance fixe,
\(\Gamma\) mémorise en poids une géométrie globale proche de
\(\Sigma^{-1}\), donc une covariance inverse servant de préconditionneur.
Quand les covariances sont aléatoirement tournées entre contextes, cette
matrice globale ne peut plus blanchir chaque tâche et la solution devient une
descente in-context non préconditionnée.

Notre tête a la même lecture géométrique générale, mais pas le même statut :
\(B_\theta(\mathcal C)\) est recalculé depuis la covariance du prompt courant,
est covariant aux rotations, et n'est low-rank que dans les directions
routées. L'apprentissage ne mémorise donc pas une covariance fixe; il apprend
une règle contextuelle qui transforme les moments du prompt en métrique.

## Correction conditionnelle minimale du complément

La fermeture à un atome de Ritz--Chebyshev \(q=2\) connaît exactement la
trace du complément, mais pas sa dispersion ni la répartition d'énergie de la
tâche. Un MLP de largeur 12 est donc autorisé à apprendre seulement trois
statistiques du prompt : (i) la porte d'énergie résolu/complément, (ii) la
balance entre deux atomes du complément et (iii) leur dispersion bornée. Les
deux nœuds restent positifs, sous le certificat spectral, et leur moyenne
pondérée conserve exactement la trace du complément. La tête de covariance,
les HVP, Ritz, le Gram solve et Clenshaw sont figés et exacts.

| méthode | risque \(H\) moyen | ratio vs correction apprise |
|---|---:|---:|
| fermeture exacte à un atome, \(q=2\) | \(9.603\times10^{-6}\) | 19.8 |
| correction conditionnelle à trois scalaires | **\(4.860\times10^{-7}\)** | 1 |
| PCG-8 pur | \(2.192\times10^{-6}\) | 4.51 |
| PCG pur à travail scalaire égal | \(4.739\times10^{-8}\) | 0.0975 |
| même tête + PCG-8 | **\(1.811\times10^{-8}\)** | 0.0373 |

Les paramètres appris sont cohérents entre graines : la masse d'énergie
attribuée aux huit modes résolus passe de \(0.853\) à
\(0.951,0.953,0.961\), la balance vaut \(0.621,0.620,0.706\), et la dispersion
du complément reste petite (\(1.36\%,1.42\%,1.81\%\) du certificat). Le gain
vient principalement d'une correction stable de la masse de tâche sur les
directions résolues, pas d'un ajustement arbitraire de nombreux nœuds.

La correction apprise bat donc PCG pur à profondeur huit, mais pas PCG pur à
travail total égal et encore moins PCG utilisant la même géométrie. On la
retient comme meilleur décodeur polynomial structuré; on ne la présente pas
comme meilleur solveur numérique universel.

![Correction spectrale apprise contre solveurs purs](pde_corrected_ritz_measure_training/pde_matrix_free_solver_comparison.png)
