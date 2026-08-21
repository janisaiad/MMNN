# Sélection spectrale en self-attention symétrique — extension expérimentale et théorique

## 1. Objet et statut des conclusions

Ce rapport reproduit et étend *Spectral Selection in Symmetric Self-Attention
Dynamics* ([arXiv:2604.26085](https://arxiv.org/abs/2604.26085)). Le papier prouve
deux résultats globaux : sélection du mode positif dominant dans un cône unilatéral,
et sélection générique du mode le plus négatif pour deux particules lorsque (V) est
négative définie. L'extension ci-dessous couvre toutes les classes qualitatives de
spectres réels symétriques (signes, zéros, dominance et multiplicités), donne une
classification complète de la stabilité des équilibres purs, et identifie une
nouvelle famille exacte d'équilibres mixtes à trois groupes.

Une précision importante : « tous les patterns de valeurs propres » ne signifie pas
que tous les points critiques du système à (n) particules sont classifiés. Cette
dernière tâche contient déjà, dans un espace propre multiple, toute la richesse du
modèle isotrope et ses configurations multi-clusters. Les résultats complets ici
sont :

1. la taxonomie des spectres, à conjugaison orthogonale près ;
2. tous les attracteurs *purs* possibles pour un spectre, un (\beta) et un partage de
   signes arbitraires ;
3. une famille explicite d'attracteurs mixtes qui explique les écarts systématiques
   au scénario bipolaire du papier ;
4. un critère matriciel exact de stabilité de cette nouvelle famille ;
5. un atlas numérique sur toutes les classes spectrales, complété par des trajectoires
   longues près des seuils et des dégénérescences.

## 2. Modèle, énergie et conséquence dynamique générale

Dans une base propre de (V=V^\top), avec valeurs propres
(lambda_1\ge\cdots\ge\lambda_d), la dynamique est

\[
\dot x_i=P_{x_i}^{\perp}\sum_jK_{ij}Vx_j,
\qquad
K_{ij}=\frac{e^{\beta\langle x_i,Vx_j\rangle}}
{\sum_m e^{\beta\langle x_i,Vx_m\rangle}}.
\]

Elle est le gradient ascendant pondéré de

\[
E_\beta(X)=\frac1{2\beta}\sum_{i,j}
e^{\beta\langle x_i,Vx_j\rangle},
\qquad
\frac{dE_\beta}{dt}=\sum_i Z_i\|\dot x_i\|^2\ge0.
\]

Le champ, la métrique et l'énergie sont analytiques sur le produit compact de
sphères. L'inégalité de Łojasiewicz pour les flots gradients analytiques implique
donc que chaque trajectoire converge vers un point critique unique. Il n'y a ni
cycle limite ni chaos asymptotique dans ce modèle : les longues phases observées à
grand (\beta) sont des métastabilités ou des convergences vers des équilibres
multi-clusters.

## 3. Taxonomie exhaustive des spectres

Une permutation des valeurs propres ou une rotation dans un espace propre ne change
pas la dynamique, à conjugaison orthogonale près. Les classes qualitatives finies
sont donc déterminées par :

- l'inertie : définie positive, semi-définie positive, nulle, semi-définie négative,
  définie négative, indéfinie ;
- la comparaison entre (lambda_+=\lambda_1) et
  (lambda_-=\lambda_d), notamment (lambda_+>|\lambda_-|),
  (lambda_+<|\lambda_-|), ou égalité ;
- la multiplicité des valeurs extrêmes ;
- les collisions et zéros intérieurs ;
- les rapports continus entre valeurs propres, qui déplacent les seuils sans créer
  une nouvelle classe combinatoire.

Les quinze représentants de `taxonomy.py` couvrent ces possibilités. Les seuils
analytiques ci-dessous couvrent les rapports continus à l'intérieur de chaque classe.

## 4. Classification complète des équilibres purs

Pour tout mode (p) et tout motif (s_i\in\{\pm1\}), la configuration
(x_i=s_i e_p) est stationnaire. Posons (n_+=|\{i:s_i=1\}|),
(n_-=n-n_+), (r=n_+/n_-), et

\[
c=e^{2\beta\lambda_p},\qquad
\sigma(c,r)=\frac{(c-r)(cr-1)}{r(c^2-1)}.
\]

Le calcul du papier, revérifié indépendamment par différences finies, donne la
classification suivante.

### 4.1 État homogène

L'état homogène sur (p) est asymptotiquement stable si et seulement si

\[
\lambda_p>0,\qquad \lambda_k<\lambda_p\quad(k\ne p).
\]

Ainsi, seul le maximum spectral positif simple peut porter un consensus stable.
Une valeur propre nulle ou un maximum multiple produit des directions neutres.

### 4.2 État non homogène à signes séparés

Pour (lambda_p>0), la stabilité équivaut à

\[
\lambda_p>\frac{|\log r|}{2\beta},\qquad
\lambda_k<\lambda_p\sigma(c,r)\quad(k\ne p).
\]

Pour (lambda_p<0), elle équivaut à

\[
\lambda_p<-\frac{|\log r|}{2\beta},\qquad
\lambda_p<\lambda_k<\lambda_p\sigma(c,r)\quad(k\ne p).
\]

Ces inégalités montrent immédiatement qu'un mode positif stable à signes séparés
doit être le maximum spectral strict, tandis qu'un mode négatif stable doit être le
minimum strict. Aucun mode intérieur ni mode nul ne peut être un attracteur pur
hyperbolique.

Pour un partage équilibré (r=1),
(sigma(e^{2\beta\lambda},1)=\tanh(\beta\lambda)). Les deux critères deviennent

\[
\lambda_2<\lambda_+\tanh(\beta\lambda_+)
\]

pour une polarisation sur le maximum positif, et

\[
\lambda_+<|\lambda_-|\tanh(\beta|\lambda_-|)
\]

pour une polarisation sur le minimum négatif. Dans un spectre indéfini, les deux
inégalités peuvent être satisfaites simultanément. On obtient alors trois types
d'attracteurs purs coexistants : consensus sur (e_+), polarisation sur (e_+), et
polarisation sur (e_-). C'est la source théorique de la forte dépendance à
l'initialisation observée hors du cône du papier.

### 4.3 Lecture par classe spectrale

| Spectre | Attracteurs purs hyperboliques possibles |
|---|---|
| (V=0) | aucun ; toute configuration est stationnaire |
| semi-définie/définie positive, (lambda_+>0) | consensus sur (e_+), plus polarisation sur (e_+) au-dessus du seuil |
| semi-définie négative, (lambda_+=0>lambda_-) | polarisation sur (e_-) ; pas de consensus hyperbolique |
| définie négative | polarisation sur (e_-), avec seuil d'imbalance pour (n_+\ne n_-) |
| indéfinie | consensus (e_+), polarisation (e_+), polarisation (e_-), dans toutes les combinaisons autorisées par les deux seuils |
| valeur extrême multiple | les points purs ont des directions neutres ; remplacer la direction par une variété dans l'espace propre |

## 5. Développement à petite température

Le développement uniforme sur l'espace compact des configurations est

\[
E_\beta(X)=\frac{n^2}{2\beta}
+\frac12 S_1(X)+\frac\beta4 S_2(X)+O(\beta^2),
\]

où

\[
S_1=\sum_{i,j}\langle x_i,Vx_j\rangle
=\left\langle\sum_i x_i,V\sum_i x_i\right\rangle,
\qquad
S_2=\sum_{i,j}\langle x_i,Vx_j\rangle^2.
\]

Si (lambda_+>0) est simple, les maximisateurs globaux du terme dominant sont les
consensus (pm e_+). Cela explique la convergence quasi universelle vers le
consensus à (\beta=0.03) dans tous les spectres possédant une valeur propre positive.

Si (V<0), le terme dominant est maximisé sur la variété
(sum_i x_i=0). Pour (n) impair, cette contrainte exclut une configuration
bipolaire pure avec deux groupes de tailles différentes. Le terme suivant et les
corrections d'ordre supérieur sélectionnent alors des triangles, polygones ou états
à trois groupes. C'est précisément le régime manquant dans le scénario
multi-particules suggéré par le papier.

## 6. Nouvelle famille exacte d'équilibres mixtes

Soient (u,v) deux vecteurs propres orthogonaux, de valeurs propres respectives
(a,b). Pour des entiers (m,k\ge1), considérons

\[
\underbrace{u,\ldots,u}_{m\text{ fois}},\qquad
\underbrace{q u+s v,\ldots,q u+s v}_{k\text{ fois}},\qquad
\underbrace{q u-s v,\ldots,q u-s v}_{k\text{ fois}},
\qquad s=\sqrt{1-q^2}.
\]

Définissons

\[
A(q)=b+(a-b)q^2,\quad
B(q)=-b+(a+b)q^2,\quad
C(q)=aq.
\]

### Proposition — équation exacte d'existence

La configuration ci-dessus est un équilibre si et seulement si (q\in(-1,1)) est
une racine de

\[
F_{a,b}^{m,k}(q)=
q\left[(a-b)e^{\beta A(q)}+(a+b)e^{\beta B(q)}\right]
+a\frac{m}{k}e^{\beta C(q)}=0.
\]

**Preuve.** Pour un jeton central, les composantes en (v) des deux groupes
polaires s'annulent et la sortie d'attention est parallèle à (u). Pour un jeton
(qu+sv), les trois scores possibles sont (A,B,C). La composante tangentielle de
la sortie dans la direction (su-qv) est, à un facteur positif près,
(F_{a,b}^{m,k}(q)). La même équation vaut par symétrie pour (qu-sv).

À (\beta=0), si (a\ne0), l'équation se réduit à

\[
q=-\frac{m}{2k},
\]

qui est exactement la condition de moyenne nulle. Pour (m=k=1), on retrouve le
triangle équilatéral (q=-1/2). Si (a>0>b), (|b|>a), et (\beta\to\infty), la
petite racine positive vérifie

\[
q\sim \frac{a m}{k(|b|-a)}e^{-\beta|b|}.
\]

Elle correspond à un sous-groupe aligné sur le mode positif et à deux sous-groupes
presque antipodaux sur le mode négatif.

### Critère exact de stabilité

Au point critique, notons (K^*) la matrice d'attention et

\[
\phi_i^*=\left\langle x_i,\sum_jK_{ij}^*Vx_j\right\rangle.
\]

La linéarisation se décompose en blocs invariants :

1. un bloc (J_{a,b}^{m,k}) dans le plan engendré par (u,v) ;
2. pour chaque autre valeur propre (lambda_\ell), un bloc

\[
L_\ell=\lambda_\ell K^*-\operatorname{Diag}(\phi_1^*,\ldots,\phi_n^*).
\]

La configuration est asymptotiquement stable si et seulement si tous ces blocs ont
leur spectre dans le demi-plan gauche. Cette décomposition vient du fait que les
variations des scores d'attention sont nulles au premier ordre pour une perturbation
orthogonale au plan propre ((u,v)). Elle fournit un test exact pour n'importe quel
spectre, y compris lorsque (a) ou (b) ne sont pas des valeurs extrêmes.

Exemples confirmés numériquement :

- ((a,b,\beta,m,k)=(2,-3,1.5,1,1)) : (q=0.0239051913) ;
- ((-0.4,-4,0.03,1,1)) : (q=-0.2341172382) ;
- ((-2,-4,0.03,1,1)) : (q=-0.4323657845) ;
- ((-0.4,-4,0.03,1,2)) : (q=-0.1141124228), donnant exactement la masse
  observée (4(1-q^2)/5=0.7895826840) sur le mode (-4).

## 7. Protocole numérique

Le calcul principal contient :

- 15 représentants spectraux ;
- (\beta\in\{0.03,0.1,0.3,0.7,1.5,3,8\}) ;
- (n\in\{2,3,4,5,8,20\}) ;
- 64 initialisations sphériques indépendantes par combinaison ;
- 40 320 trajectoires aléatoires jusqu'à (t=50) ;
- 10 080 comparaisons entre spectre analytique et Jacobienne par différences finies ;
- 240 trajectoires supplémentaires jusqu'à (t=500) ou (t=1000) ;
- une carte de 8 520 paramètres pour la famille mixte à trois jetons, contenant
  20 780 racines et 4 533 racines stables dans le plan ;
- un audit (dt\in\{0.04,0.02,0.01,0.005\}) et horizons jusqu'à (80).

L'intégrateur est RK4 avec rétraction sphérique à chaque sous-étape. L'énergie est
contrôlée croissante, la norme est conservée à l'erreur machine et le champ reste
tangent. L'audit au temps (80) donne une erreur médiane de Gram au plus
(4.8\times10^{-17}) sur les cas testés. Les 10 080 spectres de Jacobienne concordent
sans aucune erreur de signe ; l'écart spectral maximal est
(6.4\times10^{-11}).

## 8. Observations principales

1. **Petit (\beta), valeur propre positive.** À (\beta=0.03), toutes les classes
   ayant (lambda_+>0) donnent pratiquement 100 % de consensus, conformément au
   développement de l'énergie.

2. **Matrice positive, grand (\beta).** Même sans valeur propre négative, les états
   à signes séparés sur le maximum positif deviennent stables. Pour le spectre
   ((3,2,1,0.4)), la fraction bipolaire passe de 0 % à (\beta=0.03) à 81.8 % à
   (\beta=8).

3. **Spectre indéfini.** Les bassins du consensus positif, de la polarisation
   positive, de la polarisation négative et des états mixtes coexistent. Pour
   ((2,1.8,-0.5,-3)), (n=20,\beta=8), 21 trajectoires longues sur 24 convergent
   vers un état mixte, typiquement avec 20 % de masse sur (+2) et 80 % sur (-3).

4. **Négatif défini, (n) impair et petit (\beta).** Sous le seuil
   (\beta>|\log(n_+/n_-)|/(2|\lambda_-|)), la polarisation pure déséquilibrée est
   instable. Pour (n=3,\beta=0.03), toutes les trajectoires longues convergent vers
   l'une des familles mixtes exactes décrites plus haut ; pour (n=5), la famille
   (1+2+2) est sélectionnée.

5. **Dégénérescence du minimum.** Avec
   ((-0.4,-1,-4,-4)), (n=3,\beta=0.03), 16 trajectoires sur 24 convergent vers un
   triangle équilatéral dans l'espace propre minimal. À (\beta=8,n=20), 22 sur 24
   convergent vers des configurations multi-clusters stationnaires dans cet espace
   propre, et non vers une seule droite bipolaire.

6. **Dégénérescence du maximum.** Avec ((3,3,1,0.4)), (n=20,\beta=8), 18
   trajectoires sur 24 sont déjà stationnaires dans l'espace propre maximal à
   (t=500), avec plusieurs directions de clusters. La multiplicité ne fait donc
   pas que rendre le mode sélectionné non unique : elle change la géométrie de
   l'attracteur.

7. **Métastabilité.** Les cas simples négatifs à grand (\beta) finissent par devenir
   bipolaires dans l'audit long. En revanche, les vitesses proches de (10^{-6})
   presque constantes dans les espaces propres multiples signalent des plateaux
   extrêmement longs ou des équilibres non hyperboliques ; ils ne doivent pas être
   interprétés comme une sélection rapide d'une direction propre.

## 9. Conclusion théorique

Le signe du spectre ne suffit pas à prédire le régime. Les objets déterminants sont

- les deux extrêmes (lambda_+,lambda_-) et les gaps adjacents ;
- les quantités sans dimension (\beta\lambda_p) ;
- le ratio discret (n_+/n_-), donc la parité de (n) ;
- la multiplicité des espaces propres extrêmes ;
- l'existence et la stabilité de branches mixtes (F_{a,b}^{m,k}(q)=0).

Les théorèmes du papier correspondent à deux secteurs où une branche pure est
globalement sélectionnée sous des hypothèses géométriques fortes. En dehors de ces
secteurs, la description correcte est un diagramme de multistabilité. Les branches
mixtes donnent le mécanisme manquant entre consensus et bipolarisation pure et
expliquent les observations multi-particules que la théorie du papier laissait
ouvertes.

## 10. Fichiers reproductibles

- `experiments/spectral_self_attention/simulator.py` : ODE, énergie, intégrateur,
  diagnostics et Jacobiennes ;
- `experiments/spectral_self_attention/taxonomy.py` : taxonomie spectrale ;
- `experiments/spectral_self_attention/run_sweep.py` : atlas Monte-Carlo et
  validation locale ;
- `experiments/spectral_self_attention/mixed_equilibria.py` : équation exacte et
  carte de phase des nouveaux équilibres ;
- `experiments/spectral_self_attention/long_time_audit.py` : trajectoires longues ;
- `data/spectral_self_attention/full/` : résultats principaux ;
- `data/spectral_self_attention/long_time/` : audit long et états finaux ;
- `data/spectral_self_attention/theory/` : carte de phase ;
- `data/spectral_self_attention/figures/` : figures de synthèse ;
- `tests/test_spectral_self_attention.py` : dix tests de cohérence et de théorie.

