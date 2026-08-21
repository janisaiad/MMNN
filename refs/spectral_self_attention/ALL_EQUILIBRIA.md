# Tous les équilibres de la self-attention sphérique symétrique

## Résultat principal

Pour le système de [Kuehn–Yoon, arXiv:2604.26085](https://arxiv.org/abs/2604.26085),
il n'existe pas de liste finie de tous les équilibres valable pour tout
\(n,d,V,\beta\). Deux contre-exemples suffisent :

- si \(V=0\), chaque point de \((\mathbb S^{d-1})^n\) est un équilibre ;
- si une valeur propre de \(V\) est multiple, des familles continues de polygones,
  simplexes et autres codes sphériques apparaissent dans l'espace propre.

Il existe en revanche une **caractérisation nécessaire et suffisante**, finie et
constructive. Elle est donnée ci-dessous. Elle contient sans exception les
consensus, états bipolaires, polygones, configurations multi-clusters, équilibres
mixtes et continua dus aux multiplicités.

Un travail antérieur classe formellement les équilibres en consensus, bipartites,
polygonaux et « clustering », mais sa classe clustering reste définie par
l'équation stationnaire elle-même et le nombre d'attracteurs n'y est pas déterminé
([Altafini, arXiv:2511.11553](https://arxiv.org/abs/2511.11553)). La formulation
spectrale-Gram qui suit est plus précise dans le cas symétrique
\(Q^\top K=V=V^\top\) : elle quotient exactement les rotations internes aux espaces
propres et donne un système complet en matrices semi-définies positives.

## 1. Équation vectorielle exacte

Écrivons les jetons comme les lignes de \(X\in\mathbb R^{n\times d}\), avec
\(\|x_i\|=1\), et posons

\[
H=XVX^\top,\qquad
R=\exp_\circ(\beta H),
\]

où \(\exp_\circ\) désigne l'exponentielle entrée par entrée. Le dénominateur
softmax peut être supprimé dans une équation d'équilibre, car il est strictement
positif et ne fait que multiplier chaque ligne par un scalaire.

Définissons

\[
\mu_i=(RH)_{ii},\qquad
M=\operatorname{Diag}(\mu_1,\ldots,\mu_n).
\]

### Théorème 1 — caractérisation par multiplicateurs

\[
\boxed{\quad X\text{ est un équilibre}
\iff RXV=MX,\quad \operatorname{diag}(XX^\top)=\mathbf1.\quad}
\]

En effet, la ligne \(i\) du champ s'annule si et seulement si
\((RXV)_i\) est colinéaire à \(x_i\). Son coefficient de colinéarité est
nécessairement

\[
\langle x_i,(RXV)_i\rangle=(RH)_{ii}=\mu_i.
\]

Cette équation vectorielle est déjà exhaustive, mais elle dépend encore d'une base
dans chaque espace propre. La formulation suivante élimine cette redondance.

## 2. Théorème spectral-Gram : tous les équilibres modulo symétrie

Soient \(\lambda_1^\star,\ldots,\lambda_s^\star\) les valeurs propres distinctes de
\(V\), de multiplicités \(m_1,\ldots,m_s\). Décomposons

\[
X=C_1\oplus\cdots\oplus C_s,\qquad
C_\alpha\in\mathbb R^{n\times m_\alpha},
\]

et introduisons les Gram partielles

\[
G_\alpha=C_\alpha C_\alpha^\top\succeq0.
\]

Alors

\[
XX^\top=\sum_\alpha G_\alpha,\qquad
H=XVX^\top=\sum_\alpha\lambda_\alpha^\star G_\alpha.
\]

### Théorème 2 — système PSD nécessaire et suffisant

Les équilibres, modulo les rotations orthogonales à l'intérieur de chaque espace
propre, sont exactement les familles de matrices \(G_\alpha\) satisfaisant

\[
\boxed{
\begin{aligned}
&G_\alpha\succeq0,\qquad
\operatorname{rank}G_\alpha\le m_\alpha,\\
&\sum_{\alpha=1}^s\operatorname{diag}G_\alpha=\mathbf1,\\
&H=\sum_{\alpha=1}^s\lambda_\alpha^\star G_\alpha,\qquad
R=\exp_\circ(\beta H),\\
&M=\operatorname{Diag}\operatorname{diag}(RH),\\
&(\lambda_\alpha^\star R-M)G_\alpha=0
\quad\text{pour tout }\alpha.
\end{aligned}}
\]

### Preuve

En projetant \(RXV=MX\) sur l'espace propre \(E_{\lambda_\alpha^\star}\), on obtient

\[
(\lambda_\alpha^\star R-M)C_\alpha=0.
\]

En multipliant à droite par \(C_\alpha^\top\), on obtient l'équation annoncée pour
\(G_\alpha\).

Réciproquement, factorisons toute solution PSD sous la forme
\(G_\alpha=C_\alpha C_\alpha^\top\), avec au plus \(m_\alpha\) colonnes. Comme
\(\operatorname{Im}G_\alpha=\operatorname{Im}C_\alpha\), l'égalité
\((\lambda_\alpha^\star R-M)G_\alpha=0\) implique
\((\lambda_\alpha^\star R-M)C_\alpha=0\). En concaténant les \(C_\alpha\) dans des
espaces propres orthogonaux, on reconstruit \(X\) et donc \(RXV=MX\).

Deux factorisations d'une même \(G_\alpha\) diffèrent seulement par une rotation
dans \(E_{\lambda_\alpha^\star}\). Le système décrit donc exactement le quotient
par le groupe de symétrie
\(\prod_\alpha O(m_\alpha)\).

## 3. Réduction exhaustive par clusters

Toute configuration finie possède \(q\le n\) positions distinctes
\(w_1,\ldots,w_q\), de multiplicités entières \(r_1,\ldots,r_q\). Posons

\[
S_{ab}=\langle w_a,Vw_b\rangle,\qquad
\mathcal R_{ab}=e^{\beta S_{ab}}.
\]

### Théorème 3 — équations de tous les \(q\)-clusters

La configuration est un équilibre si et seulement si, pour chaque \(a\),

\[
\boxed{
\sum_{b=1}^q r_b\mathcal R_{ab}Vw_b
=\nu_a w_a,\qquad
\nu_a=\sum_{b=1}^q r_b\mathcal R_{ab}S_{ab},
\qquad \|w_a\|=1.}
\]

Cette réduction n'est pas une approximation : en regroupant les lignes identiques,
elle est équivalente au Théorème 1. Elle donne un procédé d'énumération :

1. énumérer les partitions \(r_1+\cdots+r_q=n\) ;
2. résoudre les \(q\) équations tangentielles sur
   \((\mathbb S^{d-1})^q\) ;
3. quotienter les permutations de groupes de même taille et les rotations qui
   commutent avec \(V\).

Il n'existe pas de borne universelle sur le nombre de branches quand \(n\), les
valeurs propres et \(\beta\) varient.

## 4. Toutes les familles explicites universelles

### 4.1 Matrice nulle

\[
V=0\quad\Longrightarrow\quad
\operatorname{Eq}=(\mathbb S^{d-1})^n.
\]

### 4.2 Un seul jeton

Pour \(n=1\),

\[
\operatorname{Eq}
=\bigcup_{\lambda\in\operatorname{spec}V}
\bigl(E_\lambda\cap\mathbb S^{d-1}\bigr).
\]

Avec spectre simple, ce sont les \(2d\) vecteurs propres orientés. Avec
multiplicité, ce sont des sphères entières.

### 4.3 Consensus

Pour tout \(n\),

\[
x_1=\cdots=x_n=u
\]

est un équilibre si et seulement si \(u\) appartient à un espace propre de \(V\).
Ainsi, contrairement au modèle non symétrique plus général, il n'existe pas ici de
consensus sur une direction qui ne soit pas propre.

### 4.4 Toutes les configurations sur une droite propre

Pour tout \(u\in E_\lambda\cap\mathbb S^{d-1}\) et tout motif
\(s_i\in\{\pm1\}\),

\[
x_i=s_i u
\]

est un équilibre. Cela donne \(2^n\) motifs orientés par direction propre, avant
quotient du changement de signe global.

### 4.5 Noyau de \(V\)

Toute configuration entièrement contenue dans \(\ker V\) est stationnaire :

\[
x_i\in\ker V\cap\mathbb S^{d-1}\quad\forall i.
\]

Plus généralement, l'équation spectrale-Gram du noyau est

\[
MG_0=0.
\]

Donc tout jeton ayant une composante non nulle dans le noyau doit avoir
\(\mu_i=0\), c'est-à-dire une sortie d'attention non normalisée nulle. Un jeton
entièrement dans le noyau voit tous les scores égaux à zéro et impose

\[
\sum_j Vx_j=0.
\]

Cette condition décrit exactement la manière dont des jetons de noyau peuvent être
adjoints à une configuration active.

### 4.6 Configurations dans un seul espace propre

Si tous les jetons appartiennent à \(E_\lambda\), le problème devient isotrope dans
cet espace :

\[
\sum_j e^{\beta\lambda\langle x_i,x_j\rangle}x_j
\parallel x_i\qquad(\lambda\ne0).
\]

Il contient notamment :

- tous les motifs \(\pm u\) ;
- le simplexe régulier à \(q\le m_\lambda+1\) sommets, avec multiplicités égales ;
- tout polygone régulier dans un plan de \(E_\lambda\), avec multiplicités égales ;
- les cross-polytopes, hypercubes et, plus généralement, les orbites de groupes
  orthogonaux dont le stabilisateur d'un sommet ne fixe que son axe.

Pour la dernière assertion, la somme pondérée

\[
F(u)=\sum_{v\in\mathcal O}e^{\beta\lambda\langle u,v\rangle}v
\]

est invariante par le stabilisateur de \(u\). Si l'espace fixe de ce stabilisateur
est \(\operatorname{span}(u)\), alors \(F(u)\parallel u\).

Ces orbites fournissent déjà une infinité de géométries quand \(n\) varie et des
continua lorsqu'on les fait tourner dans un espace propre multiple.

### 4.7 Équilibres à sortie nulle

Ils satisfont

\[
RXV=0.
\]

Ils correspondent à la classe parfois appelée « polygonale ». Si
\(\beta>0\) et \(V\succ0\), il
n'en existe aucun : après regroupement des positions distinctes, le noyau
exponentiel

\[
\mathcal R_{ab}=e^{\beta\langle V^{1/2}w_a,V^{1/2}w_b\rangle}
\]

est strictement défini positif, donc inversible, et
\(\mathcal R\operatorname{Diag}(r)WV=0\) forcerait \(W=0\).

Pour \(V\) singulière ou indéfinie, ces équilibres sont exactement les solutions du
système spectral-Gram avec \(M=0\).

### 4.8 Branches mixtes dans plusieurs espaces propres

Elles sont nécessaires : les expériences précédentes ont trouvé des attracteurs
stables qui combinent deux valeurs propres. Une famille exacte est

\[
\underbrace{u,\ldots,u}_{m},\qquad
\underbrace{qu+\sqrt{1-q^2}v,\ldots}_{k},\qquad
\underbrace{qu-\sqrt{1-q^2}v,\ldots}_{k},
\]

où \(Vu=au\), \(Vv=bv\). Elle existe exactement lorsque

\[
\boxed{
q\!\left[
(a-b)e^{\beta[b+(a-b)q^2]}
+(a+b)e^{\beta[-b+(a+b)q^2]}
\right]
+a\frac{m}{k}e^{\beta aq}=0.}
\]

Ce n'est qu'une sous-famille symétrique. Les Théorèmes 2 et 3 contiennent également
les branches mixtes asymétriques à deux, trois ou davantage de groupes.

## 5. Classification complète à \(\beta=0\)

À température infinie, \(K_{ij}=1/n\). Posons

\[
y=V\sum_{j=1}^n x_j.
\]

Alors \(X\) est un équilibre si et seulement si l'une des deux alternatives
suivantes se réalise :

1. \(y=0\), soit
   \[
   V\sum_jx_j=0;
   \]
   toute configuration satisfaisant cette condition est un équilibre ;
2. \(y\ne0\), auquel cas il existe un vecteur propre unitaire \(u\) associé à une
   valeur propre non nulle et un motif de signes non équilibré tel que
   \[
   x_i=\pm u,\qquad \sum_i\operatorname{sign}(x_i\cdot u)\ne0.
   \]

### Preuve

La condition est \(P_{x_i}^\perp y=0\) pour tout \(i\). Si \(y=0\), elle est
automatique. Sinon, chaque \(x_i=\pm u\), où \(u=y/\|y\|\). La somme vaut
\(c\,u\) avec \(c\ne0\), puis \(y=cVu\parallel u\), donc \(u\) est propre.

Cette classification est entièrement explicite et complète.

## 6. Réduction aux sous-espaces spectraux

Pour un équilibre à \(q\) positions distinctes, écrivons

\[
\mathcal A=\mathcal R\operatorname{Diag}(r_1,\ldots,r_q),\qquad
W=\begin{pmatrix}w_1^\top\\ \vdots\\ w_q^\top\end{pmatrix}.
\]

Les équations de clusters ont la forme

\[
\mathcal A WV=\operatorname{Diag}(\nu)W.
\]

Si \(\mathcal A\) est inversible,

\[
WV=\mathcal A^{-1}\operatorname{Diag}(\nu)W.
\]

Le sous-espace engendré par les centres est donc invariant par \(V\). Comme \(V\)
est symétrique, ce sous-espace est une somme de sous-espaces propres.

En particulier, si \(\beta>0\) et \(V\succ0\), le noyau exponentiel est strictement défini positif
sur des centres distincts ; \(\mathcal A\) est donc toujours inversible. Pour un
spectre simple, tout équilibre à \(q\) clusters vit alors dans un sous-espace
engendré par au plus \(q\) vecteurs propres. Cela réduit l'énumération constructive
à un nombre fini de choix de sous-espaces, suivi des équations du Théorème 3.

Pour \(V\) négative ou indéfinie, la même conclusion vaut chaque fois que
\(\det\mathcal R\ne0\). Les singularités de \(\mathcal R\) constituent des branches
exceptionnelles supplémentaires, déjà incluses dans le Théorème 2.

## 7. Ce que donne l'énumération des petits systèmes

Une recherche multi-start a résolu toutes les équations tangentielles planaires
trouvées pour trois paramètres représentatifs. Deux graines indépendantes donnent
exactement les mêmes catalogues après quotient des permutations et des changements
globaux de signe des axes :

| spectre, \(\beta\) | \(n\) | équilibres distincts | stables |
|---|---:|---:|---:|
| \((2,-3),\,1.5\) | 2 | 7 | 3 |
| \((2,-3),\,1.5\) | 3 | 16 | 4 |
| \((-0.4,-4),\,0.03\) | 2 | 4 | 1 |
| \((-0.4,-4),\,0.03\) | 3 | 8 | 1 |
| \((3,2),\,1.5\) | 2 | 6 | 2 |
| \((3,2),\,1.5\) | 3 | 12 | 2 |

Pour \(n=3\), on trouve, en plus des consensus et états bipolaires :

- 8 branches à trois positions dans le cas indéfini ;
- 2 branches à trois positions dans le cas négatif ;
- 5 branches à trois positions dans le cas positif.

La plupart sont des selles asymétriques. Elles montrent pourquoi une liste limitée
aux familles visuellement dominantes n'est pas exhaustive. Les résidus maximaux
sont inférieurs à \(3\times10^{-14}\). Cette recherche numérique est un audit de
découverte, pas la preuve de complétude ; la preuve exhaustive est le Théorème 2.

## 8. Réponse finale sous forme d'algorithme

Pour trouver effectivement tous les équilibres d'une instance fixée
\((n,d,V,\beta)\) :

1. diagonaliser \(V\) et regrouper ses valeurs propres distinctes ;
2. résoudre le système PSD du Théorème 2 avec les contraintes de rang ;
3. factoriser chaque \(G_\alpha=C_\alpha C_\alpha^\top\) ;
4. concaténer les \(C_\alpha\) dans les espaces propres correspondants ;
5. quotienter \(\prod_\alpha O(m_\alpha)\), les permutations de jetons et le
   changement de signe global ;
6. calculer la Jacobienne tangentielle si la stabilité est demandée.

La formulation clusters du Théorème 3 est préférable quand le nombre de positions
distinctes \(q\) est petit. La formulation spectral-Gram est préférable pour les
espaces propres multiples, les noyaux et les familles continues.

## 9. Fichiers reproductibles

- `experiments/spectral_self_attention/equilibrium_catalogue.py` :
  caractérisation spectral-Gram, réduction par clusters, cas \(\beta=0\), simplexes,
  polygones et recherche planaire ;
- `experiments/spectral_self_attention/run_equilibrium_catalogue.py` :
  catalogue des petits systèmes et audit de saturation ;
- `data/spectral_self_attention/equilibria/planar_small_systems.csv` :
  racines, résidus et stabilité ;
- `data/spectral_self_attention/equilibria/search_saturation.csv` :
  concordance des recherches indépendantes ;
- `tests/test_spectral_self_attention.py` :
  tests des équations équivalentes et des familles explicites.
