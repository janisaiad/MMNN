# Bassins multi-têtes, grande dimension et lecture de physique statistique

## Ce qui a été simulé

On fixe cinq tokens sur le cercle ou sur la sphère. Un sixième token, appelé
**sonde**, est le seul à bouger. Pour une tête `h`, on utilise une matrice
symétrique `V_h` à la fois pour calculer qui la sonde écoute et pour calculer la
direction proposée par la tête :

\[
 a_j^{(h)}(x)=
 \frac{\exp(\beta x^\top V_h y_j)}
 {\sum_k\exp(\beta x^\top V_h y_k)},\qquad
 \dot x=(I-xx^\top)\sum_h\sum_j a_j^{(h)}(x)V_hy_j .
\]

La liste des `y_j` contient aussi la sonde elle-même. Les cinq autres `y_j`
restent fixes. Cette expérience donne une **carte conditionnelle** : elle répond
à « où va ce token si tout son environnement reste immobile ? ». Elle ne donne
pas directement les bassins du système où tous les tokens bougent ensemble.

Une destination est stable lorsque la somme des poussées des têtes n'a plus de
composante le long de la sphère :

\[
 (I-xx^\top)\sum_h f_h(x)=0.
\]

Il n'est pas nécessaire que chaque tête soit séparément à l'équilibre. Deux
têtes peuvent se neutraliser tangentiellement et créer une destination qui
n'existe pour aucune tête seule.

## Résultats des cartes de bassins

Sur le cercle, avec cinq ancres fixes, `beta=5`, et deux matrices ayant le même
spectre `(2,-3)` mais des axes tournés de 60 degrés :

| dynamique | destinations stables | parts des bassins, en % |
|---|---:|---|
| tête A | 4 | 30.8, 17.2, 27.2, 24.8 |
| tête B | 5 | 32.6, 18.9, 4.9, 18.6, 25.0 |
| A + B | 7 | 34.9, 9.7, 7.1, 6.8, 12.1, 5.7, 23.8 |

Sur la sphère, avec `beta=4` et deux matrices de spectre `(2,0.5,-3)` dont les
axes propres diffèrent :

| dynamique | destinations stables | parts des bassins, en % |
|---|---:|---|
| tête A | 4 | 38.0, 10.5, 21.2, 30.3 |
| tête B | 4 | 25.1, 28.2, 16.3, 30.5 |
| A + B | 5 | 16.9, 28.0, 8.4, 13.4, 33.3 |

Ce n'est donc pas une simple superposition de deux dessins. La somme peut
couper un bassin, déplacer les frontières et créer de nouvelles destinations.
Les petites zones expliquent aussi pourquoi deux départs très proches peuvent
finir loin l'un de l'autre : ils sont de part et d'autre d'une frontière.

## Grande dimension : ce qui se découple et ce qui ne se découple pas

Protocole : 24 tokens unitaires aléatoires, six têtes symétriques indépendantes,
matrices gaussiennes denses mises à l'échelle pour garder un spectre d'ordre un,
96 répétitions par dimension.

| dimension | proximité absolue entre tokens | proximité absolue entre forces de têtes | tokens effectivement écoutés sur 24 |
|---:|---:|---:|---:|
| 2 | 0.638 | 1.000 | 11.6 |
| 8 | 0.291 | 0.312 | 15.3 |
| 32 | 0.144 | 0.144 | 20.9 |
| 128 | 0.071 | 0.070 | 23.2 |
| 256 | 0.050 | 0.049 | 23.6 |

Pour deux directions unitaires choisies au hasard en dimension `d`, la taille
moyenne de leur produit scalaire absolu vaut asymptotiquement

\[
 \sqrt{\frac{2}{\pi d}}.
\]

Les données suivent cette loi. Les forces de têtes indépendantes suivent la
même décroissance. La norme de la somme de six forces, divisée par la racine de
la somme de leurs normes au carré, tend vers `1`. C'est la signature nette de
forces presque perpendiculaires. Si les six têtes étaient identiques, ce nombre
vaudrait `sqrt(6)=2.45`.

Mais **perpendiculaire ne signifie pas que l'attention s'éteint**. Avec une
matrice aléatoire sans structure et `beta=3` fixé, les scores deviennent tous
presque égaux. Le softmax devient donc presque uniforme : en dimension 256, une
tête écoute effectivement 23.6 tokens sur 24. Il reste une interaction moyenne
globale, pas 24 systèmes indépendants.

Le cas `V=I` est différent car le score de la sonde avec elle-même reste `1`,
alors que ses scores avec les autres tendent vers `0`. Pour `n` tokens :

\[
 a_{ii}\longrightarrow \frac{e^\beta}{e^\beta+n-1}.
\]

Avec `n=24`, le poids propre limite est environ `10.6%` pour `beta=1`, `70.4%`
pour `beta=4`, et `99.2%` pour `beta=8`. Le vrai auto-découplage demande donc
`beta` nettement plus grand que `log(n)`. Pour des matrices aléatoires sans
avantage diagonal, garder des scores non triviaux en grande dimension demande
en général de faire croître `beta` comme `sqrt(d)` dans cette normalisation.

Attention à la transposition vers un Transformer standard : ici les tokens ont
une norme unitaire et les matrices un spectre d'ordre un. Dans un Transformer,
les normes des requêtes et clés et la division par `sqrt(d_head)` sont choisies
pour conserver des logits d'ordre un. L'orthogonalité brute des tokens n'impose
donc pas, à elle seule, un découplage dans un réseau réel ou entraîné.

## Le parallèle de Boltzmann, précisément

Chaque ligne d'attention est exactement une loi de Boltzmann conditionnelle :

\[
 a_j^{(h)}=\frac{e^{-E_j^{(h)}/T_h}}{\sum_k e^{-E_k^{(h)}/T_h}},
 \qquad E_j^{(h)}=-x^\top B_hy_j,\quad T_h=1/\beta_h.
\]

`beta` joue bien le rôle d'un froid : petit `beta` répartit l'attention, grand
`beta` privilégie les meilleurs scores. Mais le système simulé n'est pas un bain
thermique complet : les poids sont calculés comme des probabilités de Boltzmann,
puis les tokens suivent une trajectoire déterministe. Ils ne tirent pas au
hasard un état selon une loi de Gibbs.

Pour obtenir un vrai modèle thermique, il faudrait ajouter un bruit brownien
tangent à la sphère. Dans le cas où la dérive est réellement le gradient d'une
énergie et où l'intensité du bruit est accordée à la mobilité, la distribution
stationnaire devient une loi de Gibbs. Pour des têtes générales, non symétriques
ou mal alignées entre score et valeur, il peut ne plus exister une seule énergie
globale : on a alors un système hors équilibre avec courants persistants.

## Systèmes voisins en physique et en mémoire associative

- **Réseaux de Hopfield modernes** : c'est le parallèle le plus direct. Une
  requête choisit parmi des souvenirs par softmax; les états stables peuvent
  être un souvenir, une moyenne de souvenirs ou un état métastable.
- **Modèles XY et de Heisenberg** : des spins unitaires vivent respectivement
  sur un cercle ou une sphère. Ici, la différence importante est que les
  couplages d'attention changent avec l'état et sont normalisés ligne par ligne.
- **Kuramoto et Lohe** : ils décrivent la synchronisation sur le cercle ou sur
  des sphères. Notre projection tangentielle a la même géométrie, mais le
  softmax produit une interaction beaucoup plus sélective.
- **Potts** : lorsque seules les étiquettes finales des bassins comptent, les
  destinations jouent le rôle de couleurs discrètes.
- **Verres de spins** : plusieurs matrices aléatoires ou incompatibles peuvent
  produire frustration et nombreux états métastables. Toutefois, notre régime
  aléatoire à `beta` fixé devient surtout uniforme. Pour voir un paysage
  franchement vitreux, les expériences pertinentes sont `beta ~ sqrt(d)`, des
  matrices avec pics de faible rang, des têtes non commutantes, ou du bruit.

## Fichiers reproductibles

- `experiments/spectral_self_attention/multihead_phase.py`
- `data/spectral_self_attention/multihead/frozen_probe_basins.json`
- `data/spectral_self_attention/multihead/high_dimension_decoupling.csv`
- `tests/test_multihead_phase.py`

Références primaires utiles :

- Kuehn et Yoon, *Spectral Selection in Symmetric Self-Attention Dynamics*,
  <https://arxiv.org/abs/2604.26085>
- Pendharkar, *Gradient Flow Structure and Quantitative Dynamics of Multi-Head
  Self-Attention*, <https://arxiv.org/abs/2605.04279>
- Ramsauer et al., *Hopfield Networks is All You Need*,
  <https://arxiv.org/abs/2008.02217>
- Lipton, Mirollo et Strogatz, *The Kuramoto model on a sphere*,
  <https://arxiv.org/abs/1907.07150>
- Tiberi et al., *Dissecting the Interplay of Attention Paths in a Statistical
  Mechanics Theory of Transformers*, <https://arxiv.org/abs/2405.15926>
