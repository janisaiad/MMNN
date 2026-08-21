# Audit massif : attention sphérique suivie d'un MLP quadratique

## Conclusion en une phrase

Les quatre familles peuvent avoir plusieurs configurations finales stables et des
cycles stables de trois ou quatre couches. La séparation décisive n'est pas
« potentiel ou non » : c'est surtout la taille du pas et l'endroit où la rotation
est introduite. Dans le type 1, les cycles sont un effet de couches assez fortes ;
dans le type 3, une rotation stable peut survivre quand chaque couche devient
infinitésimale.

## Ce qui a réellement été testé

Une couche est appliquée en série : attention, remise sur la sphère, MLP
quadratique, puis nouvelle remise sur la sphère. Les quatre familles sont :

| type | attention | MLP |
|---|---|---|
| 1 | `QᵀK = V = Vᵀ` | dérive d'un potentiel |
| 2 | `QᵀK = V = Vᵀ` | quadratique général |
| 3 | `QᵀK ≠ V` | dérive d'un potentiel |
| 4 | `QᵀK ≠ V` | quadratique général |

Pour les types 3 et 4, quatre cas ont été séparés : score et valeur symétriques,
score symétrique avec valeur générale, score général avec valeur symétrique, et
les deux généraux. Tester séparément les factorisations de `QᵀK` n'apporterait
rien à la dynamique gelée : celle-ci ne voit que leur produit.

Le recensement principal contient 28 672 modèles et 2 752 512 trajectoires, avec
1 à 4 jetons, 96 départs par modèle,
1 400 couches de chauffe, des largeurs MLP 1, 2, 4 et 8, cinq régimes de pas,
cinq régimes de température et cinq rapports de force attention/MLP. Le type 3 a
été sur-échantillonné : 16 384 modèles et 1 572 864 trajectoires.

« Exhaustif » signifie ici exhaustif sur cette taxonomie et ces grilles, avec de
nombreux tirages dans chaque case. Cela ne signifie pas une preuve par balayage de
tout l'espace continu des matrices. Les équations exactes données plus bas, elles,
couvrent tout cet espace.

## Configurations fixes trouvées par résolution directe

Les nombres pour deux ou trois jetons sont des bornes inférieures : on résout les
équations depuis une grille dense et 180 départs aléatoires, mais une racine dont
le bassin de l'algorithme est minuscule peut échapper au balayage.

| type | jetons | fixes stables moyens | maximum | fixes stables non consensus moyens |
|---|---:|---:|---:|---:|
| 1 | 1 / 2 / 3 | 1,68 / 2,34 / 3,09 | 2 / 4 / 8 | 0 / 0,80 / 1,58 |
| 2 | 1 / 2 / 3 | 1,49 / 2,03 / 2,55 | 2 / 5 / 9 | 0 / 0,59 / 1,19 |
| 3 | 1 / 2 / 3 | 1,57 / 2,13 / 2,61 | 3 / 4 / 8 | 0 / 0,66 / 1,18 |
| 4 | 1 / 2 / 3 | 1,31 / 1,71 / 2,00 | 2 / 4 / 7 | 0 / 0,48 / 0,81 |

Même le type 1, entièrement construit à partir de deux sous-étapes de potentiel,
produit parfois un retour en spirale vers un point fixe. Deux cartes de gradient
appliquées l'une après l'autre ne forment pas forcément une seule carte de
gradient.

Le test construit à trois puits donne en plus exactement `3^n` configurations
stables étiquetées : 3, 9, 27, 81 et 243 pour 1 à 5 jetons. Le nombre de fins
possibles peut donc croître de façon combinatoire, même dans le type 1.

## Cycles stables

Un cycle de période 3 signifie : après trois couches on revient exactement au
même état, mais ni après une ni après deux. Chaque exemple a été raffiné en
résolvant directement `F³(x)=x` ou `F⁴(x)=x`. La stabilité a été contrôlée en
perturbant tout le cycle ; un rayon inférieur à 1 signifie que la perturbation se
rétracte.

| type | trajectoires p3 | modèles ayant un p3 | rayon du p3 certifié | trajectoires p4 | modèles ayant un p4 | rayon du p4 certifié |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 321 | 8 | 0,311 | 201 | 15 | 0,654 |
| 2 | 2 039 | 34 | 0,055 | 1 823 | 28 | 0,512 |
| 3 | 4 490 | 75 | 0,454 | 4 507 | 113 | 0,803 |
| 4 | 1 621 | 22 | 0,448 | 1 860 | 41 | 0,091 |

Des cycles primitifs stables de périodes 5 à 12 ont aussi été certifiés dans le
balayage. Ils ne sont donc pas de simples erreurs où un cycle de période 2 aurait
été compté comme période 4.

## Le motif expérimental

1. **Le pas de couche commande la transition.** Dans le type 1, aucun cycle de
   période 3 ou 4 n'est observé sous `h=0,6` dans le grand balayage. Ils apparaissent
   ensuite dans des fenêtres, pas à une valeur isolée. Dans un modèle type 1, le
   p3 occupe approximativement `h=1,229–1,800`; dans un autre, le p4 occupe
   `h=1,421–1,523`. Les exemples du type 3 donnent des fenêtres encore larges :
   `h=0,409–1,426` pour un p3 et `h=1,129–1,650` pour un p4.
2. **Le nombre de jetons n'impose pas la période.** Des p3 et p4 apparaissent avec
   un seul jeton comme avec plusieurs. Les interactions ajoutent des branches et
   changent leurs bassins, mais il n'existe pas de règle « trois jetons donnent un
   triangle temporel ».
3. **La largeur du MLP n'ordonne pas la complexité.** Les largeurs 1, 2, 4 et 8
   contiennent toutes des cycles ; augmenter la largeur ne fait pas monter leur
   fréquence de façon monotone.
4. **L'exponentielle du softmax n'est pas nécessaire.** En forçant `β=0`, donc une
   attention uniforme, les quatre types possèdent encore des p3 et p4 primitifs
   et stables.
5. **Le MLP général peut le faire seul ; le MLP potentiel, pratiquement non.** Sur
   65 536 trajectoires sans attention, le MLP général donne des p3 et p4 certifiés.
   Le MLP potentiel ne donne aucun p3 certifié et son unique candidat p4 se réduit
   à une période plus courte lors du raffinement. Pour le type 1, la richesse vient
   donc surtout de la composition attention puis MLP, pas du MLP potentiel isolé.
6. **Une attention beaucoup plus forte que le MLP favorise surtout les p2.** Les
   p3/p4 sont les plus présents lorsque les deux forces sont comparables ou que le
   MLP domine. La température et la largeur ont un effet non monotone.
7. **Dans le type 3, la matrice de valeur est le levier principal de circulation.**
   Quand `V` est générale, environ 9,3 % des trajectoires restent non périodiques
   après la chauffe ; quand `V` est symétrique, environ 2,1 %. Rendre seulement la
   matrice de score générale ne reproduit pas cet écart. Les p3 et p4 existent
   toutefois dans les quatre sous-types : une valeur non symétrique n'est pas
   nécessaire aux cycles de couche finie, mais elle l'est au mécanisme continu
   simple décrit ci-dessous.

Les balayages de l'exposant de séparation confirment que tous les p3/p4 classés ont
un exposant négatif : ils attirent vraiment. Parmi les trajectoires non classées,
une petite fraction sépare au contraire deux départs presque identiques. Douze cas
par type ont été repris pendant deux fois 5 000 couches : les 48 restent positifs,
ne ferment aucun cycle de période au plus 12 et couvrent respectivement 3, 3, 4 et
3 matrices distinctes. C'est une preuve numérique forte de chaos dans les quatre
types, y compris le type 1 entièrement potentiel par sous-étapes, mais pas un
théorème pour toutes les matrices.

## Pourquoi le type 3 est différent

Prenons des scores nuls, une valeur

```text
V = a I + ω J,       J = rotation de 90 degrés,
```

et un MLP potentiel à trois puits, de force `g`. Quand les jetons se rejoignent,
leur angle commun suit, à la limite des petites couches,

```text
dθ/dt = ω - g sin(3θ).
```

Si `|ω| > |g|`, le MLP ralentit puis accélère le mouvement, mais ne peut jamais
l'arrêter. La valeur moyenne de la vitesse est exactement
`sqrt(ω²-g²)`. Avec `ω=0,8` et `g=0,25`, la prédiction est `0,75993`.
L'expérience donne 0,5909, 0,6684, 0,7115, 0,7404, 0,7503 et 0,7547 quand le pas
passe de 0,2 à 0,005. Les 128 départs et les quatre jetons se synchronisent tous.
Les deux contrôles symétriques s'arrêtent exactement.

C'est la différence essentielle : les p3/p4 du type 1 disparaissent quand on
raffine assez la couche, tandis que la rotation du type 3 converge vers un cycle
du temps continu. Un MLP potentiel n'empêche donc pas un cycle si `QᵀK` et `V` ne
décrivent pas la même interaction.

## Équations qui couvrent tous les points fixes et tous les cycles

Notons `N(z)=z/||z||`, `A(X)` la sortie de l'attention et `M(x)` le MLP. Une couche
est exactement

```text
Y_i = N(X_i + h A_i(X))
F_i(X) = N(Y_i + h M(Y_i)).
```

Toutes les configurations fixes sont exactement les solutions de `F(X)=X`.
En introduisant deux longueurs positives `r_i,s_i`, on retire les divisions :

```text
X_i + h A_i(X) = r_i Y_i
Y_i + h M(Y_i) = s_i X_i.
```

Pour tous les cycles de période `p`, on écrit les mêmes deux lignes pour
`X⁰,…,Xᵖ⁻¹` et on ferme la boucle avec `Xᵖ=X⁰`. Pour demander une période primitive,
on exclut simplement les retours plus courts. L'exponentielle n'empêche pas cette
formulation : elle reste seulement dans `A(X)`.

Quand `h` tend vers zéro, la condition fixe devient la condition tangentielle

```text
(I - X_i X_iᵀ) [A_i(X) + M(X_i)] = 0.
```

Elle dit sans vocabulaire abstrait : la somme des deux poussées n'a plus aucune
composante capable de faire glisser le jeton sur la sphère.

## Fichiers reproductibles

- `experiments/spectral_self_attention/large_scale_cycle_census.py` : recensement
  vectorisé des attracteurs ;
- `experiments/spectral_self_attention/equilibrium_root_census.py` : résolution
  directe des points fixes ;
- `experiments/spectral_self_attention/periodic_orbit_audit.py` : certificat des
  cycles primitifs et de leur stabilité ;
- `experiments/spectral_self_attention/period_bifurcation_sweep.py` : fenêtres en
  fonction du pas ;
- `experiments/spectral_self_attention/lyapunov_census.py` : séparation entre
  cycles, rotations lentes et candidats chaotiques ;
- `experiments/spectral_self_attention/type3_continuous_rotation.py` : limite en
  petites couches du type 3.

Les résultats agrégés sont dans
`data/spectral_self_attention/large_taxonomy_summary.json`.
