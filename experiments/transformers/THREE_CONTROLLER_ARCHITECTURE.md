# Architecture retenue : une géométrie, trois contrôleurs

## Tronc commun

Les trois modèles utilisent exactement le même pipeline :

\[
\mathcal C\longrightarrow(\widehat G,\widehat b)
\longrightarrow Q_\theta(\mathcal C)
\longrightarrow B_\theta(\mathcal C)
\longrightarrow\widehat z_L.
\]

Le préconditionneur reste factorisé et fixe pendant la boucle. Le HVP
\(v\mapsto G^\top(Gv)+\lambda v\) est une contraction d'attention linéaire
fixe. Puisque \(mR>K\), le calcul principal est primal ; le dual comprimé
donne l'interprétation tokenique équivalente.

## Cellules conservées

| Cellule | Appris en plus de la tête | État | Divisions | Rôle |
|---|---|---:|---:|---|
| Heavy-Ball | deux scalaires stables, ou un intervalle prédit | deux vecteurs | non | architecture principale |
| Chebyshev | \([\widehat\mu,\widehat L]\) via un petit MLP | deux vecteurs + scalaires | récurrence fixe | meilleur polynôme non adaptatif |
| PCG | rien | cinq vecteurs/scalaires | quotients exacts | plafond numérique par HVP |
| HB certifié → PCG | seuil résiduel fixe | état HB, PCG seulement en queue | quotients au fallback | décodeur robuste final |

La récurrence Chebyshev est hardcodée. Le MLP prédit seulement un log-centre
et une log-largeur ; il n'effectue ni multiplication ni division. Une loss
unilatérale pénalise les intervalles qui ne couvrent pas le vrai spectre.
La même tête scalaire peut rendre HB adaptatif : elle prédit
\([\widehat\mu,\widehat L]\), puis les relations fixes
\[
\alpha=\frac{4}{(\sqrt{\widehat L}+\sqrt{\widehat\mu})^2},\qquad
\beta=\left(\frac{\sqrt{\widehat L}-\sqrt{\widehat\mu}}
{\sqrt{\widehat L}+\sqrt{\widehat\mu}}\right)^2
\]
construisent ses deux poids. Le MLP ne remplace donc toujours aucune opération
de la boucle.

## Règle de sélection

Avec le même \(B_\theta\), PCG minimise l'erreur dans l'espace de Krylov :

\[
\|e_L^{\rm PCG}\|_H
\leq
\min\{\|e_L^{\rm HB}\|_H,\|e_L^{\rm Cheb}\|_H\}.
\]

Heavy-Ball est retenu comme architecture finale si son risque **end-to-end**
est au plus 10 % supérieur à PCG, ou si les intervalles de confiance se
recouvrent, et si son temps de solve est inférieur. Quand PCG atteint zéro sur
le sous-problème numérique, on ne divise pas par cette erreur : l'erreur
solveur HB doit alors être négligeable devant le plancher encodeur/statistique
mesuré séparément. Chebyshev ne remplace HB que s'il réduit l'écart de manière
robuste avec une couverture spectrale fiable. Richardson reste la baseline
\(\beta=0\).

Cette règle est réalisée directement par le contrôleur
`certified_hb_pcg`. Après les \(L\) blocs HB, il calcule
\[
\eta_L=\frac{r_L^\top B_\theta r_L}{c^\top B_\theta c}.
\]
Si \(\eta_L\leq10^{-8}\), la sortie HB est conservée. Sinon, le même PCG
exact et le même préconditionneur sont exécutés uniquement sur les indices
fautifs. Le routage est un masque déterministe : aucun réseau ne choisit le
solveur.

## Protocole sans explosion combinatoire

On entraîne une tête avec chaque contrôleur, puis on croise les trois têtes
avec les trois cellules sans réentraînement. Les neuf évaluations utilisent les
mêmes prompts, seeds et budgets de HVP. On rapporte seulement : erreur de
projecteur, couverture Chebyshev, erreur \(H\), temps par solve et erreur PDE.

## Muon

Muon est un optimiseur externe, pas un quatrième solveur. AdamW reste la
référence du petit prototype. Sur l'encodeur complet, Muon pourra être testé
uniquement sur les grandes matrices cachées ; dictionnaire physique,
embeddings/slot queries, gains, biais, scalaires HB et sortie spectrale restent
sous AdamW. L'ablation garde architecture, batches, seeds et budget identiques
et mesure le temps jusqu'à une erreur de sous-espace donnée.

Le projet utilise actuellement PyTorch 2.7.1, sans `torch.optim.Muon`. Ajouter
une dépendance ou une copie locale n'est pas justifié avant le branchement du
grand encodeur.

## Évidence contrôlée actuelle

Distribution : prompts `pde_elliptic_correlated`, \(K=8\), \(m=32\), quatre
HVP, batch d'évaluation fixé.

- La tête rang 2 entraînée sous PCG a \(\kappa_{\rm eff}\approx1080\). Avec ce
  même préconditionneur, PCG donne une erreur \(H\) relative
  \(3.1\cdot10^{-4}\), Chebyshev oracle \(5.9\cdot10^{-1}\), et HB oracle
  environ \(25\).
- Après entraînement HB avec une pénalité \(0.05\log\kappa\), la tête rang 2
  descend à \(\kappa_{\rm eff}\approx211\), mais la MSE HB reste \(0.193\)
  contre \(0.058\) lorsque la même tête est utilisée dans PCG. Le rang 2 échoue
  donc au critère de non-infériorité.
- Un oracle spectral rang 4 échoue encore à quatre couches. Un oracle rang 6
  corrigeant cinq directions lentes et une rapide donne
  \(\kappa_{\rm eff}\approx4.3\). HB atteint alors une MSE
  \(2.8\cdot10^{-3}\) à profondeur 4, \(8.1\cdot10^{-6}\) à profondeur 8 ;
  Chebyshev atteint respectivement \(1.8\cdot10^{-3}\) et
  \(5.0\cdot10^{-6}\).
- Runtime solveur seul sur GPU, float64, batch 2048, préconditionneur déjà
  construit : médianes PCG-4 \(8.5\) ms, HB-8 \(15.7\) ms, Chebyshev-8
  \(25.4\) ms.
- La tête d'intervalle Chebyshev (sept statistiques de prompt, MLP caché de
  largeur 16) a d'abord été entraînée uniquement pour couvrir le spectre, puis
  affinée 750 pas avec l'erreur solveur. Pour \(m=16,32,64,128\), elle conserve
  respectivement \(99.7\%,100\%,99.8\%,96.8\%\) de couverture et obtient une
  erreur \(H\) relative \(0.101,0.100,0.105,0.100\). C'est nettement meilleur
  que l'intervalle minimax formé seulement par les deux valeurs propres
  extrêmes (environ \(0.58\)--\(0.61\) sur cette distribution), parce que la
  loss apprend un compromis distributionnel. L'arithmétique de Chebyshev reste
  entièrement fixe. PCG avec le même préconditionneur reste cependant entre
  \(3.8\cdot10^{-4}\) et \(7.8\cdot10^{-4}\).

Conclusion du solveur isolé : PCG gagne pour le décodeur explicite actuel. HB reste
le candidat loop-Transformer seulement si son erreur à huit couches est sous
le plancher end-to-end et si une réalisation réelle en blocs Transformer rend
son coût inférieur à celui du macro-bloc PCG multi-phase. Ces deux conditions
ne sont pas encore démontrées. Le résultat Chebyshev valide la séparation
« apprendre la géométrie/les coefficients, hardcoder l'algèbre ». Le test
end-to-end suivant décide si cet écart numérique est matériel ou non.

## Test end-to-end décisif

Le décodeur a ensuite été replacé dans le problème ICL complet
\( (f_i,u_i)_{i=1}^m,f_\star\mapsto u_\star \). Une tête softmax, six requêtes
de sous-espace, un Ritz exact et huit HVP sont entraînés avec le dictionnaire
encodeur. Le run HB atteint une MSE \(u_\star\) \(2.34\cdot10^{-8}\) dès 1000
pas et \(2.95\cdot10^{-8}\) à 5000 pas, contre \(3.42\cdot10^{-8}\) pour le
run historique avec solve directe (batches d'évaluation différents).

Le contrôle propre gèle ensuite encodeur, tête, prompts et budget, et change
seulement la cellule. Sur 4096 nouvelles tâches communes :

| Cellule | MSE end-to-end | erreur solveur seule sur \(u\) |
|---|---:|---:|
| HB adaptatif | \(3.116\cdot10^{-8}\) | \(3.59\cdot10^{-12}\) |
| PCG | \(3.108\cdot10^{-8}\) | \(6.70\cdot10^{-14}\) |
| Richardson, même pas que HB | \(3.296\cdot10^{-8}\) | -- |

Les demi-largeurs des IC 95 % HB et PCG valent environ \(1.01\cdot10^{-9}\)
et se recouvrent presque entièrement. HB est donc quasi-PCG selon la règle
préenregistrée ; son erreur solveur vaut environ \(1.2\cdot10^{-4}\) de la MSE
totale.

Sur GPU float32, batch 2048, \(K=8\), \(M=512\), huit HVP et préconditionneur
commun déjà construit, les médianes sont \(0.982\) ms pour la cellule HB et
\(2.186\) ms pour PCG. La politique spectrale HB complète (construction des
sept statistiques, MLP et coefficients) coûte \(0.546\) ms une fois par
prompt. HB adaptatif totalise donc environ \(1.53\) ms, encore \(30\%\) plus
rapide que PCG dans cette réalisation. Les synchronisations de diagnostic sont
exclues des deux mesures.

La petite tête d'intervalle HB a été ajustée seule, encodeur et préconditionneur
gelés, sur \(z_{\rm scale}\in[0.1,1]\). Elle conserve la non-infériorité pour
\(m=8,16,32\), un bruit prompt \(0.01\), et à amplitude OOD \(1.0\). Dans ce
dernier cas, HB obtient \(3.361\cdot10^{-4}\) contre PCG
\(3.353\cdot10^{-4}\), soit \(0.22\%\) d'écart avec des IC bien plus larges.
Une vérification séparée sur 8192 tâches nominales et 8192 tâches à amplitude
\(1.0\) ne trouve aucune violation de la condition de Jury
\(2(1+\beta)-\alpha\lambda_{\max}>0\) ; les marges minimales observées sont
respectivement \(0.557\) et \(0.440\). La couverture complète de l'intervalle
prédit n'est donc pas utilisée comme substitut à la vraie condition de
stabilité HB.

La politique a ensuite reçu le même ajustement de queue
\(z_{\rm scale}\in[0.5,1]\) et une calibration de sécurité au quantile 99 %
sur trois encodeurs indépendants. Les ratios de MSE HB/PCG sont :

| Seed | nominal | amplitude \(1.0\) | violations de Jury |
|---:|---:|---:|---:|
| 0 | 1.0014 | 1.0008 | 0/4096 |
| 1 | 1.0046 | 1.0077 | 0/4096 |
| 2 | 1.0033 | 1.0188 | 0/4096 |

Le seed pilote 0 a reçu 5000 pas encodeur contre 2000 pour les deux
réplications ; ce tableau établit la reproductibilité du critère quasi-PCG,
mais n'est donc pas présenté comme une comparaison de trois budgets
strictement identiques. Les marges de Jury minimales OOD sont
\(0.512,0.422,0.453\).

Les nombres complets sont dans `end_to_end_loop_controller_results.csv`,
`multi_seed_adaptive_hb_pcg_results.csv` et
`end_to_end_loop_runtime_results.csv`.

Décision mise à jour : **HB adaptatif est retenu comme loop-Transformer
principal ; PCG reste le contrôle numérique et Chebyshev le contrôle
polynomial.** Cette décision porte sur le risque end-to-end dans la famille
testée, pas sur une prétention à dominer PCG par HVP en arithmétique exacte.

## Validation elliptique avec sous-espace physique appris

La famille `elliptic_1d` apprend un sous-espace de perturbations de diffusion
de rang huit, tandis que \(A_0\) est fixé et que chaque atome est projeté sur
la structure elliptique connue. Une ridge euclidienne sur les coordonnées est
incorrecte dans cette paramétrisation : deux bases du même sous-espace donnent
deux pénalisations physiques différentes. Avec la métrique covariante
\[
M_{ij}=\langle A_i,A_j\rangle_F,
\qquad H=G^\top G+\lambda M,
\]
la prédiction devient invariante aux changements de base du dictionnaire.
Après 250 pas de fine-tuning covariant, le recouvrement de sous-espace vaut
`0.99999994`, la MSE exacte apprise vaut \(9.56\cdot10^{-10}\), et le plancher
avec vrai dictionnaire vaut \(8.83\cdot10^{-10}\). Le goulot précédent venait
donc de la jauge de régularisation, pas du décodeur.

Avec cet encodeur gelé, une tête softmax, six slots et 16 HVP, l'audit commun
donne :

| Régime | HB | HB certifié → PCG | PCG | taux fallback |
|---|---:|---:|---:|---:|
| \(z_{\rm scale}=0.5\), 4096 tâches | \(9.2078\cdot10^{-10}\) | \(9.2078\cdot10^{-10}\) | \(9.2064\cdot10^{-10}\) | 0 % |
| \(z_{\rm scale}=1\), 8192 tâches | \(2.2989\cdot10^{-9}\) | \(1.5906\cdot10^{-9}\) | \(1.5893\cdot10^{-9}\) | 0.171 % |

Au nominal, HB pur satisfait directement le critère quasi-PCG. En OOD, les
médianes et quantiles 99 % de HB restent au niveau de PCG, mais quelques
opérateurs très sensibles rendent la moyenne HB pure non fiable. Le certificat
élimine cette queue pour un écart moyen final de 0.079 % à PCG. Cependant, le
fallback compacté naïvement avec `nonzero` impose une synchronisation GPU : la
cellule hybride mesurée prend 2.85 ms au nominal et 3.59 ms en OOD, contre
environ 0.77 ms pour PCG-16. Elle est donc conservée comme mode de sûreté ou
file de reprise asynchrone, pas comme chemin GPU synchrone rapide.

La variante finalement retenue supprime aussi le MLP spectral. Encodeur et
tête gelés, seuls deux scalaires globaux ont été ajustés avec une loss moyenne
plus CVaR de queue et une contrainte de Jury. On compare alors à budget de
latence : HB effectue 32 HVP simples pendant que PCG effectue 16 HVP avec ses
réductions et quotients.

| Régime | HB-32 global robuste | PCG-16 | ratio HB/PCG |
|---|---:|---:|---:|
| (z_{\rm scale}=0.5), 4096 tâches | (9.1713\cdot10^{-10}) | (9.1720\cdot10^{-10}) | 0.9999 |
| (z_{\rm scale}=1), 8192 tâches | (1.6247\cdot10^{-9}) | (1.5599\cdot10^{-9}) | 1.0415 |

Les IC 95 % OOD se recouvrent et aucune violation de Jury n'est observée ; la
marge minimale vaut 0.815. Sur GPU float32, batch 2048, (K=8,M=512), les
médianes des cellules sont 0.777 ms pour HB-32 et 0.766 ms pour PCG-16. Elles
sont donc équivalentes en latence à ce niveau de mesure, tandis que HB conserve
un état et une algèbre plus simples.

Décision elliptique : **retenir HB-32 à deux scalaires, sans MLP, comme
architecture principale tant que le ratio end-to-end reste sous 1.10 ; garder
PCG-16 comme contrôle numérique et mécanisme de sûreté.** Cette sélection est
latency-matched et non HVP-matched. Richardson est nettement dominé.

## Audit causal du MLP Chebyshev

Chebyshev reste bien une architecture bouclée dans sa forme théorique. Le MLP
est appliqué une fois à un token spectral construit depuis le prompt et produit
seulement \([\widehat\mu,\widehat L]\). Le bloc lié conserve les tokens
\((z_\ell,z_{\ell-1},\alpha_\ell,\beta_\ell)\) et met exactement à jour les
deux derniers. La forme fermée vectorisée du calendrier
\((\alpha_\ell,\beta_\ell)_{\ell<L}\) est une compilation algébriquement
équivalente de cette récurrence, pas une approximation supplémentaire.

Pour vérifier que le MLP fait réellement de l'ICL plutôt qu'apprendre un
intervalle global, quatre contrôles partagent encodeur, tête softmax, prompts
et préconditionneur : conditionnement correct, features permutées entre
tâches, intervalle constant calibré sur un ensemble disjoint, et bornes oracle.
Sur 8192 tâches elliptiques OOD :

| Contrôleur | MSE \(u_\star\) | ratio à PCG-16 | couverture |
|---|---:|---:|---:|
| PCG-16 | \(1.5659\cdot10^{-9}\) | 1.0000 | -- |
| HB global-32 | \(1.7784\cdot10^{-9}\) | 1.1357 | -- |
| HB global-40 | \(1.5915\cdot10^{-9}\) | 1.0164 | -- |
| Chebyshev-32 conditionné | \(1.6451\cdot10^{-9}\) | 1.0506 | 98.22 % |
| Chebyshev-32, features permutées | \(2.0417\cdot10^{-3}\) | \(1.30\cdot10^6\) | 83.03 % |
| Chebyshev-32, intervalle constant | \(1.0542\cdot10^{-3}\) | \(6.73\cdot10^5\) | 98.13 % |
| Chebyshev-32 oracle | \(1.5659\cdot10^{-9}\) | 1.0000 | 100 % |

Le contrôle constant est décisif : sa couverture globale est presque la même
que celle du MLP, mais il couvre les mauvaises tâches et échoue sur une queue
physiquement très sensible. Le MLP apprend donc une correspondance utile entre
le prompt et les poids de la boucle, sans adaptation de paramètres à
l'inférence. C'est un effet ICL causal, pas seulement une meilleure
approximation moyenne du spectre.

La décision principale reste néanmoins **HB-40** : il est plus simple, ne
contient aucun MLP, obtient un écart OOD de 1.64 % à PCG et sa théorie se réduit
à un filtre polynomial indépendant du second membre. Chebyshev-MLP est retenu
comme contrôle adaptatif ICL et comme option pour les familles où une
profondeur HB fixe ne couvre pas la queue spectrale. PCG reste le contrôle
numérique, non le modèle dont on cherche à faire scaler la théorie replica.
