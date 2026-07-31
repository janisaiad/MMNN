# Programme first-principles : ICL, problème inverse et récupération de sous-espace

## 1. Le problème scientifique, avant le choix du solveur

Une tâche est un opérateur elliptique appartenant à une famille affine de
faible dimension,

\[
A(z)=A_0+\sum_{k=1}^K z_k A_k,
\qquad
T=\operatorname{span}(A_1,\ldots,A_K).
\]

Le prompt contient des couples \((f_i,u_i)\) produits par le même opérateur
caché. Après passage à la forme faible avec des fonctions test, il fournit

\[
G_{(i,r),k}=\langle A_k u_i,v_r\rangle,
\qquad
b_{(i,r)}=\langle f_i-A_0u_i,v_r\rangle.
\]

Le décodeur résout donc, in-context et sans modifier ses poids,

\[
z_\star=\arg\min_z
\frac{\tau}{2}\|Gz-b\|_2^2+\frac{\lambda}{2}\|z\|_2^2,
\quad
H z_\star=c,
\quad
H=\tau G^\top G+\lambda I,
\quad
c=\tau G^\top b.
\]

Le but n'est pas « apprendre PCG ». Le but est de récupérer la géométrie de
faible rang induite par le prompt, puis de l'exploiter avec un solveur exact et
minimal.

## 2. Deux sous-espaces à ne pas confondre

Le sous-espace physique

\[
T=\operatorname{span}(A_1,\ldots,A_K)
\]

est la quantité identifiable de l'encodeur/dictionnaire. Les coordonnées ne
le sont qu'à la transformation

\[
(A_1,\ldots,A_K,z)\mapsto
((A_1,\ldots,A_K)Q,Q^{-1}z),\qquad Q\in GL_K.
\]

La métrique scientifique est donc une distance entre projecteurs sur
\(T\) et \(\widehat T\), pas une erreur brute entre deux dictionnaires.

Il faut en outre fixer la jauge avant de poser la ridge. Sous
\((A_1,\ldots,A_K)\mapsto(A_1,\ldots,A_K)Q\), on a \(G\mapsto GQ\) et
\(z\mapsto Q^{-1}z\). La pénalité \(\lambda\|z\|^2\) n'est pas invariante
pour un \(Q\in GL_K\) général. Deux formulations cohérentes sont possibles :

1. orthonormaliser la dictionnaire apprise dans la métrique physique des
   opérateurs, ce qui réduit l'ambiguïté à \(O(K)\) et rend \(\lambda I\)
   intrinsèque ;
2. utiliser \(\lambda z^\top Mz\), avec
   \(M_{ij}=\langle A_i,A_j\rangle_{\mathrm{op}}\) et la transformation
   covariante \(M\mapsto Q^\top M Q\).

Le prototype minimal retient la première option. Sans ce choix, une erreur de
coordonnées et une erreur physique seraient mélangées dans la loss du
décodeur.

Pour une tâche et un prompt fixés, le sous-espace observable est

\[
S(G)=\mathcal R(G^\top),
\qquad r=\operatorname{rank}(G).
\]

Il vit dans l'espace des coefficients. Il dépend à la fois de la dictionnaire
apprise et de l'excitation fournie par le prompt. C'est ce sous-espace qui
contrôle le système normal et le préconditionnement du décodeur.

Une tête qui apprend seulement quelques vecteurs propres de \(H\) ne prouve
pas qu'elle a récupéré \(T\). Les deux erreurs doivent être mesurées
séparément :

\[
E_T=\frac{\|(I-P_{\widehat T})T\|_F}{\|T\|_F},
\qquad
E_S=\|P_{S(G)}-P_{\widehat S_\theta(\mathcal C)}\|_{\mathrm{op}}.
\]

## 3. Le préconditionneur découle de la structure low-rank

Écrivons une SVD mince \(\sqrt\tau G=U\Sigma V^\top\). Alors

\[
H^{-1}
=
\lambda^{-1}I
-
V\,\operatorname{diag}\!\left(
\frac{\sigma_j^2}{\lambda(\lambda+\sigma_j^2)}
\right)V^\top.
\]

L'écart à \(\lambda^{-1}I\) est donc une correction négative de rang
\(r\), portée exactement par \(S(G)=\mathcal R(G^\top)\). Ceci explique
pourquoi une correction uniquement positive n'était pas adaptée dans les
premières ablations.

La tête unique doit produire une base orthonormée
\(Q_\theta(\mathcal C)\in\mathbb R^{K\times s}\). Toute l'algèbre suivante
est fixée :

\[
T_\theta=Q_\theta^\top H Q_\theta,
\]

\[
B_\theta
=
\lambda^{-1}(I-Q_\theta Q_\theta^\top)
+Q_\theta T_\theta^{-1}Q_\theta^\top.
\]

Cette formule est SPD. Elle coûte \(O(Ks+s^3)\) par construction et
\(O(Ks+s^2)\) par application si elle reste factorisée. Il ne faut pas former
une matrice dense \(K\times K\).

Si \(s=r\) et \(\mathcal R(Q_\theta)=S(G)\), alors

\[
B_\theta=H^{-1}.
\]

Cela implique une frontière nette. Si le décodeur reçoit déjà le vrai \(G\),
si son rang \(r\) est connu et si l'on conserve tout le sous-espace, une SVD
ou un QR fixe récupère \(S(G)\) sans apprentissage. La tête du décodeur n'est
alors justifiée que dans trois régimes :

- compression volontaire \(s<r\), où elle doit sélectionner les directions
  les plus utiles à profondeur finie ;
- équations faibles bruitées ou produites par un encodeur imparfait ;
- calcul amorti où l'on interdit une factorisation exacte complète à chaque
  prompt.

La récupération apprise du sous-espace physique \(T\), en revanche, reste le
problème statistique principal de l'encodeur. Heavy-Ball ou PCG ne « découvrent
pas \(T\) » par eux-mêmes : ils transmettent une loss de solution à l'encodeur
et exploitent la géométrie que celui-ci a identifiée.

Ainsi la qualité du décodeur se ramène directement à une question de
récupération de sous-espace. Les itérations ne cachent plus ce que la tête a
appris.

## 4. Ce qui est appris et ce qui est fixé

| Composante | Statut | Raison |
|---|---:|---|
| dictionnaire \(\widehat T\) depuis les tâches d'entraînement | apprise | inconnue physique partagée |
| routage du prompt vers \(Q_\theta(\mathcal C)\) | une tête apprise | adaptation in-context |
| construction de \(G,b,H,c\) depuis la forme faible | fixée | identité mathématique connue |
| orthonormalisation de la sortie de tête | fixée | choix de représentation, sans contenu statistique |
| petit système de Ritz \(T_\theta\) | fixé | algèbre linéaire exacte |
| HVP \(v\mapsto \tau G^\top(Gv)+\lambda v\) | attention linéaire fixe | moment exact du prompt |
| relations Heavy-Ball | fixées | dynamique stationnaire connue |
| relations PCG et quotients | fixées | dynamique de Krylov connue |
| MLP pour multiplier ou diviser | absent | aucune approximation nécessaire |

La tête ne doit pas recevoir la solution cible. Elle lit seulement le contexte
et produit un préconditionneur fixe pendant la boucle. Si le préconditionneur
dépend du résidu courant, le second décodeur devient flexible CG et la théorie
PCG standard ne s'applique plus.

## 5. Les deux décodeurs, même géométrie apprise

### Décodeur HB

État persistant : \([Z_\ell,Z_{\ell-1}]\).

\[
r_\ell=c-Hz_\ell,
\qquad
z_{\ell+1}=z_\ell+\alpha B_\theta r_\ell
+\beta(z_\ell-z_{\ell-1}).
\]

Une itération utilise un HVP et une application factorisée de
\(B_\theta\). Les coefficients \(\alpha,\beta\) sont tied. Ils peuvent être
fixés à partir de bornes spectrales certifiées ou appris sous une
paramétrisation stable.

### Décodeur PCG

État persistant : \([Z,R,S,P,\rho]\), avec
\(S=B_\theta R\). Le token \(Q=HP\) est transitoire dans le macro-bloc.
Produits scalaires, quotients sécurisés et routage sont des primitives fixes.
Une itération utilise également un HVP et une application de
\(B_\theta\).

PCG est plus adaptatif et conserve plus d'état. HB est plus simple, ne divise
pas, et se rapproche davantage d'un bloc Transformer récurrent standard. La
comparaison porte sur le contrôleur, pas sur deux représentations apprises
différentes.

## 6. Primal ou dual : décision dictée par le rang

Le même estimateur possède les formes exactement équivalentes

\[
z_\star=(\tau G^\top G+\lambda I)^{-1}\tau G^\top b
=G^\top(\tau GG^\top+\lambda I)^{-1}\tau b.
\]

Si le nombre d'équations utiles \(n=mR\) est inférieur à \(K\), ou si le rang
observable \(r\ll K\), le dual est la représentation naturelle : l'état vit
dans l'espace des tokens et la reconstruction finale est \(z=G^\top\alpha\).
Si \(K\ll n\), le primal est plus petit. Le décodeur ne doit donc pas imposer
le primal ; il choisit le côté de dimension effective la plus faible, ou
travaille directement dans le sous-espace de rang \(r\).

Dans notre régime effectif \(n=mR>K\), le primal est le calcul le moins cher.
Pour conserver l'interprétation duale sans introduire un état de taille \(n\),
on peut utiliser le QR mince

\[
G=UR,
\qquad
(\tau RR^\top+\lambda I)\gamma=\tau U^\top b,
\qquad
z=R^\top\gamma.
\]

Ce **dual comprimé** a une dimension \(\min(n,K)=K\) dans notre cas. Il est
exactement équivalent au primal. Il sert de formulation conceptuelle
tokenique ; le primal évite le coût supplémentaire du QR lorsque seule la
vitesse numérique compte.

## 7. Trois expériences seulement pour valider l'idée

1. **Récupération.** Faire varier \(m,R,K,r\) et mesurer \(E_T\), \(E_S\) et
   les angles principaux. Sans récupération, un bon MSE solveur ne suffit pas.
2. **Causalité.** Remplacer \(Q_\theta\) par le vrai sous-espace, un sous-espace
   aléatoire, puis le sous-espace appris. Vérifier que l'erreur suit l'erreur de
   projecteur prédite par la théorie.
3. **Même préconditionneur, deux contrôleurs.** Entraîner la tête avec HB puis
   PCG, croiser les têtes, et comparer à nombre égal de HVP. Un transfert croisé
   réussi montre que la tête apprend la géométrie du problème, pas un artefact
   du solveur.

Les comparaisons murales CPU/GPU, les grands sweeps et les préconditionneurs
industriels ne viennent qu'après ces trois falsifications. Ils ne peuvent pas
remplacer la validation du mécanisme.

## 8. Frontière des affirmations

L'architecture peut prétendre être une **machine récurrente d'attention
first-principles sans MLP**. Le HVP et les contractions sont des opérations
d'attention linéaire fixes ; les quotients PCG sont des primitives exactes
hardcodées. Ce n'est pas un Transformer vanilla softmax qui aurait appris à
approcher une division.

Elle ne peut pas battre un solveur exact qui reçoit le vrai \(G\) et dispose
d'un budget illimité. La cible crédible est meilleure erreur à profondeur/HVP
fixé grâce à une géométrie amortie depuis la distribution des tâches, tout en
conservant l'adaptation in-context à chaque nouveau prompt.
