# Résumé de l'Approche de Calcul et des Concepts Explorés

Ce document synthétise à haut niveau les différentes stratégies de calcul et les concepts théoriques que nous avons abordés pour analyser l'espérance $E[\text{ReLU}(X_1)\text{ReLU}(X_2)]$ et les lois de probabilité sous-jacentes.

---

## 1. Problème Initial : Calcul de l'Espérance

L'objectif de départ était de trouver une formule pour $E[\text{ReLU}(X_1)\text{ReLU}(X_2)]$, où $(X_1, X_2)$ sont des variables gaussiennes corrélées.

**Approche Principale :**
La stratégie standard a été de standardiser les variables $(Z_1, Z_2)$ et de décomposer le problème en quatre moments tronqués :
$$E[\dots] = \sigma_1\sigma_2 I_{22} + \mu_1\sigma_2 I_{12} + \mu_2\sigma_1 I_{21} + \mu_1\mu_2 I_{11}$$
La difficulté principale a été identifiée dans le calcul du moment croisé $I_{22}$.

---

## 2. Chemins Explorés pour le Calcul de $I_{22}$

Plusieurs méthodes ont été tentées pour calculer ou simplifier $I_{22}$.

### 2.1. Chemin 1 : Utilisation d'une Identité sur les Moments (Impasse)
Une première approche consistait à utiliser l'identité reliant un moment d'ordre 1 à un moment d'ordre 0 :
$$ \int_a^{\infty} z_1 \phi_2 dz_1 = \rho z_2 \int_a^{\infty} \phi_2 dz_1 + (1-\rho^2)\phi_2(a, z_2) $$
Tenter d'appliquer cette identité au calcul de $I_{22}$ a mené à une impasse, car cela transformait le problème en un calcul de moments d'ordre encore plus élevés (contenant des termes en $z_1^2$), rendant le problème plus complexe.

### 2.2. Chemin 2 : Décomposition de la PDF (Impasse)
Une seconde approche consistait à utiliser la décomposition de la PDF bivariée en produit d'une marginale et d'une conditionnelle :
$$ \phi_2(z_1, z_2) = \phi_1(z_1) f(z_2|z_1) $$
Cette méthode, bien que fondamentale, a transformé les termes de $I_{22}$ en intégrales en une dimension qui n'avaient pas de solution analytique simple, menant également à une impasse pour le calcul algébrique.

### 2.3. Chemin 3 : Formule Algébrique Directe et Théorème de Price (Chemin Valide)
La solution s'est avérée être l'utilisation d'une formule connue, issue des propriétés de la loi normale (souvent prouvée via le théorème de Price ou des dérivations géométriques) :
$$ I_{22} = \rho I_{11} + \phi_2(a, b; \rho) $$
Cette identité a permis d'assembler la formule algébrique finale et complète de l'espérance.

---

## 3. Analyse des Lois de Probabilité Sous-jacentes

La discussion s'est ensuite orientée vers l'analyse des distributions des différentes composantes aléatoires du problème, en particulier en haute dimension.

### 3.1. La Loi de `r` (Coefficient de Corrélation)
* **Distribution Exacte :** Identifiée comme la distribution de Fisher (1915), de forme très complexe (impliquant une fonction hypergéométrique `_2F_1`).
* **Approximation :** La transformation **z de Fisher** ($z = \text{arctanh}(r)$) a été présentée comme l'approximation pratique, menant à une loi quasi-normale dont la variance est en $O(1/n)$.
* **"Curse of Dimensionality" :** Nous avons montré que si la "vraie" corrélation $\rho$ est de l'ordre de $1/\sqrt{n}$, le signal se noie dans le bruit géométrique, rendant sa détection statistiquement difficile.

### 3.2. La Loi de Kibble et les Normes au Carré $(U, V)$
* **Distribution Exacte :** La loi jointe de $U=\|X\|^2$ et $V=\|Y\|^2$ a été identifiée comme la distribution **Gamma bivariée de Kibble** (1941).
* **"Curse of Dimensionality" :** Nous avons montré un effet de concentration de la mesure :
    * L'espérance $E[U]$ et la variance $Var(U)$ croissent linéairement avec la dimension $n$.
    * La variable normalisée $U/n$ converge vers la constante 1, avec une variance en $O(1/n)$.
* **Produit $W=UV$ :** La loi du produit a été identifiée comme très complexe (séries infinies, fonctions spéciales), mais ses moments (espérance et variance) ont été calculés explicitement.

### 3.3. Autres Concepts
* **Indépendance :** Nous avons établi le résultat fondamental que $r$ (information sur l'orientation) est statistiquement indépendant du couple $(U,V)$ (information sur la magnitude).
* **Fonction de Gudermann :** Identifiée comme une curiosité mathématique reliant l'angle $\theta = \arccos(r)$ à la transformation $z$ de Fisher, mais sans utilité pratique pour simplifier les calculs.

---

## 4. Analyse d'une Formule de Noyau (NTK)

En fin de discussion, une formule spécifique de noyau NTK a été analysée.
$$ K = 1 + z_1 z_2 f(r) + (1 - \arccos(r)/\pi)f(\rho) $$
* La fonction $f(\rho)$ a été identifiée comme étant $E[\text{ReLU}(Z_1)\text{ReLU}(Z_2)]$.
* Une analyse standard prédisait une déviation standard pour le noyau en $O(1/\sqrt{n})$.
* Face à un résultat expérimental d'une déviation en $O(1/n^2)$, nous avons conclu que la définition du noyau devait contenir un **facteur de normalisation** dépendant de $n$ (par ex. $1/n^{3/2}$), une pratique courante dans la littérature NTK pour assurer une limite stable en haute dimension.