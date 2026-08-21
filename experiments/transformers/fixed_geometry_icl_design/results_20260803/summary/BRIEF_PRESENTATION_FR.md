# Brief de présentation — ICL à géométrie noyau fixée

## Message en une phrase

On peut fixer la non-linéarité softmax/RBF comme choix de modélisation, puis entraîner seulement la dynamique récurrente et la politique de mesure : la première réalise une régression bayésienne in-context proche de KRR et la seconde apprend où mesurer.

## Protocole minimal

- Une nouvelle fonction GP est tirée à chaque épisode sur 64 positions fixes.
- La longueur d’échelle RBF vaut 0,18 et n’est jamais entraînable.
- La cellule de résolution est partagée sur 12 itérations et entraînée seulement par l’erreur de prédiction.
- KRR exact et variance-greedy ne sont appelés qu’après entraînement, comme références d’évaluation.
- Cinq graines indépendantes ; 5 000 mises à jour par modèle et par politique.

## Résultats à annoncer

- Boucle noyau apparié, 12 observations : MSE 0.1205 ± 0.0025.
- Référence KRR exacte : MSE 0.1136 ± 0.0025.
- Même boucle avec mauvaise géométrie : MSE 0.209 ± 0.0015.
- Transformer standard : MSE 0.137 ± 0.0028.
- Design appris, budget 8 : MSE pondérée 0.06049 ± 0.0047.
- Design aléatoire, budget 8 : 0.2342 ± 0.0046.
- Design uniforme/maximin, budget 8 : 0.1493 ± 0.0025.

## Déroulé conseillé en six transparents

1. Hypothèse : la non-linéarité définit l’espace de features ; elle n’a pas besoin d’être apprise.
2. Architecture : softmax RBF fixé, état dual, cellule récurrente partagée.
3. Protocole : nouvelles fonctions à chaque épisode, contrôles appariés, aucune cible KRR pendant l’entraînement.
4. Résultat ICL : scaling en nombre d’observations et en nombre de boucles.
5. Design expérimental : politique aval versus aléatoire, uniforme et variance-greedy.
6. Limites : domaine 1D et géométrie fixe ; prochaine étape = LSM/PDE à géométrie physique fixée.

## Formulation prudente

Ce PoC valide le mécanisme et les contrôles causaux essentiels. Il ne constitue ni une loi d’échelle universelle, ni encore un benchmark PDE, ni une preuve que le noyau peut être identifié automatiquement.
