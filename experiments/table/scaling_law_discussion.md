# Discussion: Scaling Laws and Large-Scale Experiments

## Contexte et Hypothèse Initiale

### Hypothèse du Scaling Law

L'utilisateur a proposé l'hypothèse suivante:

> **`loss = g(L/freq)`** où:
> - La courbe est **décroissante** jusqu'à `L/freq = 7`
> - **Optimal** dans la plage `L/freq = 7-12` (entraînement parfait)
> - **Croissante** après `L/freq = 12`

**Objectif:** Construire cette courbe précise et très lisse pour vérifier l'hypothèse.

### Considération Importante sur la Fréquence

L'utilisateur a noté:

> "Il y a aussi le facteur que ce n'est pas directement linéaire avec la fréquence impliquée dans le cosinus, donc nous devrions adapter le plot"

**Implication:**
- Le `freq_multiplier` n'est pas directement la fréquence dans le cosinus
- Les fréquences réelles du cosinus sont: `[12, 24, 36, 72] × freq_multiplier`
- Fréquence maximale: `72 × freq_multiplier`
- Fréquence moyenne: `36 × freq_multiplier`
- On pourrait devoir utiliser `L / max_cosine_freq` ou `L / mean_cosine_freq` au lieu de `L / freq_multiplier`

### Fonction Cible

La fonction cible est une somme de quatre cosinus:

```python
def target_function(x, freq_multiplier):
    base_freqs = [12, 24, 36, 72]
    result = np.zeros_like(x)
    for base_freq in base_freqs:
        freq = base_freq * freq_multiplier
        if base_freq in [24, 72]:
            result += np.cos(freq * np.pi * x + 0.5)
        else:
            result += np.cos(freq * np.pi * x)
    return result
```

## Expansion de la Plage Optimale

L'utilisateur a élargi la plage d'intérêt:

> "Plus entre 7 à 20" (au lieu de 7-12)

**Raison:** Certains entraînements dans la plage 7-12 étaient "très mauvais" ("shitty"), nécessitant des expériences plus larges et une courbe plus lisse.

## Exigences d'Exécution des Expériences

### 1. Exécution Séquentielle

> "ok run tt les experiences mais pas en meme temps stp"

**Implémentation:** Toutes les configurations sont exécutées séquentiellement, une après l'autre.

### 2. Affichage du Progrès

> "avec loss en tqdm"

**Implémentation:** Utilisation de `tqdm` pour afficher une barre de progression avec la perte actuelle en temps réel.

### 3. Reconstruction des Plots

> "reconstruction du plot de loss curve tt les 1k epoch"

**Implémentation:** Le graphique `loss_evolution.png` est régénéré et sauvegardé tous les 1000 epochs.

### 4. Plots de Prédiction

> "at each 1k epoch plot prediction vs baseline"

**Implémentation:** À chaque checkpoint de 1000 epochs (si stable), un graphique `prediction_epoch{epoch}.png` est généré montrant les prédictions du modèle contre la fonction cible pour les ensembles d'entraînement et de test.

## Gestion de l'Instabilité

### Détection d'Instabilité

> "by the way if the loss goes upwards 5 it's surely due to a spike or instability, get back to former state because we want to avoid unstability in training"

**Seuil initial:** 2.0 (changé plus tard à 5.0)

**Seuil final:** 5.0

**Logique de détection:**
- Un spike est détecté si:
  1. Au moins 10 epochs se sont écoulées
  2. La perte actuelle (`epoch_loss`) > 5.0
  3. La perte minimale récente (sur les 9 epochs précédentes) était < 5.0

Cela évite les faux positifs dus à des pertes initiales naturellement élevées.

### Récupération et Arrêt

> "get back to former state"
> "after 1 instability just stop training please"

**Implémentation:**
- Lorsqu'une instabilité est détectée:
  1. Le modèle est restauré depuis le dernier checkpoint stable (`last_stable_checkpoint`)
  2. Un graphique spécial `instability_epoch{epoch}.png` est généré montrant:
     - Prédiction vs. Cible (train/test)
     - Ratio Prédiction/Cible pour visualiser l'instabilité
  3. Le compteur `instability_count` est incrémenté
  4. Si `instability_count >= max_instabilities` (1), l'entraînement s'arrête

### Sauvegarde des Checkpoints

> "do you store parameters at this point ?"

**Implémentation:**
- `checkpoint.pth` est sauvegardé tous les 1000 epochs **seulement si** `epoch_loss <= 5.0` (états stables uniquement)
- Contient: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, et historique des pertes/erreurs
- `model_parameters.pth` est sauvegardé à la fin de l'entraînement avec le `model.state_dict()` final

## Filtrage des Configurations

### Rank

> "don't use rank 10 please, only 15"

**Implémentation:** Seul `rank = 15` est utilisé dans les expériences à grande échelle.

## Plan d'Expérience à Grande Échelle

### Configurations Générées

**Frequencies:** `[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]` (13 valeurs)

**Ranks:** `[15]` uniquement

**L values:** Générés de manière très dense pour couvrir:
- **Plage principale:** `L/freq` de 7 à 20 (tous les 0.25 en ratio)
- **Contexte inférieur:** `L/freq` de 4 à 7 (tous les 0.5 en ratio)
- **Contexte supérieur:** `L/freq` de 20 à 25 (tous les 1.0 en ratio)

**Total:** ~290 configurations (après filtrage par rank)

### Stratégie de Génération

Pour chaque `freq_multiplier`:
1. Calculer la plage de `L` pour couvrir `L/freq` de 7 à 20
2. Générer des valeurs de `L` très denses (tous les 1-2 layers selon la fréquence)
3. Ajouter explicitement des valeurs de `L` correspondant aux ratios cibles (7, 7.25, 7.5, ..., 20)
4. Garantir une couverture dense de la plage optimale 7-20

## Questions et Réponses

### Question: Schéma d'Initialisation

> "what's the initialization scheme"

**Réponse:**

Le modèle utilise la **μ-parameterization (initialisation NTK)**:

- **Architecture:** Couches alternées `rank→width` (couches linéaires d'index pair) et `width→rank` (couches linéaires d'index impair)
- **Couches `rank→width` (j % 2 == 0):** Poids et biais initialisés depuis `N(0, 1/√rank)`
- **Couches `width→rank` (j % 2 == 1):** Poids et biais initialisés depuis `N(0, 1/√width)`
- **`fixWb=True` (utilisé dans les expériences):** Toutes les couches `rank→width` ont `requires_grad=False`, les rendant non-entraînables (couches de features aléatoires). Seules les couches `width→rank` sont entraînables.

### Question: L Values Actuellement en Cours

> "what are the L values currently in running ?"

**Réponse:** Analyse des configurations en cours d'exécution pour identifier les valeurs de `L` testées.

### Question: Configurations Historiquement Réussies

> "give me the configs that led to losses less than 5e-4 before in former large runs"

**Réponse:** Analyse des résultats historiques pour identifier 14 configurations avec `min_loss < 5e-4`:

**Meilleure configuration:**
- `freq0.6, rank15, L8, L/freq=13.33` → `loss=1.18e-10`

**Patterns observés:**
- Toutes les bonnes configs ont `freq <= 0.6`
- Ratios `L/freq` communs: `5.0, 8.33, 10.0, 13.33, 16.67, 60.0`
- Valeurs de `L` communes: `3, 5, 8`
- `Rank 15` apparaît fréquemment

### Question: Causes d'Instabilité

> "what do you think leads to unstable training, give me all L values for all unstable training"

**Réponse:** Analyse complète des instabilités (voir `instability_analysis.md` pour détails):

**L values instables:** `[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]`

**Causes principales:**
1. **L >= 4 est INSTABLE** (L=3 est stable)
2. **L/freq élevé (> 15) est TRÈS INSTABLE**
3. **Spikes extrêmes pour freq=0.30 avec L élevé** (L/freq > 16)

**Recommandations:**
- Éviter L < 4 (sauf L=3 qui peut être stable)
- Maintenir L/freq entre 7-12 pour stabilité optimale
- Éviter L/freq > 15 (très instable)
- Pour freq=0.30, utiliser L très petit (L=3) ou L/freq < 10

## Scripts Créés/Modifiés

### 1. `plan_large_scale_verification.py`

Génère la liste complète des configurations `(freq_multiplier, num_layers, hidden_rank)` pour l'expérience à grande échelle.

**Caractéristiques:**
- Génération dense de `L` values pour couvrir `L/freq` de 7 à 20
- Garantit la couverture de ratios spécifiques (tous les 0.25)
- Ajoute des valeurs de contexte en dehors de la plage principale

### 2. `train_large_scale_verification.py`

Script principal d'exécution de l'entraînement.

**Fonctionnalités:**
- Exécution séquentielle de toutes les configurations
- Barre de progression `tqdm` avec perte en temps réel
- Reconstruction du graphique de perte tous les 1000 epochs
- Plots de prédiction tous les 1000 epochs (si stable)
- Détection intelligente d'instabilité (seuil 5.0, après epoch 10)
- Récupération depuis checkpoint stable en cas d'instabilité
- Arrêt après 1 instabilité
- Sauvegarde de checkpoints stables uniquement
- Génération de graphiques d'instabilité pour diagnostic

### 3. `analyze_optimal_L_over_freq_range.py`

Script d'analyse pour visualiser et ajuster la courbe `loss = g(L/freq)`.

**Fonctionnalités:**
- Ajustement d'une courbe en forme de U
- Utilisation de `log10(loss)` pour la robustesse
- Contraintes sur la position du minimum (plage 7-12 initialement)
- Visualisation de la plage optimale

## Résultats et Observations

### Instabilité Générale

- **Configs stables:** 4
- **Configs instables:** 75
- **Ratio:** 18.75:1

**Observation critique:** Presque toutes les configurations testées (sauf L=3) montrent de l'instabilité, particulièrement quand `L/freq > 15`.

### Configurations Stables

- **L=3 uniquement** (4 configs)
- **L/freq moyen:** 7.01, range [3.75, 10.00]
- **freq moyen:** 0.52

### Configurations Instables

- **L values:** 4-22
- **L/freq moyen:** 14.51, range [4.00, 24.44]
- **freq moyen:** 0.81

### Spikes Extrêmes

| Configuration | freq | L | L/freq | max_loss |
|--------------|------|---|--------|----------|
| freq0.30_rank10_L5 | 0.30 | 5 | 16.67 | 1.48e+04 |
| freq0.30_rank10_L6 | 0.30 | 6 | 20.00 | 3.41e+05 |
| freq0.30_rank10_L7 | 0.30 | 7 | 23.33 | 1.20e+08 |

## Conclusions et Prochaines Étapes

### Hypothèses Validées/Invalidées

1. **L/freq optimal 7-12:** Partiellement validé - L=3 avec L/freq ~7 est stable, mais la plage 7-12 contient aussi beaucoup d'instabilités
2. **L/freq > 15 très instable:** **VALIDÉ** - Tous les spikes extrêmes ont L/freq > 16
3. **L=3 est un sweet spot:** **VALIDÉ** - Seule valeur de L stable observée

### Questions Ouvertes

1. Pourquoi L=3 est-il stable alors que L=4+ ne l'est pas?
2. Pourquoi freq=0.30 avec L élevé cause-t-elle des explosions de gradient?
3. La courbe `loss = g(L/freq)` peut-elle être construite avec les données actuelles compte tenu de l'instabilité généralisée?

### Recommandations Futures

1. **Focus sur L=3** pour explorer la zone stable
2. **Éviter L/freq > 15** dans les futures expériences
3. **Investigation théorique** sur pourquoi L=3 est stable
4. **Ajustement du learning rate** ou autres hyperparamètres pour L > 3
5. **Considérer L / max_cosine_freq** au lieu de L / freq_multiplier pour le scaling law

## Fichiers de Résultats

- **Résultats:** `experiments/table/results_large_scale_verification/`
- **Analyse d'instabilité:** `instability_analysis.md`
- **Scripts:** `plan_large_scale_verification.py`, `train_large_scale_verification.py`, `analyze_optimal_L_over_freq_range.py`

## Date

**Discussion:** 2026-01-23
