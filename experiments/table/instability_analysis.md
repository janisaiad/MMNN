# Analyse des Instabilités d'Entraînement

## Résumé Exécutif

**Statistiques:**
- Configs stables: **4**
- Configs instables: **75**
- Ratio instable/stable: **18.75:1**

## Tous les L Values avec Instabilités

**L values instables:** `[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]`

- **Total:** 19 valeurs différentes
- **Range:** 4 à 22

**L values stables:**
- **L=3 uniquement** (4 configs stables)

## Distribution par L

| L | Nombre de configs instables |
|---|----------------------------|
| 4 | 6 (le plus instable) |
| 5 | 6 |
| 6 | 5 |
| 7 | 5 |
| 8 | 4 |
| 9 | 4 |
| 10 | 4 |
| 11 | 4 |
| 12 | 4 |
| 13 | 4 |
| 14 | 5 |
| 15 | 4 |
| 16 | 4 |
| 17 | 3 |
| 18 | 2 |
| 19 | 2 |
| 20 | 1 |
| 21 | 1 |
| 22 | 1 |

## Distribution par L/freq

| Range L/freq | Nombre de configs instables |
|--------------|----------------------------|
| (0, 5] | 4 |
| (5, 7] | 6 |
| (7, 10] | 12 |
| (10, 12] | 5 |
| (12, 15] | 13 |
| (15, 20] | 17 |
| (20, 100] | 12 |

## Distribution par freq

| freq | Nombre de configs instables |
|------|----------------------------|
| 0.30 | 6 |
| 0.60 | 1 |
| 0.70 | 14 |
| 0.80 | 16 |
| 0.90 | 19 |
| 1.00 | 13 |

## Comparaison Stable vs Instable

### L/freq
- **Stable:** L/freq moyen = **7.01**, range [3.75, 10.00]
- **Instable:** L/freq moyen = **14.51**, range [4.00, 24.44]

### L (nombre de couches)
- **Stable:** L moyen = **3.00**, médian = **3.00**
- **Instable:** L moyen = **11.60**, médian = **11.00**

### freq
- **Stable:** freq moyen = **0.52**
- **Instable:** freq moyen = **0.81**

## Principales Causes d'Instabilité

### 1. L >= 4 est INSTABLE

- **L=3:** STABLE (L/freq moyen = 7.01)
- **L=4-22:** INSTABLE (L/freq moyen = 14.51)
- L très petit (4-5) peut être instable
- L élevé (> 10) est aussi instable

### 2. L/freq élevé (> 15) est TRÈS INSTABLE

- Stable: L/freq moyen = 7.01, range [3.75, 10.00]
- Instable: L/freq moyen = 14.51, range [4.00, 24.44]
- Les spikes extrêmes ont L/freq > 16

### 3. Spikes Extrêmes pour freq=0.30 avec L élevé

| Configuration | freq | L | L/freq | max_loss |
|--------------|------|---|--------|----------|
| freq0.30_rank10_L5 | 0.30 | 5 | 16.67 | 1.48e+04 |
| freq0.30_rank10_L6 | 0.30 | 6 | 20.00 | 3.41e+05 |
| freq0.30_rank10_L7 | 0.30 | 7 | 23.33 | 1.20e+08 |

## Hypothèses sur les Causes

1. **L trop petit (< 5)** → capacité insuffisante → instabilité
   - Le réseau n'a pas assez de capacité pour apprendre la fonction cible complexe
   - Cela peut mener à des oscillations ou des explosions de gradient

2. **L/freq trop élevé (> 15)** → sur-paramétrisation → instabilité
   - Trop de couches par rapport à la fréquence
   - Peut causer des problèmes de conditionnement numérique
   - Risque d'explosion de gradient

3. **Combinaisons spécifiques (freq=0.30, L élevé)** → explosion de gradient
   - Les très basses fréquences avec beaucoup de couches sont particulièrement problématiques
   - Les gradients peuvent exploser exponentiellement avec la profondeur

4. **L=3 semble être un sweet spot** pour certaines fréquences
   - Assez de capacité sans sur-paramétrisation
   - Bon équilibre entre expressivité et stabilité

## Recommandations

### Pour éviter l'instabilité:

1. **Éviter L < 4** (sauf L=3 qui peut être stable dans certains cas)
2. **Maintenir L/freq entre 7-12** pour stabilité optimale
3. **Éviter L/freq > 15** (très instable)
4. **Pour freq=0.30**, utiliser L très petit (L=3) ou L/freq < 10

### Zone de stabilité recommandée:

- **L:** 3-8 (éviter L < 3 ou L > 10)
- **L/freq:** 7-12 (zone optimale)
- **freq:** Éviter les très basses fréquences (< 0.3) avec L élevé

## Méthodologie de Détection

Une configuration est considérée comme **instable** si au moins une des conditions suivantes est vraie:

1. Présence d'un fichier `instability_epoch*.png`
2. `instability_count > 0` dans `results.json`
3. `stopped_due_to_instability = true` dans `results.json`
4. Spike détecté: `loss > 5.0` après avoir été < 5.0 (après epoch 10)
5. `max_loss > 5.0` dans l'historique d'entraînement

## Notes Techniques

- **Seuil d'instabilité:** loss > 5.0
- **Détection de spike:** Requiert au moins 10 epochs et une perte récente < 5.0 avant le spike
- **Toutes les configurations analysées:** `experiments/table/results_large_scale_verification/`
- **Date d'analyse:** 2026-01-23
