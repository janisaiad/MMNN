# Résumé des modifications - example1d.py

## 🔄 Alignement avec benchmark.py

### Paramètres modifiés pour correspondre à benchmark.py:

1. **Optimiseur**: `Adam` → `SGD`
   - Ligne 129: `optim.SGD(model.parameters(), lr=lr_init)`
   - Impact: Convergence plus lente, comportement identique à benchmark.py

2. **Initialisation des poids**: Supprimée
   - Ligne 115: Commenté l'initialisation Kaiming
   - Utilise maintenant l'initialisation par défaut de PyTorch (Xavier/Glorot)
   - Impact: Même point de départ que benchmark.py

3. **Fréquence d'évaluation**: 
   - Test errors: toutes les 50 époques (comme benchmark.py)
   - Prints détaillés: toutes les 500 époques
   - NTK computation: toutes les 5000 époques (optimisation)
   - Plots: toutes les 1000 époques

## ✅ Fonctionnalités conservées (améliorations sur benchmark.py):

### Plots supplémentaires:
- ✅ Loss complète à chaque époque (logscale et loglog)
- ✅ Evolution des erreurs test/train
- ✅ Evolution des eigenvalues NTK (min/max)
- ✅ Prédictions périodiques (toutes les 1000 époques)
- ✅ Comparaison finale (epoch 20000)

### Données sauvegardées:
- ✅ all_losses (loss à chaque epoch)
- ✅ NTK matrices et eigenvalues
- ✅ Configuration complète en JSON
- ✅ Modèle entraîné (.pth)
- ✅ Erreurs train/test (.npz)

### Organisation:
- ✅ Structure de dossiers avec timestamp
- ✅ Noms de fichiers avec config_name
- ✅ Barre de progression avec tqdm
- ✅ Prints informatifs

## 📊 Résultats attendus:

Avec SGD au lieu d'Adam:
- Loss initiale plus élevée
- Convergence plus lente (facteur ~10-100x)
- Courbe de loss plus bruitée
- Résultats finaux similaires après 20000 époques

## 🎯 Comparaison:

| Paramètre | example1d.py (nouveau) | benchmark.py | Status |
|-----------|------------------------|--------------|--------|
| Optimiseur | SGD | SGD | ✅ Identique |
| Init weights | Default | Default | ✅ Identique |
| Learning rate | 0.001 | 0.001 | ✅ Identique |
| Scheduler | StepLR | StepLR | ✅ Identique |
| Activation | ReLU | ReLU | ✅ Identique |
| fixWb | True | True | ✅ Identique |
| Batch size | 100 | 100 | ✅ Identique |
| Epochs | 20000 | 20000 | ✅ Identique |
| Plots | ✅✅✅ | ❌ | 🎉 Amélioré |
| NTK | ✅ | ❌ | 🎉 Amélioré |
| Metrics | ✅✅ | ❌ | 🎉 Amélioré |

Les résultats de loss/erreur devraient maintenant être **identiques** entre les deux scripts!

