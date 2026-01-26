# Méthodologie Complète de Plotting - MMNN Training

## 📊 Liste Complète des Plots Générés

### 1. **Plots Statiques (PNG)**

#### 1.1 `prediction_vs_baseline_epoch0_lr{lr}.png`
- **Quand**: À l'initialisation (epoch 0)
- **Contenu**:
  - Ligne bleue: Fonction cible (baseline/ground truth)
  - Ligne rouge pointillée: Prédiction du modèle à l'initialisation
  - Points verts: Points d'entraînement (x_train, y_train)
  - Points orange: Prédictions aux points d'entraînement
- **Objectif**: Visualiser l'état initial du réseau avant entraînement
- **Fonction**: `plot_prediction_vs_baseline(..., save_png=True)`

#### 1.2 `loss_evolution.png`
- **Quand**: À la fin de l'entraînement
- **Contenu**:
  - Courbe de loss d'entraînement vs epochs
  - Barres verticales rouges: Moments de réduction du learning rate
  - Ligne horizontale: Seuil d'early stopping (2e-5)
- **Objectif**: Visualiser l'évolution de la loss pendant l'entraînement
- **Fonction**: Généré dans `train_one_config()` après la boucle d'entraînement

#### 1.3 `loss_and_lr_evolution.png`
- **Quand**: À la fin de l'entraînement
- **Contenu**: 2 subplots
  - **Subplot 1 (haut)**: Loss d'entraînement vs epochs (même que `loss_evolution.png`)
  - **Subplot 2 (bas)**: Learning rate vs epochs
- **Objectif**: Visualiser simultanément l'évolution de la loss et du learning rate
- **Fonction**: Généré dans `train_one_config()` après la boucle d'entraînement

#### 1.4 `layer_{idx}_partial_functions.png`
- **Quand**: À la fin de l'entraînement (pour chaque couche)
- **Contenu**: Grille de subplots montrant les fonctions partielles apprises par chaque composante de la couche
  - Pour chaque couche `idx` (1, 2, 3, 4, 5 pour L=2):
    - Grille de `n_rows × n_cols` subplots
    - Chaque subplot montre une composante de la sortie de la couche
    - Axe X: `x ∈ [-1, 1]` (500 points fins)
    - Axe Y: Valeur de la composante
- **Objectif**: Visualiser les représentations internes apprises par chaque couche
- **Fonction**: `plot_partial_functions()` (appelée à la fin de l'entraînement)

### 2. **GIFs Animés**

#### 2.1 `prediction_vs_baseline_epochs_0_250.gif`
- **Quand**: Créé immédiatement à l'epoch 250
- **Contenu**: Animation de 251 frames (epochs 0 à 250, 1 frame/epoch)
  - Chaque frame = même contenu que `prediction_vs_baseline_epoch0.png` mais pour un epoch donné
  - Montre l'évolution de la prédiction pendant les 250 premiers epochs
- **Durée**: 0.1 secondes par frame
- **Objectif**: Visualiser la dynamique d'apprentissage au début de l'entraînement
- **Fonction**: Frames collectées dans la boucle d'entraînement, GIF créé à `epoch == 250`

#### 2.2 `layer_{idx}_partial_functions_epochs_0_250.gif`
- **Quand**: Créé immédiatement à l'epoch 250 (pour chaque couche)
- **Contenu**: Animation de 251 frames (epochs 0 à 250, 1 frame/epoch)
  - Chaque frame = même contenu que `layer_{idx}_partial_functions.png` mais pour un epoch donné
  - Montre l'évolution des fonctions partielles pendant les 250 premiers epochs
- **Durée**: 0.1 secondes par frame
- **Objectif**: Visualiser l'évolution des représentations internes au début de l'entraînement
- **Fonction**: Frames collectées dans la boucle d'entraînement, GIF créé à `epoch == 250`

---

## 🔬 Méthodologie Détaillée de Plotting

### Architecture MMNN (L=2, NTK parameterization)

Pour un réseau avec:
- **L = 2** couches
- **Ranks**: `[1, hidden_rank, hidden_rank, 1]` (ex: `[1, 15, 15, 1]`)
- **Widths**: `[hidden_width, hidden_width, hidden_width]` (ex: `[1024, 1024, 1024]`)
- **Total Linear layers**: 6 (2 par bloc de couche)

**Structure des couches**:
```
Input (rank=1)
  ↓
L0: rank→width  [1 → 1024]     (frozen si fixWb=True)
  ↓ ReLU
L1: width→rank [1024 → 15]    (trainable)
  ↓
L2: rank→width  [15 → 1024]    (frozen si fixWb=True)
  ↓ ReLU
L3: width→rank  [1024 → 15]    (trainable)
  ↓
L4: rank→width  [15 → 1024]    (frozen si fixWb=True)
  ↓ ReLU
L5: width→rank  [1024 → 1]     (trainable)
  ↓
Output (rank=1)
```

### Plotting des Fonctions Partielles

#### Méthode: Extraction des Sorties Intermédiaires

**Code dans `plot_partial_functions()`**:

```python
# Pour chaque couche idx (1, 2, 3, 4, 5):
for layer_idx in range(1, len(teacher.fcs), 1):
    # 1. Déterminer le rank de sortie
    if layer_idx % 2 == 0:
        output_rank = ranks[layer_idx//2+1]  # rank→width layer
    else:
        output_rank = min(widths[(layer_idx)//2], 36)  # width→rank layer
    
    # 2. Forward pass jusqu'à cette couche
    current = x_tensor  # [batch_size, 1]
    for i in range(layer_idx):
        current = teacher.fcs[i](current)
        if i % 2 == 0:  # Appliquer ReLU après rank→width
            current = torch.relu(current)
    
    # 3. Extraire les composantes
    output = current.cpu().numpy()  # [batch_size, output_rank]
    
    # 4. Plotter chaque composante dans un subplot
    for idx in range(output_rank):
        axes[i, j].plot(x_plot, output[:, idx], 'b-', linewidth=1)
```

**Interprétation**:
- **L1 (width→rank)**: Montre les 15 canaux low-rank après la première couche
  - Chaque canal = combinaison linéaire des 1024 activations ReLU
  - Ces 15 fonctions sont ensuite mélangées via la matrice de la couche suivante
  
- **L2 (rank→width)**: Montre les 1024 neurones après expansion (limitée à 36 pour visualisation)
  - Chaque neurone = combinaison linéaire des 15 canaux précédents
  
- **L3 (width→rank)**: Montre les 15 canaux low-rank après la deuxième couche
  
- **L4 (rank→width)**: Montre les 1024 neurones après expansion
  
- **L5 (width→rank)**: Montre la sortie finale (1 dimension)

### Plotting pour Mean Field

#### Concept Mean Field vs MMNN

Dans le contexte **mean field**, les fonctions partielles `f_k(x)` représentent:

$$f_k(x) = \mathbb{E}_{C_1}[w_1(C_1, k) \cdot \phi_1(f_1(C_1), x)]$$

Où:
- `C_1` = indices des neurones de la première couche (width)
- `w_1(C_1, k)` = poids pour le canal `k` du neurone `C_1`
- `f_1(C_1)` = features aléatoires gelées (random features)
- `\phi_1` = activation ReLU
- `x` = point d'entrée

**Implémentation Mean Field** (dans `meanfield_cosine_multifreq_experiment.py`):

```python
def compute_partial_functions(self, w1, X):
    # X: [batch_size, d] = [500, 1] pour le plot
    # f1: [n1, d] = [777, 1] (features aléatoires gelées)
    
    inner = torch.matmul(self.f1, X.t())  # [777, 500]
    phi1 = torch.relu(inner)  # [777, 500] - ReLU activations
    
    f_k = torch.zeros(self.r, batch_size)  # [15, 500]
    for k in range(self.r):  # pour chaque canal k = 0,...,14
        w1_k = w1[:, k]  # [777, 1] - poids pour le canal k
        f_k[k] = torch.mean(w1_k * phi1, dim=0)  # [500] - moyenne sur les 777 neurones
    
    return f_k  # [15, 500] - une fonction f_k(x) pour chaque canal
```

**Différence avec MMNN**:
- **Mean Field**: `f_k(x)` = moyenne sur tous les neurones de la couche
- **MMNN**: Sortie directe de la couche low-rank (pas de moyenne explicite, mais équivalent via la structure low-rank)

#### Plotting Mean Field - Fonctions Partielles

**Méthodologie**:

1. **Calculer `f_k(x)` pour tous les canaux `k`**:
   - Pour chaque point `x` dans `[-1, 1]` (500 points fins)
   - Pour chaque canal `k = 0, ..., r-1` (ex: r=15)
   - Résultat: `f_k` de shape `[r, 500]`

2. **Plotter chaque `f_k` comme une fonction de `x`**:
   ```python
   fig = plt.figure(figsize=(16, 10))
   ax = fig.add_subplot(111)
   
   colors = plt.cm.tab20(np.linspace(0, 1, r))
   for k in range(r):
       ax.plot(x_fine, f_k[k, :], color=colors[k], linewidth=2, 
               label=f'$f_{k}(x)$')
   ```

3. **Interprétation**:
   - Chaque courbe représente la contribution d'un canal low-rank
   - Les canaux peuvent se spécialiser pour différentes fréquences
   - Les fonctions sont **piecewise linear** (combinaisons de ReLU) et passent par 0

#### Plotting Mean Field - Log Ratios

**Concept**: Mesurer la spécialisation des canaux via les log-ratios:

$$R_{i,j} = \log(|f_i(x)|) - \log(|f_j(x)|)$$

**Implémentation**:

```python
def compute_log_ratios(self, f_k, x_location):
    # f_k: [r, 1] - valeurs de f_k au point x_location
    # Calculer R_{i,j} pour tous les paires (i, j)
    
    log_f_k = np.log(np.abs(f_k) + 1e-10)  # [r, 1]
    R = np.zeros((r, r))
    
    for i in range(r):
        for j in range(r):
            R[i, j] = log_f_k[i] - log_f_k[j]
    
    return R  # [r, r] - matrice de log-ratios
```

**Plots générés**:

1. **Heatmap de log-ratios** (`meanfield_log_ratio_heatmap.png`):
   - Matrice `R_{i,j}` à un temps final
   - Rouge = canal `i` domine canal `j`
   - Bleu = canal `j` domine canal `i`
   - Blanc = magnitudes similaires

2. **Évolution temporelle** (`meanfield_log_ratio_statistics_time.png`):
   - Mean, max, min, ±1 std des log-ratios vs temps
   - Montre l'évolution de la spécialisation

3. **Comparaison Mean-Field vs Finite-Width**:
   - Plot des prédictions finales
   - Plot des fonctions partielles `f_k(x)`
   - Vérification de la convergence mean-field

---

## 📝 Résumé des Fichiers Générés par Configuration

Pour chaque configuration d'entraînement, les fichiers suivants sont générés:

### Fichiers JSON
- `config.json`: Configuration complète (factor, ranks, widths, optimizer, etc.)
- `results.json`: Résultats d'entraînement (losses, epochs, temps, etc.)

### Fichiers PNG (Statiques)
- `prediction_vs_baseline_epoch0_lr{lr}.png`: État initial
- `loss_evolution.png`: Évolution de la loss
- `loss_and_lr_evolution.png`: Loss + LR
- `layer_{1,2,3,4,5}_partial_functions.png`: Fonctions partielles (fin d'entraînement)

### Fichiers GIF (Animés)
- `prediction_vs_baseline_epochs_0_250.gif`: Évolution prédiction (251 frames)
- `layer_{1,2,3,4,5}_partial_functions_epochs_0_250.gif`: Évolution fonctions partielles (251 frames)

**Note**: Les GIFs sont créés immédiatement à l'epoch 250, pas à la fin de l'entraînement complet.

---

## 🔧 Paramètres de Plotting

- **Résolution PNG**: 150-300 DPI
- **Résolution GIF**: 100 DPI (pour réduire la taille)
- **Points de plot**: 500-1000 points fins pour les courbes
- **Frames GIF**: 251 frames (epochs 0-250, 1 frame/epoch)
- **Durée par frame**: 0.1 secondes
- **Limite de visualisation**: Max 36 composantes par couche (pour éviter trop de subplots)

---

## 🎯 Objectifs des Plots

1. **Diagnostic d'entraînement**: Loss evolution, LR scheduling
2. **Qualité d'approximation**: Prediction vs baseline
3. **Compréhension interne**: Fonctions partielles (représentations apprises)
4. **Dynamique d'apprentissage**: GIFs montrant l'évolution temporelle
5. **Comparaison Mean-Field**: Log-ratios, spécialisation des canaux
