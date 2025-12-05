# Explication Complète de Tous les Codes

## Vue d'Ensemble

Nous avons créé un système complet pour:
1. **Calculer** les matrices NTK et leurs spectres (eigenvalues)
2. **Visualiser** les résultats avec des plots publication-quality
3. **Valider** la théorie (Théorèmes 3.2, 4.2, etc.)

---

# 1. Scripts de Calcul (Computation)

## 1.1. `experiments/paper/largescale.py` (Original)

### Objectif
Calculer les matrices de Gram NTK pour une **grille de paramètres**.

### Architecture du Réseau

**3 couches RF-LR**:
```
Input x ∈ R^d
   ↓ W^(1) ∈ R^(n1×d) (frozen, Gaussian)
Layer 1: h^(1) = (1/√n1) A^(1) ReLU(W^(1)T x + b^(1))
         where A^(1) ∈ R^(r×n1) (trainable, low-rank bottleneck)
   ↓ W^(2) ∈ R^(n2×r) (frozen)
Layer 2: h^(2) = (1/√n2) A^(2) ReLU(W^(2)T h^(1) + b^(2))
         where A^(2) ∈ R^(1×n2) (trainable, output)
   ↓
Output: f(x) = h^(2) + c (scalar output)
```

**Paramètres clés**:
- $n_1 = n_2 = N$ (width des couches)
- $r$ (rank du bottleneck)
- $d$ (dimension d'entrée)

### Formule du NTK 3-Couches

$$\Theta^{(2)}(x_1, x_2) = \Theta^{(1)}(\hat{\rho}_r) \left(1 - \frac{\arccos(\hat{\rho}_r)}{\pi}\right) + w_r \cdot \Sigma^{(1)}(\hat{\rho}_r) + 1$$

où:
- $h^{(1)}(x) = \frac{1}{\sqrt{n_1}} A^{(1)} \text{ReLU}(W^{(1)T} x + b^{(1)})$ (features de couche 1)
- $\hat{\rho}_r = \frac{\langle h^{(1)}(x_1), h^{(1)}(x_2) \rangle}{\|h^{(1)}(x_1)\| \|h^{(1)}(x_2)\|}$ (corrélation angulaire)
- $w_r = \frac{\|h^{(1)}(x_1)\| \|h^{(1)}(x_2)\|}{r}$ (produit radial)

### Fonctions Principales

#### `compute_ntk_3layer(x1, x2, ...)`
Calcule le NTK entre deux inputs $x_1, x_2$:
1. Forward pass: $h^{(1)}(x_1)$, $h^{(1)}(x_2)$
2. Calcule $\hat{\rho}_r$ (corrélation des features)
3. Calcule $w_r$ (produit des normes / r)
4. Évalue kernels ReLU: $\Theta^{(1)}(\hat{\rho}_r)$, $\Sigma^{(1)}(\hat{\rho}_r)$
5. Combine selon formule ci-dessus

#### `compute_ntk_gram_matrix(X, ...)`
Calcule la matrice de Gram **complète** $\mathbf{K} \in \mathbb{R}^{n \times n}$:
- $\mathbf{K}_{ij} = \Theta^{(2)}(x_i, x_j)$
- Vectorisé pour efficacité: calcule toutes les paires $(i,j)$ en une fois

**Étapes**:
1. Pre-activations: $Z^{(1)} = W^{(1)} X^T + b^{(1)}$ (matrice $n_1 \times n$)
2. ReLU: $H_{\text{relu}} = \max(0, Z^{(1)})$
3. Features layer 1: $H^{(1)} = A^{(1)} H_{\text{relu}} / \sqrt{n_1}$ (matrice $r \times n$)
4. Normes: $\text{norms}_i = \|H^{(1)}_i\|$ pour chaque sample
5. Inner products: $G = H^{(1)T} H^{(1)}$ (matrice $n \times n$)
6. Corrélations: $R_{ij} = G_{ij} / (\text{norms}_i \cdot \text{norms}_j)$
7. Produits radiaux: $W_{ij} = (\text{norms}_i \cdot \text{norms}_j) / r$
8. Kernels ReLU: $\Theta^{(1)}(R)$, $\Sigma^{(1)}(R)$ (element-wise)
9. NTK final: formule complète

**Coût**: $O(n^2 r + n_1 dr)$ FLOPs

#### `run_grid_computation(...)`
Boucle principale:
```python
for n_pow in data_powers:      # n ∈ {2^4, ..., 2^10}
    for N_pow in width_powers:  # N ∈ {2^4, ..., 2^10}
        for r_pow in rank_powers:  # r ∈ {2^4, ..., 2^10}
            n = 2^n_pow
            N = 2^N_pow
            r = 2^r_pow
            d = r  # policy: d = r
            
            # pour chaque initialization:
            for init in range(n_init):
                # 1. Générer données X
                # 2. Initialiser réseau
                # 3. Calculer matrice NTK Gram
                # 4. Eigendecompose
                # 5. Stocker eigenvalues
            
            # Agréger sur initialisations
            # Sauvegarder spectre + metadata
```

**Outputs**:
- `grid_n{n}_N{N}_r{r}_d{d}.npz`: eigenvalues moyennés
- `grid_n{n}_N{N}_r{r}_d{d}_metadata.json`: metadata avec params

---

## 1.2. `experiments/paper/largescale_largewidth.py` (Nouveau)

### Objectif
Comme `largescale.py` mais pour **LARGE WIDTH** $N \gg r, d$.

### Différences Clés

#### Grille Extensive
Au lieu de powers régulières, on spécifie explicitement:
- $N \in \{2048, ..., 131072\}$ (large widths!)
- $n \in \{16, ..., 2048\}$ (many values)
- $r \in \{16, ..., 1024\}$ (many values)
- $d \in \{16, ..., 1024\}$ (many values)

#### Contrainte de Régime
```python
if N < 8 * max(r, d):
    skip  # pas dans régime large-width
```

Cela garantit $N/\max(r,d) \geq 8$ (minimum pour limite infinie)

#### Fonction `run_extensive_grid(...)`
```python
for N_pow in width_powers:
    N = 2^N_pow
    for n in n_values:
        for r in r_values:
            for d in d_values:
                if N >= 8 * max(r, d):  # check regime
                    # compute config
                    # save to largewidth/ folder
```

**Output**: `refs/paper/data/largewidth/lw_N{N}_n{n}_r{r}_d{d}.npz`

---

# 2. Scripts de Visualisation

## 2.1. `experiments/paper/plot_all_figures.py` (Principal)

### Structure Générale

Le script contient une fonction par plot:
- `plot_rank_concentration()` → Plot 1
- `plot_tail_probability()` → Plot 1a
- `plot_ntk_concentration()` → Plot 2
- `plot_spectral_decay()` → Plot 3
- `plot_marchenko_pastur()` → Plot 4
- `plot_flops_analysis()` → Plot 5
- `plot_fisher_kibble()` → Plot 6

### Configuration matplotlib

```python
plt.rcParams['figure.figsize'] = [6, 6]
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 22
mpl.rcParams['savefig.dpi'] = 300
# ... (style publication-quality)
```

Style cohérent: STIXGeneral font, 300 DPI, couleurs Dark2

---

### Plot 1: Rank-Driven Concentration (2 panels)

#### Panel 1: Variance Decay

**Code**:
```python
# Monte Carlo sampling
for r in r_vals_empirical:
    x = rng.standard_normal((n_mc_samples, r))  # sample x ∈ R^r
    y = rng.standard_normal((n_mc_samples, r))  # sample y ∈ R^r
    W = (||x|| * ||y||) / r  # radial product
    variance_empirical[r] = Var(W)  # empirical variance

# Theoretical curve
var_theory = C / r  # Var(W_r) ~ 1/r
```

**Étapes**:
1. Pour chaque rang $r \in \{5, 10, ..., 200\}$:
   - Échantillonne 50,000 paires $(x,y) \sim \mathcal{N}(0, I_r)$
   - Calcule $W = \|x\| \|y\| / r$ pour chaque paire
   - Mesure $\text{Var}(W)$ empiriquement
2. Plot log-log: points empiriques + courbe théorique $1/r$

#### Panel 2: Std Decay

Même chose pour $\sigma = \sqrt{\text{Var}} \sim 1/\sqrt{r}$

---

### Plot 1a: Tail Probability (Standalone)

**Code**:
```python
for eps in epsilons:  # eps ∈ {0.1, 0.2, 0.5}
    for r in r_vals:
        # sample W_r
        prob = mean(|W - 1| >= eps)  # empirical probability
    
    # theoretical bound
    prob_theory = 4 * exp(-r * eps^2 / 8)
    
    # align at first rank
    shift = prob_empirical[0] / prob_theory[0]  # shift from r=5
    prob_theory_shifted = shift * prob_theory
```

**Étapes**:
1. Pour chaque $\epsilon$ et $r$: mesure $\mathbb{P}(|W_r-1| \geq \epsilon)$
2. Calcule shift pour aligner au **premier rang** (r=5)
3. Plot log-log: empirique (points) + théorique shifté (lignes)

**Résultat shifts**:
- $\epsilon=0.1$: shift ≈ 0.21
- $\epsilon=0.2$: shift ≈ 0.17
- $\epsilon=0.5$: shift ≈ 0.075

**Interprétation**: La borne théorique $4e^{-r\epsilon^2/8}$ est conservatrice (surestime), mais la **forme exponentielle** est correcte.

---

### Plot 2: NTK Concentration (4 panels)

#### Panels 1-3: NTK vs $\rho$

**Code**:
```python
for config in rank_configs:  # r ∈ {16, 32, 64}
    # load NTK-rho data
    data = np.load(f"grid_n{n}_N{N}_r{r}_d{d}_ntk_rho.npz")
    rho_vals = data["rho_vals"]  # [-1, -0.9, ..., 1.0]
    ntk_mean = data["ntk_mean"]  # empirical mean
    ntk_std = data["ntk_std"]    # empirical std
    k_infty = data["k_infty"]    # deterministic limit
    
    # plot mean + confidence band
    plot(rho_vals, ntk_mean)
    fill_between(rho_vals, ntk_mean - 2*ntk_std, ntk_mean + 2*ntk_std)
    plot(rho_vals, k_infty, '--')  # deterministic limit
```

**Ce qui est montré**:
- Empirical NTK: $\hat{\Theta}^{(2)}(\rho)$ moyenné sur samples Fisher-Kibble
- Bandes $\pm 2\sigma$: montrent variance qui décroit avec $r$
- Limite déterministe: $K_\infty(\rho)$ (formule théorique)

#### Panel 4: Std vs Rank

**Code**:
```python
# search all NTK-rho files (343 files!)
for ntk_file in all_ntk_rho_files:
    r = extract_r_from_filename(ntk_file)
    std = mean(data["ntk_std"])  # mean std across all rho
    
# select 5 representative ranks
selected_r = logspace_select(all_r, num=5)

# fit: sigma ~ C / sqrt(r)
log_fit → C_fitted

# plot empirical + fitted theory
loglog(selected_r, std_empirical, 'o')
loglog(r_theory, C_fitted / sqrt(r_theory), '--')
```

**Étapes**:
1. Charge **tous** les 343 fichiers NTK-rho
2. Extrait std moyen pour chaque rang $r$
3. Sélectionne 5 rangs représentatifs (espacement log)
4. Fit $\sigma = C/\sqrt{r}$ pour trouver $C$
5. Plot avec théorie alignée

**Résultat**: $C \approx 0.120$, pente empirique ≈ -0.70 (attendu: -0.50)

---

### Plot 3: Spectral Decay

**Code**:
```python
for config in configs:  # 4 configs with different r
    # load eigenvalues from NTK Gram matrix
    eigenvalues = np.load(f"grid_n{n}_N{N}_r{r}_d{d}.npz")["eigenvalues_mean"]
    
    # remove spike (focus on bulk)
    bulk = eigenvalues[eigenvalues < 10.0]
    
    # sort descending
    bulk_sorted = sort(bulk)[::-1]
    
    # create index array
    k = [1, 2, 3, ..., len(bulk)]
    
    # plot λ_k vs k
    loglog(k, bulk_sorted, ':', label=f'r={r}, d={d}')

# add reference k^{-0.5} aligned at LAST index
last_k, last_λ = k[-1], bulk_sorted[-1]
C = last_λ * last_k^{0.5}  # shift to match last point
ref = C * k^{-0.5}
loglog(k, ref, '--', label='k^{-0.5}')
```

**Étapes**:
1. Charge eigenvalues **réelles** de matrices NTK Gram calculées
2. Enlève spike (typiquement $\lambda_1 \sim 1500$ vs bulk $\sim 0.01$)
3. Trie bulk en ordre décroissant
4. Plot $\lambda_k$ vs index $k$ en log-log
5. Ajoute référence $k^{-0.5}$ **alignée au dernier point**

**Interprétation**: 
- Lignes pointillées = données empiriques (vraies eigenvalues)
- Ligne tirets noire = référence théorique alignée
- Si parallèle → confirme décroissance en $k^{-0.5}$

---

### Plot 4: Marchenko-Pastur (3 panels)

**Code**:
```python
for gamma in gamma_ratios:  # [0.5, 1.0, 2.0]
    # find config with this gamma
    eigenvalues = load_eigenvalues(n, N, r, d)
    
    # separate spike from bulk
    spike = eigenvalues[eigenvalues > 10.0]  # typically 1 value
    bulk = eigenvalues[eigenvalues <= 10.0]  # n-1 values
    
    # filter bulk to MP support
    a = (1 - sqrt(gamma))^2  # lower edge (theory)
    b = (1 + sqrt(gamma))^2  # upper edge (theory)
    bulk_clean = bulk[(bulk >= 0.5*a) & (bulk <= 1.5*b)]
    
    # histogram with Freedman-Diaconis binning
    IQR = percentile(bulk_clean, 75) - percentile(bulk_clean, 25)
    bin_width = 2 * IQR * n^{-1/3}  # Diaconis rule
    n_bins = (max - min) / bin_width
    
    hist(bulk_clean, bins=n_bins, density=True)
    
    # theoretical MP density
    rho_MP(λ) = (1/(2πγ)) * sqrt((b-λ)(λ-a)) / λ
    plot(lambda, rho_MP(lambda), '--')
    
    # annotate spike in text box
    text(f"Spike: λ_1 = {spike[0]:.1f}")
```

**Étapes détaillées**:
1. **Séparation spike-bulk**: Seuil adaptatif $2b$
2. **Nettoyage bulk**: Garde seulement eigenvalues près du support théorique
3. **Binning optimal**: Règle de Freedman-Diaconis (adapte aux données)
4. **Histogram**: Densité empirique
5. **MP théorique**: Overlay formule Marchenko-Pastur
6. **Outliers**: Identifiés et annotés individuellement

**Résultat**: 3 panels montrant comment bulk change avec $\gamma$

---

### Plot 5: FLOPs Analysis

**Code**:
```python
# load ALL metadata files
for meta_file in all_metadata:
    meta = json.load(meta_file)
    n = meta["n"]
    gamma = meta["gamma_ratio"]
    flops = meta["flops_config"]  # actual FLOPs used!
    
# group by n
by_n[n].append(flops)

# panel 1: FLOPs vs n
plot(n_values, mean(flops), 'o-')
plot(n_theory, n^2, '--')  # theoretical ~ n^2

# panel 2: FLOPs vs gamma
by_gamma[gamma].append(flops)
plot(gamma_values, mean(flops), 'o-')
```

**Nouveauté**: Utilise les **FLOPs réels** enregistrés pendant les calculs (pas théorique!)

**Validation**: Vérifie que coût $\sim n^2$ (dominant: Gram matrix)

---

### Plot 6: Fisher-Kibble Independence

**Code**:
```python
# generate samples across full rho range
for rho_true in [-1.0, -0.9, ..., 1.0]:  # 21 values
    # sample Fisher distribution (angular)
    rho_hat = sample_fisher(rho_true, r, n_samples=500)
    
    # sample Kibble distribution (radial)
    # approximate: norm products
    x_norm = ||rng.normal(n_samples, r)||
    y_norm = ||rng.normal(n_samples, r)||
    w_r = (x_norm * y_norm) / r
    
    # store pairs
    all_rho.append(rho_hat)
    all_w.append(w_r)
    colors.append(rho_true)  # for coloring

# scatter plot
scatter(all_rho, all_w, c=colors, cmap='coolwarm')
```

**Étapes**:
1. Pour chaque $\rho_{\text{true}} \in [-1, 1]$:
   - Échantillonne $\hat{\rho}_r$ de Fisher distribution
   - Échantillonne $w_r$ de Kibble distribution (approximation via normes)
2. Combine tous les samples avec couleurs par $\rho_{\text{true}}$
3. Plot scatter: aucune structure de corrélation → indépendance

**Zoom**: Y-axis $[0.8, 1.2]$ montre concentration serrée de $w_r$ autour de 1

---

## 2.2. `experiments/paper/plot_all_mp_configs.py`

### Objectif
Générer **un plot MP par configuration** (370+ plots!)

### Fonction `fit_mp_to_bulk(...)`

**Algorithme de Fit**:
```python
# 1. Histogram du bulk
counts, bin_edges = histogram(bulk, bins=n_bins)

# 2. Define MP density fonction to fit
def mp(lambda, center, scale):
    a = center + scale * (1 - sqrt(gamma))^2
    b = center + scale * (1 + sqrt(gamma))^2
    return (1/(2πγ)) * sqrt((b-λ)(λ-a)) / λ

# 3. Fit using scipy.optimize.curve_fit
initial_guess = [median(bulk), range/theoretical_range]
popt, pcov = curve_fit(mp, bin_centers, counts, p0=initial_guess)
center_fit, scale_fit = popt

# 4. Compute R^2 goodness of fit
y_pred = mp(x, center_fit, scale_fit)
R^2 = 1 - SS_res / SS_tot
```

**Ce que ça fait**:
- Ajuste les paramètres $(center, scale)$ de la MP pour matcher l'histogramme
- Centre: décalage horizontal de la distribution
- Scale: étirement vertical/horizontal
- $R^2$: qualité du fit (1 = parfait, 0 = mauvais)

### Fonction `plot_single_mp_config(...)`

**Étapes**:
1. Charge eigenvalues et metadata
2. Sépare spike / bulk
3. Nettoie bulk (garde near-support seulement)
4. **Fit MP** aux données
5. Crée plot avec:
   - Histogram (Diaconis binning)
   - MP ajusté (rouge)
   - MP théorique standard (noir tirets)
   - Info box avec params et $R^2$
6. Sauvegarde PNG dans `mp_individual/`

**Output**: Un fichier par config, ex: `grid_n1024_N64_r1024_d1024_mp.png`

---

# 3. Fonctions Utilitaires

## 3.1. Kernels ReLU

### $\Sigma^{(1)}(\rho)$ - Base Kernel

```python
def relu_kernel_sigma1(rho):
    rho = clip(rho, -1, 1)
    theta = arccos(rho)
    return (1/π) * (sqrt(1 - rho^2) + rho * (π - theta))
```

**Géométrie**: 
- $\theta = \arccos(\rho)$ = angle entre vecteurs
- Terme $\sin\theta = \sqrt{1-\rho^2}$
- Terme $\rho(\pi - \theta)$

### $\Theta^{(1)}(\rho)$ - Derivative Kernel

```python
def relu_kernel_theta1(rho):
    rho = clip(rho, -1, 1)
    theta = arccos(rho)
    return (1/(2π)) * ((π - theta) * rho + sqrt(1 - rho^2))
```

**Signification**: Corrélation des gradients (NTK)

### $K_\infty(\rho)$ - Deterministic NTK Limit

```python
def compute_theoretical_ntk_limit(rho):
    theta1 = relu_kernel_theta1(rho)
    sigma1 = relu_kernel_sigma1(rho)
    theta = arccos(clip(rho, -1, 1))
    
    K_infty = theta1 * (1 - theta/π) + sigma1 + 1
    return K_infty
```

**Formule complète**:
$$K_\infty(\rho) = \Theta^{(1)}(\rho) \left(1 - \frac{\arccos(\rho)}{\pi}\right) + \Sigma^{(1)}(\rho) + 1$$

**Termes**:
1. $\Theta^{(1)}(\rho)(1 - \theta/\pi)$: Contribution gradient layer 1 avec damping angulaire
2. $\Sigma^{(1)}(\rho)$: Fresh basis du layer 2
3. $+1$: Bias trainable

---

## 3.2. Marchenko-Pastur Density

### Formule Standard

```python
def _marchenko_pastur_density(x, gamma_ratio):
    a = (1 - sqrt(gamma_ratio))^2
    b = (1 + sqrt(gamma_ratio))^2
    
    rho(λ) = (1/(2π)) * sqrt((b-λ)(λ-a)) / λ  for λ ∈ [a, b]
           = 0  otherwise
```

**Forme**: Quarter-circle (quart de cercle)

**Support**: $[a, b]$ où largeur $= b - a = 4\sqrt{\gamma}$

**Paramètres**:
- $\gamma = n/r$: aspect ratio
- Plus $\gamma$ grand → support plus large

---

## 3.3. Génération de Données

### Données sur la Sphère

```python
def generate_data(n, d, covariance_type='isotropic'):
    X = rng.standard_normal((n, d))  # Gaussian
    X = X / ||X||_row  # normalize to unit sphere
    return X
```

**Pourquoi sphère?**
- Simplifie théorie (rotation-invariant)
- Kernel dépend seulement de $\langle x_i, x_j \rangle = \rho$

### Paires avec $\rho$ Fixé

```python
def _generate_pairs_fixed_rho(rho, d, num_pairs):
    # 1. Sample x ~ N(0, I), normalize
    x = normalize(rng.normal(num_pairs, d))
    
    # 2. Sample z ~ N(0, I), orthogonalize to x
    z = rng.normal(num_pairs, d)
    z_perp = z - projection(z on x)
    z_perp = normalize(z_perp)
    
    # 3. Construct y = rho*x + sqrt(1-rho^2)*z_perp
    y = rho * x + sqrt(1 - rho^2) * z_perp
    
    return x, y  # both on sphere with <x,y> = rho
```

**Géométrie**:
- $x$: vecteur aléatoire sur sphère
- $z_{\perp}$: composante orthogonale à $x$
- $y$: combinaison pour obtenir exactement $\langle x, y \rangle = \rho$

---

# 4. Pipeline Complet

## Étape 1: Génération de Données (largescale*.py)

```
Input: (n, N, r, d)
      ↓
1. Generate X ∈ R^(n×d) on sphere
2. Initialize network (W, b, A, c)
3. Compute NTK Gram K ∈ R^(n×n)
      ↓ eigvalsh
4. Eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_n
      ↓ save
Output: grid_*.npz, metadata.json
```

**Répété pour**: 343 configs (old) ou 500-1000 configs (extensive)

## Étape 2: Visualisation (plot_all_figures.py)

```
Input: grid_*.npz files
      ↓
For each plot type:
  1. Load relevant data
  2. Compute statistics (mean, std, fits)
  3. Create figure with matplotlib
  4. Save PDF + PNG
      ↓
Output: fig_plot*.{pdf,png}
```

## Étape 3: MP Individual Plots (plot_all_mp_configs.py)

```
For each config:
  1. Load eigenvalues
  2. Separate spike / bulk
  3. Fit MP to bulk (optimize center, scale)
  4. Compute R^2
  5. Plot histogram + fitted MP + theoretical MP
  6. Save to mp_individual/
```

---

# 5. Structures de Données

## 5.1. Fichiers `.npz` (Spectra)

**Contenu**:
```python
{
    'eigenvalues_mean': array([λ_1, λ_2, ..., λ_n]),  # moyenné sur inits
    'eigenvalues_per_init': list of arrays,  # chaque init séparément
    'lambda_spike_mean': float,  # spike moyen
    'lambda_spike_std': float,  # spike std
    'lambda_spike_per_init': array,  # spike par init
    'n': int, 'r': int, 'd': int, 'n1': int, 'n2': int,
    'gamma_ratio': float,
    'seeds_data': array,  # pour reproductibilité
    'seeds_init': array,
    'flops_config': float  # FLOPs utilisés
}
```

## 5.2. Fichiers `_ntk_rho.npz`

**Contenu**:
```python
{
    'rho_vals': array([-1.0, -0.9, ..., 1.0]),  # 21 buckets
    'ntk_samples': array(shape=(21, samples_per_rho)),  # raw samples
    'ntk_mean': array(21),  # mean NTK par rho
    'ntk_std': array(21),  # std NTK par rho
    'k_infty': array(21),  # K_∞(rho) théorique
    'n': int, 'r': int, 'd': int, ...
    'samples_per_rho': 2000
}
```

**Génération**: Pour chaque $\rho$:
1. Génère `samples_per_rho` paires $(x_1, x_2)$ avec $\langle x_1, x_2 \rangle = \rho$
2. Calcule $\hat{\Theta}^{(2)}(x_1, x_2)$ pour chaque paire
3. Stocke tous les samples + statistiques

## 5.3. Fichiers Metadata JSON

**Contenu complet**:
```json
{
  "computation_date": "ISO timestamp",
  "base_seed": 20250131,
  "config_seed": unique_per_config,
  "n_init": 5,
  "gamma_ratio": n/r,
  "alpha_ratio": r/d,  # (nouveau)
  "n": int,
  "r": int,
  "d": int,
  "n1": N,
  "n2": N,
  "N": N,  # (nouveau)
  "regime_check": "N/max(r,d) = X.X",  # (nouveau)
  "lambda_spike_mean": float,
  "lambda_spike_std": float,
  "mp_params": {
    "alpha": K_∞(0),
    "beta": K_∞'(0),
    "gamma": correction,
    "support": [a, b],
    "gamma_ratio": n/r
  },
  "flops_config": float,
  "python": "version",
  "platform": "system",
  "numpy": "version",
  "notes": "description"
}
```

---

# 6. Algorithmes Clés

## 6.1. Fisher-Kibble Decomposition

**Théorie**: Pour $h^{(1)}(x) \in \mathbb{R}^r$ avec composantes i.i.d.:

$$\hat{\rho}_r = \frac{\langle h^{(1)}(x_1), h^{(1)}(x_2) \rangle}{\|h^{(1)}(x_1)\| \|h^{(1)}(x_2)\|} \sim \text{Fisher}(\rho, r)$$

$$w_r = \frac{\|h^{(1)}(x_1)\| \|h^{(1)}(x_2)\|}{r} \sim \text{Kibble}(r, \rho)$$

**Indépendance**: $\hat{\rho}_r \perp w_r$ (Lemme 2.1)

**Implémentation** (approx):
```python
# Fisher: correlation of Gaussians on sphere
# Approximation: Normal(rho_true, 1/sqrt(r))
rho_hat = clip(Normal(rho_true, 1/sqrt(r)), -1, 1)

# Kibble: product of chi norms
# Approximation: product of ||X||_r where X ~ N(0, I_r)
x_norm = ||Normal(n_samples, r)||
y_norm = ||Normal(n_samples, r)||
w_r = (x_norm * y_norm) / r  # concentrates at 1
```

## 6.2. Spike Identification

**Algorithme**:
```python
def identify_spikes(eigenvalues, gamma):
    # 1. Compute theoretical bulk support
    a = (1 - sqrt(gamma))^2
    b = (1 + sqrt(gamma))^2
    
    # 2. Adaptive threshold
    threshold = max(2*b, 0.1)  # 2× bulk maximum
    
    # 3. Separate
    spikes = eigenvalues[eigenvalues > threshold]
    bulk = eigenvalues[eigenvalues <= threshold]
    
    return spikes, bulk
```

**Résultat**: 100% des configs ont **exactement 1 spike**

---

# 7. Résumé du Flow Complet

```
1. COMPUTATION (largescale*.py)
   ├─ For each (N, n, r, d):
   │  ├─ Generate data X on sphere
   │  ├─ Initialize network
   │  ├─ Compute NTK Gram matrix K
   │  ├─ Eigendecompose → λ_1, ..., λ_n
   │  └─ Save .npz + metadata.json
   └─ Output: 343-1000 config files

2. ANALYSIS
   ├─ Identify spikes (all configs have 1)
   ├─ Compute spike vs n scaling (linear!)
   └─ Group by (gamma, alpha, N)

3. VISUALIZATION (plot_all_figures.py)
   ├─ Plot 1: Variance/Std decay (Monte Carlo)
   ├─ Plot 1a: Tail probability (aligned)
   ├─ Plot 2: NTK concentration (from _ntk_rho files)
   ├─ Plot 3: Spectral decay (from eigenvalues)
   ├─ Plot 4: MP spectrum (3 gammas)
   ├─ Plot 5: FLOPs analysis (from metadata)
   └─ Plot 6: Fisher-Kibble (samples)

4. INDIVIDUAL MP PLOTS (plot_all_mp_configs.py)
   └─ For each config: fit + plot → 370+ PNG files
```

---

# 8. Questions Fréquentes

## Q: Pourquoi $d = r$ dans les données?

**R**: Policy `d_policy="equal_r"` dans largescale.py (ligne 1018)

Simplifie en ayant seulement $\gamma = n/r$ comme ratio principal.

## Q: Pourquoi seulement 1 spike?

**R**: Structure mathématique du kernel:
- Mode constant $v = (1, ..., 1)/\sqrt{n}$ → eigenvalue $\sim n$
- Modes non-constants → eigenvalues $\sim O(1)$ (MP bulk)

## Q: Pourquoi $w_r > 0$ toujours?

**R**: $w_r = \|h_1\| \|h_2\| / r$ est un **produit de normes** (toujours $\geq 0$)

## Q: Comment calculer $K_\infty(\rho)$?

**R**: Formule explicite combinant kernels ReLU:
$$K_\infty = \Theta^{(1)}(\rho)(1-\arccos(\rho)/\pi) + \Sigma^{(1)}(\rho) + 1$$

Voir `K_INFINITY_CALCULATION.md` pour détails complets.

---

**Tous les codes sont maintenant expliqués en détail!** 📚


