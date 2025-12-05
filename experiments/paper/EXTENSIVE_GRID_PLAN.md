# Extensive Large-Width Grid Computation Plan

## Nouvelle Grille BEAUCOUP Plus Grande

### Paramètres

#### Largeurs (Network Width)
$N \in \{2048, 4096, 8192, 16384, 32768, 65536, 131072\}$ 

**7 valeurs** (powers: $2^{11}$ à $2^{17}$)

#### Nombre de données
$n \in \{16, 32, 64, 128, 256, 512, 1024, 2048\}$

**8 valeurs**

#### Rang
$r \in \{16, 32, 64, 128, 256, 512, 1024\}$

**7 valeurs**

#### Dimension
$d \in \{16, 32, 64, 128, 256, 512, 1024\}$

**7 valeurs**

---

## Taille de la Grille

### Maximum Théorique
$7 \times 8 \times 7 \times 7 = 2744$ configurations possibles

### Après Contrainte Large-Width
**Contrainte**: $N \geq 8 \times \max(r, d)$ (régime large largeur)

**Estimé**: ~500-1000 configurations valides

---

## Exemples de Configurations Valides

| $N$ | $n$ | $r$ | $d$ | $\gamma$ | $\alpha$ | $N/\max(r,d)$ | Valide? |
|-----|-----|-----|-----|----------|----------|---------------|---------|
| 2048 | 128 | 64 | 64 | 2.0 | 1.0 | **32×** | ✅ |
| 4096 | 256 | 128 | 64 | 2.0 | 2.0 | **32×** | ✅ |
| 8192 | 512 | 256 | 256 | 2.0 | 1.0 | **32×** | ✅ |
| 16384 | 1024 | 512 | 512 | 2.0 | 1.0 | **32×** | ✅ |
| 32768 | 2048 | 1024 | 512 | 2.0 | 2.0 | **32×** | ✅ |
| 65536 | 2048 | 1024 | 1024 | 2.0 | 1.0 | **64×** | ✅ |
| 131072 | 2048 | 1024 | 1024 | 2.0 | 1.0 | **128×** | ✅ |

---

## Couverture des Ratios

### Ratio $\gamma = n/r$ (Aspect)

Avec $n \in \{16,...,2048\}$ et $r \in \{16,...,1024\}$:

$\gamma \in [16/1024, 2048/16] = [0.016, 128]$

**Couverture**: ~50 valeurs distinctes de $\gamma$

### Ratio $\alpha = r/d$ (Dimension-Rank)

Avec $r \in \{16,...,1024\}$ et $d \in \{16,...,1024\}$:

$\alpha \in [16/1024, 1024/16] = [0.016, 64]$

**Couverture**: ~50 valeurs distinctes de $\alpha$

---

## Coût Computationnel

### Par Configuration

- Gram matrix: $O(n^2 r)$ FLOPs
- Eigendecomposition: $O(n^3)$ FLOPs
- Temps: ~1 sec (small n) à ~10 min (n=2048)

### Total Estimé

- 500-1000 configs × 3 inits × 1-10 min = **~25-500 heures**
- **Parallélisable** sur plusieurs CPUs/GPUs

---

## Stratégie d'Exécution

### Phase 1: N ∈ {2048, 4096, 8192} (En cours)
- ~100-200 configs
- Temps: 2-6 heures
- **Status**: ⏳ Running...

### Phase 2: N ∈ {16384, 32768}
- ~150-300 configs
- Temps: 5-15 heures
- **Status**: À lancer

### Phase 3: N ∈ {65536, 131072}
- ~100-200 configs  
- Temps: 10-30 heures
- **Status**: À lancer

---

## Outputs

### Fichiers par Config

1. `lw_N{N}_n{n}_r{r}_d{d}.npz` - Eigenvalues spectra
2. `lw_N{N}_n{n}_r{r}_d{d}_metadata.json` - Metadata with regime check
3. `lw_N{N}_n{n}_r{r}_d{d}_ntk_rho.npz` - NTK-vs-rho (optionnel)

### Master Index

`refs/paper/data/largewidth/largewidth_extensive_index.json`

### MP Individual Plots

`figures/paper/mp_individual/lw_N{N}_n{n}_r{r}_d{d}_mp.png` (one per config!)

---

## Parallélisation

Le script peut être parallélisé:

```bash
# split by width power
for N_pow in {11..17}; do
    python largescale_largewidth.py --width_pow $N_pow &
done
```

Ou utiliser GNU parallel, joblib, ou cluster computing.

---

## Résultats Attendus

### 1. Convergence en N

Pour $\gamma, \alpha$ fixés, quand $N$ augmente:
- Spike → $n \times K_\infty(0)$ (stabilisation)
- Bulk → Distribution MP théorique (meilleur fit)
- Variance NTK → $O(1/r)$ (indépendant de $N$)

### 2. Dépendance en $\gamma = n/r$

- Spike: $\sim n$ (linéaire)
- Bulk width: $\sim \sqrt{\gamma}$ (MP scaling)

### 3. Dépendance en $\alpha = r/d$

- Spectral decay: peut dépendre de $d$
- Concentration: devrait dépendre de $r$ seulement

---

**Status**: Script modifié pour grille extensive  
**Action**: Relancer avec nouvelle configuration  
**Estimation**: ~500-1000 configs, 25-500 heures (parallélisable)

