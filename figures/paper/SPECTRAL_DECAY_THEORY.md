# Théorie de la Décroissance Spectrale RKHS

## Formule Théorique

Pour les réseaux de neurones ReLU, la décroissance spectrale des valeurs propres du noyau NTK suit:

$$\mu_k \sim (c_1 - c_{-1}) C(d,\nu) k^{-d-2\nu+1}$$

où:
- $\mu_k$ = $k$-ième valeur propre (ordonnée décroissante)
- $k$ = indice harmonique sphérique
- $d$ = dimension de l'espace d'entrée
- $\nu$ = régularité de Sobolev du noyau
- $c_1, c_{-1}$ = constantes liées aux valeurs de bord du noyau
- $C(d,\nu)$ = constante de normalisation dépendant de $d$ et $\nu$

---

## Paramètres pour ReLU

### Régularité de Sobolev: $\nu = 1/2$

Pour les réseaux ReLU:
- Le noyau est **Lipschitz continu** mais **non différentiable**
- Cela correspond à $\nu = 1/2$ dans l'échelle de Sobolev $W^{\nu,2}$
- Plus lisse que $C^0$ (continu), moins lisse que $C^1$ (différentiable)

### Exposant Simplifié

Avec $\nu = 1/2$:

$$-d - 2\nu + 1 = -d - 2 \cdot \frac{1}{2} + 1 = -d - 1 + 1 = -d$$

**Donc**: $\mu_k \sim C \cdot k^{-d}$

---

## Interprétation Géométrique

### Harmoniques Sphériques

Les valeurs propres correspondent aux **harmoniques sphériques** sur $\mathbb{S}^{d-1}$:
- $k = 0$: mode constant (le spike!)
- $k = 1, 2, \ldots$: modes oscillants de fréquence croissante

### Décroissance en Puissance

La loi de puissance $k^{-d}$ signifie:
- **Modes basse fréquence** ($k$ petit): grandes valeurs propres → apprises rapidement
- **Modes haute fréquence** ($k$ grand): petites valeurs propres → apprises lentement
- **Taux de décroissance**: dépend linéairement de la dimension $d$

---

## Valeurs Numériques (Plot 3)

### Configuration: $d = 64$

```
Paramètres:
  d = 64 (dimension d'entrée)
  ν = 0.5 (régularité Sobolev)
  Exposant théorique: -(d + 2ν - 1) = -64

Formule: μ_k ~ C k^{-64}
```

### Ajustement Empirique

Le Plot 3 montre:
1. **Courbes empiriques** (couleurs): valeurs propres des matrices NTK réelles
2. **Courbe ajustée** (noir tiret): régression log-log sur données empiriques
3. **Théorie RKHS** (rouge): $\mu_k \sim k^{-d}$ avec normalisation ajustée
4. **Références** (gris): $k^{-1}$ et $k^{-2}$ pour comparaison

---

## Validation Théorique

### Ce que montre le plot:

1. **Superposition des rangs**: Les courbes pour différents $r \in \{64, 128, 256, 512\}$ se superposent
   → Valide l'**équivalence RKHS** (Corollaire 3.5)

2. **Pente empirique ≈ -d**: L'ajustement log-log donne un exposant proche de $-d$
   → Valide la **régularité Sobolev** $\nu = 1/2$

3. **Pas de cassure**: Décroissance en loi de puissance sans changement de régime
   → Valide l'**homogénéité** du noyau

---

## Comparaison avec la Littérature

### Noyaux Gaussiens (Kernel Machines)

Pour le noyau Gaussien $K(x,y) = \exp(-\|x-y\|^2/\sigma^2)$:
- Régularité: $\nu = \infty$ (infiniment différentiable)
- Décroissance: **exponentielle** $\mu_k \sim \exp(-c k^{2/d})$
- Beaucoup plus rapide que ReLU!

### Noyaux Laplaciens

Pour le noyau Laplacien $K(x,y) = \exp(-\|x-y\|/\sigma)$:
- Régularité: $\nu = 1/2$ (comme ReLU)
- Décroissance: $\mu_k \sim k^{-d-1}$ (similaire à ReLU)

### ReLU NTK (Notre Cas)

- Régularité: $\nu = 1/2$ (Lipschitz mais non $C^1$)
- Décroissance: $\mu_k \sim k^{-d}$ (loi de puissance)
- Intermédiaire entre noyaux polynomiaux et Gaussiens

---

## Implications Pratiques

### 1. Vitesse d'Apprentissage

Les modes haute fréquence ($k$ grand) ont des valeurs propres $\mu_k \sim k^{-d}$ très petites:
- Pour $d = 64$ et $k = 100$: $\mu_k \sim 10^{-128}$ (quasi-nul!)
- En pratique: seuls les premiers $k \ll d^{1/(d-1)}$ modes sont appris

### 2. Dimension Effective

Le **rang effectif** du noyau est:

$$r_{\text{eff}} = \frac{(\sum_k \mu_k)^2}{\sum_k \mu_k^2} \ll n$$

Pour grande dimension $d$, $r_{\text{eff}}$ est très petit → **régularisation implicite**

### 3. Complexité d'Approximation

Pour approximer une fonction dans le RKHS à précision $\epsilon$:
- Nombre de modes nécessaires: $K \sim \epsilon^{-1/d}$
- **Malédiction de la dimension**: croissance exponentielle avec $d$

---

## Lien avec les Autres Plots

### Plot 1 (Concentration)

La décroissance $k^{-d}$ implique une concentration exponentielle:
- Variance des valeurs propres: $\text{Var}(\mu_k) \sim O(1/r^2)$
- Justifie la concentration de Fisher-Kibble

### Plot 4 (Marchenko-Pastur)

Le **spike** ($\mu_1 \sim O(n)$) est séparé du **bulk** ($\mu_k \sim O(1)$ pour $k \geq 2$):
- Spike: mode constant ($k=0$)
- Bulk: suit MP avec décroissance $k^{-d}$ interne

### Plot 7 (Puiseux)

Near $\rho \to 1$, le comportement $t^{1/2}$ vient de:
- Terme dominant du développement en harmoniques sphériques
- Lié à la régularité $\nu = 1/2$

---

## Formule Complète

En incluant tous les termes:

$$\mu_k = (c_1 - c_{-1}) \frac{\Gamma(d/2)}{\Gamma(k+d/2)} \frac{(k + d/2 - 1)!}{k! \, (d/2)!} k^{-d-2\nu+1} + o(k^{-d-2\nu+1})$$

Pour ReLU avec $\nu = 1/2$:

$$\mu_k \sim C(d) \cdot k^{-d} \quad \text{où} \quad C(d) = \frac{\Gamma(d/2)}{2\pi^{d/2}}$$

---

## Références Théoriques

1. **Sobolev Spaces**: Adams & Fournier, "Sobolev Spaces" (2003)
2. **Spherical Harmonics**: Müller, "Spherical Harmonics" (1966)
3. **NTK Theory**: Jacot et al., "Neural Tangent Kernel" (2018)
4. **RKHS Decay**: Steinwart & Christmann, "Support Vector Machines" (2008)

---

## Code d'Implémentation

```python
def theoretical_spectral_decay(k, d, nu=0.5, C_norm=1.0):
    """
    compute theoretical RKHS eigenvalue decay
    
    parameters:
    - k: harmonic index (array)
    - d: input dimension
    - nu: sobolev smoothness (0.5 for relu)
    - C_norm: normalization constant
    
    returns:
    - mu_k: eigenvalues following k^{-d-2nu+1}
    """
    exponent = -(d + 2*nu - 1)  # we compute exponent #
    mu_k = C_norm * k**exponent  # we compute decay #
    return mu_k  # we return eigenvalues #
```

---

**Résumé**: La décroissance spectrale $\mu_k \sim k^{-d}$ pour ReLU avec $\nu = 1/2$ est **validée empiriquement** dans Plot 3 et **cohérente avec la théorie RKHS**. L'exposant $-d$ reflète la **dimension de l'espace** et la **régularité Lipschitz** du noyau ReLU.

---

**Date**: 2025-01-31  
**Plot**: Figure 3 (Spectral Decay)  
**Formule**: $\mu_k \sim (c_1 - c_{-1}) C(d,\nu) k^{-d-2\nu+1}$ avec $\nu = 1/2$

