# Explication Précise: Ce qui est Ploté et Calcul du Log Ratio

## 1. Ce qui est Ploté dans les Fonctions Low-Rank

### Fonctions Plotées: `f_k(x)` pour k = 1, ..., 15

**Définition mathématique:**
$$f_k(x) = m_k(x; W) = \mathbb{E}_{C_1}[w_1(C_1, k) \cdot \phi_1(f_1(C_1), x)]$$

**Implémentation dans le code:**
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
```

**En détail:**
1. Pour chaque point $x$ dans $[-1, 1]$ (500 points):
   - On calcule `inner = f1 @ x` → shape [777, 1]
   - On applique ReLU: `phi1 = ReLU(inner)` → shape [777, 1]
   - Pour chaque canal $k$ (15 canaux):
     - On prend les poids $w_1[:, k]$ (shape [777])
     - On multiplie: `w1_k * phi1` → shape [777]
     - On fait la moyenne: `mean(w1_k * phi1)` → scalaire
   - Ce scalaire est $f_k(x)$ pour ce point $x$

2. **Résultat:** Pour chaque canal $k$, on obtient une fonction $f_k(x)$ définie sur $[-1, 1]$

**Ce que cela représente:**
- $f_k(x)$ est la **sortie du canal $k$ de la couche low-rank**
- C'est la moyenne pondérée (par $w_1$) des activations ReLU des 777 features aléatoires
- Ces 15 fonctions sont ensuite mélangées via la matrice $L$ pour former $H_2$

---

## 2. Calcul du Log Ratio

### Définition: $R_{i,j} = \log(|f_i|) - \log(|f_j|)$

**Implémentation:**
```python
def compute_log_ratios(self, f_k):
    # f_k: [r, batch_size] = [15, 1] (calculé à x ≈ 0)
    r = f_k.shape[0]  # r = 15
    
    abs_f = torch.abs(f_k) + self.epsilon  # [15, 1] - valeurs absolues
    log_f = torch.log(abs_f)  # [15, 1] - logarithmes
    
    log_ratios = torch.zeros(r, r, f_k.shape[1])  # [15, 15, 1]
    for k in range(r):
        for ell in range(r):
            log_ratios[k, ell] = log_f[k] - log_f[ell]  # R_{k,ell}
```

**Étapes:**
1. **Calcul de $f_k$ à $x \approx 0$:**
   - On utilise $x = 10^{-6}$ (pas exactement 0 car ReLU(0) = 0 donnerait $f_k = 0$)
   - On calcule $f_k(x)$ pour tous les 15 canaux → shape [15, 1]

2. **Calcul des valeurs absolues:**
   - $|f_k| = \text{abs}(f_k) + \epsilon$ où $\epsilon = 10^{-8}$ (évite log(0))

3. **Calcul des logarithmes:**
   - $\log(|f_k|)$ pour chaque canal $k$

4. **Calcul des log ratios:**
   - $R_{i,j} = \log(|f_i|) - \log(|f_j|)$ pour toutes les paires $(i, j)$
   - Résultat: matrice [15, 15] où chaque entrée $(i,j)$ est $R_{i,j}$

**Interprétation:**
- $R_{i,j} > 0$: le canal $i$ domine le canal $j$ (car $|f_i| > |f_j|$)
- $R_{i,j} < 0$: le canal $j$ domine le canal $i$ (car $|f_i| < |f_j|$)
- $R_{i,j} = 0$: les canaux ont la même magnitude
- La diagonale est toujours 0: $R_{i,i} = 0$

**Exemple concret:**
- Si $f_3(x=0) = 0.001$ et $f_5(x=0) = 0.0001$
- Alors $R_{3,5} = \log(0.001) - \log(0.0001) = -6.91 - (-9.21) = 2.30$
- Cela signifie que le canal 3 est environ $e^{2.30} \approx 10$ fois plus fort que le canal 5 à $x \approx 0$

---

## 3. Architecture Mean-Field (2 couches)

**Structure:**
```
Input x
  ↓
f1 (frozen random features) [777, 1]
  ↓
ReLU(phi1) [777, 1]
  ↓
w1 (trainable) [777, 15] → f_k [15, 1]  ← CE QUI EST PLOTÉ
  ↓
L (frozen mixing) [777, 15] → H2 [777, 1]
  ↓
ReLU(phi2) [777, 1]
  ↓
w2 (trainable) [777] → y_hat [1]
```

**Les fonctions plotées sont $f_k$**, qui sont:
- Les **sorties de la couche low-rank** (comme dans MMNN)
- Avant le mélange via $L$
- Ce sont les 15 composantes low-rank qui seront ensuite mélangées

---

## 4. Pourquoi elles peuvent sembler "ReLU-like"

Les fonctions $f_k(x)$ sont des **moyennes pondérées de ReLU activations**:
- Chaque $f_k(x) = \frac{1}{777} \sum_{i=1}^{777} w_1[i,k] \cdot \text{ReLU}(f_1[i] \cdot x)$
- Comme ce sont des combinaisons linéaires de fonctions ReLU (qui sont linéaires par morceaux), les $f_k$ sont aussi **linéaires par morceaux**
- Elles peuvent donc sembler "semi-linéaires" ou "ReLU-like" car elles héritent de cette structure

**Cependant**, après entraînement, les poids $w_1$ sont ajustés pour que les combinaisons de ces 15 fonctions (via $L$) puissent approximer la fonction cible complexe (cosinus multi-fréquences).

---

## 5. Pourquoi les Fonctions Passent par 0 et Sont "ReLU-like"

**Pourquoi $f_k(0) = 0$:**
- À $x = 0$, on a: $f_1 \cdot 0 = 0$
- Donc: $\text{ReLU}(0) = 0$
- Donc: $f_k(0) = \mathbb{E}_{C_1}[w_1(C_1, k) \cdot 0] = 0$
- **C'est NORMAL** - toutes les fonctions passent par 0 car ce sont des combinaisons de ReLU sans biais

**Pourquoi elles sont "ReLU-like" (linéaires par morceaux):**
- Les fonctions $f_k$ sont des **moyennes pondérées de fonctions ReLU**
- Chaque ReLU est linéaire par morceaux (0 pour $x < 0$, linéaire pour $x > 0$)
- Une combinaison linéaire de fonctions linéaires par morceaux est aussi linéaire par morceaux
- Donc $f_k$ sont **linéaires par morceaux** (pas des cosinus lisses)
- **C'est la structure mathématique** de ces fonctions - elles héritent de la structure ReLU

**Après entraînement:**
- Les poids $w_1$ sont ajustés pour que les **combinaisons** de ces 15 fonctions (via $L$) puissent approximer la fonction cible
- Individuellement, chaque $f_k$ reste linéaire par morceaux
- Mais ensemble, après mélange via $L$, elles peuvent créer des formes complexes

---

## 6. Résumé

**Ce qui est ploté:**
- Les 15 fonctions $f_k(x)$ pour $x \in [-1, 1]$
- Où $f_k(x) = \mathbb{E}_{C_1}[w_1(C_1, k) \cdot \text{ReLU}(f_1(C_1) \cdot x)]$
- Ce sont les sorties de la couche low-rank (comme `fcs[1]` dans MMNN)
- **Propriétés:** Linéaires par morceaux, passent par 0 (normal pour ReLU sans biais)

**Comment le log ratio est calculé:**
- On calcule $f_k$ à $x \approx 0$ (15 valeurs, très petites mais non nulles)
- On calcule $R_{i,j} = \log(|f_i| + \epsilon) - \log(|f_j| + \epsilon)$ pour toutes les paires
- Résultat: matrice [15, 15] montrant la spécialisation des canaux
