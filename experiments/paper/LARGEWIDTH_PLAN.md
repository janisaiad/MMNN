# Large-Width Computation Plan

## Problem Identified ✋

**Current data has N too small!**

Example from Plot 3:
- $r = 512$, $d = 512$, but $N = 64$
- Ratio: $N/\max(r,d) = 64/512 = 0.125$ ❌ (N is SMALLER than r!)

**This is NOT the large-width regime!** We need $N \gg r, d$ for infinite-width limit.

---

## New Computation Plan

### Width Range (Large!)

$N \in \{2048, 4096, 8192, 16384, 32768, 65536\}$ (powers: $2^{11}$ to $2^{16}$)

### Key Ratios to Vary

#### 1. **Aspect Ratio** $\gamma = n/r$
Values: $\{0.25, 0.5, 1.0, 2.0, 4.0\}$
- 0.25: rank-rich ($r = 4n$)
- 0.5: rank-rich ($r = 2n$)
- 1.0: balanced
- 2.0: data-rich ($n = 2r$)
- 4.0: data-rich ($n = 4r$)

#### 2. **Dimension-Rank Ratio** $\alpha = r/d$
Values: $\{0.5, 1.0, 2.0\}$
- 0.5: $d = 2r$ (high-dimensional input)
- 1.0: $d = r$ (balanced)
- 2.0: $d = r/2$ (low-dimensional input)

### Constraints

1. **Large-width regime**: $N \geq 4 \times \max(r, d)$ (minimum 4× separation)
2. **Memory**: $n \leq 2048$ (Gram matrix size)
3. **Base rank**: $r \sim 64$-$128$ (scales with width)

### Example Configs

| $N$ | $r$ | $d$ | $n$ | $\gamma$ | $\alpha$ | $N/\max(r,d)$ | Regime |
|-----|-----|-----|-----|----------|----------|---------------|---------|
| 2048 | 64 | 64 | 32 | 0.5 | 1.0 | **32×** | ✅ Large-width |
| 4096 | 128 | 64 | 128 | 1.0 | 2.0 | **32×** | ✅ Large-width |
| 8192 | 256 | 256 | 512 | 2.0 | 1.0 | **32×** | ✅ Large-width |

---

## Computational Cost Estimate

### Per Configuration

- Gram matrix: $O(n^2 r)$ FLOPs (dominant)
- Eigendecomposition: $O(n^3)$ FLOPs
- Total per init: $\sim n^2(r + n)$ FLOPs

### Example (N=2048, n=128, r=64):
- FLOPs per init: $\sim 128^2 \times (64 + 128) \sim 3 \times 10^6$
- 3 inits: $\sim 10^7$ FLOPs
- Time: ~1-5 minutes per config

### Total Grid

- Configs: $3 \times 3 \times 3 = 27$ (for N ∈ {2048, 4096, 8192})
- Full grid: $6 \times 5 \times 3 = 90$ configs (if extend to all widths)
- Estimated time: **~2-10 hours** (parallelizable)

---

## Phased Execution

### Phase 1 (Testing): N ∈ {2048, 4096, 8192}
- 3 widths × 3 gammas × 3 alphas = **27 configs**
- Time: ~1-2 hours
- Purpose: Validate large-width convergence

### Phase 2 (Extended): Add N ∈ {16384, 32768}
- +2 widths × 3 × 3 = **+18 configs**
- Time: ~2-4 hours
- Purpose: Show width scaling

### Phase 3 (Full): Add N = 65536
- +1 width × 3 × 3 = **+9 configs**
- Time: ~2-4 hours
- Purpose: Extreme large-width limit

---

## Expected Results

### 1. Convergence with N

- **Spike value**: Should stabilize at $n \times K_\infty(0) \approx 1.318n$ as $N \to \infty$
- **Bulk width**: Should stabilize to theoretical MP support
- **NTK variance**: Should decay as $O(1/r)$ regardless of $N$ (once $N$ large enough)

### 2. Ratio Independence

With $N$ large:
- Results should depend only on $\gamma = n/r$ and $\alpha = r/d$
- Width $N$ should have minimal effect (finite-width corrections vanish)

---

## Script Created

**File**: `experiments/paper/largescale_largewidth.py`

**Usage**:
```bash
# run computation (Phase 1)
python experiments/paper/largescale_largewidth.py

# run in background for long computation
nohup python experiments/paper/largescale_largewidth.py > largewidth_run.log 2>&1 &
```

**Output**: `refs/paper/data/largewidth/*.npz` and metadata

---

## Next Steps

1. ✅ **Script created**: Ready to run
2. ⏳ **Start Phase 1**: N ∈ {2048, 4096, 8192} (testing)
3. ⏳ **Validate**: Check convergence vs width
4. ⏳ **Extend**: Add larger widths if needed
5. ⏳ **Update plots**: Use large-width data

---

**Status**: Ready to launch large-width computation!  
**Hardware needed**: Sufficient RAM for N=65536 networks (~16GB recommended)

