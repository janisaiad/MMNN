# Baseline sweep summary

Target: cos(2 π factor x), N = base×factor, bs ∈ {1,2,4,8,16}, L ∈ {1..2×factor}. **Worked** = final test error < 0.01; **Failed** = test error ≥ 0.5 or NaN/Inf.

## Best configs (by final test error)

| config | factor | N | bs | L | final_test_err | final_train_err | epochs |
|--------|--------|---|-----|---|----------------|-----------------|--------|
| f3_N128_bs8_L3 | 3 | 128 | 8 | 3 | 5.0374e-01 | 5.0785e-01 | 771 |
| f3_N128_bs8_L4 | 3 | 128 | 8 | 4 | 5.4244e-01 | 5.5074e-01 | 818 |
| f3_N128_bs16_L3 | 3 | 128 | 16 | 3 | 5.6673e-01 | 5.7410e-01 | 1076 |
| f3_N128_bs16_L4 | 3 | 128 | 16 | 4 | 6.0924e-01 | 6.2114e-01 | 1382 |
| f3_N128_bs4_L3 | 3 | 128 | 4 | 3 | 6.5741e-01 | 6.7773e-01 | 576 |
| f3_N128_bs2_L3 | 3 | 128 | 2 | 3 | 6.6088e-01 | 6.8412e-01 | 477 |
| f3_N128_bs1_L3 | 3 | 128 | 1 | 3 | 6.7177e-01 | 6.9262e-01 | 676 |
| f3_N128_bs8_L1 | 3 | 128 | 8 | 1 | 7.1981e-01 | 7.3498e-01 | 775 |
| f3_N128_bs16_L1 | 3 | 128 | 16 | 1 | 7.6352e-01 | 7.8024e-01 | 761 |
| f4_N256_bs1_L3 | 4 | 256 | 1 | 3 | 7.7963e-01 | 7.9026e-01 | 572 |
| f3_N128_bs4_L4 | 3 | 128 | 4 | 4 | 8.1078e-01 | 8.4253e-01 | 642 |
| f3_N128_bs4_L2 | 3 | 128 | 4 | 2 | 8.2199e-01 | 8.5513e-01 | 476 |
| f3_N128_bs16_L2 | 3 | 128 | 16 | 2 | 8.3104e-01 | 8.5646e-01 | 618 |
| f3_N128_bs8_L2 | 3 | 128 | 8 | 2 | 8.3221e-01 | 8.5990e-01 | 503 |
| f3_N128_bs2_L2 | 3 | 128 | 2 | 2 | 8.4959e-01 | 8.8764e-01 | 468 |
| f3_N64_bs4_L2 | 3 | 64 | 4 | 2 | 8.5323e-01 | 9.2203e-01 | 613 |
| f3_N128_bs2_L4 | 3 | 128 | 2 | 4 | 8.5956e-01 | 8.9672e-01 | 447 |
| f3_N128_bs1_L2 | 3 | 128 | 1 | 2 | 8.6026e-01 | 9.0000e-01 | 391 |
| f4_N256_bs2_L4 | 4 | 256 | 2 | 4 | 8.6567e-01 | 8.7619e-01 | 543 |
| f4_N256_bs2_L3 | 4 | 256 | 2 | 3 | 8.6689e-01 | 8.7491e-01 | 548 |

## Worked vs did not

- **Worked** (test err < 0.01): 0 configs.
- **Failed** (test err ≥ 0.5 or NaN/Inf): 196 configs.
- Total completed: 196.

## All worked configs (sorted by final test error)

| # | config | factor | N | bs | L | final_test_err | final_train_err | min_loss | epochs |
|---|--------|--------|---|-----|---|----------------|-----------------|----------|--------|

### Worked (representative)


### Did not work (representative)

- `f3_N128_bs16_L1` test_err=7.6352e-01 L=1 N=128 bs=16
- `f3_N128_bs16_L2` test_err=8.3104e-01 L=2 N=128 bs=16
- `f3_N128_bs16_L3` test_err=5.6673e-01 L=3 N=128 bs=16
- `f3_N128_bs16_L4` test_err=6.0924e-01 L=4 N=128 bs=16
- `f3_N128_bs16_L5` test_err=2.0040e+00 L=5 N=128 bs=16
- `f3_N128_bs16_L6` test_err=2.0040e+00 L=6 N=128 bs=16
- `f3_N128_bs1_L1` test_err=1.0330e+00 L=1 N=128 bs=1
- `f3_N128_bs1_L2` test_err=8.6026e-01 L=2 N=128 bs=1
- `f3_N128_bs1_L3` test_err=6.7177e-01 L=3 N=128 bs=1
- `f3_N128_bs1_L4` test_err=9.2000e-01 L=4 N=128 bs=1
- `f3_N128_bs1_L5` test_err=2.0108e+00 L=5 N=128 bs=1
- `f3_N128_bs1_L6` test_err=2.0107e+00 L=6 N=128 bs=1
- `f3_N128_bs2_L1` test_err=9.5540e-01 L=1 N=128 bs=2
- `f3_N128_bs2_L2` test_err=8.4959e-01 L=2 N=128 bs=2
- `f3_N128_bs2_L3` test_err=6.6088e-01 L=3 N=128 bs=2
- `f3_N128_bs2_L4` test_err=8.5956e-01 L=4 N=128 bs=2
- `f3_N128_bs2_L5` test_err=2.0189e+00 L=5 N=128 bs=2
- `f3_N128_bs2_L6` test_err=2.0178e+00 L=6 N=128 bs=2
- `f3_N128_bs4_L1` test_err=8.7111e-01 L=1 N=128 bs=4
- `f3_N128_bs4_L2` test_err=8.2199e-01 L=2 N=128 bs=4
- `f3_N128_bs4_L3` test_err=6.5741e-01 L=3 N=128 bs=4
- `f3_N128_bs4_L4` test_err=8.1078e-01 L=4 N=128 bs=4
- `f3_N128_bs4_L5` test_err=2.0061e+00 L=5 N=128 bs=4
- `f3_N128_bs4_L6` test_err=2.0045e+00 L=6 N=128 bs=4
- `f3_N128_bs8_L1` test_err=7.1981e-01 L=1 N=128 bs=8
- `f3_N128_bs8_L2` test_err=8.3221e-01 L=2 N=128 bs=8
- `f3_N128_bs8_L3` test_err=5.0374e-01 L=3 N=128 bs=8
- `f3_N128_bs8_L4` test_err=5.4244e-01 L=4 N=128 bs=8
- `f3_N128_bs8_L5` test_err=2.0041e+00 L=5 N=128 bs=8
- `f3_N128_bs8_L6` test_err=2.0042e+00 L=6 N=128 bs=8
- ... and 166 more.

## Table by factor (mean test error, count worked/total)

| factor | N range | total | worked | failed | best test err |
|--------|---------|-------|--------|--------|---------------|
| 3 | 32–128 | 90 | 0 | 90 | 5.0374e-01 |
| 4 | 64–256 | 106 | 0 | 106 | 7.7963e-01 |

---

Full results CSV: `baseline_sweep_results.csv`. Histogram: `baseline_sweep_loss_histogram.png`.