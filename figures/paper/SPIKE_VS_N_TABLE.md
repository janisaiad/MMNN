# Spike Values vs n (Number of Data Points)

## Summary Table

| n | # Configs | Spike Count | Mean Spike | Std | Min | Max | Spike/n |
|---|-----------|-------------|------------|-----|-----|-----|---------|
| 16 | 49 | 1 | 23.8 | 0.5 | 22.9 | 25.3 | 1.490 |
| 32 | 49 | 1 | 47.6 | 1.1 | 45.1 | 50.6 | 1.486 |
| 64 | 49 | 1 | 95.2 | 2.1 | 90.5 | 101.3 | 1.488 |
| 128 | 49 | 1 | 190.5 | 4.4 | 182.2 | 202.4 | 1.488 |
| 256 | 49 | 1 | 381.4 | 8.6 | 364.6 | 403.6 | 1.490 |
| 512 | 49 | 1 | 761.1 | 16.5 | 726.3 | 800.1 | 1.487 |
| 1024 | 49 | 1 | 1523.6 | 34.2 | 1448.0 | 1602.2 | 1.488 |

## Interpretation

The table shows that:
- **All configurations have exactly 1 spike** (100% consistency)
- **Spike scales linearly with n**: Mean ratio spike/n = 1.488 ± 0.001
- **Theoretical prediction**: Spike ~ n × K_∞(0) where K_∞(0) ≈ 1.318

This validates Theorem 4.2: The spike eigenvalue is O(n).