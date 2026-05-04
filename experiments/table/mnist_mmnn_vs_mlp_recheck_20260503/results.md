# MNIST: MLP vs MMNN

## Run 2 — MLP vs low-rank MMNN with random/frozen features (fixWb)

| Model | Config | Params | Trainable | Test acc | Test loss |
|-------|--------|--------|-----------|----------|-----------|
| **MLP** | 784→512→512→10 | 669,706 | 669,706 | **98.28%** | 0.1085 |
| **MMNN R=15** | fixWb=True (random features) fact=128 | 187,417 | 12,825 | **96.03%** | 0.1376 |
| **MMNN R=25** | fixWb=True (random features) fact=128 | 197,667 | 17,955 | **96.74%** | 0.1211 |
| **MMNN R=50** | fixWb=True (random features) fact=128 | 223,292 | 30,780 | **96.81%** | 0.1110 |

