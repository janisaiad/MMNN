Done: [download the CUDA PyTorch script](sandbox:/mnt/data/witness_slot_retrieval_experiments.py).

It implements the synthetic **Option A first** benchmark:

[
P_T={(x_t,y_t)}*{t=1}^{T},
\qquad
\beta\in\mathbb R^K,
\qquad
y_t=M*{r_t}\beta+\xi_t,
]

with a one-head (K)-slot model:

[
h_t=E_\alpha(x_t,y_t),
]

[
k_t=W_Kh_t,
\qquad
v_t=W_Vh_t,
]

[
A=\operatorname{softmax}_t(CK_T^\top)\in\mathbb R^{K\times T},
]

[
O=AV_T,
]

[
\hat\beta=R_\theta(O),
]

[
\hat y_\star=\phi(x_\star)^\top\hat\beta.
]

So this tests exactly:

[
\boxed{
\text{encoder separability}
\Rightarrow
\text{slot-token retrieval}
\Rightarrow
\text{coefficient recovery}
\Rightarrow
\text{query prediction}.
}
]

---

### Quick run

```bash
python witness_slot_retrieval_experiments.py \
  --mode single \
  --K 8 \
  --d-h 8 \
  --d-model 128 \
  --train-tasks 4096 \
  --train-prompt-len 128 \
  --test-prompt-len 128 \
  --steps 2000 \
  --batch-size 128 \
  --device cuda \
  --outdir runs_slot_A
```

---

### Scaling in (d_h) versus (K)

This checks the rank/separability threshold:

```bash
python witness_slot_retrieval_experiments.py \
  --mode sweep \
  --sweep-ks 4,8,16,32 \
  --sweep-dh 2,4,8,16,32,64 \
  --train-tasks 8192 \
  --train-prompt-len 256 \
  --test-prompt-len 256 \
  --steps 2000 \
  --batch-size 128 \
  --device cuda \
  --outdir runs_dh_vs_K
```

You want to see a transition around:

[
d_h\approx K
]

or more generally:

[
d_h\approx r_{\rm eff}.
]

The script logs:

[
R_{\rm correct}
===============

\text{mass slot }k\text{ puts on witness class }k,
]

[
\gamma
======

\text{slot-token class margin},
]

[
\operatorname{rank}(CK_T^\top),
]

[
\operatorname{effrank}(CK_T^\top),
]

[
\text{test MSE},
\qquad
\text{beta MSE}.
]

---

### Dataset-size scaling (N)

Finite pretraining dataset size:

```bash
python witness_slot_retrieval_experiments.py \
  --mode sweep \
  --sweep-ks 8 \
  --sweep-dh 4,8,16 \
  --sweep-train-tasks 128,512,2048,8192,32768 \
  --train-prompt-len 256 \
  --test-prompt-len 256 \
  --steps 2500 \
  --batch-size 128 \
  --device cuda \
  --outdir runs_N_scaling
```

This tests the analogue of the (N)-task generalization scaling.

---

### Test prompt-size scaling (m)

Train with large prompts, evaluate on many test prompt lengths:

```bash
python witness_slot_retrieval_experiments.py \
  --mode single \
  --K 8 \
  --d-h 8 \
  --train-tasks 16384 \
  --train-prompt-len 256 \
  --eval-prompt-grid 8,16,32,64,128,256,512 \
  --steps 3000 \
  --batch-size 128 \
  --device cuda \
  --outdir runs_test_prompt_scaling
```

This tests:

[
m\uparrow
\Rightarrow
\text{better witness coverage}
\Rightarrow
R_{\rm correct}\uparrow,
\quad
\text{MSE}\downarrow.
]

---

### Train prompt-size scaling (n)

Very large dataset, vary the prompt length used in training:

```bash
python witness_slot_retrieval_experiments.py \
  --mode sweep \
  --K 8 \
  --d-h 8 \
  --sweep-train-prompts 8,16,32,64,128,256 \
  --train-tasks 32768 \
  --test-prompt-len 256 \
  --steps 2500 \
  --batch-size 128 \
  --device cuda \
  --outdir runs_train_prompt_scaling
```

---

### Harder inverse problem: mixed witnesses

Identity mixing is:

[
y_t=\beta_{r_t}+\xi_t.
]

Harder mode:

[
y_t=M_{r_t}\beta+\xi_t,
]

so the readout must learn (M^{-1}):

```bash
python witness_slot_retrieval_experiments.py \
  --mode sweep \
  --mixing random_wellcond \
  --mixing-cond 5.0 \
  --sweep-ks 4,8,16 \
  --sweep-dh 4,8,16,32 \
  --train-tasks 16384 \
  --train-prompt-len 256 \
  --steps 3000 \
  --device cuda \
  --outdir runs_mixed_witnesses
```

---

### Option B is included but not default

The script also has a simple Gram/self-attention pre-layer:

```bash
--model option_b
```

but your requested **slot-token matrix first** setup is the default:

```bash
--model option_a
```

So the first experiments should use `option_a`.
