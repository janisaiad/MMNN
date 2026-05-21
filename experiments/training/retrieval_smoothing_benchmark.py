#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_smoothing_benchmark.py

Toy benchmark for the retrieval-vs-smoothing tradeoff in a one-head Transformer-like
ResNet stack.

It compares:
  - softmax attention
  - softmax + entropy regularization
  - strict Sinkhorn attention
  - unbalanced Sinkhorn attention
  - residual vs non-residual dynamics
  - Q=K vs Q!=K
  - with / without MLP
  - varying d_model / d_head, temperature tau, Sinkhorn strength eta

Core measured observables layer by layer:
  R_l    = query-row attention mass on true spike token
  S_l    = intra-token variance
  C_l    = column imbalance ||A^T 1 - 1||^2 / n_tokens
  gap_l  = 1 - |lambda_2(A)| using eigenvalues of A
  H_l    = mean attention entropy

The benchmark is deliberately synthetic: each sequence contains one query token and
n_mem memory tokens. One memory token is the true spike. The query is generated from
that spike, optionally through a nonlinear transform; the model must learn to put
attention mass on the spike after L layers.

Example quick run:
  python retrieval_smoothing_benchmark.py --mode single --norm softmax --d-model 32 --tau 0.3 --steps 500

Example sweep:
  python retrieval_smoothing_benchmark.py --mode sweep --steps 300 --outdir ./runs/retrieval_sweep

Dependencies:
  torch, numpy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

NormType = Literal["softmax", "softmax_entropy", "sinkhorn", "unbalanced_sinkhorn"]
DataKind = Literal["linear", "quadratic"]


# -----------------------------
# Configs
# -----------------------------

@dataclasses.dataclass
class DataConfig:
    n_mem: int = 64              # number of memory tokens; total tokens = n_mem + 1 query
    d_star: int = 32             # intrinsic dimension of spike-generating latent
    d_model: int = 32            # observed/model dimension; for one head, d_head = d_model
    noise: float = 0.20          # noise in the query latent
    obs_noise: float = 0.03      # noise in observed token embeddings
    kind: DataKind = "linear"    # linear or quadratic spike relation
    device: str = "cpu"
    seed: int = 0


@dataclasses.dataclass
class ModelConfig:
    d_model: int = 32
    d_head: Optional[int] = None
    n_tokens: int = 65
    n_layers: int = 4
    tau: float = 0.3
    norm: NormType = "softmax"
    sinkhorn_iters: int = 12
    unbalanced_eta: float = 1.0  # eta -> infinity approximates strict Sinkhorn; eta -> 0 approximates softmax
    q_equals_k: bool = False
    use_mlp: bool = True
    residual: bool = True
    use_layernorm: bool = True
    mask_self: bool = True
    step_size: float = 0.5


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 128
    steps: int = 500
    lr: float = 2e-3
    weight_decay: float = 1e-4
    entropy_coef: float = 0.03   # only used for softmax_entropy: positive encourages smoother attention
    log_every: int = 100
    seed: int = 0
    device: str = "cpu"


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


# -----------------------------
# Data generator
# -----------------------------

class SpikeBatcher:
    """Generates one-query + n_mem-memory sequences with one true spike token.

    Total tokens: index 0 = query, indices 1..n_mem = memory tokens.
    Target index in the sequence = j_star + 1.

    Linear mode:
        query latent q = z_spike + noise.
        Linear QK can solve if d_model/d_head preserve the geometry.

    Quadratic mode:
        query latent q = psi(z_spike) + noise, psi(z)=z^2-1.
        Memory input is still z. A purely linear QK has difficulty; an MLP stack can
        create nonlinear features over layers.
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        # Fixed random observation map from intrinsic dimension to model dimension.
        # Scaling preserves distances approximately when d_model is large enough.
        R = torch.randn(cfg.d_star, cfg.d_model, device=self.device) / math.sqrt(max(1, cfg.d_model))
        self.R = R

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        B, n, ds, dm = batch_size, c.n_mem, c.d_star, c.d_model
        z_mem = torch.randn(B, n, ds, device=self.device)
        j_star = torch.randint(low=0, high=n, size=(B,), device=self.device)
        z_spike = z_mem[torch.arange(B, device=self.device), j_star]

        if c.kind == "linear":
            q_latent = z_spike + c.noise * torch.randn_like(z_spike)
            mem_latent_for_obs = z_mem
            query_latent_for_obs = q_latent
        elif c.kind == "quadratic":
            # Query lives in nonlinear feature coordinates; memory still observed in raw z.
            # The MLP has to create matching nonlinear features from memory tokens.
            q_latent = (z_spike.pow(2) - 1.0) + c.noise * torch.randn_like(z_spike)
            mem_latent_for_obs = z_mem
            query_latent_for_obs = q_latent
        else:
            raise ValueError(f"unknown data kind: {c.kind}")

        mem_obs = mem_latent_for_obs @ self.R
        query_obs = query_latent_for_obs @ self.R
        X = torch.cat([query_obs[:, None, :], mem_obs], dim=1)
        X = X + c.obs_noise * torch.randn_like(X)
        target_idx = j_star + 1
        return X, target_idx


# -----------------------------
# Attention normalizations
# -----------------------------

def apply_mask_self(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, N, N]
    B, N, _ = logits.shape
    eye = torch.eye(N, device=logits.device, dtype=torch.bool)[None, :, :]
    return logits.masked_fill(eye, -1e4)


def strict_sinkhorn_from_logits(
    logits: torch.Tensor,
    tau: float,
    n_iters: int = 12,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Doubly stochastic attention with row sums and column sums approximately 1.

    Starting kernel K = exp(logits/tau). Alternating row/column normalization.
    Output rows sum to 1, so it can be used as an attention matrix.
    """
    K = torch.exp((logits / max(tau, 1e-6)).clamp(min=-80.0, max=80.0))
    A = K + eps
    for _ in range(n_iters):
        A = A / (A.sum(dim=-1, keepdim=True) + eps)  # rows
        A = A / (A.sum(dim=-2, keepdim=True) + eps)  # cols
    # Final row normalization to remove tiny numerical drift.
    A = A / (A.sum(dim=-1, keepdim=True) + eps)
    return A


def unbalanced_sinkhorn_from_logits(
    logits: torch.Tensor,
    tau: float,
    eta: float = 1.0,
    n_iters: int = 12,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Unbalanced Sinkhorn-like attention.

    This interpolates between row-softmax and strict Sinkhorn.
    We use the standard unbalanced Sinkhorn exponent
        rho = eta / (eta + eps_ot)
    with eps_ot identified with tau.

    eta -> 0: weak marginal constraints, close to row-softmax after final row norm.
    eta -> infinity: strong marginal constraints, close to strict Sinkhorn.
    """
    eps_ot = max(tau, 1e-6)
    rho = float(eta / (eta + eps_ot)) if eta > 0 else 0.0
    K = torch.exp((logits / eps_ot).clamp(min=-80.0, max=80.0)) + eps
    B, N, _ = K.shape
    # Target marginals: row and col sums = 1 for an attention matrix.
    r = torch.ones(B, N, device=K.device)
    c = torch.ones(B, N, device=K.device)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(n_iters):
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + eps
        u = (r / Kv).pow(rho)
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + eps
        v = (c / KTu).pow(rho)
    A = u.unsqueeze(-1) * K * v.unsqueeze(1)
    A = A / (A.sum(dim=-1, keepdim=True) + eps)  # attention rows sum to one
    return A


def normalize_attention(
    logits: torch.Tensor,
    cfg: ModelConfig,
) -> torch.Tensor:
    if cfg.mask_self:
        logits = apply_mask_self(logits)
    if cfg.norm in ("softmax", "softmax_entropy"):
        return F.softmax(logits / max(cfg.tau, 1e-6), dim=-1)
    if cfg.norm == "sinkhorn":
        return strict_sinkhorn_from_logits(logits, cfg.tau, cfg.sinkhorn_iters)
    if cfg.norm == "unbalanced_sinkhorn":
        return unbalanced_sinkhorn_from_logits(
            logits, cfg.tau, eta=cfg.unbalanced_eta, n_iters=cfg.sinkhorn_iters
        )
    raise ValueError(f"unknown norm: {cfg.norm}")


# -----------------------------
# Model
# -----------------------------

class OneHeadLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        dh = cfg.d_head or cfg.d_model
        self.q = nn.Linear(d, dh, bias=False)
        if cfg.q_equals_k:
            self.k = self.q
        else:
            self.k = nn.Linear(d, dh, bias=False)
        self.v = nn.Linear(d, dh, bias=False)
        self.o = nn.Linear(dh, d, bias=False)
        self.ln1 = nn.LayerNorm(d) if cfg.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(d) if cfg.use_layernorm else nn.Identity()
        if cfg.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.ReLU(),
                nn.Linear(4 * d, d),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # X: [B, N, d]
        Xn = self.ln1(X)
        Q = self.q(Xn)
        K = self.k(Xn)
        V = self.v(Xn)
        logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
        A = normalize_attention(logits, self.cfg)
        attn_update = self.o(torch.bmm(A, V))
        if self.cfg.residual:
            Y = X + self.cfg.step_size * attn_update
        else:
            Y = self.cfg.step_size * attn_update

        if self.cfg.use_mlp:
            Yn = self.ln2(Y)
            mlp_update = self.mlp(Yn)
            if self.cfg.residual:
                Y = Y + self.cfg.step_size * mlp_update
            else:
                Y = Y + self.cfg.step_size * mlp_update
        return Y, A, logits


class OneHeadStack(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([OneHeadLayer(cfg) for _ in range(cfg.n_layers)])

    def forward(self, X: torch.Tensor, return_all: bool = True):
        X = X + self.pos
        As, logits_list, Xs = [], [], [X]
        for layer in self.layers:
            X, A, logits = layer(X)
            As.append(A)
            logits_list.append(logits)
            Xs.append(X)
        if return_all:
            return X, As, logits_list, Xs
        return X, As[-1]


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def attention_metrics(A: torch.Tensor, X: torch.Tensor, target_idx: torch.Tensor) -> Dict[str, float]:
    """Metrics for one layer.

    A: [B, N, N], X: [B, N, d], target_idx: [B] in [1,N-1]
    """
    B, N, _ = A.shape
    device = A.device
    row0 = A[:, 0, :]
    R = row0[torch.arange(B, device=device), target_idx].mean().item()

    Xc = X - X.mean(dim=1, keepdim=True)
    S = (Xc.pow(2).sum(dim=-1).mean() / X.shape[-1]).item()

    colsum = A.sum(dim=1)  # [B, N], target is one because rows sum to one and N rows
    C = ((colsum - 1.0).pow(2).mean()).item()

    H = (-(A.clamp_min(1e-12) * safe_log(A)).sum(dim=-1).mean()).item()
    H_row0 = (-(row0.clamp_min(1e-12) * safe_log(row0)).sum(dim=-1).mean()).item()

    # Spectral gap of row-stochastic matrix; approximate with eigenvalues for small/medium N.
    # For speed, use first min(B, 16) matrices.
    B_eval = min(B, 16)
    gaps = []
    for b in range(B_eval):
        vals = torch.linalg.eigvals(A[b]).abs().real
        vals_sorted = torch.sort(vals, descending=True).values
        if len(vals_sorted) >= 2:
            gaps.append((1.0 - vals_sorted[1]).item())
    gap = float(np.mean(gaps)) if gaps else float("nan")

    spike_col_load = colsum[torch.arange(B, device=device), target_idx].mean().item()
    max_col_load = colsum.max(dim=-1).values.mean().item()

    return {
        "R_spike": R,
        "S_token_var": S,
        "C_col_imbalance": C,
        "gap": gap,
        "H_attn": H,
        "H_query": H_row0,
        "spike_col_load": spike_col_load,
        "max_col_load": max_col_load,
    }


@torch.no_grad()
def evaluate_layers(model: OneHeadStack, batcher: SpikeBatcher, batch_size: int) -> pd.DataFrame:
    model.eval()
    X, target_idx = batcher.sample(batch_size)
    _, As, _, Xs = model(X, return_all=True)
    rows = []
    # Metrics after each layer use Xs[l+1].
    for ell, A in enumerate(As, start=1):
        row = {"layer": ell}
        row.update(attention_metrics(A, Xs[ell], target_idx))
        rows.append(row)
    return pd.DataFrame(rows)


def retrieval_ce_loss(A_final: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    row0 = A_final[:, 0, :]
    prob = row0[torch.arange(row0.shape[0], device=row0.device), target_idx].clamp_min(1e-12)
    return -torch.log(prob).mean()


def row0_entropy(A_final: torch.Tensor) -> torch.Tensor:
    row0 = A_final[:, 0, :]
    return (-(row0.clamp_min(1e-12) * safe_log(row0)).sum(dim=-1)).mean()


# -----------------------------
# Training / experiments
# -----------------------------

def train_one(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    verbose: bool = True,
) -> Tuple[OneHeadStack, pd.DataFrame, pd.DataFrame]:
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    data_cfg = dataclasses.replace(data_cfg, device=train_cfg.device, d_model=model_cfg.d_model)
    batcher = SpikeBatcher(data_cfg)
    model_cfg = dataclasses.replace(model_cfg, n_tokens=data_cfg.n_mem + 1)
    model = OneHeadStack(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    logs = []
    for step in range(1, train_cfg.steps + 1):
        model.train()
        X, target_idx = batcher.sample(train_cfg.batch_size)
        _, As, _, _ = model(X, return_all=True)
        A_final = As[-1]
        loss = retrieval_ce_loss(A_final, target_idx)
        if model_cfg.norm == "softmax_entropy" and train_cfg.entropy_coef != 0.0:
            # Positive entropy_coef encourages smoother query-row attention.
            loss = loss - train_cfg.entropy_coef * row0_entropy(A_final)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % train_cfg.log_every == 0 or step == 1 or step == train_cfg.steps:
            df_layers = evaluate_layers(model, batcher, train_cfg.batch_size)
            final = df_layers.iloc[-1].to_dict()
            record = {"step": step, "loss": float(loss.detach().cpu())}
            record.update({k: float(v) for k, v in final.items() if k != "layer"})
            logs.append(record)
            if verbose:
                print(
                    f"step={step:5d} loss={record['loss']:.4f} "
                    f"R={record['R_spike']:.3f} Hq={record['H_query']:.2f} "
                    f"C={record['C_col_imbalance']:.3f} gap={record['gap']:.3f}"
                )
    train_log = pd.DataFrame(logs)
    layer_metrics = evaluate_layers(model, batcher, train_cfg.batch_size)
    return model, train_log, layer_metrics


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    os.makedirs(args.outdir, exist_ok=True)
    base_train = TrainConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        log_every=max(1, args.steps // 3),
        seed=args.seed,
        device=args.device,
        entropy_coef=args.entropy_coef,
    )
    results = []
    d_models = [int(x) for x in args.d_grid.split(",")]
    taus = [float(x) for x in args.tau_grid.split(",")]
    norms = [x.strip() for x in args.norm_grid.split(",")]
    etas = [float(x) for x in args.eta_grid.split(",")]

    run_id = 0
    for norm in norms:
        for d_model in d_models:
            for tau in taus:
                eta_values = etas if norm == "unbalanced_sinkhorn" else [args.unbalanced_eta]
                for eta in eta_values:
                    for seed_offset in range(args.seeds):
                        seed = args.seed + 1000 * run_id + seed_offset
                        print("\n=== RUN", run_id, "norm", norm, "d", d_model, "tau", tau, "eta", eta, "seed", seed, "===")
                        data_cfg = DataConfig(
                            n_mem=args.n_mem,
                            d_star=args.d_star,
                            d_model=d_model,
                            noise=args.noise,
                            obs_noise=args.obs_noise,
                            kind=args.data_kind,
                            seed=seed,
                            device=args.device,
                        )
                        model_cfg = ModelConfig(
                            d_model=d_model,
                            d_head=d_model,
                            n_tokens=args.n_mem + 1,
                            n_layers=args.layers,
                            tau=tau,
                            norm=norm,  # type: ignore
                            sinkhorn_iters=args.sinkhorn_iters,
                            unbalanced_eta=eta,
                            q_equals_k=args.q_equals_k,
                            use_mlp=not args.no_mlp,
                            residual=not args.no_residual,
                            use_layernorm=not args.no_layernorm,
                            mask_self=not args.no_mask_self,
                            step_size=args.step_size,
                        )
                        train_cfg = dataclasses.replace(base_train, seed=seed)
                        _, log_df, layer_df = train_one(data_cfg, model_cfg, train_cfg, verbose=False)
                        final = layer_df.iloc[-1].to_dict()
                        rec = {
                            "run_id": run_id,
                            "seed": seed,
                            "norm": norm,
                            "d_model": d_model,
                            "tau": tau,
                            "eta": eta,
                            "q_equals_k": args.q_equals_k,
                            "use_mlp": not args.no_mlp,
                            "residual": not args.no_residual,
                            "data_kind": args.data_kind,
                            "steps": args.steps,
                        }
                        rec.update({k: float(v) for k, v in final.items() if k != "layer"})
                        results.append(rec)
                        # save detailed layer metrics for this run
                        layer_df.assign(**rec).to_csv(os.path.join(args.outdir, f"layers_run_{run_id}.csv"), index=False)
                    run_id += 1

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.outdir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved", csv_path)
    make_plots(df, args.outdir)
    return df


def make_plots(df: pd.DataFrame, outdir: str) -> None:
    if plt is None or df.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    # Aggregate over seeds.
    agg = df.groupby(["norm", "d_model", "tau", "eta"], as_index=False).agg(
        R_spike=("R_spike", "mean"),
        C_col_imbalance=("C_col_imbalance", "mean"),
        H_query=("H_query", "mean"),
        gap=("gap", "mean"),
        S_token_var=("S_token_var", "mean"),
    )
    for norm in agg["norm"].unique():
        sub = agg[agg["norm"] == norm]
        # For unbalanced, plot each eta separately.
        for eta in sorted(sub["eta"].unique()):
            sube = sub[sub["eta"] == eta]
            pivot = sube.pivot(index="d_model", columns="tau", values="R_spike")
            if pivot.empty:
                continue
            plt.figure(figsize=(7, 5))
            plt.imshow(pivot.values, aspect="auto", origin="lower")
            plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
            plt.colorbar(label="final R_spike")
            plt.xlabel("temperature tau")
            plt.ylabel("d_model = d_head")
            title = f"R_spike heatmap: {norm}"
            if norm == "unbalanced_sinkhorn":
                title += f" eta={eta}"
            plt.title(title)
            plt.tight_layout()
            fname = f"heatmap_R_{norm}_eta_{eta}.png".replace("/", "_")
            plt.savefig(os.path.join(outdir, fname), dpi=160)
            plt.close()

    # Scatter: retrieval vs column imbalance.
    plt.figure(figsize=(7, 5))
    for norm in agg["norm"].unique():
        sub = agg[agg["norm"] == norm]
        plt.scatter(sub["C_col_imbalance"], sub["R_spike"], label=norm, alpha=0.8)
    plt.xlabel("column imbalance C")
    plt.ylabel("final R_spike")
    plt.title("Retrieval vs column imbalance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_R_vs_C.png"), dpi=160)
    plt.close()


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval-vs-smoothing toy benchmark")
    p.add_argument("--mode", choices=["single", "sweep"], default="single")
    p.add_argument("--outdir", type=str, default="./retrieval_smoothing_runs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # data
    p.add_argument("--n-mem", type=int, default=64)
    p.add_argument("--d-star", type=int, default=32)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--data-kind", choices=["linear", "quadratic"], default="linear")
    p.add_argument("--noise", type=float, default=0.20)
    p.add_argument("--obs-noise", type=float, default=0.03)

    # model
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--norm", choices=["softmax", "softmax_entropy", "sinkhorn", "unbalanced_sinkhorn"], default="softmax")
    p.add_argument("--sinkhorn-iters", type=int, default=12)
    p.add_argument("--unbalanced-eta", type=float, default=1.0)
    p.add_argument("--q-equals-k", action="store_true")
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--no-layernorm", action="store_true")
    p.add_argument("--no-mask-self", action="store_true")
    p.add_argument("--step-size", type=float, default=0.5)

    # training
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--entropy-coef", type=float, default=0.03)

    # sweep
    p.add_argument("--d-grid", type=str, default="4,8,16,32,64")
    p.add_argument("--tau-grid", type=str, default="0.1,0.2,0.4,0.8")
    p.add_argument("--eta-grid", type=str, default="0.05,0.2,1.0,5.0")
    p.add_argument("--norm-grid", type=str, default="softmax,sinkhorn,unbalanced_sinkhorn")
    p.add_argument("--seeds", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.mode == "single":
        set_seed(args.seed)
        data_cfg = DataConfig(
            n_mem=args.n_mem,
            d_star=args.d_star,
            d_model=args.d_model,
            noise=args.noise,
            obs_noise=args.obs_noise,
            kind=args.data_kind,
            seed=args.seed,
            device=args.device,
        )
        model_cfg = ModelConfig(
            d_model=args.d_model,
            d_head=args.d_model,
            n_tokens=args.n_mem + 1,
            n_layers=args.layers,
            tau=args.tau,
            norm=args.norm,  # type: ignore
            sinkhorn_iters=args.sinkhorn_iters,
            unbalanced_eta=args.unbalanced_eta,
            q_equals_k=args.q_equals_k,
            use_mlp=not args.no_mlp,
            residual=not args.no_residual,
            use_layernorm=not args.no_layernorm,
            mask_self=not args.no_mask_self,
            step_size=args.step_size,
        )
        train_cfg = TrainConfig(
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            device=args.device,
            log_every=max(1, args.steps // 5),
        )
        model, log_df, layer_df = train_one(data_cfg, model_cfg, train_cfg, verbose=True)
        log_path = os.path.join(args.outdir, "single_train_log.csv")
        layer_path = os.path.join(args.outdir, "single_layer_metrics.csv")
        log_df.to_csv(log_path, index=False)
        layer_df.to_csv(layer_path, index=False)
        print("\nFinal layer metrics:")
        print(layer_df.to_string(index=False))
        print("Saved", log_path)
        print("Saved", layer_path)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
