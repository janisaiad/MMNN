#!/usr/bin/env python3
"""
Witness-slot retrieval experiments.

Goal
----
Synthetic non-PDE benchmark for the architecture:
    one attention head + K learned query slots + value readout.

The task is coefficient retrieval. Each task has latent beta in R^K. A prompt
contains T witness tokens (x_t, y_t). Each token belongs to a hidden witness
class r_t in {1,...,K}; for identity mixing, y_t = beta_{r_t} + noise. More
generally, y_t = M[r_t] @ beta + noise for a fixed invertible mixing matrix M.
A successful model must retrieve one statistic per slot and reconstruct beta.

Main knobs:
    K                  number of coefficients / witness slots
    d_h                head/key/query dimension
    T train/test       prompt length
    N train_tasks      finite pretraining dataset size
    model              option_a = slots directly attend to tokens
                       option_b = one Gram/self-attn token layer, then slots

Metrics:
    test_mse           query prediction error
    beta_mse           coefficient error
    R_correct_mass     attention mass slot r puts on witness class r
    margin             slot-token logit class margin
    rank_logits        numerical rank of slot-token logits
    effrank_logits     entropy effective rank of slot-token logits
    slot_entropy       attention entropy
    slot_overlap       ||A A^T - diag|| off-diagonal overlap

Examples
--------
Single run:
    python witness_slot_retrieval_experiments.py --mode single --K 8 --d-h 8 --steps 2000

Sweep d_h vs K:
    python witness_slot_retrieval_experiments.py --mode sweep \
      --sweep-ks 4,8,16 --sweep-dh 2,4,8,16,32 \
      --train-tasks 4096 --steps 1500 --outdir runs_slot_A

Finite dataset-size scaling:
    python witness_slot_retrieval_experiments.py --mode sweep \
      --sweep-ks 8 --sweep-dh 4,8,16 --sweep-train-tasks 128,512,2048,8192 \
      --steps 2000 --outdir runs_N_scaling

Prompt-size test scaling after training on large prompts:
    python witness_slot_retrieval_experiments.py --mode single --K 8 --d-h 8 \
      --train-prompt-len 256 --eval-prompt-grid 8,16,32,64,128,256 \
      --train-tasks 8192 --steps 3000 --outdir runs_prompt_test

Train-prompt scaling with large dataset:
    python witness_slot_retrieval_experiments.py --mode sweep \
      --sweep-train-prompts 8,16,32,64,128 --train-tasks 16384 \
      --K 8 --d-h 8 --steps 2000 --outdir runs_train_prompt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ----------------------------- utilities -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


@dataclass
class Config:
    # task/data
    K: int = 8
    d_star: int = 8
    raw_dim: int = 0  # if 0, raw_dim=max(K,d_star)
    train_prompt_len: int = 128
    test_prompt_len: int = 128
    n_queries: int = 8
    train_tasks: int = 4096  # 0 = online infinite tasks
    test_tasks: int = 1024
    ensure_coverage: bool = True
    x_noise: float = 0.35
    y_noise: float = 0.05
    beta_scale: float = 1.0
    mixing: str = "identity"  # identity | random_orthogonal | random_wellcond
    mixing_cond: float = 3.0

    # model
    model: str = "option_a"  # option_a | option_b
    d_model: int = 128
    d_h: int = 8
    d_v: int = 32
    encoder_depth: int = 2
    mlp_hidden: int = 128
    normalize_qk: bool = False
    tau: float = 0.35
    q_equals_k: bool = False  # for option_b token Gram layer only
    residual_gram: bool = True
    gram_strength: float = 1.0

    # training
    steps: int = 2000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    beta_aux: float = 0.0
    slot_div_reg: float = 0.0
    log_every: int = 100
    eval_every: int = 500
    grad_clip: float = 1.0

    # runtime/output
    seed: int = 0
    device: str = "cuda"
    outdir: str = "runs_witness_slot"
    run_name: str = "run"
    eval_prompt_grid: str = ""  # comma list, optional


# ----------------------------- data -----------------------------


class WitnessTaskGenerator:
    """Generates finite or online datasets for coefficient witness retrieval."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        K = cfg.K
        d_star = cfg.d_star
        raw_dim = cfg.raw_dim if cfg.raw_dim > 0 else max(K, d_star)
        self.raw_dim = raw_dim

        # Prototypes in raw_dim. If raw_dim >= K, use orthogonal-like first K axes.
        # Else use random normalized prototypes, deliberately making separation harder.
        if raw_dim >= K:
            proto = torch.zeros(K, raw_dim, device=device)
            proto[:, :K] = torch.eye(K, device=device)
            # Add a small random rotation/noise so the task is not totally trivial.
            proto = proto + 0.05 * torch.randn_like(proto)
            proto = F.normalize(proto, dim=-1)
        else:
            proto = torch.randn(K, raw_dim, device=device)
            proto = F.normalize(proto, dim=-1)
        self.prototypes = proto

        self.M = self._make_mixing(K, cfg.mixing, cfg.mixing_cond, device)
        self.M_inv = torch.linalg.pinv(self.M)

        self.train_cache = None
        if cfg.train_tasks > 0:
            self.train_cache = self.sample_tasks(cfg.train_tasks, cfg.train_prompt_len, cfg.n_queries)

        self.test_cache = self.sample_tasks(cfg.test_tasks, cfg.test_prompt_len, cfg.n_queries)

    @staticmethod
    def _make_mixing(K: int, mixing: str, cond: float, device: torch.device) -> torch.Tensor:
        if mixing == "identity":
            return torch.eye(K, device=device)
        A = torch.randn(K, K, device=device)
        U, _, Vh = torch.linalg.svd(A, full_matrices=True)
        if mixing == "random_orthogonal":
            return U @ Vh
        if mixing == "random_wellcond":
            # Singular values from 1 to cond.
            s = torch.linspace(1.0, cond, K, device=device)
            return U @ torch.diag(s) @ Vh
        raise ValueError(f"unknown mixing={mixing}")

    def _sample_labels(self, B: int, T: int) -> torch.Tensor:
        K = self.cfg.K
        if self.cfg.ensure_coverage and T >= K:
            first = torch.arange(K, device=self.device).repeat(B, 1)
            rest = torch.randint(0, K, (B, T - K), device=self.device)
            labels = torch.cat([first, rest], dim=1)
            # Shuffle each row.
            noise = torch.rand(B, T, device=self.device)
            perm = noise.argsort(dim=1)
            labels = labels.gather(1, perm)
        else:
            labels = torch.randint(0, K, (B, T), device=self.device)
        return labels

    def sample_tasks(self, B: int, T: int, Q: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        labels = self._sample_labels(B, T)
        beta = cfg.beta_scale * torch.randn(B, cfg.K, device=self.device)

        proto = self.prototypes[labels]  # [B,T,raw_dim]
        x = proto + cfg.x_noise * torch.randn(B, T, self.raw_dim, device=self.device)

        # Witness functional observed by each class: z_r = M[r] beta.
        z = beta @ self.M.T  # [B,K]
        y = z.gather(1, labels) + cfg.y_noise * torch.randn(B, T, device=self.device)

        # Query basis values phi_star; target = phi_star^T beta.
        phi_q = torch.randn(B, Q, cfg.K, device=self.device)
        # normalize to make MSE scales comparable as K changes
        phi_q = phi_q / math.sqrt(cfg.K)
        y_q = torch.einsum('bqk,bk->bq', phi_q, beta)
        return {"x": x, "y": y, "labels": labels, "beta": beta, "phi_q": phi_q, "y_q": y_q}

    def batch(self, B: int, split: str = "train", T_override: Optional[int] = None) -> Dict[str, torch.Tensor]:
        Q = self.cfg.n_queries
        if split == "train" and self.train_cache is not None and T_override is None:
            N = self.cfg.train_tasks
            idx = torch.randint(0, N, (B,), device=self.device)
            return {k: v[idx] for k, v in self.train_cache.items()}
        if split == "test" and T_override is None:
            N = self.cfg.test_tasks
            idx = torch.randint(0, N, (B,), device=self.device)
            return {k: v[idx] for k, v in self.test_cache.items()}
        T = T_override if T_override is not None else (self.cfg.train_prompt_len if split == "train" else self.cfg.test_prompt_len)
        return self.sample_tasks(B, T, Q)


# ----------------------------- model -----------------------------


class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = inp
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WitnessSlotModel(nn.Module):
    """Option A: slots directly attend prompt tokens.
       Option B: one token-token Gram/self-attn layer before slots.
    """

    def __init__(self, cfg: Config, raw_dim: int):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K
        self.raw_dim = raw_dim
        token_in = raw_dim + 1
        self.encoder = MLP(token_in, cfg.d_model, cfg.mlp_hidden, cfg.encoder_depth)
        self.key = nn.Linear(cfg.d_model, cfg.d_h, bias=False)
        self.value = nn.Linear(cfg.d_model, cfg.d_v, bias=False)
        self.slots = nn.Parameter(torch.randn(cfg.K, cfg.d_h) / math.sqrt(cfg.d_h))

        # Optional Gram/self-attn layer for option_b.
        if cfg.model == "option_b":
            self.gram_key = nn.Linear(cfg.d_model, cfg.d_h, bias=False)
            if cfg.q_equals_k:
                self.gram_query = self.gram_key
            else:
                self.gram_query = nn.Linear(cfg.d_model, cfg.d_h, bias=False)
            self.gram_value = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.gram_out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.gram_ln = nn.LayerNorm(cfg.d_model)

        self.beta_net = nn.Sequential(
            nn.LayerNorm(cfg.K * cfg.d_v),
            nn.Linear(cfg.K * cfg.d_v, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.K),
        )

    def _maybe_norm(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.normalize_qk:
            return F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor, y: torch.Tensor, phi_q: torch.Tensor, return_debug: bool = False):
        # x: [B,T,raw_dim], y: [B,T], phi_q: [B,Q,K]
        B, T, _ = x.shape
        inp = torch.cat([x, y[..., None]], dim=-1)
        h = self.encoder(inp)

        debug = {}
        if self.cfg.model == "option_b":
            q0 = self._maybe_norm(self.gram_query(h))
            k0 = self._maybe_norm(self.gram_key(h))
            logits_tt = torch.einsum('btd,bsd->bts', q0, k0) / math.sqrt(self.cfg.d_h)
            A_tt = F.softmax(logits_tt / self.cfg.tau, dim=-1)
            msg = torch.bmm(A_tt, self.gram_value(h))
            msg = self.gram_out(msg)
            if self.cfg.residual_gram:
                h = self.gram_ln(h + self.cfg.gram_strength * msg)
            else:
                h = self.gram_ln(self.cfg.gram_strength * msg)
            debug["token_attn"] = A_tt
            debug["token_logits"] = logits_tt

        keys = self._maybe_norm(self.key(h))  # [B,T,d_h]
        values = self.value(h)  # [B,T,d_v]
        slots = self._maybe_norm(self.slots)  # [K,d_h]
        logits = torch.einsum('kd,btd->bkt', slots, keys) / math.sqrt(self.cfg.d_h)
        A = F.softmax(logits / self.cfg.tau, dim=-1)
        O = torch.bmm(A, values)  # [B,K,d_v]
        beta_hat = self.beta_net(O.reshape(B, -1))
        y_hat = torch.einsum('bqk,bk->bq', phi_q, beta_hat)

        if return_debug:
            debug.update({"attn": A, "logits": logits, "keys": keys, "values": values, "O": O})
            return y_hat, beta_hat, debug
        return y_hat, beta_hat


# ----------------------------- metrics -----------------------------


def compute_metrics(y_hat: torch.Tensor, beta_hat: torch.Tensor, batch: Dict[str, torch.Tensor], debug: Dict[str, torch.Tensor]) -> Dict[str, float]:
    with torch.no_grad():
        y_q = batch["y_q"]
        beta = batch["beta"]
        labels = batch["labels"]
        A = debug["attn"]  # [B,K,T]
        logits = debug["logits"]  # [B,K,T]
        B, K, T = A.shape

        out: Dict[str, float] = {}
        out["test_mse"] = F.mse_loss(y_hat, y_q).item()
        out["beta_mse"] = F.mse_loss(beta_hat, beta).item()

        # Attention mass slot r puts on witness class r.
        eye_labels = torch.arange(K, device=A.device)[None, :, None]
        correct_mask = (labels[:, None, :] == eye_labels).float()
        R_correct = (A * correct_mask).sum(dim=-1).mean()
        out["R_correct_mass"] = R_correct.item()

        # Class confusion matrix: rows slots, columns true classes.
        conf = []
        for c in range(K):
            mask = (labels == c).float()[:, None, :]
            conf.append((A * mask).sum(dim=-1).mean(dim=0))  # [K]
        conf_mat = torch.stack(conf, dim=1)  # [K,K]
        out["conf_diag"] = torch.diag(conf_mat).mean().item()
        off = conf_mat - torch.diag(torch.diag(conf_mat))
        out["conf_offdiag_mean"] = (off.sum() / max(K * (K - 1), 1)).item()

        # Class logit margins: mean logit on correct class minus best other class.
        class_logits = []
        for c in range(K):
            mask = (labels == c).float()[:, None, :]
            denom = mask.sum(dim=-1).clamp_min(1.0)
            class_logits.append((logits * mask).sum(dim=-1) / denom)  # [B,K]
        class_logits = torch.stack(class_logits, dim=2).mean(dim=0)  # [K slots, K classes]
        diag = torch.diag(class_logits)
        other = class_logits.masked_fill(torch.eye(K, device=A.device).bool(), -1e9).max(dim=1).values
        margin = diag - other
        out["margin_mean"] = margin.mean().item()
        out["margin_min"] = margin.min().item()

        # Attention entropy.
        entropy = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(dim=-1)
        out["slot_entropy"] = entropy.mean().item()

        # Slot overlap: off-diagonal mass of A A^T normalized.
        AA = torch.bmm(A, A.transpose(1, 2))  # [B,K,K]
        diag_AA = torch.diagonal(AA, dim1=1, dim2=2)
        off_AA = AA - torch.diag_embed(diag_AA)
        out["slot_overlap"] = off_AA.pow(2).mean().item()
        out["slot_self_mass"] = diag_AA.mean().item()

        # Column load imbalance: how many slots use same token.
        col_load = A.sum(dim=1)  # [B,T]
        # For K row-stochastic distributions, uniform token load is K/T.
        out["col_load_max"] = col_load.max(dim=1).values.mean().item()
        out["col_load_var"] = col_load.var(dim=1).mean().item()

        # Singular values/rank of slot-token logits.
        # [B,K,T], rank <= min(K,d_h,T). Average singular values after padding.
        sv = torch.linalg.svdvals(logits.float())  # [B,min(K,T)]
        smean = sv.mean(dim=0)
        out["sv1_logits"] = smean[0].item() if smean.numel() > 0 else 0.0
        out["svK_logits"] = smean[min(K, smean.numel()) - 1].item() if smean.numel() > 0 else 0.0
        thresh = 1e-3 * sv[:, :1].clamp_min(1e-12)
        out["rank_logits"] = (sv > thresh).float().sum(dim=1).mean().item()
        ps = sv / sv.sum(dim=1, keepdim=True).clamp_min(1e-12)
        effrank = torch.exp(-(ps.clamp_min(1e-12) * ps.clamp_min(1e-12).log()).sum(dim=1))
        out["effrank_logits"] = effrank.mean().item()

        if "token_logits" in debug:
            tl = debug["token_logits"].float()
            # Symmetric part spectral values for option_b.
            tlsym = 0.5 * (tl + tl.transpose(1, 2))
            eigs = torch.linalg.eigvalsh(tlsym)
            out["gram_top_eig"] = eigs[:, -1].mean().item()
            out["gram_second_eig"] = eigs[:, -2].mean().item() if eigs.shape[1] > 1 else 0.0
            out["gram_gap"] = (eigs[:, -1] - eigs[:, -2]).mean().item() if eigs.shape[1] > 1 else 0.0
        return out


# ----------------------------- train / eval -----------------------------


def loss_fn(y_hat, beta_hat, batch, cfg: Config, debug=None):
    loss = F.mse_loss(y_hat, batch["y_q"])
    if cfg.beta_aux > 0:
        loss = loss + cfg.beta_aux * F.mse_loss(beta_hat, batch["beta"])
    if cfg.slot_div_reg > 0 and debug is not None:
        A = debug["attn"]
        AA = torch.bmm(A, A.transpose(1, 2))
        diag = torch.diagonal(AA, dim1=1, dim2=2)
        off = AA - torch.diag_embed(diag)
        loss = loss + cfg.slot_div_reg * off.pow(2).mean()
    return loss


@torch.no_grad()
def evaluate(model: WitnessSlotModel, gen: WitnessTaskGenerator, cfg: Config, T: Optional[int] = None, batches: int = 8) -> Dict[str, float]:
    model.eval()
    acc: Dict[str, List[float]] = {}
    for _ in range(batches):
        batch = gen.batch(cfg.batch_size, split="test", T_override=T)
        y_hat, beta_hat, debug = model(batch["x"], batch["y"], batch["phi_q"], return_debug=True)
        metrics = compute_metrics(y_hat, beta_hat, batch, debug)
        for k, v in metrics.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def train_single(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    set_seed(cfg.seed)
    gen = WitnessTaskGenerator(cfg, device)
    model = WitnessSlotModel(cfg, gen.raw_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rows = []
    best = {"test_mse": float("inf")}

    for step in range(1, cfg.steps + 1):
        model.train()
        batch = gen.batch(cfg.batch_size, split="train")
        y_hat, beta_hat, debug = model(batch["x"], batch["y"], batch["phi_q"], return_debug=True)
        loss = loss_fn(y_hat, beta_hat, batch, cfg, debug)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            train_metrics = compute_metrics(y_hat.detach(), beta_hat.detach(), batch, debug)
            row = {"step": step, "split": "train", "loss": loss.item(), **train_metrics}
            rows.append(row)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            test_metrics = evaluate(model, gen, cfg, T=cfg.test_prompt_len, batches=8)
            row = {"step": step, "split": "test", "loss": np.nan, **test_metrics}
            rows.append(row)
            if test_metrics["test_mse"] < best.get("test_mse", float("inf")):
                best = {**test_metrics, "step": step}

    # Optional prompt test grid, after training.
    if cfg.eval_prompt_grid:
        for T in parse_int_list(cfg.eval_prompt_grid):
            m = evaluate(model, gen, cfg, T=T, batches=8)
            rows.append({"step": cfg.steps, "split": f"test_T={T}", "loss": np.nan, **m, "eval_T": T})

    df = pd.DataFrame(rows)
    final = {**best, "K": cfg.K, "d_h": cfg.d_h, "train_tasks": cfg.train_tasks,
             "train_prompt_len": cfg.train_prompt_len, "test_prompt_len": cfg.test_prompt_len,
             "model": cfg.model, "seed": cfg.seed, "mixing": cfg.mixing}
    return df, final


# ----------------------------- sweeps / plotting -----------------------------


def plot_heatmap(df: pd.DataFrame, outdir: Path, value: str = "test_mse") -> None:
    if plt is None or df.empty:
        return
    # Use final rows from sweep summary.
    if not {"K", "d_h", value}.issubset(df.columns):
        return
    for K, sub in df.groupby("K"):
        piv = sub.pivot_table(index="d_h", columns="train_tasks", values=value, aggfunc="mean")
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(np.log10(piv.values + 1e-12), aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([str(x) for x in piv.columns], rotation=45)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([str(x) for x in piv.index])
        ax.set_xlabel("train_tasks N")
        ax.set_ylabel("d_h")
        ax.set_title(f"log10({value}) for K={K}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / f"heatmap_{value}_K{K}.png", dpi=160)
        plt.close(fig)


def plot_scaling(df: pd.DataFrame, outdir: Path) -> None:
    if plt is None or df.empty:
        return
    if "test_mse" not in df.columns:
        return
    # MSE vs d_h/K ratio.
    if {"K", "d_h"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        temp = df.copy()
        temp["dh_over_K"] = temp["d_h"] / temp["K"]
        ax.scatter(temp["dh_over_K"], temp["test_mse"], c=temp["K"], s=45)
        ax.set_yscale("log")
        ax.set_xlabel("d_h / K")
        ax.set_ylabel("test MSE")
        ax.set_title("Capacity threshold: d_h / K")
        fig.tight_layout()
        fig.savefig(outdir / "scatter_mse_vs_dh_over_K.png", dpi=160)
        plt.close(fig)
    # MSE vs N loglog.
    if "train_tasks" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (K, dh), sub in df.groupby(["K", "d_h"]):
            sub = sub.sort_values("train_tasks")
            if sub["train_tasks"].nunique() > 1:
                ax.plot(sub["train_tasks"], sub["test_mse"], marker="o", label=f"K={K},dh={dh}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("train_tasks N")
        ax.set_ylabel("test MSE")
        ax.set_title("Dataset-size scaling")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / "scaling_mse_vs_N.png", dpi=160)
        plt.close(fig)


def run_sweep(args) -> None:
    base = Config(**vars(args_to_config_namespace(args)))
    outdir = ensure_dir(base.outdir)
    summary_rows = []

    Ks = parse_int_list(args.sweep_ks) if args.sweep_ks else [base.K]
    dhs = parse_int_list(args.sweep_dh) if args.sweep_dh else [base.d_h]
    Ns = parse_int_list(args.sweep_train_tasks) if args.sweep_train_tasks else [base.train_tasks]
    train_prompts = parse_int_list(args.sweep_train_prompts) if args.sweep_train_prompts else [base.train_prompt_len]
    seeds = parse_int_list(args.sweep_seeds) if args.sweep_seeds else [base.seed]

    total = len(Ks) * len(dhs) * len(Ns) * len(train_prompts) * len(seeds)
    idx = 0
    for K in Ks:
        for dh in dhs:
            for N in Ns:
                for Ttr in train_prompts:
                    for seed in seeds:
                        idx += 1
                        cfg = Config(**asdict(base))
                        cfg.K = K
                        cfg.d_star = max(cfg.d_star, K) if args.auto_dstar else cfg.d_star
                        cfg.raw_dim = max(cfg.raw_dim, K) if (cfg.raw_dim > 0 and args.auto_rawdim) else cfg.raw_dim
                        cfg.d_h = dh
                        cfg.train_tasks = N
                        cfg.train_prompt_len = Ttr
                        cfg.seed = seed
                        cfg.run_name = f"K{K}_dh{dh}_N{N}_Ttr{Ttr}_seed{seed}_{cfg.model}"
                        print(f"[{idx}/{total}] running {cfg.run_name}", flush=True)
                        hist, final = train_single(cfg)
                        hist.to_csv(outdir / f"history_{cfg.run_name}.csv", index=False)
                        summary_rows.append(final)
                        pd.DataFrame(summary_rows).to_csv(outdir / "sweep_summary_partial.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "sweep_summary.csv", index=False)
    plot_heatmap(summary, outdir, "test_mse")
    plot_heatmap(summary, outdir, "R_correct_mass")
    plot_scaling(summary, outdir)
    print(f"Saved sweep to {outdir}")


# argparse helper: only fields in Config
class Obj:
    pass


def args_to_config_namespace(args):
    o = Obj()
    cfg_fields = set(Config.__dataclass_fields__.keys())
    for k, v in vars(args).items():
        if k in cfg_fields:
            setattr(o, k, v)
    return o


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["single", "sweep"], default="single")

    # Config fields
    p.add_argument("--K", type=int, default=Config.K)
    p.add_argument("--d-star", dest="d_star", type=int, default=Config.d_star)
    p.add_argument("--raw-dim", dest="raw_dim", type=int, default=Config.raw_dim)
    p.add_argument("--train-prompt-len", dest="train_prompt_len", type=int, default=Config.train_prompt_len)
    p.add_argument("--test-prompt-len", dest="test_prompt_len", type=int, default=Config.test_prompt_len)
    p.add_argument("--n-queries", dest="n_queries", type=int, default=Config.n_queries)
    p.add_argument("--train-tasks", dest="train_tasks", type=int, default=Config.train_tasks)
    p.add_argument("--test-tasks", dest="test_tasks", type=int, default=Config.test_tasks)
    p.add_argument("--no-ensure-coverage", dest="ensure_coverage", action="store_false", default=Config.ensure_coverage)
    p.add_argument("--x-noise", dest="x_noise", type=float, default=Config.x_noise)
    p.add_argument("--y-noise", dest="y_noise", type=float, default=Config.y_noise)
    p.add_argument("--beta-scale", dest="beta_scale", type=float, default=Config.beta_scale)
    p.add_argument("--mixing", choices=["identity", "random_orthogonal", "random_wellcond"], default=Config.mixing)
    p.add_argument("--mixing-cond", dest="mixing_cond", type=float, default=Config.mixing_cond)

    p.add_argument("--model", choices=["option_a", "option_b"], default=Config.model)
    p.add_argument("--d-model", dest="d_model", type=int, default=Config.d_model)
    p.add_argument("--d-h", dest="d_h", type=int, default=Config.d_h)
    p.add_argument("--d-v", dest="d_v", type=int, default=Config.d_v)
    p.add_argument("--encoder-depth", dest="encoder_depth", type=int, default=Config.encoder_depth)
    p.add_argument("--mlp-hidden", dest="mlp_hidden", type=int, default=Config.mlp_hidden)
    p.add_argument("--normalize-qk", dest="normalize_qk", action="store_true", default=Config.normalize_qk)
    p.add_argument("--tau", type=float, default=Config.tau)
    p.add_argument("--q-equals-k", dest="q_equals_k", action="store_true", default=Config.q_equals_k)
    p.add_argument("--no-residual-gram", dest="residual_gram", action="store_false", default=Config.residual_gram)
    p.add_argument("--gram-strength", dest="gram_strength", type=float, default=Config.gram_strength)

    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--beta-aux", dest="beta_aux", type=float, default=Config.beta_aux)
    p.add_argument("--slot-div-reg", dest="slot_div_reg", type=float, default=Config.slot_div_reg)
    p.add_argument("--log-every", dest="log_every", type=int, default=Config.log_every)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=Config.eval_every)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=Config.grad_clip)

    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--device", default=Config.device)
    p.add_argument("--outdir", default=Config.outdir)
    p.add_argument("--run-name", dest="run_name", default=Config.run_name)
    p.add_argument("--eval-prompt-grid", dest="eval_prompt_grid", default=Config.eval_prompt_grid)

    # Sweep-only
    p.add_argument("--sweep-ks", default="")
    p.add_argument("--sweep-dh", default="")
    p.add_argument("--sweep-train-tasks", default="")
    p.add_argument("--sweep-train-prompts", default="")
    p.add_argument("--sweep-seeds", default="")
    p.add_argument("--auto-dstar", action="store_true", help="in sweeps set d_star=max(d_star,K)")
    p.add_argument("--auto-rawdim", action="store_true", help="in sweeps set raw_dim=max(raw_dim,K) when raw_dim>0")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = Config(**vars(args_to_config_namespace(args)))
    outdir = ensure_dir(cfg.outdir)
    with open(outdir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if args.mode == "single":
        hist, final = train_single(cfg)
        hist_path = outdir / f"history_{cfg.run_name}.csv"
        final_path = outdir / f"final_{cfg.run_name}.json"
        hist.to_csv(hist_path, index=False)
        with open(final_path, "w") as f:
            json.dump(final, f, indent=2)
        print(json.dumps(final, indent=2))
        print(f"Saved history to {hist_path}")
        if plt is not None and not hist.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for split, sub in hist.groupby("split"):
                if "test_mse" in sub.columns:
                    ax.plot(sub["step"], sub["test_mse"], marker="o", label=split)
            ax.set_yscale("log")
            ax.set_xlabel("step")
            ax.set_ylabel("MSE")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(outdir / f"curve_{cfg.run_name}.png", dpi=160)
            plt.close(fig)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
