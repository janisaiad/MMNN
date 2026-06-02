#!/usr/bin/env python3
"""
Unified softmax-Bayes and Q/K/V-Richardson attention lab.

Purpose
-------
This file explicitly separates and unifies three mechanisms:

1) Softmax empirical Bayes / kernel posterior averaging.
   - This is the mechanism behind softmax attention as a Bayes posterior over
     empirical memory atoms.

2) Linear Q/K/V attention Richardson.
   - This constructively implements the signed product G^T r required by ridge/KRR.

3) Hybrid softmax routing + signed values.
   - Softmax chooses relevant weak equations; values carry signed residual evidence
     g_i (b_i - g_i^T beta). This is the bridge between local Bayes retrieval and
     global Richardson/KRR correction.

No Sinkhorn. No unbalanced attention.

Tasks
-----
weak_inverse:
    G beta = b + noise
    H = noise_prec G^T G + prior_prec I
    c = noise_prec G^T b
    beta_* = H^{-1} c

    Recurrent update:
        beta_{l+1} = beta_l + B m_l

    Different methods produce m_l:
        linear_richardson:
            m_l = noise_prec G^T (b - G beta_l) - prior_prec beta_l
            computed through explicit Q/K/V linear attention.

        softmax_scalar:
            negative control; softmax over scalar residual values.
            Does not compute signed G^T r.

        softmax_vector:
            softmax routing over equations, but value is signed vector
            v_i = g_i (b_i - g_i^T beta_l). This can approximate gradient if routing
            covers the relevant equations and scaling is chosen properly.

eb_denoise:
    Empirical Bayes denoising with memory atoms x_i and noisy query y_q.
    Softmax posterior:
        p(i | y_q) ∝ exp(-||y_q - x_i||^2 / (2 sigma^2))
    prediction:
        E[x | y_q] ≈ sum_i p(i | y_q) x_i

Typical commands
----------------
Smoke weak inverse linear Richardson:
    python unified_softmax_richardson_attention_lab.py --mode smoke --task weak_inverse --device cuda

Depth sweep, exact linear Q/K/V Richardson:
    python unified_softmax_richardson_attention_lab.py --mode sweep_depth --task weak_inverse \
      --method linear_richardson --K 16 --heads 1 --d-head 16 \
      --depth-grid 1,2,4,8,16,32,64 --device cuda

Capacity sweep:
    python unified_softmax_richardson_attention_lab.py --mode sweep_capacity --task weak_inverse \
      --method linear_richardson --K-grid 8,16,32 --heads-grid 1,2,4,8 \
      --d-head-grid 1,2,4,8,16,32 --device cuda

Softmax negative control:
    python unified_softmax_richardson_attention_lab.py --mode sweep_depth --task weak_inverse \
      --method softmax_scalar --K 16 --heads 1 --d-head 16 --temperature 0.5 --device cuda

Hybrid softmax routing + signed values:
    python unified_softmax_richardson_attention_lab.py --mode sweep_depth --task weak_inverse \
      --method softmax_vector --K 16 --heads 4 --d-head 4 --temperature 1.0 \
      --softmax-vector-scale context_mean --device cuda

Empirical Bayes softmax:
    python unified_softmax_richardson_attention_lab.py --mode sweep_prompt --task eb_denoise \
      --method softmax_eb --eb-dim 4 --prompt-grid 16,32,64,128,256,512 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_outdir(outdir: str) -> Path:
    p = Path(outdir)
    if p.is_absolute():
        return p
    return project_root() / "data" / "transformers" / p.name


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_csv(path: Path, row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def parse_grid(s: str, typ=int) -> List:
    return [typ(x) for x in str(s).split(",") if str(x).strip()]


def batch_eye(B: int, K: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.eye(K, device=device, dtype=dtype).expand(B, K, K)


def inv_spd(A: Tensor, jitter: float = 1e-7) -> Tensor:
    K = A.shape[-1]
    return torch.linalg.inv(A + jitter * torch.eye(K, device=A.device, dtype=A.dtype))


def gaussian_nll(y: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = var.clamp_min(1e-10)
    return 0.5 * (torch.log(2.0 * torch.pi * var) + (y - mean).pow(2) / var)


def coverage_width(y: Tensor, mean: Tensor, var: Tensor) -> Tuple[float, float]:
    z = 1.959963984540054
    std = var.clamp_min(1e-10).sqrt()
    lo, hi = mean - z * std, mean + z * std
    cov = ((y >= lo) & (y <= hi)).float().mean().item()
    width = (hi - lo).mean().item()
    return cov, width


# -----------------------------------------------------------------------------
# Weak inverse task
# -----------------------------------------------------------------------------

@dataclass
class WeakCfg:
    K: int = 16
    prompt_len: int = 128
    prior_var: float = 1.0
    noise_var: float = 0.02
    design: str = "isotropic"  # isotropic | correlated | spiked
    cond: float = 10.0
    spike_strength: float = 4.0
    dtype: str = "float32"


@dataclass
class WeakBatch:
    G: Tensor
    b: Tensor
    gq: Tensor
    yq: Tensor
    beta_true: Tensor
    H: Tensor
    c: Tensor
    beta_post: Tensor
    cov_post: Tensor
    mean_exact: Tensor
    var_exact: Tensor
    eigvals: Tensor


def design_sqrt(K: int, cfg: WeakCfg, device: torch.device, dtype: torch.dtype) -> Tensor:
    if cfg.design == "isotropic":
        return torch.eye(K, device=device, dtype=dtype)
    if cfg.design == "correlated":
        vals = torch.logspace(0, math.log10(cfg.cond), K, device=device, dtype=dtype)
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    if cfg.design == "spiked":
        vals = torch.ones(K, device=device, dtype=dtype)
        vals[0] = cfg.spike_strength
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    raise ValueError(cfg.design)


def sample_weak_batch(B: int, cfg: WeakCfg, device: torch.device) -> WeakBatch:
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    K, m = cfg.K, cfg.prompt_len
    Csqrt = design_sqrt(K, cfg, device, dtype)
    G = torch.randn(B, m, K, device=device, dtype=dtype) @ Csqrt.T / math.sqrt(K)
    gq = torch.randn(B, K, device=device, dtype=dtype) @ Csqrt.T / math.sqrt(K)
    beta = torch.randn(B, K, device=device, dtype=dtype) * math.sqrt(cfg.prior_var)
    b = torch.einsum("bmk,bk->bm", G, beta)
    b = b + torch.randn(B, m, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)
    yq = torch.einsum("bk,bk->b", gq, beta)
    yq = yq + torch.randn(B, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)

    noise_prec = 1.0 / cfg.noise_var
    prior_prec = 1.0 / cfg.prior_var
    I = batch_eye(B, K, device, dtype)
    H = noise_prec * torch.einsum("bmk,bml->bkl", G, G) + prior_prec * I
    c = noise_prec * torch.einsum("bmk,bm->bk", G, b)
    cov = inv_spd(H)
    beta_post = torch.einsum("bkl,bl->bk", cov, c)
    mean = torch.einsum("bk,bk->b", gq, beta_post)
    var = cfg.noise_var + torch.einsum("bk,bkl,bl->b", gq, cov, gq).clamp_min(0.0)
    eigvals = torch.linalg.eigvalsh(H)
    return WeakBatch(G, b, gq, yq, beta, H, c, beta_post, cov, mean, var, eigvals)


# -----------------------------------------------------------------------------
# Empirical Bayes denoising task
# -----------------------------------------------------------------------------

@dataclass
class EBCfg:
    dim: int = 4
    prompt_len: int = 128
    noise_var: float = 0.05
    prior: str = "gaussian_mixture"  # gaussian | gaussian_mixture
    mixture_sep: float = 2.0
    dtype: str = "float32"


@dataclass
class EBBatch:
    x_mem: Tensor       # clean memory atoms [B,n,d]
    y_mem: Tensor       # noisy memory observations [B,n,d]
    x_true: Tensor      # query clean target [B,d]
    y_query: Tensor     # noisy query [B,d]
    mean_emp: Tensor    # empirical posterior mean using clean x_mem [B,d]
    var_emp: Tensor     # empirical posterior coordinate variance [B,d]


def sample_eb_batch(B: int, cfg: EBCfg, device: torch.device) -> EBBatch:
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    n, d = cfg.prompt_len, cfg.dim

    def sample_prior(shape):
        if cfg.prior == "gaussian":
            return torch.randn(*shape, device=device, dtype=dtype)
        if cfg.prior == "gaussian_mixture":
            signs = torch.randint(0, 2, shape[:-1] + (1,), device=device).to(dtype) * 2 - 1
            mean = signs * cfg.mixture_sep
            return mean + torch.randn(*shape, device=device, dtype=dtype)
        raise ValueError(cfg.prior)

    x_mem = sample_prior((B, n, d))
    y_mem = x_mem + torch.randn(B, n, d, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)
    x_true = sample_prior((B, d))
    y_query = x_true + torch.randn(B, d, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)

    # Empirical Bayes posterior over memory atoms, using clean atoms as empirical prior.
    diff = y_query[:, None, :] - x_mem
    logits = -0.5 * diff.pow(2).sum(-1) / cfg.noise_var
    w = torch.softmax(logits, dim=-1)
    mean = torch.einsum("bn,bnd->bd", w, x_mem)
    second = torch.einsum("bn,bnd->bd", w, x_mem.pow(2))
    var = (second - mean.pow(2)).clamp_min(1e-10)
    return EBBatch(x_mem, y_mem, x_true, y_query, mean, var)


# -----------------------------------------------------------------------------
# Heads/projections and preconditioners
# -----------------------------------------------------------------------------

def make_heads(K: int, H: int, dh: int, scheme: str, device: torch.device, dtype: torch.dtype, seed: int = 0) -> List[Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 17)
    Ps: List[Tensor] = []
    if scheme == "coordinate":
        ptr = 0
        for _ in range(H):
            P = torch.zeros(dh, K, device=device, dtype=dtype)
            for a in range(dh):
                if ptr < K:
                    P[a, ptr] = 1.0
                    ptr += 1
            Ps.append(P)
        return Ps
    if scheme == "cyclic_coordinate":
        for h in range(H):
            P = torch.zeros(dh, K, device=device, dtype=dtype)
            for a in range(dh):
                P[a, (h * dh + a) % K] = 1.0
            Ps.append(P)
        return Ps
    if scheme == "random_orthogonal":
        M = torch.randn(K, K, device=device, dtype=dtype, generator=gen)
        Q, _ = torch.linalg.qr(M)
        ptr = 0
        for _ in range(H):
            rows = []
            for _a in range(dh):
                rows.append(Q[:, ptr % K])
                ptr += 1
            Ps.append(torch.stack(rows, dim=0))
        return Ps
    raise ValueError(scheme)


def projection_correction(Ps: List[Tensor], K: int, mode: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return C such that corrected gradient = C sum_h P_h^T o_h.

    none: identity.
    pinv: pseudo-inverse of M_P=sum_h P_h^T P_h on its range.
    diag: diagonal inverse correction.
    """
    M = torch.zeros(K, K, device=device, dtype=dtype)
    for P in Ps:
        M = M + P.T @ P
    if mode == "none":
        return torch.eye(K, device=device, dtype=dtype)
    if mode == "pinv":
        return torch.linalg.pinv(M.float()).to(dtype)
    if mode == "diag":
        d = torch.diagonal(M).clamp_min(1e-8)
        return torch.diag(1.0 / d)
    raise ValueError(mode)


def build_precond(Hmat: Tensor, precond: str, eta_multiplier: float, heads: int, d_head: int) -> Tuple[Tensor, Dict[str, float]]:
    B, K, _ = Hmat.shape
    dev, dtype = Hmat.device, Hmat.dtype
    I = batch_eye(B, K, dev, dtype)
    eig, U = torch.linalg.eigh(Hmat)
    lmin, lmax = eig[:, 0], eig[:, -1]
    eta_opt = eta_multiplier * 2.0 / (lmax + lmin)
    if precond == "scalar_opt":
        return eta_opt[:, None, None] * I, {"eta_mean": eta_opt.mean().item(), "precond_rank": 0.0}
    if precond == "scalar_lmax":
        eta = eta_multiplier / lmax
        return eta[:, None, None] * I, {"eta_mean": eta.mean().item(), "precond_rank": 0.0}
    if precond == "jacobi":
        Dinv = torch.diag_embed(1.0 / torch.diagonal(Hmat, dim1=-2, dim2=-1).clamp_min(1e-8))
        M = torch.einsum("bkl,blm->bkm", Dinv, Hmat)
        ev = torch.linalg.eigvals(M).abs().real.max(-1).values
        eta = eta_multiplier / ev.clamp_min(1e-8)
        return eta[:, None, None] * Dinv, {"eta_mean": eta.mean().item(), "precond_rank": float(K)}
    if precond == "spectral_full":
        inv_e = 1.0 / eig.clamp_min(1e-8)
        Bmat = torch.einsum("bkr,br,blr->bkl", U, inv_e, U)
        return Bmat, {"eta_mean": 1.0, "precond_rank": float(K)}
    if precond == "lowrank_spectral":
        r = min(K, heads * d_head)
        Usel, esel = U[:, :, :r], eig[:, :r]
        Bsel = torch.einsum("bkr,br,blr->bkl", Usel, 1.0 / esel.clamp_min(1e-8), Usel)
        Psel = torch.einsum("bkr,blr->bkl", Usel, Usel)
        Bmat = Bsel + eta_opt[:, None, None] * (I - Psel)
        return Bmat, {"eta_mean": eta_opt.mean().item(), "precond_rank": float(r)}
    raise ValueError(precond)


# -----------------------------------------------------------------------------
# Weak inverse unified attention messages
# -----------------------------------------------------------------------------

def weak_message(
    G: Tensor,
    residual: Tensor,
    beta: Tensor,
    Ps: List[Tensor],
    method: str,
    temperature: float,
    noise_prec: float,
    prior_prec: float,
    softmax_vector_scale: str,
    proj_corr: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    """Return message approximating c-H beta = noise_prec G^T r - prior_prec beta."""
    B, m, K = G.shape
    dev, dtype = G.device, G.dtype
    raw = torch.zeros(B, K, device=dev, dtype=dtype)
    ranks, effranks, entropies = [], [], []

    if method == "linear_richardson":
        # Q canonical, K=P_h g_i, V=residual. Output P_h G^T residual.
        for P in Ps:
            Kh = torch.einsum("dk,bmk->bmd", P, G)  # [B,m,dh]
            o = torch.einsum("bmd,bm->bd", Kh, residual)
            raw = raw + torch.einsum("dk,bd->bk", P, o)
            S0 = Kh[0].T.detach()
            sv = torch.linalg.svdvals(S0.float())
            ranks.append((sv > 1e-6 * sv.max().clamp_min(1e-8)).float().sum().item())
            p = sv / sv.sum().clamp_min(1e-12)
            effranks.append(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())
        raw = torch.einsum("kl,bl->bk", proj_corr, raw)
        return noise_prec * raw - prior_prec * beta, {
            "attn_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
            "attn_effrank_mean": float(np.mean(effranks)) if effranks else 0.0,
            "attn_entropy_mean": 0.0,
        }

    if method == "softmax_scalar":
        # Negative control: softmax weights on scalar residual values.
        for P in Ps:
            Kh = torch.einsum("dk,bmk->bmd", P, G)
            logits = Kh.transpose(1, 2) / max(temperature, 1e-8)  # [B,dh,m]
            A = torch.softmax(logits, dim=-1)
            o = torch.einsum("bdm,bm->bd", A, residual)
            raw = raw + torch.einsum("dk,bd->bk", P, o)
            entropies.append((-(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(-1)).mean().item())
            S0 = logits[0].detach()
            sv = torch.linalg.svdvals(S0.float())
            ranks.append((sv > 1e-6 * sv.max().clamp_min(1e-8)).float().sum().item())
            p = sv / sv.sum().clamp_min(1e-12)
            effranks.append(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())
        raw = torch.einsum("kl,bl->bk", proj_corr, raw)
        return noise_prec * raw - prior_prec * beta, {
            "attn_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
            "attn_effrank_mean": float(np.mean(effranks)) if effranks else 0.0,
            "attn_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        }

    if method == "softmax_vector":
        # Bridge: softmax routes equations, value is signed vector g_i r_i.
        signed_values = G * residual[:, :, None]  # [B,m,K]
        slot_messages = []
        for P in Ps:
            Kh = torch.einsum("dk,bmk->bmd", P, G)
            logits = Kh.transpose(1, 2) / max(temperature, 1e-8)  # [B,dh,m]
            A = torch.softmax(logits, dim=-1)
            msg = torch.einsum("bdm,bmk->bdk", A, signed_values)  # [B,dh,K]
            slot_messages.append(msg)
            entropies.append((-(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(-1)).mean().item())
            S0 = logits[0].detach()
            sv = torch.linalg.svdvals(S0.float())
            ranks.append((sv > 1e-6 * sv.max().clamp_min(1e-8)).float().sum().item())
            p = sv / sv.sum().clamp_min(1e-12)
            effranks.append(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())
        M = torch.cat(slot_messages, dim=1)  # [B,S,K]
        if softmax_vector_scale == "none":
            raw = M.mean(dim=1)
        elif softmax_vector_scale == "context":
            raw = M.mean(dim=1) * float(m)
        elif softmax_vector_scale == "context_sum":
            raw = M.sum(dim=1) * float(m) / max(M.shape[1], 1)
        elif softmax_vector_scale == "sqrt_context":
            raw = M.mean(dim=1) * math.sqrt(float(m))
        else:
            raise ValueError(softmax_vector_scale)
        return noise_prec * raw - prior_prec * beta, {
            "attn_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
            "attn_effrank_mean": float(np.mean(effranks)) if effranks else 0.0,
            "attn_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        }

    raise ValueError(method)


@dataclass
class WeakRunCfg:
    depth: int = 8
    heads: int = 1
    d_head: int = 16
    method: str = "linear_richardson"  # linear_richardson | softmax_scalar | softmax_vector
    head_scheme: str = "coordinate"
    precond: str = "scalar_opt"
    temperature: float = 0.5
    eta_multiplier: float = 1.0
    projection_correction: str = "none"  # none | pinv | diag
    softmax_vector_scale: str = "context"  # none | context | context_sum | sqrt_context
    batch_size: int = 1024
    eval_batches: int = 8
    seed: int = 0


def run_weak_loop(batch: WeakBatch, task: WeakCfg, run: WeakRunCfg, Ps: List[Tensor]) -> Dict:
    G, b, Hmat, c = batch.G, batch.b, batch.H, batch.c
    Bsz, m, K = G.shape
    noise_prec, prior_prec = 1.0 / task.noise_var, 1.0 / task.prior_var
    Bpre, pst = build_precond(Hmat, run.precond, run.eta_multiplier, run.heads, run.d_head)
    dtype, dev = G.dtype, G.device
    proj_corr = projection_correction(Ps, K, run.projection_correction, dev, dtype)

    beta = torch.zeros(Bsz, K, device=dev, dtype=dtype)
    layer_post, layer_grad, ranks, effs, ents = [], [], [], [], []
    for _ in range(run.depth):
        residual = b - torch.einsum("bmk,bk->bm", G, beta)
        msg, ast = weak_message(
            G, residual, beta, Ps, run.method, run.temperature,
            noise_prec, prior_prec, run.softmax_vector_scale, proj_corr,
        )
        exact_grad = c - torch.einsum("bkl,bl->bk", Hmat, beta)
        layer_grad.append((msg - exact_grad).pow(2).mean().item())
        ranks.append(ast["attn_rank_mean"])
        effs.append(ast["attn_effrank_mean"])
        ents.append(ast["attn_entropy_mean"])
        beta = beta + torch.einsum("bkl,bl->bk", Bpre, msg)
        layer_post.append((beta - batch.beta_post).pow(2).mean().item())

    mean_iter = torch.einsum("bk,bk->b", batch.gq, beta)
    var = batch.var_exact
    rho = torch.linalg.eigvals(batch_eye(Bsz, K, dev, dtype) - torch.einsum("bkl,blm->bkm", Bpre, Hmat)).abs().real.max(dim=-1).values

    Mproj = torch.zeros(K, K, device=dev, dtype=dtype)
    for P in Ps:
        Mproj = Mproj + P.T @ P
    evproj = torch.linalg.eigvalsh(Mproj.float())

    return {
        "beta_mse_post": (beta - batch.beta_post).pow(2).mean().item(),
        "beta_mse_true": (beta - batch.beta_true).pow(2).mean().item(),
        "pred_mse_y": (mean_iter - batch.yq).pow(2).mean().item(),
        "mean_mse_exact": (mean_iter - batch.mean_exact).pow(2).mean().item(),
        "exact_pred_mse_y": (batch.mean_exact - batch.yq).pow(2).mean().item(),
        "nll_iter": gaussian_nll(batch.yq, mean_iter, var).mean().item(),
        "nll_exact": gaussian_nll(batch.yq, batch.mean_exact, batch.var_exact).mean().item(),
        "coverage_iter": coverage_width(batch.yq, mean_iter, var)[0],
        "width_iter": coverage_width(batch.yq, mean_iter, var)[1],
        "coverage_exact": coverage_width(batch.yq, batch.mean_exact, batch.var_exact)[0],
        "width_exact": coverage_width(batch.yq, batch.mean_exact, batch.var_exact)[1],
        "contraction_radius_mean": rho.mean().item(),
        "contraction_radius_max": rho.max().item(),
        "precond_rank": pst["precond_rank"],
        "eta_mean": pst["eta_mean"],
        "proj_rank": float(torch.linalg.matrix_rank(Mproj.float()).item()),
        "proj_max": evproj.max().item(),
        "proj_min_nonzero": evproj[evproj > 1e-7].min().item() if (evproj > 1e-7).any() else 0.0,
        "capacity_rank": float(run.heads * run.d_head),
        "capacity_ge_K": int(run.heads * run.d_head >= K),
        "first_grad_mse": layer_grad[0],
        "last_grad_mse": layer_grad[-1],
        "attn_rank_first": ranks[0],
        "attn_rank_last": ranks[-1],
        "attn_effrank_first": effs[0],
        "attn_effrank_last": effs[-1],
        "attn_entropy_first": ents[0],
        "attn_entropy_last": ents[-1],
        "layer_first_beta_mse_post": layer_post[0],
        "layer_last_beta_mse_post": layer_post[-1],
        "layer_decay_ratio": layer_post[-1] / max(layer_post[0], 1e-30),
    }


# -----------------------------------------------------------------------------
# Empirical Bayes softmax evaluation
# -----------------------------------------------------------------------------

@dataclass
class EBRunCfg:
    method: str = "softmax_eb"
    temperature: float = 1.0
    batch_size: int = 1024
    eval_batches: int = 8
    seed: int = 0


def eval_eb_once(cfg: EBCfg, run: EBRunCfg, device: torch.device) -> Dict:
    rows = []
    for _ in range(run.eval_batches):
        batch = sample_eb_batch(run.batch_size, cfg, device)
        # softmax posterior over empirical clean memory atoms; temperature rescales likelihood
        diff = batch.y_query[:, None, :] - batch.x_mem
        logits = -0.5 * diff.pow(2).sum(-1) / (cfg.noise_var * run.temperature)
        w = torch.softmax(logits, dim=-1)
        mean = torch.einsum("bn,bnd->bd", w, batch.x_mem)
        second = torch.einsum("bn,bnd->bd", w, batch.x_mem.pow(2))
        var = (second - mean.pow(2)).clamp_min(1e-10)
        mse_emp = (mean - batch.mean_emp).pow(2).mean().item()
        mse_true = (mean - batch.x_true).pow(2).mean().item()
        var_mse = (var - batch.var_emp).pow(2).mean().item()
        # Average coordinate NLL under diagonal Gaussian approximation
        nll = gaussian_nll(batch.x_true, mean, var + cfg.noise_var).mean().item()
        exact_nll = gaussian_nll(batch.x_true, batch.mean_emp, batch.var_emp + cfg.noise_var).mean().item()
        entropy = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1).mean().item()
        rows.append({
            "mean_mse_emp": mse_emp,
            "mean_mse_true": mse_true,
            "var_mse_emp": var_mse,
            "nll": nll,
            "exact_nll": exact_nll,
            "attn_entropy": entropy,
            "max_weight": w.max(dim=-1).values.mean().item(),
        })
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def save_row(csv_path: Path, row: Dict) -> None:
    row = dict(row)
    if "beta_mse_post" in row:
        row["log10_beta_mse_post"] = math.log10(max(row["beta_mse_post"], 1e-30))
        row["log10_beta_mse_true"] = math.log10(max(row["beta_mse_true"], 1e-30))
    append_csv(csv_path, row)
    print(pd.Series(row).to_string())


def eval_weak(task: WeakCfg, run: WeakRunCfg, device: torch.device) -> Dict:
    dtype = torch.float64 if task.dtype == "float64" else torch.float32
    Ps = make_heads(task.K, run.heads, run.d_head, run.head_scheme, device, dtype, run.seed)
    rows = []
    for _ in range(run.eval_batches):
        rows.append(run_weak_loop(sample_weak_batch(run.batch_size, task, device), task, run, Ps))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="single", choices=["smoke", "single", "sweep_depth", "sweep_prompt", "sweep_capacity", "sweep_method", "sweep_temperature", "sweep_precond"])
    ap.add_argument("--task", default="weak_inverse", choices=["weak_inverse", "eb_denoise"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=str(resolve_outdir("runs_unified_softmax_richardson")))
    ap.add_argument("--seed", type=int, default=0)

    # weak task
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--noise-var", type=float, default=0.02)
    ap.add_argument("--prior-var", type=float, default=1.0)
    ap.add_argument("--design", default="isotropic", choices=["isotropic", "correlated", "spiked"])
    ap.add_argument("--cond", type=float, default=10.0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    # eb task
    ap.add_argument("--eb-dim", type=int, default=4)
    ap.add_argument("--eb-prior", default="gaussian_mixture", choices=["gaussian", "gaussian_mixture"])
    ap.add_argument("--eb-mixture-sep", type=float, default=2.0)

    # recurrent / attention
    ap.add_argument("--method", default="linear_richardson", choices=["linear_richardson", "softmax_scalar", "softmax_vector", "softmax_eb"])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--d-head", type=int, default=16)
    ap.add_argument("--head-scheme", default="coordinate", choices=["coordinate", "cyclic_coordinate", "random_orthogonal"])
    ap.add_argument("--precond", default="scalar_opt", choices=["scalar_opt", "scalar_lmax", "jacobi", "lowrank_spectral", "spectral_full"])
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--eta-multiplier", type=float, default=1.0)
    ap.add_argument("--projection-correction", default="none", choices=["none", "pinv", "diag"])
    ap.add_argument("--softmax-vector-scale", default="context", choices=["none", "context", "context_sum", "sqrt_context"])

    # eval
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-batches", type=int, default=8)

    # grids
    ap.add_argument("--depth-grid", default="1,2,4,8,16,32,64")
    ap.add_argument("--prompt-grid", default="16,32,64,128,256,512")
    ap.add_argument("--K-grid", default="8,16,32")
    ap.add_argument("--heads-grid", default="1,2,4,8")
    ap.add_argument("--d-head-grid", default="1,2,4,8,16,32")
    ap.add_argument("--method-grid", default="linear_richardson,softmax_scalar,softmax_vector")
    ap.add_argument("--temperature-grid", default="0.1,0.2,0.5,1.0,2.0,5.0")
    ap.add_argument("--precond-grid", default="scalar_opt,scalar_lmax,jacobi,lowrank_spectral,spectral_full")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    outdir = ensure_dir(resolve_outdir(args.outdir))
    csv_path = outdir / "results.csv"

    def weak_task(K=None, prompt=None):
        return WeakCfg(
            K=args.K if K is None else K,
            prompt_len=args.prompt_len if prompt is None else prompt,
            prior_var=args.prior_var,
            noise_var=args.noise_var,
            design=args.design,
            cond=args.cond,
            dtype=args.dtype,
        )

    def weak_run(depth=None, heads=None, d_head=None, method=None, temp=None, precond=None):
        return WeakRunCfg(
            depth=args.depth if depth is None else depth,
            heads=args.heads if heads is None else heads,
            d_head=args.d_head if d_head is None else d_head,
            method=args.method if method is None else method,
            head_scheme=args.head_scheme,
            precond=args.precond if precond is None else precond,
            temperature=args.temperature if temp is None else temp,
            eta_multiplier=args.eta_multiplier,
            projection_correction=args.projection_correction,
            softmax_vector_scale=args.softmax_vector_scale,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )

    def eb_task(prompt=None):
        return EBCfg(
            dim=args.eb_dim,
            prompt_len=args.prompt_len if prompt is None else prompt,
            noise_var=args.noise_var,
            prior=args.eb_prior,
            mixture_sep=args.eb_mixture_sep,
            dtype=args.dtype,
        )

    def eb_run(temp=None):
        return EBRunCfg(
            method=args.method,
            temperature=args.temperature if temp is None else temp,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )

    def run_one(tag: str, task_override=None, run_override=None):
        if args.task == "weak_inverse":
            task = task_override or weak_task()
            run = run_override or weak_run()
            extra = eval_weak(task, run, device)
            row = {
                "task": "weak_inverse",
                "tag": tag,
                "K": task.K,
                "prompt_len": task.prompt_len,
                "design": task.design,
                "cond": task.cond,
                "noise_var": task.noise_var,
                "prior_var": task.prior_var,
                "method": run.method,
                "depth": run.depth,
                "heads": run.heads,
                "d_head": run.d_head,
                "capacity_rank": run.heads * run.d_head,
                "head_scheme": run.head_scheme,
                "precond": run.precond,
                "temperature": run.temperature,
                "projection_correction": run.projection_correction,
                "softmax_vector_scale": run.softmax_vector_scale,
                "seed": run.seed,
            }
            row.update(extra)
            save_row(csv_path, row)
            return row
        else:
            task = task_override or eb_task()
            run = run_override or eb_run()
            extra = eval_eb_once(task, run, device)
            row = {
                "task": "eb_denoise",
                "tag": tag,
                "prompt_len": task.prompt_len,
                "noise_var": task.noise_var,
                "method": "softmax_eb",
                "temperature": run.temperature,
                "eb_dim": task.dim,
                "eb_prior": task.prior,
                "eb_mixture_sep": task.mixture_sep,
                "seed": run.seed,
            }
            row.update(extra)
            save_row(csv_path, row)
            return row

    if args.mode == "smoke":
        if args.task == "weak_inverse":
            run_one("smoke", weak_task(K=8, prompt=64), weak_run(depth=8, heads=1, d_head=8, method="linear_richardson"))
        else:
            run_one("smoke", eb_task(prompt=64), eb_run())
        return

    if args.mode == "single":
        run_one("single")
        return

    if args.mode == "sweep_depth":
        for d in parse_grid(args.depth_grid, int):
            run_one(f"depth_{d}", run_override=weak_run(depth=d))
        return

    if args.mode == "sweep_prompt":
        for m in parse_grid(args.prompt_grid, int):
            if args.task == "weak_inverse":
                run_one(f"prompt_{m}", task_override=weak_task(prompt=m))
            else:
                run_one(f"prompt_{m}", task_override=eb_task(prompt=m))
        return

    if args.mode == "sweep_capacity":
        for K in parse_grid(args.K_grid, int):
            for H in parse_grid(args.heads_grid, int):
                for dh in parse_grid(args.d_head_grid, int):
                    run_one(f"K{K}_H{H}_dh{dh}", task_override=weak_task(K=K), run_override=weak_run(heads=H, d_head=dh))
        return

    if args.mode == "sweep_method":
        if args.task != "weak_inverse":
            raise ValueError("sweep_method is for weak_inverse")
        for method in [x for x in args.method_grid.split(",") if x]:
            run_one(f"method_{method}", run_override=weak_run(method=method))
        return

    if args.mode == "sweep_temperature":
        for temp in parse_grid(args.temperature_grid, float):
            if args.task == "weak_inverse":
                run_one(f"temp_{temp}", run_override=weak_run(temp=temp))
            else:
                run_one(f"temp_{temp}", run_override=eb_run(temp=temp))
        return

    if args.mode == "sweep_precond":
        if args.task != "weak_inverse":
            raise ValueError("sweep_precond is for weak_inverse")
        for pc in [x for x in args.precond_grid.split(",") if x]:
            run_one(f"precond_{pc}", run_override=weak_run(precond=pc))
        return


if __name__ == "__main__":
    main()
