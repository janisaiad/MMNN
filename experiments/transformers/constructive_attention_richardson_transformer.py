#!/usr/bin/env python3
"""
Constructive Q/K/V attention-Richardson for weak-form inverse ICL.

This is the corrected constructive experiment: the update is implemented by an
explicit attention forward pass, not by directly multiplying G^T r.

Weak inverse task:
    G beta = b + noise

Bayes/ridge target:
    H = noise_prec G^T G + prior_prec I
    c = noise_prec G^T b
    beta_* = H^{-1} c

Richardson:
    beta_{l+1} = beta_l + B (c - H beta_l)

Attention realization of the residual gradient:
    r_i^l = b_i - g_i^T beta_l

For each head h with projection P_h in R^{d_h x K}:
    keys:   k_i^h = P_h g_i
    values: v_i^l = r_i^l
    queries: canonical basis e_a in R^{d_h}

Linear attention:
    o_a^h = sum_i <e_a, k_i^h> v_i^l
          = [P_h G^T r^l]_a
    mapped back: P_h^T o_h

Summing heads gives an approximation to G^T r. If the projections cover R^K,
this is exact. Softmax is included only as a negative/control because signed
Richardson requires bilinear/linear attention unless signs are encoded in V.

No Sinkhorn. No learned model by default. This checks constructively:
  - depth = number of Richardson steps
  - heads * d_head = rank/capacity of the QK attention gradient
  - Q/K/V forward computes the weak residual gradient
  - preconditioner/head rank controls convergence
"""
from __future__ import annotations

import argparse, csv, math, random
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


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


def append_csv(path: Path, row: Dict):
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)


def parse_grid(s: str, typ=int):
    return [typ(x) for x in str(s).split(',') if str(x).strip()]


def batch_eye(B, K, device, dtype):
    return torch.eye(K, device=device, dtype=dtype).expand(B, K, K)


def inv_spd(A: Tensor, jitter=1e-7):
    K = A.shape[-1]
    return torch.linalg.inv(A + jitter * torch.eye(K, device=A.device, dtype=A.dtype))


def gaussian_nll(y, mean, var):
    var = var.clamp_min(1e-10)
    return 0.5 * (torch.log(2 * torch.pi * var) + (y - mean).pow(2) / var)


def coverage_width(y, mean, var):
    z = 1.959963984540054
    std = var.clamp_min(1e-10).sqrt()
    lo, hi = mean - z * std, mean + z * std
    return ((y >= lo) & (y <= hi)).float().mean().item(), (hi - lo).mean().item()


@dataclass
class TaskCfg:
    K: int = 16
    prompt_len: int = 128
    prior_var: float = 1.0
    noise_var: float = 0.02
    design: str = 'isotropic'   # isotropic | correlated | spiked
    cond: float = 10.0
    spike_strength: float = 4.0
    dtype: str = 'float32'


@dataclass
class Batch:
    G: Tensor; b: Tensor; gq: Tensor; yq: Tensor; beta_true: Tensor
    H: Tensor; c: Tensor; beta_post: Tensor; cov_post: Tensor
    mean_exact: Tensor; var_exact: Tensor; eigvals: Tensor


def design_sqrt(K, cfg: TaskCfg, device, dtype):
    if cfg.design == 'isotropic':
        return torch.eye(K, device=device, dtype=dtype)
    if cfg.design == 'correlated':
        vals = torch.logspace(0, math.log10(cfg.cond), K, device=device, dtype=dtype)
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    if cfg.design == 'spiked':
        vals = torch.ones(K, device=device, dtype=dtype); vals[0] = cfg.spike_strength
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    raise ValueError(cfg.design)


def sample_batch(B: int, cfg: TaskCfg, device) -> Batch:
    dtype = torch.float64 if cfg.dtype == 'float64' else torch.float32
    K, m = cfg.K, cfg.prompt_len
    Csqrt = design_sqrt(K, cfg, device, dtype)
    G = torch.randn(B, m, K, device=device, dtype=dtype) @ Csqrt.T / math.sqrt(K)
    gq = torch.randn(B, K, device=device, dtype=dtype) @ Csqrt.T / math.sqrt(K)
    beta = torch.randn(B, K, device=device, dtype=dtype) * math.sqrt(cfg.prior_var)
    b = torch.einsum('bmk,bk->bm', G, beta) + torch.randn(B, m, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)
    yq = torch.einsum('bk,bk->b', gq, beta) + torch.randn(B, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)

    noise_prec, prior_prec = 1.0 / cfg.noise_var, 1.0 / cfg.prior_var
    I = batch_eye(B, K, device, dtype)
    H = noise_prec * torch.einsum('bmk,bml->bkl', G, G) + prior_prec * I
    c = noise_prec * torch.einsum('bmk,bm->bk', G, b)
    cov = inv_spd(H)
    beta_post = torch.einsum('bkl,bl->bk', cov, c)
    mean = torch.einsum('bk,bk->b', gq, beta_post)
    var = cfg.noise_var + torch.einsum('bk,bkl,bl->b', gq, cov, gq).clamp_min(0)
    eigvals = torch.linalg.eigvalsh(H)
    return Batch(G, b, gq, yq, beta, H, c, beta_post, cov, mean, var, eigvals)


def make_heads(K: int, H: int, dh: int, scheme: str, device, dtype, seed=0) -> List[Tensor]:
    """Projection heads P_h [d_h,K]. Q slots are the canonical basis in each head."""
    gen = torch.Generator(device=device); gen.manual_seed(seed + 17)
    Ps = []
    if scheme == 'coordinate':
        ptr = 0
        for _ in range(H):
            P = torch.zeros(dh, K, device=device, dtype=dtype)
            for a in range(dh):
                if ptr < K:
                    P[a, ptr] = 1.0
                    ptr += 1
            Ps.append(P)
        return Ps
    if scheme == 'cyclic_coordinate':
        for h in range(H):
            P = torch.zeros(dh, K, device=device, dtype=dtype)
            for a in range(dh):
                P[a, (h * dh + a) % K] = 1.0
            Ps.append(P)
        return Ps
    if scheme == 'random_orthogonal':
        M = torch.randn(K, K, device=device, dtype=dtype, generator=gen)
        Q, _ = torch.linalg.qr(M)
        ptr = 0
        for _ in range(H):
            rows = []
            for _a in range(dh):
                rows.append(Q[:, ptr % K]); ptr += 1
            Ps.append(torch.stack(rows, dim=0))
        return Ps
    raise ValueError(scheme)


def qkv_attention_gradient(G: Tensor, residual: Tensor, Ps: List[Tensor], attention: str,
                           temperature: float, scale_linear_by_m: bool) -> Tuple[Tensor, Dict[str, float]]:
    """Approximate G^T residual via explicit Q/K/V attention.

    Linear mode is exact when projections cover R^K.
    Softmax mode is a diagnostic/control, not exact signed Richardson.
    """
    B, m, K = G.shape
    grad = torch.zeros(B, K, device=G.device, dtype=G.dtype)
    ranks, effranks, entropies = [], [], []
    for P in Ps:
        Kh = torch.einsum('dk,bmk->bmd', P, G)  # [B,m,dh] keys
        if attention == 'linear':
            # o[d] = sum_i key_i[d] * residual_i ; Q slots are e_d
            o = torch.einsum('bmd,bm->bd', Kh, residual)
            if scale_linear_by_m: o = o / float(m)
            gh = torch.einsum('dk,bd->bk', P, o)
            S0 = Kh[0].T.detach()  # QK^T for canonical Q
        elif attention == 'softmax':
            logits = Kh.transpose(1, 2) / max(temperature, 1e-8)  # [B,dh,m]
            A = torch.softmax(logits, dim=-1)
            o = torch.einsum('bdm,bm->bd', A, residual)
            gh = torch.einsum('dk,bd->bk', P, o)
            S0 = logits[0].detach()
            ent = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(-1).mean().item()
            entropies.append(ent)
        else:
            raise ValueError(attention)
        grad = grad + gh
        sv = torch.linalg.svdvals(S0.float())
        ranks.append((sv > 1e-6 * sv.max().clamp_min(1e-8)).float().sum().item())
        p = sv / sv.sum().clamp_min(1e-12)
        effranks.append(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())
    return grad, {
        'attn_rank_mean': float(np.mean(ranks)) if ranks else 0.0,
        'attn_effrank_mean': float(np.mean(effranks)) if effranks else 0.0,
        'attn_entropy_mean': float(np.mean(entropies)) if entropies else 0.0,
    }


def build_precond(Hmat: Tensor, precond: str, eta_multiplier: float, heads: int, d_head: int):
    B, K, _ = Hmat.shape; dev, dtype = Hmat.device, Hmat.dtype
    I = batch_eye(B, K, dev, dtype)
    eig, U = torch.linalg.eigh(Hmat)
    lmin, lmax = eig[:, 0], eig[:, -1]
    eta_opt = eta_multiplier * 2.0 / (lmax + lmin)
    if precond == 'scalar_opt':
        return eta_opt[:, None, None] * I, {'eta_mean': eta_opt.mean().item(), 'precond_rank': 0.0}
    if precond == 'scalar_lmax':
        eta = eta_multiplier / lmax
        return eta[:, None, None] * I, {'eta_mean': eta.mean().item(), 'precond_rank': 0.0}
    if precond == 'jacobi':
        Dinv = torch.diag_embed(1.0 / torch.diagonal(Hmat, dim1=-2, dim2=-1).clamp_min(1e-8))
        M = torch.einsum('bkl,blm->bkm', Dinv, Hmat)
        ev = torch.linalg.eigvals(M).abs().real.max(-1).values
        eta = eta_multiplier / ev.clamp_min(1e-8)
        return eta[:, None, None] * Dinv, {'eta_mean': eta.mean().item(), 'precond_rank': float(K)}
    if precond == 'spectral_full':
        inv_e = 1.0 / eig.clamp_min(1e-8)
        Bmat = torch.einsum('bkr,br,blr->bkl', U, inv_e, U)
        return Bmat, {'eta_mean': 1.0, 'precond_rank': float(K)}
    if precond == 'lowrank_spectral':
        r = min(K, heads * d_head)
        Usel, esel = U[:, :, :r], eig[:, :r]
        Bsel = torch.einsum('bkr,br,blr->bkl', Usel, 1.0 / esel.clamp_min(1e-8), Usel)
        Psel = torch.einsum('bkr,blr->bkl', Usel, Usel)
        Bmat = Bsel + eta_opt[:, None, None] * (I - Psel)
        return Bmat, {'eta_mean': eta_opt.mean().item(), 'precond_rank': float(r)}
    raise ValueError(precond)


@dataclass
class RunCfg:
    depth: int = 8
    heads: int = 1
    d_head: int = 16
    attention: str = 'linear'
    head_scheme: str = 'coordinate'
    precond: str = 'scalar_opt'
    temperature: float = 0.5
    eta_multiplier: float = 1.0
    scale_linear_by_m: bool = False
    batch_size: int = 1024
    eval_batches: int = 8
    seed: int = 0


def run_loop(batch: Batch, task: TaskCfg, run: RunCfg, Ps: List[Tensor]) -> Dict:
    G, b, Hmat, c = batch.G, batch.b, batch.H, batch.c
    Bsz, m, K = G.shape
    noise_prec, prior_prec = 1.0 / task.noise_var, 1.0 / task.prior_var
    Bpre, pst = build_precond(Hmat, run.precond, run.eta_multiplier, run.heads, run.d_head)
    beta = torch.zeros(Bsz, K, device=G.device, dtype=G.dtype)
    layer_post, layer_grad, ranks, effs = [], [], [], []
    for _ in range(run.depth):
        residual = b - torch.einsum('bmk,bk->bm', G, beta)
        raw_grad, ast = qkv_attention_gradient(G, residual, Ps, run.attention, run.temperature, run.scale_linear_by_m)
        attn_grad = noise_prec * raw_grad - prior_prec * beta
        exact_grad = c - torch.einsum('bkl,bl->bk', Hmat, beta)
        layer_grad.append((attn_grad - exact_grad).pow(2).mean().item())
        ranks.append(ast['attn_rank_mean']); effs.append(ast['attn_effrank_mean'])
        beta = beta + torch.einsum('bkl,bl->bk', Bpre, attn_grad)
        layer_post.append((beta - batch.beta_post).pow(2).mean().item())
    mean_iter = torch.einsum('bk,bk->b', batch.gq, beta)
    var = batch.var_exact
    rho = torch.linalg.eigvals(batch_eye(Bsz, K, G.device, G.dtype) - torch.einsum('bkl,blm->bkm', Bpre, Hmat)).abs().real.max(dim=-1).values
    cov_proj = torch.zeros(K, K, device=G.device, dtype=G.dtype)
    for P in Ps: cov_proj = cov_proj + P.T @ P
    evals_proj = torch.linalg.eigvalsh(cov_proj.float())
    return {
        'beta_mse_post': (beta - batch.beta_post).pow(2).mean().item(),
        'beta_mse_true': (beta - batch.beta_true).pow(2).mean().item(),
        'pred_mse_y': (mean_iter - batch.yq).pow(2).mean().item(),
        'mean_mse_exact': (mean_iter - batch.mean_exact).pow(2).mean().item(),
        'exact_pred_mse_y': (batch.mean_exact - batch.yq).pow(2).mean().item(),
        'nll_iter': gaussian_nll(batch.yq, mean_iter, var).mean().item(),
        'nll_exact': gaussian_nll(batch.yq, batch.mean_exact, batch.var_exact).mean().item(),
        'coverage_iter': coverage_width(batch.yq, mean_iter, var)[0],
        'width_iter': coverage_width(batch.yq, mean_iter, var)[1],
        'coverage_exact': coverage_width(batch.yq, batch.mean_exact, batch.var_exact)[0],
        'width_exact': coverage_width(batch.yq, batch.mean_exact, batch.var_exact)[1],
        'contraction_radius_mean': rho.mean().item(),
        'contraction_radius_max': rho.max().item(),
        'precond_rank': pst['precond_rank'],
        'eta_mean': pst['eta_mean'],
        'proj_rank': float(torch.linalg.matrix_rank(cov_proj.float()).item()),
        'proj_max': evals_proj.max().item(),
        'proj_min_nonzero': evals_proj[evals_proj > 1e-7].min().item() if (evals_proj > 1e-7).any() else 0.0,
        'capacity_rank': float(run.heads * run.d_head),
        'capacity_ge_K': int(run.heads * run.d_head >= K),
        'first_grad_mse': layer_grad[0],
        'last_grad_mse': layer_grad[-1],
        'attn_rank_first': ranks[0],
        'attn_rank_last': ranks[-1],
        'attn_effrank_first': effs[0],
        'attn_effrank_last': effs[-1],
        'layer_first_beta_mse_post': layer_post[0],
        'layer_last_beta_mse_post': layer_post[-1],
        'layer_decay_ratio': layer_post[-1] / max(layer_post[0], 1e-30),
    }


def eval_once(task: TaskCfg, run: RunCfg, device) -> Dict:
    dtype = torch.float64 if task.dtype == 'float64' else torch.float32
    Ps = make_heads(task.K, run.heads, run.d_head, run.head_scheme, device, dtype, run.seed)
    rows = []
    for _ in range(run.eval_batches):
        rows.append(run_loop(sample_batch(run.batch_size, task, device), task, run, Ps))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def row_base(task: TaskCfg, run: RunCfg, extra: Dict, tag: str) -> Dict:
    row = {
        'tag': tag, 'K': task.K, 'prompt_len': task.prompt_len, 'design': task.design, 'cond': task.cond,
        'noise_var': task.noise_var, 'prior_var': task.prior_var, 'depth': run.depth, 'heads': run.heads,
        'd_head': run.d_head, 'capacity_rank': run.heads * run.d_head, 'attention': run.attention,
        'head_scheme': run.head_scheme, 'precond': run.precond, 'temperature': run.temperature,
        'eta_multiplier': run.eta_multiplier, 'scale_linear_by_m': run.scale_linear_by_m, 'seed': run.seed,
        'batch_size': run.batch_size, 'eval_batches': run.eval_batches,
    }
    row.update(extra)
    row['log10_beta_mse_post'] = math.log10(max(row['beta_mse_post'], 1e-30))
    row['log10_beta_mse_true'] = math.log10(max(row['beta_mse_true'], 1e-30))
    return row


def run_save(task, run, csv_path, device, tag):
    row = row_base(task, run, eval_once(task, run, device), tag)
    append_csv(csv_path, row)
    print(pd.Series(row).to_string())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='single', choices=['smoke','single','sweep_depth','sweep_prompt','sweep_capacity','sweep_precond'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir', default='runs_constructive_attention_richardson')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--prompt-len', type=int, default=128)
    ap.add_argument('--noise-var', type=float, default=0.02)
    ap.add_argument('--prior-var', type=float, default=1.0)
    ap.add_argument('--design', default='isotropic', choices=['isotropic','correlated','spiked'])
    ap.add_argument('--cond', type=float, default=10.0)
    ap.add_argument('--dtype', default='float32', choices=['float32','float64'])
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--heads', type=int, default=1)
    ap.add_argument('--d-head', type=int, default=16)
    ap.add_argument('--attention', default='linear', choices=['linear','softmax'])
    ap.add_argument('--head-scheme', default='coordinate', choices=['coordinate','cyclic_coordinate','random_orthogonal'])
    ap.add_argument('--precond', default='scalar_opt', choices=['scalar_opt','scalar_lmax','jacobi','lowrank_spectral','spectral_full'])
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--eta-multiplier', type=float, default=1.0)
    ap.add_argument('--scale-linear-by-m', action='store_true')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--eval-batches', type=int, default=8)
    ap.add_argument('--depth-grid', default='1,2,4,8,16,32,64')
    ap.add_argument('--prompt-grid', default='16,32,64,128,256,512')
    ap.add_argument('--K-grid', default='8,16,32')
    ap.add_argument('--heads-grid', default='1,2,4,8')
    ap.add_argument('--d-head-grid', default='1,2,4,8,16,32')
    ap.add_argument('--precond-grid', default='scalar_opt,scalar_lmax,jacobi,lowrank_spectral,spectral_full')
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    outdir = ensure_dir(args.outdir); csv_path = outdir / 'results.csv'

    def task(K=None, prompt=None):
        return TaskCfg(K=args.K if K is None else K, prompt_len=args.prompt_len if prompt is None else prompt,
                       prior_var=args.prior_var, noise_var=args.noise_var, design=args.design, cond=args.cond,
                       dtype=args.dtype)
    def run(depth=None, heads=None, d_head=None, precond=None):
        return RunCfg(depth=args.depth if depth is None else depth, heads=args.heads if heads is None else heads,
                      d_head=args.d_head if d_head is None else d_head, attention=args.attention,
                      head_scheme=args.head_scheme, precond=args.precond if precond is None else precond,
                      temperature=args.temperature, eta_multiplier=args.eta_multiplier,
                      scale_linear_by_m=args.scale_linear_by_m, batch_size=args.batch_size,
                      eval_batches=args.eval_batches, seed=args.seed)

    if args.mode == 'smoke':
        run_save(task(K=8, prompt=64), run(depth=8, heads=1, d_head=8), csv_path, device, 'smoke'); return
    if args.mode == 'single':
        run_save(task(), run(), csv_path, device, 'single'); return
    if args.mode == 'sweep_depth':
        for d in parse_grid(args.depth_grid, int):
            run_save(task(), run(depth=d), csv_path, device, f'depth_{d}')
        return
    if args.mode == 'sweep_prompt':
        for m in parse_grid(args.prompt_grid, int):
            run_save(task(prompt=m), run(), csv_path, device, f'prompt_{m}')
        return
    if args.mode == 'sweep_capacity':
        for K in parse_grid(args.K_grid, int):
            for H in parse_grid(args.heads_grid, int):
                for dh in parse_grid(args.d_head_grid, int):
                    run_save(task(K=K), run(heads=H, d_head=dh), csv_path, device, f'K{K}_H{H}_dh{dh}')
        if plt is not None:
            df = pd.read_csv(csv_path)
            for K, sub in df.groupby('K'):
                plt.figure(figsize=(7,5))
                for H, ss in sub.groupby('heads'):
                    ss = ss.sort_values('capacity_rank')
                    plt.plot(ss['capacity_rank'], ss['beta_mse_post'], marker='o', label=f'H={H}')
                plt.axvline(K, color='k', ls='--', alpha=.4)
                plt.yscale('log'); plt.xlabel('capacity_rank=H*d_head'); plt.ylabel('beta_mse_post')
                plt.title(f'Q/K/V attention Richardson capacity K={int(K)}'); plt.grid(True, alpha=.3); plt.legend()
                plt.tight_layout(); plt.savefig(outdir / f'capacity_K{int(K)}.png', dpi=160); plt.close()
        return
    if args.mode == 'sweep_precond':
        for pc in [x for x in args.precond_grid.split(',') if x]:
            run_save(task(), run(precond=pc), csv_path, device, f'precond_{pc}')
        return

if __name__ == '__main__':
    main()
