#!/usr/bin/env python3
"""
True finite-network NTK validation for the low-rank NTK paper.

This script computes empirical NTKs J J^T by PyTorch autograd on actual
finite networks, not just proxy recurrences.

It checks:
  1) r=n pathwise audit: isometric factorized coordinates and full MLP
     coordinates have identical empirical NTK up to numerical precision.
  2) operator fluctuation scaling of true finite-network NTKs.
  3) rank-defect scaling through epsilon_{n,r}=1/n+gamma_{n,r}.

Examples
--------
Quick sanity run:
    python run_true_finite_ntk_validation_fixed.py --quick

Cleaner run:
    python run_true_finite_ntk_validation_fixed.py \
      --n 64 --m 8 --d 12 \
      --L-values 1,2,3,4,5,6,8 \
      --r-values 4,8,16,32,64 \
      --reps 50 \
      --outdir true_finite_ntk_outputs
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    out: List[int] = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted(set(int(v) for v in values))


def set_paper_style() -> None:
    # The font family may not exist on all systems; matplotlib will fallback.
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.weight'] = 'normal'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 22
    mpl.rcParams['axes.formatter.limits'] = (-6, 6)
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True


def gamma_defect(n: int, r: int) -> float:
    """Exact Haar-projector rank defect gamma_{n,r}."""
    if r == n:
        return 0.0
    if not (1 <= r <= n):
        raise ValueError(f"rank r must satisfy 1 <= r <= n; got r={r}, n={n}")
    return n * (n - r) / (r * (n - 1) * (n + 2))


def epsilon_nr(n: int, r: int) -> float:
    return 1.0 / n + gamma_defect(n, r)


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    return gen


def orthonormal_columns(n: int, r: int, *, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(n, r, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(a, mode='reduced')
    return q[:, :r].contiguous()


@dataclass
class FactorizedParams:
    U: List[torch.Tensor]      # frozen n x r matrices, no grad
    B: List[torch.Tensor]      # trainable r x input_dim matrices
    a: torch.Tensor            # trainable readout vector
    scales: List[float]


@dataclass
class FullParams:
    Wraw: List[torch.Tensor]   # trainable raw full matrices
    a: torch.Tensor
    scales: List[float]


def init_factorized(d: int, n: int, r: int, L: int, seed: int, dtype: torch.dtype) -> FactorizedParams:
    """Initialize signed/isometric low-rank network.

    Layer map: h -> ReLU(h @ (scale * U B)^T), with
    scale = sqrt(2/in_dim) * sqrt(n/r).
    If r=n and U is square orthogonal, B -> U B is an isometry and the
    network is exactly a full-width He MLP in reparameterized coordinates.
    """
    gen = make_generator(seed)
    U_list: List[torch.Tensor] = []
    B_list: List[torch.Tensor] = []
    scales: List[float] = []
    in_dim = d
    for _ in range(L):
        U = orthonormal_columns(n, r, generator=gen, dtype=dtype)
        B = torch.randn(r, in_dim, generator=gen, dtype=dtype)
        B.requires_grad_(True)
        scale = math.sqrt(2.0 / in_dim) * math.sqrt(n / r)
        U_list.append(U.detach())
        B_list.append(B)
        scales.append(scale)
        in_dim = n
    a = torch.randn(n, generator=gen, dtype=dtype) / math.sqrt(n)
    a.requires_grad_(True)
    return FactorizedParams(U=U_list, B=B_list, a=a, scales=scales)


def factorized_to_full(p: FactorizedParams) -> FullParams:
    """Convert factorized coordinates to raw full coordinates Wraw = U B.

    For r=n this is an orthogonal coordinate change and should preserve the NTK.
    """
    Wraw_list: List[torch.Tensor] = []
    for U, B in zip(p.U, p.B):
        Wraw = (U @ B).detach().clone()
        Wraw.requires_grad_(True)
        Wraw_list.append(Wraw)
    a = p.a.detach().clone()
    a.requires_grad_(True)
    return FullParams(Wraw=Wraw_list, a=a, scales=list(p.scales))


def forward_factorized(p: FactorizedParams, X: torch.Tensor) -> torch.Tensor:
    h = X
    for U, B, scale in zip(p.U, p.B, p.scales):
        W = scale * (U @ B)
        h = torch.relu(h @ W.T)
    return h @ p.a


def forward_full(p: FullParams, X: torch.Tensor) -> torch.Tensor:
    h = X
    for Wraw, scale in zip(p.Wraw, p.scales):
        h = torch.relu(h @ (scale * Wraw).T)
    return h @ p.a


def flatten_grads(grads: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def empirical_ntk(outputs: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
    """Compute empirical scalar-output NTK K_ij=<grad f_i, grad f_j>."""
    rows: List[torch.Tensor] = []
    for i in range(outputs.shape[0]):
        grads = torch.autograd.grad(
            outputs[i], params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        rows.append(flatten_grads(grads))
    J = torch.stack(rows, dim=0)
    return J @ J.T


def compute_factorized_ntk(p: FactorizedParams, X: torch.Tensor) -> torch.Tensor:
    return empirical_ntk(forward_factorized(p, X), list(p.B) + [p.a])


def compute_full_ntk(p: FullParams, X: torch.Tensor) -> torch.Tensor:
    return empirical_ntk(forward_full(p, X), list(p.Wraw) + [p.a])


def op_norm_symmetric(A: np.ndarray) -> float:
    vals = np.linalg.eigvalsh((A + A.T) / 2.0)
    return float(np.max(np.abs(vals)))


def make_dataset(m: int, d: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    gen = make_generator(seed)
    X = torch.randn(m, d, generator=gen, dtype=dtype)
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_matching(outdir: Path, d: int, n: int, L_values: List[int], m: int, seed: int, dtype: torch.dtype) -> None:
    X = make_dataset(m, d, seed + 1000, dtype)
    rows: List[dict] = []
    for L in L_values:
        p_lr = init_factorized(d, n, n, L, seed + 10 * L, dtype)
        p_full = factorized_to_full(p_lr)
        K_lr = compute_factorized_ntk(p_lr, X).detach().cpu().numpy()
        K_full = compute_full_ntk(p_full, X).detach().cpu().numpy()
        max_abs = float(np.max(np.abs(K_lr - K_full)))
        rel = max_abs / max(1e-12, float(np.max(np.abs(K_full))))
        rows.append({
            'L': L, 'n': n, 'r': n, 'm': m,
            'max_abs_ntk_difference': max_abs,
            'relative_difference': rel,
        })
    write_csv(outdir / 'true_ntk_r_equals_n_matching.csv', rows)


def run_operator_scaling(
    outdir: Path, d: int, n: int, ranks: List[int], L_values: List[int],
    m: int, reps: int, seed: int, dtype: torch.dtype, make_plots: bool,
) -> None:
    X = make_dataset(m, d, seed + 2000, dtype)
    rows: List[dict] = []
    for r in ranks:
        eps = epsilon_nr(n, r)
        for L in L_values:
            Ks: List[np.ndarray] = []
            for s in range(reps):
                p = init_factorized(d, n, r, L, seed + 10000 * r + 100 * L + s, dtype)
                Ks.append(compute_factorized_ntk(p, X).detach().cpu().numpy())
            Kstack = np.stack(Ks, axis=0)
            Kmean = Kstack.mean(axis=0)
            opvals = np.array([op_norm_symmetric(K - Kmean) for K in Kstack])
            median_op = float(np.median(opvals))
            mean_op = float(np.mean(opvals))
            scale = (L ** 1.5) * math.sqrt(m * eps)
            rows.append({
                'n': n, 'r': r, 'L': L, 'm': m, 'reps': reps,
                'epsilon': eps,
                'median_op_fluctuation': median_op,
                'mean_op_fluctuation': mean_op,
                'scale_L32_sqrt_m_eps': scale,
                'median_ratio': median_op / max(scale, 1e-12),
                'mean_ratio': mean_op / max(scale, 1e-12),
            })
    write_csv(outdir / 'true_finite_ntk_operator_scaling.csv', rows)

    if make_plots and rows:
        set_paper_style()
        fig, ax = plt.subplots()
        for r in ranks:
            xs = [row['L'] for row in rows if row['r'] == r]
            ys = [row['median_ratio'] for row in rows if row['r'] == r]
            ax.plot(xs, ys, marker='o', label=fr'$r={r}$')
        ax.set_xlabel(r'depth $L$')
        ax.set_ylabel(r'median $\|K-\bar K\|_{\mathrm{op}}/(L^{3/2}\sqrt{m\epsilon})$')
        ax.legend(frameon=False, fontsize=14)
        fig.tight_layout()
        fig.savefig(outdir / 'true_finite_ntk_operator_ratio.pdf')
        fig.savefig(outdir / 'true_finite_ntk_operator_ratio.png')
        plt.close(fig)


def run_rank_defect(
    outdir: Path, d: int, n: int, ranks: List[int], L: int,
    m: int, reps: int, seed: int, dtype: torch.dtype, make_plots: bool,
) -> None:
    X = make_dataset(m, d, seed + 3000, dtype)
    rows: List[dict] = []
    for r in ranks:
        eps = epsilon_nr(n, r)
        Ks: List[np.ndarray] = []
        for s in range(reps):
            p = init_factorized(d, n, r, L, seed + 777 * r + s, dtype)
            Ks.append(compute_factorized_ntk(p, X).detach().cpu().numpy())
        Kstack = np.stack(Ks, axis=0)
        Kmean = Kstack.mean(axis=0)
        opvals = np.array([op_norm_symmetric(K - Kmean) for K in Kstack])
        median_op = float(np.median(opvals))
        scale = (L ** 1.5) * math.sqrt(m * eps)
        rows.append({
            'n': n, 'r': r, 'L': L, 'm': m,
            'epsilon': eps, 'gamma': gamma_defect(n, r),
            'median_op_fluctuation': median_op,
            'scale_L32_sqrt_m_eps': scale,
            'ratio': median_op / max(scale, 1e-12),
        })
    write_csv(outdir / 'true_finite_ntk_rank_defect_scaling.csv', rows)

    if make_plots and rows:
        set_paper_style()
        fig, ax = plt.subplots()
        ax.loglog([row['epsilon'] for row in rows], [row['median_op_fluctuation'] for row in rows], marker='o')
        ax.set_xlabel(r'$\epsilon_{n,r}$')
        ax.set_ylabel(r'median $\|K-\bar K\|_{\mathrm{op}}$')
        fig.tight_layout()
        fig.savefig(outdir / 'true_finite_ntk_rank_defect_scaling.pdf')
        fig.savefig(outdir / 'true_finite_ntk_rank_defect_scaling.png')
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='True finite-network empirical NTK validation.')
    parser.add_argument('--outdir', type=str, default='true_finite_ntk_validation_outputs')
    parser.add_argument('--quick', action='store_true', help='Small, fast sanity run.')
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--n', type=int, default=32)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--float32', action='store_true', help='Use float32 instead of float64.')
    parser.add_argument('--no-plots', action='store_true', help='Only write CSVs, no plots.')
    parser.add_argument('--L-values', type=str, default=None, help='Comma-separated depths, e.g. 1,2,3,4,6,8')
    parser.add_argument('--r-values', type=str, default=None, help='Comma-separated ranks, e.g. 4,8,16,32')
    parser.add_argument('--r', type=int, default=None, help='Optional single rank to include in ranks list.')
    args = parser.parse_args()

    if args.n < 1:
        raise ValueError('--n must be positive')
    if args.d < 1 or args.m < 1:
        raise ValueError('--d and --m must be positive')

    dtype = torch.float32 if args.float32 else torch.float64
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    L_values = parse_int_list(args.L_values)
    ranks = parse_int_list(args.r_values)

    if L_values is None:
        L_values = [1, 2, 3, 4] if args.quick else [1, 2, 3, 4, 5, 6]
    if ranks is None:
        if args.quick:
            ranks = [max(1, args.n // 3), max(1, 2 * args.n // 3), args.n]
        else:
            ranks = [max(1, args.n // 8), max(1, args.n // 4), max(1, args.n // 2), args.n]
    if args.r is not None:
        ranks.append(args.r)

    L_values = unique_sorted([L for L in L_values if L >= 1])
    ranks = unique_sorted([r for r in ranks if 1 <= r <= args.n])
    if not L_values:
        raise ValueError('No valid L values provided.')
    if not ranks:
        raise ValueError('No valid ranks provided; each rank must satisfy 1 <= r <= n.')

    reps = min(args.reps, 6) if args.quick else args.reps
    m = min(args.m, 6) if args.quick else args.m

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('Running true finite NTK validation')
    print(f'  outdir={outdir}')
    print(f'  d={args.d}, n={args.n}, m={m}, reps={reps}, dtype={dtype}')
    print(f'  L_values={L_values}')
    print(f'  ranks={ranks}')

    run_matching(outdir, args.d, args.n, L_values, m, args.seed, dtype)
    run_operator_scaling(outdir, args.d, args.n, ranks, L_values, m, reps, args.seed, dtype, not args.no_plots)
    run_rank_defect(outdir, args.d, args.n, ranks, max(L_values), m, reps, args.seed, dtype, not args.no_plots)

    with open(outdir / 'README.txt', 'w') as f:
        f.write('True finite-network empirical NTK validation outputs.\n')
        f.write('Computed by PyTorch autograd from empirical NTK J J^T.\n')
        f.write('CSV files:\n')
        f.write('  true_ntk_r_equals_n_matching.csv\n')
        f.write('  true_finite_ntk_operator_scaling.csv\n')
        f.write('  true_finite_ntk_rank_defect_scaling.csv\n')
        f.write('The r=n matching file should be at numerical precision.\n')
    print(f'Wrote outputs to {outdir}')


if __name__ == '__main__':
    main()
