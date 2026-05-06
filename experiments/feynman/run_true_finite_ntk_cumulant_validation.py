#!/usr/bin/env python3
"""
True finite-network cumulant validation for the low-rank NTK paper.

This script is intentionally heavier than the quick NTK-audit script. It computes
empirical NTKs J J^T by PyTorch autograd on actual finite low-rank networks and
estimates second cumulant blocks across random initializations:

  V  ~ Cov(K_NNGP, K_NNGP)
  C  ~ Cov(K_NNGP, Theta_NTK)      (proxy for D/F blocks)
  A  ~ Cov(Theta_NTK, Theta_NTK)   (proxy for A/B blocks)

Here K_NNGP is the final hidden feature Gram h_L h_L^T / n and Theta_NTK is the
empirical NTK with respect to the trainable right factors B and final readout a.

The main purpose is NOT to prove the exact universal constants from the proxy
model inside a small finite network. Instead it tests the true-network scalings

  V = O(epsilon L),    C = O(epsilon L^2),    A = O(epsilon L^3),

where epsilon = 1/n + gamma_{n,r}. It also reports normalized ratios and log-log
slopes in L.

Examples
--------
Quick sanity run:
    python run_true_finite_ntk_cumulant_validation.py --quick

Longer paper-style run:
    python run_true_finite_ntk_cumulant_validation.py \
      --n 128 --m 8 --d 16 \
      --L-values 2,3,4,5,6,8,10,12 \
      --r-values 8,16,32,64,128 \
      --reps 500 \
      --outdir true_finite_ntk_cumulants_long

Very long run, no plots until the end:
    python run_true_finite_ntk_cumulant_validation.py \
      --n 128 --m 8 --d 16 --L-values 2,3,4,5,6,8,10,12 \
      --r-values 8,16,32,64,128 --reps 2000 --no-plots \
      --outdir true_finite_ntk_cumulants_very_long
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Utilities and plotting
# -----------------------------------------------------------------------------


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    out: List[int] = []
    for part in str(s).split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted(set(int(v) for v in values))


def set_paper_style() -> None:
    # Matplotlib will fall back gracefully if some fonts are unavailable.
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


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device.type if device.type != 'mps' else 'cpu')
    gen.manual_seed(int(seed))
    return gen


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-30 else float('nan')


# -----------------------------------------------------------------------------
# Network definition
# -----------------------------------------------------------------------------


@dataclass
class FactorizedParams:
    U: List[torch.Tensor]      # frozen n x r matrices, no grad
    B: List[torch.Tensor]      # trainable r x input_dim matrices
    a: torch.Tensor            # trainable readout vector
    scales: List[float]


def orthonormal_columns(n: int, r: int, *, generator: torch.Generator,
                        dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    a = torch.randn(n, r, generator=generator, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(a, mode='reduced')
    return q[:, :r].contiguous()


def init_factorized(d: int, n: int, r: int, L: int, seed: int,
                    dtype: torch.dtype, device: torch.device) -> FactorizedParams:
    """Signed/isometric low-rank network.

    Layer map: h -> ReLU(h @ (scale * U B)^T), with
    scale = sqrt(2/in_dim) * sqrt(n/r).
    If r=n and U is square orthogonal, this is an isometric coordinate change
    from full MLP parameters to B coordinates.
    """
    gen = make_generator(seed, device)
    U_list: List[torch.Tensor] = []
    B_list: List[torch.Tensor] = []
    scales: List[float] = []
    in_dim = d
    for _ in range(L):
        U = orthonormal_columns(n, r, generator=gen, dtype=dtype, device=device)
        B = torch.randn(r, in_dim, generator=gen, dtype=dtype, device=device)
        B.requires_grad_(True)
        scale = math.sqrt(2.0 / in_dim) * math.sqrt(n / r)
        U_list.append(U.detach())
        B_list.append(B)
        scales.append(scale)
        in_dim = n
    a = torch.randn(n, generator=gen, dtype=dtype, device=device) / math.sqrt(n)
    a.requires_grad_(True)
    return FactorizedParams(U=U_list, B=B_list, a=a, scales=scales)


def forward_factorized_hidden(p: FactorizedParams, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h = X
    for U, B, scale in zip(p.U, p.B, p.scales):
        W = scale * (U @ B)
        h = torch.relu(h @ W.T)
    out = h @ p.a
    return out, h


def flatten_grads(grads: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    pieces = [g.reshape(-1) for g in grads if g is not None]
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def empirical_ntk(outputs: torch.Tensor, params: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute empirical scalar-output NTK K_ij=<grad f_i, grad f_j>."""
    rows: List[torch.Tensor] = []
    for i in range(outputs.shape[0]):
        grads = torch.autograd.grad(
            outputs[i], list(params),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        rows.append(flatten_grads(grads))
    J = torch.stack(rows, dim=0)
    return J @ J.T


def compute_feature_gram_and_ntk(p: FactorizedParams, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    out, h = forward_factorized_hidden(p, X)
    # Output NNGP kernel induced by the final readout variance a_i ~ N(0, 1/n).
    K_feat = (h.detach() @ h.detach().T) / h.shape[1]
    Theta = empirical_ntk(out, list(p.B) + [p.a])
    return K_feat.detach().cpu().numpy(), Theta.detach().cpu().numpy()


def make_dataset(m: int, d: int, seed: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    gen = make_generator(seed, device)
    X = torch.randn(m, d, generator=gen, dtype=dtype, device=device)
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


# -----------------------------------------------------------------------------
# Cumulant estimators
# -----------------------------------------------------------------------------

Entry = Tuple[int, int]


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2:
        return float('nan')
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def entry_values(stack: np.ndarray, entry: Entry) -> np.ndarray:
    i, j = entry
    return stack[:, i, j]


def all_diag_entries(m: int) -> List[Entry]:
    return [(i, i) for i in range(m)]


def all_off_entries(m: int) -> List[Entry]:
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def disjoint_off_entry_pairs(m: int) -> List[Tuple[Entry, Entry]]:
    offs = all_off_entries(m)
    out: List[Tuple[Entry, Entry]] = []
    for a_idx, e in enumerate(offs):
        se = set(e)
        for f in offs[a_idx + 1:]:
            if se.isdisjoint(set(f)):
                out.append((e, f))
    return out


def mean_cov_same_entries(X: np.ndarray, Y: np.ndarray, entries: List[Entry]) -> float:
    vals: List[float] = []
    for e in entries:
        vals.append(covariance(entry_values(X, e), entry_values(Y, e)))
    return float(np.mean(vals)) if vals else float('nan')


def mean_cov_entry_pairs(X: np.ndarray, Y: np.ndarray, pairs: List[Tuple[Entry, Entry]]) -> float:
    vals: List[float] = []
    for e, f in pairs:
        vals.append(covariance(entry_values(X, e), entry_values(Y, f)))
    return float(np.mean(vals)) if vals else float('nan')


def pairing_contractions(Kstack: np.ndarray, Tstack: np.ndarray) -> Dict[str, float]:
    """Three fixed sample pairings on indices 0,1,2,3 if available.

    These are useful for the tensor-pairing audit:
      12|34, 13|24, 14|23.
    """
    m = Kstack.shape[1]
    result: Dict[str, float] = {}
    if m < 4:
        return result
    pairings = {
        '12_34': ((0, 1), (2, 3)),
        '13_24': ((0, 2), (1, 3)),
        '14_23': ((0, 3), (1, 2)),
    }
    for name, (e, f) in pairings.items():
        result[f'V_pair_{name}'] = covariance(entry_values(Kstack, e), entry_values(Kstack, f))
        result[f'C_pair_{name}'] = covariance(entry_values(Kstack, e), entry_values(Tstack, f))
        result[f'A_pair_{name}'] = covariance(entry_values(Tstack, e), entry_values(Tstack, f))
    return result


def summarize_cumulants(Kstack: np.ndarray, Tstack: np.ndarray, eps: float, L: int) -> Dict[str, float]:
    m = Kstack.shape[1]
    diag_entries = all_diag_entries(m)
    off_entries = all_off_entries(m)
    disjoint_pairs = disjoint_off_entry_pairs(m)

    V_diag = mean_cov_same_entries(Kstack, Kstack, diag_entries)
    C_diag = mean_cov_same_entries(Kstack, Tstack, diag_entries)
    A_diag = mean_cov_same_entries(Tstack, Tstack, diag_entries)

    V_off = mean_cov_same_entries(Kstack, Kstack, off_entries)
    C_off = mean_cov_same_entries(Kstack, Tstack, off_entries)
    A_off = mean_cov_same_entries(Tstack, Tstack, off_entries)

    V_dis = mean_cov_entry_pairs(Kstack, Kstack, disjoint_pairs)
    C_dis = mean_cov_entry_pairs(Kstack, Tstack, disjoint_pairs)
    A_dis = mean_cov_entry_pairs(Tstack, Tstack, disjoint_pairs)

    row: Dict[str, float] = {
        'V_same_diag': V_diag,
        'C_same_diag': C_diag,
        'A_same_diag': A_diag,
        'V_same_off': V_off,
        'C_same_off': C_off,
        'A_same_off': A_off,
        'V_disjoint_off': V_dis,
        'C_disjoint_off': C_dis,
        'A_disjoint_off': A_dis,
        'V_same_off_over_eps_L': safe_div(V_off, eps * L),
        'C_same_off_over_eps_L2': safe_div(C_off, eps * (L ** 2)),
        'A_same_off_over_eps_L3': safe_div(A_off, eps * (L ** 3)),
        'abs_V_same_off_over_eps_L': safe_div(abs(V_off), eps * L),
        'abs_C_same_off_over_eps_L2': safe_div(abs(C_off), eps * (L ** 2)),
        'abs_A_same_off_over_eps_L3': safe_div(abs(A_off), eps * (L ** 3)),
        'V_disjoint_off_over_eps_L': safe_div(V_dis, eps * L),
        'C_disjoint_off_over_eps_L2': safe_div(C_dis, eps * (L ** 2)),
        'A_disjoint_off_over_eps_L3': safe_div(A_dis, eps * (L ** 3)),
    }
    row.update(pairing_contractions(Kstack, Tstack))
    return row


def fit_slope(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (np.abs(y) > 0)
    x = x[mask]
    y = np.abs(y[mask])
    if x.size < 2:
        return float('nan'), float('nan')
    coeff = np.polyfit(np.log(x), np.log(y), deg=1)
    slope, intercept = float(coeff[0]), float(coeff[1])
    return slope, intercept


# -----------------------------------------------------------------------------
# Main experiment routines
# -----------------------------------------------------------------------------


def run_cumulants(
    outdir: Path,
    d: int,
    n: int,
    ranks: List[int],
    L_values: List[int],
    m: int,
    reps: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    save_raw: bool,
    progress_every: int,
) -> List[dict]:
    X = make_dataset(m, d, seed + 4000, dtype, device)
    rows: List[dict] = []
    t0 = time.time()

    for r in ranks:
        eps = epsilon_nr(n, r)
        gam = gamma_defect(n, r)
        for L in L_values:
            K_list: List[np.ndarray] = []
            T_list: List[np.ndarray] = []
            combo_start = time.time()
            for s in range(reps):
                p = init_factorized(d, n, r, L, seed + 1_000_000 * r + 10_000 * L + s, dtype, device)
                K, T = compute_feature_gram_and_ntk(p, X)
                K_list.append(K)
                T_list.append(T)
                if progress_every > 0 and (s + 1) % progress_every == 0:
                    elapsed = time.time() - combo_start
                    print(f'  r={r:4d} L={L:3d}: rep {s+1:5d}/{reps} elapsed={elapsed:8.1f}s')

            Kstack = np.stack(K_list, axis=0)
            Tstack = np.stack(T_list, axis=0)
            summary = summarize_cumulants(Kstack, Tstack, eps, L)
            row = {
                'n': n, 'r': r, 'L': L, 'm': m, 'reps': reps,
                'epsilon': eps, 'gamma': gam,
            }
            row.update(summary)
            rows.append(row)

            if save_raw:
                np.savez_compressed(
                    outdir / f'raw_KTheta_n{n}_r{r}_L{L}_m{m}_reps{reps}.npz',
                    Kstack=Kstack,
                    Thetastack=Tstack,
                )
            write_csv(outdir / 'true_finite_ntk_cumulant_summary.csv', rows)
            print(f'DONE r={r} L={L}: V/(eps L)={row["V_same_off_over_eps_L"]:.4g}, '
                  f'C/(eps L^2)={row["C_same_off_over_eps_L2"]:.4g}, '
                  f'A/(eps L^3)={row["A_same_off_over_eps_L3"]:.4g}')

    print(f'Total elapsed: {time.time() - t0:.1f}s')
    return rows


def compute_slope_rows(rows: List[dict], ranks: List[int]) -> List[dict]:
    out: List[dict] = []
    keys = [
        'V_same_off', 'C_same_off', 'A_same_off',
        'V_disjoint_off', 'C_disjoint_off', 'A_disjoint_off',
    ]
    for r in ranks:
        sub = sorted([row for row in rows if row['r'] == r], key=lambda z: z['L'])
        Ls = [row['L'] for row in sub]
        row_out: Dict[str, float] = {'r': r, 'num_depths': len(Ls)}
        for key in keys:
            slope, intercept = fit_slope(Ls, [row.get(key, float('nan')) for row in sub])
            row_out[f'slope_{key}_vs_L'] = slope
            row_out[f'intercept_{key}_vs_L'] = intercept
        out.append(row_out)
    return out


def make_plots(outdir: Path, rows: List[dict], ranks: List[int]) -> None:
    if not rows:
        return
    set_paper_style()

    # Normalized ratios versus depth.
    ratio_specs = [
        ('V_same_off_over_eps_L', r'$V/(\epsilon L)$'),
        ('C_same_off_over_eps_L2', r'$C/(\epsilon L^2)$'),
        ('A_same_off_over_eps_L3', r'$A/(\epsilon L^3)$'),
    ]
    for key, ylabel in ratio_specs:
        fig, ax = plt.subplots()
        for r in ranks:
            sub = sorted([row for row in rows if row['r'] == r], key=lambda z: z['L'])
            if not sub:
                continue
            ax.plot([row['L'] for row in sub], [row[key] for row in sub], marker='o', label=fr'$r={r}$')
        ax.axhline(0.0, lw=0.8)
        ax.set_xlabel(r'depth $L$')
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f'cumulant_ratio_{key}.pdf')
        fig.savefig(outdir / f'cumulant_ratio_{key}.png')
        plt.close(fig)

    # Log-log raw magnitudes versus depth.
    raw_specs = [
        ('V_same_off', r'$|V|$'),
        ('C_same_off', r'$|C|$'),
        ('A_same_off', r'$|A|$'),
    ]
    for key, ylabel in raw_specs:
        fig, ax = plt.subplots()
        for r in ranks:
            sub = sorted([row for row in rows if row['r'] == r], key=lambda z: z['L'])
            xs = np.array([row['L'] for row in sub], dtype=float)
            ys = np.abs(np.array([row[key] for row in sub], dtype=float))
            mask = np.isfinite(ys) & (ys > 0)
            if np.sum(mask) < 2:
                continue
            ax.loglog(xs[mask], ys[mask], marker='o', label=fr'$r={r}$')
        ax.set_xlabel(r'depth $L$')
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f'cumulant_loglog_{key}.pdf')
        fig.savefig(outdir / f'cumulant_loglog_{key}.png')
        plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='True finite-network NTK cumulant validation.')
    parser.add_argument('--outdir', type=str, default='true_finite_ntk_cumulant_outputs')
    parser.add_argument('--quick', action='store_true', help='Small, fast sanity run.')
    parser.add_argument('--d', type=int, default=12)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--m', type=int, default=8)
    parser.add_argument('--reps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--float32', action='store_true', help='Use float32 instead of float64.')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda if available.')
    parser.add_argument('--no-plots', action='store_true', help='Only write CSVs, no plots.')
    parser.add_argument('--save-raw', action='store_true', help='Save raw K/Theta stacks as compressed NPZ files.')
    parser.add_argument('--progress-every', type=int, default=50)
    parser.add_argument('--L-values', type=str, default=None, help='Comma-separated depths, e.g. 2,3,4,5,6,8,10')
    parser.add_argument('--r-values', type=str, default=None, help='Comma-separated ranks, e.g. 8,16,32,64')
    args = parser.parse_args()

    if args.n < 1 or args.d < 1 or args.m < 2:
        raise ValueError('--n and --d must be positive, and --m must be at least 2')

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: CUDA requested but not available; falling back to CPU.')
        args.device = 'cpu'
    device = torch.device(args.device)
    dtype = torch.float32 if args.float32 else torch.float64
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    L_values = parse_int_list(args.L_values)
    ranks = parse_int_list(args.r_values)

    if args.quick:
        if L_values is None:
            L_values = [2, 3]
        if ranks is None:
            ranks = [max(1, args.n // 2), args.n]
        reps = min(args.reps, 8)
        m = min(args.m, 6)
    else:
        if L_values is None:
            L_values = [2, 3, 4, 5, 6, 8, 10]
        if ranks is None:
            ranks = [max(1, args.n // 8), max(1, args.n // 4), max(1, args.n // 2), args.n]
        reps = args.reps
        m = args.m

    L_values = unique_sorted([L for L in L_values if L >= 1])
    ranks = unique_sorted([r for r in ranks if 1 <= r <= args.n])
    if not L_values:
        raise ValueError('No valid L values provided.')
    if not ranks:
        raise ValueError('No valid ranks provided; each rank must satisfy 1 <= r <= n.')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('Running true finite-network NTK cumulant validation')
    print(f'  outdir={outdir}')
    print(f'  d={args.d}, n={args.n}, m={m}, reps={reps}, dtype={dtype}, device={device}')
    print(f'  L_values={L_values}')
    print(f'  ranks={ranks}')
    print('  measured blocks: V=Cov(K,K), C=Cov(K,Theta), A=Cov(Theta,Theta)')

    rows = run_cumulants(
        outdir=outdir,
        d=args.d,
        n=args.n,
        ranks=ranks,
        L_values=L_values,
        m=m,
        reps=reps,
        seed=args.seed,
        dtype=dtype,
        device=device,
        save_raw=args.save_raw,
        progress_every=args.progress_every,
    )
    slope_rows = compute_slope_rows(rows, ranks)
    write_csv(outdir / 'true_finite_ntk_cumulant_slopes.csv', slope_rows)

    if not args.no_plots:
        make_plots(outdir, rows, ranks)

    with open(outdir / 'README.txt', 'w') as f:
        f.write('True finite-network NTK cumulant validation outputs.\n')
        f.write('This is a heavy autodiff experiment on actual low-rank finite networks.\n')
        f.write('K_NNGP = final hidden feature Gram h_L h_L^T / n.\n')
        f.write('Theta_NTK = empirical NTK with respect to trainable right factors B and readout a.\n')
        f.write('Main CSV files:\n')
        f.write('  true_finite_ntk_cumulant_summary.csv\n')
        f.write('  true_finite_ntk_cumulant_slopes.csv\n')
        f.write('Expected scaling, not exact finite-network constants:\n')
        f.write('  V = O(epsilon L), C = O(epsilon L^2), A = O(epsilon L^3).\n')
        f.write('Use large reps and n for smoother ratios.\n')

    print(f'Wrote outputs to {outdir}')


if __name__ == '__main__':
    main()
