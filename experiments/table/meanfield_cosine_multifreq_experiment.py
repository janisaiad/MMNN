#!/usr/bin/env python3
"""
we implement mean-field analysis for multi-frequency cosine function
using config from results_multi_frequency_benchmark
we compute log ratios at x=0 only with LaTeX formatting
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")  # we use non-interactive backend
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.integrate import solve_ivp
import sys

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# we configure matplotlib for LaTeX formatting
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

# we optionally enable LaTeX rendering (requires LaTeX installed)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


class MultiFrequencyCosineFunction:
    """we create multi-frequency cosine function from config"""
    def __init__(self):
        """
        we create: cos(12πx) + cos(24πx + 0.5) + cos(36πx) + cos(72πx + 0.5)
        """
        pass
    
    def __call__(self, x):
        """we evaluate the multi-frequency cosine function"""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        result = (np.cos(12 * np.pi * x_np) + 
                 np.cos(24 * np.pi * x_np + 0.5) +
                 np.cos(36 * np.pi * x_np) +
                 np.cos(72 * np.pi * x_np + 0.5))
        
        if isinstance(x, torch.Tensor):
            return torch.tensor(result, dtype=x.dtype, device=x.device)
        return result
    
    def get_training_dataset(self, n_samples=5000, x_range=(-1, 1)):
        """we create training dataset"""
        x_points = np.linspace(x_range[0], x_range[1], n_samples)
        y_points = self(x_points)
        return x_points, y_points


class MeanFieldODESolver:
    """we solve the mean-field ODEs for low-rank networks (2-layer version)"""
    def __init__(self, n1=777, n2=777, r=15, d=1, device="cpu"):
        """
        we initialize mean-field solver
        n1, n2: number of neurons in first and second layers
        r: rank (number of channels)
        d: input dimension
        """
        self.n1 = n1
        self.n2 = n2
        self.r = r
        self.d = d
        self.device = device
        
        # we initialize random features (frozen)
        torch.manual_seed(42)
        self.f1 = torch.randn(n1, d, device=device)  # random features for first layer
        
        # we initialize mixing matrix L (frozen, random Gaussian)
        self.L = torch.randn(n2, r, device=device)  # Gaussian N(0, 1)
        
        # we initialize weights as Gaussian O(1) (not zero)
        self.w1_0 = torch.randn(n1, r, device=device)  # Gaussian N(0, 1) - order 1
        self.w2_0 = torch.randn(n2, device=device)  # Gaussian N(0, 1) - order 1
        
        # we store trajectory
        self.trajectory = []
        self.times = []
    
    def compute_H2(self, w1, w2, X):
        """we compute H2 = sum_k L_{c2,k} m_k where m_k = E_C1[w1(C1,k) phi1(f1(C1), X)]"""
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        
        inner = torch.matmul(self.f1, X.t())  # [n1, batch_size]
        phi1 = torch.relu(inner)  # [n1, batch_size]
        
        m_k = torch.zeros(self.r, batch_size, device=self.device)
        for k in range(self.r):
            w1_k = w1[:, k].unsqueeze(1)  # [n1, 1]
            m_k[k] = torch.mean(w1_k * phi1, dim=0)  # [batch_size]
        
        H2 = torch.zeros(self.n2, batch_size, device=self.device)
        for j2 in range(self.n2):
            H2[j2] = torch.sum(self.L[j2, :].unsqueeze(1) * m_k, dim=0)  # [batch_size]
        
        return H2, m_k
    
    def compute_backprop_signal(self, w2, H2):
        """we compute B_k = E_C2[L_{C2,k} phi2'(H2) w2]"""
        phi2_prime = (H2 > 0).float()  # ReLU derivative
        B_k = torch.zeros(self.r, H2.shape[1], device=self.device)
        for k in range(self.r):
            L_k = self.L[:, k].unsqueeze(1)  # [n2, 1]
            B_k[k] = torch.mean(L_k * phi2_prime * w2.unsqueeze(1), dim=0)  # [batch_size]
        return B_k
    
    def compute_output(self, w1, w2, X):
        """we compute network output y_hat = E_C2[w2(C2) phi2(H2)]"""
        H2, _ = self.compute_H2(w1, w2, X)
        phi2 = torch.relu(H2)  # [n2, batch_size]
        y_hat = torch.mean(w2.unsqueeze(1) * phi2, dim=0)  # [batch_size]
        return y_hat
    
    def ode_rhs(self, t, y, X_data, y_data, xi1=1.0, xi2=1.0):
        """we compute right-hand side of mean-field ODEs"""
        w1_flat = y[:self.n1 * self.r]
        w2_flat = y[self.n1 * self.r:]
        w1 = w1_flat.reshape(self.n1, self.r)
        w2 = w2_flat
        
        w1_t = torch.tensor(w1, device=self.device, dtype=torch.float32)
        w2_t = torch.tensor(w2, device=self.device, dtype=torch.float32)
        X_t = torch.tensor(X_data, device=self.device, dtype=torch.float32)
        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(1)
        y_data_t = torch.tensor(y_data, device=self.device, dtype=torch.float32)
        
        H2, m_k = self.compute_H2(w1_t, w2_t, X_t)
        B_k = self.compute_backprop_signal(w2_t, H2)
        
        y_hat = self.compute_output(w1_t, w2_t, X_t)
        dL = y_hat - y_data_t  # square loss derivative [batch_size]
        
        inner_prod = torch.matmul(self.f1, X_t.t())  # [n1, batch_size]
        phi1_vals = torch.relu(inner_prod)  # [n1, batch_size]
        
        dw1 = torch.zeros_like(w1_t)
        for k in range(self.r):
            grad_k = torch.mean(dL.unsqueeze(0) * phi1_vals * B_k[k].unsqueeze(0), dim=1)  # [n1]
            dw1[:, k] = -xi1 * grad_k
        
        phi2_vals = torch.relu(H2)  # [n2, batch_size]
        dw2 = -xi2 * torch.mean(dL.unsqueeze(0) * phi2_vals, dim=1)  # [n2]
        
        dw1_flat = dw1.cpu().numpy().flatten()
        dw2_flat = dw2.cpu().numpy()
        return np.concatenate([dw1_flat, dw2_flat])
    
    def solve(self, X_data, y_data, t_span=(0, 1000), dt=1.0, xi1=1.0, xi2=1.0):
        """we solve the mean-field ODEs"""
        y0 = np.concatenate([
            self.w1_0.cpu().numpy().flatten(),
            self.w2_0.cpu().numpy()
        ])
        
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        sol = solve_ivp(
            lambda t, y: self.ode_rhs(t, y, X_data, y_data, xi1=xi1, xi2=xi2),
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-5,
            atol=1e-7
        )
        
        self.times = sol.t
        self.trajectory = sol.y.T
        
        return sol
    
    def get_weights_at_time(self, t_idx):
        """we extract weights at a given time index"""
        y = self.trajectory[t_idx]
        w1_flat = y[:self.n1 * self.r]
        w2_flat = y[self.n1 * self.r:]
        w1 = w1_flat.reshape(self.n1, self.r)
        w2 = w2_flat
        return torch.tensor(w1, device=self.device), torch.tensor(w2, device=self.device)
    
    def compute_partial_functions(self, w1, X):
        """we compute f_k(t,x) = m_k(t;x,W) for each channel"""
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        
        inner = torch.matmul(self.f1, X.t())  # [n1, batch_size]
        phi1 = torch.relu(inner)  # [n1, batch_size]
        
        f_k = torch.zeros(self.r, batch_size, device=self.device)
        for k in range(self.r):
            w1_k = w1[:, k].unsqueeze(1)  # [n1, 1]
            f_k[k] = torch.mean(w1_k * phi1, dim=0)  # [batch_size]
        
        return f_k  # returns [r, batch_size]


class ChannelSpecializationMetrics:
    """we compute channel specialization metrics"""
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def compute_channel_shares(self, f_k):
        """we compute channel shares s_k = |f_k| / (sum_j |f_j| + epsilon)"""
        abs_f = torch.abs(f_k)  # [r, batch_size]
        sum_abs = torch.sum(abs_f, dim=0, keepdim=True)  # [1, batch_size]
        shares = abs_f / (sum_abs + self.epsilon)  # [r, batch_size]
        return shares  # [r, batch_size]
    
    def compute_log_ratios(self, f_k):
        """we compute log ratios R_{k,ell} = log(|f_k| + epsilon) - log(|f_ell| + epsilon)"""
        r = f_k.shape[0]
        abs_f = torch.abs(f_k) + self.epsilon  # [r, batch_size]
        log_f = torch.log(abs_f)  # [r, batch_size]
        
        log_ratios = torch.zeros(r, r, f_k.shape[1], device=f_k.device)  # [r, r, batch_size]
        for k in range(r):
            for ell in range(r):
                log_ratios[k, ell] = log_f[k] - log_f[ell]
        
        return log_ratios  # [r, r, batch_size]


def run_experiment():
    """we run the mean-field analysis experiment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # we use config from the attached file
    num_layers = 8  # from config (but we use 2-layer mean-field for now)
    hidden_width = 777  # from config
    hidden_rank = 15  # from config
    num_training_samples = 5000  # from config
    interval = [-1, 1]  # from config
    
    # we create multi-frequency cosine function
    cosine_func = MultiFrequencyCosineFunction()
    
    # we create training dataset
    x_train, y_train = cosine_func.get_training_dataset(n_samples=num_training_samples, x_range=tuple(interval))
    print(f"\nTraining dataset: {len(x_train)} points on {interval}")
    print(f"Target function: cos(12πx) + cos(24πx + 0.5) + cos(36πx) + cos(72πx + 0.5)")
    
    # we set up mean-field solver (2-layer version with config parameters)
    # note: config has 8 layers, but we use 2-layer mean-field approximation
    n1, n2 = hidden_width, hidden_width
    r = hidden_rank
    t_span = (0, 1000)  # we solve for 1000 time units
    dt = 1.0
    
    print(f"\nMean-Field Architecture (2-layer approximation of {num_layers}-layer network):")
    print(f"  Config: {num_layers} layers, width={hidden_width}, rank={hidden_rank}")
    print(f"  Mean-field: Layer 1: n1={n1} neurons, Layer 2: n2={n2} neurons")
    print(f"  Rank: r={r} channels (low-rank bottleneck)")
    print(f"  Time span: {t_span}")
    print(f"  Training samples: {num_training_samples}")
    
    # we solve mean-field ODEs
    print("\n" + "="*80)
    print("Solving Mean-Field ODEs (this will evolve weights during training)...")
    print("="*80)
    mf_solver = MeanFieldODESolver(n1=n1, n2=n2, r=r, d=1, device=device)
    
    # we check initial weights (before solving)
    w1_init = mf_solver.w1_0.clone()
    w2_init = mf_solver.w2_0.clone()
    print(f"Initial weights: w1 shape={w1_init.shape}, w2 shape={w2_init.shape}")
    print(f"  w1 mean={torch.mean(w1_init).item():.6f}, std={torch.std(w1_init).item():.6f}")
    print(f"  w2 mean={torch.mean(w2_init).item():.6f}, std={torch.std(w2_init).item():.6f}")
    
    sol = mf_solver.solve(x_train, y_train, t_span=t_span, dt=dt)
    
    print(f"Mean-field ODE solved. Trajectory shape: {mf_solver.trajectory.shape}")
    
    # we check final weights to verify they evolved
    w1_final, w2_final = mf_solver.get_weights_at_time(-1)
    print(f"Final weights: w1 mean={torch.mean(w1_final).item():.6f}, std={torch.std(w1_final).item():.6f}")
    print(f"  w2 mean={torch.mean(w2_final).item():.6f}, std={torch.std(w2_final).item():.6f}")
    print(f"  Weight change: w1 diff={torch.mean(torch.abs(w1_final - w1_init)).item():.6f}")
    print(f"  Weight change: w2 diff={torch.mean(torch.abs(w2_final - w2_init)).item():.6f}")
    
    # we compute log ratios at x=0 only
    print("\n" + "="*80)
    print("Computing Log Ratios at x=0...")
    print("="*80)
    
    # we use x=0, but note that ReLU(0)=0, so we use a small epsilon to avoid exact zero
    # mathematically, at x=0, f1 @ 0 = 0, so we use x ≈ 0 for numerical stability
    x_analyze = 1e-6  # we use small epsilon near 0
    x_tensor = torch.tensor([[x_analyze]], dtype=torch.float32, device=device)
    print(f"Note: Using x={x_analyze} (small epsilon near 0) because ReLU(0)=0 makes partial functions zero at exact x=0")
    
    spec_metrics = ChannelSpecializationMetrics()
    
    # we compute at different time points
    time_indices = [0, len(mf_solver.times)//4, len(mf_solver.times)//2, -1]
    results = {}
    
    for t_idx in time_indices:
        t = mf_solver.times[t_idx]
        w1_t, w2_t = mf_solver.get_weights_at_time(t_idx)
        
        # we check weight statistics
        w1_mean = torch.mean(torch.abs(w1_t)).item()
        w2_mean = torch.mean(torch.abs(w2_t)).item()
        
        f_k = mf_solver.compute_partial_functions(w1_t, x_tensor)  # [r, 1]
        log_ratios = spec_metrics.compute_log_ratios(f_k)  # [r, r, 1]
        
        results[f'time_{t:.1f}'] = {
            'time': float(t),
            'log_ratios': log_ratios.squeeze(2).detach().cpu().numpy()  # [r, r]
        }
        
        # we compute statistics
        triu_indices = np.triu_indices_from(log_ratios.squeeze(2).detach().cpu().numpy(), k=1)
        all_log_ratios = log_ratios.squeeze(2).detach().cpu().numpy()[triu_indices]
        f_k_vals = f_k.squeeze(1).detach().cpu().numpy()
        
        print(f"\nTime t={t:.1f}:")
        print(f"  Weight stats: |w1|_mean={w1_mean:.6f}, |w2|_mean={w2_mean:.6f}")
        print(f"  Partial functions f_k: min={np.min(f_k_vals):.6f}, max={np.max(f_k_vals):.6f}, mean={np.mean(f_k_vals):.6f}, std={np.std(f_k_vals):.6f}")
        print(f"  Mean log-ratio: {np.mean(all_log_ratios):.6f}")
        print(f"  Max log-ratio: {np.max(all_log_ratios):.6f}")
        print(f"  Min log-ratio: {np.min(all_log_ratios):.6f}")
    
    # we create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations (separate plots)...")
    print("="*80)
    
    output_dir = Path(__file__).parent / "meanfield_cosine_multifreq_results"
    output_dir.mkdir(exist_ok=True)
    
    # we prepare common data
    final_time_key = list(results.keys())[-1]
    log_ratios_final = results[final_time_key]['log_ratios']
    final_time = results[final_time_key]['time']
    w1_final, w2_final = mf_solver.get_weights_at_time(-1)
    w1_flat = w1_final.detach().cpu().numpy().flatten()
    w2_flat = w2_final.detach().cpu().numpy().flatten()
    all_weights = np.concatenate([w1_flat, w2_flat])
    abs_weights = np.abs(all_weights)
    abs_weights = abs_weights[abs_weights > 0]  # we remove zeros for log plot
    
    # we create common info text (moved to bottom to avoid collapsing)
    common_info_text = (f'Mean-Field Analysis: Multi-Frequency Cosine Function. '
                       f'Target: $\\cos(12\\pi x) + \\cos(24\\pi x + 0.5) + \\cos(36\\pi x) + \\cos(72\\pi x + 0.5)$. '
                       f'Architecture: 2-layer mean-field (approximating {num_layers}-layer), width $n={n1}$, rank $r={r}$, '
                       f'training samples $N={num_training_samples}$. '
                       f'Note: Using $x=10^{{-6}}$ because $\\mathrm{{ReLU}}(0)=0$ makes $f_k=0$ at exact $x=0$.')
    
    # we plot 1: log ratio heatmap at final time (separate figure)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    vmax = max(10, np.max(np.abs(log_ratios_final)))
    vmin = -vmax
    im = ax.imshow(log_ratios_final, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Channel $j$', fontsize=24)
    ax.set_ylabel('Channel $i$', fontsize=24)
    ax.set_title(f'Log-Ratio Matrix $R_{{i,j}} = \\log(|f_i|) - \\log(|f_j|)$ at $x \\approx 0$ (Time $t={final_time:.1f}$)', fontsize=22)
    cbar = plt.colorbar(im, ax=ax, label='$R_{i,j}$', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, common_info_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_log_ratio_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_log_ratio_heatmap.png'}")
    plt.close()
    
    # we plot 2: log ratio statistics over time (separate figure)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    times = [results[k]['time'] for k in results.keys()]
    mean_log_ratios = []
    max_log_ratios = []
    min_log_ratios = []
    std_log_ratios = []
    for key in results.keys():
        log_ratios = results[key]['log_ratios']
        triu_indices = np.triu_indices_from(log_ratios, k=1)
        all_log_ratios = log_ratios[triu_indices]
        mean_log_ratios.append(np.mean(all_log_ratios))
        max_log_ratios.append(np.max(all_log_ratios))
        min_log_ratios.append(np.min(all_log_ratios))
        std_log_ratios.append(np.std(all_log_ratios))
    
    ax.plot(times, mean_log_ratios, 'b-o', label='Mean $R_{i,j}$', linewidth=3, markersize=10)
    ax.plot(times, max_log_ratios, 'r-s', label='Max $R_{i,j}$', linewidth=3, markersize=10)
    ax.plot(times, min_log_ratios, 'g-^', label='Min $R_{i,j}$', linewidth=3, markersize=10)
    ax.fill_between(times,
                   np.array(mean_log_ratios) - np.array(std_log_ratios),
                   np.array(mean_log_ratios) + np.array(std_log_ratios),
                   alpha=0.2, label='±1 std', color='blue')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Time $t$', fontsize=24)
    ax.set_ylabel('Log-Ratio $R_{i,j}$', fontsize=24)
    n_pairs = r * (r - 1) // 2
    ax.set_title(f'Log-Ratio Statistics Over Time (at $x \\approx 0$, all ${n_pairs}$ channel pairs)', fontsize=22)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, common_info_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_log_ratio_statistics_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_log_ratio_statistics_time.png'}")
    plt.close()
    
    # we plot 3: log ratio distribution at final time (separate figure)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    log_ratios_final_flat = log_ratios_final[np.triu_indices_from(log_ratios_final, k=1)]
    ax.hist(log_ratios_final_flat, bins=50, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Log-Ratio $R_{i,j}$', fontsize=24)
    ax.set_ylabel('Count', fontsize=24)
    n_pairs = r * (r - 1) // 2
    ax.set_title(f'Log-Ratio Distribution (Final Time $t={final_time:.1f}$, all ${n_pairs}$ pairs, rank $r={r}$)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, common_info_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_log_ratio_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_log_ratio_distribution.png'}")
    plt.close()
    
    # we plot 3b: log ratio distributions across different time points (separate figure)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for key in results.keys():
        t = results[key]['time']
        log_ratios = results[key]['log_ratios']
        triu_indices = np.triu_indices_from(log_ratios, k=1)
        all_log_ratios = log_ratios[triu_indices]
        ax.hist(all_log_ratios, bins=30, alpha=0.5, label=f'$t={t:.1f}$', density=True)
    ax.set_xlabel('Log-Ratio $R_{i,j}$', fontsize=24)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_title(f'Log-Ratio Distribution Evolution Over Time (at $x \\approx 0$, rank $r={r}$)', fontsize=22)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, common_info_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_log_ratio_distribution_time_evolution.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_log_ratio_distribution_time_evolution.png'}")
    plt.close()
    
    # we plot 3c: channel shares at final time (separate figure)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    f_k_final = mf_solver.compute_partial_functions(w1_final, x_tensor)
    shares = spec_metrics.compute_channel_shares(f_k_final).squeeze(1).detach().cpu().numpy()
    channel_indices = np.arange(1, r + 1)
    ax.bar(channel_indices, shares, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Channel $k$', fontsize=24)
    ax.set_ylabel('Channel Share $s_k = |f_k| / \\sum_j |f_j|$', fontsize=24)
    ax.set_title(f'Channel Shares at Final Time $t={final_time:.1f}$ (at $x \\approx 0$, rank $r={r}$)', fontsize=22)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, common_info_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_channel_shares.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_channel_shares.png'}")
    plt.close()
    
    # we plot 4: weight density normal scale (separate figure)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.hist(all_weights, bins=50, alpha=0.7, edgecolor='black', linewidth=1.5, density=True)
    ax.set_xlabel('Weight Value $w$', fontsize=24)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_title(f'Weight Density (Normal Scale, Final Time $t={final_time:.1f}$)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, f'{common_info_text} Total weights: ${n1*r + n2}$ ($w_1$: ${n1*r}$, $w_2$: ${n2}$).', ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_weight_density_normal.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_weight_density_normal.png'}")
    plt.close()
    
    # we plot 5: weight density log-log scale (separate figure)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    log_bins = np.logspace(np.log10(abs_weights.min()), np.log10(abs_weights.max()), 50)
    hist, bins = np.histogram(abs_weights, bins=log_bins)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean for log scale
    non_zero = hist > 0
    ax.plot(bin_centers[non_zero], hist[non_zero], 'b-o', linewidth=2, markersize=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$|w|$ (absolute weight)', fontsize=24)
    ax.set_ylabel('Count', fontsize=24)
    ax.set_title(f'Weight Density (Log-Log Scale, Final Time $t={final_time:.1f}$)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, f'{common_info_text} Total weights: ${n1*r + n2}$.', ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_weight_density_loglog.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_weight_density_loglog.png'}")
    plt.close()
    
    # we plot 5b: weight density evolution across time points (separate figure)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    for t_idx in time_indices:
        t = mf_solver.times[t_idx]
        w1_t, w2_t = mf_solver.get_weights_at_time(t_idx)
        w1_flat_t = w1_t.detach().cpu().numpy().flatten()
        w2_flat_t = w2_t.detach().cpu().numpy().flatten()
        all_weights_t = np.concatenate([w1_flat_t, w2_flat_t])
        ax.hist(all_weights_t, bins=30, alpha=0.5, label=f'$t={t:.1f}$', density=True)
    ax.set_xlabel('Weight Value $w$', fontsize=24)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_title(f'Weight Density Evolution Over Time (width $n={n1}$, rank $r={r}$)', fontsize=22)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)
    fig.text(0.5, 0.02, common_info_text, ha='center', fontsize=12, wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'meanfield_weight_density_time_evolution.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_weight_density_time_evolution.png'}")
    plt.close()
    
    # we save results
    def convert_to_serializable(obj):
        """we convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    save_results = {
        'target_function': 'cos(12πx) + cos(24πx + 0.5) + cos(36πx) + cos(72πx + 0.5)',
        'analysis_location': 0.0,
        'architecture': {
            'n1': n1,
            'n2': n2,
            'rank': r
        },
        'training': {
            'num_samples': num_training_samples,
            'x_range': interval
        },
        'results': convert_to_serializable(results)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=4)
    print(f"Saved results to {output_dir / 'results.json'}")
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)


if __name__ == "__main__":
    run_experiment()
