#!/usr/bin/env python3
"""
we implement mean-field coupling and channel specialization metrics for 2-step function
this is a working example before large-scale experiments
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.integrate import solve_ivp
from scipy.stats import norm
import sys

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN

# we configure matplotlib
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class TwoStepFunction:
    """
    we create a two-sided step function (staircase with indicator functions) as defined in the paper:
    y(x) = A * (1{|x-x0| <= delta} - 1{|x-x1| <= delta})
    this is a true step function with sharp discontinuities (not smooth)
    """
    def __init__(self, x0=-0.5, x1=0.5, A=1.0, delta=0.05):
        """
        we create a two-sided step: +A bump at x0, -A bump at x1
        delta: half-width of each bump (indicator function support)
        """
        self.x0 = x0  # center of positive spike (-0.5)
        self.x1 = x1  # center of negative spike (0.5)
        self.A = A
        self.delta = delta
    
    def __call__(self, x):
        """
        we evaluate the two-sided step function (staircase with indicator functions):
        y(x) = A * (1{|x-x0| <= delta} - 1{|x-x1| <= delta})
        this is a true step function with sharp discontinuities, not smooth
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        # we use actual indicator functions (true step function)
        # indicator for positive bump: 1 if |x - x0| <= delta, else 0
        indicator0 = (np.abs(x_np - self.x0) <= self.delta).astype(float) * self.A
        # indicator for negative bump: 1 if |x - x1| <= delta, else 0
        indicator1 = (np.abs(x_np - self.x1) <= self.delta).astype(float) * (-self.A)
        result = indicator0 + indicator1
        
        if isinstance(x, torch.Tensor):
            return torch.tensor(result, dtype=x.dtype, device=x.device)
        return result
    
    def get_4point_dataset(self):
        """we create the 4-point dataset on [-1, 1] with spikes at -0.5 and 0.5"""
        x_points = np.array([-1, -0.5, 0.5, 1])
        y_points = np.array([0, self.A, -self.A, 0])
        return x_points, y_points
    
    def get_2spike_dataset(self):
        """we create the 2-spike dataset (just the two non-zero points) on [-1, 1]"""
        x_points = np.array([-0.5, 0.5])
        y_points = np.array([self.A, -self.A])
        return x_points, y_points


class MeanFieldODESolver:
    """we solve the mean-field ODEs for low-rank networks"""
    def __init__(self, n1=1000, n2=1000, r=2, d=1, device="cpu"):
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
        # L should be O(1) with 1/width normalization in the forward pass
        self.L = torch.randn(n2, r, device=device)  # Gaussian N(0, 1)
        
        # we initialize weights as Gaussian O(1) (not zero)
        # weights are O(1), normalization comes from 1/width in mean operations (torch.mean)
        self.w1_0 = torch.randn(n1, r, device=device)  # Gaussian N(0, 1) - order 1
        self.w2_0 = torch.randn(n2, device=device)  # Gaussian N(0, 1) - order 1
        
        # we store trajectory
        self.trajectory = []
        self.times = []
    
    def compute_H2(self, w1, w2, X):
        """we compute H2 = sum_k L_{c2,k} m_k where m_k = E_C1[w1(C1,k) phi1(f1(C1), X)]"""
        # we handle both single point and batch
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        
        # we compute m_k for each channel
        # m_k = (1/n1) sum_{j1} w1[j1,k] * ReLU(f1[j1] @ X)
        # X: [batch_size, d], f1: [n1, d]
        inner = torch.matmul(self.f1, X.t())  # [n1, batch_size]
        phi1 = torch.relu(inner)  # [n1, batch_size]
        
        # we compute m_k: [r, batch_size]
        m_k = torch.zeros(self.r, batch_size, device=self.device)
        for k in range(self.r):
            w1_k = w1[:, k].unsqueeze(1)  # [n1, 1]
            m_k[k] = torch.mean(w1_k * phi1, dim=0)  # [batch_size]
        
        # we compute H2: [n2, batch_size]
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
        # we reshape y into w1 and w2
        w1_flat = y[:self.n1 * self.r]
        w2_flat = y[self.n1 * self.r:]
        w1 = w1_flat.reshape(self.n1, self.r)
        w2 = w2_flat
        
        # we convert to torch
        w1_t = torch.tensor(w1, device=self.device, dtype=torch.float32)
        w2_t = torch.tensor(w2, device=self.device, dtype=torch.float32)
        X_t = torch.tensor(X_data, device=self.device, dtype=torch.float32)
        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(1)
        y_data_t = torch.tensor(y_data, device=self.device, dtype=torch.float32)
        
        # we compute H2 and backprop signal
        H2, m_k = self.compute_H2(w1_t, w2_t, X_t)
        B_k = self.compute_backprop_signal(w2_t, H2)
        
        # we compute output and loss derivative
        y_hat = self.compute_output(w1_t, w2_t, X_t)
        dL = y_hat - y_data_t  # square loss derivative [batch_size]
        
        # we compute gradients
        # dw1/dt = -xi1 * E[dL * phi1(f1, X) * B_k]
        # we compute phi1: [n1, batch_size]
        inner_prod = torch.matmul(self.f1, X_t.t())  # [n1, batch_size]
        phi1_vals = torch.relu(inner_prod)  # [n1, batch_size]
        
        dw1 = torch.zeros_like(w1_t)
        for k in range(self.r):
            # we average over batch: E[dL * phi1 * B_k]
            grad_k = torch.mean(dL.unsqueeze(0) * phi1_vals * B_k[k].unsqueeze(0), dim=1)  # [n1]
            dw1[:, k] = -xi1 * grad_k
        
        # dw2/dt = -xi2 * E[dL * phi2(H2)]
        phi2_vals = torch.relu(H2)  # [n2, batch_size]
        dw2 = -xi2 * torch.mean(dL.unsqueeze(0) * phi2_vals, dim=1)  # [n2]
        
        # we flatten and convert to numpy
        dw1_flat = dw1.cpu().numpy().flatten()
        dw2_flat = dw2.cpu().numpy()
        return np.concatenate([dw1_flat, dw2_flat])
    
    def solve(self, X_data, y_data, t_span=(0, 10), dt=0.01, xi1=1.0, xi2=1.0):
        """we solve the mean-field ODEs"""
        # we prepare initial condition
        y0 = np.concatenate([
            self.w1_0.cpu().numpy().flatten(),
            self.w2_0.cpu().numpy()
        ])
        
        # we create time points
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        # we solve ODE
        sol = solve_ivp(
            lambda t, y: self.ode_rhs(t, y, X_data, y_data, xi1=xi1, xi2=xi2),
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-5,
            atol=1e-7
        )
        
        # we store trajectory
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
        """
        we compute f_k(t,x) = m_k(t;x,W) for each channel
        these are the LOW-RANK COMPONENTS before mixing via L
        
        f_k(t,x) = m_k(t;x,W) = E_C1[w1(t,C1,k) * phi1(f1(C1), x)]
        where:
        - w1(t,C1,k) are the first-layer weights for channel k
        - f1(C1) are the frozen random features
        - phi1 is ReLU activation
        
        these partial functions are then mixed via L to form H2:
        H2(t,c2;X) = sum_{k=1}^r L_{c2,k} * m_k(t;X)
        """
        # we handle both single point and batch
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        
        # X: [batch_size, d], f1: [n1, d]
        inner = torch.matmul(self.f1, X.t())  # [n1, batch_size]
        phi1 = torch.relu(inner)  # [n1, batch_size]
        
        # we compute m_k = E_C1[w1(C1,k) * phi1(f1(C1), x)]
        f_k = torch.zeros(self.r, batch_size, device=self.device)
        for k in range(self.r):
            w1_k = w1[:, k].unsqueeze(1)  # [n1, 1]
            f_k[k] = torch.mean(w1_k * phi1, dim=0)  # [batch_size] - this is m_k
        
        return f_k  # returns [r, batch_size] - the low-rank components


class ChannelSpecializationMetrics:
    """we compute channel specialization metrics"""
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
    
    def compute_channel_shares(self, f_k):
        """
        we compute channel shares s_k(t,x) = |f_k(t,x)| / (sum_j |f_j(t,x)| + epsilon)
        f_k: [r, batch_size] tensor
        returns: [r, batch_size] tensor
        """
        abs_f = torch.abs(f_k)  # [r, batch_size]
        sum_abs = torch.sum(abs_f, dim=0, keepdim=True) + self.epsilon  # [1, batch_size]
        shares = abs_f / sum_abs  # [r, batch_size]
        return shares
    
    def compute_log_ratios(self, f_k):
        """
        we compute log-ratios R_{k,ell}(t,x) = log((|f_k| + epsilon) / (|f_ell| + epsilon))
        f_k: [r, batch_size] tensor
        returns: [r, r, batch_size] tensor where result[k, ell, i] = R_{k,ell}(t, x_i)
        """
        r, batch_size = f_k.shape
        abs_f = torch.abs(f_k) + self.epsilon  # [r, batch_size]
        
        log_ratios = torch.zeros(r, r, batch_size, device=f_k.device)
        for k in range(r):
            for ell in range(r):
                if k != ell:
                    log_ratios[k, ell] = torch.log(abs_f[k] / abs_f[ell])
        
        return log_ratios
    
    def compute_dominance_metrics(self, f_k, spike_locations):
        """
        we compute dominance metrics at spike locations
        returns: dict with channel shares and log-ratios at each spike
        """
        shares = self.compute_channel_shares(f_k)  # [r, batch_size]
        log_ratios = self.compute_log_ratios(f_k)  # [r, r, batch_size]
        
        metrics = {}
        for i, x_loc in enumerate(spike_locations):
            metrics[f'spike_{i}'] = {
                'location': x_loc,
                'shares': shares[:, i].cpu().numpy(),
                'log_ratios': log_ratios[:, :, i].cpu().numpy()
            }
        
        return metrics


class CouplingMetrics:
    """we compute coupling distance between mean-field and finite-width networks"""
    def __init__(self):
        pass
    
    def compute_distance(self, w1_mf, w2_mf, w1_fw, w2_fw):
        """
        we compute D_T(W, W_fw) = sup{|w1_mf - w1_fw|, |w2_mf - w2_fw|}
        """
        # we compute max differences
        diff_w1 = torch.max(torch.abs(w1_mf - w1_fw))
        diff_w2 = torch.max(torch.abs(w2_mf - w2_fw))
        
        distance = max(diff_w1.item(), diff_w2.item())
        
        return {
            'distance': distance,
            'w1_max_diff': diff_w1.item(),
            'w2_max_diff': diff_w2.item(),
            'w1_mean_diff': torch.mean(torch.abs(w1_mf - w1_fw)).item(),
            'w2_mean_diff': torch.mean(torch.abs(w2_mf - w2_fw)).item()
        }


def train_finite_width_network(X_train, y_train, mf_solver, 
                               num_epochs=1000, lr=0.001, device="cpu"):
    """we train a finite-width network with same architecture as mean-field"""
    n1 = mf_solver.n1
    n2 = mf_solver.n2
    r = mf_solver.r
    
    # we create a simplified 2-layer model matching mean-field structure
    class SimpleLowRankNet(nn.Module):
        def __init__(self, n1, n2, r, f1_init, L_init, w1_init, w2_init):
            super().__init__()
            self.n1 = n1
            self.n2 = n2
            self.r = r
            
            # we freeze random features (like mean-field) - use same as mean-field
            self.f1 = nn.Parameter(f1_init.clone(), requires_grad=False)
            
            # we trainable weights - initialize same as mean-field
            self.w1 = nn.Parameter(w1_init.clone())
            self.w2 = nn.Parameter(w2_init.clone())
            
            # we freeze mixing matrix L - use same as mean-field
            self.L = nn.Parameter(L_init.clone(), requires_grad=False)
        
        def forward(self, x):
            # we handle input shape: x can be [batch_size, d] or [batch_size]
            if x.dim() == 1:
                x = x.unsqueeze(1)
            batch_size = x.shape[0]
            
            # we compute m_k
            # x: [batch_size, d], f1: [n1, d]
            inner = torch.matmul(self.f1, x.t())  # [n1, batch_size]
            phi1 = torch.relu(inner)  # [n1, batch_size]
            
            m_k = torch.zeros(self.r, batch_size, device=x.device)
            for k in range(self.r):
                m_k[k] = torch.mean(self.w1[:, k].unsqueeze(1) * phi1, dim=0)
            
            # we compute H2
            H2 = torch.zeros(self.n2, batch_size, device=x.device)
            for j2 in range(self.n2):
                H2[j2] = torch.sum(self.L[j2, :].unsqueeze(1) * m_k, dim=0)
            
            # we compute output
            phi2 = torch.relu(H2)  # [n2, batch_size]
            y_hat = torch.mean(self.w2.unsqueeze(1) * phi2, dim=0)
            
            return y_hat
        
        def compute_partial_functions(self, x):
            """
            we compute partial functions f_k = m_k from the trained finite-width network
            this matches the mean-field compute_partial_functions method
            """
            # we handle input shape
            if x.dim() == 1:
                x = x.unsqueeze(0)
            batch_size = x.shape[0]
            
            # we compute m_k = E_C1[w1(C1,k) * phi1(f1(C1), x)]
            inner = torch.matmul(self.f1, x.t())  # [n1, batch_size]
            phi1 = torch.relu(inner)  # [n1, batch_size]
            
            f_k = torch.zeros(self.r, batch_size, device=x.device)
            for k in range(self.r):
                w1_k = self.w1[:, k].unsqueeze(1)  # [n1, 1]
                f_k[k] = torch.mean(w1_k * phi1, dim=0)  # [batch_size] - this is m_k
            
            return f_k  # returns [r, batch_size] - the low-rank components
    
    # we use exact same initialization as mean-field
    model = SimpleLowRankNet(
        n1, n2, r,
        mf_solver.f1,
        mf_solver.L,
        mf_solver.w1_0,
        mf_solver.w2_0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    if X_tensor.dim() == 1:
        X_tensor = X_tensor.unsqueeze(1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    return model, losses


def run_experiment(output_dir):
    """we run the complete experiment"""
    print("="*80)
    print("Mean-Field Coupling and Channel Specialization Experiment")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # we create 2-step function on [-1, 1] with spikes at -0.5 and 0.5
    # this removes inductive bias from norm (0 is centered)
    # this is the two-sided step: y(x) = A*(1{|x-x0|<=delta} - 1{|x-x1|<=delta})
    step_func = TwoStepFunction(x0=-0.5, x1=0.5, A=1.0, delta=0.05)
    
    # we create training data (4-point dataset on [-1, 1])
    x_train, y_train = step_func.get_4point_dataset()
    print(f"\nTraining data: x={x_train}, y={y_train}")
    
    # we also create a fine grid for visualization on [-1, 1]
    x_fine = np.linspace(-1, 1, 100)
    y_fine = step_func(x_fine)
    
    # we set up parameters
    n1, n2 = 1000, 1000  # we use width 1000
    r = 2  # we use 2 channels
    t_span = (0, 100)  # we solve for 100 time units
    dt = 0.1
    
    print(f"\nNetwork Architecture:")
    print(f"  First layer: n1={n1} neurons with frozen random features f1")
    print(f"  Low-rank mixing: r={r} channels with mixing matrix L (n2 x r = {n2} x {r})")
    print(f"  Second layer: n2={n2} neurons with trainable weights w2")
    print(f"  Partial functions: f_k(t,x) = m_k(t;x,W) = E_C1[w1(C1,k) * phi1(f1(C1), x)]")
    print(f"  Second layer pre-activation: H2(t,c2;X) = sum_k L_{{c2,k}} * m_k(t;X)")
    print(f"  Output: y_hat = E_C2[w2(C2) * phi2(H2(C2))]")
    print(f"\nParameters: n1={n1}, n2={n2}, r={r}, d=1 (input dimension)")
    
    # we solve mean-field ODEs
    print("\n" + "="*80)
    print("Solving Mean-Field ODEs...")
    print("="*80)
    mf_solver = MeanFieldODESolver(n1=n1, n2=n2, r=r, d=1, device=device)
    
    # we verify Gaussian initialization (not zero)
    print(f"\nWeight Initialization (Gaussian, not zero):")
    print(f"  w1_0: shape [{n1}, {r}], mean={mf_solver.w1_0.mean().item():.6f}, std={mf_solver.w1_0.std().item():.6f}")
    print(f"  w2_0: shape [{n2}], mean={mf_solver.w2_0.mean().item():.6f}, std={mf_solver.w2_0.std().item():.6f}")
    sol = mf_solver.solve(x_train, y_train, t_span=t_span, dt=dt)
    
    print(f"Mean-field ODE solved. Trajectory shape: {mf_solver.trajectory.shape}")
    
    # we train finite-width network
    print("\n" + "="*80)
    print("Training Finite-Width Network...")
    print("="*80)
    fw_model, fw_losses = train_finite_width_network(
        x_train, y_train, mf_solver,
        num_epochs=500, lr=0.001, device=device
    )
    
    # we compute coupling metrics at final time
    print("\n" + "="*80)
    print("Computing Coupling Metrics...")
    print("="*80)
    w1_mf, w2_mf = mf_solver.get_weights_at_time(-1)
    w1_fw = fw_model.w1.data
    w2_fw = fw_model.w2.data
    
    coupling = CouplingMetrics()
    coupling_dist = coupling.compute_distance(w1_mf, w2_mf, w1_fw, w2_fw)
    print(f"Coupling distance: {coupling_dist['distance']:.6f}")
    print(f"  w1 max diff: {coupling_dist['w1_max_diff']:.6f}")
    print(f"  w2 max diff: {coupling_dist['w2_max_diff']:.6f}")
    
    # we compute channel specialization metrics
    print("\n" + "="*80)
    print("Computing Channel Specialization Metrics...")
    print("="*80)
    spec_metrics = ChannelSpecializationMetrics()
    
    # we compute at spike locations
    spike_locations = [-0.5, 0.5]  # spikes at -0.5 and 0.5 from 4-point dataset
    spike_indices = [1, 2]  # indices in x_train (x_train = [-1, -0.5, 0.5, 1])
    
    results = {}
    time_indices = [0, len(mf_solver.times)//4, len(mf_solver.times)//2, -1]
    
    for t_idx in time_indices:
        t = mf_solver.times[t_idx]
        w1_t, w2_t = mf_solver.get_weights_at_time(t_idx)
        
        # we compute partial functions at spike locations
        X_spikes = torch.tensor(x_train[spike_indices], device=device, dtype=torch.float32).unsqueeze(1)
        f_k = mf_solver.compute_partial_functions(w1_t, X_spikes)
        
        # we compute metrics
        metrics = spec_metrics.compute_dominance_metrics(f_k, spike_locations)
        
        results[f'time_{t:.2f}'] = {
            'time': t,
            'metrics': metrics
        }
        
        print(f"\nTime t={t:.2f}:")
        for spike_name, spike_data in metrics.items():
            print(f"  {spike_name} (x={spike_data['location']:.3f}):")
            print(f"    Channel shares: {spike_data['shares']}")
            if spike_data['shares'][0] > 0.6:
                print(f"    -> Channel 0 dominates!")
            elif spike_data['shares'][1] > 0.6:
                print(f"    -> Channel 1 dominates!")
    
    # we create visualizations
    print("\n" + "="*80)
    print("Creating Visualizations...")
    print("="*80)
    
    # we plot 1: coupling distance over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # we plot mean-field vs finite-width predictions
    ax = axes[0, 0]
    X_fine_tensor = torch.tensor(x_fine, device=device, dtype=torch.float32)
    if X_fine_tensor.dim() == 1:
        X_fine_tensor = X_fine_tensor.unsqueeze(1)
    y_mf_final = mf_solver.compute_output(w1_mf, w2_mf, X_fine_tensor)
    y_fw_final = fw_model(X_fine_tensor)
    
    ax.plot(x_fine, y_fine, 'k-', label='Target', linewidth=2)
    ax.plot(x_fine, y_mf_final.detach().cpu().numpy(), 'b--', label='Mean-Field', linewidth=2)
    ax.plot(x_fine, y_fw_final.detach().cpu().numpy(), 'r:', label='Finite-Width', linewidth=2)
    ax.scatter(x_train, y_train, color='red', s=100, zorder=5, label='Training points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mean-Field vs Finite-Width Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # we plot channel shares over time at spike locations
    ax = axes[0, 1]
    times_plot = [mf_solver.times[i] for i in time_indices]
    shares_ch0_spike0 = [results[f'time_{t:.2f}']['metrics']['spike_0']['shares'][0] for t in times_plot]
    shares_ch1_spike0 = [results[f'time_{t:.2f}']['metrics']['spike_0']['shares'][1] for t in times_plot]
    shares_ch0_spike1 = [results[f'time_{t:.2f}']['metrics']['spike_1']['shares'][0] for t in times_plot]
    shares_ch1_spike1 = [results[f'time_{t:.2f}']['metrics']['spike_1']['shares'][1] for t in times_plot]
    
    ax.plot(times_plot, shares_ch0_spike0, 'b-o', label='Channel 0 at x=-0.5', linewidth=2)
    ax.plot(times_plot, shares_ch1_spike0, 'b--s', label='Channel 1 at x=-0.5', linewidth=2)
    ax.plot(times_plot, shares_ch0_spike1, 'r-o', label='Channel 0 at x=0.5', linewidth=2)
    ax.plot(times_plot, shares_ch1_spike1, 'r--s', label='Channel 1 at x=0.5', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Channel Share')
    ax.set_title('Channel Specialization Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # we plot log-ratios
    ax = axes[1, 0]
    log_ratios_spike0 = [results[f'time_{t:.2f}']['metrics']['spike_0']['log_ratios'][0, 1] for t in times_plot]
    log_ratios_spike1 = [results[f'time_{t:.2f}']['metrics']['spike_1']['log_ratios'][1, 0] for t in times_plot]
    
    ax.plot(times_plot, log_ratios_spike0, 'b-o', label='R_{0,1} at x=-0.5', linewidth=2)
    ax.plot(times_plot, log_ratios_spike1, 'r-o', label='R_{1,0} at x=0.5', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Log-Ratio R_{k,ell}')
    ax.set_title('Log-Ratio Growth (Channel Dominance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # we plot partial functions f_k at final time (low-rank components m_k)
    # we plot both mean-field (trained via ODE) and finite-width (trained via SGD)
    ax = axes[1, 1]
    f_k_mf = mf_solver.compute_partial_functions(w1_mf, X_fine_tensor)  # mean-field (trained)
    f_k_fw = fw_model.compute_partial_functions(X_fine_tensor)  # finite-width (trained)
    
    for k in range(r):
        ax.plot(x_fine, f_k_mf[k].detach().cpu().numpy(), 
               label=f'MF Channel {k} (trained)', linewidth=2, linestyle='--')
        ax.plot(x_fine, f_k_fw[k].detach().cpu().numpy(), 
               label=f'FW Channel {k} (trained)', linewidth=2, linestyle=':')
    ax.scatter(x_train[spike_indices], [0, 0], color='red', s=100, zorder=5, label='Spikes')
    ax.set_xlabel('x')
    ax.set_ylabel('f_k(x) = m_k(t;x,W)')
    ax.set_title('Low-Rank Partial Functions: Mean-Field (MF) vs Finite-Width (FW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'meanfield_channel_specialization.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'meanfield_channel_specialization.png'}")
    
    # we create a new figure for weight distributions through time
    print("\nCreating Weight Distribution Plots...")
    time_points_for_dist = [0, 20, 40, 60, 80, 100]  # we plot distributions at these times
    time_indices_dist = [int(t / dt) for t in time_points_for_dist if t <= t_span[1]]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # we plot w1 distributions for each channel at different times
    for k in range(r):
        ax = axes[0, k]
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices_dist)))
        
        for idx, t_idx in enumerate(time_indices_dist):
            if t_idx < len(mf_solver.times):
                w1_t, _ = mf_solver.get_weights_at_time(t_idx)
                w1_k = w1_t[:, k].detach().cpu().numpy()  # weights for channel k
                t = mf_solver.times[t_idx]
                
                # we plot histogram with auto bins
                ax.hist(w1_k, bins='auto', alpha=0.6, label=f't={t:.1f}', 
                       color=colors[idx], density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(f'w1[:, {k}] (First-layer weights for channel {k})')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of w1[:, {k}] Through Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # we plot w2 distribution at different times
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices_dist)))
    
    for idx, t_idx in enumerate(time_indices_dist):
        if t_idx < len(mf_solver.times):
            _, w2_t = mf_solver.get_weights_at_time(t_idx)
            w2_vals = w2_t.detach().cpu().numpy()  # second-layer weights
            t = mf_solver.times[t_idx]
            
            # we plot histogram with auto bins
            ax.hist(w2_vals, bins='auto', alpha=0.6, label=f't={t:.1f}', 
                   color=colors[idx], density=True, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('w2 (Second-layer weights)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of w2 Through Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # we plot statistics over time (mean and std)
    ax = axes[1, 1]
    all_times = mf_solver.times
    w1_means = {k: [] for k in range(r)}
    w1_stds = {k: [] for k in range(r)}
    w2_means = []
    w2_stds = []
    
    for t_idx in range(len(all_times)):
        w1_t, w2_t = mf_solver.get_weights_at_time(t_idx)
        w2_vals = w2_t.detach().cpu().numpy()
        w2_means.append(np.mean(w2_vals))
        w2_stds.append(np.std(w2_vals))
        
        for k in range(r):
            w1_k = w1_t[:, k].detach().cpu().numpy()
            w1_means[k].append(np.mean(w1_k))
            w1_stds[k].append(np.std(w1_k))
    
    # we plot means
    for k in range(r):
        ax.plot(all_times, w1_means[k], label=f'Mean w1[:, {k}]', linewidth=2, linestyle='--')
        ax.fill_between(all_times, 
                       np.array(w1_means[k]) - np.array(w1_stds[k]),
                       np.array(w1_means[k]) + np.array(w1_stds[k]),
                       alpha=0.2, label=f'±1 std w1[:, {k}]')
    
    ax.plot(all_times, w2_means, label='Mean w2', linewidth=2, linestyle='-')
    ax.fill_between(all_times,
                   np.array(w2_means) - np.array(w2_stds),
                   np.array(w2_means) + np.array(w2_stds),
                   alpha=0.2, label='±1 std w2')
    
    ax.set_xlabel('Time t')
    ax.set_ylabel('Weight Value')
    ax.set_title('Weight Statistics (Mean ± Std) Through Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_distributions_through_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved weight distribution figure to {output_dir / 'weight_distributions_through_time.png'}")
    
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
        else:
            return obj
    
    results_summary = {
        'coupling_distance': convert_to_serializable(coupling_dist),
        'channel_specialization': {k: {
            'time': v['time'],
            'metrics': convert_to_serializable(v['metrics'])
        } for k, v in results.items()},
        'parameters': {
            'n1': n1, 'n2': n2, 'r': r,
            't_span': list(t_span), 'dt': dt
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nSaved results to {output_dir / 'results.json'}")
    
    # we print summary of network structure
    print("\n" + "="*80)
    print("Network Structure Summary:")
    print("="*80)
    print(f"Architecture: 2-layer low-rank network")
    print(f"  Layer 1: {n1} neurons with frozen random features f1 (shape: [{n1}, {1}])")
    print(f"  Low-rank: {r} channels with mixing matrix L (shape: [{n2}, {r}])")
    print(f"  Layer 2: {n2} neurons with trainable weights w2 (shape: [{n2}])")
    print(f"\nPartial Functions (Low-Rank Components):")
    print(f"  f_k(t,x) = m_k(t;x,W) = E_C1[w1(C1,k) * ReLU(f1(C1) @ x)]")
    print(f"  These are the {r} low-rank components BEFORE mixing via L")
    print(f"  They get mixed: H2 = sum_k L_{{c2,k}} * m_k")
    print(f"  Then: y_hat = E_C2[w2(C2) * ReLU(H2(C2))]")
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    
    return results_summary


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "meanfield_two_step_results"
    run_experiment(output_dir)
