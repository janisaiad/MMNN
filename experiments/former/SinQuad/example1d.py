# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
matplotlib.use('Agg')  # we use non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import time
import os
import json
import nets
from tqdm import tqdm
import plotly.graph_objects as go

torch.set_printoptions(
    precision=3,      # we set decimal precision
    threshold=float('inf'),  # we show all elements (no truncation with ...)
    edgeitems=10,     # we show more edge items
    linewidth=200,    # we set wider line width
    sci_mode=False    # we disable scientific notation
)

# torch.set_default_dtype(torch.float64)
mydtype = torch.get_default_dtype()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"training on device: {device}")

def func(x):
    """we define the 1d oscillatory function"""
    y = torch.cos(36*np.pi* x**2) - 0.8*torch.cos(12*np.pi* x**2)  # we compute oscillatory function
    return y

def compute_ntk_gram(model, x):
    """we compute ntk using vectorized jacobian computation"""
    n = x.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    
    if len(params) == 0:
        return torch.zeros((n, n)), torch.zeros(n)
    
    # we compute all jacobians at once
    jacobians = []
    for i in range(n):
        x_i = x[i:i+1].requires_grad_(True)
        y_i = model(x_i)
        
        if not y_i.requires_grad:
            jacobians.append(torch.zeros(sum(p.numel() for p in params), device=device))
            continue
        
        grads = torch.autograd.grad(y_i.sum(), params, create_graph=False, allow_unused=True)
        jac = torch.cat([g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device) 
                         for g, p in zip(grads, params)])
        jacobians.append(jac)
    
    # we stack all jacobians: shape (n, n_params_total)
    J = torch.stack(jacobians)
    
    # we compute ntk as J @ J^T
    ntk = J @ J.T  # we perform matrix multiplication
    
    ntk_cpu = ntk.cpu()
    eigenvalues = torch.linalg.eigvalsh(ntk_cpu)
    
    return ntk_cpu, eigenvalues

# we set hyperparameters
num_epochs = 20000
batch_size = 100
num_training_samples = 1000  # we set uniform grid samples
num_test_samples = 1234  # we set random samples
  
# we define learning rate schedule: lr_init*lr_gamma**floor(k/lr_step_size)
lr_init = 0.001
lr_gamma = 0.9
lr_step_size = 400

interval = [-1, 1]
ranks = [1] + [36]*5 + [1]
widths = [666]*6

# we create config name
depth = len(widths)
width = widths[0]  # we assume all widths are same
rank = ranks[1]  # we assume all middle ranks are same
config_name = f"d{depth}_w{width}_r{rank}_1d"

# we setup paths for data storage
timestamp = time.strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.join(script_dir, "../../data/storage/1d_experiments", f"results_1d_{timestamp}")
base_folder = os.path.normpath(base_folder)
os.makedirs(base_folder, exist_ok=True)

print("="*80)
print("starting 1d mmnn training experiment")
print("="*80)
print(f"config: {config_name}")
print(f"results will be saved to: {base_folder}")

# we generate training data
x_train_np = np.linspace(*interval, num_training_samples).reshape([-1, 1])
x_train = torch.tensor(x_train_np, device=device, dtype=mydtype)
y_train = func(x_train)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# we create model
model = nets.MMNN(ranks=ranks, widths=widths, device=device, ResNet=False, fixWb=True, act_kind=["ReLU"]*len(widths))

# we MUST initialize weights properly (especially important with fixWb=True!)
# even frozen weights need good initialization
for layer in model.fcs:
    torch.nn.init.xavier_uniform_(layer.weight)  # we use xavier initialization (default in PyTorch for Linear)
    torch.nn.init.zeros_(layer.bias)  # we zero-initialize biases

params = [p for p in model.parameters() if p.requires_grad]
print(f"\ntrainable parameters: {sum(p.numel() for p in params)}")

# we setup optimizer and criterion
time1 = time.time()
all_losses = []  # we store loss at every epoch
errors_train = []
errors_test = []
errors_test_max = []
ntk_matrices = {}
ntk_eigenvalues = {}

optimizer = optim.Adam(model.parameters(), lr=lr_init)  # we use SGD to match benchmark.py
scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
criterion = nn.MSELoss()

# we start training loop
pbar = tqdm(range(1, 1+num_epochs), desc="training")
for epoch in pbar:
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # we check for nan/inf before backward
        if not torch.isfinite(loss):
            print(f"\nWARNING: Non-finite loss detected at epoch {epoch}: {loss.item()}")
            print("Skipping this batch...")
            continue
            
        loss.backward()
        
        # we clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    # we store loss at every epoch
    all_losses.append(loss.item())
    
    scheduler.step()
              
    if epoch % 50 == 0:
        training_error = loss.item()
        pbar.set_postfix({"loss": f"{training_error:.2e}"})
        errors_train.append(training_error)
    
        # we evaluate on test set
        with torch.no_grad():
            x_test_np = np.random.rand(num_test_samples) * 2 - 1
            x_test = torch.tensor(x_test_np.reshape([-1, 1]), dtype=mydtype, device=device)
            y_test_pred = model(x_test).cpu().numpy().reshape([-1])
            y_test_true = func(x_test).cpu().numpy().reshape([-1])
        
        # we calculate errors
        e = y_test_pred - y_test_true
        e_max = np.max(np.abs(e))
        e_mse = np.mean(e**2)
        errors_test.append(e_mse)
        errors_test_max.append(e_max)
        
        if epoch % 500 == 0:
            print(f"\nepoch {epoch}/{num_epochs} ({epoch/num_epochs*100:.2f}%)")
            print(f"training error (MSE): {training_error:.2e}")
            print(f"test errors (MAX and MSE): {e_max:.2e} and {e_mse:.2e}")
            print(f"time used: {time.time() - time1:.2f}s")
            
        if epoch % 5000 == 0:
            # we compute NTK
            ntk, eigenvalues = compute_ntk_gram(model, x_train)
            ntk_matrices[epoch] = ntk
            ntk_eigenvalues[epoch] = eigenvalues
            print(f"ntk eigenvalues: min={eigenvalues[0]:.3e}, max={eigenvalues[-1]:.3e}")
        
        if epoch % 1000 == 0:
            # we plot the results (every 1000 epochs)
            x_plot = np.linspace(-1, 1, 1000)
            x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), dtype=mydtype, device=device)
            with torch.no_grad():
                y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
            y_plot_true = func(x_plot_tensor).cpu().numpy().reshape([-1])
            
            fig = plt.figure(figsize=(8, 5))
            plt.plot(x_plot, y_plot_true, 'b-', label='true function', linewidth=2)
            plt.plot(x_plot, y_plot_nn, 'r--', label='learned network', linewidth=2)
            plt.xticks(np.linspace(*interval, 5))
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.grid(True, axis='both', color='#AAAAAA', linestyle='--', linewidth=1.4)
            plt.title(f'true function and learned network (epoch {epoch})', fontsize=14)
            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.tight_layout()
            plt.legend(loc="upper center", fontsize=13, ncol=2)
            plt.savefig(os.path.join(base_folder, f"{config_name}_prediction_epoch{epoch}.png"), dpi=150)
            plt.close()

print(f"\n{'='*80}")
print("training completed")
print(f"{'='*80}")

# we plot final comparison
x_plot_final = np.linspace(-1, 1, 1000)
x_plot_tensor_final = torch.tensor(x_plot_final.reshape([-1, 1]), dtype=mydtype, device=device)
with torch.no_grad():
    y_plot_nn_final = model(x_plot_tensor_final).cpu().numpy().reshape([-1])
y_plot_true_final = func(x_plot_tensor_final).cpu().numpy().reshape([-1])

fig = plt.figure(figsize=(10, 6))
plt.plot(x_plot_final, y_plot_true_final, 'b-', label='true function', linewidth=2.5)
plt.plot(x_plot_final, y_plot_nn_final, 'r--', label='learned network', linewidth=2, alpha=0.8)
plt.xticks(np.linspace(*interval, 5))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, axis='both', color='#AAAAAA', linestyle='--', linewidth=1.4)
plt.title(f'final comparison: true function vs learned network (epoch {num_epochs})', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.tight_layout()
plt.legend(loc="upper center", fontsize=13, ncol=2)
plt.savefig(os.path.join(base_folder, f"{config_name}_final_comparison.png"), dpi=150)
plt.close()
print(f"final comparison plot saved")

# we save model parameters
torch.save(model.state_dict(), os.path.join(base_folder, f'{config_name}_model_parameters.pth'))

# we save errors and losses
np.savez(os.path.join(base_folder, f"{config_name}_errors.npz"), 
         test=np.array(errors_test), 
         testmax=np.array(errors_test_max), 
         train=np.array(errors_train),
         all_losses=np.array(all_losses),
         time=time.time()-time1)

# we save NTK data
ntk_save_path = os.path.join(base_folder, f"{config_name}_ntk_data.pt")
torch.save({
    "ntk_matrices": ntk_matrices,
    "ntk_eigenvalues": ntk_eigenvalues
}, ntk_save_path)

# we plot complete loss evolution (all epochs)
fig = plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1, alpha=0.7, label='training loss (all epochs)')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss (log scale)', fontsize=12)
plt.title(f'complete loss evolution - {config_name}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(base_folder, f"{config_name}_loss_evolution_complete.png"), dpi=150)
plt.close()

# we plot loss evolution (loglog scale)
fig = plt.figure(figsize=(10, 6))
plt.loglog(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1, alpha=0.7, label='training loss')
plt.xlabel('epoch (log scale)', fontsize=12)
plt.ylabel('loss (log scale)', fontsize=12)
plt.title(f'loss evolution (loglog) - {config_name}', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(base_folder, f"{config_name}_loss_evolution_loglog.png"), dpi=150)
plt.close()

# we plot error evolution (sampled every 50 epochs)
fig = plt.figure(figsize=(10, 6))
n = len(errors_test) 
m = len(errors_train)
plt.plot(np.linspace(1, m, m)*50, np.log10(errors_train), 'b-', label="log10 training error", linewidth=2)
plt.plot(np.linspace(1, n, n)*50, np.log10(errors_test), 'r--', label="log10 test error", linewidth=2)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('log10(error)', fontsize=12)
plt.title(f'error evolution (sampled) - {config_name}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(base_folder, f"{config_name}_error_evolution_sampled.png"), dpi=150)
plt.close()

# we plot NTK eigenvalues evolution
if len(ntk_eigenvalues) > 0:
    epochs_list = sorted(ntk_eigenvalues.keys())
    max_eigenvalues = []
    min_eigenvalues = []
    
    for ep in epochs_list:
        eigs = ntk_eigenvalues[ep]
        max_eigenvalues.append(eigs[-1].item())
        min_eigenvalues.append(eigs[0].item())
    
    # we plot maximum eigenvalue
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, max_eigenvalues, 'b-', linewidth=2)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('max eigenvalue', fontsize=12)
    plt.title(f'ntk maximum eigenvalue evolution - {config_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, f"{config_name}_ntk_max_eigenvalue.png"), dpi=150)
    plt.close()
    
    # we plot minimum eigenvalue
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, min_eigenvalues, 'r-', linewidth=2)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('min eigenvalue', fontsize=12)
    plt.title(f'ntk minimum eigenvalue evolution - {config_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, f"{config_name}_ntk_min_eigenvalue.png"), dpi=150)
    plt.close()

# we save configuration
config_dict = {
    "config_name": config_name,
    "depth": depth,
    "width": width,
    "rank": rank,
    "ranks": ranks,
    "widths": widths,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "num_training_samples": num_training_samples,
    "num_test_samples": num_test_samples,
    "lr_init": lr_init,
    "lr_gamma": lr_gamma,
    "lr_step_size": lr_step_size,
    "interval": interval,
    "optimizer": "SGD",
    "activation": "ReLU",
    "resnet": False,
    "fix_wb": True,
    "trainable_params": sum(p.numel() for p in params),
    "total_time": time.time() - time1,
    "final_train_loss": all_losses[-1] if len(all_losses) > 0 else None,
    "final_test_error": errors_test[-1] if len(errors_test) > 0 else None
}

with open(os.path.join(base_folder, f"{config_name}_config.json"), "w") as f:
    json.dump(config_dict, f, indent=4)

print(f"\nall results saved to: {base_folder}")
print(f"total training time: {time.time() - time1:.2f}s")