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
num_epochs = 1000
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

# we setup paths for data storagein
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
        
        if epoch % 100 == 0:
            print(f"\nepoch {epoch}/{num_epochs} ({epoch/num_epochs*100:.2f}%)")
            print(f"training error (MSE): {training_error:.2e}")
            print(f"test errors (MAX and MSE): {e_max:.2e} and {e_mse:.2e}")
            print(f"time used: {time.time() - time1:.2f}s")
            
                        
                        
            # we compute loss landscape around final trained parameters
            print("\n" + "="*80)
            print("computing loss landscape (2d affine projection)")
            print("="*80)

            # we create landscape folder
            landscape_folder = os.path.join(base_folder, "figures", f"landscape_{epoch}")
            os.makedirs(landscape_folder, exist_ok=True)

            # we extract final trainable parameters as a flat vector
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            final_params_flat = torch.cat([p.data.view(-1) for p in trainable_params])  # we flatten all trainable params
            n_params = final_params_flat.shape[0]
            print(f"number of trainable parameters: {n_params}")

            # we save final model parameters
            torch.save({
                'model_state_dict': model.state_dict(),
                'final_params_flat': final_params_flat,
                'trainable_param_shapes': [p.shape for p in trainable_params]
            }, os.path.join(landscape_folder, f"{config_name}_final_model_{epoch}.pt"))
            print(f"saved final model parameters")

            # we generate 2 random orthogonal directions
            torch.manual_seed(42)  # we set seed for reproducibility
            direction1 = torch.randn(n_params, device=device)  # we generate random direction 1
            direction1 = direction1 / torch.norm(direction1)  # we normalize

            direction2 = torch.randn(n_params, device=device)  # we generate random direction 2
            direction2 = direction2 - (direction2 @ direction1) * direction1  # we orthogonalize using Gram-Schmidt
            direction2 = direction2 / torch.norm(direction2)  # we normalize

            print(f"generated 2 orthogonal random directions")
            print(f"direction 1 norm: {torch.norm(direction1).item():.6f}")
            print(f"direction 2 norm: {torch.norm(direction2).item():.6f}")
            print(f"directions dot product: {(direction1 @ direction2).item():.6e} (should be ~0)")

            # we set up grid for loss landscape
            n_grid = 210  # we use 21x21 grid
            alpha_range = np.linspace(-10.0, 10.0, n_grid)  # we set range for direction 1
            beta_range = np.linspace(-10.0, 10.0, n_grid)  # we set range for direction 2
            loss_grid = np.zeros((n_grid, n_grid))  # we initialize loss grid

            print(f"evaluating loss on {n_grid}x{n_grid} grid...")

            # we evaluate loss at each grid point
            model.eval()
            with torch.no_grad():
                for i, alpha in enumerate(tqdm(alpha_range, desc="alpha (direction 1)")):
                    for j, beta in enumerate(beta_range):
                        # we compute perturbed parameters: theta = theta_final + alpha*d1 + beta*d2
                        perturbed_params = final_params_flat + alpha * direction1 + beta * direction2
                        
                        # we load perturbed parameters into model
                        idx = 0
                        for p in trainable_params:
                            numel = p.numel()
                            p.data = perturbed_params[idx:idx+numel].view(p.shape)
                            idx += numel
                        
                        # we compute loss on training set
                        total_loss = 0.0
                        n_batches = 0
                        for inputs, targets in train_loader:
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            total_loss += loss.item()
                            n_batches += 1
                        
                        avg_loss = total_loss / n_batches
                        loss_grid[i, j] = avg_loss

            # we restore final parameters
            idx = 0
            for p in trainable_params:
                numel = p.numel()
                p.data = final_params_flat[idx:idx+numel].view(p.shape)
                idx += numel

            print(f"loss landscape computed")
            print(f"min loss: {np.min(loss_grid):.6e}")
            print(f"max loss: {np.max(loss_grid):.6e}")
            print(f"loss at final point (0,0): {loss_grid[n_grid//2, n_grid//2]:.6e}")

            # we save loss landscape data
            np.savez(os.path.join(landscape_folder, f"{config_name}_loss_landscape_{epoch}.npz"),
                    loss_grid=loss_grid,
                    alpha_range=alpha_range,
                    beta_range=beta_range,
                    direction1=direction1.cpu().numpy(),
                    direction2=direction2.cpu().numpy(),
                    final_params=final_params_flat.cpu().numpy())

            # we create 3d plotly visualization
            Alpha, Beta = np.meshgrid(alpha_range, beta_range)

            fig = go.Figure(data=[go.Surface(
                x=Alpha,
                y=Beta,
                z=loss_grid.T,  # we transpose to match meshgrid convention
                colorscale='Viridis',
                colorbar=dict(title='Loss'),
                name='Loss Landscape'
            )])

            # we add marker at final point (0, 0)
            final_loss = loss_grid[n_grid//2, n_grid//2]
            fig.add_trace(go.Scatter3d(
                x=[0],
                y=[0],
                z=[final_loss],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='Final Parameters'
            ))

            fig.update_layout(
                title=f'Loss Landscape - {config_name}<br>Affine 2D projection around final trained parameters',
                scene=dict(
                    xaxis_title='α (direction 1)',
                    yaxis_title='β (direction 2)',
                    zaxis_title='Loss',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                width=1000,
                height=800,
                font=dict(size=12)
            )

            # we save interactive html
            html_path = os.path.join(landscape_folder, f"{config_name}_loss_landscape_{epoch}.html")
            fig.write_html(html_path)
            print(f"saved interactive loss landscape to: {html_path}")

            # we also create a 2d contour plot
            fig_contour = go.Figure(data=[go.Contour(
                x=alpha_range,
                y=beta_range,
                z=loss_grid.T,
                colorscale='Viridis',
                colorbar=dict(title='Loss'),
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=10, color='white')
                )
            )])

            # we add marker at final point
            fig_contour.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='star'),
                text=['Final'],
                textposition='top center',
                textfont=dict(size=14, color='red'),
                name='Final Parameters'
            ))

            fig_contour.update_layout(
                title=f'Loss Landscape (Contour) - {config_name}<br>Affine 2D projection around final trained parameters',
                xaxis_title='α (direction 1)',
                yaxis_title='β (direction 2)',
                width=900,
                height=800,
                font=dict(size=12)
            )

            # we save contour html
            html_contour_path = os.path.join(landscape_folder, f"{config_name}_loss_landscape_contour_{epoch}.html")
            fig_contour.write_html(html_contour_path)
            print(f"saved interactive contour plot to: {html_contour_path}")

            print("\n" + "="*80)
            print("loss landscape analysis completed")
            print("="*80)


            
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
            #plt.savefig(os.path.join(base_folder, f"{config_name}_prediction_epoch{epoch}.png"), dpi=150)
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
#plt.savefig(os.path.join(base_folder, f"{config_name}_final_comparison.png"), dpi=150)
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
#plt.savefig(os.path.join(base_folder, f"{config_name}_loss_evolution_complete.png"), dpi=150)
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
#plt.savefig(os.path.join(base_folder, f"{config_name}_loss_evolution_loglog.png"), dpi=150)
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
#plt.savefig(os.path.join(base_folder, f"{config_name}_error_evolution_sampled.png"), dpi=150)
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
    #plt.savefig(os.path.join(base_folder, f"{config_name}_ntk_max_eigenvalue.png"), dpi=150)
    plt.close()
    
    # we plot minimum eigenvalue
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, min_eigenvalues, 'r-', linewidth=2)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('min eigenvalue', fontsize=12)
    plt.title(f'ntk minimum eigenvalue evolution - {config_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig(os.path.join(base_folder, f"{config_name}_ntk_min_eigenvalue.png"), dpi=150)
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
    "total_time": float(time.time() - time1),  # we convert to native python float
    "final_train_loss": float(all_losses[-1]) if len(all_losses) > 0 else None,  # we convert to native python float
    "final_test_error": float(errors_test[-1]) if len(errors_test) > 0 else None  # we convert to native python float
}

with open(os.path.join(base_folder, f"{config_name}_config.json"), "w") as f:
    json.dump(config_dict, f, indent=4)

print(f"\nall results saved to: {base_folder}")
print(f"total training time: {time.time() - time1:.2f}s")
