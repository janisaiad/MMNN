# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import time
import nets
import torch
from itertools import product
from tqdm import tqdm
import os
import json
import matplotlib
matplotlib.use('Agg')  # we use non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt  # NOW we can import pyplot safely


# Après les imports existants, ajoutez :
import plotly.graph_objects as go
from plotly.subplots import make_subplots


torch.set_printoptions(
    precision=3,      # we set decimal precision
    threshold=float('inf'),  # we show all elements (no truncation with ...)
    edgeitems=10,     # we show more edge items
    linewidth=200,    # we set wider line width
    sci_mode=False    # we disable scientific notation
)




device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"training on device: {device}")
torch.set_default_dtype(torch.float32)
mydtype = torch.get_default_dtype()

def oscillatory_function_1d(x):
    """we define the 1d oscillatory function f1(x) = cos(20π|x|^1.4) + 0.5cos(12π|x|^1.6)"""
    return torch.cos(20 * np.pi * torch.abs(x)**1.4) + 0.5 * torch.cos(12 * np.pi * torch.abs(x)**1.6)

def oscillatory_function_2d(x1, x2):
    """we define the 2d oscillatory function with given parameters"""
    s = 2
    a = torch.tensor([[0.3, 0.2], [0.2, 0.3]], dtype=torch.float32, device=x1.device)
    b = torch.tensor([2*np.pi, 4*np.pi], dtype=torch.float32, device=x1.device)
    c = torch.tensor([[2*np.pi, 4*np.pi], [8*np.pi, 4*np.pi]], dtype=torch.float32, device=x1.device)
    d = torch.tensor([[4*np.pi, 6*np.pi], [8*np.pi, 6*np.pi]], dtype=torch.float32, device=x1.device)
    
    result = torch.zeros_like(x1)
    for i in range(2):
        for j in range(2):
            term = a[i, j] * torch.sin(s * b[i] * x1 + s * c[i, j] * x1 * x2) * \
                   torch.cos(s * b[j] * x2 + s * d[i, j] * x1**2)
            result = result + term
    
    return result

def generate_data_1d(n_samples=100, x_range=(0, 1), device="cuda"):
    """we generate training data for 1d function"""
    x = torch.linspace(x_range[0], x_range[1], n_samples, device=device).reshape(-1, 1)
    y = oscillatory_function_1d(x)
    return x, y

def generate_data_2d(n_samples=100, x_range=(-1, 1), device="cuda"):
    """we generate training data for 2d function uniformly on unit circle"""
    # we sample uniformly on the unit circle (radius = 1)
    theta = np.pi * torch.linspace(0, 1, n_samples, device=device)
    
    x1 = torch.cos(theta)
    x2 = torch.sin(theta)
    
    x = torch.stack([x1, x2], dim=1)
    y = oscillatory_function_2d(x[:, 0:1], x[:, 1:2])
    # we rescale y to have unit variance
    y = y / torch.std(y)
    return x, y




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
    ntk = J @ J.T  # matrix multiplication: much faster!
    
    ntk_cpu = ntk.cpu()
    eigenvalues = torch.linalg.eigvalsh(ntk_cpu)
    
    return ntk_cpu, eigenvalues

def compute_loss_landscape_2d_projection(model, x_train, y_train, weight_snapshots, 
                                        n_grid=50, grid_range=4.0, use_pca=False, seed=42):
    """
    we compute loss landscape along 2 principal directions (PCA) or random directions
    """
    criterion = torch.nn.MSELoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    def get_weight_vector():
        return torch.cat([p.data.view(-1) for p in params])
    
    def set_weight_vector(w_vector):
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(w_vector[offset:offset+numel].view(p.shape))
            offset += numel
    
    w_final = get_weight_vector().clone()
    w_init = weight_snapshots[0].clone()
    
    if use_pca and len(weight_snapshots) > 2 :
        print("using PCA to find principal directions...")
        
        # we center the weight snapshots
        weight_matrix = torch.stack(weight_snapshots)  # shape: (n_snapshots, n_params)
        weight_centered = weight_matrix - weight_matrix.mean(dim=0)
        
        # we compute PCA using SVD
        U, S, Vt = torch.linalg.svd(weight_centered.T, full_matrices=False)
        
        # we take the 2 principal components
        d1 = U[:, 0]  # first principal component
        d2 = U[:, 1]  # second principal component
        
        # we print explained variance
        explained_var = (S**2) / (S**2).sum()
        print(f"PC1 explains {explained_var[0]*100:.2f}% of variance")
        print(f"PC2 explains {explained_var[1]*100:.2f}% of variance")
        print(f"PC1+PC2 explain {(explained_var[0]+explained_var[1])*100:.2f}% of variance")
        
    else:
        print("using random directions...")
        torch.manual_seed(seed)
        d1 = torch.randn_like(w_final)
        d1 = d1 / torch.norm(d1)
        
        d2 = torch.randn_like(w_final)
        d2 = d2 - torch.dot(d1, d2) * d1
        d2 = d2 / torch.norm(d2)
    
    print(f"d1 norm = {torch.norm(d1):.3f}, d2 norm = {torch.norm(d2):.3f}")
    print(f"orthogonality: d1·d2 = {torch.dot(d1, d2):.6f}")
    
    # we project trajectory
    trajectory_2d = []
    for w_snapshot in weight_snapshots:
        w_diff = w_snapshot - w_init
        alpha = torch.dot(w_diff, d1).item()
        beta = torch.dot(w_diff, d2).item()
        trajectory_2d.append((alpha, beta))
    
    alphas = [t[0] for t in trajectory_2d]
    betas = [t[1] for t in trajectory_2d]
    
    print(f"trajectory range α: [{min(alphas):.4f}, {max(alphas):.4f}]")
    print(f"trajectory range β: [{min(betas):.4f}, {max(betas):.4f}]")
    
    # we compute grid centered on trajectory
    alpha_center = (max(alphas) + min(alphas)) / 2
    beta_center = (max(betas) + min(betas)) / 2
    alpha_range = max(abs(max(alphas) - min(alphas)), 0.001) * grid_range
    beta_range = max(abs(max(betas) - min(betas)), 0.001) * grid_range
    
    alpha_grid = np.linspace(alpha_center - alpha_range, alpha_center + alpha_range, n_grid)
    beta_grid = np.linspace(beta_center - beta_range, beta_center + beta_range, n_grid)
    
    Alpha, Beta = np.meshgrid(alpha_grid, beta_grid)
    loss_landscape = np.zeros_like(Alpha)
    
    print(f"computing loss landscape on {n_grid}x{n_grid} grid...")
    
    for i in range(n_grid):
        if i % 10 == 0:
            print(f"  row {i}/{n_grid}")
        for j in range(n_grid):
            w_new = w_init + Alpha[i, j] * d1 + Beta[i, j] * d2
            set_weight_vector(w_new)
            
            with torch.no_grad():
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                loss_landscape[i, j] = loss.item()
    
    set_weight_vector(w_final)
    
    print(f"\nLoss landscape: min={loss_landscape.min():.6f}, max={loss_landscape.max():.6f}, mean={loss_landscape.mean():.6f}")
    
    return Alpha, Beta, loss_landscape, trajectory_2d



def plot_2d_landscape_with_trajectory(Alpha, Beta, loss_landscape, trajectory_2d, 
                                      loss_trajectory, config_name, save_path):
    """we plot 2d projected loss landscape with training trajectory"""
    
    alphas = [t[0] for t in trajectory_2d]
    betas = [t[1] for t in trajectory_2d]
    
    # we create 3d surface
    surface = go.Surface(
        x=Alpha, y=Beta, z=loss_landscape,
        colorscale='Viridis',
        opacity=0.9,
        name='loss landscape',
        colorbar=dict(title='loss', x=1.15)
    )
    
    # we create trajectory line in 3d
    trajectory = go.Scatter3d(
        x=alphas, y=betas, z=loss_trajectory,
        mode='lines+markers',
        line=dict(color='red', width=8),
        marker=dict(
            size=5,
            color=list(range(len(alphas))),
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title='epoch', x=1.0, len=0.5, y=0.25)
        ),
        name='training trajectory'
    )
    
    # we mark start and end points
    start_marker = go.Scatter3d(
        x=[alphas[0]], y=[betas[0]], z=[loss_trajectory[0]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='start'
    )
    
    end_marker = go.Scatter3d(
        x=[alphas[-1]], y=[betas[-1]], z=[loss_trajectory[-1]],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='diamond'),
        name='end'
    )
    
    fig = go.Figure(data=[surface, trajectory, start_marker, end_marker])
    
    fig.update_layout(
        title=f'loss landscape (2d random projection) - {config_name}',
        scene=dict(
            xaxis_title='direction 1 (α)',
            yaxis_title='direction 2 (β)',
            zaxis_title='loss',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1200,
        height=900
    )
    
    fig.write_html(save_path)
    print(f"saved 2d projection landscape to {save_path}")
    
    
    
def train_one_config(model, x_train, y_train, n_epochs, lr, config_dict, save_folder, 
                     compute_ntk_every=10, patience=10, min_delta=1e-12,
                     store_weight_snapshots=True, snapshot_every=100):
    """we train one mmnn configuration with early stopping on plateau"""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr)
    criterion = torch.nn.MSELoss()
    
    losses = []
    ntk_matrices = {}
    ntk_eigenvalues = {}
    
    # we store complete weight snapshots for landscape visualization
    weight_snapshots = []
    loss_snapshots = []
    
    # we save initial weights
    if store_weight_snapshots:
        initial_weights = torch.cat([p.data.view(-1).clone() for p in params])
        weight_snapshots.append(initial_weights)
        loss_snapshots.append(float('inf'))  # we set placeholder for initial loss
    
    print(f"\ntraining: {config_dict['config_name']}")
    print(f"trainable parameters: {sum(p.numel() for p in params)}")
    
    
    
    
    
    best_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    stop_epoch = n_epochs
    
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        
        # we store weight snapshots
        if store_weight_snapshots and epoch % snapshot_every == 0:
            snapshot = torch.cat([p.data.view(-1).clone() for p in params])
            weight_snapshots.append(snapshot)
            loss_snapshots.append(current_loss)
            
        if epoch % (10 * compute_ntk_every) == 0:
                
            
            print(f"epoch {epoch}/{n_epochs}")
            print('loss: ', current_loss)
            
            ntk, eigenvalues = compute_ntk_gram(model, x_train)
            # plt.close()
            # plt.figure()
            # # we add a bar
            # plt.matshow(ntk)
            # plt.colorbar()
            # plt.savefig(os.path.join(save_folder, f"{config_dict['config_name']}_ntk_{epoch}.png"))
            # plt.close()
            ntk_matrices[epoch] = ntk
            ntk_eigenvalues[epoch] = eigenvalues
        
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}: loss plateau detected")
            print(f"best loss: {best_loss:.6e}, current loss: {current_loss:.6e}")
            early_stopped = True
            stop_epoch = epoch
            break
    
    print(f"final loss: {losses[-1]:.6e}")
    if early_stopped:
        print(f"training stopped early at epoch {stop_epoch}")
        # we compute and plot 2d projected loss landscape
    
    
    if len(weight_snapshots) > 1:
        print("\ncomputing 2d projected loss landscape...")
        Alpha, Beta, loss_landscape, trajectory_2d = compute_loss_landscape_2d_projection(
            model, x_train, y_train, weight_snapshots,
            n_grid=50,
            grid_range=0.8,
            use_pca=True
        )
        
        landscape_path = os.path.join(save_folder, f"{config_dict['config_name']}_landscape_2d.html")
        plot_2d_landscape_with_trajectory(
            Alpha, Beta, loss_landscape, trajectory_2d,
            loss_snapshots, config_dict['config_name'], landscape_path
        )
        
    plt.close()
    plt.figure()
    plt.loglog(losses)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title(f'loss evolution - {config_dict["config_name"]}', fontsize=14)
    plt.savefig(os.path.join(save_folder, f"{config_dict['config_name']}_loss.png"))
    plt.close()
    os.makedirs(save_folder, exist_ok=True)
    
    
    # we create plots of eigenvalues vs epochs
    if len(ntk_eigenvalues) > 0:
        epochs_list = sorted(ntk_eigenvalues.keys())
        max_eigenvalues = []
        min_eigenvalues = []
        
        for epoch in epochs_list:
            eigs = ntk_eigenvalues[epoch]
            max_eigenvalues.append(eigs[-1].item())
            min_eigenvalues.append(eigs[0].item())
        
        #  maximum eigenvalue
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, max_eigenvalues, 'b-', linewidth=2)
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('eigenvalue', fontsize=12)
        plt.title(f'ntk maximum eigenvalue evolution - {config_dict["config_name"]}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        max_plot_path = os.path.join(save_folder, f"{config_dict['config_name']}_max_eigenvalue.png")
        plt.savefig(max_plot_path, dpi=150)
        plt.close()
        
        # minimum eigenvalue
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, min_eigenvalues, 'r-', linewidth=2)
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('eigenvalue', fontsize=12)
        plt.title(f'ntk minimum eigenvalue evolution - {config_dict["config_name"]}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        min_plot_path = os.path.join(save_folder, f"{config_dict['config_name']}_min_eigenvalue.png")
        plt.savefig(min_plot_path, dpi=150)
        plt.close()
        # plot prediction vs ground truth
        plt.figure()
        plt.plot(range(len(y_train)), y_train.cpu().numpy(), 'b-', label='true')
        plt.plot(range(len(y_train)), model(x_train).detach().cpu().numpy(), 'r--', label='predicted') 
        plt.legend()
        plt.savefig(os.path.join(save_folder, f"{config_dict['config_name']}_prediction.png"))
        plt.close()
        print("\n" + "="*80)
        print("all experiments completed")
        print(f"results in: {save_folder}")
        print("="*80)
        
            
        print(f"eigenvalue plots saved to {max_plot_path} and {min_plot_path}")
    
    config_path = os.path.join(save_folder, f"{config_dict['config_name']}_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    results = {
        "config_name": config_dict["config_name"],
        "losses": torch.tensor(losses),
        "ntk_matrices": ntk_matrices,
        "ntk_eigenvalues": ntk_eigenvalues,
        "final_loss": losses[-1],
        "model_config": config_dict,
        "early_stopped": early_stopped,
        "stop_epoch": stop_epoch,
        "best_loss": best_loss,
        "total_epochs": len(losses)
    }
    
    results_path = os.path.join(save_folder, f"{config_dict['config_name']}.pt")
    torch.save(results, results_path)
    
    print(f"saved to {results_path}")
    
    return results





def main():
    """we run all training experiments"""
    
    depths = [2]
    widths = [512, 1024, 2048, 4096, 8192]
    ranks = [5, 10, 15, 20, 25, 30, 40, 50]

    
    n_samples_1d = 30
    n_samples_2d = 100
    n_epochs = 100000
    lr = 0.001
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_folder = os.path.join(script_dir, "../../data/storage/mmnn_ntk_values", f"results_ntk_{timestamp}")
    base_folder = os.path.normpath(base_folder)
    os.makedirs(base_folder, exist_ok=True)
    
    print("="*80)
    print("starting mmnn training experiments")
    print("="*80)
    
    x_train_1d, y_train_1d = generate_data_1d(n_samples_1d, device=device)
    print(f"\ngenerated 1d data: x {x_train_1d.shape}, y {y_train_1d.shape}")
    
    x_train_2d, y_train_2d = generate_data_2d(n_samples_2d, device=device)
    print(f"generated 2d data: x {x_train_2d.shape}, y {y_train_2d.shape}")
    
    to_plot = y_train_2d.cpu().numpy()
    plt.figure()
    plt.plot(range(n_samples_2d), to_plot)
    plt.xlabel('sample index')
    plt.ylabel('x2')
    plt.title('2d data')
    plt.savefig(os.path.join(base_folder, "2d_data.png"))
    plt.close()
    
    
    
    configs = []
    for depth in depths:
        for width in widths:
            for rank in ranks:
                for data_type in ["2d"]:
                    configs.append({
                        "depth": depth,
                        "width": width,
                        "rank": rank,
                        "data_type": data_type
                    })
    
    print(f"\ntotal configurations: {len(configs)}")
    
    pbar = tqdm(configs)
    for config in pbar:
        depth = config["depth"]
        width = config["width"]
        rank = config["rank"]
        data_type = config["data_type"]
        
        pbar.set_description(f"D:{depth} W:{width} R:{rank} T:{data_type}")
        
        if data_type == "1d":
            input_dim = 1
            x_train = x_train_1d
            y_train = y_train_1d
        else:
            input_dim = 2
            x_train = x_train_2d
            y_train = y_train_2d
        
        ranks_list = [input_dim] + [rank] * (depth - 1) + [1]
        widths_list = [width] * depth
        
        config_name = f"d{depth}_w{width}_r{rank}_{data_type}"
        
        model = nets.MMNN(
            ranks=ranks_list,
            widths=widths_list,
            device=device,
            ResNet=False,
            fixWb=True,
            act_kind=["ReLU"]*depth
        )
        for layer in model.fcs:
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.zeros_(layer.bias)

        
        config_dict = {
            "depth": depth,
            "width": width,
            "rank": rank,
            "data_type": data_type,
            "ranks_list": ranks_list,
            "widths_list": widths_list,
            "n_epochs": n_epochs,
            "learning_rate": lr,
            "optimizer": "Adam",
            "activation": "ReLU",
            "resnet": False,
            "fix_wb": True,
            "n_samples": x_train.shape[0],
            "input_dim": input_dim,
            "config_name": config_name,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-12
        }
        
        try:
            train_one_config(
                model=model,
                x_train=x_train,
                y_train=y_train,
                n_epochs=n_epochs,
                lr=lr,
                config_dict=config_dict,
                save_folder=base_folder,
                compute_ntk_every=1000,
                patience=20,
                min_delta=1e-12,
                store_weight_snapshots=True,
                snapshot_every=10,
                
            )
        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()
            continue
        

if __name__ == "__main__":
    main()

