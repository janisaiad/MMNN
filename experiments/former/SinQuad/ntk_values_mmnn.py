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

def train_one_config(model, x_train, y_train, n_epochs, lr, config_dict, save_folder, compute_ntk_every=10, 
                     patience=10, min_delta=1e-12):
    """we train one mmnn configuration with early stopping on plateau"""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = torch.nn.MSELoss()
    
    losses = []
    ntk_matrices = {}
    ntk_eigenvalues = {}
    
    print(f"\ntraining: {config_dict['config_name']}")
    print(f"trainable parameters: {sum(p.numel() for p in params)}")
    
    best_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    stop_epoch = n_epochs
    
    for epoch in range(n_epochs):
        print(f"epoch {epoch}/{n_epochs}")
        optimizer.zero_grad()
        
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        
        
        print(f"epoch {epoch}/{n_epochs}")
        print('loss: ', current_loss)
        
        if epoch % compute_ntk_every == 0:
            
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
    
    depths = [2]#,4, 6, 8, 10]
    widths = [2048,4096,8192]
    ranks = [20, 25, 30, 40, 50]

    
    n_samples_1d = 30
    n_samples_2d = 100
    n_epochs = 20000
    lr = 0.0001
    
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
            act_kind=["R"]*depth
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
            "optimizer": "SGD",
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
                compute_ntk_every=10,
                patience=20,
                min_delta=1e-12
            )
        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()
            continue
        

if __name__ == "__main__":
    main()

