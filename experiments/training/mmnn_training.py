import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.mmnn.mmnn import MMNN

def oscillatory_function_1d(x):
    """
    we define the 1d oscillatory function f1(x) = cos(20π|x|^1.4) + 0.5cos(12π|x|^1.6)
    """
    return torch.cos(20 * np.pi * torch.abs(x)**1.4) + 0.5 * torch.cos(12 * np.pi * torch.abs(x)**1.6)

def oscillatory_function_2d(x1, x2):
    """
    we define the 2d oscillatory function with given parameters
    """
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

def generate_data_1d(n_samples=1000, x_range=(-1, 1), device="cuda"):
    """
    we generate training data for 1d function
    """
    x = torch.linspace(x_range[0], x_range[1], n_samples, device=device).reshape(-1, 1)
    y = oscillatory_function_1d(x)
    return x, y

def generate_data_2d(n_samples=1000, x_range=(-1, 1), device="cuda"):
    """
    we generate training data for 2d function
    """
    n_per_dim = int(np.sqrt(n_samples))
    x1 = torch.linspace(x_range[0], x_range[1], n_per_dim, device=device)
    x2 = torch.linspace(x_range[0], x_range[1], n_per_dim, device=device)
    x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing='ij')
    x1_flat = x1_grid.reshape(-1, 1)
    x2_flat = x2_grid.reshape(-1, 1)
    x = torch.cat([x1_flat, x2_flat], dim=1)
    y = oscillatory_function_2d(x[:, 0:1], x[:, 1:2])
    return x, y

def compute_ntk_gram(model, x, device="cuda"):
    """
    we compute the neural tangent kernel gram matrix for given model and data
    """
    n = x.shape[0]
    model.eval()
    
    ntk = torch.zeros((n, n), device=device)
    
    for i in range(n):
        model.zero_grad()
        x_i = x[i:i+1]
        output_i = model(x_i)
        
        grads_i = []
        for param in model.parameters():
            if param.requires_grad:
                model.zero_grad()
                output_i.backward(retain_graph=True)
                if param.grad is not None:
                    grads_i.append(param.grad.view(-1).clone())
        
        grad_i = torch.cat(grads_i)
        
        for j in range(n):
            model.zero_grad()
            x_j = x[j:j+1]
            output_j = model(x_j)
            
            grads_j = []
            for param in model.parameters():
                if param.requires_grad:
                    model.zero_grad()
                    output_j.backward(retain_graph=True)
                    if param.grad is not None:
                        grads_j.append(param.grad.view(-1).clone())
            
            grad_j = torch.cat(grads_j)
            ntk[i, j] = torch.dot(grad_i, grad_j)
    
    model.train()
    return ntk.cpu()

def train_mmnn(model, x_train, y_train, n_epochs=20000, lr=0.001, device="cuda", 
               config_name="default", save_dir="results", compute_ntk_every=1, 
               model_config=None):
    """
    we train the mmnn model with gradient descent and store ntk matrices
    """
    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    ntk_matrices = {}
    
    print(f"\ntraining configuration: {config_name}")
    print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"using relu activation: True")
    print(f"using resnet: {model.ResNet}")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"epoch {epoch}/{n_epochs}, loss: {loss.item():.6e}")
        
        if epoch % compute_ntk_every == 0:
            with torch.no_grad():
                ntk = compute_ntk_gram(model, x_train, device)
                ntk_matrices[epoch] = ntk
    
    final_loss = losses[-1]
    print(f"final loss: {final_loss:.6e}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if model_config is not None:
        config_path = os.path.join(save_dir, f"{config_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=4)
        print(f"configuration saved to {config_path}")
    
    results = {
        "config_name": config_name,
        "losses": torch.tensor(losses),
        "ntk_matrices": ntk_matrices,
        "final_loss": final_loss,
        "model_config": model_config
    }
    
    save_path = os.path.join(save_dir, f"{config_name}.pt")
    torch.save(results, save_path)
    
    print(f"results saved to {save_path}")
    
    return results

def main():
    """
    we run all training experiments with various configurations
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    
    depths = [2, 4, 6, 8, 10]
    widths = [128, 256, 512]
    ranks = [5, 10, 15, 20, 25, 30, 40, 50]
    
    n_samples_1d = 100
    n_samples_2d = 100
    n_epochs = 20000
    lr = 0.001
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/training/results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 80)
    print("starting mmnn training experiments")
    print("=" * 80)
    
    x_train_1d, y_train_1d = generate_data_1d(n_samples_1d, device=device)
    print(f"\ngenerated 1d training data: x shape {x_train_1d.shape}, y shape {y_train_1d.shape}")
    
    x_train_2d, y_train_2d = generate_data_2d(n_samples_2d, device=device)
    print(f"generated 2d training data: x shape {x_train_2d.shape}, y shape {y_train_2d.shape}")
    
    all_configs = []
    
    for depth in depths:
        for width in widths:
            for rank in ranks:
                for data_type in ["1d", "2d"]:
                    if data_type == "1d":
                        input_dim = 1
                        x_train = x_train_1d
                        y_train = y_train_1d
                    else:
                        input_dim = 2
                        x_train = x_train_2d
                        y_train = y_train_2d
                    
                    ranks_list = [input_dim] + [rank] * depth + [1]
                    widths_list = [width] * depth
                    
                    config_name = f"d{depth}_w{width}_r{rank}_{data_type}"
                    all_configs.append({
                        "depth": depth,
                        "width": width,
                        "rank": rank,
                        "data_type": data_type,
                        "config_name": config_name,
                        "ranks_list": ranks_list,
                        "widths_list": widths_list,
                        "x_train": x_train,
                        "y_train": y_train
                    })
    
    print(f"\ntotal number of configurations: {len(all_configs)}")
    
    for idx, config in enumerate(all_configs):
        print(f"\n{'=' * 80}")
        print(f"configuration {idx + 1}/{len(all_configs)}")
        print(f"{'=' * 80}")
        
        model = MMNN(
            ranks=config["ranks_list"],
            widths=config["widths_list"],
            device=device,
            ResNet=False,
            fixWb=True
        ).to(device)
        
        model_config_dict = {
            "depth": config["depth"],
            "width": config["width"],
            "rank": config["rank"],
            "data_type": config["data_type"],
            "ranks_list": config["ranks_list"],
            "widths_list": config["widths_list"],
            "n_epochs": n_epochs,
            "learning_rate": lr,
            "optimizer": "SGD",
            "activation": "ReLU",
            "resnet": False,
            "fix_wb": True,
            "n_samples": config["x_train"].shape[0],
            "input_dim": config["x_train"].shape[1]
        }
        
        try:
            results = train_mmnn(
                model=model,
                x_train=config["x_train"],
                y_train=config["y_train"],
                n_epochs=n_epochs,
                lr=lr,
                device=device,
                config_name=config["config_name"],
                save_dir=save_dir,
                compute_ntk_every=1,
                model_config=model_config_dict
            )
        except Exception as e:
            print(f"error during training: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("all experiments completed")
    print(f"results saved in: {save_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

