import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")  # we use non-interactive backend
import matplotlib.pyplot as plt
import time
import os
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys
from datetime import datetime


class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cuda", 
                 ResNet = False,
                 fixWb = False):
        super().__init__()
        """
        A class to configure the neural network model.
    
        Attributes:
            ranks (list[int]): A list where the i-th element represents the output dimension of the i-th layer.
                               For the j-th layer, ranks[j-1] is the input dimension and ranks[j] is the output dimension.
            
            widths (list[int]): A list where each element specifies the width of the corresponding layer.
            
            device (str): The device (CPU/GPU) on which the PyTorch code will be executed.
            
            ResNet (bool): Indicates whether to use ResNet architecture, which includes identity connections between layers.
            
            fixWb (bool): If True, the weights and biases are not updated during training.
        """
        
        self.product = 1
        for j in range(1,len(ranks)):
            self.product *= np.sqrt(widths[j-1] *ranks[j])
        self.ranks = ranks # 
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        self.inverse_widths = [1/width for width in self.widths]
        self.inverse_product = 1/self.product
        fc_sizes = [ ranks[0] ] 
        for j in range(self.depth):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs) # list of nn.Linear layers
        # mu-parameterization init: we zero biases and scale output heads by 1/sqrt(width)  # we implement mu parameterization with minimal changes
        for j, fc in enumerate(self.fcs):
            with torch.no_grad():
                
                if j % 2 == 1:  # we scale width→rank weights (odd indices) by 1/sqrt(hidden width)
                    hidden_width = widths[j // 2]  # we get hidden width for this block
                    fc.weight.normal_(mean=0.0, std=1.0/np.sqrt(hidden_width))  # we apply mu scaling
                    fc.bias.normal_(mean=0.0, std=1/np.sqrt(hidden_width))  # we set bias to normal 
                else:
                    fc.weight.normal_(mean=0.0, std=1/np.sqrt(ranks[j//2]))  # we keep unit variance for rank→width weights
                    fc.bias.normal_(mean=0.0, std=1/np.sqrt(ranks[j//2]))  # we keep unit variance for rank→width weights
        
        if fixWb: # if True, the weights and biases are not updated during training
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
 
    
    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            x = self.fcs[2*j](x)
            x = torch.relu(x)
            x = self.fcs[2*j+1](x) 
            if self.ResNet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x[:,:n] = x[:,:n] + x_id[:,:n]
        return x


# we define placeholder dataset classes for PDE benchmarks with PINN support
# the user will implement these later when datasets are installed
class PlaceholderPINNDataset(Dataset):
    """we provide a placeholder dataset class for PINN benchmarks"""
    def __init__(self, benchmark_name: str, split: str, n_samples: int = 1000, input_dim: int = 1, output_dim: int = 1, 
                 n_collocation: int = 1000, n_boundary: int = 100, n_initial: int = 100):
        self.benchmark_name = benchmark_name  # we store benchmark name
        self.split = split  # we store split
        self.n_samples = n_samples  # we store number of data samples
        self.n_collocation = n_collocation  # we store number of collocation points
        self.n_boundary = n_boundary  # we store number of boundary points
        self.n_initial = n_initial  # we store number of initial condition points
        self.input_dim = input_dim  # we store input dimension
        self.output_dim = output_dim  # we store output dimension
        
        # we generate dummy data for now
        np.random.seed(42)  # we set seed for reproducibility
        self.x_data = np.random.randn(n_samples, input_dim).astype(np.float32)  # we generate dummy data inputs
        self.y_data = np.random.randn(n_samples, output_dim).astype(np.float32)  # we generate dummy data outputs
        self.x_colloc = np.random.randn(n_collocation, input_dim).astype(np.float32)  # we generate collocation points
        self.x_boundary = np.random.randn(n_boundary, input_dim).astype(np.float32)  # we generate boundary points
        self.x_initial = np.random.randn(n_initial, input_dim).astype(np.float32)  # we generate initial condition points
        
    def __len__(self):
        return max(self.n_samples, self.n_collocation)  # we return max length
        
    def __getitem__(self, idx):
        # we return data points, collocation points, boundary points, and initial points
        idx_data = idx % self.n_samples if self.n_samples > 0 else 0  # we wrap index
        idx_colloc = idx % self.n_collocation if self.n_collocation > 0 else 0  # we wrap index
        idx_boundary = idx % self.n_boundary if self.n_boundary > 0 else 0  # we wrap index
        idx_initial = idx % self.n_initial if self.n_initial > 0 else 0  # we wrap index
        
        return {
            "x_data": torch.from_numpy(self.x_data[idx_data]),
            "y_data": torch.from_numpy(self.y_data[idx_data]),
            "x_colloc": torch.from_numpy(self.x_colloc[idx_colloc]),
            "x_boundary": torch.from_numpy(self.x_boundary[idx_boundary]),
            "x_initial": torch.from_numpy(self.x_initial[idx_initial]),
        }  # we return sample


class PlaceholderPDEDataset(Dataset):
    """we provide a placeholder dataset class for regular PDE benchmarks (non-PINN)"""
    def __init__(self, benchmark_name: str, split: str, n_samples: int = 1000, input_dim: int = 1, output_dim: int = 1):
        self.benchmark_name = benchmark_name  # we store benchmark name
        self.split = split  # we store split
        self.n_samples = n_samples  # we store number of samples
        self.input_dim = input_dim  # we store input dimension
        self.output_dim = output_dim  # we store output dimension
        
        # we generate dummy data for now
        np.random.seed(42)  # we set seed for reproducibility
        self.x = np.random.randn(n_samples, input_dim).astype(np.float32)  # we generate dummy inputs
        self.y = np.random.randn(n_samples, output_dim).astype(np.float32)  # we generate dummy outputs
        
    def __len__(self):
        return self.n_samples  # we return length
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])  # we return sample


def load_pde_dataset(benchmark_name: str, split: str, use_pinn: bool = False, **kwargs) -> Dataset:
    """
    we load a PDE benchmark dataset
    TODO: implement actual dataset loading when datasets are installed
    """
    # we define which benchmarks use PINN losses
    pinn_benchmarks = ["pinnacle"]  # we define PINN benchmarks
    
    benchmark_lower = benchmark_name.lower()  # we normalize name
    
    if benchmark_lower in pinn_benchmarks or use_pinn:  # we check if should use PINN
        # we use PINN dataset
        return PlaceholderPINNDataset(
            benchmark_lower, 
            split, 
            n_samples=kwargs.get("n_samples", 1000),
            input_dim=kwargs.get("input_dim", 2), 
            output_dim=kwargs.get("output_dim", 1),
            n_collocation=kwargs.get("n_collocation", 1000),
            n_boundary=kwargs.get("n_boundary", 100),
            n_initial=kwargs.get("n_initial", 100)
        )  # we return PINN dataset
    else:
        # we use regular dataset
        dataset_loaders = {
            "flowbench": lambda s: PlaceholderPDEDataset("flowbench", s, input_dim=kwargs.get("input_dim", 2), output_dim=kwargs.get("output_dim", 1)),
            "pdearena": lambda s: PlaceholderPDEDataset("pdearena", s, input_dim=kwargs.get("input_dim", 2), output_dim=kwargs.get("output_dim", 1)),
            "pdegym": lambda s: PlaceholderPDEDataset("pdegym", s, input_dim=kwargs.get("input_dim", 2), output_dim=kwargs.get("output_dim", 1)),
            "pdebench": lambda s: PlaceholderPDEDataset("pdebench", s, input_dim=kwargs.get("input_dim", 2), output_dim=kwargs.get("output_dim", 1)),
        }  # we define regular dataset loaders
        
        if benchmark_lower not in dataset_loaders:
            raise ValueError(f"unknown benchmark: {benchmark_name}")  # we raise error
        
        return dataset_loaders[benchmark_lower](split)  # we return dataset


@dataclass
class AblationConfig:
    """we store ablation study configuration"""
    benchmark_name: str  # we store benchmark name
    fixWb: bool  # we store fixWb flag
    rank: int  # we store rank (15 or width)
    num_layers: int = 6  # we set number of layers
    hidden_width: int = 1024  # we set hidden width
    num_epochs: int = 5000  # we set number of epochs
    batch_size: int = 100  # we set batch size
    lr_init: float = 0.001  # we set initial learning rate
    device: str = "cuda"  # we set device
    seed: int = 42  # we set seed
    n_train_samples: int = 1000  # we set training samples
    n_test_samples: int = 500  # we set test samples
    input_dim: int = 1  # we set input dimension
    output_dim: int = 1  # we set output dimension
    log_every: int = 50  # we log every N epochs
    save_every: int = 500  # we save plots every N epochs
    use_pinn: bool = False  # we set whether to use PINN losses
    n_collocation: int = 1000  # we set number of collocation points for PINN
    n_boundary: int = 100  # we set number of boundary points for PINN
    n_initial: int = 100  # we set number of initial condition points for PINN
    lambda_data: float = 1.0  # we set weight for data loss in PINN
    lambda_physics: float = 1.0  # we set weight for physics loss in PINN
    lambda_boundary: float = 1.0  # we set weight for boundary loss in PINN
    lambda_initial: float = 1.0  # we set weight for initial condition loss in PINN


def generate_ablation_configs() -> List[AblationConfig]:
    """we generate all ablation study configurations"""
    configs = []  # we initialize list
    
    benchmarks = ["flowbench", "pdearena", "pdegym", "pdebench", "pinnacle"]  # we define benchmarks (including PINNacle)
    fixWb_options = [False, True]  # we define fixWb options
    rank_options = [3, 6, 10, 15, 25, 50, None]  # we test multiple ranks, None indicates rank=width
    
    for benchmark in benchmarks:
        use_pinn = benchmark.lower() == "pinnacle"  # we use PINN for PINNacle
        for fixWb in fixWb_options:
            for rank_val in rank_options:
                if rank_val is None:  # we use width as rank
                    actual_rank = 1024  # we set rank to width
                    rank_label = "width"  # we set label
                else:
                    actual_rank = rank_val  # we use specified rank
                    rank_label = str(rank_val)  # we set label
                
                config = AblationConfig(
                    benchmark_name=benchmark,
                    fixWb=fixWb,
                    rank=actual_rank,
                    num_layers=6,
                    hidden_width=1024,
                    use_pinn=use_pinn,  # we set PINN flag
                )  # we create config
                configs.append(config)  # we append config
    
    return configs  # we return configs


def train_one_config(config: AblationConfig, output_dir: Path) -> Dict:
    """we train one configuration and save results"""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")  # we set device
    mydtype = torch.get_default_dtype()  # we get dtype
    
    # we setup logging to file
    log_file = output_dir / "training.log"  # we set log file path
    log_file.parent.mkdir(parents=True, exist_ok=True)  # we ensure directory exists
    
    class Tee:  # we create tee class to log to both file and console
        def __init__(self, file_path):
            self.file = open(file_path, 'w')  # we open file
            self.stdout = sys.stdout  # we save stdout
            
        def write(self, text):
            self.file.write(text)  # we write to file
            self.file.flush()  # we flush file
            self.stdout.write(text)  # we write to stdout
            
        def flush(self):
            self.file.flush()  # we flush file
            self.stdout.flush()  # we flush stdout
    
    tee = Tee(log_file)  # we create tee
    sys.stdout = tee  # we redirect stdout
    sys.stderr = tee  # we redirect stderr
    
    try:
        # we set random seeds
        torch.manual_seed(config.seed)  # we set torch seed
        np.random.seed(config.seed)  # we set numpy seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)  # we set cuda seed
        
        # we add FULL MLP label if rank equals width
        arch_label = "FULL_MLP (rank=width)" if config.rank == config.hidden_width else f"rank={config.rank}"
        print(f"\n{'='*80}")  # we print separator
        print(f"Training: {config.benchmark_name} | fixWb={config.fixWb} | {arch_label}")  # we print config
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # we print start time
        print(f"{'='*80}")  # we print separator
        
        # we load datasets
        try:
            train_dataset = load_pde_dataset(
                config.benchmark_name, 
                "train", 
                use_pinn=config.use_pinn,  # we pass PINN flag
                input_dim=config.input_dim, 
                output_dim=config.output_dim,
                n_samples=config.n_train_samples,
                n_collocation=config.n_collocation,
                n_boundary=config.n_boundary,
                n_initial=config.n_initial
            )  # we load train dataset
            test_dataset = load_pde_dataset(
                config.benchmark_name, 
                "test", 
                use_pinn=config.use_pinn,  # we pass PINN flag
                input_dim=config.input_dim, 
                output_dim=config.output_dim,
                n_samples=config.n_test_samples,
                n_collocation=config.n_collocation,
                n_boundary=config.n_boundary,
                n_initial=config.n_initial
            )  # we load test dataset
        except Exception as e:
            print(f"warning: could not load dataset for {config.benchmark_name}: {e}")  # we print warning
            print("using placeholder dataset")  # we print message
            if config.use_pinn:  # we check if PINN
                train_dataset = PlaceholderPINNDataset(config.benchmark_name, "train", config.n_train_samples, config.input_dim, config.output_dim, config.n_collocation, config.n_boundary, config.n_initial)  # we use PINN placeholder
                test_dataset = PlaceholderPINNDataset(config.benchmark_name, "test", config.n_test_samples, config.input_dim, config.output_dim, config.n_collocation, config.n_boundary, config.n_initial)  # we use PINN placeholder
            else:
                train_dataset = PlaceholderPDEDataset(config.benchmark_name, "train", config.n_train_samples, config.input_dim, config.output_dim)  # we use placeholder
                test_dataset = PlaceholderPDEDataset(config.benchmark_name, "test", config.n_test_samples, config.input_dim, config.output_dim)  # we use placeholder
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)  # we create train loader
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)  # we create test loader
        
        # we build model
        ranks = [config.input_dim] + [config.rank] * config.num_layers + [config.output_dim]  # we build ranks
        widths = [config.hidden_width] * (config.num_layers + 1)  # we build widths
        
        model = MMNN(
            ranks=ranks,
            widths=widths,
            device=device,
            ResNet=False,
            fixWb=config.fixWb
        )  # we create model
        
        print(f"total parameters: {sum(p.numel() for p in model.parameters())}")  # we print total params
        print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")  # we print trainable params
        
        # we setup training
        optimizer = optim.Adam(model.parameters(), lr=config.lr_init)  # we create optimizer
        criterion = nn.MSELoss()  # we create loss function
        
        # we define PINN loss function if needed
        def compute_pinn_loss(model, batch, device, dtype, config):
            """we compute PINN loss with data, physics, boundary, and initial condition terms"""
            if not config.use_pinn:  # we check if PINN is enabled
                return None  # we return None if not PINN
            
            x_data = batch["x_data"].to(device, dtype=dtype)  # we move data points
            y_data = batch["y_data"].to(device, dtype=dtype)  # we move data targets
            x_colloc = batch["x_colloc"].to(device, dtype=dtype)  # we move collocation points
            x_boundary = batch["x_boundary"].to(device, dtype=dtype)  # we move boundary points
            x_initial = batch["x_initial"].to(device, dtype=dtype)  # we move initial points
            
            # we compute data loss
            y_pred_data = model(x_data)  # we predict on data points
            loss_data = criterion(y_pred_data, y_data)  # we compute data loss
            
            # we compute physics loss (PDE residual)
            # TODO: implement actual PDE residual computation based on benchmark
            # for now we use a placeholder that requires gradients
            x_colloc.requires_grad_(True)  # we enable gradients for collocation points
            u_colloc = model(x_colloc)  # we predict on collocation points
            
            # we compute gradients for PDE residual
            # placeholder: we compute laplacian-like term as example
            if x_colloc.shape[1] >= 2:  # we check if 2D or higher
                u_x = torch.autograd.grad(u_colloc.sum(), x_colloc, create_graph=True, retain_graph=True)[0]  # we compute first derivatives
                u_xx = torch.autograd.grad(u_x.sum(), x_colloc, create_graph=True, retain_graph=True)[0]  # we compute second derivatives
                # we compute PDE residual (placeholder: -Δu - f, where f is approximated)
                pde_residual = u_xx.sum(dim=1, keepdim=True)  # we sum over spatial dimensions (simplified)
            else:  # we handle 1D case
                u_x = torch.autograd.grad(u_colloc.sum(), x_colloc, create_graph=True, retain_graph=True)[0]  # we compute first derivative
                u_xx = torch.autograd.grad(u_x.sum(), x_colloc, create_graph=True, retain_graph=True)[0]  # we compute second derivative
                pde_residual = u_xx  # we use second derivative as residual
            
            loss_physics = torch.mean(pde_residual ** 2)  # we compute physics loss
            
            # we compute boundary loss (placeholder: u should be close to boundary values)
            u_boundary = model(x_boundary)  # we predict on boundary points
            # TODO: implement actual boundary condition (e.g., u = g on boundary)
            loss_boundary = torch.mean(u_boundary ** 2)  # we use zero boundary condition as placeholder
            
            # we compute initial condition loss (placeholder: u should match initial values)
            u_initial = model(x_initial)  # we predict on initial points
            # TODO: implement actual initial condition (e.g., u(t=0) = u0)
            loss_initial = torch.mean(u_initial ** 2)  # we use zero initial condition as placeholder
            
            # we combine losses with weights
            total_loss = (config.lambda_data * loss_data + 
                         config.lambda_physics * loss_physics +
                         config.lambda_boundary * loss_boundary +
                         config.lambda_initial * loss_initial)  # we combine losses
            
            return total_loss, {
                "loss_data": loss_data.item(),
                "loss_physics": loss_physics.item(),
                "loss_boundary": loss_boundary.item(),
                "loss_initial": loss_initial.item(),
            }  # we return total loss and components
        
        # we initialize tracking variables
        errors_train = []  # we store training errors
        errors_test = []  # we store test errors
        errors_test_max = []  # we store max test errors
        all_losses = []  # we store all losses
        losses_std = []  # we store loss std
        epoch_durations = []  # we store epoch durations
        pinn_loss_components = []  # we store PINN loss components if using PINN
        
        time_start = time.time()  # we start timer
        
        # we train
        model.train()  # we set training mode
        for epoch in tqdm(range(1, config.num_epochs + 1), desc=f"{config.benchmark_name} fixWb={config.fixWb} rank={config.rank}"):  # we loop epochs
            epoch_start = time.time()  # we start epoch timer
            
            epoch_losses = []  # we store epoch losses
            epoch_pinn_components = []  # we store PINN loss components
            
            for batch in train_loader:  # we loop batches
                optimizer.zero_grad()  # we zero gradients
                
                if config.use_pinn:  # we check if using PINN
                    # we compute PINN loss
                    pinn_result = compute_pinn_loss(model, batch, device, mydtype, config)  # we compute PINN loss
                    if pinn_result is not None:  # we check if result is valid
                        loss, pinn_components = pinn_result  # we unpack result
                        loss.backward()  # we backward pass
                        epoch_pinn_components.append(pinn_components)  # we store components
                    else:
                        # we fallback to regular loss if PINN computation fails
                        x_data = batch["x_data"].to(device, dtype=mydtype)  # we move data
                        y_data = batch["y_data"].to(device, dtype=mydtype)  # we move targets
                        outputs = model(x_data)  # we forward pass
                        loss = criterion(outputs, y_data)  # we compute loss
                        loss.backward()  # we backward pass
                else:
                    # we use regular supervised loss
                    inputs, targets = batch  # we unpack batch
                    inputs = inputs.to(device, dtype=mydtype)  # we move to device
                    targets = targets.to(device, dtype=mydtype)  # we move to device
                    outputs = model(inputs)  # we forward pass
                    loss = criterion(outputs, targets)  # we compute loss
                    loss.backward()  # we backward pass
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # we clip gradients
                optimizer.step()  # we update weights
                
                epoch_losses.append(loss.item())  # we store loss
            
            avg_loss = np.mean(epoch_losses)  # we compute average loss
            all_losses.append(avg_loss)  # we store loss
            
            # we store PINN loss components
            if config.use_pinn and len(epoch_pinn_components) > 0:  # we check if PINN
                avg_pinn = {k: np.mean([c[k] for c in epoch_pinn_components]) for k in epoch_pinn_components[0].keys()}  # we average components
                pinn_loss_components.append(avg_pinn)  # we store components
            
            epoch_duration = time.time() - epoch_start  # we compute duration
            epoch_durations.append(epoch_duration)  # we store duration
            
            # we evaluate periodically
            if epoch % config.log_every == 0:  # we check if should log
                model.eval()  # we set eval mode
                with torch.no_grad():
                    # we compute training error
                    train_error = avg_loss  # we use current loss
                    errors_train.append(train_error)  # we store error
                    
                    # we compute test error
                    test_losses = []  # we store test losses
                    test_errors_abs = []  # we store absolute errors
                    for test_batch in test_loader:  # we loop test batches
                        if config.use_pinn:  # we check if PINN
                            test_x_data = test_batch["x_data"].to(device, dtype=mydtype)  # we move data
                            test_y_data = test_batch["y_data"].to(device, dtype=mydtype)  # we move targets
                            test_outputs = model(test_x_data)  # we forward pass
                            test_loss = criterion(test_outputs, test_y_data)  # we compute loss
                            test_errors_abs.append(torch.abs(test_outputs - test_y_data).cpu().numpy())  # we store abs errors
                        else:
                            test_inputs, test_targets = test_batch  # we unpack batch
                            test_inputs = test_inputs.to(device, dtype=mydtype)  # we move to device
                            test_targets = test_targets.to(device, dtype=mydtype)  # we move to device
                            test_outputs = model(test_inputs)  # we forward pass
                            test_loss = criterion(test_outputs, test_targets)  # we compute loss
                            test_errors_abs.append(torch.abs(test_outputs - test_targets).cpu().numpy())  # we store abs errors
                        test_losses.append(test_loss.item())  # we store loss
                    
                    test_error = np.mean(test_losses)  # we compute mean test error
                    test_error_max = np.max([np.max(err) for err in test_errors_abs])  # we compute max error
                    errors_test.append(test_error)  # we store error
                    errors_test_max.append(test_error_max)  # we store max error
                    
                    # we compute loss std
                    if len(all_losses) >= 50:  # we check if enough losses
                        losses_std.append(np.std(np.log10(all_losses[-50:])))  # we compute std
                    else:
                        losses_std.append(0.0)  # we set zero
                    
                    print(f"\nepoch {epoch}/{config.num_epochs}")  # we print epoch
                    print(f"  train error (MSE): {train_error:.2e}")  # we print train error
                    print(f"  test error (MSE): {test_error:.2e}")  # we print test error
                    print(f"  test error (MAX): {test_error_max:.2e}")  # we print max error
                    if config.use_pinn and len(epoch_pinn_components) > 0:  # we check if PINN
                        avg_components = {k: np.mean([c[k] for c in epoch_pinn_components]) for k in epoch_pinn_components[0].keys()}  # we average components
                        print(f"  PINN losses - data: {avg_components.get('loss_data', 0):.2e}, physics: {avg_components.get('loss_physics', 0):.2e}, boundary: {avg_components.get('loss_boundary', 0):.2e}, initial: {avg_components.get('loss_initial', 0):.2e}")  # we print PINN components
                    print(f"  time: {time.time() - time_start:.2f}s")  # we print time
                
                model.train()  # we set training mode
            
            # we save plots periodically
            if epoch % config.save_every == 0:  # we check if should save
                model.eval()  # we set eval mode
                with torch.no_grad():
                    # we get sample predictions
                    sample_batch = next(iter(test_loader))  # we get sample
                    if config.use_pinn:  # we check if PINN
                        sample_inputs = sample_batch["x_data"].to(device, dtype=mydtype)  # we move data
                        sample_targets = sample_batch["y_data"]  # we get targets
                    else:
                        sample_inputs, sample_targets = sample_batch  # we unpack batch
                        sample_inputs = sample_inputs.to(device, dtype=mydtype)  # we move to device
                    sample_outputs = model(sample_inputs)  # we forward pass
                    
                    # we plot if 1D
                    if config.input_dim == 1:  # we check if 1D
                        x_plot = sample_inputs.cpu().numpy().flatten()  # we get x values
                        y_true = sample_targets.numpy().flatten()  # we get true y
                        y_pred = sample_outputs.cpu().numpy().flatten()  # we get pred y
                        
                        # we sort for plotting
                        sort_idx = np.argsort(x_plot)  # we get sort indices
                        x_plot = x_plot[sort_idx]  # we sort x
                        y_true = y_true[sort_idx]  # we sort y_true
                        y_pred = y_pred[sort_idx]  # we sort y_pred
                        
                        fig, ax = plt.subplots(figsize=(8, 5))  # we create figure
                        ax.plot(x_plot, y_true, 'b-', label='true', linewidth=2)  # we plot true
                        ax.plot(x_plot, y_pred, 'r--', label='predicted', linewidth=2)  # we plot pred
                        ax.set_xlabel('x')  # we set xlabel
                        ax.set_ylabel('y')  # we set ylabel
                        # we add FULL MLP label if rank equals width
                        arch_label = "FULL_MLP" if config.rank == config.hidden_width else f"rank={config.rank}"
                        ax.set_title(f'{config.benchmark_name} | fixWb={config.fixWb} | {arch_label} | epoch {epoch}')  # we set title
                        ax.legend()  # we add legend
                        ax.grid(True, alpha=0.3)  # we add grid
                        plt.tight_layout()  # we adjust layout
                        plt.savefig(output_dir / f'prediction_epoch{epoch}.png', dpi=100)  # we save plot
                        plt.close()  # we close figure
                
                model.train()  # we set training mode
        
        total_time = time.time() - time_start  # we compute total time
        
        # we save model
        torch.save(model.state_dict(), output_dir / 'model_parameters.pth')  # we save model
        
        # we save all tensors for plotting after training
        print("\nsaving all tensors for post-training analysis...")  # we print message
        model.eval()  # we set eval mode
        with torch.no_grad():
            # we collect all test data
            all_test_inputs = []  # we store test inputs
            all_test_targets = []  # we store test targets
            all_test_predictions = []  # we store test predictions
            all_test_errors = []  # we store test errors
            
            for test_batch in test_loader:  # we loop test batches
                if config.use_pinn:  # we check if PINN
                    test_x_data = test_batch["x_data"].to(device, dtype=mydtype)  # we move data
                    test_y_data = test_batch["y_data"].to(device, dtype=mydtype)  # we move targets
                    test_pred = model(test_x_data)  # we forward pass
                    all_test_inputs.append(test_x_data.cpu())  # we store inputs
                    all_test_targets.append(test_y_data.cpu())  # we store targets
                    all_test_predictions.append(test_pred.cpu())  # we store predictions
                    all_test_errors.append((test_pred.cpu() - test_y_data.cpu()))  # we store errors
                else:
                    test_inputs, test_targets = test_batch  # we unpack batch
                    test_inputs = test_inputs.to(device, dtype=mydtype)  # we move to device
                    test_targets = test_targets.to(device, dtype=mydtype)  # we move to device
                    test_pred = model(test_inputs)  # we forward pass
                    all_test_inputs.append(test_inputs.cpu())  # we store inputs
                    all_test_targets.append(test_targets.cpu())  # we store targets
                    all_test_predictions.append(test_pred.cpu())  # we store predictions
                    all_test_errors.append((test_pred.cpu() - test_targets.cpu()))  # we store errors
            
            # we concatenate all test data
            test_inputs_tensor = torch.cat(all_test_inputs, dim=0)  # we concatenate inputs
            test_targets_tensor = torch.cat(all_test_targets, dim=0)  # we concatenate targets
            test_predictions_tensor = torch.cat(all_test_predictions, dim=0)  # we concatenate predictions
            test_errors_tensor = torch.cat(all_test_errors, dim=0)  # we concatenate errors
            
            # we collect training data (sample for memory efficiency)
            all_train_inputs = []  # we store train inputs
            all_train_targets = []  # we store train targets
            all_train_predictions = []  # we store train predictions
            n_train_samples_to_save = min(1000, len(train_dataset))  # we limit samples to save memory
            
            for i, batch in enumerate(train_loader):  # we loop train batches
                if i * config.batch_size >= n_train_samples_to_save:  # we check if enough samples
                    break  # we break loop
                if config.use_pinn:  # we check if PINN
                    train_x_data = batch["x_data"].to(device, dtype=mydtype)  # we move data
                    train_y_data = batch["y_data"].to(device, dtype=mydtype)  # we move targets
                    train_pred = model(train_x_data)  # we forward pass
                    all_train_inputs.append(train_x_data.cpu())  # we store inputs
                    all_train_targets.append(train_y_data.cpu())  # we store targets
                    all_train_predictions.append(train_pred.cpu())  # we store predictions
                else:
                    train_inputs, train_targets = batch  # we unpack batch
                    train_inputs = train_inputs.to(device, dtype=mydtype)  # we move to device
                    train_targets = train_targets.to(device, dtype=mydtype)  # we move to device
                    train_pred = model(train_inputs)  # we forward pass
                    all_train_inputs.append(train_inputs.cpu())  # we store inputs
                    all_train_targets.append(train_targets.cpu())  # we store targets
                    all_train_predictions.append(train_pred.cpu())  # we store predictions
            
            # we concatenate training data
            if len(all_train_inputs) > 0:  # we check if we have data
                train_inputs_tensor = torch.cat(all_train_inputs, dim=0)  # we concatenate inputs
                train_targets_tensor = torch.cat(all_train_targets, dim=0)  # we concatenate targets
                train_predictions_tensor = torch.cat(all_train_predictions, dim=0)  # we concatenate predictions
            else:
                train_inputs_tensor = torch.empty(0)  # we create empty tensor
                train_targets_tensor = torch.empty(0)  # we create empty tensor
                train_predictions_tensor = torch.empty(0)  # we create empty tensor
            
            # we initialize PINN tensors dict (we define it early to avoid scope issues)
            pinn_tensors = {}  # we initialize empty dict
            
            # we save PINN-specific tensors if using PINN
            if config.use_pinn:  # we check if PINN
                # we collect collocation, boundary, and initial points
                all_colloc_inputs = []  # we store collocation inputs
                all_colloc_predictions = []  # we store collocation predictions
                all_boundary_inputs = []  # we store boundary inputs
                all_boundary_predictions = []  # we store boundary predictions
                all_initial_inputs = []  # we store initial inputs
                all_initial_predictions = []  # we store initial predictions
                
                # we sample from train loader for PINN points
                for i, batch in enumerate(train_loader):  # we loop batches
                    if i >= 10:  # we limit to first 10 batches
                        break  # we break
                    x_colloc = batch["x_colloc"].to(device, dtype=mydtype)  # we move collocation
                    x_boundary = batch["x_boundary"].to(device, dtype=mydtype)  # we move boundary
                    x_initial = batch["x_initial"].to(device, dtype=mydtype)  # we move initial
                    
                    u_colloc = model(x_colloc)  # we predict on collocation
                    u_boundary = model(x_boundary)  # we predict on boundary
                    u_initial = model(x_initial)  # we predict on initial
                    
                    all_colloc_inputs.append(x_colloc.cpu())  # we store collocation inputs
                    all_colloc_predictions.append(u_colloc.cpu())  # we store collocation predictions
                    all_boundary_inputs.append(x_boundary.cpu())  # we store boundary inputs
                    all_boundary_predictions.append(u_boundary.cpu())  # we store boundary predictions
                    all_initial_inputs.append(x_initial.cpu())  # we store initial inputs
                    all_initial_predictions.append(u_initial.cpu())  # we store initial predictions
                
                if len(all_colloc_inputs) > 0:  # we check if we have data
                    colloc_inputs_tensor = torch.cat(all_colloc_inputs, dim=0)  # we concatenate
                    colloc_predictions_tensor = torch.cat(all_colloc_predictions, dim=0)  # we concatenate
                    boundary_inputs_tensor = torch.cat(all_boundary_inputs, dim=0)  # we concatenate
                    boundary_predictions_tensor = torch.cat(all_boundary_predictions, dim=0)  # we concatenate
                    initial_inputs_tensor = torch.cat(all_initial_inputs, dim=0)  # we concatenate
                    initial_predictions_tensor = torch.cat(all_initial_predictions, dim=0)  # we concatenate
                    
                    # we add PINN tensors to saved dict
                    pinn_tensors = {
                        "colloc_inputs": colloc_inputs_tensor,  # we save collocation inputs
                        "colloc_predictions": colloc_predictions_tensor,  # we save collocation predictions
                        "boundary_inputs": boundary_inputs_tensor,  # we save boundary inputs
                        "boundary_predictions": boundary_predictions_tensor,  # we save boundary predictions
                        "initial_inputs": initial_inputs_tensor,  # we save initial inputs
                        "initial_predictions": initial_predictions_tensor,  # we save initial predictions
                    }  # we create PINN tensors dict
            
            # we save all tensors
            save_torch_dict = {
                "test_inputs": test_inputs_tensor,  # we save test inputs
                "test_targets": test_targets_tensor,  # we save test targets
                "test_predictions": test_predictions_tensor,  # we save test predictions
                "test_errors": test_errors_tensor,  # we save test errors
                "config": asdict(config),  # we save config for reference
            }  # we create base dict
            
            if len(train_inputs_tensor) > 0:  # we check if we have train data
                save_torch_dict.update({
                    "train_inputs": train_inputs_tensor,  # we save train inputs
                    "train_targets": train_targets_tensor,  # we save train targets
                    "train_predictions": train_predictions_tensor,  # we save train predictions
                })  # we add train data
            
            if config.use_pinn and len(pinn_tensors) > 0:  # we check if PINN
                save_torch_dict.update(pinn_tensors)  # we add PINN tensors
            
            torch.save(save_torch_dict, output_dir / 'all_tensors.pt')  # we save tensors
            
            # we also save as numpy arrays for easier plotting
            save_npz_dict = {
                "test_inputs": test_inputs_tensor.numpy(),  # we save test inputs
                "test_targets": test_targets_tensor.numpy(),  # we save test targets
                "test_predictions": test_predictions_tensor.numpy(),  # we save test predictions
                "test_errors": test_errors_tensor.numpy(),  # we save test errors
            }  # we create base dict
            
            if len(train_inputs_tensor) > 0:  # we check if we have train data
                save_npz_dict.update({
                    "train_inputs": train_inputs_tensor.numpy(),  # we save train inputs
                    "train_targets": train_targets_tensor.numpy(),  # we save train targets
                    "train_predictions": train_predictions_tensor.numpy(),  # we save train predictions
                })  # we add train data
            
            if config.use_pinn and len(pinn_tensors) > 0:  # we check if PINN
                save_npz_dict.update({
                    "colloc_inputs": pinn_tensors["colloc_inputs"].numpy(),  # we save collocation inputs
                    "colloc_predictions": pinn_tensors["colloc_predictions"].numpy(),  # we save collocation predictions
                    "boundary_inputs": pinn_tensors["boundary_inputs"].numpy(),  # we save boundary inputs
                    "boundary_predictions": pinn_tensors["boundary_predictions"].numpy(),  # we save boundary predictions
                    "initial_inputs": pinn_tensors["initial_inputs"].numpy(),  # we save initial inputs
                    "initial_predictions": pinn_tensors["initial_predictions"].numpy(),  # we save initial predictions
                })  # we add PINN data
            
            np.savez(output_dir / 'all_tensors.npz', **save_npz_dict)  # we save as numpy
            
            print(f"  saved test tensors: inputs {test_inputs_tensor.shape}, targets {test_targets_tensor.shape}, predictions {test_predictions_tensor.shape}")  # we print info
            if len(train_inputs_tensor) > 0:  # we check if we have train data
                print(f"  saved train tensors: inputs {train_inputs_tensor.shape}, targets {train_targets_tensor.shape}, predictions {train_predictions_tensor.shape}")  # we print info
        
        # we save errors and losses
        save_dict = {
            "test": np.array(errors_test),
            "testmax": np.array(errors_test_max),
            "train": np.array(errors_train),
            "all_losses": np.array(all_losses),
            "losses_std": np.array(losses_std),
            "time": total_time
        }  # we create save dict
        
        if config.use_pinn and len(pinn_loss_components) > 0:  # we check if PINN components exist
            # we save PINN loss components
            pinn_dict = {k: np.array([c[k] for c in pinn_loss_components]) for k in pinn_loss_components[0].keys()}  # we extract components
            save_dict.update(pinn_dict)  # we add PINN components
        
        np.savez(output_dir / "errors.npz", **save_dict)  # we save errors
        
        # we save results json
        results = {
            "config": asdict(config),  # we save config
            "final_train_error": float(errors_train[-1]) if len(errors_train) > 0 else None,  # we save final train error
            "final_test_error": float(errors_test[-1]) if len(errors_test) > 0 else None,  # we save final test error
            "final_test_error_max": float(errors_test_max[-1]) if len(errors_test_max) > 0 else None,  # we save final max error
            "training_time_seconds": float(total_time),  # we save training time
            "total_parameters": int(sum(p.numel() for p in model.parameters())),  # we save total params
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),  # we save trainable params
            "epochs_run": int(len(all_losses)),  # we save epochs run
            "mean_epoch_time_seconds": float(np.mean(epoch_durations)) if len(epoch_durations) > 0 else None,  # we save mean epoch time
            "epoch_durations": [float(d) for d in epoch_durations],  # we save epoch durations
        }  # we create results dict
        
        with open(output_dir / "results.json", "w") as f:  # we open file
            json.dump(results, f, indent=4)  # we save json
        
        # we save config json
        with open(output_dir / "config.json", "w") as f:  # we open file
            json.dump(asdict(config), f, indent=4)  # we save config
        
        # we plot loss evolution
        if len(all_losses) > 0:  # we check if losses exist
            fig = plt.figure(figsize=(8, 5))  # we create figure
            plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1)  # we plot losses
            plt.xlabel('Epoch')  # we set xlabel
            plt.ylabel('Loss (log scale)')  # we set ylabel
            # we add FULL MLP label if rank equals width
            arch_label = "FULL_MLP" if config.rank == config.hidden_width else f"rank={config.rank}"
            plt.title(f'Training Loss Evolution\n{config.benchmark_name} | fixWb={config.fixWb} | {arch_label}')  # we set title
            plt.grid(True, alpha=0.3)  # we add grid
            plt.tight_layout()  # we adjust layout
            plt.savefig(output_dir / 'loss_evolution.png', dpi=100)  # we save plot
            plt.close()  # we close figure
        
        # we plot error evolution
        if len(errors_test) > 0 and len(errors_train) > 0:  # we check if errors exist
            fig = plt.figure(figsize=(8, 5))  # we create figure
            n_train = len(errors_train)  # we get train length
            n_test = len(errors_test)  # we get test length
            plt.plot(np.linspace(1, n_train, n_train)*config.log_every, np.log10(errors_train), label="log10 training error", linewidth=2)  # we plot train
            plt.plot(np.linspace(1, n_test, n_test)*config.log_every, np.log10(errors_test), label="log10 test error", linewidth=2)  # we plot test
            plt.xlabel('Epoch')  # we set xlabel
            plt.ylabel('log10(error)')  # we set ylabel
            # we add FULL MLP label if rank equals width
            arch_label = "FULL_MLP" if config.rank == config.hidden_width else f"rank={config.rank}"
            plt.title(f'Error Evolution\n{config.benchmark_name} | fixWb={config.fixWb} | {arch_label}')  # we set title
            plt.grid(True, alpha=0.3)  # we add grid
            plt.legend()  # we add legend
            plt.tight_layout()  # we adjust layout
            plt.savefig(output_dir / 'error_evolution.png', dpi=100)  # we save plot
            plt.close()  # we close figure
        
        print(f"\nresults saved to {output_dir}")  # we print message
        print(f"final test error: {errors_test[-1]:.2e}" if len(errors_test) > 0 else "no test errors")  # we print final error
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # we print completion time
        
    finally:
        # we restore stdout/stderr and close log file
        sys.stdout = tee.stdout  # we restore stdout
        sys.stderr = tee.stdout  # we restore stderr
        tee.file.close()  # we close file
    
    return results  # we return results


def train_config_wrapper(args):
    """we wrap training function for parallel execution"""
    config_dict, output_dir_str, device_str = args  # we unpack arguments
    config = AblationConfig(**config_dict)  # we recreate config
    output_dir = Path(output_dir_str)  # we create path
    config.device = device_str  # we set device
    try:
        results = train_one_config(config, output_dir)  # we train config
        return results  # we return results
    except Exception as e:
        print(f"error training {config.benchmark_name} fixWb={config.fixWb} rank={config.rank}: {e}")  # we print error
        import traceback  # we import traceback
        traceback.print_exc()  # we print traceback
        return None  # we return None on error


def main():
    """we run the ablation study"""
    # we setup main log file
    base_output_dir = Path("experiments/table/results")  # we set base dir
    base_output_dir.mkdir(parents=True, exist_ok=True)  # we create dir
    main_log_file = base_output_dir / f"main_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # we set main log file
    
    class Tee:  # we create tee class for main logging
        def __init__(self, file_path):
            self.file = open(file_path, 'w')  # we open file
            self.stdout = sys.stdout  # we save stdout
            
        def write(self, text):
            self.file.write(text)  # we write to file
            self.file.flush()  # we flush file
            self.stdout.write(text)  # we write to stdout
            
        def flush(self):
            self.file.flush()  # we flush file
            self.stdout.flush()  # we flush stdout
    
    tee = Tee(main_log_file)  # we create tee
    sys.stdout = tee  # we redirect stdout
    sys.stderr = tee  # we redirect stderr
    
    try:
        # we check available devices
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0  # we count GPUs
        n_cpus = mp.cpu_count()  # we count CPUs
        use_parallel = n_gpus > 1 or (n_gpus == 0 and n_cpus > 1)  # we decide if parallel
        
        print(f"ablation study started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # we print start time
        print(f"main log file: {main_log_file}")  # we print log file location
        print(f"available GPUs: {n_gpus}, CPUs: {n_cpus}")  # we print info
        print(f"parallel training: {use_parallel}")  # we print info
    
        # we generate all configurations
        configs = generate_ablation_configs()  # we generate configs
        print(f"\ngenerated {len(configs)} configurations")  # we print count
        
        # we prepare configurations for parallel execution
        all_results = []  # we store all results
        
        if use_parallel and len(configs) > 1:  # we check if should use parallel
            print(f"\nrunning {len(configs)} configurations in parallel...")  # we print message
            
            # we prepare arguments for parallel execution
            parallel_args = []  # we initialize list
            for idx, config in enumerate(configs):  # we loop configs
                output_dir = base_output_dir / f"{config.benchmark_name}_fixWb{config.fixWb}_rank{config.rank}_run{idx}"  # we set output dir
                output_dir.mkdir(parents=True, exist_ok=True)  # we create dir
                
                # we assign device based on GPU availability
                if n_gpus > 0:  # we check if GPUs available
                    device_str = f"cuda:{idx % n_gpus}"  # we assign GPU by round-robin
                else:
                    device_str = "cpu"  # we use CPU
                
                config_dict = asdict(config)  # we convert to dict
                parallel_args.append((config_dict, str(output_dir), device_str))  # we append args
            
            # we run in parallel
            max_workers = min(n_gpus if n_gpus > 0 else n_cpus, len(configs))  # we set max workers
            with ProcessPoolExecutor(max_workers=max_workers) as executor:  # we create executor
                futures = {executor.submit(train_config_wrapper, args): args for args in parallel_args}  # we submit jobs
                
                # we collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="training configurations"):  # we loop futures
                    args = futures[future]  # we get args
                    try:
                        result = future.result()  # we get result
                        if result is not None:  # we check if valid
                            all_results.append(result)  # we append result
                    except Exception as e:  # we catch exceptions
                        print(f"error in parallel execution: {e}")  # we print error
                        import traceback  # we import traceback
                        traceback.print_exc()  # we print traceback
        else:
            # we run sequentially
            print(f"\nrunning {len(configs)} configurations sequentially...")  # we print message
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # we set device
            
            for idx, config in enumerate(tqdm(configs, desc="training configurations")):  # we loop configs
                config.output_dir = base_output_dir / f"{config.benchmark_name}_fixWb{config.fixWb}_rank{config.rank}_run{idx}"  # we set output dir
                config.output_dir.mkdir(parents=True, exist_ok=True)  # we create dir
                config.device = str(device)  # we set device
                
                try:
                    results = train_one_config(config, config.output_dir)  # we train config
                    all_results.append(results)  # we append results
                except Exception as e:
                    print(f"error training {config.benchmark_name} fixWb={config.fixWb} rank={config.rank}: {e}")  # we print error
                    import traceback  # we import traceback
                    traceback.print_exc()  # we print traceback
                    continue  # we continue
        
        # we save summary
        summary_path = base_output_dir / "ablation_summary.json"  # we set summary path
        with open(summary_path, "w") as f:  # we open file
            json.dump(all_results, f, indent=4)  # we save summary
        
        print(f"\n{'='*80}")  # we print separator
        print(f"ablation study complete")  # we print message
        print(f"completed {len(all_results)}/{len(configs)} configurations")  # we print completion
        print(f"results saved to {base_output_dir}")  # we print path
        print(f"summary saved to {summary_path}")  # we print summary path
        print(f"main log saved to {main_log_file}")  # we print main log location
        print(f"completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # we print completion time
        print(f"{'='*80}")  # we print separator
        
    finally:
        # we restore stdout/stderr and close log file
        sys.stdout = tee.stdout  # we restore stdout
        sys.stderr = tee.stdout  # we restore stderr
        tee.file.close()  # we close file


if __name__ == "__main__":
    main()  # we run main
