# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:25:00 2025

@author: Shijun Zhang 
This script runs a large-scale experiment to generate and save optimization
landscapes for a variety of neural network configurations and target functions.
"""
from __future__ import print_function
import numpy as np
import time
import nets
import myplotly
import torch
import os
from itertools import product
from tqdm import tqdm

# we set up the device
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"training on device: {device}")
torch.set_default_dtype(torch.float32)
mydtype = torch.get_default_dtype()

# --- Helper Functions for Landscape Generation ---

def get_weights(net):
    """ we extract parameters from net """
    return [p.data for p in net.parameters()]

def set_weights(net, weights):
    """ we set parameters of net """
    for p, w in zip(net.parameters(), weights):
        p.data.copy_(w)

def get_random_weights(weights):
    """ we create a random direction with the same dimension as weights """
    return [torch.randn(p.size()).to(p.device) for p in weights]

def normalize_directions_for_weights(direction, weights):
    """ we normalize the direction wrt the weights """
    for d, w in zip(direction, weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))
        
# --- Building blocks for complex target functions ---

def main_y(x, k=128):
    """ The highly oscillatory sawtooth-like base function. """
    y = torch.abs(x)**2
    y = k*y - 2*torch.floor((k*y+1)/2)
    return torch.abs(y)**2

def y1_rational(x):
    """ A smooth rational function. """
    return (8*x**4 + 1) / (1 + 10*x**2)

def y1_trig_v1(x):
    """ A high-frequency trigonometric function. """
    return 0.6*torch.sin(150*torch.pi*x) + 0.8*torch.cos(100*torch.pi*x**2)

def y1_trig_v2(x):
    """ Another high-frequency trigonometric function. """
    return 0.6*torch.sin(200*torch.pi*x) + 0.8*torch.cos(160*torch.pi*x**2)


# --- Main Experiment Logic ---

def generate_landscape_data(model, f_true_func, interval, num_samples, n_grid, plot_range):
    """ we generate the loss landscape for a given model and target function. """
    p_initial = get_weights(model)
    d1 = get_random_weights(p_initial)
    d2 = get_random_weights(p_initial)

    normalize_directions_for_weights(d1, p_initial)
    normalize_directions_for_weights(d2, p_initial)
    
    x_grid = np.linspace(-1, 1, n_grid) * plot_range
    y_grid = np.linspace(-1, 1, n_grid) * plot_range
    X, Y = np.meshgrid(x_grid, y_grid)
    
    x_train = torch.linspace(interval[0], interval[1], num_samples, device=device, dtype=mydtype).view(-1, 1)
    # we call f_true_func directly, as x_train is already on the correct device
    y_true = f_true_func(x_train)
    h = (interval[1] - interval[0]) / num_samples
    loss_grid = np.zeros_like(X)
    
    for i in range(n_grid):
        for j in range(n_grid):
            new_weights = [p + X[i, j] * d + Y[i, j] * v for p, d, v in zip(p_initial, d1, d2)]
            set_weights(model, new_weights)
            
            with torch.no_grad():
                y_nn = model(x_train)
                loss = torch.sum((y_nn - y_true)**2) * h
                loss_grid[i, j] = loss.item()
    
    d1_list = [d.cpu().numpy() for d in d1]
    d2_list = [d.cpu().numpy() for d in d2]

    return X, Y, loss_grid, d1_list, d2_list

def run_experiment():
    """ we run the main experiment loop. """
    base_seed = 1000
    runs_per_config = 10
    output_base_dir = "figures/bigrun"

    # --- Definition of Target Functions ---
    # the functions now only take 'x' as input, since x is already on the correct device
    f_targets = {
        "T01_MainY": lambda x: main_y(x),
        "T02_MainY_plus_Rational": lambda x: main_y(x) + y1_rational(x),
        "T03_MainY_plus_TrigV2": lambda x: main_y(x) + y1_trig_v2(x),
        "T04_MainY_div_x2_plus_TrigV1": lambda x: main_y(x)/(1+x**2) + y1_trig_v1(x),
        "T05_Just_Rational": lambda x: y1_rational(x),
        "T06_Just_TrigV1": lambda x: y1_trig_v1(x),
        "T07_Just_TrigV2": lambda x: y1_trig_v2(x),
        "T08_Rational_plus_TrigV1": lambda x: y1_rational(x) + y1_trig_v1(x),
        "T09_SmoothSin": lambda x: torch.sin(x * 4 * torch.pi) / (1 + x**2),
        "T10_Zero": lambda x: torch.zeros_like(x),
    }

    # --- Definition of Network Configurations ---
    depths = [2, 4, 8, 16]
    widths = [16, 64, 256, 1024]
    activations = ["Sin", "SinT1", "Cos", "ReLU", "GELU", "Tanh"]
    
    # we reverse order to test biggest models first
    depths.reverse()
    widths.reverse()
    
    configs = []
    for depth, width, act in product(depths, widths, activations):
        configs.append({"depth": depth, "width": width, "act_kind": [act] * depth, "name": act})
        
    for depth, width in product(depths, widths):
        if depth > 2:
            configs.append({"depth": depth, "width": width, "act_kind": ["ReLU"]*(depth-2)+["Sin"], "name": "Hybrid_ReLU_Sin"})
            configs.append({"depth": depth, "width": width, "act_kind": ["ReLU"]*(depth-2)+["SinT1"], "name": "Hybrid_ReLU_SinT1"})

    # --- Main Loop ---
    all_experiments = list(product(f_targets.items(), configs, range(runs_per_config)))
    pbar = tqdm(all_experiments, desc="Overall Progress")

    for (f_name, f_func), config, run_idx in pbar:
        conf_name = f"FCNN_depth-{config['depth']}_width-{config['width']}_act-{config['name']}"
        conf_dir = os.path.join(output_base_dir, f"F_target-{f_name}", conf_name)
        os.makedirs(conf_dir, exist_ok=True)
        
        seed = base_seed + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        pbar.set_description(f"F:{f_name} D:{config['depth']} W:{config['width']} A:{config['name']} R:{run_idx}")
        
        model = nets.FCNN(in_out_dim=[1, 1],
                            widths=[config['width']] * config['depth'],
                            device=device, 
                            ResNet=False,
                            act_kind=config['act_kind']
                          ).to(device)

        X, Y, Z, d1_list, d2_list = generate_landscape_data(
            model=model, f_true_func=f_func, interval=[-1, 1],
            num_samples=2000, n_grid=50, plot_range=240.0
        )

        run_base_name = f"run_{run_idx:03d}_seed-{seed}"
        html_path = os.path.join(conf_dir, f"{run_base_name}.html")
        directions_path = os.path.join(conf_dir, f"{run_base_name}_directions")

        landscape_data = (X, Y, Z)
        myplotly.plot(landscape_data, landscape_data, html_path)
        
        save_dict = {'seed': seed}
        for i, arr in enumerate(d1_list):
            save_dict[f'd1_{i}'] = arr
        for i, arr in enumerate(d2_list):
            save_dict[f'd2_{i}'] = arr
        
        np.savez_compressed(directions_path, **save_dict)

    pbar.close()
    print("\n--- all experiments finished ---")

if __name__ == '__main__':
    run_experiment()
