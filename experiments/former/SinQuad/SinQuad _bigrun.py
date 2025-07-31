# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:25:00 2025

@author: Shijun Zhang 
This script runs a large-scale experiment by looping over the logic from SinQuad.py,
generating landscapes by varying two specific weights of the network.
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

# --- Target Functions (based on user's SinQuad.py) ---

def f_target_01(x, k=128): # Sawtooth only
    y=(torch.abs(x)**10)**(1/5)
    y = k*y - 2*torch.floor( (k*y+1)/2 )
    y = abs(y)**2
    return y

def f_target_02(x, k=128): # Sawtooth + Rational
    y=(torch.abs(x)**10)**(1/5)
    y = k*y - 2*torch.floor( (k*y+1)/2 )
    y = abs(y)**2
    y1 = (8*x**4 + 1) / (1 + 10*x**2)
    return y + y1
    
def f_target_03(x, k=128): # Sawtooth + Trig v1
    y=(torch.abs(x)**10)**(1/5)
    y = k*y - 2*torch.floor( (k*y+1)/2 )
    y = abs(y)**2
    y1 = 0.6*torch.sin(150*torch.pi*x)+0.8*torch.cos(100*torch.pi*x**2)
    return y + y1
    
def f_target_04(x, k=128): # Sawtooth + Trig v2
    y=(torch.abs(x)**10)**(1/5)
    y = k*y - 2*torch.floor( (k*y+1)/2 )
    y = abs(y)**2
    y1 = 0.6*torch.sin(200*torch.pi*x)+0.8*torch.cos(160*torch.pi*x**2)
    return y + y1

def f_target_05(x): # Just Rational
    return (8*x**4 + 1) / (1 + 10*x**2)

def f_target_06(x): # Just Trig v1
    return 0.6*torch.sin(150*torch.pi*x)+0.8*torch.cos(100*torch.pi*x**2)

def f_target_07(x): # Just Trig v2
    return 0.6*torch.sin(200*torch.pi*x)+0.8*torch.cos(160*torch.pi*x**2)
    
def f_target_08(x): # Damped Sawtooth
    y=(torch.abs(x)**10)**(1/5)
    y = 128*y - 2*torch.floor( (128*y+1)/2 )
    y = abs(y)**2
    return y / (1 + x**2)
    
def f_target_09(x): # High Freq Smooth Sin
    return torch.sin(x * 400 * torch.pi) / (1 + x**2)

def f_target_10(x): # Zero
    return torch.zeros_like(x)

# --- Main Experiment Function ---

def run_experiment():
    """ we run the main experiment loop. """
    # --- Configuration ---
    base_seed = 1000
    runs_per_config = 10
    output_base_dir = "figures/bigrun_v2"

    f_targets = {
        "T01_Sawtooth": f_target_01, "T02_Sawtooth_p_Rational": f_target_02,
        "T03_Sawtooth_p_TrigV1": f_target_03, "T04_Sawtooth_p_TrigV2": f_target_04,
        "T05_Rational": f_target_05, "T06_TrigV1": f_target_06, "T07_TrigV2": f_target_07,
        "T08_DampedSawtooth": f_target_08, "T09_HighFreqSin": f_target_09, "T10_Zero": f_target_10
    }

    depths = [2, 4, 8, 16]
    widths = [16, 64, 256, 1024]
    activations = ["Sin", "SinT1", "Cos", "ReLU", "GELU", "Tanh"]
    
    depths.reverse()
    widths.reverse()
    
    configs = []
    for depth, width, act in product(depths, widths, activations):
        configs.append({"depth": depth, "width": width, "act_kind": [act] * depth, "name": act})
    for depth, width in product(depths, widths):
        if depth > 2:
            configs.append({"depth": depth, "width": width, "act_kind": ["ReLU"]*(depth-2)+["Sin"], "name": "Hybrid_ReLU_Sin"})
            configs.append({"depth": depth, "width": width, "act_kind": ["ReLU"]*(depth-2)+["SinT1"], "name": "Hybrid_ReLU_SinT1"})

    all_experiments = list(product(f_targets.items(), configs, range(runs_per_config)))
    pbar = tqdm(all_experiments, desc="Overall Progress")

    for (f_name, f_func), config, run_idx in pbar:
        # --- Setup for the current run ---
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

        # --- Core Logic from SinQuad.py ---
        n_grid = 50
        plot_range = 240.0
        num_samples = 2000
        interval = np.array([-1, 1]) * np.pi
        
        # we will vary weights of the first hidden layer, which is fcs[1] for depth > 1
        # this is consistent with W_idx=1 in the original file and robust for all tested depths
        fc_to_vary = model.fcs[1]

        x_grid = np.linspace(-1, 1, n_grid) * plot_range
        y_grid = np.linspace(-1, 1, n_grid) * plot_range
        X, Y = np.meshgrid(x_grid, y_grid)
        
        x_train = torch.linspace(interval[0], interval[1], num_samples, device=device).view(-1, 1)
        y_true = f_func(x_train.to(device)).to(device)
        h = (interval[1] - interval[0]) / num_samples
        loss_grid = np.zeros_like(X)
        
        # we store the original values of the two weights we are going to modify
        w_orig_1 = fc_to_vary.weight.data[0, 0].clone()
        w_orig_2 = fc_to_vary.weight.data[-1, -1].clone()

        for i in range(n_grid):
            for j in range(n_grid):
                # we set the weights to the grid values
                fc_to_vary.weight.data[0, 0] = X[i, j]
                fc_to_vary.weight.data[-1, -1] = Y[i, j]
                
                with torch.no_grad():
                    y_nn = model(x_train)
                    loss = torch.sum((y_nn - y_true)**2) * h
                    loss_grid[i, j] = loss.item()

        # we restore the original weights before the next run
        fc_to_vary.weight.data[0, 0] = w_orig_1
        fc_to_vary.weight.data[-1, -1] = w_orig_2

        # --- Saving results ---
        run_base_name = f"run_{run_idx:03d}_seed-{seed}"
        html_path = os.path.join(conf_dir, f"{run_base_name}.html")
        
        landscape_data = (X, Y, loss_grid)
        # we call the plot function by passing the same data twice to match its signature
        myplotly.plot(landscape_data, landscape_data, html_path)
        
    pbar.close()
    print("\n--- all experiments finished ---")

if __name__ == '__main__':
    run_experiment()
