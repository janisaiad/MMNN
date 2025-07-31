# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import time
import nets,myplotly
import torch
from itertools import product
from tqdm import tqdm
import os

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
torch.set_default_dtype(torch.float32)
mydtype = torch.get_default_dtype()

# Configuration identique à SinQuad.py
acts= [
    "Sin",
    "SinT1",
    "Cos",
    "ReLU", 
    "GELU",
    "Tanh",
]

n=50  # grid size for plot
s=60*4  # range for plot  
num_samples = 2000
interval=np.array([-1,1])*np.pi

def f_true(x, k=128):
    y=(torch.abs(x)**10)**(1/5)
    y = k*y - 2*torch.floor( (k*y+1)/2 )
    y = abs(y)**2
    y1 = (8*x**4 + 1) / (1 + 10*x**2)
    return y + y1

def get_data(net_size, act_idx, W_idx):   
    fc_idx=W_idx
    widths = [ net_size[0] ]*net_size[2]
    ResNet = True if net_size[2]>8.5 else False
    act_kind=[ acts[act_idx] ]*net_size[2]
    
    model = nets.FCNN(in_out_dim=[1, 1],
                        widths = widths,
                        device = device, 
                        ResNet = ResNet,
                        act_kind= act_kind
                      )
    
    x=np.linspace(-1,1,n)*s
    y=np.linspace(-1,1,n)*s
    X,Y = np.meshgrid(x, y)
    
    x_train = np.linspace(*interval, num_samples).reshape([-1, 1])
    x_train = torch.tensor(x_train, device=device, dtype=mydtype)   
    y_true = f_true(x_train)
    h = ( interval[1] - interval[0] ) / num_samples
    loss=np.zeros_like(X)
    
    for i in range(n):
        for j in range(n):
            model.fcs[fc_idx].weight.data[0,0] = X[i,j]
            model.fcs[fc_idx].weight.data[-1,-1] = Y[i,j]
        
            y_nn = model(x_train)
            y= (y_nn-y_true)**2
            loss[i,j] = h * (0.5 * y[0] + 0.5 * y[-1] + torch.sum(y[1:-1]))
    
    return X, Y, loss

# Boucles sur les configs
depths = [16, 8, 4, 2]
widths = [1024, 256, 64, 16] 
runs_per_config = 10

configs = []
for depth in depths:
    for width in widths:
        for act_idx in range(len(acts)):
            for run in range(runs_per_config):
                configs.append((depth, width, act_idx, run))

pbar = tqdm(configs)
for depth, width, act_idx, run in pbar:
    pbar.set_description(f"D:{depth} W:{width} A:{acts[act_idx]} R:{run}")
    
    torch.manual_seed(1000 + run)
    net_size = [width, 0, depth]
    W_idx = 1
    
    data1 = get_data(net_size, act_idx, W_idx)
    data2 = get_data([width//2, 0, 2], act_idx, W_idx)  # deuxième config pour plot
    
    folder = f"figures/bigrun_v4/D{depth}_W{width}_{acts[act_idx]}"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/run_{run:03d}.html"
    
    myplotly.plot(data1, data2, filename)

print("Done!")
