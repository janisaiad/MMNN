# -*- coding: utf-8 -*-
# %% [markdown]
"""
Created on Tue Feb 18 20:19:55 2025

@author: Shijun Zhang 
"""

# %%
from __future__ import print_function
import argparse
import numpy as np
import scipy 
import mpmath as mp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import nets,myplotly # okay this works very well for small projects also, don't need to put htis on uv it's cool
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random

# matplotlib.rcParams['text.usetex']=True
# plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
# matplotlib.rcParams['text.usetex']=False
seed = random.randint(0, 1000000)
torch.manual_seed(seed)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)
mydtype = torch.get_default_dtype()

# %%


net_size, W_idx = [128, 0, 4], 1

# net_size, W_idx =[128, 32, 2], 1

# net_size, W_idx =[128, 32, 2], 2


# act_idx=1 # 1 for sin
act_idx=2 # 2 for SinTU_0


n=100 # grid size for plot


# %%
acts= [
    "Sin",
    "SinT1",
    "Cos",
    # "CosShift",
    "ReLU",
    # "ELU",
    "GELU",
    # "Sigmoid",
    "Tanh",
    ]



nn_type= "MMNN" if net_size[1]>0 else "FCNN"

PN_save="d" #f"Landscape{nn_type}{W_idx}Act{act_idx}"
s=60*4 # range for plot

num_samples = 10000
interval=np.array([-1,1])*np.pi # integral range


# %%
# def f_true(x):
    # return 1/(1+100*x**2)
list_of_func = [
    # Base functions for y
    lambda x, k=128: (torch.abs(x)**10)**(1/5) * k - 2*torch.floor((k*(torch.abs(x)**10)**(1/5)+1)/2),  # Original y
    lambda x: ((torch.abs(x)**10)**(1/5)) / (1 + x**2),  # y with division
    lambda x: ((torch.abs(x)**10)**(1/5))*(6*x**8 + 1) / (1 + 8*x**6),  # y with multiplication 1
    lambda x: ((torch.abs(x)**10)**(1/5))*(8*x**8 + 1) / (1 + 10*x**4),  # y with multiplication 2
    
    # Functions for y1
    lambda x: 0.6*torch.sin(150*np.pi*x) + 0.8*torch.cos(100*np.pi*x**2),  # Original y1
    lambda x: (8*x**4 + 1) / (1 + 10*x**2),  # Alternative y1 polynomial
    lambda x: 0.6*torch.sin(200*np.pi*x) + 0.8*torch.cos(100*np.pi*x**2),  # y1 with different frequencies
    lambda x: 0.6*torch.sin(200*np.pi*x) + 0.8*torch.cos(160*torch.pi*x**2)  # y1 with torch.pi
]

index_y=0
index_y1=4
combine=True
def f_true(x, k=128, index_y=index_y, index_y1=index_y1, combine=combine):
    y = list_of_func[index_y](x, k) if index_y == 0 else list_of_func[index_y](x)
    if combine:
        y1 = list_of_func[index_y1](x)
        return y + y1
    return y


# %%
def get_data(net_size):   
    fc_idx=W_idx #[1] if net_size[1]>0 else W_idx[0]
    # acts=nets.ActFun_list
    widths = [ net_size[0] ]*net_size[2]
    ResNet = True if net_size[2]>8.5 else False
    act_kind=[ acts[act_idx-1] ]*net_size[2]
    ranks = [1] + [ net_size[1] ]*(net_size[2]-1) + [1]
    if net_size[1]>0.5:
        ranks = [1] + [ net_size[1] ]*(net_size[2]-1) + [1]
        model = nets.MMNN(ranks = ranks, 
                          widths = widths,
                          device = device,
                          # ResNet = False,
                          ResNet = ResNet,
                          # ResNet = True,
                          act_kind = act_kind
                          )
    else:   
        model = nets.FCNN(in_out_dim=[1, 1],
                            widths = widths,
                            device = device, 
                            ResNet = ResNet,
                            # ResNet = True,
                            act_kind= act_kind
                          )
    
    # num_samples = 3000
    # interval=[-1,1]
    # x_train = np.linspace(*interval, num_samples+1).reshape([-1, 1])
    # x_train = torch.tensor(x_train, device=device, dtype=mydtype)   
    # y=model(x_train)
    
    # print(y)
    # print(model.fcs[0].weight.data[0,0])
    # model.fcs[0].weight.data[0,0]=100
    # print(model.fcs[0].weight.data[0,0])
    # y=model(x_train)
    # print(y)
    
    # int_type="composite_trapezoidal"
    # int_type="quad"
    
    x=np.linspace(-1,1,n)*s
    y=np.linspace(-1,1,n)*s
    
    X,Y = np.meshgrid(x, y)
    
    time1=time.time()
    
    
    x_train = np.linspace(*interval, num_samples).reshape([-1, 1])
    x_train = torch.tensor(x_train, device=device, dtype=mydtype)   
    y_true = f_true(x_train)
    h = ( interval[1] - interval[0] ) / num_samples
    loss=np.zeros_like(X)
    for i in range(n):
        print(f"{i}  /  {n};  {net_size}; W_idx = {W_idx}; {act_kind}")
        for j in range(n):
            model.fcs[fc_idx].weight.data[0,0] = X[i,j]
            model.fcs[fc_idx].weight.data[-1,-1] = Y[i,j]
            # print(model.fcs[0].weight.data[0,0])
        
            y_nn = model(x_train)
            y= (y_nn-y_true)**2
            loss[i,j] = h * (0.5 * y[0] + 0.5 * y[-1] + torch.sum(y[1:-1]))
    
    print(f"time used: {time.time()-time1:.2f}s", )
    return X, Y, loss

# %%
data1=get_data(net_size)
# X,Y,Z=data1
# np.savez( f"{PN_save}.npz",X=X,Y=Y,Z=Z) 
net_size2 =[128, 32, 4] 


data2=get_data(net_size2)
import datetime
import json
with open(f"figures/sinquad/config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}_{act_idx}_{s}_{num_samples}_{W_idx}_{net_size[0]}_{net_size[1]}_{net_size[2]}_{net_size2[0]}_{net_size2[1]}_{net_size2[2]}.json", "w") as f:
    # now date time
    f.write(f"now date time: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    f.write(f"seed: {seed}")
    f.write(f"act_idx: {act_idx}")
    f.write(f"s: {s}")
    f.write(f"num_samples: {num_samples}")
    f.write(f"W_idx: {W_idx}")
    f.write(f"net_size: {net_size}")
    f.write(f"net_size2: {net_size2}")
    f.write(f'index_y: {index_y}')
    f.write(f'index_y1: {index_y1}')
    f.write(f'combine: {combine}')
myplotly.plot(data1, data2, f"figures/sinquad/test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}_{act_idx}_{s}_{num_samples}_{W_idx}_{net_size[0]}_{net_size[1]}_{net_size[2]}_{net_size2[0]}_{net_size2[1]}_{net_size2[2]}.html",seed)
