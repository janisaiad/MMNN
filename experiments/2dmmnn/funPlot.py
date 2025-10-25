# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:22:00 2024

@author: Shijun Zhang 
"""

from __future__ import print_function
import argparse,os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import torch.nn.init as init
import matplotlib


##########################  
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

cart2pol2=lambda x,y: cart2pol(2*x-1,2*y-1)
dim=2
def fun(x):
    r, theta = cart2pol(x[:,0], x[:,1])
    r0 = 0.2 +  0.02*np.cos(8*theta)
    z0 = 0.2 - 8*(r-r0)
    m=np.pi**2
    r1 = 0.66+ 0.08*np.cos(m*theta**2)
    z1 = 0.5 - 3*(r-r1)
    def g(z):        
        z = np.maximum(z, 0)
        z = np.minimum(z,1)
        return(z)
    y=g(z1)-g(z0)
    # y*=(x[:,1]>=0.5)
    return(y)

def fun(x):
    # x has a size batch_size * dim
    a=[  [0.3,0.2], 
                [0.2,0.3]
        ]
        
    b=[2*np.pi,4*np.pi]
    
    c=[ [2*np.pi,4*np.pi],
        [8*np.pi,4*np.pi]
    ]
    
    d=[ [4*np.pi,6*np.pi],
        [8*np.pi,6*np.pi]
    ]
    
    dim=2
    r=2
    a=np.array(a)
    b=np.array(b)*r
    c=np.array(c)*r
    d=np.array(d)*r
    y=np.zeros_like(x[:,0])
    for i in range(dim):
        for j in range(dim):
            y=y+a[i,j]*np.sin(b[i]*x[:,i]+c[i,j]*x[:,i]*x[:,j])*np.cos(b[j]*x[:,j]+d[i,j]*x[:,i]**2)
    return(y)

interval=[-1,1]
# sqrt_n1=n_plot
sqrt_n1=320
x_n1=np.linspace(*interval,sqrt_n1)
y_n1=np.linspace(*interval,sqrt_n1)
xv,yv=np.meshgrid(x_n1,y_n1)
x_plot_in=np.concatenate([np.reshape(xv,[-1,1]),np.reshape(yv,[-1,1])],axis=1)
y_f = fun(x_plot_in)
fig=plt.figure(figsize=(6,4.8))
# fig = plt.figure(figsize=[8,6])
ax = plt.gca()    
ctf=ax.contourf(xv,yv,y_f.reshape([sqrt_n1,sqrt_n1]), 100,
                alpha=0.8, cmap="coolwarm")
fig.colorbar(ctf, shrink=0.99, aspect=8)
plt.xticks(np.linspace(*interval,5))
plt.yticks(np.linspace(*interval,5))
plt.tick_params(axis='both', 
                which='major', labelsize=12)
# plt.grid(True, axis='both', color='#AAAAAA', 
#           linestyle='--', linewidth=1.14)
plt.tight_layout()
plt.savefig(f"vsFCNNf2D.png",dpi=100)




def fun(x):
    # y1 = np.sin(20*np.pi*x)*(np.abs(x-0.70)<0.05)
    # y2 = np.sin(12*np.pi*x)*(x<0.3)
    # y = y1+y2
    # y = np.sin(20*np.pi*x)*(np.abs(x-0.40)<0.05)
    # y = np.sin(10*np.pi*x)
    # y = np.sin(21*np.pi*x)#*(x>=0)
    # # y = np.sin(30*np.pi*np.abs(x)**1.5)
    # s = (x+1.5)*0.6
    # s = -x**2+1
    # # s *=0.1
    # # s = x**2
    # y *= s
    # y = np.sin(18*np.pi*np.abs(x)**1.2)
    # y *= (np.abs(x-0.5)<0.2)+(np.abs(x+0.5)<0.2)
    # y = np.sin(21*np.pi*x)#*(x>=0)
    # y = np.sin(10*np.pi*x)*(np.abs(x-0.40)<0.1)+ 0
    # y = np.sin(
    #     50*np.pi*x)#*(np.abs(x+0.0)<0.02)
    y = np.sin(17*np.pi*x)+np.cos(7*np.pi*x)
    # y = np.sin(21*np.pi*x) + np.cos(9*np.pi*x)
    y=np.cos(20*np.pi*np.abs(x)**1.4) + 0.5*np.cos(
                    12*np.pi*np.abs(x)**1.6)
    # y = np.sin(
    #     40*np.pi*x)*(np.abs(x+0.0)<0.025)
    return(y)


x_in=np.linspace(-1,1,1000)
y_f=fun(x_in)
fig=plt.figure(figsize=(6,4.6))
plt.plot(x_in, y_f, lw=2, label='true function')
# for i in range(f_net.num_channels):
#     plt.plot(x_in, y_nns[i], label=f'learned nn{i+1}')
# plt.plot(x_in, y_nn, label='learned network')
# plt.plot(x_in, first, label='first')
# plt.grid(axis="both",linestyle="--",lw=1.0)  
plt.xticks(np.linspace(-1,1,5))
plt.tick_params(axis='both', 
                which='major', labelsize=12)
plt.grid(True, axis='both', color='#AAAAAA', 
          linestyle='--', linewidth=1.4)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('true function and learned nn')
# plt.title(errors_txt,fontsize=17)
plt.tight_layout()
# plt.legend(loc="upper center" , fontsize=13,  ncol=2,
#     )
ax=plt.gca()
# Get current y-axis limits
current_ylim = ax.get_ylim()
# print("Current Y-Axis Limits:", current_ylim)

# Adjust y-axis limits (example: widen the range by 10)
e=current_ylim[1]-current_ylim[0]
new_ylim = (current_ylim[0], current_ylim[0]+1.0*e)

# Set new y-axis limits
ax.set_ylim(new_ylim)
# plt.legend(loc="lower center")
FPN1="./figures/"
if not os.path.exists(FPN1):
    os.makedirs(FPN1)
plt.savefig(f"vsFCNNf1D.pdf",dpi=100)


