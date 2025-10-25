# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:08:28 2023

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

matplotlib.rcParams['text.usetex']=True
# plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
matplotlib.rcParams['text.usetex']=False

# parser = argparse.ArgumentParser(description='PyTorch Example')
# parser.add_argument('--kind', type=int, default=1, metavar='N',help=' ')
# args = parser.parse_args()
mydtype=torch.float32
# mydtype=torch.float64
# torch.set_default_dtype(mydtype)

train=1
train=0

kind=0
# kind=1 
# kind=2
# kind=3

if kind==1:
    width, rank, depth = 100, 0, 6   
    width, rank, depth = 150, 0, 6   
elif kind==2 or kind==3:
    width, rank, depth = 321, 15, 6
    width, rank, depth = 546, 20, 6
    
if kind==0:
    width, rank, depth  = 83, 0, 6
elif kind==1:
    width, rank, depth = 120, 0, 6
else:
    width, rank, depth = 388, 18, 6
    
    
ResNet=0 # 1 ResNet, 0 No Res



Opt="SGD"
Opt="Adam"
momentum=0.90
betas=(0.9, 0.999)
if Opt=="SGD":
    lr1_init=0.001
else:
    lr1_init=0.001
lr2_init=lr1_init
lr_gamma=0.9
epochs_same_lr=400
num_epochs = 20000

num_samples = 1000
num_samples_test=1234
batch_size = 100



wrd=f"w{width}r{rank}d{depth}kind{kind}"
save_idx=0
save_plot=True
save_plot=False
print_param=True
# print_param=False
gpu=0
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Training on device: {device}")

##############################

def fun(x):
    # y = np.sin(17*np.pi*x)+np.cos(7*np.pi*x)
    # # y = np.sin(21*np.pi*x) + np.cos(9*np.pi*x)
    y=np.cos(20*np.pi*np.abs(x)**1.4) + 0.5*np.cos(
                    12*np.pi*np.abs(x)**1.6)
    # y = np.sin(
    #     40*np.pi*x)*(np.abs(x+0.0)<0.025)
    return(y)

# fig=plt.figure(figsize=(6,4))
# x=np.linspace(-1,1,1000)
# y=fun(x)
# # plt.plot(x, y, label='learned nn')
# plt.plot(x, y, label='true function')
# plt.grid(True, axis='both', color='#AAAAAA', 
#           linestyle='--', linewidth=1.4)

# Step 1: Check for the availability of CUDA (GPU)


# Step 2: Move your neural networks and data to the GPU
if rank>0:
    class F(nn.Module):
        def __init__(self,rank=rank, width=width, depth=depth):
            super().__init__()
            self.rank=rank
            self.width=width
            self.fcs=[]
            ws=[1] + [width, self.rank]*(depth-1) + [width, 1]
            self.ws=ws
            self.depth=depth
            for j in range(len(ws)-1):
                # print(ws[j], ws[j+1])
                fc = nn.Linear(ws[j], ws[j+1], device=device) 
                # if j==0:
                #     if i<3:
                #         init.normal_(fc.bias, 
                #         mean=0.5, std=0.01) # std=0.01 vs 0.1
                setattr(self, f"fc{j}", fc)
                self.fcs.append(fc)

            # for idx,_ in self.named_parameters():
            #     print(idx)
            #     print( getattr(_, "requires_grad") )
                
            # 
            # for j in range(len(ws)-1):
            #     if j % 2 == 0:
            #     # if len(ws)-j-1>2:
            #         p = getattr(self,f"fc{j}")
            #         p.weight.requires_grad = False
            #         p.bias.requires_grad = False
            #     # if not len(ws)-j-1>2:
            #     #     p = getattr(model,f"fc{j}")
            #     #     p.weight.requires_grad = True
            #     #     p.bias.requires_grad = True
                        
            # # for idx,_ in self.named_parameters():
            # #     print(idx)
            # #     print( getattr(_, "requires_grad") )
        
        def act(self,x):
            # y=torch.relu(1-torch.abs(100*x))
            # y=torch.exp(-100*x**2)
            y=torch.relu(x)
            return(y)        
    
        def forward(self, x):
            fcs=self.fcs
            for j in range(self.depth):
                if ResNet>0.5:
                    if j>0 and j<self.depth-1:
                        # print(j)
                        x0 = x + 0
                # x = torch.relu(fcs[2*j](x))
                x = self.act(fcs[2*j](x))
                x = fcs[2*j+1](x) 
                if ResNet>0.5:
                    if j>0 and j<self.depth-1:
                        x = x + x0
            return x
else:
    class F(nn.Module):
        def __init__(self,rank=rank, width=width, depth=depth):
            super().__init__()
            self.rank=rank
            self.width=width
            self.fcs=[]
            ws=[1] + [width]*(depth-1) + [width, 1]
            self.ws=ws
            self.depth=depth
            for j in range(len(ws)-1):
                # print(ws[j], ws[j+1])
                fc = nn.Linear(ws[j], ws[j+1], device=device) 
                # if j==0:
                #     if i<3:
                #         init.normal_(fc.bias, 
                #         mean=0.5, std=0.01) # std=0.01 vs 0.1
                setattr(self, f"fc{j}", fc)
                self.fcs.append(fc)

            # for idx,_ in self.named_parameters():
            #     print(idx)
            #     print( getattr(_, "requires_grad") )
                
            # 
            # for j in range(len(ws)-1):
            #     if j % 2 == 0:
            #     # if len(ws)-j-1>2:
            #         p = getattr(self,f"fc{j}")
            #         p.weight.requires_grad = False
            #         p.bias.requires_grad = False
            #     # if not len(ws)-j-1>2:
            #     #     p = getattr(model,f"fc{j}")
            #     #     p.weight.requires_grad = True
            #     #     p.bias.requires_grad = True
                        
            # # for idx,_ in self.named_parameters():
            # #     print(idx)
            # #     print( getattr(_, "requires_grad") )
        
        def act(self,x):
            # y=torch.relu(1-torch.abs(100*x))
            # y=torch.exp(-100*x**2)
            y=torch.relu(x)
            return(y)        
    
        def forward(self, x):
            fcs=self.fcs
            for j in range(self.depth):
                if ResNet>0.5:
                    if j>0 and j<self.depth-1:
                        # print(j)
                        x0 = x + 0
                # x = torch.relu(fcs[2*j](x))
                x = fcs[j](x)
                x = self.act(x)
                # print(fcs[j])
                # x = fcs[2*j+1](x) 
                if ResNet>0.5:
                    if j>0 and j<self.depth-1:
                        x = x + x0
            x=fcs[-1](x)
            return x

model = F()#.to(device)
if rank>0 and kind==3:
    for j in range(len(model.ws)-1):
        if j % 2 == 0:
        # if len(ws)-j-1>2:
            p = getattr(model,f"fc{j}")
            p.weight.requires_grad = False
            p.bias.requires_grad = False

# x_in_test=Plot(sqrt_num_samples_test).samples()
# x=torch.tensor(x_in_test, dtype=mydtype).to(device)
# print(model(x).shape)
# Define two groups of parameters
group1_params = []
group2_params = []
for name, param in model.named_parameters():
    test_group1=False
    for i in range(model.depth):
        if f"fc{2*i}" in name:
            test_group1=True
    if test_group1:
        if "weight" in name:
            print("g1", name, param.requires_grad, param.data.shape)
        group1_params.append(param)
    else:
        if "weight" in name:
            print("g2", name, param.requires_grad, param.data.shape)
        group2_params.append(param)

param_groups = [
    {'params': group1_params, 'lr': lr1_init},
    {'params': group2_params, 'lr': lr2_init}
]

        
criterion_f = nn.MSELoss()
if Opt=="SGD":
    optimizer_f = optim.SGD(model.parameters(), lr=lr1_init,
                            momentum=momentum)
else:
    optimizer_f = optim.Adam(param_groups,betas=betas)
# optimizer_f = optim.RAdam(model.parameters(), lr=0.003)

scheduler = StepLR(optimizer_f, step_size=1, gamma=lr_gamma)


x_data=np.linspace(-1, 1, num_samples).reshape([-1, 1])
# x_data2=np.linspace(-1, 1, num_samples).reshape([-1, 1])
x_train_f = x_data
y_train_f = fun(x_train_f)
x_train_f = torch.tensor(x_train_f, device=device, dtype=mydtype)
y_train_f = torch.tensor(y_train_f, device=device, dtype=mydtype)

# Step 5: Train f_1 and f_2 separately on their respective datasets using mini-batches
# batch_size = 50  # Adjust the batch size as needed
train_dataset_f = torch.utils.data.TensorDataset(x_train_f, y_train_f)
train_loader_f = torch.utils.data.DataLoader(train_dataset_f, 
                                              batch_size=batch_size, shuffle=True)


time1=time.time()
errors_train=[]
errors_test=[]
errors_test_max=[]
if train>0.5:
    for epoch in range(1,1+num_epochs):
        for inputs, targets in train_loader_f:
            # inputs, targets = inputs.to(device), targets.to(device)
            optimizer_f.zero_grad()
            outputs_f = model(inputs)
            loss_f = criterion_f(outputs_f, targets)
            loss_f.backward()
            optimizer_f.step()
        e1=loss_f.item()
        print("Training loss in Epoch {}: {:.8f}".format(epoch,e1))
        errors_train.append(e1)
        print(f"Time used: {time.time()-time1:.2f}\n")
        if epoch % epochs_same_lr == 0:
            scheduler.step()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)        
        print("num of param:", pytorch_total_params)    
        
        if epoch % 1 == 0:
        # Step 6: Compose the functions f_2∘f_1 (move input_data to GPU)
        
            def learned_nn(x):
                x=x.reshape([-1,1])
                input_data = torch.tensor(x, dtype=mydtype).to(device)
                out = model(input_data)
                out = out.cpu().detach().numpy().reshape([-1])
                return out     
            
            x_in = np.linspace(-1, 1, num_samples_test)
            y_nn = learned_nn(x_in)
            y_f = fun(x_in)
            # Calculate errors
            e = y_f - y_nn
            e_max = np.linalg.norm(e, ord=np.inf)
            e_mse = np.mean(e**2)
            errors_test.append(e_mse)
            errors_test_max.append(e_max)
            
            errors_txt=" "
            errors_txt+=f"Epoch: {epoch}; " #"  {Opt}; "
            errors_txt+=f"width={width}; rank={rank}; depth={depth}"
            # errors_txt+=f"For each layer: keep the parameters within $\sigma$,\n"
            # errors_txt+=" and only train those that lie outside of it."
            # "     Errors max and mse: "
            errors_txt+=f";\n {e_max:.2e} and {e_mse:.2e}"
            print(errors_txt)
        if epoch % 100 == 0 and save_plot:
            # Plot the results
            fig=plt.figure(figsize=(6,4))
            plt.plot(x_in, y_f, label='true function')
            # for i in range(model.num_channels):
            #     plt.plot(x_in, y_nns[i], label=f'learned nn{i+1}')
            plt.plot(x_in, y_nn, label='learned network')
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
            plt.legend(loc="upper center" , fontsize=13,  ncol=2,
                )
            ax=plt.gca()
            # Get current y-axis limits
            current_ylim = ax.get_ylim()
            # print("Current Y-Axis Limits:", current_ylim)
            
            # Adjust y-axis limits (example: widen the range by 10)
            e=current_ylim[1]-current_ylim[0]
            new_ylim = (current_ylim[0], current_ylim[0]+1.145*e)
            
            # Set new y-axis limits
            ax.set_ylim(new_ylim)
            # plt.legend(loc="lower center")
            FPN1="./figures/"
            if not os.path.exists(FPN1):
                os.makedirs(FPN1)
            plt.savefig(f"{FPN1}epoch{epoch}_idx{save_idx}_{wrd}.pdf",dpi=100)
            plt.show()
    
    fig=plt.figure(figsize=(6,4.8))
    # fig = plt.figure(figsize=[8,6])
    ax = plt.gca()    
    torch.save(model.state_dict(), f'1Dmodel_{wrd}.pth')
    
    n=len(errors_test) 
    m=len(errors_train)
    k=round(m/n)
    np.savez(f"1Derrors_{wrd}", 
             test=np.array(errors_test), 
             testmax=np.array(errors_test_max), 
             train = np.array(errors_train), 
             time=time.time()-time1
             )
    t=np.linspace(1,n,n)   
    plt.plot(t, np.log10(errors_train[::k]), label="training error")
    plt.plot(t, np.log10(errors_test), label="test error")
    plt.legend()
else:
    model.load_state_dict(
            torch.load(f'1Dmodel_{wrd}.pth',
                       map_location=device))
    
    def learned_nn(x):
        x=x.reshape([-1,1])
        input_data = torch.tensor(x, dtype=mydtype).to(device)
        out = model(input_data)
        out = out.cpu().detach().numpy().reshape([-1])
        return out   
    
    fig=plt.figure(figsize=(6,4.3))
    # fig = plt.figure(figsize=[8,6])
    ax = plt.gca() 
    x=np.linspace(-1,1,1234)
    # plt.plot(x,fun(x),lw=5)
    plt.plot(x,learned_nn(x)-fun(x),lw=2)
    plt.grid(True,axis='both',color='#b0b0b0', 
             linestyle='--', linewidth=1.2,)
    # plt.legend(fontsize=42)

    # plt.yticks(np.arange(-7,2))
    # plt.ylim([-7,1])
    # plt.legend(
    #            fontsize=legend_fs, 
    #            loc="upper right",
    #            borderpad=0.46,
    #            ncols=1,
    #            handletextpad=0.73)
    ax.tick_params(labelsize=18)
    ax=plt.gca()
    # Get current y-axis limits
    current_ylim = ax.get_ylim()
    # print("Current Y-Axis Limits:", current_ylim)            
    # Adjust y-axis limits (example: widen the range by 10)
    e=current_ylim[1]-current_ylim[0]
    # ra=1.15 if initParam==3 else 1
    ra1=0.0
    ra2=0.0
    new_ylim = (current_ylim[0]-ra1*e, current_ylim[1]+ra2*e)
    ax.set_xticks(np.linspace(-1,1,5))
    # Set new y-axis limits
    ax.set_ylim(new_ylim)
    plt.tight_layout()
    # ax.set_position([0.1,0.1,0.73,0.8])
    pos=ax.get_position()
    ax.set_position([0.168, pos.y0, 0.65, pos.y1-pos.y0+0.00361])
    plt.savefig(f"1DvsFCNN_Diff_{wrd}.pdf",dpi=100)


