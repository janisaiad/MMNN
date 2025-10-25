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

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--kind', type=int, default=0, metavar='N',help=' ')
parser.add_argument('--gpu', type=int, default=0, metavar='N',help=' ')
parser.add_argument('--epochs', type=int, default=800, metavar='N',help=' ')
parser.add_argument('--train', type=int, default=0, metavar='N',help=' ')
#   0 load; 1 train from random
args = parser.parse_args()
mydtype=torch.float32
# mydtype=torch.float64
# torch.set_default_dtype(mydtype)


kind=args.kind
# kind=1 # FCNN, train all
# kind=2 # MMNN, train all
# kind=3 # MMNN, fix W,b

if kind==1:
    width, rank, depth = 150, 0, 12
else:
    width, rank, depth = 500, 22, 12

if kind==1:
    width, rank, depth = 150, 0, 9
else:
    width, rank, depth = 478, 23, 9

if kind==1:
    width, rank, depth = 120, 0, 9
else:
    width, rank, depth = 388, 18, 9
    
if kind==1:
    width, rank, depth = 120, 0, 12
else:
    width, rank, depth = 390, 18, 12

if kind==1:
    width, rank, depth = 200, 0, 12
else:
    width, rank, depth = 656, 30, 12

if kind==0:
    width, rank, depth  = 168, 0, 12
elif kind==1:
    width, rank, depth = 240, 0, 12
else:
    width, rank, depth = 789, 36, 12
    
ResNet=0  # 1 ResNet, 0 No Res    


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
epochs_same_lr=16
num_epochs = 800
num_epochs = args.epochs
epoch_skip=1

sqrt_num_samples = 600
sqrt_num_samples_test=520
batch_size = 1000



save_idx=1
save_plot=True
save_plot=False

print_param=True
print_param=False

wrd=f"w{width}r{rank}d{depth}kind{kind}"
gpu=args.gpu
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Training on device: {device}")
############################### case 2
##########################  
##########################  
# def cart2pol(x, y):
#     rho = np.sqrt(x**2 + y**2)
#     phi = np.arctan2(y, x)
#     return(rho, phi)

# def fun(x):
#     r, theta = cart2pol(x[:,0], x[:,1])
#     r0 = 0.2 +  0.02*np.cos(8*theta)
#     z0 = 0.2 - 8*(r-r0)
#     m=np.pi**2
#     r1 = 0.66+ 0.08*np.cos(m*theta**2)
#     z1 = 0.5 - 3*(r-r1)
#     def g(z):        
#         z = np.maximum(z, 0)
#         z = np.minimum(z,1)
#         return(z)
#     y=g(z1)-g(z0)
#     # y*=(x[:,1]>=0.5)
#     return(y)
   
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
    return 10/np.sqrt(5)*(y)

class Plot(object):
    def __init__(self,sqrt_n1=250, interval=[-1,1]):
        super().__init__()
        # self.curves=curves
        self.sqrt_n1=sqrt_n1
        self.interval=interval
    
    def samples(self):
        x_n1=np.linspace(*self.interval,self.sqrt_n1)
        y_n1=np.linspace(*self.interval,self.sqrt_n1)
        xv,yv=np.meshgrid(x_n1,y_n1)
        x_plot_in=np.concatenate([np.reshape(xv,[-1,1]),
                                  np.reshape(yv,[-1,1])],axis=1)
        self.xv, self.yv =xv, yv
        return x_plot_in
        
    def myplot(self, myfun):
        x_plot_in = self.samples()
        y_f = myfun(x_plot_in)
        ax=plt.gca()
        fig=plt.gcf()
        ctf=ax.contourf(self.xv,self.yv,
                        y_f.reshape([self.sqrt_n1,self.sqrt_n1]), 
                        100,
                        alpha=0.8, cmap="coolwarm")
        cbar =fig.colorbar(ctf, shrink=0.99, aspect=8)
        cbar.ax.tick_params(labelsize=16)
        plt.xticks(np.linspace(*self.interval,5))
        plt.yticks(np.linspace(*self.interval,5))
        plt.tick_params(axis='both', 
                        which='major', labelsize=18)
        # plt.grid(True, axis='both', color='#AAAAAA', 
        #           linestyle='--', linewidth=1.14)
        plt.tight_layout()      
        # plt.legend()
        return(y_f)

# fig=plt.figure(figsize=(6,4.8))
# # fig = plt.figure(figsize=[8,6])
# ax = plt.gca()    
# Plot().myplot(fun)
# plt.tight_layout()
# plt.savefig(f"LHF2D.png",dpi=100)

# Step 2: Move your neural networks and data to the GPU
if rank>0:
    class F(nn.Module):
        def __init__(self,rank=rank, width=width, depth=depth):
            super().__init__()
            self.rank=rank
            self.width=width
            self.fcs=[]
            ws=[2] + [width, self.rank]*(depth-1) + [width, 1]
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
            ws=[2] + [width]*(depth-1) + [width, 1]
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

x_train_f=Plot(sqrt_num_samples).samples()
z_train_f = fun(x_train_f).reshape([-1,1])
x_train_f = torch.tensor(x_train_f, device=device, dtype=mydtype)
z_train_f = torch.tensor(z_train_f, device=device, dtype=mydtype)

# Step 5: Train f_1 and f_2 separately on their respective datasets using mini-batches
# batch_size = 50  # Adjust the batch size as needed
train_dataset_f = torch.utils.data.TensorDataset(x_train_f, z_train_f)
train_loader_f = torch.utils.data.DataLoader(train_dataset_f, 
                                              batch_size=batch_size, shuffle=True)

time1=time.time()
errors_train=[]
errors_test=[]
errors_test_max=[]
x_in_test=Plot(sqrt_num_samples_test).samples()
y_true = fun(x_in_test)

pytorch_total_params1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params2 = sum(p.numel() for p in model.parameters())

if args.train>0.5:
    for epoch in range(1,1+num_epochs):
        for myidx,param_group in enumerate(optimizer_f.param_groups):
            if myidx==0:
                print(f"\n\nLearning rate: {param_group['lr']}")
        for inputs, targets in train_loader_f:
            # inputs, targets = inputs.to(device), targets.to(device)
            optimizer_f.zero_grad()
            outputs_f = model(inputs)
            loss_f = criterion_f(outputs_f, targets)
            loss_f.backward()
            optimizer_f.step()
            
        e1=loss_f.item()
        print("Training loss in Epoch {}: {:.4e}".format(epoch,e1))
        errors_train.append(e1)
        print(f"Time used: {time.time()-time1:.2f}")
        if epoch % epochs_same_lr == 0:
            scheduler.step()
        
                
        print(f"#param: {pytorch_total_params1} / {pytorch_total_params2}")      
            
        
        if epoch % epoch_skip == 0:
        # Step 6: Compose the functions f_2∘f_1 (move input_data to GPU)
        
            def learned_nn(x):
                input_data = torch.tensor(x, dtype=mydtype).to(device)
                out = model(input_data)
                out = out.cpu().detach().numpy().reshape([-1])
                return out        
            
            y_nn= learned_nn(x_in_test)
            e = y_true - y_nn
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
            errors_txt+=f";   {e_max:.4e} and {e_mse:.4e}"
            print(errors_txt)
            args_dict = vars(args)
            print(args_dict)
            
            if save_plot:
                fig=plt.figure(figsize=(6,4.8))
                # fig = plt.figure(figsize=[8,6])
                ax = plt.gca()    
                Plot().myplot(learned_nn)
                plt.tight_layout()
                # plt.legend(loc="lower center")
                FPN1="./figures2D/"
                if not os.path.exists(FPN1):
                    os.makedirs(FPN1)
                plt.savefig(f"{FPN1}epoch{epoch}_idx{save_idx}_{wrd}_f.png", dpi=100)
                # plt.show()


    # fig=plt.figure(figsize=(6,4.8))
    # # fig = plt.figure(figsize=[8,6])
    # ax = plt.gca()    
    torch.save(model.state_dict(), f'2Dmodel_{wrd}.pth')
    # n=len(errors_train) 
    # t=np.linspace(1,n,n)      
    # plt.plot(t, errors_train, label="training error")
    n=len(errors_test) 
    np.savez(f"2Derrors_{wrd}",
             time=time.time()-time1,
             test=np.array(errors_test), 
             testmax=np.array(errors_test_max), 
              train = np.array(errors_train)
              )
    
    
    # t=np.linspace(1,n,n)   
    # plt.plot(t, errors_train, label="training error")
    # plt.plot(t, errors_test, label="test error")
    # plt.legend()
else:
    model.load_state_dict(
            torch.load(f'2Dmodel_{wrd}.pth',
                       map_location=device))
    
    def learned_nn(x):
        input_data = torch.tensor(x, dtype=mydtype).to(device)
        out = model(input_data)
        out = out.cpu().detach().numpy().reshape([-1])
        return out 
    
    fig=plt.figure(figsize=(6,4.3))
    # fig = plt.figure(figsize=[8,6])
    ax = plt.gca() 
    myfun_e=lambda x: learned_nn(x)-fun(x)
    Plot(sqrt_num_samples_test).myplot(myfun_e)
    plt.tight_layout()
    # pos=ax.get_position()
    # ax.set_position([0.17, pos.y0, 0.525, pos.y1-pos.y0])
    plt.savefig(f"2DvsFCNN_Diff_{wrd}.png",dpi=100)
    
    

