# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 13:40:42 2022


@author: shijun
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex']=True
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
matplotlib.rcParams['text.usetex']=False

plt.rcParams['figure.dpi'] = 30
# ind=1 # 
# PN0="{}/".format(ind)
PN0=""
# width, rank, depth = 800, 30, 8


# size=1 # 
# # size=2

# initParam=1
# # initParam=2
idx_list=[1,2,3]
idx_list=[3,0,2,1]
averOrNot=True
# averOrNot=False

dim=1
dim=2


# size=1 # 
# # size=2 

if dim==1:
    step=2
else:
    step=1



def get_e(idx):
    kind=idx
    if dim==2:
        if kind==1:
            width, rank, depth = 200, 0, 6
        else:
            width, rank, depth = 653, 30, 6  
            # width, rank, depth = 500, 20, 8
        
        if kind==1:
            width, rank, depth = 150, 0, 6
        else:
            width, rank, depth = 544, 20, 6
            
        if kind==1:
            width, rank, depth = 150, 0, 12
        else:
            width, rank, depth = 500, 22, 12
        
        if kind==1:
            width, rank, depth = 150, 0, 9
        else:
            width, rank, depth = 478, 23, 9
        
        if kind==1:
            width, rank, depth = 200, 0, 12
        else:
            width, rank, depth = 656, 30, 12
            
        if kind==1:
            width, rank, depth = 240, 0, 12
        else:
            width, rank, depth = 789, 36, 12
        
        if kind==0:
            width, rank, depth  = 168, 0, 12
        elif kind==1:
            width, rank, depth = 240, 0, 12
        else:
            width, rank, depth = 789, 36, 12
            
    else:
        if kind==1:
            width, rank, depth = 100, 0, 6   
        else:
            width, rank, depth = 321, 15, 6
        
        if kind==1:
            width, rank, depth = 120, 0, 6
        else:
            width, rank, depth = 388, 18, 6
        
        if kind==0:
            width, rank, depth  = 83, 0, 6
        elif kind==1:
            width, rank, depth = 120, 0, 6
        else:
            width, rank, depth = 388, 18, 6
            
        # if kind==1:
        #     width, rank, depth = 150, 0, 6   
        # elif kind==2 or kind==3:
        #     width, rank, depth = 546, 20, 6
    
    wrd=f"w{width}r{rank}d{depth}kind{idx}"
    # wrd=f"w{width}r{rank}d{depth}"
    # def fcNP(width, depth):
    #     return (1+1)*width + (width+1)*width*(depth-1) + (width+1)
    # if idx==1:
    #     numP = (width+1)*rank *(depth-1) + (width+1)
    #     if initParam>1:
    #         numP+= (1+1)*width + (rank+1)*width *(depth-1) 
    # else:
    #     numP=fcNP(width, depth)
    
    def text_mathrm(s_str):
        s_list=s_str.split(" ")
        for i,si in enumerate(s_list):
            if i==0:
                s_list[i]=r"$\mathrm{"+si+"}$"
            else:
                s_list[i]=r" $\mathrm{"+si+"}$"
        return("".join(s_list))
    

    
    t="1D" if dim==1 else "2D"
    PN=PN0+f"{t}errors_{wrd}.npz"
    # PN=PN0+"{}aver_loss.npz".format(i)
    print(PN)
    with np.load(PN) as a:
        # train_errors=errors["train_errors"]
        # print("train_error: ",np.sum(train_errors[-100:])/100)
        # test_errors=errors["test_errors"]
        # print("test_error: ",np.sum(test_errors[-100:])/100)
        
        
        # time_used=a['time']
        # print('Time used: {:.4f} s'.format(time_used))
        # width=a['width']
        # print('width: {:d}'.format(width))
        y=a['train']
        z= a['test']
        u=a['testmax']
        mm=100
        def sci_to_latex(sci_str):
            base, exponent = sci_str.split('e')
            if int(exponent)>=0:
                out=float(base) * 10**int(exponent)
                out=f"{out:.2f}"
            else:
                out=f"{base} \\times 10^{{{int(exponent)}}}"
            return out
        print(f"{wrd}\n mse: {z[-mm:].mean():.2e}")
        print(f" max: {u[-mm:].mean():.2e}")
        txt= "$"+ sci_to_latex(f"{z[-mm:].mean():.2e}") 
        txt+= "$  &  $"
        txt+= sci_to_latex(f"{u[-mm:].mean():.2e}")+"$"
        print(txt)
        print(f"time used: {a['time']:.2f}")
    return( y, z, width, rank, depth )

# endm=10000
# y=y[2:endm]
# z=z[2:endm]






grid_color='#b0b0b0'
tick_fs=60
legend_fs=49
label_fs=61
matplotlib.rc('ytick', labelsize=tick_fs) 
matplotlib.rc('xtick', labelsize=tick_fs) 
fig=plt.figure(figsize=(28,16))
ax=fig.add_subplot(1,1,1)
# ax.set_xlabel(r'$\mathrm{epoch}$',fontsize=label_fs,labelpad=7)
# myylabel="base 10 logarithm of loss"
# ax.set_ylabel(text_mathrm(myylabel),fontsize=label_fs,labelpad=7)
colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
tns=["training error", "test error"]

# ylim=np.array([-0.01,1])*0.1**3
# ax.set_ylim([-6.5,1.5])
# ax.set_xticks(np.linspace(1,n,6))
# ax.set_xlim([-0.5,n+0.5])
# ax.set_ylabel('loss',fontsize=label_fs,labelpad=12)


# ax.set_title(tns[ind_list[0]],fontsize=48)


mylw=5
myls=['dashed','dotted','solid','dashdot']
ax=plt.gca()


def aver(v):
    n=len(v)
    m=100
    w=np.zeros(n)
    for i in range(n):
        i1=max(0, i-m)
        i2=i+m
        # print(i1,i2)
        v2=v[i1:i2]
        w[i]=v2.mean()
    return(w)


for idx in idx_list:
    y, z, width, rank, depth = get_e(idx)
    y=y[::step]
    z=z[::step]
    n=len(y)
    m=len(z)
    print(n,m)
    k=round(n/m)
    x=np.linspace(1,m,m)*step #*100


            
    # y=np.log10(y)
    # z=np.log10(z)
    # y=aver(y)
    # z=aver(z)

    # mm=3000
    # x=x[:mm]
    # z=z[:mm]
    # y=y[:mm]
    
    if averOrNot:
        y=aver(y)
        z=aver(z)  
    y=np.log10(y)
    z=np.log10(z)
    # ax.plot(x,y[::k],
    #             linewidth=6,
    #             label=r"$\log_{10}($training-error$)$ vs. epoch"
    #             # label="training error"
    #             # label=label2,
    #             # ls=myls[r-1],
    #             # zorder=r+4
    #             )
    
    
    def fcNP(width, depth):
        return (dim+1)*width + (width+1)*width*(depth-1) + (width+1)
    
    if rank>0:
        param0 = (width+1)*rank *(depth-1) + (width+1)
    
        print('Number of param (MMNN-AC):', param0)
        param = param0 + (dim+1)*width + (rank+1)*width *(depth-1) 

        print('Number of param (MMNN-all):', param)
    else:
        param = fcNP(width, depth)
        print('Number of param (FC):', param)


    
    if idx==3:
        txt=f"MMNN (S1, train {param0} of the {param} parameters)"
    elif idx==2:
        txt=f"MMNN (S2, train all {param} parameters)"
    elif idx==1:
        txt=f"FCNN (train all {param} parameters)"
    elif idx==0:
        txt=f"FCNN (train all {param} parameters)"
    # if idx==2:

    #     txt=f"MMNN ({param} parameters;  width {width}, rank {rank}, depth {depth})"
    # else:
    #     txt=f"FCNN ({param} parameters;  width {width}, depth {depth})"
    ax.plot(x,z,
                linewidth=9,
                label=txt,
                color=colors[idx]
    # label=fr"{txt}: $\log_{{10}}($test-error-aver$)$ vs. epoch"
                # label="test error"
                # label=label2,
                # ls=myls[r-1],
                # zorder=r+4
                )


# ax.plot(x,y2,color='blue',linewidth=2.8,label=r"$M_{}$".format(2))
# plt.grid(True,axis='both',color='gray', linestyle='--', linewidth=1.3)
# plt.tight_layout()
# plt.legend(loc="upper center",fontsize=30)
plt.grid(True,axis='both',color=grid_color, linestyle='--', linewidth=mylw/3,)
# plt.legend(fontsize=42)

# if initParam<5:
# plt.yticks(np.arange(-7,2))
# plt.ylim([-7,1])
plt.legend(
           fontsize=legend_fs, 
            loc="upper right",
           # loc="upper center",
           borderpad=0.46,
           ncols=1,
           handletextpad=0.73)
# elif initParam==3:
#     plt.legend(
#                fontsize=legend_fs*0.81, 
#                loc="upper center",
#                borderpad=0.5,
#                ncols=2,
#                handletextpad=0.6)
# plt.savefig(PN0+'{}_loss_epochs.pdf'.format(ind_list[0]))

ax=plt.gca()
# Get current y-axis limits
current_ylim = ax.get_ylim()
# print("Current Y-Axis Limits:", current_ylim)            
# Adjust y-axis limits (example: widen the range by 10)
e=current_ylim[1]-current_ylim[0]
# ra=1.15 if initParam==3 else 1
ra1=0.03
ra2=0.25840
new_ylim = (current_ylim[0]-ra1*e, current_ylim[1]+ra2*e)

# Set new y-axis limits
ax.set_ylim(new_ylim)
if dim==1:
    plt.xticks(np.linspace(0,20000,5))
else:
    plt.xticks(np.linspace(0,800,5))
plt.tight_layout()
# plt.tight_layout(pad=0.8,rect=(-0.0045,-0.012,1.00,1.00))
# plt.tight_layout()

# plt_backend=matplotlib.rcParams["backend"]
# if plt_backend.lower()!="pgf":
#     plt.show()
# wrd=f"w{width}r{rank}d{depth}"
plt.savefig(PN0+f'{dim}Derrors_vsFCNN.pdf')





