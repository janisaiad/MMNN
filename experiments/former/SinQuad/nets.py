import torch, math
import torch.nn as nn
import torch.nn.functional as F

ActFun_list= [
    "ReLU",
    # "ELU",
    "GELU",
    # "Sigmoid",
    "Tanh",
    "Sin",
    "Cos",
    # "CosShift",
    ]

# ActFun_list += [f"SinTr{i}"  for i in range(1,5)]

ActFun_list += [f"SinT{i}"  for i in range(1,4)]

# ActFun_list += [f"PSinT{i}"  for i in range(1,4)]

# print(ActFun_list)
# print(len(ActFun_list))

def f1d(f, x, h=1e-3):
    """
    Computes the first derivative of function f at points x using
    the 4th-order accurate 5-point central difference method.

    Parameters:
    - f: Callable function
    - x: Points where the derivative is evaluated (scalar or numpy array)
    - h: Step size (default is 1e-5)

    Returns:
    - Approximate derivative of f at x
    """
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

def f2d(f, x, h=2e-3):
    """
    Computes the second derivative of function f at points x using
    the 4th-order accurate 5-point central difference method.

    Parameters:
    - f: Callable function
    - x: Points where the second derivative is evaluated (scalar or numpy array)
    - h: Step size (default is 1e-5)

    Returns:
    - Approximate second derivative of f at x
    """
    return (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h**2)

class PSinT(nn.Module):
    def __init__(self, num_features, c = -12, ParamSharing=True, device="cpu"):
        super(PSinT, self).__init__()
        # Create a learnable parameter for each feature dimension
        if ParamSharing:
            a = torch.tensor(c*torch.pi, device=device)
        else:
            a = torch.rand(1, num_features, device=device)
            a = c*a
            a = torch.pi*torch.round(a)
        self.alpha = nn.Parameter(a)  # Shape [1, num_features]
        # self.beta = nn.Parameter(torch.zeros(1, num_features,device=device))
        
    def forward(self, x):
        x = torch.relu(x - self.alpha) + self.alpha
        # y = self.alpha * torch.sin(x) + self.beta
        # y = torch.relu(y)
        # y = self.alpha * torch.relu(x) + self.beta
        return torch.sin(x)

def SinT(x, s = -torch.pi*2):
    x = torch.relu(x-s)+s
    return torch.sin(x)

def myActFunc(x, act_str):
    if act_str=="ReLU":
        x = torch.relu(x)
    elif act_str=="ELU":
        x = F.elu(x)
    elif act_str=="GELU":
        x = F.gelu(x)
    elif act_str=="Sigmoid":
        x = torch.sigmoid(x)
    elif act_str=="Tanh":
        x = torch.tanh(x)
    elif act_str=="Sin":
        x = torch.sin(x)
    elif act_str=="Cos":
        x = torch.cos(x)
    elif act_str=="CosShift":
        x = torch.cos(x-torch.pi/4)
    elif act_str=="SinT1":
        x = SinT(x, 0)
    elif act_str=="SinT2":
        x = SinT(x, -torch.pi*1)
    elif act_str=="SinT3":
        x = SinT(x, -torch.pi*2)
    elif act_str=="SinT4":
        x = SinT(x, -torch.pi*4)
    elif act_str=="SinT5":
        x = SinT(x, -torch.pi*8)
    elif act_str=="SinT6":
        x = SinT(x, -torch.pi*16)
    elif act_str=="SinT7":
        x = SinT(x, -torch.pi*32)
    elif act_str=="SinT8":
        x = SinT(x, -torch.pi*64)
    return x


    
class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cpu", 
                 ResNet = False,
                 fixWb = True,
                 act_kind=["R"]*6):
        super().__init__()
        
        self.ranks = ranks
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        self.act_kind=act_kind
        
        fc_sizes = [ ranks[0] ] 
        for j in range(self.depth):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)
        
        if fixWb:
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
        
        if "PSinT" in self.act_kind[0]:
            actfuns=[]
            for j in range(self.depth):
                if "PSinT1"==self.act_kind[j]:
                    act = PSinT(widths[j], c=0, ParamSharing=True,device=device)
                elif "PSinT2"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-1, ParamSharing=True,device=device)
                elif "PSinT3"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-2, ParamSharing=True,device=device)
                elif "PSinT4"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-4, ParamSharing=True,device=device)
                elif "PSinT5"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-12, ParamSharing=True,device=device)
                elif "PSinT6"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-36, ParamSharing=True,device=device)
                elif "PSinT7"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-108, ParamSharing=True,device=device)
                else:
                    act = PSinT(widths[j], ParamSharing=False,device=device)
                setattr(self, f"act{j}", act)
                actfuns.append(act)
            self.actfuns = actfuns
           
    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            x = self.fcs[2*j](x)
            
            if "PSinT" in self.act_kind[j]:
                x = self.actfuns[j](x)
            else:
                x = myActFunc(x, self.act_kind[j])
                
            x = self.fcs[2*j+1](x) 
            # if j<self.depth-1:
            #     m = round(x.shape[1]*0.2)
            #     x[:,:m] = torch.relu(x[:,:m])
                
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x = x + x_id
                    # n = min(x.shape[1], x_id.shape[1])
                    # x[:,:n] = x[:,:n] + x_id[:,:n]
        return x

class FCNN(nn.Module):
    def __init__(self, 
                 in_out_dim=[1,1],
                 widths = [366]*6,
                 device = "cpu", 
                 ResNet = False,
                 act_kind=1):
        super().__init__()

        
        self.in_out_dim = in_out_dim
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        self.act_kind=act_kind
        
        fc_sizes = [ in_out_dim[0] ] + widths + [ in_out_dim[1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)
        
        if "PSinT" in self.act_kind[0]:
            actfuns=[]
            for j in range(self.depth):
                if "PSinT1"==self.act_kind[j]:
                    act = PSinT(widths[j], c=0, ParamSharing=True,device=device)
                elif "PSinT2"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-1, ParamSharing=True,device=device)
                elif "PSinT3"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-2, ParamSharing=True,device=device)
                elif "PSinT4"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-4, ParamSharing=True,device=device)
                elif "PSinT5"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-12, ParamSharing=True,device=device)
                elif "PSinT6"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-36, ParamSharing=True,device=device)
                elif "PSinT7"==self.act_kind[j]:
                    act = PSinT(widths[j], c=-108, ParamSharing=True,device=device)
                else:
                    act = PSinT(widths[j], ParamSharing=False,device=device)
                setattr(self, f"act{j}", act)
                actfuns.append(act)
            self.actfuns = actfuns
 

    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth:
                    x_id = x + 0
            x = self.fcs[j](x)

            if "PSinT" in self.act_kind[j]:
                x = self.actfuns[j](x)
            else:
                x = myActFunc(x, self.act_kind[j])
                
            if self.ResNet:
                if 0 < j < self.depth:
                    x=x+x_id
                    # n = min(x.shape[1], x_id.shape[1])
                    # print(x.shape[1], x_id.shape[1])
                    # print("ddd", x.shape , x_id.shape )
                    # temp = x_id[:, :n].clone() + x[:, :n].clone()
                    # x = temp[:, :]
                    # # x=x+x_id
                    
        x = self.fcs[-1](x)
        return x
    

