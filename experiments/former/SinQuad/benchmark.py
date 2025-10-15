import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time,os
import mmnn

def compute_ntk_gram(model, x, device):
    """we compute ntk using vectorized jacobian computation"""
    n = x.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    
    if len(params) == 0:
        return torch.zeros((n, n)), torch.zeros(n)
    
    jacobians = []
    for i in range(n):
        x_i = x[i:i+1].requires_grad_(True)
        y_i = model(x_i)
        
        if not y_i.requires_grad:
            jacobians.append(torch.zeros(sum(p.numel() for p in params), device=device))
            continue
        
        grads = torch.autograd.grad(y_i.sum(), params, create_graph=False, allow_unused=True)
        jac = torch.cat([g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device) 
                         for g, p in zip(grads, params)])
        jacobians.append(jac)
    
    J = torch.stack(jacobians)
    ntk = J @ J.T
    
    ntk_cpu = ntk.cpu()
    eigenvalues = torch.linalg.eigvalsh(ntk_cpu)
    
    return ntk_cpu, eigenvalues

# torch.set_default_dtype(torch.float64)
mydtype = torch.get_default_dtype()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
##############################
def func(x):
    y = np.cos(36*np.pi* x**2) - 0.8*np.cos(12*np.pi* x**2)
    return y


num_epochs = 2000
batch_size = 100
num_training_samples = 1000 # uniform grid samples
num_test_samples = 1234 # random samples
  
# learning rate in epoch k is 
# lr_init*lr_gamma**floor(k/lr_step_size)
lr_init=0.001
lr_gamma=0.9
lr_step_size= 400


# Set this to False if running the code on a remote server.
# Set this to True if running the code on a local PC 
# to monitor the training process.
show_plot = False

interval=[-1,1]
ranks = [1] + [36]*5 + [1]
widths = [666]*6
model = mmnn.MMNN(ranks = ranks, 
                 widths = widths,
                 device = device,
                 ResNet = False)



# nous vérifions l'initialisation
print("\n=== WEIGHT INITIALIZATION CHECK ===")
for j, layer in enumerate(model.fcs):
    w_norm = layer.weight.norm().item()
    b_norm = layer.bias.norm().item()
    w_mean = layer.weight.mean().item()
    w_std = layer.weight.std().item()
    frozen = "FROZEN" if not layer.weight.requires_grad else "trainable"
    
    print(f"Layer {j} ({frozen}): weight_norm={w_norm:.3f}, weight_mean={w_mean:.6f}, weight_std={w_std:.6f}, bias_norm={b_norm:.3f}")


x_train = np.linspace(*interval, num_training_samples).reshape([-1, 1])
y_train = func(x_train)
x_train = torch.tensor(x_train, device=device, dtype=mydtype)
y_train = torch.tensor(y_train, device=device, dtype=mydtype)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, shuffle=True)


time1=time.time()
errors_train=[]
errors_test=[]
errors_test_max=[]
all_losses=[]  # we store all losses
ntk_eigenvalues_full={}  # we store all ntk eigenvalues (full spectrum)
ntk_eigenvalues={}  # we store ntk eigenvalues min/max



optimizer = optim.AdamW(model.parameters(), lr=lr_init)
scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
criterion = nn.MSELoss()

for epoch in range(1,1+num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    all_losses.append(loss.item())  # we store loss
    scheduler.step()
              
    if epoch % 50 == 0:
        training_error = loss.item()
        print(f"\nEpoch {epoch} / {num_epochs}" + 
              f"  ( {epoch/num_epochs*100:.2f}% )" +
              f"\nTraining error (MSE): { training_error :.2e}" + 
              f"\nTime used: { time.time() - time1 :.2f}s")
        errors_train.append(training_error)
    
        def learned_nn(x): # input and output are numpy.ndarray  
            x=x.reshape([-1, 1]) 
            input_data = torch.tensor(x, dtype=mydtype).to(device)
            y = model(input_data)
            y = y.cpu().detach().numpy().reshape([-1])
            return y     
        
        
        x = np.random.rand(num_test_samples) * 2 - 1
        y_nn = learned_nn(x)
        y_true = func(x)
        
        # Calculate errors
        e = y_nn - y_true
        e_max = np.max(np.abs(e))
        e_mse = np.mean(e**2)
        errors_test.append(e_mse)
        errors_test_max.append(e_max)
        
        print("Test errors (MAX and MSE): " + 
              f"{e_max:.2e} and {e_mse:.2e}")
        
        # we compute NTK every 50 epochs (min/max only for print)
        if epoch % 50 == 0:
            ntk, eigenvalues = compute_ntk_gram(model, x_train, device)
            ntk_eigenvalues[epoch] = eigenvalues
            print(f"NTK eigenvalues: min={eigenvalues[0]:.3e}, max={eigenvalues[-1]:.3e}")
    
    # we store full eigenvalue spectrum every 1000 epochs for detailed analysis
    if epoch % 100 == 0:
        ntk, eigenvalues = compute_ntk_gram(model, x_train, device)
        ntk_eigenvalues_full[epoch] = eigenvalues.cpu().numpy()  # we store as numpy array
        print(f"Stored full NTK spectrum at epoch {epoch}: {len(eigenvalues)} eigenvalues")
    
    if epoch % 50 == 0:
        if epoch % 100 == 0:
            # Plot the results
            x = np.linspace(-1, 1, 1000)
            y_nn = learned_nn(x)
            y_true = func(x)
            fig=plt.figure(figsize=(6,4))
            plt.plot(x, y_true, label='true function')
            plt.plot(x, y_nn, label='learned network')
            plt.xticks(np.linspace(*interval,5))
            plt.tick_params(axis='both', 
                            which='major', labelsize=12)
            plt.grid(True, axis='both', color='#AAAAAA', 
                      linestyle='--', linewidth=1.4)
            plt.title(f'true function and learned network in (Epoch {epoch})')
            plt.tight_layout()
            plt.legend(loc="upper center" , fontsize=13,  ncol=2,
                )
    
            FPN = os.path.join("figuressgd", "SinQuad", f"rank{ranks[-1]}_width{widths[0]}")
            os.makedirs(FPN, exist_ok=True)
            plt.savefig(os.path.join(FPN, f"mmnn_epoch{epoch}_1D.png"), dpi=50)
            if show_plot:
                plt.show()

torch.save(model.state_dict(), 'model_parameters1D.pth')
np.savez("errors1D", 
         test=np.array(errors_test), 
         testmax=np.array(errors_test_max), 
         train = np.array(errors_train),
         all_losses=np.array(all_losses),
         time=time.time()-time1
         )

# we plot loss evolution
fig=plt.figure(figsize=(8,5))
plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss Evolution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./figuressgd/loss_evolution.png', dpi=100)
plt.close()

# we plot errors
fig=plt.figure(figsize=(8,5))
n=len(errors_test) 
m=len(errors_train)
plt.plot(np.linspace(1,m,m)*50, np.log10(errors_train), 
         label="log10 training error", linewidth=2)
plt.plot(np.linspace(1,n,n)*50, np.log10(errors_test), 
         label="log10 test error", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('log10(error)')
plt.title('Error Evolution')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('./figuressgd/error_evolution.png', dpi=100)
plt.close()

# we plot NTK eigenvalues (min/max)
if len(ntk_eigenvalues) > 0:
    epochs_list = sorted(ntk_eigenvalues.keys())
    max_eigenvalues = [ntk_eigenvalues[ep][-1].item() for ep in epochs_list]
    min_eigenvalues = [ntk_eigenvalues[ep][0].item() for ep in epochs_list]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    ax1.plot(epochs_list, max_eigenvalues, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Max Eigenvalue')
    ax1.set_title('NTK Maximum Eigenvalue Evolution')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_list, min_eigenvalues, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Min Eigenvalue')
    ax2.set_title('NTK Minimum Eigenvalue Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figuressgd/ntk_eigenvalues_minmax.png', dpi=100)
    plt.close()

# we plot full NTK eigenvalue spectrum (every 1000 epochs)
if len(ntk_eigenvalues_full) > 0:
    epochs_full = sorted(ntk_eigenvalues_full.keys())
    
    # we create a plot with all eigenvalues at each epoch
    fig = plt.figure(figsize=(10, 6))
    
    for epoch in epochs_full:
        eigs = ntk_eigenvalues_full[epoch]
        indices = np.arange(len(eigs))
        plt.semilogy(indices, eigs, '-o', markersize=3, label=f'Epoch {epoch}', alpha=0.7)
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('Full NTK Eigenvalue Spectrum Evolution')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figuressgd/ntk_full_spectrum.png', dpi=100)
    plt.close()
    
    # we plot first 5 and last 5 eigenvalues over time
    fig, axes = plt.subplots(10, 1, figsize=(10, 16))
    fig.suptitle('NTK Eigenvalues Evolution (First 5 and Last 5)', fontsize=14, y=0.995)
    
    # we plot first 5 eigenvalues (smallest)
    for i in range(5):
        ax = axes[i]
        eigenvalues_over_time = [ntk_eigenvalues_full[ep][i] for ep in epochs_full]
        ax.plot(epochs_full, eigenvalues_over_time, 'o-', linewidth=2, markersize=6)
        ax.set_ylabel(f'λ_{i+1}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        if i < 4:
            ax.set_xticklabels([])
    
    # we plot last 5 eigenvalues (largest)
    n_eigs = len(ntk_eigenvalues_full[epochs_full[0]])
    for i in range(5):
        ax = axes[5 + i]
        idx = n_eigs - 5 + i
        eigenvalues_over_time = [ntk_eigenvalues_full[ep][idx] for ep in epochs_full]
        ax.plot(epochs_full, eigenvalues_over_time, 'o-', linewidth=2, markersize=6, color='C1')
        ax.set_ylabel(f'λ_{idx+1}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        if i < 4:
            ax.set_xticklabels([])
    
    axes[-1].set_xlabel('Epoch', fontsize=12)
    plt.tight_layout()
    plt.savefig('./figuressgd/ntk_first_last_eigenvalues.png', dpi=100)
    plt.close()
    print("NTK first/last 5 eigenvalues plot saved")

# we plot final prediction/fit
x_plot = np.linspace(-1, 1, 1000)
x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), dtype=mydtype).to(device)
with torch.no_grad():
    y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
y_plot_true = func(x_plot_tensor.cpu().numpy())

fig=plt.figure(figsize=(8,5))
plt.plot(x_plot, y_plot_true, 'b-', label='True function', linewidth=2)
plt.plot(x_plot, y_plot_nn, 'r--', label='Learned network', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Prediction vs True Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('./figuressgd/final_prediction.png', dpi=100)
plt.close()

print("\nAll plots saved to ./figuressgd/")
print(f"Total training time: {time.time()-time1:.2f}s")