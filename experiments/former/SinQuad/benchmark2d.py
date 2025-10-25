# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import json


# %%


# %%

class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cuda", 
                 ResNet = False,
                 fixWb = True):
        super().__init__()
        """
        A class to configure the neural network model.
    
        Attributes:
            ranks (list[int]): A list where the i-th element represents the output dimension of the i-th layer.
                               For the j-th layer, ranks[j-1] is the input dimension and ranks[j] is the output dimension.
            
            widths (list[int]): A list where each element specifies the width of the corresponding layer.
            
            device (str): The device (CPU/GPU) on which the PyTorch code will be executed.
            
            ResNet (bool): Indicates whether to use ResNet architecture, which includes identity connections between layers.
            
            fixWb (bool): If True, the weights and biases are not updated during training.
        """
        
        self.product = 1
        for j in range(1,len(ranks)):
            self.product *= np.sqrt(widths[j-1] *ranks[j])
        self.ranks = ranks # 
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        
        fc_sizes = [ ranks[0] ] 
        for j in range(self.depth):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs) # list of nn.Linear layers
        
        if fixWb: # if True, the weights and biases are not updated during training
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
 

    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            x = self.fcs[2*j](x)
            x = torch.relu(x)
            x = self.fcs[2*j+1](x) 
            if self.ResNet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x[:,:n] = x[:,:n] + x_id[:,:n]
        return x/self.product

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import json


def compute_ntk_gram(model, x, device):
    """we compute ntk using vectorized jacobian computation

    to load saved ntk matrices:
        data = np.load('ntk_matrices.npz')
        ntk_at_epoch_100 = data['epoch_100']  # loads ntk matrix at epoch 100
        all_epochs = [int(key.split('_')[1]) for key in data.keys()]  # gets all stored epochs
    """
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
    try:
        eigenvalues = torch.linalg.eigvalsh(ntk_cpu)
    except:
        eigenvalues = torch.zeros(ntk_cpu.shape[0])

    return ntk_cpu, eigenvalues

# torch.set_default_dtype(torch.float64)
mydtype = torch.get_default_dtype()
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
##############################

# Generate configs with different depths, hidden ranks and widths
configs = []
for lr_init in [0.001, 0.0001]:
    for batch_size in [1000]:
        for num_layers in [12,15,10,20,8,25]:
            for hidden_width in [512,768,1024,1536,164,128,256,1024,2048,4096,8192]:
                for hidden_rank in [25,30,35,40,45,50]:
                    configs.append({
                        # architecture hyperparameters
                        "num_layers": num_layers,
                        "hidden_width": hidden_width,
                        "hidden_rank": hidden_rank,
                        "input_rank": 2,
                        "output_rank": 1,
                        "use_resnet": False,

                        # training hyperparameters
                        "num_epochs": 2000,
                        "batch_size": batch_size,
                        "num_training_samples": 600,
                        "num_test_samples": 133,

                        # learning rate schedule
                        "lr_init": lr_init,
                        "lr_gamma": 0.99,
                        "lr_step_size": 16,

                        # problem setup
                        "interval": [-1, 1],
                        "function": "0.3*sin(2*pi*x1) * sin(2*pi*x2) + 0.2*sin(2*pi*x1) * sin(4*pi*x2) + 0.2*sin(4*pi*x1) * sin(2*pi*x2) + 0.3*sin(4*pi*x1) * sin(4*pi*x2)",

                        # monitoring
                        "show_plot": False,
                        "device": str(device),
                        "dtype": str(mydtype)
                    })




# we define configuration dictionary with all hyperparameters
'''
config = {
    # architecture hyperparameters
    "num_layers": 6,  # number of hidden layers
    "hidden_width": 666,  # width of each hidden layer
    "hidden_rank": 15,  # rank of each hidden layer
    "input_rank": 1,  # rank of input layer
    "output_rank": 1,  # rank of output layer
    "use_resnet": False,  # whether to use resnet architecture

    # training hyperparameters
    "num_epochs": 3000,
    "batch_size": 100,
    "num_training_samples": 1000,  # uniform grid samples
    "num_test_samples": 1234,  # random samples

    # learning rate schedule: lr_init*lr_gamma**floor(k/lr_step_size)
    "lr_init": 0.001,
    "lr_gamma": 0.9,
    "lr_step_size": 100,

    # problem setup
    "interval": [-1, 1],
    "function": "cos(36*pi*x^2) - 0.8*cos(12*pi*x^2)",

    # monitoring
    "show_plot": False,
    "device": str(device),
    "dtype": str(mydtype)
}
'''
for config in configs:
    print(f"Training config: {config}")
    # we construct ranks and widths from config
    ranks = [config["input_rank"]] + [config["hidden_rank"]] * config["num_layers"] + [config["output_rank"]]
    widths = [config["hidden_width"]] * (config["num_layers"] + 1)

    # we create folder name from config
    folder_name = (f"mmnn_L{config['num_layers']}_"
                f"W{config['hidden_width']}_"
                f"R{config['hidden_rank']}_"
                f"E{config['num_epochs']}_"
                f"lr{config['lr_init']}_"
                f"bs{config['batch_size']}_"
                f"ntr{config['num_training_samples']}")

    # we create output directory
    output_dir = os.path.join("/Data/janis.aiad/", "mmnn_training2d", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # we save config to json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n=== CONFIGURATION ===")
    print(f"Output directory: {output_dir}")
    print(f"Ranks: {ranks}")
    print(f"Widths: {widths}")
    print(f"Config: {json.dumps(config, indent=2)}")

    model = MMNN(ranks=ranks,
                    widths=widths,
                    device=device,
                    ResNet=config["use_resnet"])
    
    def func(x):
        x1, x2 = x[:, 0], x[:, 1]
        s = 2
        a = np.array([[0.3, 0.2], 
                     [0.2, 0.3]])
        b = np.array([np.pi, 2*np.pi])
        c = np.array([[np.pi, 3*np.pi],
                     [np.pi, np.pi]])
        d = np.array([[2*np.pi, np.pi],
                     [np.pi, 2*np.pi]])
        y = 0
        for i in range(2):
            for j in range(2):
                y += a[i,j] * np.sin(s*b[i]*x1 + s*c[i,j]*x1*x2) * np.cos(s*b[j]*x2 + s*d[i,j]*x1**2)
        return y*10/np.sqrt(5)




    # nous vérifions l'initialisation
    print("\n=== WEIGHT INITIALIZATION CHECK ===")
    for j, layer in enumerate(model.fcs):
        w_norm = layer.weight.norm().item()
        b_norm = layer.bias.norm().item()
        w_mean = layer.weight.mean().item()
        w_std = layer.weight.std().item()
        frozen = "FROZEN" if not layer.weight.requires_grad else "trainable"

        print(f"Layer {j} ({frozen}): weight_norm={w_norm:.3f}, weight_mean={w_mean:.6f}, weight_std={w_std:.6f}, bias_norm={b_norm:.3f}")
        
        
        
    x1 = np.linspace(*config["interval"], config["num_training_samples"])
    x2 = np.linspace(*config["interval"], config["num_training_samples"]) 
    X1, X2 = np.meshgrid(x1, x2)
    X = np.concatenate([np.reshape(X1,[-1,1]),
                       np.reshape(X2,[-1,1])], axis=1)
    Y = func(X).reshape([-1,1])
    x_train = torch.tensor(X, device=device, dtype=mydtype)
    y_train = torch.tensor(Y, device=device, dtype=mydtype)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=config["batch_size"], shuffle=True)


    time1=time.time()
    errors_train=[]
    errors_test=[]
    errors_test_max=[]
    all_losses=[]  # we store all losses
    parameters_snapshots={}  # we store model parameters at selected epochs

    losses_std = []

    optimizer = optim.Adam(model.parameters(), lr=config["lr_init"])
    scheduler = StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])
    criterion = nn.MSELoss()

    has_changed_optimizer = False
    has_changed_optimizer_2 = False
    for epoch in range(1, 1 + config["num_epochs"]):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        all_losses.append(loss.item())  # we store loss
        scheduler.step()


        if epoch % 50 == 0:
            # we print the day hour etc ;;
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            training_error = loss.item()
            print(f"\nEpoch {epoch} / {config['num_epochs']}" +
                f"  ( {epoch/config['num_epochs']*100:.2f}% )" +
                f"\nTraining error (MSE): { training_error :.2e}" +
                f"\nTime used: { time.time() - time1 :.2f}s")
            errors_train.append(training_error)
            # we compute the std for the last 50 losses in the log space
            losses_std.append(np.std(np.log10(all_losses[-50:])))

            def learned_nn(x):  # input and output are numpy.ndarray
                x = x.reshape([-1, 2])
                input_data = torch.tensor(x, dtype=mydtype).to(device)
                y = model(input_data)
                y = y.cpu().detach().numpy().reshape([-1])
                return y


            x=np.linspace(-1, 1, 100)
            x1, x2 = np.meshgrid(x, x)
            X = np.concatenate([np.reshape(x1,[-1,1]),
                               np.reshape(x2,[-1,1])], axis=1)
            x = X
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

        if epoch % 250 == 0:
            # Track epoch progress
            print(f"Completed epoch {epoch} ({epoch/config['num_epochs']*100:.1f}% done)")
            
            if epoch % 100 == 0:
                # we plot the results as 2d heatmap
                n_plot = 100
                x1_plot = np.linspace(-1, 1, n_plot)
                x2_plot = np.linspace(-1, 1, n_plot)
                X1_plot, X2_plot = np.meshgrid(x1_plot, x2_plot)
                X_plot = np.concatenate([np.reshape(X1_plot,[-1,1]),
                                       np.reshape(X2_plot,[-1,1])], axis=1)
                
                y_nn_plot = learned_nn(X_plot).reshape(n_plot, n_plot)
                y_true_plot = func(X_plot).reshape(n_plot, n_plot)
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # we plot true function
                im0 = axes[0].imshow(y_true_plot, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
                axes[0].set_title('True function')
                axes[0].set_xlabel('$x_1$')
                axes[0].set_ylabel('$x_2$')
                plt.colorbar(im0, ax=axes[0])
                
                # we plot learned network
                im1 = axes[1].imshow(y_nn_plot, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
                axes[1].set_title('Learned network')
                axes[1].set_xlabel('$x_1$')
                axes[1].set_ylabel('$x_2$')
                plt.colorbar(im1, ax=axes[1])
                
                # we plot difference
                diff = np.abs(y_nn_plot - y_true_plot)
                im2 = axes[2].imshow(diff, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
                axes[2].set_title(f'Absolute error (max={np.max(diff):.2e})')
                axes[2].set_xlabel('$x_1$')
                axes[2].set_ylabel('$x_2$')
                plt.colorbar(im2, ax=axes[2])
                
                config_str = f"L={config['num_layers']}, W={config['hidden_width']}, R={config['hidden_rank']}"
                fig.suptitle(f'Epoch {epoch}\n{config_str}', fontsize=14)
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_dir, f"mmnn_epoch{epoch}_2D.png"), dpi=100)
                plt.close()
                if config["show_plot"]:
                    plt.show()

    torch.save(model.state_dict(), os.path.join(output_dir, 'model_parameters.pth'))
    np.savez(os.path.join(output_dir, "errors.npz"),
            test=np.array(errors_test),
            testmax=np.array(errors_test_max),
            train=np.array(errors_train),
            all_losses=np.array(all_losses),
            losses_std=np.array(losses_std),
            time=time.time()-time1
            )


    # we save model parameters evolution
    if len(parameters_snapshots) > 0:
        # we convert epoch keys to strings for npz format
        #params_str_keys = {f"epoch_{epoch}": params for epoch, params in parameters_snapshots.items()}
        #np.savez(os.path.join(output_dir, "parameters_evolution.npz"), **params_str_keys)
        #print(f"\nSaved {len(parameters_snapshots)} parameter snapshots to parameters_evolution.npz")
        # we also save epochs list for reference
        params_epochs = sorted(parameters_snapshots.keys())
        print(f"Parameters stored at epochs: {params_epochs}")
        print(f"Parameter vector size: {len(parameters_snapshots[params_epochs[0]])}")

    # we save results to json
    results = {
        "config": config,
        "final_train_error": float(errors_train[-1]) if len(errors_train) > 0 else None,
        "final_test_error": float(errors_test[-1]) if len(errors_test) > 0 else None,
        "final_test_error_max": float(errors_test_max[-1]) if len(errors_test_max) > 0 else None,
        "training_time_seconds": float(time.time()-time1),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters_epochs_stored": sorted(parameters_snapshots.keys()) if len(parameters_snapshots) > 0 else [],
        "parameter_vector_size": len(parameters_snapshots[list(parameters_snapshots.keys())[0]]) if len(parameters_snapshots) > 0 else None,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {os.path.join(output_dir, 'results.json')}")

    # we plot loss evolution
    config_str = f"L={config['num_layers']}, W={config['hidden_width']}, R={config['hidden_rank']}, lr={config['lr_init']}"
    fig = plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Training Loss Evolution\n{config_str}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_evolution.png'), dpi=100)
    plt.close()

    # we plot errors
    fig = plt.figure(figsize=(8, 5))
    n = len(errors_test)
    m = len(errors_train)
    plt.plot(np.linspace(1, m, m)*50, np.log10(errors_train),
            label="log10 training error", linewidth=2)
    plt.plot(np.linspace(1, n, n)*50, np.log10(errors_test),
            label="log10 test error", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('log10(error)')
    plt.title(f'Error Evolution\n{config_str}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_evolution.png'), dpi=100)
    plt.close()


    # we plot losses std
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(1, m, m)*50, losses_std, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Std')
    plt.title(f'Loss Std Evolution\n{config_str}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_std_evolution.png'), dpi=100)
    plt.close()

    # we plot final prediction/fit as 2d heatmaps
    n_plot = 150
    x1_plot = np.linspace(-1, 1, n_plot)
    x2_plot = np.linspace(-1, 1, n_plot)
    X1_plot, X2_plot = np.meshgrid(x1_plot, x2_plot)
    X_plot = np.concatenate([np.reshape(X1_plot,[-1,1]),
                           np.reshape(X2_plot,[-1,1])], axis=1)

    with torch.no_grad():
        X_plot_tensor = torch.tensor(X_plot, dtype=mydtype).to(device)  # we convert to tensor
        y_plot_nn = model(X_plot_tensor).cpu().numpy().reshape(n_plot, n_plot)  # we get predictions and convert back to numpy
    y_plot_true = func(X_plot).reshape(n_plot, n_plot)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # we plot true function
    im0 = axes[0].imshow(y_plot_true, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    axes[0].set_title('True function')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    plt.colorbar(im0, ax=axes[0])

    # we plot learned network
    im1 = axes[1].imshow(y_plot_nn, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    axes[1].set_title('Learned network')
    axes[1].set_xlabel('$x_1$')
    axes[1].set_ylabel('$x_2$')
    plt.colorbar(im1, ax=axes[1])

    # we plot absolute error as heatmap
    diff = np.abs(y_plot_nn - y_plot_true)
    im2 = axes[2].imshow(diff, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    axes[2].set_title(f'Absolute error (max={np.max(diff):.2e})')
    axes[2].set_xlabel('$x_1$')
    axes[2].set_ylabel('$x_2$')
    plt.colorbar(im2, ax=axes[2])

    config_str = f"L={config['num_layers']}, W={config['hidden_width']}, R={config['hidden_rank']}, lr={config['lr_init']}"
    fig.suptitle(f'Final Prediction vs True Function\n{config_str}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_prediction.png'), dpi=150)
    plt.close()

    print(f"\nAll plots saved to {output_dir}")
    print(f"Total training time: {time.time()-time1:.2f}s")
    # we visualize functions learned by each layer (adapted for 2d)
    teacher = MMNN(ranks=ranks,
                widths=widths,
                device=device,
                ResNet=config["use_resnet"])
    teacher.load_state_dict(model.state_dict())

    # we create 2d grid for layer visualization
    n_viz = 50  # we use smaller grid for speed
    x1_viz = np.linspace(-1, 1, n_viz)
    x2_viz = np.linspace(-1, 1, n_viz)
    X1_viz, X2_viz = np.meshgrid(x1_viz, x2_viz)
    X_viz = np.concatenate([np.reshape(X1_viz,[-1,1]),
                       np.reshape(X2_viz,[-1,1])], axis=1)
    x_tensor = torch.tensor(X_viz, dtype=mydtype).to(device)  # we create 2d tensor input

    # we iterate through layers
    for layer_idx in range(1, len(teacher.fcs), 1):
        # we determine output rank for this layer
        if layer_idx % 2 == 0:
            output_rank = ranks[layer_idx//2+1]
        else:
            output_rank = min(widths[(layer_idx)//2], 16)  # we limit to 16 for visualization

        print(f"Plotting layer {layer_idx} with output rank {output_rank}")
        
        # we compute layer output
        with torch.no_grad():
            current = x_tensor
            for i in range(layer_idx):
                current = teacher.fcs[i](current)
                if i % 2 == 0:  # we apply relu after first part of each layer
                    current = torch.relu(current)
            
            output = current.cpu().numpy()  # shape: (n_viz*n_viz, output_rank)
        
        # we plot components as 2d heatmaps
        n_rows = int(np.ceil(np.sqrt(output_rank)))
        n_cols = int(np.ceil(output_rank / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])  # we make 2d array for consistent indexing
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        fig.suptitle(f'Functions learned by Layer {layer_idx} (rank {output_rank})', fontsize=16)
        
        for idx in range(output_rank):
            i = idx // n_cols
            j = idx % n_cols
            
            # we reshape component to 2d grid
            component_2d = output[:, idx].reshape(n_viz, n_viz)
            
            # we plot as heatmap
            im = axes[i,j].imshow(component_2d, extent=[-1, 1, -1, 1], 
                                  origin='lower', cmap='viridis')
            axes[i,j].set_title(f'Component {idx+1}')
            axes[i,j].set_xlabel('$x_1$')
            axes[i,j].set_ylabel('$x_2$')
            plt.colorbar(im, ax=axes[i,j])
        
        # we hide unused subplots
        for idx in range(output_rank, n_rows * n_cols):
            i = idx // n_cols
            j = idx % n_cols
            axes[i,j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'layer_{layer_idx}_components_2d.png'), dpi=100)
        plt.close()

    print(f"Layer component plots saved to {output_dir}")
