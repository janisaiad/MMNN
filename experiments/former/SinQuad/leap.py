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
import random
import threading
import sys
import select


# %%


# %%

class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cuda", 
                 ResNet = False,
                 fixWb = False):
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
        return x

# %%

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


# we create global flags for keyboard control
skip_current_config = False
quit_all_training = False

def keyboard_listener():
    """we listen for keyboard input in a separate thread"""
    global skip_current_config, quit_all_training
    print("\n=== KEYBOARD CONTROLS ===")
    print("Press 's' + ENTER to skip current config")
    print("Press 'q' + ENTER to quit all training")
    print("========================\n")
    
    while not quit_all_training:
        try:
            # we check if input is available (works on linux)
            user_input = input()
            if user_input.lower() == 's':
                skip_current_config = True
                print("\n>>> SKIP REQUESTED - will skip current config after current epoch <<<\n")
            elif user_input.lower() == 'q':
                quit_all_training = True
                print("\n>>> QUIT REQUESTED - will stop all training after current epoch <<<\n")
        except:
            pass

# we start keyboard listener thread
listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
listener_thread.start()


configs = []
for alpha in [2,4,8,16]:
    for lr_init in [0.001]:
        for batch_size in [10,20,50,100,200,500,1000]:
            for num_layers in [0,1,2,3,4,5]:
                for hidden_width in [64,256,512]:
                    for hidden_rank in [1,3,5,8,15]:
                        seed = np.random.randint(0, 1000000)
                        
                        configs.append({
                            # architecture hyperparameters
                            "seed": seed,
                            "device": str(device),
                            "dtype": str(mydtype),
                            "num_layers": num_layers,
                            "hidden_width": hidden_width,
                            "hidden_rank": hidden_rank,
                            "input_rank": 1,
                            "output_rank": 1,
                            "use_resnet": False,

                            # training hyperparameters
                            "num_epochs": 5000,
                            "batch_size": batch_size,
                            "num_training_samples": 600,
                            "num_test_samples": 1000,

                            # learning rate schedule
                            "lr_init": lr_init,
                            "lr_gamma": 0.99,
                            "lr_step_size": 300,

                            # problem setup
                            "interval": [-1, 1],
                            "function": f"cos(2*pi*x) + 2*cos(2*pi*alpha*x)+0.5*cos(4*pi*x)",
                            "alpha": alpha,
                            
                            'show_plot': False,
                        })
            random.shuffle(configs)
            temp = []
            for num_layers in [6,8,10,12,15]:
                for hidden_width in [64,128,256,512,1024,2048]:
                    for hidden_rank in [1,2,3,4,5,6,7,8,9,10,20,30,50]:
                        seed = np.random.randint(0, 1000000)
                        
                        temp.append({
                            # architecture hyperparameters
                            "seed": seed,
                            "device": str(device),
                            "dtype": str(mydtype),
                            "num_layers": num_layers,
                            "hidden_width": hidden_width,
                            "hidden_rank": hidden_rank,
                            "input_rank": 1,
                            "output_rank": 1,
                            "use_resnet": False,

                            # training hyperparameters
                            "num_epochs": 5000,
                            "batch_size": batch_size,
                            "num_training_samples": 600,
                            "num_test_samples": 1000,

                            # learning rate schedule
                            "lr_init": lr_init,
                            "lr_gamma": 0.99,
                            "lr_step_size": 300,

                            # problem setup
                            "interval": [-1, 1],
                            "function": f"cos(2*pi*x) + 2*cos(2*pi*alpha*x)+0.5*cos(4*pi*x)",
                            "alpha": alpha,
                            
                            'show_plot': False,
                        })
            temp = []
            random.shuffle(temp)
            configs+=temp




plot_threshold = 0.5
plot_threshold=0.2
for config_idx, config in enumerate(configs):
    
    # we check if quit was requested
    if quit_all_training:
        print(f"\n=== QUIT REQUESTED - Stopping all training (processed {config_idx}/{len(configs)} configs) ===\n")
        break
    
    # we reset skip flag for this config
    skip_current_config = False
    
    # we set seeds 
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    # we create subfolder associated to alpha
    
    # we create folder name from config
    folder_name = (f"wavelet_L{config['num_layers']}_"
                f"W{config['hidden_width']}_"
                f"R{config['hidden_rank']}_"
                f"E{config['num_epochs']}_"
                f"lr{config['lr_init']}_"
                f"bs{config['batch_size']}_"
                f"ntr{config['num_training_samples']}")

    sub_folder_name = f"alpha{config['alpha']}"
    base_dir = os.path.join("/Data/janis.aiad/", "wavelet_leap", sub_folder_name)
    os.makedirs(base_dir, exist_ok=True)

    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(os.path.join(output_dir, 'final_prediction.png')): 
        
        print(f"\n{'='*80}")
        print(f"CONFIG {config_idx+1}/{len(configs)} - Training config: {config}")
        print(f"{'='*80}\n")
        # we construct ranks and widths from config
        ranks = [config["input_rank"]] + [config["hidden_rank"]] * config["num_layers"] + [config["output_rank"]]
        widths = [config["hidden_width"]] * (config["num_layers"] + 1)


        # we create output directory
        
        # we write the seed to the config
            
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
            y = np.cos(2*np.pi*x) + 2*np.cos(2*np.pi*config["alpha"]*x)+0.5*np.cos(4*np.pi*x)
            return y




        # nous vérifions l'initialisation
        print("\n=== WEIGHT INITIALIZATION CHECK ===")
        for j, layer in enumerate(model.fcs):
            w_norm = layer.weight.norm().item()
            b_norm = layer.bias.norm().item()
            w_mean = layer.weight.mean().item()
            w_std = layer.weight.std().item()
            frozen = "FROZEN" if not layer.weight.requires_grad else "trainable"

            print(f"Layer {j} ({frozen}): weight_norm={w_norm:.3f}, weight_mean={w_mean:.6f}, weight_std={w_std:.6f}, bias_norm={b_norm:.3f}")


        x_train = np.linspace(*config["interval"], config["num_training_samples"]).reshape([-1, 1])
        y_train = func(x_train)
        x_train = torch.tensor(x_train, device=device, dtype=mydtype)
        y_train = torch.tensor(y_train, device=device, dtype=mydtype)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config["batch_size"], shuffle=True)


        time1=time.time()
        errors_train=[]
        errors_test=[]
        errors_test_max=[]
        all_losses=[]  # we store all losses
        ntk_eigenvalues_full={}  # we store all ntk eigenvalues (full spectrum)
        ntk_eigenvalues={}  # we store ntk eigenvalues min/max
        ntk_matrices={}  # we store full ntk matrices at selected epochs
        parameters_snapshots={}  # we store model parameters at selected epochs

        losses_std = []

        optimizer = optim.Adam(model.parameters(), lr=config["lr_init"])
        scheduler = StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])
        criterion = nn.MSELoss()

        has_changed_optimizer = False
        has_changed_optimizer_2 = False
        
        
        initial_loss = criterion(model(x_train), y_train)
        next_plot_threshold = initial_loss * 0.8
        halving_index = 0
        
        for epoch in range(1, 1 + config["num_epochs"]):
            # we check if skip or quit was requested
            if skip_current_config:
                print(f"\n=== SKIP REQUESTED - Skipping config at epoch {epoch}/{config['num_epochs']} ===\n")
                break
            if quit_all_training:
                print(f"\n=== QUIT REQUESTED - Stopping training at epoch {epoch}/{config['num_epochs']} ===\n")
                break
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            all_losses.append(loss.item())  # we store loss
            scheduler.step()
            
            
            if float(loss.item()) <= 1e-5:
                print(f"\n=== CONVERGED - Loss below threshold at epoch {epoch} ===\n")
                break
            
            if float(loss.item()) <= float(next_plot_threshold):
                # we update the halving index
                halving_index += 1
                # we update the next plot threshold
                next_plot_threshold = initial_loss * plot_threshold ** halving_index
                
                
                # we print the day hour, halving index, and the full config, matching benchmark.py style

                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                training_error = loss.item()
                print(f"\nEpoch {epoch} / {config['num_epochs']}"
                      f"  ( {epoch/config['num_epochs']*100:.2f}% )"
                      f"\nTraining error (MSE): {training_error:.2e}"
                      f"\nTime used: {time.time() - time1:.2f}s"
                      f"\nOptimizer: Adam")
                print(f"halving_index={halving_index}", end="  |  ")
                print("CONFIG:",
                      f"num_layers={config['num_layers']}",
                      f"hidden_width={config['hidden_width']}",
                      f"hidden_rank={config['hidden_rank']}",
                      f"input_rank={config['input_rank']}",
                      f"output_rank={config['output_rank']}",
                      f"use_resnet={config['use_resnet']}",
                      f"num_epochs={config['num_epochs']}",
                      f"batch_size={config['batch_size']}",
                      f"alpha={config['alpha']}",
                      f"num_training_samples={config['num_training_samples']}",
                      f"num_test_samples={config['num_test_samples']}",
                      f"lr_init={config['lr_init']}",
                      f"lr_gamma={config['lr_gamma']}",
                      f"lr_step_size={config['lr_step_size']}",
                      f"interval={config['interval']}",
                      f"function={config['function']}",
                      f"show_plot={config['show_plot']}",
                      f"device={config['device']}",
                      f"dtype={config['dtype']}",
                        sep="  |  ")
                errors_train.append(training_error)
                
                
                # we compute the std for the last 50 losses in the log space
                losses_std.append(np.std(np.log10(all_losses[-50:])))

                def learned_nn(x):  # input and output are numpy.ndarray
                    x = x.reshape([-1, 1])
                    input_data = torch.tensor(x, dtype=mydtype).to(device)
                    y = model(input_data)
                    y = y.cpu().detach().numpy().reshape([-1])
                    return y


                x = np.random.rand(config["num_test_samples"]) * 2 - 1
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
                
                

                new_output_dir = os.path.join(output_dir, f"halving_{halving_index}_epoch_{epoch}")
                os.makedirs(new_output_dir, exist_ok=True)
                # ---- Do plot: prediction/output + all partial functions ----
                # 1. Plot final prediction/fit at this "halving" milestone
                x_plot = np.linspace(-1, 1, 1000)
                x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), dtype=mydtype).to(device)
                with torch.no_grad():
                    y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
                y_plot_true = func(x_plot_tensor.cpu().numpy())

                config_str = f"L={config['num_layers']}, W={config['hidden_width']}, R={config['hidden_rank']}, halving={halving_index}"

                fig = plt.figure(figsize=(8, 5))
                plt.plot(x_plot, y_plot_true, 'b-', label='True function', linewidth=2)
                plt.plot(x_plot, y_plot_nn, 'r--', label='Learned network', linewidth=2)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Halving idx {halving_index}: Prediction vs True Function\n{config_str}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plot_basename = f'prediction_halving_{halving_index}.png'
                plt.savefig(os.path.join(new_output_dir, plot_basename), dpi=100)
                plt.close()
                
                # we store model parameters as a flattened tensor
                params_flat = torch.cat([p.data.view(-1).cpu() for p in model.parameters()])
                parameters_snapshots[epoch] = params_flat.numpy()
               
                
                

                # 2. Plot all partial/layer functions and save it also in sepated folder for each layer_idx and for each idx
                
                
                teacher = MMNN(ranks=ranks,
                                widths=widths,
                                device=device,
                                ResNet=config["use_resnet"])
                teacher.load_state_dict(model.state_dict())

                x = np.linspace(-1, 1, 1000)
                x_tensor = torch.tensor(x.reshape([-1, 1]), dtype=mydtype).to(device)

                # we compute all layer outputs linearly in one forward pass
                layer_outputs = {}  # we store output after each layer operation, keyed by layer_idx
                with torch.no_grad():
                    current = x_tensor
                    for i in range(len(teacher.fcs)):
                        current = teacher.fcs[i](current)
                        layer_outputs[i] = current.cpu().numpy()  # we store output after linear layer (before relu if applicable)
                        if i % 2 == 0:  # we apply relu after first part of each layer
                            current = torch.relu(current)
                            # we update with post-relu output for even layers
                            layer_outputs[i] = current.cpu().numpy()

                # we now plot each layer using precomputed outputs
                for layer_idx in range(1, len(teacher.fcs), 1):
                    # we create a separated folder for each layer
                    layer_output_dir = os.path.join(new_output_dir, f"layer_{layer_idx}")
                    os.makedirs(layer_output_dir, exist_ok=True)
                    
                    # we get precomputed output for this layer
                    output = layer_outputs[layer_idx]
                    
                    # we ensure output is 2d
                    if len(output.shape) == 1:
                        output = output.reshape(-1, 1)
                    
                    # we use actual output shape instead of theoretical rank
                    output_rank = output.shape[1]
                    
                    
                    
                    output_rank = min(output_rank, 36)
                    
                    # we skip if output_rank is incompatible
                    if output_rank > output.shape[1]:
                        print(f"Warning: skipping layer {layer_idx}, output_rank={output_rank} > actual shape {output.shape}")
                        continue
                    
                    n_rows = int(np.ceil(np.sqrt(output_rank)))
                    n_cols = int(np.ceil(output_rank / n_rows))
                    
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
                    if n_rows == 1 and n_cols == 1:
                        axes = np.array([[axes]])
                    elif n_rows == 1 or n_cols == 1:
                        axes = axes.reshape(n_rows, n_cols)
                    
                    fig.suptitle(f'[halving {halving_index}] Functions learned by Layer {layer_idx} (rank {output_rank})', fontsize=16)

                    for idx in range(output_rank):
                        i = idx // n_cols
                        j = idx % n_cols
                        # we plot on the big combined figure
                        axes[i, j].plot(x, output[:, idx], 'b-', linewidth=1)
                        axes[i, j].set_title(f'Component {idx+1}')
                        axes[i, j].grid(True, alpha=0.3)
                        axes[i, j].set_xticks([-1, 0, 1])
                        
                        # we create individual plot for this component
                        subfig = plt.figure(figsize=(8, 5))
                        plt.plot(x, output[:, idx], 'b-', linewidth=2)
                        plt.title(f'Layer {layer_idx} Component {idx+1}')
                        plt.xlabel('x')
                        plt.ylabel('value')
                        plt.grid(True, alpha=0.3)
                        plt.xticks([-1, 0, 1])
                        plt.tight_layout()
                        plt.savefig(os.path.join(layer_output_dir, f"component_{idx}.png"), dpi=100)
                        plt.close(subfig)
                        
                        
                    # we save the big combined plot
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    components_plot_basename = f'layer_{layer_idx}_components_halving_{halving_index}.png'
                    plt.savefig(os.path.join(new_output_dir, components_plot_basename), dpi=100)
                    plt.close(fig)
                        
            
                print(f"\n[Halving {halving_index}] component plots (combined + individual) saved to {new_output_dir} at epoch {epoch}\n")
            

        

            if epoch % 50 == 0:
                
                
                # we print the day hour, halving index, and the full config, matching benchmark.py style

                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                training_error = loss.item()
                print(f"\nEpoch {epoch} / {config['num_epochs']}"
                      f"  ( {epoch/config['num_epochs']*100:.2f}% )"
                      f"\nTraining error (MSE): {training_error:.2e}"
                      f"\nTime used: {time.time() - time1:.2f}s"
                      f"\nOptimizer: Adam")
                print(f"halving_index={halving_index}", end="  |  ")
                print("CONFIG:",
                      f"num_layers={config['num_layers']}",
                      f"hidden_width={config['hidden_width']}",
                      f"hidden_rank={config['hidden_rank']}",
                      f"input_rank={config['input_rank']}",
                      f"output_rank={config['output_rank']}",
                      f"use_resnet={config['use_resnet']}",
                      f"num_epochs={config['num_epochs']}",
                      f"batch_size={config['batch_size']}",
                      f"alpha={config['alpha']}",
                      f"num_training_samples={config['num_training_samples']}",
                      f"num_test_samples={config['num_test_samples']}",
                      f"lr_init={config['lr_init']}",
                      f"lr_gamma={config['lr_gamma']}",
                      f"lr_step_size={config['lr_step_size']}",
                      f"interval={config['interval']}",
                      f"function={config['function']}",
                      f"show_plot={config['show_plot']}",
                      f"device={config['device']}",
                      f"dtype={config['dtype']}",
                        sep="  |  ")
                errors_train.append(training_error)
                
                
                # we compute the std for the last 50 losses in the log space
                losses_std.append(np.std(np.log10(all_losses[-50:])))

                def learned_nn(x):  # input and output are numpy.ndarray
                    x = x.reshape([-1, 1])
                    input_data = torch.tensor(x, dtype=mydtype).to(device)
                    y = model(input_data)
                    y = y.cpu().detach().numpy().reshape([-1])
                    return y


                x = np.random.rand(config["num_test_samples"]) * 2 - 1
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
                
             

        torch.save(model.state_dict(), os.path.join(output_dir, 'model_parameters.pth'))
        np.savez(os.path.join(output_dir, "errors.npz"),
                test=np.array(errors_test),
                testmax=np.array(errors_test_max),
                train=np.array(errors_train),
                all_losses=np.array(all_losses),
                losses_std=np.array(losses_std),
                time=time.time()-time1
                )

        # we save ntk matrices evolution
        if len(ntk_matrices) > 0:
            # we convert epoch keys to strings for npz format
            ntk_matrices_str_keys = {f"epoch_{epoch}": matrix for epoch, matrix in ntk_matrices.items()}
            np.savez(os.path.join(output_dir, "ntk_matrices.npz"), **ntk_matrices_str_keys)
            print(f"\nSaved {len(ntk_matrices)} NTK matrices to ntk_matrices.npz")
            # we also save epochs list for reference
            ntk_epochs = sorted(ntk_matrices.keys())
            print(f"NTK matrices stored at epochs: {ntk_epochs}")
            print(f"NTK matrix shape: {ntk_matrices[ntk_epochs[0]].shape}")

        # we save model parameters evolution
        if len(parameters_snapshots) > 0:
            # we convert epoch keys to strings for npz format
            params_str_keys = {f"epoch_{epoch}": params for epoch, params in parameters_snapshots.items()}
            np.savez(os.path.join(output_dir, "parameters_evolution.npz"), **params_str_keys)
            print(f"\nSaved {len(parameters_snapshots)} parameter snapshots to parameters_evolution.npz")
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
            "ntk_epochs_stored": sorted(ntk_matrices.keys()) if len(ntk_matrices) > 0 else [],
            "ntk_matrix_shape": ntk_matrices[list(ntk_matrices.keys())[0]].shape if len(ntk_matrices) > 0 else None,
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

        # we plot errors (note: test/train errors are sampled, not every epoch)
        if len(errors_test) > 0 and len(errors_train) > 0:
            fig = plt.figure(figsize=(8, 5))
            n = len(errors_test)
            m = len(errors_train)
            plt.plot(np.linspace(1, m, m)*50, np.log10(errors_train),
                    label="log10 training error", linewidth=2, marker='o')
            plt.plot(np.linspace(1, n, n)*50, np.log10(errors_test),
                    label="log10 test error", linewidth=2, marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('log10(error)')
            plt.title(f'Error Evolution (sampled)\n{config_str}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_evolution.png'), dpi=100)
            plt.close()


        # we plot losses std (note: losses_std is sampled, not every epoch)
        if len(losses_std) > 0:
            fig = plt.figure(figsize=(8, 5))
            plt.plot(np.linspace(1, len(losses_std), len(losses_std))*50, losses_std, 'b-', linewidth=2, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Std (log10 space, window=50)')
            plt.title(f'Loss Std Evolution (sampled)\n{config_str}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'loss_std_evolution.png'), dpi=100)
            plt.close()

        # we plot NTK eigenvalues (min/max)
        if len(ntk_eigenvalues) > 0:
            epochs_list = sorted(ntk_eigenvalues.keys())
            max_eigenvalues = [ntk_eigenvalues[ep][-1].item() for ep in epochs_list]
            min_eigenvalues = [abs(ntk_eigenvalues[ep][0].item()) for ep in epochs_list]  # we use absolute value

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            fig.suptitle(f'NTK Eigenvalue Evolution\n{config_str}', fontsize=12)

            ax1.plot(epochs_list, max_eigenvalues, 'b-', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Max Eigenvalue')
            ax1.set_title('NTK Maximum Eigenvalue')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            ax2.plot(epochs_list, min_eigenvalues, 'r-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('|Min Eigenvalue|')
            ax2.set_title('NTK Minimum Eigenvalue (Absolute Value)')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ntk_eigenvalues_minmax.png'), dpi=100)
            plt.close()

        # we plot full NTK eigenvalue spectrum (every 1000 epochs)
        if len(ntk_eigenvalues_full) > 0:
            epochs_full = sorted(ntk_eigenvalues_full.keys())

            # we create a plot with all eigenvalues at each epoch
            fig = plt.figure(figsize=(10, 6))

            for epoch in epochs_full[:10]:
                eigs = np.abs(ntk_eigenvalues_full[epoch])  # we use absolute value
                indices = np.arange(len(eigs))
                plt.semilogy(indices, eigs, '-o', markersize=3, label=f'Epoch {epoch}', alpha=0.7)

            plt.xlabel('Eigenvalue Index')
            plt.ylabel('|Eigenvalue| (log scale)')
            plt.title(f'Full NTK Eigenvalue Spectrum Evolution (Absolute Value)\n{config_str}')
            # i want to put the legend on the right side of the plot
            plt.legend(loc='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ntk_full_spectrum.png'), dpi=100)
            plt.close()

            # we plot first 5 and last 5 eigenvalues over time
            fig, axes = plt.subplots(10, 1, figsize=(10, 16))
            fig.suptitle(f'NTK Eigenvalues Evolution (First 5 and Last 5, Absolute Value)\n{config_str}', fontsize=14, y=0.995)

            # we plot first 5 eigenvalues (smallest)
            for i in range(5):
                ax = axes[i]
                eigenvalues_over_time = [abs(ntk_eigenvalues_full[ep][i]) for ep in epochs_full]  # we use absolute value
                ax.plot(epochs_full, eigenvalues_over_time, 'o-', linewidth=2, markersize=6)
                ax.set_ylabel(f'|λ_{i+1}|', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                if i < 4:
                    ax.set_xticklabels([])

            # we plot last 5 eigenvalues (largest)
            n_eigs = len(ntk_eigenvalues_full[epochs_full[0]])
            for i in range(5):
                ax = axes[5 + i]
                idx = n_eigs - 5 + i
                eigenvalues_over_time = [abs(ntk_eigenvalues_full[ep][idx]) for ep in epochs_full]  # we use absolute value
                ax.plot(epochs_full, eigenvalues_over_time, 'o-', linewidth=2, markersize=6, color='C1')
                ax.set_ylabel(f'|λ_{idx+1}|', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                if i < 4:
                    ax.set_xticklabels([])

            axes[-1].set_xlabel('Epoch', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ntk_first_last_eigenvalues.png'), dpi=100)
            plt.close()
            print("NTK first/last 5 eigenvalues plot saved")





        # we plot final prediction/fit
        final_output_dir = os.path.join(output_dir, 'final_prediction')
        os.makedirs(final_output_dir, exist_ok=True)
        x_plot = np.linspace(-1, 1, 1000)
        x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), dtype=mydtype).to(device)
        with torch.no_grad():
            y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
        y_plot_true = func(x_plot_tensor.cpu().numpy())

        fig = plt.figure(figsize=(8, 5))
        plt.plot(x_plot, y_plot_true, 'b-', label='True function', linewidth=2)
        plt.plot(x_plot, y_plot_nn, 'r--', label='Learned network', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Final Prediction vs True Function\n{config_str}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_prediction.png'), dpi=100)
        plt.close()

        print(f"\nAll plots saved to {output_dir}")
        print(f"Total training time: {time.time()-time1:.2f}s")
        
        # Plot functions learned by each low rank layer
        teacher = MMNN(ranks=ranks,
                        widths=widths,
                        device=device,
                        ResNet=config["use_resnet"])
        teacher.load_state_dict(model.state_dict())

        x = np.linspace(-1, 1, 1000)
        x_tensor = torch.tensor(x.reshape([-1, 1]), dtype=mydtype).to(device)

        
        
        # very bad complexity O(n^2)
        for layer_idx in range(1, len(teacher.fcs), 1):  # Even indices correspond to first part of each layer
            # if layer_idx is odd, we plot the first part of the layer, that means relu(something)
            # if layer_idx is even, we plot the second part of the layer, that means something
            # so we need to plot the first part of the layer if layer_idx is even, and the second part of the layer if layer_idx is odd
            if layer_idx % 2 == 0:
                output_rank = ranks[layer_idx//2+1]
            else:
                output_rank = min(widths[(layer_idx)//2], 36)


            print(f"Plotting layer {layer_idx} with output rank {output_rank}")
            # Plot components in a roughly rectangular grid
            n_rows = int(np.ceil(np.sqrt(output_rank)))
            n_cols = int(np.ceil(output_rank / n_rows))
            # Create subplot figure with dimensions based on output rank
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])  # Make 2D array for consistent indexing
            elif n_rows == 1 or n_cols == 1:
                axes = axes.reshape(n_rows, n_cols)
            fig.suptitle(f'Functions learned by Layer {layer_idx} (rank {output_rank})', fontsize=16)

            # Get layer output
            with torch.no_grad():
                # Apply layers up to current one
                current = x_tensor
                for i in range(layer_idx ):
                    current = teacher.fcs[i](current)
                    if i % 2 == 0:  # Apply ReLU after first part of each layer
                        current = torch.relu(current)

                output = current.cpu().numpy()

                for idx in range(output_rank):
                    i = idx // n_cols
                    j = idx % n_cols
                    axes[i,j].plot(x, output[:,idx], 'b-', linewidth=1)
                    axes[i,j].set_title(f'Component {idx+1}')
                    axes[i,j].grid(True, alpha=0.3)
                    axes[i, j].set_xticks([-1, 0, 1])

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(final_output_dir, f'layer_{layer_idx}_components.png'), dpi=100)
                plt.close()

            print(f"Layer component plots saved to {final_output_dir}")
        # we save the losses as a JSON file
        import json
        loss_history = [{"epoch": epoch, "loss": float(loss)} for epoch, loss in enumerate(all_losses)]
        with open(os.path.join(output_dir, 'loss.json'), 'w') as f:
            json.dump(loss_history, f, indent=2)
            
        
