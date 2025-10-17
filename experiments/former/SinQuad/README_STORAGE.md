# Training Data Storage Structure

This document explains how data is stored during training and how to load it.

## Saved Files

All files are saved in a folder whose name contains the configuration:
```
figures/mmnn_training/largescaletraining/mmnn_L{layers}_W{width}_R{rank}_E{epochs}_lr{lr}_bs{batch_size}_ntr{n_train}/
```

### 1. `config.json`
Contains all hyperparameters of the configuration.

### 2. `results.json`
Contains the final training results:
- Final errors (train, test, test max)
- Training time
- Number of parameters
- Epochs where data was stored

### 3. `errors.npz`
Contains training and test errors:
- `test`: MSE errors on the test set
- `testmax`: maximum errors on the test set  
- `train`: MSE errors on the training set
- `all_losses`: all training losses (each epoch)
- `losses_std`: standard deviation of losses
- `time`: total training time

### 4. `ntk_matrices.npz`
Contains complete NTK matrices stored every 100 epochs (up to 3000):
- `epoch_100`: NTK matrix at epoch 100
- `epoch_200`: NTK matrix at epoch 200
- etc.

**Format**: matrices of size `(n_samples, n_samples)`

### 5. `parameters_evolution.npz`
Contains snapshots of model parameters every 100 epochs (up to 3000):
- `epoch_100`: flattened parameter vector at epoch 100
- `epoch_200`: flattened parameter vector at epoch 200
- etc.

**Format**: 1D vectors containing all concatenated weights and biases

### 6. `model_parameters.pth`
Final model parameters in PyTorch format (state_dict).

### 7. Generated Plots
- `loss_evolution.png`: loss evolution
- `error_evolution.png`: train/test error evolution
- `loss_std_evolution.png`: loss standard deviation evolution
- `ntk_eigenvalues_minmax.png`: min/max NTK eigenvalues
- `ntk_full_spectrum.png`: complete NTK spectrum
- `ntk_first_last_eigenvalues.png`: evolution of first and last eigenvalues
- `final_prediction.png`: final prediction vs true function
- `layer_{i}_components.png`: components learned by each layer

## How to Load Data

### Load Errors
```python
import numpy as np

data = np.load("errors.npz")
train_errors = data['train']
test_errors = data['test']
all_losses = data['all_losses']
```

### Load NTK Matrices
```python
ntk_data = np.load("ntk_matrices.npz")

# we get all available epochs
all_epochs = sorted([int(key.split('_')[1]) for key in ntk_data.keys()])

# we load ntk at epoch 100
ntk_100 = ntk_data['epoch_100']

# we compute eigenvalues
eigenvalues = np.linalg.eigvalsh(ntk_100)
```

### Load Parameter Evolution
```python
params_data = np.load("parameters_evolution.npz")

# we get all available epochs
params_epochs = sorted([int(key.split('_')[1]) for key in params_data.keys()])

# we load parameters at epoch 100
params_100 = params_data['epoch_100']

# we compute distance traveled
params_initial = params_data[f'epoch_{params_epochs[0]}']
params_final = params_data[f'epoch_{params_epochs[-1]}']
distance = np.linalg.norm(params_final - params_initial)
```

### Load Final Model
```python
import torch
import mmnn

# we recreate model architecture (use same config)
model = mmnn.MMNN(ranks=ranks, widths=widths, device=device, ResNet=False)

# we load saved parameters
model.load_state_dict(torch.load("model_parameters.pth"))
```

## Example Script

The `load_ntk_example.py` script shows complete analysis examples:
- Loading and visualizing NTK matrices
- Eigenvalue analysis
- Parameter evolution
- Computing distances in parameter space
- Generating plots

To use it:
```bash
python load_ntk_example.py
```

