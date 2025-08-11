# -*- coding: utf-8 -*-
"""
i created this on tue feb 18 20:25:05 2025

@author: gemini

this script plots the output of several randomly initialized 1-layer mmnns
to visualize the types of 1d functions they can represent out of the box.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from ntk.ntk_infinite import relu as jax_relu, sin as jax_sin

# i add project root to python path to allow imports from the 'model' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mmnn.mmnn_jax import MMNNJax

num_plots = 5 # i define the number of random networks to plot
net_depth = 1 # i want a 1-layer mmnn
net_width = 1024 # i set the width of the hidden layer, a large width allows for more complexity

ranks = [1, 1] # i define the ranks for the mmnn layers
widths = [net_width] * net_depth # i define the widths for the mmnn layers

plt.figure(figsize=(12, 8)) # i create the figure for plotting

x_domain = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 2000).reshape(-1, 1) # i define the input range for our 1d function

# i initialize the jax model
model_jax = MMNNJax(ranks=ranks, widths=widths, resnet=False, fix_wb=False, activation_fn=jax_sin)

# we jit the apply function for better performance
@jax.jit
def apply_model(params, x):
    return model_jax.apply({'params': params}, x)

key = jax.random.PRNGKey(0)

for i in range(num_plots): # i loop to create multiple plots
    print(f"plotting network {i+1}/{num_plots}...")
    key, subkey = jax.random.split(key)
    
    # we initialize parameters for each network instance
    params = model_jax.init(subkey, x_domain)['params']
    
    # we compute the model's output
    y_output = apply_model(params, x_domain)
    
    plt.plot(x_domain, y_output, label=f'initialization {i+1}') # i plot the result

plt.title(f"{num_plots} random 1-layer MMNNs (width={net_width}, activation=relu)") # i set the plot title
plt.xlabel("x") # i set the x-axis label
plt.ylabel("model(x)") # i set the y-axis label
plt.legend() # i display the legend
plt.grid(True) # i enable the grid
plt.ylim(-3, 3) # i limit y for better visualization as random weights can sometimes lead to large outputs

output_dir = 'figures/mmnn_plots' # i define the output directory
os.makedirs(output_dir, exist_ok=True) # i create the output directory if it doesn't exist
output_path = os.path.join(output_dir, '1_layer_mmnn_random_inits.png') # i define the output file path
plt.savefig(output_path) # i save the figure
print(f"plot saved to {output_path}")

plt.show() # i display the plot