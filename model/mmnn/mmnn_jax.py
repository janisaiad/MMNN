import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Optional

class MMNN(nn.Module):
    ranks: List[int] 
    widths: List[int] 
    resnet: bool 
    fix_wb: bool 

    def __init__(self, ranks, widths, resnet, fix_wb):
        self.ranks = ranks # i list where the i-th element represents output dimension of i-th layer
        self.widths = widths # list specifying width of each layer
        self.resnet = resnet # whether to use resnet architecture with identity connections
        self.fix_wb = fix_wb # if true, weights and biases not updated during training

    def setup(self):
        """Initialize the model layers."""
        self.depth = len(self.widths)
        
        fc_sizes = [self.ranks[0]]
        for j in range(self.depth):
            fc_sizes += [self.widths[j], self.ranks[j+1]]
        self.fcs = [nn.Dense(fc_sizes[j+1], use_bias=True) for j in range(len(fc_sizes)-1)]

    def __call__(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        for j in range(self.depth):
            if self.resnet:
                if 0 < j < self.depth-1:
                    x_id = x + 0.0  # make a copy
                    
            x = self.fcs[2*j](x)
            x = jax.nn.relu(x)
            x = self.fcs[2*j+1](x)
            
            if self.resnet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x = x.at[:,:n].add(x_id[:,:n])
                    
        return x
