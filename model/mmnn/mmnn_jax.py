import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Optional



class MMNNJax(nn.Module):
    ranks: List[int] # i list where the i-th element represents output dimension of i-th layer
    widths: List[int] # list specifying width of each layer
    resnet: bool # whether to use resnet architecture with identity connections
    fix_wb: bool # if true, weights and biases not updated during training
    learning_rate: float = 0.01 # learning rate for training

    
    def setup(self):
        """Initialize the model layers."""
        self.depth = len(self.widths)
        
        fc_sizes = [self.ranks[0]]
        for j in range(self.depth):
            fc_sizes += [self.widths[j], self.ranks[j+1]]
        
        self.fc_sizes = fc_sizes
        # we create a list of Dense layers
        fcs = []
        for j in range(len(fc_sizes)):
            fc = nn.Dense(fc_sizes[j], use_bias=True)
            fcs.append(fc)
        self.fcs =fcs
            
                
                
# Handle fix_wb parameter by marking parameters as trainable/non-trainable
# Note: In Flax/JAX, parameter updates are controlled during training
# We'll need to handle the fix_wb logic in the training loop
# by filtering the parameters

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
                    x_id = x + 0.0  # make a copy to avoid inplace operations
                    
            x = self.fcs[2*j](x)
            x = jax.nn.relu(x)
            x = self.fcs[2*j+1](x)
            
            if self.resnet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x = x.at[:,:n].add(x_id[:,:n])
    
        x = jax.nn.relu(x)
        x = self.fcs[-1](x)
        return x
