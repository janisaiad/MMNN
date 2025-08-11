import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Optional, Callable



class MMNNJax(nn.Module):
    ranks: List[int] # i list where the i-th element represents output dimension of i-th layer
    widths: List[int] # list specifying width of each layer
    resnet: bool # whether to use resnet architecture with identity connections
    fix_wb: bool # if true, weights and biases not updated during training
    learning_rate: float = 0.01 # learning rate for training
    activation_fn: Callable = jax.nn.relu # activation function
    beta: float = 1.0 # beta parameter for the activation function
    sigma_c: float = 1.0 # sigma_c parameter for the activation function
    sigma_A: float = 1.0 # sigma_A parameter for the activation function
    
    def setup(self):
        """Initialize the model layers."""
        self.depth = len(self.widths)
        
        fc_sizes = [self.ranks[0]]
        for j in range(self.depth):
            fc_sizes += [self.widths[j], self.ranks[j+1]]
        
        self.fc_sizes = fc_sizes
        
        # we create a list of Dense layers
        fcs = []
        for j in range(len(fc_sizes)): # we skip the first and last layer, we will only have inputs and outputs weights
            fc = nn.Dense(fc_sizes[j], use_bias=True,bias_init=nn.initializers.normal(stddev=1)) # the variance of the weights is 1/d_0
            fcs.append(fc)
        self.fcs =fcs[1:]
        
            
                
                
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
        
        for j in range(self.depth): # we skip the first and last layer
            if self.resnet:
                if 0 < j < self.depth-1:
                    x_id = x + 0.0  # make a copy to avoid inplace operations
                    
            x = self.fcs[2*j](x)
            x = self.activation_fn(x)
            x = self.fcs[2*j+1](x)
            
            if self.resnet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x = x.at[:,:n].add(x_id[:,:n])
        return x


if __name__ == "__main__":
    from ntk.ntk_infinite import relu as jax_relu
    key = jax.random.PRNGKey(0) # we create a random key
    dummy_input = jnp.ones((1, 1)) # we create a dummy input, e.g., batch_size=1, features=10

    model = MMNNJax( # we initialize the model
        ranks=[1, 1, 1], 
        widths=[10, 10], 
        resnet=False, 
        fix_wb=True, 
        activation_fn=jax.nn.relu
    )

    print("--- model architecture summary ---") # we print the model architecture
    # we use the tabulate method to get a string representation of the model's architecture.
    # this requires a prngkey and a dummy input to trace the model.
    # i've set the width to 120 to avoid wrapping.
    print(model.tabulate(key, dummy_input, console_kwargs={'width': 120}))
    
    # we initialize the parameters. note that tabulate also calls init internally.
    params = model.init(key, dummy_input)['params']
    print("\n--- initialized parameters ---") # we print the initialized parameters
    #print(params)