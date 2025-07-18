import jax
import jax.numpy as jnp
import pytest
from mmnn.mmnn_jax import MMNNJax # type: ignore
import optax
from flax.training.train_state import TrainState


def test_mmnn_init():
    ranks = [2, 4, 1]  # i list the input dimension, hidden dimension, output dimension
    widths = [10, 10]  # list specifying width of each layer
    model = MMNNJax(ranks=ranks, widths=widths, resnet=False, fix_wb=False)
    assert model.ranks == ranks  # we check if ranks are properly initialized
    assert model.widths == widths  # we check if widths are properly initialized
    assert not model.resnet  # we check if resnet flag is properly initialized
    assert not model.fix_wb  # we check if fix_wb flag is properly initialized

def test_mmnn_forward():
    ranks = [2, 4, 1] 
    widths = [10, 10]
    model = MMNNJax(ranks=ranks, widths=widths, resnet=False, fix_wb=False)
    
    batch_size = 3
    x = jnp.ones((batch_size, ranks[0]))  # we create input tensor
    variables = model.init(jax.random.PRNGKey(0), x)  # we initialize model parameters
    y = model.apply(variables, x)  # we apply forward pass
    
    assert y.shape == (batch_size, ranks[-1])  # we check output shape

def test_mmnn_resnet():
    ranks = [2, 4, 4, 1]
    widths = [10, 10, 10]
    model = MMNNJax(ranks=ranks, widths=widths, resnet=True, fix_wb=False)
    
    batch_size = 3
    x = jnp.ones((batch_size, ranks[0]))
    variables = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(variables, x)
    
    assert y.shape == (batch_size, ranks[-1])  # we check output shape with resnet



    
class MMNNJaxTrainState(TrainState):
    pass
    
def create_train_state(module, rng, learning_rate, dummy_input):
    """we create and initialize the training state."""
    params = module.init(rng, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return MMNNJaxTrainState.create(apply_fn=module.apply, params=params, tx=tx)

@jax.jit # we jit-compile the function for huge performance boost
def train_step(state, batch_x, batch_y):
    """we perform a single training step."""
    def loss_fn(params):
        y_pred = state.apply_fn({'params': params}, batch_x)
        loss = jnp.mean(jnp.square(y_pred - batch_y))
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state

    
def test_mmnn_train():
    
    ranks = [2, 4, 1]
    widths = [10, 10]
    learning_rate = 0.01
    batch_size = 3
    num_epochs = 10
    
    model = MMNNJax(ranks=ranks, widths=widths, resnet=False, fix_wb=False)
    
    
    x = jnp.ones((1, ranks[0]))
    y = jnp.ones((1, ranks[-1]))
    
    state = create_train_state(model, jax.random.PRNGKey(0), learning_rate, x)
    for epoch in range(num_epochs):
        state = train_step(state, x, y)
        print(f"Epoch {epoch+1}/{num_epochs} completed")
    
    variables = {'params': state.params}
    assert model.apply(variables, x).shape == y.shape

    
if __name__ == "__main__":
    test_mmnn_init()
    test_mmnn_forward()
    test_mmnn_resnet()
    test_mmnn_train()