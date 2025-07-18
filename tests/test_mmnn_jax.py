import jax
import jax.numpy as jnp
import pytest
from mmnn.mmnn_jax import MMNNJax # type: ignore

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
    widths = [3, 3]
    model = MMNNJax(ranks=ranks, widths=widths, resnet=True, fix_wb=False)
    
    batch_size = 3
    x = jnp.ones((batch_size, ranks[0]))
    variables = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(variables, x)
    
    assert y.shape == (batch_size, ranks[-1])  # we check output shape with resnet

def test_invalid_ranks_widths():
    with pytest.raises(AssertionError):  # we check if invalid dimensions raise error
        ranks = [2]  # invalid ranks (need at least input and output dimensions)
        widths = [3]
        MMNNJax(ranks=ranks, widths=widths, resnet=False, fix_wb=False)


if __name__ == "__main__":
    test_mmnn_init()
    test_mmnn_forward()
    test_mmnn_resnet()
    test_invalid_ranks_widths()