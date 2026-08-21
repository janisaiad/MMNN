import torch

from mmnn.right_factor import RightFactorMMNN


def test_only_right_factor_is_trainable() -> None:
    model = RightFactorMMNN(
        feature_width=7,
        outer_width=9,
        rank=3,
        seed=0,
        device=torch.device("cpu"),
    )

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == ["V"]
    assert model.U.shape == (9, 3)
    assert model.V.shape == (7, 3)


def test_right_factor_tangent_kernel_factorization() -> None:
    model = RightFactorMMNN(
        feature_width=8,
        outer_width=10,
        rank=3,
        seed=3,
        device=torch.device("cpu"),
    )
    x = torch.tensor([0.37, 1.41])
    predictions = model(x)
    gradients = [
        torch.autograd.grad(predictions[index], model.V, retain_graph=True)[0]
        for index in range(2)
    ]
    empirical_kernel = torch.sum(gradients[0] * gradients[1])

    features = model.first_features(x)
    preactivations = (features @ model.V) @ model.U.T + model.b2
    gates = (preactivations > 0).to(features.dtype)
    backpropagated = (gates * model.readout) @ model.U
    factorized_kernel = torch.dot(features[0], features[1]) * torch.dot(
        backpropagated[0], backpropagated[1]
    )

    torch.testing.assert_close(empirical_kernel, factorized_kernel)
