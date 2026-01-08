import torch
import torch.nn as nn


class MLP(nn.Module):  # we implement a simple mlp #
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int, activation: str = "relu"):  # we store sizes #
        super().__init__()  # we init #
        self.in_dim = int(in_dim)  # we store #
        self.out_dim = int(out_dim)  # we store #
        self.hidden_dim = int(hidden_dim)  # we store #
        self.depth = int(depth)  # we store #
        if activation == "relu":  # we select activation #
            act = nn.ReLU()  # we set #
        elif activation == "gelu":  # we select activation #
            act = nn.GELU()  # we set #
        else:  # we fail #
            raise ValueError(f"unknown activation={activation!r}")  # we raise #
        layers = []  # we collect layers #
        if self.depth <= 1:  # we handle linear case #
            layers.append(nn.Linear(self.in_dim, self.out_dim))  # we append #
        else:  # we handle deep case #
            layers.append(nn.Linear(self.in_dim, self.hidden_dim))  # we append #
            layers.append(act)  # we append #
            for _ in range(self.depth - 2):  # we add hidden layers #
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))  # we append #
                layers.append(act)  # we append #
            layers.append(nn.Linear(self.hidden_dim, self.out_dim))  # we append #
        self.net = nn.Sequential(*layers)  # we build sequential #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # we forward #
        return self.net(x)  # we return #

