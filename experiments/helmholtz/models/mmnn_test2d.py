import numpy as np
import torch
import torch.nn as nn


class MMNNTest2D(nn.Module):  # we mirror the mmnn used in experiments/former/SinQuad/test2d.py #
    def __init__(
        self,
        ranks: list[int],
        widths: list[int],
        device: str,
        ResNet: bool = False,
        fixWb: bool = True,
        normalize_output: bool = True,
    ):  # we store hyperparameters #
        super().__init__()  # we init #
        self.product = 1.0  # we precompute output normalization #
        for j in range(1, len(ranks)):  # we accumulate product #
            self.product *= float(np.sqrt(widths[j - 1] * ranks[j]))  # we multiply #
        self.ranks = ranks  # we store #
        self.widths = widths  # we store #
        self.ResNet = bool(ResNet)  # we store #
        self.depth = len(widths)  # we store #
        self.normalize_output = bool(normalize_output)  # we store flag #

        fc_sizes = [ranks[0]]  # we set input size #
        for j in range(self.depth):  # we build alternating sizes #
            fc_sizes += [widths[j], ranks[j + 1]]  # we append #

        fcs = []  # we collect layers #
        for j in range(len(fc_sizes) - 1):  # we build linear list #
            fc = nn.Linear(fc_sizes[j], fc_sizes[j + 1], device=device)  # we create #
            fcs.append(fc)  # we append #
        self.fcs = nn.ModuleList(fcs)  # we store #

        if bool(fixWb):  # we optionally freeze random features layers #
            for j in range(len(fcs)):  # we loop #
                if j % 2 == 0:  # we freeze the first linear of each block #
                    self.fcs[j].weight.requires_grad = False  # we freeze #
                    self.fcs[j].bias.requires_grad = False  # we freeze #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # we forward #
        for j in range(self.depth):  # we loop blocks #
            if self.ResNet:  # we optionally store skip #
                if 0 < j < self.depth - 1:  # we apply internal skips #
                    x_id = x + 0  # we copy #
            x = self.fcs[2 * j](x)  # we apply first linear #
            x = torch.relu(x)  # we apply relu #
            x = self.fcs[2 * j + 1](x)  # we apply second linear #
            if self.ResNet:  # we apply skip #
                if 0 < j < self.depth - 1:  # we apply internal skips #
                    n = min(x.shape[1], x_id.shape[1])  # we match dims #
                    x[:, :n] = x[:, :n] + x_id[:, :n]  # we add skip #
        if self.normalize_output:  # we optionally normalize #
            return x / float(self.product)  # we normalize output #
        return x  # we return raw output #

