import torch
import torch.nn as nn

from experiments.helmholtz.models.mlp import MLP
from experiments.helmholtz.models.mmnn_test2d import MMNNTest2D


class DeepONet(nn.Module):  # we implement a deeponet u(x) = <B(input), T(coord)> #
    def __init__(
        self,
        branch: nn.Module,
        trunk: nn.Module,
        latent_dim: int,
        out_bias: bool = True,
    ):  # we store modules #
        super().__init__()  # we init #
        self.branch = branch  # we store #
        self.trunk = trunk  # we store #
        self.latent_dim = int(latent_dim)  # we store #
        self.bias = nn.Parameter(torch.zeros(1)) if bool(out_bias) else None  # we store bias #

    def forward(self, branch_in: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:  # we compute output at coords #
        b = self.branch(branch_in)  # we compute branch code (B, P) #
        t = self.trunk(coords)  # we compute trunk features (B, M, P) or (M,P) #
        if t.ndim == 2:  # we broadcast trunk if shared across batch #
            t = t.unsqueeze(0).expand(b.shape[0], -1, -1)  # we expand #
        u = torch.einsum("bp,bmp->bm", b, t)  # we form dot product #
        if self.bias is not None:  # we add bias #
            u = u + self.bias  # we add #
        return u  # we return (B,M) #


def build_trunk_mlp(latent_dim: int, hidden_dim: int, depth: int, activation: str = "relu") -> nn.Module:  # we build a trunk network #
    class Trunk(nn.Module):  # we define trunk wrapper #
        def __init__(self):  # we init #
            super().__init__()  # we init #
            self.mlp = MLP(in_dim=2, out_dim=int(latent_dim), hidden_dim=int(hidden_dim), depth=int(depth), activation=activation)  # we build #

        def forward(self, coords: torch.Tensor) -> torch.Tensor:  # we forward coords (M,2) or (B,M,2) #
            if coords.ndim == 2:  # we handle (M,2) #
                return self.mlp(coords)  # we return (M,P) #
            b, m, d = coords.shape  # we unpack #
            flat = coords.reshape(b * m, d)  # we flatten #
            out = self.mlp(flat).reshape(b, m, -1)  # we reshape #
            return out  # we return #

    return Trunk()  # we return #


def build_trunk_mmnn(
    latent_dim: int,
    depth: int,
    width: int,
    rank: int,
    device: str,
    fix_wb: bool,
    resnet: bool = False,
    normalize_output: bool = False,
) -> nn.Module:  # we build a trunk mmnn that acts as a high frequency basis #
    ranks = [2] + [int(rank)] * int(depth - 1) + [int(latent_dim)]  # we build ranks for coord input #
    widths = [int(width)] * int(depth)  # we build widths #
    model = MMNNTest2D(
        ranks=ranks,
        widths=widths,
        device=str(device),
        ResNet=bool(resnet),
        fixWb=bool(fix_wb),
        normalize_output=bool(normalize_output),
    )  # we build mmnn #
    return model  # we return #


def build_branch_mlp(in_dim: int, latent_dim: int, hidden_dim: int, depth: int, activation: str = "relu") -> nn.Module:  # we build branch mlp #
    return MLP(in_dim=int(in_dim), out_dim=int(latent_dim), hidden_dim=int(hidden_dim), depth=int(depth), activation=activation)  # we return #


def build_branch_mmnn(
    in_dim: int,
    latent_dim: int,
    depth: int,
    width: int,
    rank: int,
    device: str,
    fix_wb: bool,
    resnet: bool = False,
) -> nn.Module:  # we build a mmnn branch from your code #
    ranks = [int(in_dim)] + [int(rank)] * int(depth - 1) + [int(latent_dim)]  # we build rank list #
    widths = [int(width)] * int(depth)  # we build width list #
    model = MMNNTest2D(ranks=ranks, widths=widths, device=str(device), ResNet=bool(resnet), fixWb=bool(fix_wb))  # we build mmnn #
    return model  # we return #


class DeepONetGrid(nn.Module):  # we implement a grid deeponet with shared trunk over grid #
    def __init__(
        self,
        branch: nn.Module,
        trunk: nn.Module,
        coords_hw: torch.Tensor,
        latent_dim: int,
        trunk_is_fixed: bool,
    ):  # we store modules and coords #
        super().__init__()  # we init #
        self.branch = branch  # we store #
        self.trunk = trunk  # we store #
        self.latent_dim = int(latent_dim)  # we store #
        coords = coords_hw.reshape(-1, 2).detach().clone()  # we store flattened coords #
        self.register_buffer("coords", coords)  # we register coords #
        self.trunk_is_fixed = bool(trunk_is_fixed)  # we store flag #
        self.register_buffer("trunk_phi", torch.empty(0))  # we init optional cached basis #

    def maybe_cache_trunk(self) -> None:  # we cache trunk features when trunk is fixed #
        if not self.trunk_is_fixed:  # we skip if trainable #
            return  # we return #
        if self.trunk_phi.numel() > 0:  # we skip if cached #
            return  # we return #
        with torch.no_grad():  # we avoid grads #
            phi = self.trunk(self.coords)  # we compute (M,P) #
        self.trunk_phi = phi.detach()  # we cache #

    def forward(self, branch_in: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:  # we predict u on the grid #
        b = self.branch(branch_in)  # we compute coefficients (B,P) #
        if self.trunk_is_fixed and self.trunk_phi.numel() > 0:  # we use cached #
            phi = self.trunk_phi  # we read (M,P) #
        else:  # we compute fresh #
            phi = self.trunk(self.coords)  # we compute (M,P) #
        u_flat = b @ phi.T  # we compute (B,M) #
        h, w = int(hw[0]), int(hw[1])  # we unpack #
        u = u_flat.reshape(b.shape[0], 1, h, w)  # we reshape #
        return u  # we return #

