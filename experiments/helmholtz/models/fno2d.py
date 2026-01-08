import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):  # we implement a 2d spectral convolution layer #
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):  # we store dims #
        super().__init__()  # we init module #
        self.in_channels = int(in_channels)  # we store #
        self.out_channels = int(out_channels)  # we store #
        self.modes1 = int(modes1)  # we store #
        self.modes2 = int(modes2)  # we store #
        scale = 1.0 / (self.in_channels * self.out_channels)  # we scale init #
        self.weight = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, 2)
        )  # we store complex weights as (re,im) #

    def compl_mul2d(self, input_ft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:  # we multiply complex tensors #
        w = torch.view_as_complex(weight)  # we cast weight to complex #
        return torch.einsum("bixy,ioxy->boxy", input_ft, w)  # we contract #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # we apply spectral convolution #
        b, c, n1, n2 = x.shape  # we unpack #
        x_ft = torch.fft.rfft2(x, norm="ortho")  # we compute rfft2 #
        out_ft = torch.zeros(
            (b, self.out_channels, n1, n2 // 2 + 1),
            device=x.device,
            dtype=torch.cfloat,
        )  # we allocate output fourier coeffs #
        w = self.weight  # we read parameter #
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], w
        )  # we fill low modes #
        out = torch.fft.irfft2(out_ft, s=(n1, n2), norm="ortho")  # we invert #
        return out  # we return #


class FNO2d(nn.Module):  # we implement a small 2d fno #
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        width: int = 64,
        depth: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        padding: int = 6,
    ):  # we store hyperparameters #
        super().__init__()  # we init #
        self.in_channels = int(in_channels)  # we store #
        self.out_channels = int(out_channels)  # we store #
        self.width = int(width)  # we store #
        self.depth = int(depth)  # we store #
        self.padding = int(padding)  # we store #

        self.fc0 = nn.Conv2d(self.in_channels, self.width, kernel_size=1)  # we lift to width #
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, modes1=modes1, modes2=modes2) for _ in range(self.depth)]
        )  # we create spectral layers #
        self.w_layers = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(self.depth)]
        )  # we create pointwise layers #
        self.act = nn.GELU()  # we set activation #

        self.fc1 = nn.Conv2d(self.width, self.width, kernel_size=1)  # we postprocess #
        self.fc2 = nn.Conv2d(self.width, self.out_channels, kernel_size=1)  # we project #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # we run the fno forward #
        x = self.fc0(x)  # we lift #
        if self.padding > 0:  # we pad for non-periodic boundaries #
            x = nn.functional.pad(x, (0, self.padding, 0, self.padding))  # we pad (w,h) #
        for k in range(self.depth):  # we iterate layers #
            x1 = self.spectral_layers[k](x)  # we apply spectral conv #
            x2 = self.w_layers[k](x)  # we apply pointwise conv #
            x = self.act(x1 + x2)  # we combine #
        if self.padding > 0:  # we unpad #
            x = x[:, :, : -self.padding, : -self.padding]  # we crop #
        x = self.act(self.fc1(x))  # we apply #
        x = self.fc2(x)  # we project to output #
        return x  # we return #

