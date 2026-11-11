"""2D Fourier Neural Operator, architecture-compatible with PDEBench's pretrained checkpoints.

Adapted from PDEBench ``pdebench/models/fno/fno.py`` (MIT License), which in turn follows
Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (ICLR 2021)
and https://github.com/zongyi-li/fourier_neural_operator (MIT License).

Parameter names (``fc0``, ``conv{i}.weights{1,2}``, ``w{i}``, ``fc1``, ``fc2``) are kept identical
to PDEBench so ``load_state_dict(..., strict=True)`` works on the published ``.pt`` files.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    """Fourier layer: FFT -> multiply the lowest ``modes1 x modes2`` modes -> inverse FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(self.scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in, x, y), (in, out, x, y) -> (batch, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            b, self.out_channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class FNO2d(nn.Module):
    """Autoregressive 2D FNO.

    Input:  ``x`` of shape ``[b, nx, ny, initial_step * num_channels]`` (the last
            ``initial_step`` frames, time-major) and ``grid`` of shape ``[b, nx, ny, 2]``
            holding the raw (x, y) cell-centre coordinates.
    Output: the next frame, shape ``[b, nx, ny, 1, num_channels]``.
    """

    def __init__(
        self,
        num_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 20,
        initial_step: int = 10,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.initial_step = initial_step
        self.num_channels = num_channels
        self.padding = 2  # pad the (non-periodic) domain on the high side

        self.fc0 = nn.Linear(initial_step * num_channels + 2, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        p = self.padding
        x = F.pad(x, [0, p, 0, p])

        blocks = ((self.conv0, self.w0), (self.conv1, self.w1),
                  (self.conv2, self.w2), (self.conv3, self.w3))
        for i, (conv, w) in enumerate(blocks):
            x = conv(x) + w(x)
            if i < len(blocks) - 1:
                x = F.gelu(x)

        x = x[..., :-p, :-p].permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)
