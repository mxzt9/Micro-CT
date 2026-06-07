from __future__ import annotations

import torch
import torch.nn as nn


def choose_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct3D(nn.Module):
    """Две 3D-свертки с GroupNorm для обучения с любым размером батча."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv3D(ConvGNAct3D):
    """Совместимое имя для старых ноутбуков и FiLM-модулей."""


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()

        bc = base_channels
        self.enc1 = DoubleConv3D(in_channels, bc)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(bc, bc * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(bc * 2, bc * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(bc * 4, bc * 8, dropout=0.2)

        self.up3 = nn.ConvTranspose3d(bc * 8, bc * 4, 2, stride=2)
        self.dec3 = DoubleConv3D(bc * 8, bc * 4)
        self.up2 = nn.ConvTranspose3d(bc * 4, bc * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(bc * 4, bc * 2)
        self.up1 = nn.ConvTranspose3d(bc * 2, bc, 2, stride=2)
        self.dec1 = DoubleConv3D(bc * 2, bc)
        self.out_conv = nn.Conv3d(bc, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
