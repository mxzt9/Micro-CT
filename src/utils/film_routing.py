from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import DoubleConv3D
except ImportError:  # pragma: no cover - прямой запуск из папки пакета
    from common import DoubleConv3D


class ContextSources3D(nn.Module):
    """Собирает четыре компактных источника контекста для FiLM-маршрутизации."""

    def __init__(self, enc1_ch: int, bottleneck_ch: int, ctx_dim: int = 64, ph_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.ph_dim = ph_dim

        self.proj_global = nn.Linear(bottleneck_ch, ctx_dim)
        self.proj_intensity = nn.Linear(4, ctx_dim)
        topo_in = ph_dim if ph_dim > 0 else bottleneck_ch
        self.proj_topo = nn.Linear(topo_in, ctx_dim)
        self.proj_texture = nn.Linear(2, ctx_dim)

    @staticmethod
    def _intensity_stats(x: torch.Tensor) -> torch.Tensor:
        dims = (1, 2, 3, 4)
        return torch.stack(
            [
                x.mean(dim=dims),
                x.std(dim=dims),
                x.amin(dim=dims),
                x.amax(dim=dims),
            ],
            dim=1,
        )

    @staticmethod
    def _texture_stats(feat: torch.Tensor) -> torch.Tensor:
        gz = feat[:, :, 1:, :, :] - feat[:, :, :-1, :, :]
        gy = feat[:, :, :, 1:, :] - feat[:, :, :, :-1, :]
        gx = feat[:, :, :, :, 1:] - feat[:, :, :, :, :-1]
        gz_mean = gz.abs().mean(dim=(1, 2, 3, 4))
        gy_mean = gy.abs().mean(dim=(1, 2, 3, 4))
        gx_mean = gx.abs().mean(dim=(1, 2, 3, 4))
        gmag = gz_mean + gy_mean + gx_mean
        aniso = gz_mean - gx_mean
        return torch.stack([gmag, aniso], dim=1)

    def forward(
        self,
        x_in: torch.Tensor,
        enc1: torch.Tensor,
        bottleneck: torch.Tensor,
        ph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b_glob = bottleneck.mean(dim=(2, 3, 4))
        c0 = self.proj_global(b_glob)
        c1 = self.proj_intensity(self._intensity_stats(x_in))
        if self.ph_dim > 0 and ph_features is not None:
            c2 = self.proj_topo(ph_features)
        else:
            c2 = self.proj_topo(b_glob)
        c3 = self.proj_texture(self._texture_stats(enc1))
        return torch.stack([c0, c1, c2, c3], dim=1)


class FiLMRouter(nn.Module):
    """Обучаемая маршрутизация alpha[level, source] и FiLM-головы по уровням."""

    def __init__(
        self,
        level_channels: list[int],
        num_sources: int = 4,
        ctx_dim: int = 64,
        diag_init: float = 4.0,
    ):
        super().__init__()
        self.L = len(level_channels)
        self.K = num_sources

        logits = torch.zeros(self.L, self.K)
        for level in range(min(self.L, self.K)):
            logits[level, level] = diag_init
        self.alpha_logits = nn.Parameter(logits)

        self.level_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ctx_dim, ctx_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(ctx_dim, 2 * channels),
                )
                for channels in level_channels
            ]
        )
        self.level_channels = level_channels

    def alpha(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=1)

    def forward(self, context: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        alpha = self.alpha()
        mixed = torch.einsum("lk,bkd->bld", alpha, context)
        params = []
        for level, mlp in enumerate(self.level_mlps):
            channels = self.level_channels[level]
            gamma_beta = mlp(mixed[:, level, :])
            gamma, beta = gamma_beta[:, :channels], gamma_beta[:, channels:]
            gamma = 1.0 + gamma
            view = (gamma.shape[0], channels, 1, 1, 1)
            params.append((gamma.reshape(view), beta.reshape(view)))
        return params


def film_modulate(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return gamma * h + beta


class FiLMRoutedUNet3D(nn.Module):
    """UNet3D с послойной FiLM-маршрутизацией по четырем источникам контекста."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        ctx_dim: int = 64,
        ph_dim: int = 0,
        return_embeddings: bool = True,
    ):
        super().__init__()
        bc = base_channels
        self.return_embeddings = return_embeddings

        self.enc1 = DoubleConv3D(in_channels, bc)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(bc, bc * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(bc * 2, bc * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(bc * 4, bc * 8, dropout=0.2)

        self.context = ContextSources3D(enc1_ch=bc, bottleneck_ch=bc * 8, ctx_dim=ctx_dim, ph_dim=ph_dim)
        self.router = FiLMRouter(level_channels=[bc * 8, bc * 4, bc * 2, bc], ctx_dim=ctx_dim)

        self.up3 = nn.ConvTranspose3d(bc * 8, bc * 4, 2, stride=2)
        self.dec3 = DoubleConv3D(bc * 8, bc * 4)
        self.up2 = nn.ConvTranspose3d(bc * 4, bc * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(bc * 4, bc * 2)
        self.up1 = nn.ConvTranspose3d(bc * 2, bc, 2, stride=2)
        self.dec1 = DoubleConv3D(bc * 2, bc)
        self.out_conv = nn.Conv3d(bc, out_channels, 1)

    def forward(self, x: torch.Tensor, ph_features: Optional[torch.Tensor] = None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.bottleneck(self.pool3(e3))

        ctx = self.context(x, e1, bottleneck, ph_features)
        (g1, b1), (g2, b2), (g3, b3), (g4, b4) = self.router(ctx)

        bottleneck = film_modulate(bottleneck, g1, b1)
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d3 = film_modulate(d3, g2, b2)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d2 = film_modulate(d2, g3, b3)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = film_modulate(d1, g4, b4)

        logits = self.out_conv(d1)
        if self.return_embeddings:
            return logits, d1
        return logits
