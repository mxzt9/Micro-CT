from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import choose_groups, match_size, pad_to_multiple_3d, unpad_3d
from .topology import TOPOLOGY_FEATURE_DIM


def _choose_attention_heads(ctx_dim: int, max_heads: int = 4) -> int:
    for heads in range(min(max_heads, ctx_dim), 0, -1):
        if ctx_dim % heads == 0:
            return heads
    return 1


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.skip(x) + self.net(x))


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x.mean(dim=(2, 3, 4))).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * weights


class AttentionGate3D(nn.Module):
    def __init__(self, feat_ch: int, gate_ch: int):
        super().__init__()
        inter_ch = max(feat_ch // 2, 1)
        self.feat_proj = nn.Conv3d(feat_ch, inter_ch, 1, bias=False)
        self.gate_proj = nn.Conv3d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv3d(inter_ch, 1, 1)

    def forward(self, feat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = match_size(gate, feat.shape[-3:])
        alpha = torch.sigmoid(self.psi(F.gelu(self.feat_proj(feat) + self.gate_proj(gate))))
        return feat * alpha


class TrilinearUpsampleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.channel_proj = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return self.channel_proj(x)


class AdaGN3D(nn.Module):
    def __init__(self, ctx_dim: int, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(choose_groups(channels), channels, affine=False)
        self.mlp = nn.Linear(ctx_dim, 2 * channels)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mlp(z).chunk(2, dim=-1)
        view = (h.shape[0], h.shape[1], 1, 1, 1)
        scale = scale.reshape(view)
        shift = shift.reshape(view)
        return self.norm(h) * (1.0 + scale) + shift


class MultiScaleContext3D(nn.Module):
    def __init__(
        self,
        feature_channels: list[int],
        ctx_dim: int,
        *,
        ph_dim: int = 0,
    ):
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.ph_dim = int(ph_dim)
        self.feature_projs = nn.ModuleList(nn.Linear(channels, ctx_dim) for channels in feature_channels)
        self.input_proj = nn.Linear(4, ctx_dim)
        self.texture_proj = nn.Linear(2, ctx_dim)
        self.topology_proj = nn.Linear(ph_dim, ctx_dim) if ph_dim > 0 else None
        self.num_sources = len(feature_channels) + 2 + (1 if ph_dim > 0 else 0)

    @staticmethod
    def _input_stats(x: torch.Tensor) -> torch.Tensor:
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
        return torch.stack([gz_mean + gy_mean + gx_mean, gz_mean - gx_mean], dim=1)

    def forward(
        self,
        x_in: torch.Tensor,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        ph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = [self.input_proj(self._input_stats(x_in))]
        tokens.append(self.texture_proj(self._texture_stats(features[0])))
        for feat, proj in zip(features, self.feature_projs):
            tokens.append(proj(feat.mean(dim=(2, 3, 4))))

        if self.topology_proj is not None:
            if ph_features is None:
                raise ValueError("ph_features are required when ph_dim > 0")
            ph_features = ph_features.to(device=x_in.device, dtype=x_in.dtype)
            tokens.append(self.topology_proj(ph_features))

        return torch.stack(tokens, dim=1)


class DynamicContextRouter(nn.Module):
    def __init__(self, num_sources: int, num_levels: int, ctx_dim: int):
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_levels = int(num_levels)
        self.ctx_dim = int(ctx_dim)
        heads = _choose_attention_heads(ctx_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, ctx_dim))
        nn.init.normal_(self.query_token, std=0.02)
        self.attn_pool = nn.MultiheadAttention(ctx_dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, num_levels * num_sources),
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        query = self.query_token.expand(batch_size, -1, -1)
        pooled, _ = self.attn_pool(query, context, context)
        pooled = pooled.squeeze(1)
        logits = self.mlp(pooled).reshape(batch_size, self.num_levels, self.num_sources)
        alpha = F.softmax(logits, dim=-1)
        mixed = torch.einsum("blk,bkd->bld", alpha, context)
        return mixed, alpha, pooled


class _AdaptiveRoutedUNet3DBase(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        ctx_dim: int = 64,
        ph_dim: int = 0,
        return_embeddings: bool = True,
        rock_embedding_dim: int = 128,
        topology_dim: int = 0,
    ):
        super().__init__()
        bc = int(base_channels)
        self.return_embeddings = return_embeddings
        self.rock_embedding_dim = int(rock_embedding_dim)
        self.ph_dim = int(ph_dim)
        self.topology_dim = int(topology_dim)

        self.enc1 = ResBlock3D(in_channels, bc)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResBlock3D(bc, bc * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResBlock3D(bc * 2, bc * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(ResBlock3D(bc * 4, bc * 8, dropout=0.2), SEBlock3D(bc * 8))

        level_channels = [bc * 8, bc * 4, bc * 2, bc]
        self.context = MultiScaleContext3D([bc, bc * 2, bc * 4, bc * 8], ctx_dim, ph_dim=ph_dim)
        self.router = DynamicContextRouter(
            num_sources=self.context.num_sources,
            num_levels=len(level_channels),
            ctx_dim=ctx_dim,
        )
        self.adapters = nn.ModuleList(AdaGN3D(ctx_dim, channels) for channels in level_channels)

        self.up3 = TrilinearUpsampleConv3D(bc * 8, bc * 4)
        self.gate3 = AttentionGate3D(bc * 4, bc * 4)
        self.dec3 = ResBlock3D(bc * 8, bc * 4)
        self.up2 = TrilinearUpsampleConv3D(bc * 4, bc * 2)
        self.gate2 = AttentionGate3D(bc * 2, bc * 2)
        self.dec2 = ResBlock3D(bc * 4, bc * 2)
        self.up1 = TrilinearUpsampleConv3D(bc * 2, bc)
        self.gate1 = AttentionGate3D(bc, bc)
        self.dec1 = ResBlock3D(bc * 2, bc)
        self.out_conv = nn.Conv3d(bc, out_channels, 1)

        self.rock_embedding_head = nn.Sequential(
            nn.Linear(bc * 8, rock_embedding_dim),
            nn.LayerNorm(rock_embedding_dim),
            nn.GELU(),
        )
        self.porosity_head = nn.Linear(rock_embedding_dim, 1)
        self.percolation_head = nn.Linear(rock_embedding_dim, 3)
        self.topology_head = nn.Linear(rock_embedding_dim, topology_dim) if topology_dim > 0 else None

    def _decode(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        bottleneck: torch.Tensor,
        mixed: torch.Tensor,
    ) -> torch.Tensor:
        bottleneck = self.adapters[0](bottleneck, mixed[:, 0, :])
        up3 = self.up3(bottleneck, e3.shape[-3:])
        skip3 = self.gate3(e3, up3)
        d3 = self.dec3(torch.cat([up3, skip3], dim=1))
        d3 = self.adapters[1](d3, mixed[:, 1, :])

        up2 = self.up2(d3, e2.shape[-3:])
        skip2 = self.gate2(e2, up2)
        d2 = self.dec2(torch.cat([up2, skip2], dim=1))
        d2 = self.adapters[2](d2, mixed[:, 2, :])

        up1 = self.up1(d2, e1.shape[-3:])
        skip1 = self.gate1(e1, up1)
        d1 = self.dec1(torch.cat([up1, skip1], dim=1))
        return self.adapters[3](d1, mixed[:, 3, :])

    def forward(
        self,
        x: torch.Tensor,
        ph_features: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        x, pad = pad_to_multiple_3d(x, multiple=8)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.bottleneck(self.pool3(e3))

        rock_embedding = self.rock_embedding_head(bottleneck.mean(dim=(2, 3, 4)))
        context = self.context(x, [e1, e2, e3, bottleneck], ph_features=ph_features)
        mixed, router_alpha, context_embedding = self.router(context)
        d1 = self._decode(e1, e2, e3, bottleneck, mixed)

        logits = unpad_3d(self.out_conv(d1), pad)
        decoder_embedding = unpad_3d(d1, pad)
        output = {
            "logits": logits,
            "rock_embedding": rock_embedding,
            "decoder_embedding": decoder_embedding,
            "context_embedding": context_embedding,
            "router_alpha": router_alpha,
            "porosity_logit": self.porosity_head(rock_embedding).squeeze(-1),
            "percolation_logits": self.percolation_head(rock_embedding),
        }
        if self.topology_head is not None:
            output["topology_pred"] = self.topology_head(rock_embedding)

        if return_dict:
            return output
        if self.return_embeddings:
            return logits, decoder_embedding
        return logits


class TopologyAdaptiveRoutedUNet3D(_AdaptiveRoutedUNet3DBase):
    """Adaptive 3D U-Net conditioned on raw-derived PH features."""

    def __init__(self, *, ph_dim: int = TOPOLOGY_FEATURE_DIM, topology_dim: int = TOPOLOGY_FEATURE_DIM, **kwargs):
        super().__init__(ph_dim=ph_dim, topology_dim=topology_dim, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        ph_features: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        if ph_features is None:
            raise ValueError("TopologyAdaptiveRoutedUNet3D requires ph_features")
        return super().forward(x, ph_features=ph_features, return_dict=return_dict)
