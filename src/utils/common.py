from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct3D(nn.Module):
    """Two 3D convolutions with GroupNorm for stable small-batch training."""

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
    """Compatibility alias used by notebooks and FiLM modules."""


def pad_to_multiple_3d(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int]]:
    """Pad [B,C,D,H,W] so D/H/W are divisible by multiple."""

    if x.ndim != 5:
        raise ValueError("pad_to_multiple_3d expects a [B,C,D,H,W] tensor")
    if multiple <= 0:
        raise ValueError("multiple must be positive")

    d, h, w = x.shape[-3:]
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = (0, pw, 0, ph, 0, pd)
    if pd == 0 and ph == 0 and pw == 0:
        return x, pad
    return F.pad(x, pad), pad


def unpad_3d(x: torch.Tensor, pad: tuple[int, int, int, int, int, int]) -> torch.Tensor:
    """Remove padding returned by pad_to_multiple_3d."""

    if x.ndim != 5:
        raise ValueError("unpad_3d expects a [B,C,D,H,W] tensor")
    _, pw, _, ph, _, pd = pad
    d_end = x.shape[-3] - pd if pd > 0 else x.shape[-3]
    h_end = x.shape[-2] - ph if ph > 0 else x.shape[-2]
    w_end = x.shape[-1] - pw if pw > 0 else x.shape[-1]
    return x[..., :d_end, :h_end, :w_end]


def match_size(x: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
    """Center-crop or pad x to target D/H/W."""

    if x.ndim != 5:
        raise ValueError("match_size expects a [B,C,D,H,W] tensor")

    for axis, target in enumerate(target_size, start=-3):
        current = x.shape[axis]
        if current > target:
            start = (current - target) // 2
            end = start + target
            slices = [slice(None)] * x.ndim
            slices[axis] = slice(start, end)
            x = x[tuple(slices)]

    d, h, w = x.shape[-3:]
    td, th, tw = target_size
    pd, ph, pw = max(td - d, 0), max(th - h, 0), max(tw - w, 0)
    if pd or ph or pw:
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2)
        x = F.pad(x, pad)
    return x


def safe_concat(skip: torch.Tensor, upsampled: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Concatenate decoder and skip tensors after matching spatial size to skip."""

    return torch.cat([match_size(upsampled, skip.shape[-3:]), skip], dim=dim)


def _blend_window(size: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axes = []
    for n in size:
        if n <= 1:
            axes.append(torch.ones(n, device=device, dtype=dtype))
        else:
            axes.append(torch.hann_window(n, periodic=False, device=device, dtype=dtype).clamp_min(1.0e-3))
    return axes[0].view(-1, 1, 1) * axes[1].view(1, -1, 1) * axes[2].view(1, 1, -1)


@torch.no_grad()
def sliding_window_inference_3d(
    model: nn.Module,
    x: torch.Tensor,
    window_size: int | tuple[int, int, int] = 128,
    overlap: float = 0.5,
    *,
    ph_features: torch.Tensor | None = None,
    output_key: str = "logits",
) -> torch.Tensor:
    """Run tiled inference for large [B,C,D,H,W] volumes and blend logits."""

    if x.ndim != 5:
        raise ValueError("sliding_window_inference_3d expects a [B,C,D,H,W] tensor")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    if isinstance(window_size, int):
        wd = wh = ww = int(window_size)
    else:
        wd, wh, ww = (int(v) for v in window_size)

    _, _, d, h, w = x.shape
    wd, wh, ww = min(wd, d), min(wh, h), min(ww, w)
    strides = [max(int(round(size * (1.0 - overlap))), 1) for size in (wd, wh, ww)]

    def starts(total: int, window: int, stride: int) -> list[int]:
        values = list(range(0, max(total - window, 0) + 1, stride))
        last = total - window
        if values[-1] != last:
            values.append(last)
        return values

    out: torch.Tensor | None = None
    weight_sum: torch.Tensor | None = None
    blend = _blend_window((wd, wh, ww), x.device, x.dtype).view(1, 1, wd, wh, ww)

    for z in starts(d, wd, strides[0]):
        for y in starts(h, wh, strides[1]):
            for x0 in starts(w, ww, strides[2]):
                patch = x[..., z : z + wd, y : y + wh, x0 : x0 + ww]
                pred = model(patch, ph_features=ph_features)
                if isinstance(pred, dict):
                    pred = pred[output_key]
                elif isinstance(pred, tuple):
                    pred = pred[0]
                if out is None:
                    out = torch.zeros(x.shape[0], pred.shape[1], d, h, w, device=pred.device, dtype=pred.dtype)
                    weight_sum = torch.zeros_like(out)
                    blend = blend.to(device=pred.device, dtype=pred.dtype)
                out[..., z : z + wd, y : y + wh, x0 : x0 + ww] += pred * blend
                weight_sum[..., z : z + wd, y : y + wh, x0 : x0 + ww] += blend

    assert out is not None and weight_sum is not None
    return out / weight_sum.clamp_min(1.0e-8)


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
        x, pad = pad_to_multiple_3d(x, multiple=8)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(safe_concat(e3, self.up3(b)))
        d2 = self.dec2(safe_concat(e2, self.up2(d3)))
        d1 = self.dec1(safe_concat(e1, self.up1(d2)))
        return unpad_3d(self.out_conv(d1), pad)
