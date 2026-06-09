from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss, bce_loss, dice_loss


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    preds = (torch.sigmoid(logits) >= threshold).float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean()


def embedding_consistency_loss(embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Cosine consistency loss for multi-noise or multi-scale rock embeddings."""

    if len(embeddings) < 2:
        raise ValueError("embedding_consistency_loss needs at least two embeddings")
    losses = []
    for left, right in zip(embeddings[:-1], embeddings[1:]):
        losses.append(1.0 - F.cosine_similarity(left, right, dim=1).mean())
    return torch.stack(losses).mean()


def auxiliary_physics_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    porosity_target: torch.Tensor | None = None,
    percolation_target: torch.Tensor | None = None,
    porosity_weight: float = 0.05,
    percolation_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss for porosity and percolation auxiliary heads."""

    loss = targets.new_tensor(0.0)
    parts: dict[str, torch.Tensor] = {}

    if porosity_target is None:
        porosity_target = targets.float().mean(dim=(1, 2, 3, 4))
    porosity_loss = F.mse_loss(torch.sigmoid(outputs["porosity_logit"]), porosity_target.float())
    parts["porosity_loss"] = porosity_loss
    loss = loss + porosity_weight * porosity_loss

    if percolation_target is not None and "percolation_logits" in outputs:
        percolation_loss = F.binary_cross_entropy_with_logits(outputs["percolation_logits"], percolation_target.float())
        parts["percolation_loss"] = percolation_loss
        loss = loss + percolation_weight * percolation_loss

    return loss, parts


def topology_prediction_loss(
    outputs: dict[str, torch.Tensor],
    topology_target: torch.Tensor | None,
    *,
    topology_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Auxiliary loss for PH summary prediction.

    The dataset stores raw PH summaries. Training compares them after log1p
    compression so component counts do not dominate segmentation losses.
    """

    if topology_target is None or "topology_pred" not in outputs or topology_weight <= 0:
        device_tensor = next(iter(outputs.values()))
        return device_tensor.new_tensor(0.0), {}

    target = torch.log1p(topology_target.float().clamp_min(0.0))
    pred = outputs["topology_pred"]
    loss = F.smooth_l1_loss(pred, target)
    return topology_weight * loss, {"topology_loss": loss}
