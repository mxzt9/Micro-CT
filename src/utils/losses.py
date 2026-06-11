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


# ──────────────────────────────────────────────────────────────────────────────
# clDice: топологический лосс на связность скелета (Shit et al., CVPR 2021)
# ──────────────────────────────────────────────────────────────────────────────


def _soft_erode_3d(img: torch.Tensor) -> torch.Tensor:
    """Мягкая эрозия: min-пулинг тремя направленными ядрами (3D, NCDHW)."""
    p1 = -F.max_pool3d(-img, (3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate_3d(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, (3, 3, 3), stride=1, padding=1)


def _soft_open_3d(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate_3d(_soft_erode_3d(img))


def _skel_init(img: torch.Tensor) -> torch.Tensor:
    return F.relu(img - _soft_open_3d(img))


def _skel_iter(img: torch.Tensor, skel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    img = _soft_erode_3d(img)
    delta = F.relu(img - _soft_open_3d(img))
    return img, skel + F.relu(delta - skel * delta)


def soft_skeletonize_3d(
    img: torch.Tensor,
    num_iters: int = 10,
    use_checkpoint: bool | None = None,
) -> torch.Tensor:
    """Дифференцируемый скелет: итеративное вычитание «открытий».

    num_iters должен быть не меньше максимального радиуса структур (в вокселях),
    иначе толстые поры не успеют «истончиться» до скелета. Для пор радиусом
    до ~10 вокселей хватает 10 итераций.

    Память: без чекпоинтинга autograd хранит ~12 полноразмерных тензоров на
    итерацию (num_iters=10 на кубе 192³ fp32 ≈ 3.4 ГБ только на скелет — OOM на
    8 ГБ GPU). С use_checkpoint граф каждой итерации пересчитывается в backward,
    в памяти живут лишь границы итераций (img, skel) — ~6x меньше при +~30%
    времени на сами пулинги (доля в общем шаге обучения мала).
    По умолчанию чекпоинтинг включается автоматически, когда вход требует
    градиент; для таргетов (без градиента) граф и так не строится.
    """
    if use_checkpoint is None:
        use_checkpoint = img.requires_grad and torch.is_grad_enabled()
    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint

        skel = checkpoint(_skel_init, img, use_reentrant=False)
        for _ in range(num_iters):
            img, skel = checkpoint(_skel_iter, img, skel, use_reentrant=False)
    else:
        skel = _skel_init(img)
        for _ in range(num_iters):
            img, skel = _skel_iter(img, skel)
    return skel


class SoftClDiceLoss(nn.Module):
    """clDice-лосс: штрафует разрывы и ложные перемычки скелета пор.

    Дополняет воксельный Dice: разрыв горла в 3 вокселя почти не меняет Dice,
    но рушит связность (и любые производные свойства). clDice сравнивает
    мягкие скелеты предсказания и таргета:
      tprec = |skel(pred) ∩ target| / |skel(pred)|   (нет ложных ветвей)
      tsens = |skel(target) ∩ pred| / |skel(target)| (скелет таргета покрыт)
      loss  = 1 − 2·tprec·tsens / (tprec + tsens)

    Скелетизация — на float32 (под AMP min/max-пулинги в fp16 теряют точность).
    """

    def __init__(self, num_iters: int = 10, smooth: float = 1e-6, downsample: int = 1):
        """downsample > 1 — считать clDice на уменьшенном объёме (avg_pool3d).

        Фактор 2 экономит ~8x памяти, но скелет считается на половинном
        разрешении: горла тоньше ~2 вокселей размываются. Использовать как
        крайнюю меру для больших кубов на малой видеопамяти; по умолчанию
        выключено.
        """
        super().__init__()
        self.num_iters = num_iters
        self.smooth = smooth
        self.downsample = int(downsample)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        if self.downsample > 1:
            probs = F.avg_pool3d(probs, self.downsample)
            targets = F.avg_pool3d(targets, self.downsample)
        skel_pred = soft_skeletonize_3d(probs, self.num_iters)
        with torch.no_grad():
            skel_true = soft_skeletonize_3d(targets, self.num_iters)
        dims = tuple(range(1, probs.dim()))
        tprec = ((skel_pred * targets).sum(dim=dims) + self.smooth) / (
            skel_pred.sum(dim=dims) + self.smooth
        )
        tsens = ((skel_true * probs).sum(dim=dims) + self.smooth) / (
            skel_true.sum(dim=dims) + self.smooth
        )
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens)
        return 1.0 - cl_dice.mean()