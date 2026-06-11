"""Метрики качества сегментации пористой среды (numpy, по одному объёму).

Воксельные метрики (Dice) почти не чувствуют топологические ошибки:
разрыв горла в 3 вокселя меняет Dice на ~0.001, а связную пористость и
любые производные свойства — в разы. Этот модуль дополняет Dice метриками,
которые ловят именно такие ошибки:

- ``cl_dice_score``        — связность скелета (hard-аналог SoftClDiceLoss)
- ``betti_numbers``        — b0 (компоненты), b1 (туннели), b2 (полости)
- ``connected_porosity``   — доля пор в перколирующих кластерах
- ``percolation_vector``   — перколяция по осям z/y/x
- ``segmentation_quality_report`` — всё сразу в одном dict

Все функции принимают бинарные 3D-массивы (D, H, W).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize


_STRUCT_26 = np.ones((3, 3, 3), dtype=bool)
_STRUCT_6 = ndimage.generate_binary_structure(3, 1)


def _as_bool_volume(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 3:
        raise ValueError(f"ожидался 3D-объём, получено shape={mask.shape}")
    return mask.astype(bool)


def cl_dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Hard clDice: топологическая точность/полнота по скелетам масок."""
    pred = _as_bool_volume(pred)
    target = _as_bool_volume(target)
    if not pred.any() and not target.any():
        return 1.0
    skel_pred = skeletonize(pred)
    skel_true = skeletonize(target)
    tprec = (np.logical_and(skel_pred, target).sum() + smooth) / (skel_pred.sum() + smooth)
    tsens = (np.logical_and(skel_true, pred).sum() + smooth) / (skel_true.sum() + smooth)
    return float(2.0 * tprec * tsens / (tprec + tsens))


def betti_numbers(mask: np.ndarray) -> tuple[int, int, int]:
    """Числа Бетти бинарного 3D-объёма: (b0, b1, b2).

    b0 — связные компоненты пор (26-связность),
    b2 — полости: компоненты фона (6-связность), не касающиеся границы,
    b1 — туннели, из эйлеровой характеристики: chi = b0 - b1 + b2.

    Использует дуальную пару связностей (26 для пор / 6 для фона) —
    стандарт для цифровой топологии.
    """
    mask = _as_bool_volume(mask)
    if not mask.any():
        return 0, 0, 0

    _, b0 = ndimage.label(mask, structure=_STRUCT_26)

    bg_labels, bg_num = ndimage.label(~mask, structure=_STRUCT_6)
    border = np.zeros_like(mask, dtype=bool)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    touching = np.unique(bg_labels[border])
    touching = set(touching[touching > 0].tolist())
    b2 = int(bg_num - len(touching))

    chi = int(measure.euler_number(mask, connectivity=3))
    b1 = max(b0 + b2 - chi, 0)
    return int(b0), int(b1), int(b2)


def percolation_vector(mask: np.ndarray) -> np.ndarray:
    """[percolates_z, percolates_y, percolates_x] — есть ли сквозной кластер."""
    mask = _as_bool_volume(mask)
    result = np.zeros(3, dtype=np.float32)
    if not mask.any():
        return result
    labels, num = ndimage.label(mask, structure=_STRUCT_6)
    if num == 0:
        return result
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = 0
        high[axis] = -1
        low_labels = np.unique(labels[tuple(low)])
        high_labels = np.unique(labels[tuple(high)])
        if np.intersect1d(low_labels[low_labels > 0], high_labels[high_labels > 0]).size:
            result[axis] = 1.0
    return result


def connected_porosity(mask: np.ndarray) -> float:
    """Доля объёма в порах, входящих хотя бы в один перколирующий кластер.

    Именно эта величина (а не общая пористость) определяет транспортные
    свойства: изолированные поры не проводят.
    """
    mask = _as_bool_volume(mask)
    if not mask.any():
        return 0.0
    labels, num = ndimage.label(mask, structure=_STRUCT_6)
    if num == 0:
        return 0.0
    percolating: set[int] = set()
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = 0
        high[axis] = -1
        low_labels = np.unique(labels[tuple(low)])
        high_labels = np.unique(labels[tuple(high)])
        common = np.intersect1d(low_labels[low_labels > 0], high_labels[high_labels > 0])
        percolating.update(common.tolist())
    if not percolating:
        return 0.0
    volume = np.isin(labels, sorted(percolating)).sum()
    return float(volume) / float(mask.size)


def segmentation_quality_report(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Полный отчёт по одному объёму: воксельные + топологические метрики."""
    pred = _as_bool_volume(pred)
    target = _as_bool_volume(target)

    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    dice = float((2.0 * intersection + 1e-6) / (denom + 1e-6))

    b0_p, b1_p, b2_p = betti_numbers(pred)
    b0_t, b1_t, b2_t = betti_numbers(target)

    perc_p = percolation_vector(pred)
    perc_t = percolation_vector(target)

    por_p = float(pred.mean())
    por_t = float(target.mean())
    cpor_p = connected_porosity(pred)
    cpor_t = connected_porosity(target)

    return {
        "dice": dice,
        "cl_dice": cl_dice_score(pred, target),
        "porosity_pred": por_p,
        "porosity_true": por_t,
        "porosity_abs_err": abs(por_p - por_t),
        "connected_porosity_pred": cpor_p,
        "connected_porosity_true": cpor_t,
        "connected_porosity_abs_err": abs(cpor_p - cpor_t),
        "betti0_pred": float(b0_p),
        "betti0_true": float(b0_t),
        "betti0_abs_err": float(abs(b0_p - b0_t)),
        "betti1_pred": float(b1_p),
        "betti1_true": float(b1_t),
        "betti1_abs_err": float(abs(b1_p - b1_t)),
        "betti2_pred": float(b2_p),
        "betti2_true": float(b2_t),
        "betti2_abs_err": float(abs(b2_p - b2_t)),
        "percolation_match": float((perc_p == perc_t).mean()),
    }
