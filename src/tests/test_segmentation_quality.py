"""Тесты clDice-лосса и топологических метрик качества сегментации."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.losses import SoftClDiceLoss, soft_skeletonize_3d
from utils.seg_metrics import (
    betti_numbers,
    cl_dice_score,
    connected_porosity,
    percolation_vector,
    segmentation_quality_report,
)


def _tube_mask(size: int = 32, radius: int = 3) -> np.ndarray:
    """Сквозной цилиндрический канал вдоль оси z."""
    zz, yy, xx = np.meshgrid(
        np.arange(size), np.arange(size), np.arange(size), indexing="ij"
    )
    center = size // 2
    return ((yy - center) ** 2 + (xx - center) ** 2) <= radius**2


def _logits_from_mask(mask: np.ndarray, scale: float = 12.0) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32) * 2.0 - 1.0).mul(scale)[None, None]


# ── SoftClDiceLoss ────────────────────────────────────────────────────────────


def test_soft_cldice_zero_for_perfect_prediction():
    mask = _tube_mask()
    logits = _logits_from_mask(mask)
    target = torch.from_numpy(mask.astype(np.float32))[None, None]
    loss = SoftClDiceLoss(num_iters=8)(logits, target)
    assert float(loss) < 0.05


def test_soft_cldice_penalizes_broken_throat():
    mask = _tube_mask()
    broken = mask.copy()
    broken[14:17] = False  # разрыв канала в 3 вокселя
    target = torch.from_numpy(mask.astype(np.float32))[None, None]

    loss_intact = SoftClDiceLoss(num_iters=8)(_logits_from_mask(mask), target)
    loss_broken = SoftClDiceLoss(num_iters=8)(_logits_from_mask(broken), target)

    # Воксельный Dice от такого разрыва почти не меняется, clDice — должен.
    assert float(loss_broken) > 0.02
    assert float(loss_broken) > 100.0 * float(loss_intact)


def test_soft_cldice_gradient_flows():
    mask = _tube_mask(size=24)
    logits = _logits_from_mask(mask, scale=2.0).requires_grad_(True)
    target = torch.from_numpy(mask.astype(np.float32))[None, None]
    loss = SoftClDiceLoss(num_iters=5)(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_soft_skeleton_thinner_than_input():
    mask = _tube_mask()
    img = torch.from_numpy(mask.astype(np.float32))[None, None]
    skel = soft_skeletonize_3d(img, num_iters=8)
    assert float(skel.sum()) < float(img.sum())
    assert float(skel.max()) <= 1.0 + 1e-5


# ── seg_metrics ───────────────────────────────────────────────────────────────


def test_cl_dice_score_perfect_and_empty():
    mask = _tube_mask()
    assert cl_dice_score(mask, mask) == pytest.approx(1.0, abs=1e-4)
    empty = np.zeros_like(mask)
    assert cl_dice_score(empty, empty) == 1.0


def test_betti_numbers_solid_cube():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    assert betti_numbers(mask) == (1, 0, 0)


def test_betti_numbers_ring_has_tunnel():
    # Квадратное «кольцо» в одном слое: b0=1, b1=1, b2=0
    mask = np.zeros((8, 16, 16), dtype=bool)
    mask[3:5, 4:12, 4:12] = True
    mask[3:5, 6:10, 6:10] = False
    assert betti_numbers(mask) == (1, 1, 0)


def test_betti_numbers_cavity():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    mask[7:9, 7:9, 7:9] = False  # полость внутри
    assert betti_numbers(mask) == (1, 0, 1)


def test_percolation_and_connected_porosity():
    mask = _tube_mask()
    perc = percolation_vector(mask)
    assert perc[0] == 1.0 and perc[1] == 0.0 and perc[2] == 0.0
    # весь канал перколирует по z → связная пористость == общая
    assert connected_porosity(mask) == pytest.approx(mask.mean(), rel=1e-6)

    blocked = mask.copy()
    blocked[16] = False
    assert percolation_vector(blocked)[0] == 0.0
    assert connected_porosity(blocked) == 0.0

    # изолированная пора не добавляет связной пористости
    with_isolated = mask.copy()
    with_isolated[2:4, 2:4, 2:4] = True
    assert connected_porosity(with_isolated) == pytest.approx(mask.mean(), rel=1e-6)


def test_segmentation_quality_report_keys_and_values():
    mask = _tube_mask()
    broken = mask.copy()
    broken[14:17] = False
    report = segmentation_quality_report(broken, mask)

    expected_keys = {
        "dice", "cl_dice",
        "porosity_pred", "porosity_true", "porosity_abs_err",
        "connected_porosity_pred", "connected_porosity_true", "connected_porosity_abs_err",
        "betti0_pred", "betti0_true", "betti0_abs_err",
        "betti1_pred", "betti1_true", "betti1_abs_err",
        "betti2_pred", "betti2_true", "betti2_abs_err",
        "percolation_match",
    }
    assert expected_keys.issubset(report.keys())

    # разрыв канала: dice высокий, но топология поймана
    assert report["dice"] > 0.9
    assert report["betti0_abs_err"] == 1.0  # 2 компоненты вместо 1
    assert report["percolation_match"] < 1.0  # перколяция по z потеряна
    assert report["connected_porosity_abs_err"] > 0.5 * mask.mean()
