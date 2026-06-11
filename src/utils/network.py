"""Извлечение поровой сети из бинарной маски (PoreSpy SNOW2 → OpenPNM).

Используется только для визуализации и морфологического анализа
(graph-режим в ``scripts/visualize.py``). Расчёт проницаемости и весь
GNN/PNM-контур из проекта удалены — фокус на качестве сегментации.

porespy и openpnm — опциональные зависимости: импортируются лениво,
сегментационный пайплайн без них работает полностью.
"""

from __future__ import annotations

import numpy as np


def extract_porespy_openpnm_network(
    pore_mask: np.ndarray,
    voxel_size: float = 1.0,
    sigma: float = 0.4,
    r_max: int = 4,
):
    import openpnm as op
    import porespy as ps

    pore_mask = np.asarray(pore_mask).astype(bool)
    snow = ps.networks.snow2(
        phases=pore_mask.astype(int),
        voxel_size=voxel_size,
        sigma=sigma,
        r_max=r_max,
    )
    pn = op.io.network_from_porespy(snow.network)
    cleanup_openpnm_network(pn)
    return pn


def cleanup_openpnm_network(pn):
    import openpnm as op

    health = op.utils.check_network_health(pn)
    disconnected = health.get("disconnected_pores", [])
    if len(disconnected) > 0:
        op.topotools.trim(network=pn, pores=disconnected)
    return pn
