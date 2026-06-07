from __future__ import annotations

import numpy as np
import pytest

from utils.network import extract_porespy_openpnm_network, openpnm_to_pore_network_data


def test_porespy_openpnm_synthetic_network_smoke():
    pytest.importorskip("porespy")
    pytest.importorskip("openpnm")

    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[10:14, 10:14, :] = True
    mask[8:16, 8:16, 4:8] = True
    mask[8:16, 8:16, 16:20] = True

    try:
        pn = extract_porespy_openpnm_network(mask, voxel_size=1.0, sigma=0.2, r_max=3)
    except Exception as exc:  # pragma: no cover - library-version smoke guard
        pytest.skip(f"PoreSpy/OpenPNM could not extract this synthetic smoke network: {exc}")

    data = openpnm_to_pore_network_data(pn, domain_size=(24.0, 24.0, 24.0), include_ph=False)

    assert data.coords.ndim == 2 and data.coords.shape[1] == 3
    assert data.edge_index.ndim == 2 and data.edge_index.shape[0] == 2
    assert data.node_attr.shape[0] == data.coords.shape[0]
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
