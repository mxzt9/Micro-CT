from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from utils.dependencies import require_gudhi
from utils.film_routing import FiLMRoutedUNet3D
from utils.network import openpnm_to_pore_network_data
from utils.pnm_gnn import DifferentiablePNMSolver, PoreNetworkPermeabilityModel, ThroatConductanceGNN


def test_film_unet_forward_backward_and_alpha_rows():
    model = FiLMRoutedUNet3D(in_channels=1, out_channels=1, base_channels=4, ctx_dim=16)
    x = torch.randn(1, 1, 16, 16, 16)

    logits, embeddings = model(x)
    loss = logits.mean() + embeddings.square().mean()
    loss.backward()

    alpha = model.router.alpha()
    assert logits.shape == (1, 1, 16, 16, 16)
    assert embeddings.shape == (1, 4, 16, 16, 16)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(alpha.shape[0]))
    assert model.router.alpha_logits.grad is not None
    assert torch.isfinite(model.router.alpha_logits.grad).all()


def test_pnm_solver_matches_series_and_parallel_graphs():
    solver = DifferentiablePNMSolver(mu=1.0, eps=1e-12)

    coords_series = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    edges_series = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    g_series = torch.tensor([2.0, 3.0], dtype=torch.float64)
    k_series = solver.solve_axis(g_series, edges_series, coords_series, axis=0, domain_length=1.0, cross_area=1.0)
    assert k_series.item() == pytest.approx(1.2)

    coords_parallel = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    edges_parallel = torch.tensor([[0, 1], [0, 1]], dtype=torch.long).t()
    g_parallel = torch.tensor([1.0, 2.0], dtype=torch.float64)
    k_parallel = solver.solve_axis(
        g_parallel,
        edges_parallel,
        coords_parallel,
        axis=0,
        domain_length=1.0,
        cross_area=1.0,
    )
    assert k_parallel.item() == pytest.approx(3.0)


def test_gnn_starts_from_hagen_poiseuille_baseline():
    torch.manual_seed(0)
    model = ThroatConductanceGNN(node_in=3, edge_in=2, hidden=8, layers=2)
    node_attr = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 2)
    log_g_hp = torch.tensor([-1.0, -2.0, -3.0])

    log_g = model(node_attr, edge_index, edge_attr, log_g_hp=log_g_hp)
    assert torch.allclose(log_g, log_g_hp)


def test_pore_network_data_conversion_shapes_without_ph():
    pn = {
        "pore.coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "throat.conns": np.array([[0, 1], [1, 2]]),
        "pore.diameter": np.array([0.2, 0.3, 0.2]),
        "throat.diameter": np.array([0.1, 0.1]),
        "throat.length": np.array([0.5, 0.5]),
        "pore.volume": np.array([1.0, 1.5, 1.0]),
    }

    data = openpnm_to_pore_network_data(pn, domain_size=(1.0, 1.0, 1.0), include_ph=False)

    assert data.coords.shape == (3, 3)
    assert data.edge_index.shape == (2, 2)
    assert data.node_attr.shape[0] == 3
    assert data.edge_attr.shape[0] == 2
    assert data.log_g_hp.shape == (2,)
    assert data.metadata["node_feature_dim"] == data.node_attr.shape[1]
    assert data.metadata["edge_feature_dim"] == data.edge_attr.shape[1]


def test_gudhi_requirement_has_clear_error_when_missing():
    if importlib.util.find_spec("gudhi") is not None:
        pytest.skip("gudhi is installed in this environment")

    with pytest.raises(ImportError, match="gudhi is required"):
        require_gudhi()


def test_graph_model_forward_on_converted_network_without_ph():
    pn = {
        "pore.coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "throat.conns": np.array([[0, 1], [1, 2]]),
        "pore.diameter": np.array([0.2, 0.3, 0.2]),
        "throat.diameter": np.array([0.1, 0.1]),
        "throat.length": np.array([0.5, 0.5]),
        "pore.volume": np.array([1.0, 1.5, 1.0]),
    }
    data = openpnm_to_pore_network_data(pn, domain_size=(1.0, 1.0, 1.0), include_ph=False)
    model = PoreNetworkPermeabilityModel(
        node_in=data.node_attr.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden=16,
        layers=2,
        mu=1.0,
    )

    k, log_g = model(data.node_attr, data.edge_index, data.edge_attr, data.coords, data.domain_size, data.log_g_hp)

    assert k.shape == (3,)
    assert log_g.shape == (2,)
    assert torch.isfinite(k).all()
    assert torch.isfinite(log_g).all()
