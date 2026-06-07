from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add(0, idx, src)


class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, node_dim),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        i, j = edge_index[0], edge_index[1]
        m_ij = self.msg(torch.cat([h[i], h[j], edge_attr], dim=-1))
        m_ji = self.msg(torch.cat([h[j], h[i], edge_attr], dim=-1))
        agg = scatter_sum(m_ij, j, h.size(0)) + scatter_sum(m_ji, i, h.size(0))
        h_new = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class ThroatConductanceGNN(nn.Module):
    """Предсказывает log проводимости горл как базу Хагена-Пуазейля плюс поправку."""

    def __init__(self, node_in: int, edge_in: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)
        self.mp = nn.ModuleList([MessagePassingLayer(hidden, hidden, hidden) for _ in range(layers)])
        self.edge_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        log_g_hp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.node_enc(node_attr)
        ea = self.edge_enc(edge_attr)
        for layer in self.mp:
            h = layer(h, edge_index, ea)

        i, j = edge_index[0], edge_index[1]
        delta = self.edge_head(torch.cat([h[i], h[j], ea], dim=-1)).squeeze(-1)
        if log_g_hp is None:
            log_g_hp = edge_attr[:, 0]
        return log_g_hp + delta


class DifferentiablePNMSolver(nn.Module):
    """PNM-решатель баланса масс с граничными условиями Дирихле по давлению."""

    def __init__(self, mu: float = 1.0e-3, eps: float = 1e-12):
        super().__init__()
        self.mu = mu
        self.eps = eps

    @staticmethod
    def _build_laplacian(g: torch.Tensor, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        i, j = edge_index[0], edge_index[1]
        laplacian = g.new_zeros(n, n)
        laplacian = laplacian.index_put((i, j), -g, accumulate=True)
        laplacian = laplacian.index_put((j, i), -g, accumulate=True)
        laplacian = laplacian.index_put((i, i), g, accumulate=True)
        laplacian = laplacian.index_put((j, j), g, accumulate=True)
        return laplacian

    def solve_axis(
        self,
        g: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        axis: int,
        domain_length: float,
        cross_area: float,
        frac: float = 0.05,
    ) -> torch.Tensor:
        n = coords.size(0)
        ca = coords[:, axis]
        span = ca.max() - ca.min()
        lo = ca.min() + frac * span
        hi = ca.max() - frac * span
        inlet = ca <= lo
        outlet = ca >= hi
        fixed = inlet | outlet
        free = ~fixed
        if inlet.sum() == 0 or outlet.sum() == 0:
            return g.new_tensor(0.0)

        pressure = g.new_zeros(n)
        pressure[inlet] = 1.0
        pressure[outlet] = 0.0

        if free.sum() > 0:
            laplacian = self._build_laplacian(g, edge_index, n)
            idx_free = free.nonzero(as_tuple=True)[0]
            idx_fixed = fixed.nonzero(as_tuple=True)[0]
            l_ff = laplacian[idx_free][:, idx_free]
            l_fb = laplacian[idx_free][:, idx_fixed]
            rhs = -l_fb @ pressure[idx_fixed]
            l_ff = l_ff + self.eps * torch.eye(l_ff.size(0), device=g.device, dtype=g.dtype)
            pressure = pressure.clone()
            pressure[idx_free] = torch.linalg.solve(l_ff, rhs)

        i_e, j_e = edge_index[0], edge_index[1]
        flow_e = g * (pressure[i_e] - pressure[j_e])
        inlet_f = inlet.to(dtype=g.dtype)
        flow_rate = (flow_e * inlet_f[i_e]).sum() - (flow_e * inlet_f[j_e]).sum()
        permeability = flow_rate * self.mu * domain_length / (cross_area + self.eps)
        return permeability.abs()

    def forward(
        self,
        g: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        domain_size,
    ) -> torch.Tensor:
        lz, ly, lx = domain_size
        kz = self.solve_axis(g, edge_index, coords, 0, lz, ly * lx)
        ky = self.solve_axis(g, edge_index, coords, 1, ly, lz * lx)
        kx = self.solve_axis(g, edge_index, coords, 2, lx, lz * ly)
        return torch.stack([kx, ky, kz])


class PoreNetworkPermeabilityModel(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 64, layers: int = 3, mu: float = 1.0e-3):
        super().__init__()
        self.gnn = ThroatConductanceGNN(node_in, edge_in, hidden, layers)
        self.solver = DifferentiablePNMSolver(mu=mu)

    def forward(self, node_attr, edge_index, edge_attr, coords, domain_size, log_g_hp=None):
        log_g = self.gnn(node_attr, edge_index, edge_attr, log_g_hp)
        g = torch.exp(log_g)
        k = self.solver(g, edge_index, coords, domain_size)
        return k, log_g
