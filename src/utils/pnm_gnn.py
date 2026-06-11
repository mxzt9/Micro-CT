from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """PNM-решатель баланса масс с граничными условиями Дирихле по давлению.

    Использует float64 для сборки лапласиана и решения СЛАУ для численной стабильности.
    """

    def __init__(self, mu: float = 1.0e-3, eps: float = 1e-12, use_double: bool = True):
        super().__init__()
        self.mu = mu
        self.eps = eps
        self.use_double = use_double

    @staticmethod
    def _build_laplacian(g: torch.Tensor, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        i, j = edge_index[0], edge_index[1]
        laplacian = g.new_zeros(n, n)
        laplacian = laplacian.index_put((i, j), -g, accumulate=True)
        laplacian = laplacian.index_put((j, i), -g, accumulate=True)
        laplacian = laplacian.index_put((i, i), g, accumulate=True)
        laplacian = laplacian.index_put((j, j), g, accumulate=True)
        return laplacian

    @staticmethod
    def _compute_flow_residual(
        pressure: torch.Tensor,
        g: torch.Tensor,
        edge_index: torch.Tensor,
        n: int,
        interior_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Невязка баланса массы: для каждой поры |Σ(flow_e)|.

        ВАЖНО: невязка осмысленна только для внутренних (free) узлов.
        На граничных (inlet/outlet) узлах она по определению равна
        втекающему/вытекающему потоку и НЕ является нарушением физики —
        включать её в loss нельзя (это штрафовало бы сам поток).

        Args:
            interior_mask: булева маска внутренних узлов; если None — все узлы.

        Возвращает скаляр — средний модуль невязки на внутреннюю пору.
        """
        i_e, j_e = edge_index[0], edge_index[1]
        flow_e = g * (pressure[i_e] - pressure[j_e])  # [E]
        residual = scatter_sum(flow_e, i_e, n) - scatter_sum(flow_e, j_e, n)
        if interior_mask is not None:
            if interior_mask.sum() == 0:
                return g.new_tensor(0.0)
            residual = residual[interior_mask]
        return residual.abs().mean()

    def solve_axis(
        self,
        g: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        axis: int,
        domain_length: float,
        cross_area: float,
        frac: float = 0.05,
        return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            zero = g.new_tensor(0.0)
            if return_all:
                return zero, zero, g.new_tensor(0.0)
            return zero

        pressure = g.new_zeros(n)
        pressure[inlet] = 1.0
        pressure[outlet] = 0.0

        if free.sum() > 0:
            orig_dtype = g.dtype
            if self.use_double:
                g_solve = g.double()
                laplacian = self._build_laplacian(g_solve, edge_index, n)
                idx_free = free.nonzero(as_tuple=True)[0]
                idx_fixed = fixed.nonzero(as_tuple=True)[0]
                l_ff = laplacian[idx_free][:, idx_free]
                l_fb = laplacian[idx_free][:, idx_fixed]
                rhs = -l_fb @ pressure.to(g_solve.dtype)[idx_fixed]
                # ВАЖНО: регуляризация ОТНОСИТЕЛЬНАЯ (масштабируется на средний
                # диагональный элемент). Абсолютный eps=1e-12 для реальных СИ
                # проводимостей (g ~ 1e-17) был на ~5 порядков БОЛЬШЕ самих
                # элементов матрицы и полностью искажал решение (k завышалась
                # в разы). Для g ~ 1 (юнит-тесты) разницы нет.
                diag_scale = l_ff.diagonal().abs().mean().clamp_min(torch.finfo(g_solve.dtype).tiny)
                l_ff = l_ff + (self.eps * diag_scale) * torch.eye(l_ff.size(0), device=g.device, dtype=g_solve.dtype)
                p_free = torch.linalg.solve(l_ff, rhs)
                pressure = pressure.clone()
                pressure[idx_free] = p_free.to(orig_dtype)
            else:
                laplacian = self._build_laplacian(g, edge_index, n)
                idx_free = free.nonzero(as_tuple=True)[0]
                idx_fixed = fixed.nonzero(as_tuple=True)[0]
                l_ff = laplacian[idx_free][:, idx_free]
                l_fb = laplacian[idx_free][:, idx_fixed]
                rhs = -l_fb @ pressure[idx_fixed]
                diag_scale = l_ff.diagonal().abs().mean().clamp_min(torch.finfo(g.dtype).tiny)
                l_ff = l_ff + (self.eps * diag_scale) * torch.eye(l_ff.size(0), device=g.device, dtype=g.dtype)
                pressure = pressure.clone()
                pressure[idx_free] = torch.linalg.solve(l_ff, rhs)

        i_e, j_e = edge_index[0], edge_index[1]
        flow_e = g * (pressure[i_e] - pressure[j_e])
        inlet_f = inlet.to(dtype=g.dtype)
        flow_rate = (flow_e * inlet_f[i_e]).sum() - (flow_e * inlet_f[j_e]).sum()
        permeability = flow_rate * self.mu * domain_length / (cross_area + self.eps)
        perm = permeability.abs()

        if return_all:
            flow_residual = self._compute_flow_residual(pressure, g, edge_index, n, interior_mask=free)
            return perm, pressure.detach(), flow_residual

        return perm

    def forward(
        self,
        g: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        domain_size,
        return_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        lz, ly, lx = domain_size

        if return_all:
            kz, _, res_z = self.solve_axis(g, edge_index, coords, 0, lz, ly * lx, return_all=True)
            ky, _, res_y = self.solve_axis(g, edge_index, coords, 1, ly, lz * lx, return_all=True)
            kx, _, res_x = self.solve_axis(g, edge_index, coords, 2, lx, lz * ly, return_all=True)
            k = torch.stack([kx, ky, kz])
            flow_residual = (res_z + res_y + res_x) / 3.0
            return k, flow_residual

        kz = self.solve_axis(g, edge_index, coords, 0, lz, ly * lx)
        ky = self.solve_axis(g, edge_index, coords, 1, ly, lz * lx)
        kx = self.solve_axis(g, edge_index, coords, 2, lx, lz * ly)
        return torch.stack([kx, ky, kz])


class PoreNetworkPermeabilityModel(nn.Module):
    """GNN + дифференцируемый PNM-решатель для предсказания проницаемости.

    Особенности:
    - GNN предсказывает поправку к Hagen-Poiseuille baseline
    - Дифференцируемый решатель Стокса с float64 для стабильности
    - Опциональный физический auxiliary loss (массовая невязка)
    """

    def __init__(self, node_in: int, edge_in: int, hidden: int = 64, layers: int = 3, mu: float = 1.0e-3):
        super().__init__()
        self.gnn = ThroatConductanceGNN(node_in, edge_in, hidden, layers)
        self.solver = DifferentiablePNMSolver(mu=mu)

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coords: torch.Tensor,
        domain_size,
        log_g_hp: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: GNN → проводимости → PNM → проницаемость.

        Returns:
            k: [3] тензор (kx, ky, kz)
            log_g: [E] тензор логарифмов проводимости
        """
        log_g = self.gnn(node_attr, edge_index, edge_attr, log_g_hp)
        g = torch.exp(log_g)
        k = self.solver(g, edge_index, coords, domain_size)
        return k, log_g

    def forward_with_physics_loss(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coords: torch.Tensor,
        domain_size,
        log_g_hp: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass + физическая невязка (auxiliary loss).

        Returns:
            k: [3] тензор (kx, ky, kz)
            log_g: [E] тензор логарифмов проводимости
            flow_residual: скаляр — средняя массовая невязка на пору
        """
        log_g = self.gnn(node_attr, edge_index, edge_attr, log_g_hp)
        g = torch.exp(log_g)
        k, flow_residual = self.solver(g, edge_index, coords, domain_size, return_all=True)
        return k, log_g, flow_residual