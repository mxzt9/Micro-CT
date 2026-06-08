from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .dependencies import require_gudhi


@dataclass
class PoreNetworkData:
    coords: torch.Tensor
    edge_index: torch.Tensor
    node_attr: torch.Tensor
    edge_attr: torch.Tensor
    log_g_hp: torch.Tensor
    domain_size: tuple[float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "PoreNetworkData":
        return PoreNetworkData(
            coords=self.coords.to(device),
            edge_index=self.edge_index.to(device),
            node_attr=self.node_attr.to(device),
            edge_attr=self.edge_attr.to(device),
            log_g_hp=self.log_g_hp.to(device),
            domain_size=self.domain_size,
            metadata=self.metadata,
        )


def hagen_poiseuille_log_conductance(
    radius: np.ndarray,
    length: np.ndarray,
    mu: float = 1.0e-3,
    eps: float = 1.0e-30,
) -> np.ndarray:
    radius = np.maximum(np.asarray(radius, dtype=np.float64), eps)
    length = np.maximum(np.asarray(length, dtype=np.float64), eps)
    g = np.pi * radius**4 / (8.0 * mu * length)
    return np.log(np.maximum(g, eps))


def persistent_homology_summary(
    coords: np.ndarray,
    max_points: int = 512,
    max_dimension: int = 1,
) -> np.ndarray:
    gudhi = require_gudhi()
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.zeros(6, dtype=np.float32)

    if coords.shape[0] > max_points:
        idx = np.linspace(0, coords.shape[0] - 1, max_points).astype(int)
        coords = coords[idx]

    rips = gudhi.RipsComplex(points=coords)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    persistence = simplex_tree.persistence()

    features: list[float] = []
    for dim in (0, 1):
        lifetimes = []
        for p_dim, (birth, death) in persistence:
            if p_dim != dim or not np.isfinite(death):
                continue
            lifetimes.append(float(max(death - birth, 0.0)))

        if lifetimes:
            arr = np.asarray(lifetimes, dtype=np.float64)
            features.extend([float(len(arr)), float(arr.sum()), float(arr.max())])
        else:
            features.extend([0.0, 0.0, 0.0])

    return np.asarray(features, dtype=np.float32)


def _connected_components(num_nodes: int, conns: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    adjacency = [[] for _ in range(num_nodes)]
    for edge_id, (i, j) in enumerate(conns):
        adjacency[int(i)].append((int(j), edge_id))
        adjacency[int(j)].append((int(i), edge_id))

    comp_id = np.full(num_nodes, -1, dtype=np.int64)
    comp_sizes: list[int] = []
    components: list[list[int]] = []
    for start in range(num_nodes):
        if comp_id[start] >= 0:
            continue
        cid = len(comp_sizes)
        stack = [start]
        comp_id[start] = cid
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for nbr, _ in adjacency[node]:
                if comp_id[nbr] < 0:
                    comp_id[nbr] = cid
                    stack.append(nbr)
        comp_sizes.append(len(nodes))
        components.append(nodes)
    return comp_id, np.asarray(comp_sizes, dtype=np.float64), components


def _bridge_edges(num_nodes: int, conns: np.ndarray) -> np.ndarray:
    adjacency = [[] for _ in range(num_nodes)]
    for edge_id, (i, j) in enumerate(conns):
        adjacency[int(i)].append((int(j), edge_id))
        adjacency[int(j)].append((int(i), edge_id))

    timer = 0
    tin = np.full(num_nodes, -1, dtype=np.int64)
    low = np.full(num_nodes, -1, dtype=np.int64)
    bridges = np.zeros(conns.shape[0], dtype=np.float32)

    def dfs(v: int, parent_edge: int = -1) -> None:
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        for to, edge_id in adjacency[v]:
            if edge_id == parent_edge:
                continue
            if tin[to] >= 0:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, edge_id)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges[edge_id] = 1.0

    for node in range(num_nodes):
        if tin[node] < 0:
            dfs(node)
    return bridges


def graph_topology_features(
    coords: np.ndarray,
    conns: np.ndarray,
    throat_radius: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute local graph-topology features for pores and throats."""

    coords = np.asarray(coords, dtype=np.float64)
    conns = np.asarray(conns, dtype=np.int64)
    num_pores = coords.shape[0]
    num_throats = conns.shape[0]
    if num_pores == 0:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((num_throats, 4), dtype=np.float32), {}

    comp_id, comp_sizes, _ = _connected_components(num_pores, conns)
    comp_size_per_node = comp_sizes[comp_id] if len(comp_sizes) else np.ones(num_pores)
    comp_size_ratio = comp_size_per_node / max(float(num_pores), 1.0)
    largest_component_ratio = float(comp_sizes.max() / max(float(num_pores), 1.0)) if len(comp_sizes) else 0.0

    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    coord_span = np.maximum(coord_max - coord_min, 1.0e-12)
    coords_norm = (coords - coord_min) / coord_span
    distance_to_min = coords_norm
    distance_to_max = 1.0 - coords_norm

    comp_percolates = np.zeros((len(comp_sizes), 3), dtype=bool)
    tolerance = 1.0e-6
    for cid in range(len(comp_sizes)):
        nodes = comp_id == cid
        comp_coords = coords_norm[nodes]
        if comp_coords.size == 0:
            continue
        for axis in range(3):
            touches_low = bool((comp_coords[:, axis] <= tolerance).any())
            touches_high = bool((comp_coords[:, axis] >= 1.0 - tolerance).any())
            comp_percolates[cid, axis] = touches_low and touches_high
    is_percolating_component = comp_percolates[comp_id].any(axis=1).astype(np.float64)
    percolates = comp_percolates.any(axis=0).astype(bool)

    degree = np.bincount(conns.reshape(-1), minlength=num_pores).astype(np.float64) if num_throats else np.zeros(num_pores)
    dead_end_ratio = float((degree <= 1).sum() / max(float(num_pores), 1.0))
    bridges = _bridge_edges(num_pores, conns) if num_throats else np.zeros(0, dtype=np.float32)
    bridge_incident = np.zeros(num_pores, dtype=np.float64)
    if num_throats:
        np.add.at(bridge_incident, conns[:, 0], bridges)
        np.add.at(bridge_incident, conns[:, 1], bridges)
    bridge_score_node = bridge_incident / np.maximum(degree, 1.0)

    edge_lengths = (
        np.linalg.norm(coords[conns[:, 1]] - coords[conns[:, 0]], axis=1)
        if num_throats
        else np.zeros(0, dtype=np.float64)
    )
    skeleton_length = float(edge_lengths.sum())
    euler_number = int(num_pores - num_throats + len(comp_sizes))

    node_topology = np.concatenate(
        [
            comp_size_ratio[:, None],
            is_percolating_component[:, None],
            bridge_score_node[:, None],
            distance_to_min,
            distance_to_max,
        ],
        axis=1,
    ).astype(np.float32)

    if num_throats:
        edge_comp_ratio = np.minimum(comp_size_ratio[conns[:, 0]], comp_size_ratio[conns[:, 1]])
        edge_perc = (
            is_percolating_component[conns[:, 0]].astype(bool)
            & is_percolating_component[conns[:, 1]].astype(bool)
        ).astype(np.float64)
        if throat_radius is not None and len(throat_radius) == num_throats:
            inv_radius = 1.0 / np.maximum(np.asarray(throat_radius, dtype=np.float64), 1.0e-12)
            bottleneck_score = inv_radius / np.maximum(inv_radius.max(), 1.0e-12)
        else:
            bottleneck_score = bridges.astype(np.float64)
        edge_topology = np.stack([bridges, edge_perc, edge_comp_ratio, bottleneck_score], axis=1).astype(np.float32)
    else:
        edge_topology = np.zeros((0, 4), dtype=np.float32)

    metadata = {
        "percolates_z": bool(percolates[0]),
        "percolates_y": bool(percolates[1]),
        "percolates_x": bool(percolates[2]),
        "largest_component_ratio": largest_component_ratio,
        "num_components": int(len(comp_sizes)),
        "dead_end_ratio": dead_end_ratio,
        "skeleton_length": skeleton_length,
        "euler_number": euler_number,
        "bridge_fraction": float(bridges.mean()) if len(bridges) else 0.0,
    }
    return node_topology, edge_topology, metadata


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


def _array_from_network(pn, candidates: list[str], default: np.ndarray | float | None = None) -> np.ndarray:
    for key in candidates:
        if key in pn.keys():
            return np.asarray(pn[key], dtype=np.float64)
    if default is None:
        raise KeyError(f"None of the network properties exist: {candidates}")
    return np.asarray(default, dtype=np.float64)


def openpnm_to_pore_network_data(
    pn,
    domain_size: tuple[float, float, float] | None = None,
    mu: float = 1.0e-3,
    include_ph: bool = True,
) -> PoreNetworkData:
    coords = _array_from_network(pn, ["pore.coords"])
    conns = _array_from_network(pn, ["throat.conns"]).astype(np.int64)
    num_pores = coords.shape[0]
    num_throats = conns.shape[0]

    if domain_size is None:
        spans = coords.max(axis=0) - coords.min(axis=0)
        domain_size = tuple(float(max(span, 1.0)) for span in spans)

    pore_d = _array_from_network(
        pn,
        ["pore.inscribed_diameter", "pore.equivalent_diameter", "pore.diameter"],
        default=np.ones(num_pores),
    )
    throat_d = _array_from_network(
        pn,
        ["throat.inscribed_diameter", "throat.equivalent_diameter", "throat.diameter"],
        default=np.ones(num_throats),
    )
    pore_volume = _array_from_network(pn, ["pore.volume"], default=np.zeros(num_pores))

    length_default = np.linalg.norm(coords[conns[:, 0]] - coords[conns[:, 1]], axis=1)
    throat_length = _array_from_network(
        pn,
        ["throat.total_length", "throat.length", "throat.direct_length"],
        default=length_default,
    )
    throat_length = np.maximum(throat_length, 1.0e-12)

    coordination = np.bincount(conns.reshape(-1), minlength=num_pores).astype(np.float64)
    coord_min = coords.min(axis=0)
    coord_span = np.maximum(coords.max(axis=0) - coord_min, 1.0e-12)
    coords_norm = (coords - coord_min) / coord_span

    pore_radius = 0.5 * np.maximum(pore_d, 1.0e-12)
    throat_radius = 0.5 * np.maximum(throat_d, 1.0e-12)
    log_g_hp = hagen_poiseuille_log_conductance(throat_radius, throat_length, mu=mu)
    node_topology, edge_topology, topology_metadata = graph_topology_features(coords, conns, throat_radius)

    ph_summary = persistent_homology_summary(coords) if include_ph else np.zeros(6, dtype=np.float32)
    node_ph = np.repeat(ph_summary[None, :], num_pores, axis=0)
    edge_ph = np.repeat(ph_summary[None, :], num_throats, axis=0)

    node_attr = np.concatenate(
        [
            pore_radius[:, None],
            pore_volume[:, None],
            coordination[:, None],
            coords_norm,
            node_topology,
            node_ph,
        ],
        axis=1,
    ).astype(np.float32)

    edge_vec = coords[conns[:, 1]] - coords[conns[:, 0]]
    edge_attr = np.concatenate(
        [
            log_g_hp[:, None],
            throat_length[:, None],
            throat_radius[:, None],
            edge_vec.astype(np.float64),
            edge_topology,
            edge_ph,
        ],
        axis=1,
    ).astype(np.float32)

    return PoreNetworkData(
        coords=torch.as_tensor(coords.astype(np.float32)),
        edge_index=torch.as_tensor(conns.T, dtype=torch.long),
        node_attr=torch.as_tensor(node_attr),
        edge_attr=torch.as_tensor(edge_attr),
        log_g_hp=torch.as_tensor(log_g_hp.astype(np.float32)),
        domain_size=domain_size,
        metadata={
            "num_pores": int(num_pores),
            "num_throats": int(num_throats),
            "node_feature_dim": int(node_attr.shape[1]),
            "edge_feature_dim": int(edge_attr.shape[1]),
            "ph_summary": ph_summary.tolist(),
            "topology": topology_metadata,
        },
    )


def calculate_openpnm_stokes_permeability(
    pn,
    domain_size: tuple[float, float, float],
    mu: float = 1.0e-3,
) -> dict[str, float]:
    import openpnm as op

    coords = np.asarray(pn["pore.coords"], dtype=np.float64)
    conns = np.asarray(pn["throat.conns"], dtype=np.int64)
    throat_d = _array_from_network(
        pn,
        ["throat.inscribed_diameter", "throat.equivalent_diameter", "throat.diameter"],
        default=np.ones(conns.shape[0]),
    )
    throat_radius = 0.5 * np.maximum(throat_d, 1.0e-12)
    throat_length = _array_from_network(
        pn,
        ["throat.total_length", "throat.length", "throat.direct_length"],
        default=np.linalg.norm(coords[conns[:, 0]] - coords[conns[:, 1]], axis=1),
    )
    hydraulic_conductance = np.exp(hagen_poiseuille_log_conductance(throat_radius, throat_length, mu=mu))

    phase = op.phase.Phase(network=pn)
    phase["throat.hydraulic_conductance"] = hydraulic_conductance

    result: dict[str, float] = {}
    axis_names = ("z", "y", "x")
    for axis, name in enumerate(axis_names):
        ca = coords[:, axis]
        inlet = np.where(ca <= ca.min())[0]
        outlet = np.where(ca >= ca.max())[0]
        if len(inlet) == 0 or len(outlet) == 0:
            result[f"k{name}"] = 0.0
            continue

        flow = op.algorithms.StokesFlow(network=pn, phase=phase)
        flow.set_value_BC(pores=inlet, values=1.0)
        flow.set_value_BC(pores=outlet, values=0.0)
        flow.run()
        rate = abs(float(np.asarray(flow.rate(pores=inlet, mode="group")).reshape(-1)[0]))

        length = domain_size[axis]
        other = [i for i in range(3) if i != axis]
        area = domain_size[other[0]] * domain_size[other[1]]
        result[f"k{name}"] = rate * mu * length / max(area, 1.0e-30)

    return {"kx": result["kx"], "ky": result["ky"], "kz": result["kz"]}
