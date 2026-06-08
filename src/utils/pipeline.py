from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .common import sliding_window_inference_3d
from .film_routing import FiLMRoutedUNet3D
from .network import (
    PoreNetworkData,
    calculate_openpnm_stokes_permeability,
    extract_porespy_openpnm_network,
    openpnm_to_pore_network_data,
)
from .pnm_gnn import PoreNetworkPermeabilityModel


@dataclass
class SegmentationResult:
    logits: torch.Tensor | None
    probability: torch.Tensor
    mask: np.ndarray
    embeddings: torch.Tensor | None = None
    rock_embedding: torch.Tensor | None = None
    aux_outputs: dict[str, torch.Tensor] | None = None


@dataclass
class PermeabilityResult:
    k_gnn_pnm: torch.Tensor
    log_g: torch.Tensor
    k_openpnm: dict[str, float] | None


@dataclass
class PipelineResult:
    segmentation: SegmentationResult
    network: PoreNetworkData
    permeability: PermeabilityResult
    openpnm_network: Any


class DigitalCorePipeline:
    """Единый пайплайн CNN -> PoreSpy/OpenPNM -> GNN -> дифференцируемый PNM."""

    def __init__(
        self,
        segmentation_model: torch.nn.Module | None = None,
        graph_model: PoreNetworkPermeabilityModel | None = None,
        device: str | torch.device | None = None,
        threshold: float = 0.5,
        voxel_size: float = 1.0,
        mu: float = 1.0e-3,
        sliding_window_size: int | tuple[int, int, int] | None = None,
        sliding_overlap: float = 0.5,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.segmentation_model = segmentation_model.to(self.device) if segmentation_model is not None else None
        self.graph_model = graph_model.to(self.device) if graph_model is not None else None
        self.threshold = threshold
        self.voxel_size = voxel_size
        self.mu = mu
        self.sliding_window_size = sliding_window_size
        self.sliding_overlap = sliding_overlap

    @classmethod
    def with_default_models(
        cls,
        base_channels: int = 16,
        ctx_dim: int = 64,
        threshold: float = 0.5,
        voxel_size: float = 1.0,
        mu: float = 1.0e-3,
        device: str | torch.device | None = None,
        sliding_window_size: int | tuple[int, int, int] | None = None,
        sliding_overlap: float = 0.5,
    ) -> "DigitalCorePipeline":
        segmentation_model = FiLMRoutedUNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            ctx_dim=ctx_dim,
            return_embeddings=True,
        )
        return cls(
            segmentation_model=segmentation_model,
            threshold=threshold,
            voxel_size=voxel_size,
            mu=mu,
            device=device,
            sliding_window_size=sliding_window_size,
            sliding_overlap=sliding_overlap,
        )

    def _cube_to_tensor(self, cube: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(cube, np.ndarray):
            tensor = torch.from_numpy(cube.astype(np.float32, copy=False))
        else:
            tensor = cube.detach().float()

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 5:
            raise ValueError("cube должен иметь форму [D,H,W], [C,D,H,W] или [B,C,D,H,W]")

        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.to(self.device)

    def segment_cube(self, cube: np.ndarray | torch.Tensor, ph_features: torch.Tensor | None = None) -> SegmentationResult:
        if self.segmentation_model is None:
            tensor = self._cube_to_tensor(cube)
            probability = tensor[:, :1].clamp(0.0, 1.0)
            return SegmentationResult(
                logits=None,
                probability=probability,
                mask=(probability[0, 0].detach().cpu().numpy() >= self.threshold),
                embeddings=None,
                rock_embedding=None,
                aux_outputs=None,
            )

        tensor = self._cube_to_tensor(cube)
        self.segmentation_model.eval()
        with torch.no_grad():
            if self.sliding_window_size is not None:
                logits = sliding_window_inference_3d(
                    self.segmentation_model,
                    tensor,
                    window_size=self.sliding_window_size,
                    overlap=self.sliding_overlap,
                    ph_features=ph_features,
                )
                aux_outputs = self.segmentation_model(tensor, ph_features=ph_features, return_dict=True)
                embeddings = aux_outputs.get("decoder_embedding")
                rock_embedding = aux_outputs.get("rock_embedding")
            else:
                output = self.segmentation_model(tensor, ph_features=ph_features, return_dict=True)
                if isinstance(output, dict):
                    logits = output["logits"]
                    embeddings = output.get("decoder_embedding")
                    rock_embedding = output.get("rock_embedding")
                    aux_outputs = output
                elif isinstance(output, tuple):
                    logits, embeddings = output
                    rock_embedding = None
                    aux_outputs = None
                else:
                    logits, embeddings = output, None
                    rock_embedding = None
                    aux_outputs = None
            probability = torch.sigmoid(logits)

        return SegmentationResult(
            logits=logits,
            probability=probability,
            mask=(probability[0, 0].detach().cpu().numpy() >= self.threshold),
            embeddings=embeddings,
            rock_embedding=rock_embedding,
            aux_outputs=aux_outputs,
        )

    def extract_network(
        self,
        pore_mask: np.ndarray,
        domain_size: tuple[float, float, float] | None = None,
        include_ph: bool = True,
    ) -> tuple[Any, PoreNetworkData]:
        pn = extract_porespy_openpnm_network(
            pore_mask=pore_mask,
            voxel_size=self.voxel_size,
        )
        network_data = openpnm_to_pore_network_data(
            pn,
            domain_size=domain_size,
            mu=self.mu,
            include_ph=include_ph,
        )
        return pn, network_data

    def _ensure_graph_model(self, network: PoreNetworkData) -> PoreNetworkPermeabilityModel:
        if self.graph_model is None:
            self.graph_model = PoreNetworkPermeabilityModel(
                node_in=network.node_attr.shape[1],
                edge_in=network.edge_attr.shape[1],
                hidden=64,
                layers=3,
                mu=self.mu,
            ).to(self.device)
        return self.graph_model

    def predict_permeability(self, network: PoreNetworkData) -> tuple[torch.Tensor, torch.Tensor]:
        network = network.to(self.device)
        graph_model = self._ensure_graph_model(network)
        graph_model.eval()
        with torch.no_grad():
            k, log_g = graph_model(
                network.node_attr,
                network.edge_index,
                network.edge_attr,
                network.coords,
                network.domain_size,
                log_g_hp=network.log_g_hp,
            )
        return k, log_g

    def run_cube(
        self,
        cube: np.ndarray | torch.Tensor,
        input_is_pore_mask: bool = False,
        domain_size: tuple[float, float, float] | None = None,
        include_ph: bool = True,
        compute_openpnm_baseline: bool = True,
    ) -> PipelineResult:
        if input_is_pore_mask:
            pore_mask = np.asarray(cube).astype(bool)
            segmentation = SegmentationResult(
                logits=None,
                probability=torch.from_numpy(pore_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                mask=pore_mask,
                embeddings=None,
                rock_embedding=None,
                aux_outputs=None,
            )
        else:
            segmentation = self.segment_cube(cube)
            pore_mask = segmentation.mask

        pn, network_data = self.extract_network(
            pore_mask=pore_mask,
            domain_size=domain_size,
            include_ph=include_ph,
        )
        k_openpnm = (
            calculate_openpnm_stokes_permeability(pn, network_data.domain_size, mu=self.mu)
            if compute_openpnm_baseline
            else None
        )
        k_gnn_pnm, log_g = self.predict_permeability(network_data)

        return PipelineResult(
            segmentation=segmentation,
            network=network_data,
            permeability=PermeabilityResult(
                k_gnn_pnm=k_gnn_pnm,
                log_g=log_g,
                k_openpnm=k_openpnm,
            ),
            openpnm_network=pn,
        )
