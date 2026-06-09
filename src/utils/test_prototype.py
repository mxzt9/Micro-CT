from __future__ import annotations

import torch

from utils import TopologyAdaptiveRoutedUNet3D, PoreNetworkPermeabilityModel


def main() -> None:
    torch.manual_seed(0)

    print("=" * 60)
    print("1) Topology-adaptive routed UNet3D")
    print("=" * 60)
    unet = TopologyAdaptiveRoutedUNet3D(in_channels=1, out_channels=1, base_channels=8, ctx_dim=32, ph_dim=6, topology_dim=6)
    x = torch.randn(2, 1, 64, 64, 64)
    ph_features = torch.randn(2, 6)
    out = unet(x, ph_features=ph_features, return_dict=True)
    logits = out["logits"]
    emb = out["decoder_embedding"]
    print("input        :", tuple(x.shape))
    print("mask logits  :", tuple(logits.shape))
    print("voxel embeds :", tuple(emb.shape))

    alpha = out["router_alpha"]
    print("alpha shape:", alpha.shape)
    print("alpha row sums:", alpha.sum(dim=-1).detach())

    loss = logits.mean() + emb.pow(2).mean()
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in unet.router.parameters())
    print("grad reached router:", grad_ok)

    print()
    print("=" * 60)
    print("2) GNN conductance + differentiable PNM")
    print("=" * 60)
    node_attr = torch.rand(3, 6)
    coords = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    log_g_hp = torch.tensor([0.0, 0.0])
    edge_attr = torch.stack([log_g_hp, torch.ones_like(log_g_hp), torch.ones_like(log_g_hp)], dim=1)

    model = PoreNetworkPermeabilityModel(node_in=6, edge_in=3, hidden=32, layers=2, mu=1.0)
    k, log_g = model(node_attr, edge_index, edge_attr, coords, (1.0, 1.0, 1.0), log_g_hp=log_g_hp)
    print("k (kx, ky, kz):", k.detach())
    print("log_g:", log_g.detach())


if __name__ == "__main__":
    main()
