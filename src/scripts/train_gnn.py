from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.pnm_gnn import PoreNetworkPermeabilityModel  # noqa: E402
from utils.training import EarlyStopping, MetricTracker  # noqa: E402


def load_networks(network_dir: Path, prefix: str = "") -> list[Any]:
    """Загружает .pt файлы поровых сетей из директории."""
    pattern = f"{prefix}*.pt" if prefix else "*.pt"
    paths = sorted(network_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No network .pt files found in {network_dir} matching '{pattern}'")
    networks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    return networks


def target_k_from_metadata(network: Any) -> torch.Tensor:
    """Извлекает target проницаемость из metadata['openpnm_k']."""
    k = network.metadata.get("openpnm_k")
    if k is None:
        raise ValueError("No target k found. Add OpenPNM target to network.metadata['openpnm_k'].")
    return torch.tensor([k["kx"], k["ky"], k["kz"]], dtype=torch.float32)


def compute_network_loss(
    model: PoreNetworkPermeabilityModel,
    network: Any,
    device: torch.device,
    physics_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Вычисляет loss для одной сети.

    Примечания:
    - Нормализация на медиану k удалена: loss считается в log-пространстве,
      где деление на константу — это вычитание одинаковой константы из pred
      и target, которое сокращается в smooth_l1. Старый код её и так нигде
      не использовал (мёртвый параметр).
    - Неперколирующие оси (k=0) маскируются: иначе clamp(1e-30) даёт
      log(k) = -69, и такая цель доминирует над loss всех остальных осей.

    Args:
        model: GNN+PNM модель
        network: PoreNetworkData
        device: устройство
        physics_weight: вес физического auxiliary loss (0 = выключен)

    Returns:
        dict с ключами: loss, k_loss, pred_k_log, target_k_log,
        physics_residual, phys_loss, valid_axes
    """
    network = network.to(device)
    target_k_raw = target_k_from_metadata(network).to(device)
    # Маска перколирующих осей: k <= 0 означает отсутствие связного пути
    valid_axes = target_k_raw > 0
    target_k = target_k_raw.clamp_min(1e-30)
    target_k_log = torch.log(target_k)

    if physics_weight > 0:
        pred_k, log_g, flow_residual = model.forward_with_physics_loss(
            network.node_attr,
            network.edge_index,
            network.edge_attr,
            network.coords,
            network.domain_size,
            log_g_hp=network.log_g_hp,
        )
    else:
        pred_k, log_g = model(
            network.node_attr,
            network.edge_index,
            network.edge_attr,
            network.coords,
            network.domain_size,
            log_g_hp=network.log_g_hp,
        )
        flow_residual = torch.tensor(0.0, device=device)

    pred_k = pred_k.clamp_min(1e-30)
    pred_k_log = torch.log(pred_k)

    # Loss в log-space только по перколирующим осям
    if valid_axes.any():
        k_loss = F.smooth_l1_loss(pred_k_log[valid_axes], target_k_log[valid_axes])
    else:
        k_loss = pred_k_log.sum() * 0.0

    # Физический auxiliary loss (невязка только по внутренним узлам решателя)
    phys_loss = physics_weight * flow_residual

    total_loss = k_loss + phys_loss

    return {
        "loss": total_loss,
        "k_loss": k_loss.detach(),
        "pred_k_log": pred_k_log.detach(),
        "target_k_log": target_k_log.detach(),
        "physics_residual": flow_residual.detach(),
        "phys_loss": phys_loss.detach(),
        "valid_axes": valid_axes.detach(),
    }


@torch.no_grad()
def evaluate(
    model: PoreNetworkPermeabilityModel,
    networks: list[Any],
    device: torch.device,
    physics_weight: float = 0.0,
) -> dict[str, float]:
    """Оценка на валидационной выборке."""
    if not networks:
        return {"loss": 0.0, "k_loss": 0.0}
    model.eval()
    stats = MetricTracker()
    for network in networks:
        result = compute_network_loss(model, network, device, physics_weight=physics_weight)
        stats.update("loss", float(result["loss"].cpu()), 1)
        stats.update("k_loss", float(result["k_loss"].cpu()), 1)
        stats.update("physics_residual", float(result["physics_residual"].cpu()), 1)
        stats.update("phys_loss", float(result["phys_loss"].cpu()), 1)
    return stats.as_dict()


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PoreNetworkPermeabilityModel (GNN+PNM) on extracted pore networks."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--network-dir", type=Path, default=ROOT / "outputs" / "networks")
    parser.add_argument("--network-prefix", default="network_train_",
                        help="Prefix for training network files (e.g. 'network_train_')")
    parser.add_argument("--val-prefix", default=None,
                        help="Optional prefix for validation network files. If not set, uses random split from train files.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "gnn_pnm_best.pth")
    parser.add_argument("--history-csv", type=Path, default=None)

    # Модель
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)

    # Обучение
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm. 0 to disable.")

    # Физический loss
    # Default 0: после точного решения СЛАУ невязка во внутренних узлах ~0,
    # так что этот лосс почти не информативен; включать только осознанно
    # (например, при переходе на итеративный/неточный решатель).
    parser.add_argument("--physics-weight", type=float, default=0.0,
                        help="Weight for physics auxiliary loss (interior mass conservation). 0 to disable.")

    # Мини-батчи: шаг оптимизатора каждые N сетей (раньше был 1 шаг на эпоху —
    # full-batch GD, всего ~200 шагов за всё обучение)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Networks per optimizer step.")

    # Split
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of networks for validation (if no val prefix files)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Загрузка сетей
    # Если val_prefix не указан, считаем что все сети в network-dir с префиксом — train
    # и делаем random split
    train_networks = load_networks(args.network_dir, prefix=args.network_prefix)
    print(f"Loaded {len(train_networks)} training networks from {args.network_dir}")

    # Валидационные сети: либо по префиксу (если указан), либо разделяем train
    val_networks: list[Any] = []
    if args.val_prefix:
        try:
            val_networks = load_networks(args.network_dir, prefix=args.val_prefix)
            print(f"Loaded {len(val_networks)} validation networks using prefix '{args.val_prefix}'")
        except FileNotFoundError:
            print(f"No validation files with prefix '{args.val_prefix}'. Falling back to random split.")
            args.val_prefix = None

    if not args.val_prefix:
        # Random split
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(train_networks), generator=generator).tolist()
        val_count = max(1, int(round(len(train_networks) * args.val_fraction)))
        val_count = min(val_count, len(train_networks) - 1) if len(train_networks) > 1 else 0
        if val_count > 0:
            val_indices = set(order[:val_count])
            all_train = train_networks
            train_networks = [n for idx, n in enumerate(all_train) if idx not in val_indices]
            val_networks = [n for idx, n in enumerate(all_train) if idx in val_indices]
        print(f"Random split: {len(train_networks)} train, {len(val_networks)} val")

    if not train_networks:
        raise ValueError("No training networks available!")

    # Определяем размерности
    first = train_networks[0]
    node_in = first.node_attr.shape[1]
    edge_in = first.edge_attr.shape[1]
    print(f"node_in={node_in}, edge_in={edge_in}")

    # Модель и оптимизатор
    model = PoreNetworkPermeabilityModel(
        node_in=node_in,
        edge_in=edge_in,
        hidden=args.hidden,
        layers=args.layers,
        mu=1.0e-3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # CosineAnnealing scheduler (шаг — по эпохам)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode="min")

    params_count = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {params_count:,}")
    print(f"train networks: {len(train_networks)}, val networks: {len(val_networks)}")
    print(f"epochs: {args.epochs}, lr: {args.lr}, patience: {args.patience}")
    print(f"physics_weight: {args.physics_weight}, batch_size: {args.batch_size}")
    print(f"grad_clip: {args.grad_clip}")
    print()

    history = []
    batch_size = max(1, int(args.batch_size))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_stats = MetricTracker()

        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_order = torch.randperm(len(train_networks), generator=generator).tolist()

        optimizer.zero_grad(set_to_none=True)
        in_batch = 0

        def optimizer_step() -> None:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for idx in tqdm(train_order, desc=f"epoch {epoch:3d}", leave=False):
            network = train_networks[idx]

            result = compute_network_loss(
                model, network, device,
                physics_weight=args.physics_weight,
            )
            # Усредняем по мини-батчу
            (result["loss"] / batch_size).backward()
            in_batch += 1
            if in_batch == batch_size:
                optimizer_step()
                in_batch = 0

            train_stats.update("loss", float(result["loss"].detach().cpu()), 1)
            train_stats.update("k_loss", float(result["k_loss"].cpu()), 1)
            train_stats.update("physics_residual", float(result["physics_residual"].cpu()), 1)
            train_stats.update("phys_loss", float(result["phys_loss"].cpu()), 1)

        # Хвост неполного мини-батча
        if in_batch > 0:
            optimizer_step()
        scheduler.step()

        train_metrics = train_stats.as_dict()

        # Валидация
        val_metrics = evaluate(
            model, val_networks, device,
            physics_weight=args.physics_weight,
        )

        # Единая метрика для early stopping: k_loss (на val, иначе на train).
        # Раньше сравнивались разные величины (val total loss vs train k_loss).
        monitor_loss = val_metrics.get("k_loss", train_metrics["k_loss"])

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_physics_residual": train_metrics.get("physics_residual", 0.0),
            "val_loss": val_metrics.get("loss", 0.0),
            "val_physics_residual": val_metrics.get("physics_residual", 0.0),
            "lr": float(scheduler.get_last_lr()[0]),
        })

        print(
            f"epoch={epoch:3d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics.get('loss', 0.0):.6f} "
            f"physics_res={train_metrics.get('physics_residual', 0.0):.2e} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping + save
        if early.step(monitor_loss, epoch=epoch):
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "checkpoint_type": "gnn_pnm",
                    "target_name": "openpnm_k",
                    "model_state_dict": model.state_dict(),
                    "node_in": node_in,
                    "edge_in": edge_in,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "epoch": epoch,
                    "monitor_loss": monitor_loss,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics.get("loss"),
                    "history": history,
                },
                args.checkpoint,
            )
            print(f"  saved: {args.checkpoint}")
        elif early.should_stop:
            print(f"  early stop at epoch={epoch}; best loss={early.best:.6f}")
            break

    if args.history_csv is not None:
        write_history_csv(args.history_csv, history)
        print(f"history: {args.history_csv}")


if __name__ == "__main__":
    main()