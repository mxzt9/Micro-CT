from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import (  # noqa: E402
    BCEDiceLoss,
    DEFAULT_CUBE_SIZES,
    BereaPatchDataset,
    CubeSizeBatchSampler,
    FiLMRoutedUNet3D,
    auxiliary_physics_loss,
    dice_score_from_logits,
)
from utils.training import EarlyStopping, MetricTracker  # noqa: E402


def parse_weights(value: str | None) -> dict[int, float] | None:
    if not value:
        return None
    result: dict[int, float] = {}
    for item in value.split(","):
        size, weight = item.split(":")
        result[int(size)] = float(weight)
    return result


def parse_int_map(value: str | None) -> dict[int, int] | None:
    if not value:
        return None
    result: dict[int, int] = {}
    for item in value.split(","):
        size, count = item.split(":")
        result[int(size)] = int(count)
    return result


def make_loader(
    dataset,
    batch_size: int | dict[int, int],
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    *,
    seed: int,
) -> DataLoader:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    if hasattr(dataset, "df") and hasattr(dataset, "sample_index") and "cube_size" in dataset.df.columns:
        sampler = CubeSizeBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)
        kwargs["batch_sampler"] = sampler
    else:
        if isinstance(batch_size, dict):
            raise TypeError("per-cube batch sizes require a BereaPatchDataset-like dataset")
        kwargs.update({"batch_size": batch_size, "shuffle": shuffle})
    return DataLoader(dataset, **kwargs)


def run_epoch(model, loader, criterion, optimizer, scaler, device, args, train: bool) -> dict[str, float]:
    model.train(train)
    stats = MetricTracker()
    limit = args.max_train_batches if train else args.max_val_batches
    desc = "train" if train else "val"
    iterator = tqdm(loader, desc=desc, leave=False)

    for batch_idx, batch in enumerate(iterator):
        if limit is not None and batch_idx >= limit:
            break
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        porosity = batch["porosity"].to(device, non_blocking=True)
        percolates = batch["percolates"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            out = model(x, return_dict=True)
            logits = out["logits"]
            seg_loss, bce_loss, dice_loss = criterion(logits, y)
            aux_loss, _ = auxiliary_physics_loss(
                out,
                y,
                porosity_target=porosity,
                percolation_target=percolates,
                porosity_weight=args.aux_weight,
                percolation_weight=args.aux_weight,
            )
            loss = seg_loss + aux_loss

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = x.size(0)
        with torch.no_grad():
            dice = dice_score_from_logits(logits, y)
        stats.update("loss", float(loss.detach().cpu()), batch_size)
        stats.update("seg_loss", float(seg_loss.detach().cpu()), batch_size)
        stats.update("aux_loss", float(aux_loss.detach().cpu()), batch_size)
        stats.update("bce", float(bce_loss.detach().cpu()), batch_size)
        stats.update("dice_loss", float(dice_loss.detach().cpu()), batch_size)
        stats.update("dice", float(dice.detach().cpu()), batch_size)
        iterator.set_postfix(stats.postfix("loss", "dice"))

    return stats.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FiLMRoutedUNet3D segmentation outside Jupyter.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--cube-sizes", nargs="+", type=int, default=list(DEFAULT_CUBE_SIZES))
    parser.add_argument("--size-weights", default="64:0.50,128:0.35,192:0.15")
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--batch-size-by-cube-size",
        default=None,
        help="Optional mapping like 64:8,128:2,192:1. Keeps each batch at one cube size.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "film_routed_unet3d_best.pth")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.mode == "full":
        args.samples_per_group = None
        args.max_train_batches = None
        args.max_val_batches = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = not args.no_amp
    pin_memory = device.type == "cuda"
    loader_batch_size = parse_int_map(args.batch_size_by_cube_size) or args.batch_size
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    train_ds = BereaPatchDataset(
        args.root,
        split="train",
        cube_size=args.cube_sizes,
        balance=True,
        samples_per_group=args.samples_per_group,
        size_sampling_weights=parse_weights(args.size_weights),
    )
    val_ds = BereaPatchDataset(
        args.root,
        split="val",
        cube_size=args.cube_sizes,
        noise_types=["none"],
        balance=False,
        samples_per_group=args.samples_per_group,
    )
    train_loader = make_loader(train_ds, loader_batch_size, True, args.num_workers, pin_memory, seed=42)
    val_loader = make_loader(val_ds, loader_batch_size, False, args.num_workers, pin_memory, seed=43)

    model = FiLMRoutedUNet3D(base_channels=args.base_channels, ctx_dim=args.ctx_dim).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode="min")
    history = []

    print(f"device={device} mode={args.mode} train={len(train_ds)} val={len(val_ds)} workers={args.num_workers}")
    print("train groups:")
    print(train_ds.df.groupby(["rock", "cube_size", "split"]).size().rename("samples").reset_index())

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, args, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, args, train=False)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        if early.step(val_metrics["loss"], epoch=epoch):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_dice": val_metrics["dice"],
                    "history": history,
                    "base_channels": args.base_channels,
                    "ctx_dim": args.ctx_dim,
                },
                args.checkpoint,
            )
            print(f"saved: {args.checkpoint}")
        elif early.should_stop:
            print(f"early stop at epoch={epoch}; best val_loss={early.best:.4f}")
            break


if __name__ == "__main__":
    main()
