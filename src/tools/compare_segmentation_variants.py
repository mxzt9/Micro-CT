from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_last_history_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick 64-cube comparison for adaptive/topology segmenters.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cube-size", type=int, default=64)
    parser.add_argument("--models", nargs="+", choices=("adaptive", "topology"), default=["adaptive", "topology"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--ctx-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    parser.add_argument("--topology-weight", type=float, default=0.01)
    parser.add_argument("--topology-max-size", type=int, default=32)
    parser.add_argument("--topology-cache-dir", type=Path, default=ROOT / "outputs" / "topology_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "segmentation_variant_comparison_64.csv")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int]] = []
    train_script = ROOT / "src" / "tools" / "train_segmentation.py"

    for model in args.models:
        checkpoint = ROOT / "models" / f"{model}_quick_{args.cube_size}.pth"
        history_csv = args.output.with_name(f"{args.output.stem}_{model}_history.csv")
        cmd = [
            sys.executable,
            str(train_script),
            "--root",
            str(args.root),
            "--model",
            model,
            "--mode",
            "quick",
            "--cube-sizes",
            str(args.cube_size),
            "--epochs",
            str(args.epochs),
            "--samples-per-group",
            str(args.samples_per_group),
            "--max-train-batches",
            str(args.max_train_batches),
            "--max-val-batches",
            str(args.max_val_batches),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--base-channels",
            str(args.base_channels),
            "--ctx-dim",
            str(args.ctx_dim),
            "--lr",
            str(args.lr),
            "--aux-weight",
            str(args.aux_weight),
            "--topology-weight",
            str(args.topology_weight),
            "--topology-max-size",
            str(args.topology_max_size),
            "--topology-cache-dir",
            str(args.topology_cache_dir),
            "--checkpoint",
            str(checkpoint),
            "--history-csv",
            str(history_csv),
        ]
        if args.no_amp:
            cmd.append("--no-amp")

        start = time.perf_counter()
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - start

        row = read_last_history_row(history_csv)
        row.update(
            {
                "model": model,
                "cube_size": args.cube_size,
                "elapsed_sec": f"{elapsed:.3f}",
                "checkpoint": str(checkpoint),
                "history_csv": str(history_csv),
            }
        )
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
