"""Precompute topology (PH) cache for all train/val cubes.

Run once before training to avoid GPU idle time during data loading.
Uses BereaPatchDataset internally — same code path as training.

Usage:
    python src/precompute_topology_cache.py
    python src/precompute_topology_cache.py --cube-sizes 64 128 192
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import BereaPatchDataset, CubeSizeBatchSampler, DEFAULT_CUBE_SIZES


def count_cached(cache_dir: Path, rock: str, prefix: str) -> int:
    """Count .npy files matching a prefix in the cache dir."""
    rock_dir = cache_dir / rock
    if not rock_dir.exists():
        return 0
    return len(list(rock_dir.glob(f"{prefix}*.npy")))


def precompute(
    cube_sizes: list[int],
    topology_max_size: int = 32,
    batch_size: int = 1,
) -> None:
    cache_dir = ROOT / "outputs" / "topology_cache"

    for split in ("train", "val"):
        print(f"\n=== {split} ===")

        # Count cached before
        cached_before = 0
        for rock_dir in sorted((ROOT / "datasets").iterdir()):
            if rock_dir.is_dir():
                for cs in cube_sizes:
                    prefix = f"cs{cs}_"
                    cached_before += count_cached(cache_dir, rock_dir.name, prefix)

        ds = BereaPatchDataset(
            ROOT,
            split=split,
            cube_size=cube_sizes,
            use_raw_gray=False,
            balance=False,
            samples_per_group=None,
            return_topology=True,
            topology_max_size=topology_max_size,
            return_aux_targets=False,
        )
        sampler = CubeSizeBatchSampler(ds, {cs: batch_size for cs in cube_sizes}, shuffle=False)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

        total = len(ds)
        print(f"  {total} samples in dataset, ~{cached_before} already cached")

        start = time.perf_counter()
        for batch in tqdm(loader, desc=f"  {split}", mininterval=2.0):
            _ = batch["ph_features"]  # triggers caching via __getitem__

        elapsed = time.perf_counter() - start

        # Count cached after
        cached_after = 0
        for rock_dir in sorted((ROOT / "datasets").iterdir()):
            if rock_dir.is_dir():
                for cs in cube_sizes:
                    prefix = f"cs{cs}_"
                    cached_after += count_cached(cache_dir, rock_dir.name, prefix)

        new = cached_after - cached_before
        print(f"  {new} new, {cached_after} total in {elapsed:.1f}s ({max(new, 1) / max(elapsed, 0.1):.1f}/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PH cache")
    parser.add_argument("--cube-sizes", type=int, nargs="+", default=list(DEFAULT_CUBE_SIZES))
    parser.add_argument("--topology-max-size", type=int, default=32)
    args = parser.parse_args()

    precompute(
        cube_sizes=args.cube_sizes,
        topology_max_size=args.topology_max_size,
    )


if __name__ == "__main__":
    main()