"""Fast topology / persistent homology cache precompute.

This script computes the same .npy cache format that BereaPatchDataset already uses:

    outputs/topology_cache/<rock>/cs128_z0_y128_x256_raw_m32.npy
    outputs/topology_cache/<rock>/cs128_z0_y128_x256_target_m32.npy

Why this is faster than the old script:
- no DataLoader;
- no Dataset.__getitem__;
- no noise augmentation;
- no torch batch creation;
- direct index CSV reading;
- direct memmap slicing;
- optional multiprocessing.

Run from project root:

    python src/fast_precompute_topology_cache.py --source both --cube-sizes 64 128 192 --topology-max-size 32 --workers 8

Examples:

    # Only input PH features
    python src/fast_precompute_topology_cache.py --source raw --cube-sizes 64 128 --topology-max-size 24 --workers 8

    # Full current training cache: raw + target
    python src/fast_precompute_topology_cache.py --source both --cube-sizes 64 128 192 --topology-max-size 32 --workers 8

    # Safe single-process mode for debugging
    python src/fast_precompute_topology_cache.py --source both --workers 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def find_project_root(start: Path | None = None) -> Path:
    root = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (root, *root.parents):
        if (candidate / "src" / "utils").is_dir():
            return candidate
    raise RuntimeError(
        "Project root with src/utils was not found. "
        "Run this script from project root or place it inside src/."
    )


ROOT = find_project_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.data import DEFAULT_CUBE_SIZES, discover_rock_volumes, resolve_patch_index_path
from utils.topology import TOPOLOGY_FEATURE_DIM, cubical_persistence_summary


_WORKER_VOLUMES: dict[tuple[str, tuple[int, int, int]], np.memmap] = {}


def get_memmap(path: str, shape: tuple[int, int, int]) -> np.memmap:
    key = (path, shape)
    if key not in _WORKER_VOLUMES:
        _WORKER_VOLUMES[key] = np.memmap(path, dtype=np.uint8, mode="r", shape=shape)
    return _WORKER_VOLUMES[key]


def cache_path(
    cache_dir: Path,
    rock: str,
    cube_size: int,
    z: int,
    y: int,
    x: int,
    source: str,
    topology_max_size: int | None,
) -> Path:
    max_label = "full" if topology_max_size is None else str(int(topology_max_size))
    return cache_dir / rock / f"cs{int(cube_size)}_z{int(z)}_y{int(y)}_x{int(x)}_{source}_m{max_label}.npy"


def read_index_for_spec(spec: Any, cube_size: int, split: str) -> pd.DataFrame:
    index_path = resolve_patch_index_path(spec.index_dir, int(cube_size), spec.name)
    if index_path is None:
        raise FileNotFoundError(
            f"Index not found for rock={spec.name}, cube_size={cube_size}, dir={spec.index_dir}"
        )

    df = pd.read_csv(index_path)

    required = {"z", "y", "x", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{index_path} is missing columns: {sorted(missing)}")

    if "cube_size" in df.columns:
        df = df[df["cube_size"].astype(int) == int(cube_size)]

    df = df[df["split"] == split].copy()
    df["rock"] = spec.name
    df["cube_size"] = int(cube_size)
    df["gray_path"] = str(spec.gray_path)
    df["binary_path"] = str(spec.binary_path)
    df["shape_z"] = int(spec.shape[0])
    df["shape_y"] = int(spec.shape[1])
    df["shape_x"] = int(spec.shape[2])
    return df


def build_tasks(
    *,
    root: Path,
    cube_sizes: list[int],
    splits: list[str],
    source_mode: str,
    cache_dir: Path,
    topology_max_size: int | None,
    use_raw_gray: bool,
    pore_value: int,
    rocks: list[str] | None,
) -> list[dict[str, Any]]:
    if source_mode not in {"raw", "target", "both"}:
        raise ValueError("--source must be raw, target, or both")

    sources = ["raw", "target"] if source_mode == "both" else [source_mode]

    specs = discover_rock_volumes(
        root,
        use_raw_gray=use_raw_gray,
        rocks=rocks,
    )

    tasks: list[dict[str, Any]] = []

    for spec in specs:
        for split in splits:
            for cs in cube_sizes:
                df = read_index_for_spec(spec, int(cs), split)

                for row in df.itertuples(index=False):
                    z, y, x = int(row.z), int(row.y), int(row.x)

                    for source in sources:
                        out_path = cache_path(
                            cache_dir=cache_dir,
                            rock=spec.name,
                            cube_size=int(cs),
                            z=z,
                            y=y,
                            x=x,
                            source=source,
                            topology_max_size=topology_max_size,
                        )

                        if out_path.exists():
                            continue

                        tasks.append(
                            {
                                "rock": spec.name,
                                "cube_size": int(cs),
                                "z": z,
                                "y": y,
                                "x": x,
                                "source": source,
                                "gray_path": str(spec.gray_path),
                                "binary_path": str(spec.binary_path),
                                "shape": tuple(int(v) for v in spec.shape),
                                "out_path": str(out_path),
                                "topology_max_size": topology_max_size,
                                "pore_value": int(pore_value),
                            }
                        )

    return tasks


def compute_one_task(task: dict[str, Any]) -> tuple[str, bool, str | None]:
    """Compute one cache file. Returns (path, created, error)."""
    try:
        out_path = Path(task["out_path"])
        if out_path.exists():
            return str(out_path), False, None

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cs = int(task["cube_size"])
        z, y, x = int(task["z"]), int(task["y"]), int(task["x"])
        shape = tuple(int(v) for v in task["shape"])
        source = str(task["source"])

        if source == "raw":
            volume = get_memmap(task["gray_path"], shape)
            cube = np.asarray(volume[z : z + cs, y : y + cs, x : x + cs])
            features = cubical_persistence_summary(cube, max_size=task["topology_max_size"])

        elif source == "target":
            volume = get_memmap(task["binary_path"], shape)
            cube = np.asarray(volume[z : z + cs, y : y + cs, x : x + cs])
            mask = cube == int(task["pore_value"])
            features = cubical_persistence_summary(mask, max_size=task["topology_max_size"])

        else:
            raise ValueError(f"Unknown source: {source}")

        features = np.asarray(features, dtype=np.float32)
        if features.shape != (TOPOLOGY_FEATURE_DIM,):
            raise ValueError(f"Bad feature shape: {features.shape}")

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            np.save(f, features)
        os.replace(tmp_path, out_path)

        return str(out_path), True, None

    except Exception as exc:
        return str(task.get("out_path", "")), False, repr(exc)


def count_cache_files(cache_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in cache_dir.rglob("*.npy"):
        rock = path.parent.name
        parts = path.stem.split("_")
        try:
            cs = int(parts[0][2:])
            source = parts[4]
            max_size = parts[5][1:]
        except Exception:
            cs = -1
            source = "unknown"
            max_size = "unknown"

        rows.append(
            {
                "rock": rock,
                "cube_size": cs,
                "source": source,
                "topology_max_size": max_size,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["rock", "cube_size", "source", "topology_max_size", "count"])

    df = pd.DataFrame(rows)
    return (
        df.groupby(["rock", "cube_size", "source", "topology_max_size"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["rock", "cube_size", "source", "topology_max_size"])
    )


def run(tasks: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    if not tasks:
        print("No new tasks. Cache is already complete for selected settings.")
        return {"created": 0, "skipped": 0, "errors": [], "elapsed": 0.0, "speed": 0.0}

    start = time.perf_counter()
    created = 0
    skipped = 0
    errors: list[tuple[str, str]] = []

    if workers <= 1:
        iterator = (compute_one_task(task) for task in tasks)
        for path, was_created, error in tqdm(iterator, total=len(tasks), desc="precompute"):
            if error:
                errors.append((path, error))
            elif was_created:
                created += 1
            else:
                skipped += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(compute_one_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="precompute"):
                path, was_created, error = future.result()
                if error:
                    errors.append((path, error))
                elif was_created:
                    created += 1
                else:
                    skipped += 1

    elapsed = time.perf_counter() - start
    speed = created / max(elapsed, 1e-9)

    print()
    print("Done.")
    print("created:", created)
    print("skipped:", skipped)
    print("errors :", len(errors))
    print(f"elapsed: {elapsed:.1f} sec")
    print(f"speed  : {speed:.2f} files/sec")

    if errors:
        print()
        print("First errors:")
        for path, error in errors[:10]:
            print(path, "->", error)

    return {
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "elapsed": elapsed,
        "speed": speed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast topology cache precompute without DataLoader")
    parser.add_argument(
        "--cube-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUBE_SIZES),
        help="Cube sizes to process, e.g. --cube-sizes 64 128 192",
    )
    parser.add_argument(
        "--source",
        choices=["raw", "target", "both"],
        default="both",
        help="raw = input PH only, target = label PH only, both = raw + target",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--topology-max-size",
        type=int,
        default=32,
        help="Downsample largest side before GUDHI. Use 16/24 for speed, 32 for default quality.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes. Use 1 for debugging.",
    )
    parser.add_argument(
        "--pore-value",
        type=int,
        default=0,
        help="Pore value in binary volume.",
    )
    parser.add_argument(
        "--use-raw-gray",
        action="store_true",
        help="Use raw grayscale instead of filtered grayscale.",
    )
    parser.add_argument(
        "--rocks",
        nargs="+",
        default=None,
        help="Optional list of rock names to process.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(ROOT / "outputs" / "topology_cache"),
        help="Output cache dir.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print cache stats and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("ROOT:", ROOT)
    print("CACHE_DIR:", cache_dir)
    print("cube_sizes:", args.cube_sizes)
    print("source:", args.source)
    print("splits:", args.splits)
    print("topology_max_size:", args.topology_max_size)
    print("workers:", args.workers)
    print("pore_value:", args.pore_value)
    print("use_raw_gray:", args.use_raw_gray)
    print("rocks:", args.rocks)

    if args.stats_only:
        print(count_cache_files(cache_dir))
        return

    tasks = build_tasks(
        root=ROOT,
        cube_sizes=[int(x) for x in args.cube_sizes],
        splits=list(args.splits),
        source_mode=str(args.source),
        cache_dir=cache_dir,
        topology_max_size=int(args.topology_max_size),
        use_raw_gray=bool(args.use_raw_gray),
        pore_value=int(args.pore_value),
        rocks=args.rocks,
    )

    print("New tasks:", len(tasks))
    if tasks:
        print("First task:", tasks[0])

    result = run(tasks, workers=int(args.workers))

    print()
    print("Cache stats:")
    print(count_cache_files(cache_dir))

    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    # Required for Windows multiprocessing safety.
    import multiprocessing as mp

    mp.freeze_support()
    main()
