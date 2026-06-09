from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import extract_porespy_openpnm_network  # noqa: E402


def parse_zyx(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("value must look like z,y,x")
    return tuple(int(part.strip()) for part in parts)


def load_subcube(path: Path, shape: tuple[int, int, int], cube_size: int, origin: tuple[int, int, int]):
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=shape)
    z, y, x = origin
    cube = np.asarray(raw[z : z + cube_size, y : y + cube_size, x : x + cube_size]).copy()
    del raw
    return cube


def coords_to_indices(coords: np.ndarray, perm: tuple[int, int, int], flips: tuple[bool, bool, bool], shape):
    arr = coords[:, perm].copy()
    low = arr.min(axis=0)
    span = np.maximum(arr.max(axis=0) - low, 1.0e-6)
    arr = (arr - low) / span

    for axis, do_flip in enumerate(flips):
        if do_flip:
            arr[:, axis] = 1.0 - arr[:, axis]

    scale = np.asarray(shape, dtype=np.float32) - 1.0
    return np.rint(arr * scale).astype(np.int64)


def score_orientation(coords: np.ndarray, mask: np.ndarray, perm, flips):
    dist = distance_transform_edt(mask)
    idx = coords_to_indices(coords, perm, flips, mask.shape)
    idx[:, 0] = np.clip(idx[:, 0], 0, mask.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, mask.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, mask.shape[2] - 1)

    samples = dist[idx[:, 0], idx[:, 1], idx[:, 2]]
    inside = mask[idx[:, 0], idx[:, 1], idx[:, 2]]
    return {
        "perm": perm,
        "flips": flips,
        "inside": float(inside.mean()),
        "mean_dist": float(samples.mean()),
        "median_dist": float(np.median(samples)),
        "p90_dist": float(np.percentile(samples, 90)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find graph coordinate orientation against a binary pore mask.")
    parser.add_argument("--binary", type=Path, default=ROOT / "data" / "Berea_2d25um_binary.raw")
    parser.add_argument("--shape", type=parse_zyx, default=(1000, 1000, 1000))
    parser.add_argument("--origin", type=parse_zyx, default=(468, 468, 468))
    parser.add_argument("--cube-size", type=int, default=64)
    parser.add_argument("--pore-value", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--r-max", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cube = load_subcube(args.binary, args.shape, args.cube_size, args.origin)
    mask = cube == int(args.pore_value)
    print(f"mask shape: {mask.shape}, pore_fraction={mask.mean():.4f}, origin={args.origin}")

    pn = extract_porespy_openpnm_network(mask, voxel_size=1.0, sigma=args.sigma, r_max=args.r_max)
    coords = np.asarray(pn["pore.coords"], dtype=np.float32)
    print(f"pores={coords.shape[0]}, throats={len(pn['throat.conns'])}")

    results = []
    for perm in itertools.permutations((0, 1, 2)):
        for flips in itertools.product((False, True), repeat=3):
            results.append(score_orientation(coords, mask, perm, flips))

    results.sort(key=lambda row: (row["inside"], row["mean_dist"], row["median_dist"]), reverse=True)
    print("top orientations, perm maps pore.coords columns -> mask z,y,x:")
    for row in results[:12]:
        flips = "".join("1" if value else "0" for value in row["flips"])
        print(
            f"perm={row['perm']} flips={flips} "
            f"inside={row['inside']:.4f} mean_dist={row['mean_dist']:.3f} "
            f"median={row['median_dist']:.3f} p90={row['p90_dist']:.3f}"
        )

    best = results[0]
    print()
    print(f"best_perm={best['perm']} best_flips={best['flips']}")


if __name__ == "__main__":
    main()
