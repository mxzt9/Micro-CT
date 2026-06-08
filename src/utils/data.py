from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import generate_binary_structure, label
from torch.utils.data import Dataset, Sampler


DEFAULT_CUBE_SIZES = (64, 128, 192)
REQUIRED_INDEX_COLUMNS = {"z", "y", "x", "split"}


@dataclass(frozen=True)
class RockVolumeSpec:
    name: str
    data_dir: Path
    index_dir: Path
    gray_path: Path
    binary_path: Path
    shape: tuple[int, int, int]


def _as_cube_sizes(cube_size: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(cube_size, int):
        sizes = (cube_size,)
    else:
        sizes = tuple(int(size) for size in cube_size)
    if not sizes:
        raise ValueError("cube_size must contain at least one size")
    if any(size <= 0 for size in sizes):
        raise ValueError("cube_size values must be positive")
    return sizes


def _shape_for_rock(
    shape: tuple[int, int, int] | dict[str, tuple[int, int, int]],
    rock_name: str,
) -> tuple[int, int, int]:
    if isinstance(shape, dict):
        if rock_name not in shape:
            raise KeyError(f"shape for rock '{rock_name}' was not provided")
        return tuple(int(v) for v in shape[rock_name])
    return tuple(int(v) for v in shape)


def _find_raw_file(data_dir: Path, rock_name: str, kind: str, use_raw_gray: bool) -> Path | None:
    if kind == "binary":
        names = (
            f"{rock_name}_binary.raw",
            "binary.raw",
            "segmented.raw",
            "mask.raw",
            "Berea_2d25um_binary.raw",
        )
        patterns = ("*binary*.raw", "*segmented*.raw", "*mask*.raw")
    elif use_raw_gray:
        names = (
            f"{rock_name}_grayscale.raw",
            f"{rock_name}_gray.raw",
            "grayscale.raw",
            "gray.raw",
            "Berea_2d25um_grayscale.raw",
        )
        patterns = ("*grayscale*.raw", "*gray*.raw")
    else:
        names = (
            f"{rock_name}_grayscale_filtered.raw",
            f"{rock_name}_gray_filtered.raw",
            "grayscale_filtered.raw",
            "gray_filtered.raw",
            "Berea_2d25um_grayscale_filtered.raw",
        )
        patterns = ("*grayscale*filtered*.raw", "*gray*filtered*.raw", "*grayscale*.raw", "*gray*.raw")

    for name in names:
        path = data_dir / name
        if path.exists():
            return path

    for pattern in patterns:
        matches = sorted(path for path in data_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def _legacy_index_dir(root_dir: Path) -> Path | None:
    path = root_dir / "dataset_128"
    return path if path.exists() else None


def _index_dir_for_rock(root_dir: Path, index_root: Path, rock_name: str, data_dir: Path) -> Path:
    candidates = (
        index_root / rock_name,
        root_dir / f"dataset_{rock_name}",
        data_dir,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return index_root / rock_name


def discover_rock_volumes(
    root_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    index_root: str | Path | None = None,
    rocks: Sequence[str] | None = None,
    shape: tuple[int, int, int] | dict[str, tuple[int, int, int]] = (1000, 1000, 1000),
    use_raw_gray: bool = False,
) -> list[RockVolumeSpec]:
    """Discover rock volumes from data folders.

    Supported layouts:
    - legacy: data/Berea_*.raw + dataset_128/index_128.csv
    - multi-rock: data/<rock>/*.raw + datasets/<rock>/index_<cube_size>.csv
    """

    root = Path(root_dir)
    data_base = Path(data_root) if data_root is not None else root / "data"
    index_base = Path(index_root) if index_root is not None else root / "datasets"
    requested = set(rocks) if rocks is not None else None

    specs: list[RockVolumeSpec] = []

    if data_base.exists():
        direct_gray = _find_raw_file(data_base, "Berea", "gray", use_raw_gray)
        direct_binary = _find_raw_file(data_base, "Berea", "binary", use_raw_gray)
        if direct_gray is not None and direct_binary is not None and (requested is None or "Berea" in requested):
            legacy_index = _legacy_index_dir(root)
            specs.append(
                RockVolumeSpec(
                    name="Berea",
                    data_dir=data_base,
                    index_dir=legacy_index or _index_dir_for_rock(root, index_base, "Berea", data_base),
                    gray_path=direct_gray,
                    binary_path=direct_binary,
                    shape=_shape_for_rock(shape, "Berea"),
                )
            )

        for data_dir in sorted(path for path in data_base.iterdir() if path.is_dir()):
            rock_name = data_dir.name
            if requested is not None and rock_name not in requested:
                continue
            gray_path = _find_raw_file(data_dir, rock_name, "gray", use_raw_gray)
            binary_path = _find_raw_file(data_dir, rock_name, "binary", use_raw_gray)
            if gray_path is None or binary_path is None:
                continue
            specs.append(
                RockVolumeSpec(
                    name=rock_name,
                    data_dir=data_dir,
                    index_dir=_index_dir_for_rock(root, index_base, rock_name, data_dir),
                    gray_path=gray_path,
                    binary_path=binary_path,
                    shape=_shape_for_rock(shape, rock_name),
                )
            )

    if requested is not None:
        found = {spec.name for spec in specs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"rock volumes were not found for: {missing}")
    if not specs:
        raise FileNotFoundError(
            f"no rock volumes found in {data_base}. Expected legacy data/*.raw or data/<rock>/*.raw"
        )
    return specs


def resolve_patch_index_path(index_dir: str | Path, cube_size: int, rock_name: str | None = None) -> Path | None:
    index_dir = Path(index_dir)
    names = [f"index_{cube_size}.csv"]
    if rock_name:
        names.extend(
            (
                f"{rock_name}_index_{cube_size}.csv",
                f"index_{rock_name}_{cube_size}.csv",
                f"{rock_name}_{cube_size}.csv",
            )
        )
    names.append("index.csv")

    for name in names:
        path = index_dir / name
        if path.exists():
            return path
    return None


def build_patch_index(
    shape: tuple[int, int, int],
    cube_size: int,
    *,
    stride: int | None = None,
    split: str = "train",
    val_fraction: float = 0.2,
    seed: int = 42,
    rock: str | None = None,
) -> pd.DataFrame:
    """Build a grid index for subcubes and assign train/val splits."""

    cube_size = int(cube_size)
    stride = cube_size if stride is None else int(stride)
    if cube_size <= 0 or stride <= 0:
        raise ValueError("cube_size and stride must be positive")
    if cube_size > min(shape):
        raise ValueError(f"cube_size={cube_size} does not fit shape={shape}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    starts = [list(range(0, int(dim) - cube_size + 1, stride)) for dim in shape]
    for axis, dim in enumerate(shape):
        last = int(dim) - cube_size
        if starts[axis][-1] != last:
            starts[axis].append(last)

    rows = [{"z": z, "y": y, "x": x, "split": split} for z in starts[0] for y in starts[1] for x in starts[2]]
    df = pd.DataFrame(rows)
    if val_fraction > 0 and len(df) > 0:
        rng = np.random.default_rng(seed)
        val_count = int(round(len(df) * val_fraction))
        if val_count > 0:
            val_idx = rng.choice(df.index.to_numpy(), size=val_count, replace=False)
            df.loc[val_idx, "split"] = "val"
    df["cube_size"] = cube_size
    if rock is not None:
        df["rock"] = rock
    return df


def add_aux_targets_to_index(
    df: pd.DataFrame,
    binary_volume: np.ndarray,
    cube_size: int,
    *,
    pore_value: int = 0,
) -> pd.DataFrame:
    """Add porosity and percolation labels to an index dataframe once, during preparation."""

    df = df.copy()
    porosity_values: list[float] = []
    percolation_values: list[np.ndarray] = []
    for row in df.itertuples(index=False):
        z, y, x = int(getattr(row, "z")), int(getattr(row, "y")), int(getattr(row, "x"))
        cube = binary_volume[z : z + cube_size, y : y + cube_size, x : x + cube_size]
        target = np.asarray(cube == pore_value, dtype=bool)
        porosity_values.append(float(target.mean()))
        percolation_values.append(percolation_labels(target))

    percolation = np.stack(percolation_values) if percolation_values else np.zeros((0, 3), dtype=np.float32)
    df["porosity"] = porosity_values
    df["percolates_z"] = percolation[:, 0] if len(percolation) else []
    df["percolates_y"] = percolation[:, 1] if len(percolation) else []
    df["percolates_x"] = percolation[:, 2] if len(percolation) else []
    return df


def write_patch_indices(
    root_dir: str | Path,
    *,
    cube_sizes: Sequence[int] = DEFAULT_CUBE_SIZES,
    data_root: str | Path | None = None,
    index_root: str | Path | None = None,
    rocks: Sequence[str] | None = None,
    shape: tuple[int, int, int] | dict[str, tuple[int, int, int]] = (1000, 1000, 1000),
    stride_by_size: dict[int, int] | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    use_raw_gray: bool = False,
    compute_aux_targets: bool = False,
    pore_value: int = 0,
) -> pd.DataFrame:
    """Write index_<size>.csv files for every discovered rock."""

    root = Path(root_dir)
    index_base = Path(index_root) if index_root is not None else root / "datasets"
    specs = discover_rock_volumes(
        root,
        data_root=data_root,
        index_root=index_base,
        rocks=rocks,
        shape=shape,
        use_raw_gray=use_raw_gray,
    )
    records: list[pd.DataFrame] = []
    for spec in specs:
        out_dir = index_base / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        binary = None
        if compute_aux_targets:
            binary = np.memmap(spec.binary_path, dtype=np.uint8, mode="r", shape=spec.shape)
        for size in cube_sizes:
            stride = stride_by_size.get(int(size)) if stride_by_size else None
            df = build_patch_index(
                spec.shape,
                int(size),
                stride=stride,
                val_fraction=val_fraction,
                seed=seed,
                rock=spec.name,
            )
            if compute_aux_targets and binary is not None:
                df = add_aux_targets_to_index(df, binary, int(size), pore_value=pore_value)
            path = out_dir / f"index_{int(size)}.csv"
            df.to_csv(path, index=False)
            records.append(df.assign(path=str(path)))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


class BereaPatchDataset(Dataset):
    """Patch dataset for one or many micro-CT rock volumes."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        cube_size: int | Sequence[int] = 128,
        shape: tuple[int, int, int] | dict[str, tuple[int, int, int]] = (1000, 1000, 1000),
        pore_value: int = 0,
        use_raw_gray: bool = False,
        noise_types: list[str] | None = None,
        seed: int = 42,
        data_root: str | Path | None = None,
        index_root: str | Path | None = None,
        rocks: Sequence[str] | None = None,
        balance: bool | None = None,
        balance_by: str = "rock_size",
        samples_per_group: int | None = None,
        max_samples: int | None = None,
        return_aux_targets: bool = True,
        size_sampling_weights: dict[int, float] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.cube_sizes = _as_cube_sizes(cube_size)
        self.shape = shape
        self.pore_value = pore_value
        self.rng = np.random.default_rng(seed)
        self.balance_by = balance_by
        self.samples_per_group = samples_per_group
        self.max_samples = max_samples
        self.return_aux_targets = return_aux_targets
        self.size_sampling_weights = {int(k): float(v) for k, v in (size_sampling_weights or {}).items()}
        self._aux_cache: dict[int, tuple[float, np.ndarray]] = {}

        self.specs = discover_rock_volumes(
            self.root_dir,
            data_root=data_root,
            index_root=index_root,
            rocks=rocks,
            shape=shape,
            use_raw_gray=use_raw_gray,
        )
        self.volumes = {
            spec.name: {
                "gray": np.memmap(spec.gray_path, dtype=np.uint8, mode="r", shape=spec.shape),
                "binary": np.memmap(spec.binary_path, dtype=np.uint8, mode="r", shape=spec.shape),
                "shape": spec.shape,
            }
            for spec in self.specs
        }
        self.df = self._ensure_index_metadata(self._limit_rows(self._load_index_rows(), seed=seed))
        if len(self.df) == 0:
            raise ValueError(f"no rows found for split='{split}' and cube_size={self.cube_sizes}")

        if balance is None:
            balance = split == "train"
        self.sample_index = self._build_sample_index(seed) if balance else np.arange(len(self.df))

        if noise_types is not None:
            self.noise_types = noise_types
        elif split == "train":
            self.noise_types = [
                "none",
                "gaussian_low",
                "gaussian_mid",
                "gaussian_high",
                "salt_pepper",
                "contrast_shift",
                "mixed",
            ]
        else:
            self.noise_types = ["none", "gaussian_mid", "mixed"]

    @property
    def cube_size(self) -> int:
        if len(self.cube_sizes) != 1:
            raise AttributeError("dataset has multiple cube sizes; use cube_sizes")
        return self.cube_sizes[0]

    def _load_index_rows(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for spec in self.specs:
            for size in self.cube_sizes:
                index_path = resolve_patch_index_path(spec.index_dir, size, spec.name)
                if index_path is None:
                    missing.append(f"{spec.name}:{size} ({spec.index_dir})")
                    continue
                df = pd.read_csv(index_path)
                missing_columns = REQUIRED_INDEX_COLUMNS - set(df.columns)
                if missing_columns:
                    raise ValueError(f"{index_path} is missing columns: {sorted(missing_columns)}")
                if "cube_size" in df.columns:
                    df = df[df["cube_size"].astype(int) == int(size)]
                df = df[df["split"] == self.split].copy()
                if df.empty:
                    continue
                df["rock"] = spec.name
                df["cube_size"] = int(size)
                df["index_path"] = str(index_path)
                frames.append(df)

        if missing and not frames:
            raise FileNotFoundError(
                "no patch index files were found. Missing: "
                + ", ".join(missing)
                + ". Run src/notebooks/00_prepare_data.ipynb to create them."
            )
        if missing:
            warnings.warn("some rock/size indices were skipped: " + ", ".join(missing), stacklevel=2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _limit_rows(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        if df.empty:
            return df
        rng_seed = int(seed)
        if self.samples_per_group is not None:
            n = int(self.samples_per_group)
            if n <= 0:
                raise ValueError("samples_per_group must be positive")
            df = (
                df.groupby(["rock", "cube_size"], group_keys=False, sort=True)
                .apply(lambda group: group.sample(n=min(len(group), n), random_state=rng_seed))
                .reset_index(drop=True)
            )
        if self.max_samples is not None and len(df) > int(self.max_samples):
            df = df.sample(n=int(self.max_samples), random_state=rng_seed).reset_index(drop=True)
        return df.reset_index(drop=True)

    def _ensure_index_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if "rock" not in df.columns:
            if "index_path" in df.columns:
                df["rock"] = df["index_path"].map(lambda value: Path(str(value)).parent.name)
            elif len(self.specs) == 1:
                df["rock"] = self.specs[0].name
            else:
                raise ValueError("patch index metadata is missing 'rock'; rerun 00_prepare_data.ipynb")
        if "cube_size" not in df.columns:
            if "index_path" in df.columns:
                def infer_size(value: str) -> int:
                    stem = Path(str(value)).stem
                    try:
                        return int(stem.split("_")[-1])
                    except ValueError as exc:
                        raise ValueError("patch index metadata is missing 'cube_size'") from exc

                df["cube_size"] = df["index_path"].map(infer_size)
            elif len(self.cube_sizes) == 1:
                df["cube_size"] = self.cube_sizes[0]
            else:
                raise ValueError("patch index metadata is missing 'cube_size'; rerun 00_prepare_data.ipynb")
        df["cube_size"] = df["cube_size"].astype(int)
        return df.reset_index(drop=True)

    def _balance_columns(self) -> list[str]:
        if self.balance_by == "rock":
            return ["rock"]
        if self.balance_by == "cube_size":
            return ["cube_size"]
        if self.balance_by == "rock_size":
            return ["rock", "cube_size"]
        if self.balance_by in {"none", ""}:
            return []
        raise ValueError("balance_by must be one of: rock, cube_size, rock_size, none")

    def _build_sample_index(self, seed: int) -> np.ndarray:
        requested_columns = self._balance_columns()
        columns = [column for column in requested_columns if column in self.df.columns]
        if not columns:
            return np.arange(len(self.df))
        rng = np.random.default_rng(seed)
        grouped = list(self.df.groupby(columns, sort=True))
        groups = [group.index.to_numpy() for _, group in grouped]
        max_len = max(len(group) for group in groups)
        balanced = []
        if self.size_sampling_weights and "cube_size" in self.df.columns:
            raw_weights = []
            for _, group_df in grouped:
                size = int(group_df["cube_size"].iloc[0])
                raw_weights.append(max(self.size_sampling_weights.get(size, 1.0), 0.0))
            weights = np.asarray(raw_weights, dtype=np.float64)
            if weights.sum() <= 0:
                raise ValueError("size_sampling_weights must contain at least one positive weight")
            weights = weights / weights.mean()
            target_counts = np.maximum(np.rint(weights * max_len).astype(int), 1)
            for group, target_count in zip(groups, target_counts):
                if len(group) < target_count:
                    group = rng.choice(group, size=int(target_count), replace=True)
                elif len(group) > target_count:
                    group = rng.choice(group, size=int(target_count), replace=False)
                balanced.extend(group.tolist())
        else:
            for group in groups:
                if len(group) < max_len:
                    group = rng.choice(group, size=max_len, replace=True)
                balanced.extend(group.tolist())
        balanced = np.asarray(balanced, dtype=np.int64)
        rng.shuffle(balanced)
        return balanced

    def __len__(self) -> int:
        return len(self.sample_index)

    @staticmethod
    def normalize_uint8(cube: np.ndarray) -> np.ndarray:
        normalized = cube.astype(np.float32, copy=True)
        normalized *= 1.0 / 255.0
        return normalized

    def add_noise(self, cube: np.ndarray, noise_type: str) -> np.ndarray:
        if noise_type == "none":
            return cube if cube.dtype == np.float32 else cube.astype(np.float32, copy=True)

        img = cube.astype(np.float32, copy=True)
        should_clip = True

        if noise_type == "gaussian_low":
            img += self.rng.normal(0.0, 0.03, size=img.shape).astype(np.float32)
        elif noise_type == "gaussian_mid":
            img += self.rng.normal(0.0, 0.06, size=img.shape).astype(np.float32)
        elif noise_type == "gaussian_high":
            img += self.rng.normal(0.0, 0.10, size=img.shape).astype(np.float32)
        elif noise_type == "salt_pepper":
            prob = 0.02
            mask = self.rng.random(img.shape)
            img[mask < prob / 2] = 0.0
            img[(mask >= prob / 2) & (mask < prob)] = 1.0
            should_clip = False
        elif noise_type == "contrast_shift":
            img *= float(self.rng.uniform(0.75, 1.25))
            img += float(self.rng.uniform(-0.10, 0.10))
        elif noise_type == "mixed":
            img += self.rng.normal(0.0, 0.05, size=img.shape).astype(np.float32)
            img *= float(self.rng.uniform(0.8, 1.2))
            img += float(self.rng.uniform(-0.08, 0.08))
            prob = 0.01
            mask = self.rng.random(img.shape)
            img[mask < prob / 2] = 0.0
            img[(mask >= prob / 2) & (mask < prob)] = 1.0
        else:
            raise ValueError(f"unknown noise type: {noise_type}")

        if should_clip:
            np.clip(img, 0.0, 1.0, out=img)
        return img

    def get_cube(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[int(self.sample_index[idx])]
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        cs = int(row["cube_size"])
        rock = str(row["rock"])
        volume = self.volumes[rock]
        shape = volume["shape"]
        if z < 0 or y < 0 or x < 0 or z + cs > shape[0] or y + cs > shape[1] or x + cs > shape[2]:
            raise IndexError(f"cube {rock}:{(z, y, x)} size={cs} is outside shape={shape}")
        gray_cube = volume["gray"][z : z + cs, y : y + cs, x : x + cs]
        binary_cube = volume["binary"][z : z + cs, y : y + cs, x : x + cs]
        return {"gray": gray_cube, "binary": binary_cube, "coord": (z, y, x), "rock": rock, "cube_size": cs}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cube = self.get_cube(idx)
        clean_x = self.normalize_uint8(cube["gray"])
        noise_type = self.rng.choice(self.noise_types)
        noisy_x = self.add_noise(clean_x, str(noise_type))
        target = (cube["binary"] == self.pore_value).astype(np.float32)
        row_idx = int(self.sample_index[idx])
        row = self.df.iloc[row_idx]
        if self.return_aux_targets and {"porosity", "percolates_z", "percolates_y", "percolates_x"}.issubset(self.df.columns):
            porosity = float(row["porosity"])
            percolates = np.asarray(
                [row["percolates_z"], row["percolates_y"], row["percolates_x"]],
                dtype=np.float32,
            )
        elif self.return_aux_targets:
            if row_idx in self._aux_cache:
                porosity, percolates = self._aux_cache[row_idx]
            else:
                porosity = float(target.mean())
                percolates = percolation_labels(target > 0.5)
                self._aux_cache[row_idx] = (porosity, percolates)
        else:
            porosity = 0.0
            percolates = np.zeros(3, dtype=np.float32)

        return {
            "x": torch.from_numpy(noisy_x).unsqueeze(0),
            "y": torch.from_numpy(target).unsqueeze(0),
            "coord": torch.tensor(cube["coord"], dtype=torch.long),
            "rock": cube["rock"],
            "cube_size": torch.tensor(cube["cube_size"], dtype=torch.long),
            "porosity": torch.tensor(porosity, dtype=torch.float32),
            "percolates": torch.tensor(percolates, dtype=torch.float32),
            "noise": str(noise_type),
        }


class CubeSizeBatchSampler(Sampler[list[int]]):
    """Yield batches whose samples all have the same cube_size."""

    def __init__(
        self,
        dataset: BereaPatchDataset,
        batch_size: int | dict[int, int],
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if not hasattr(dataset, "df") or not hasattr(dataset, "sample_index"):
            raise TypeError("CubeSizeBatchSampler expects a BereaPatchDataset-like object")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if isinstance(batch_size, dict):
            self.batch_size_by_size = {int(size): int(value) for size, value in batch_size.items()}
            if any(value <= 0 for value in self.batch_size_by_size.values()):
                raise ValueError("batch_size values must be positive")
        else:
            if int(batch_size) <= 0:
                raise ValueError("batch_size must be positive")
            self.batch_size_by_size = {}
            self.default_batch_size = int(batch_size)

        effective = dataset.df.iloc[dataset.sample_index].reset_index(drop=True)
        self.indices_by_size = {
            int(size): group.index.to_numpy(dtype=np.int64)
            for size, group in effective.groupby("cube_size", sort=True)
        }
        if not self.indices_by_size:
            raise ValueError("dataset has no samples to batch")

        unknown_sizes = set(self.indices_by_size) - set(self.batch_size_by_size)
        if isinstance(batch_size, dict) and unknown_sizes:
            raise ValueError(f"batch_size is missing cube sizes: {sorted(unknown_sizes)}")

    def _batch_size_for(self, cube_size: int) -> int:
        if int(cube_size) in self.batch_size_by_size:
            return self.batch_size_by_size[int(cube_size)]
        return self.default_batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for size, indices in self.indices_by_size.items():
            indices = indices.copy()
            if self.shuffle:
                rng.shuffle(indices)
            batch_size = self._batch_size_for(size)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size].tolist()
                if len(batch) == batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for size, indices in self.indices_by_size.items():
            batch_size = self._batch_size_for(size)
            full, remainder = divmod(len(indices), batch_size)
            total += full
            if remainder and not self.drop_last:
                total += 1
        return total


def percolation_labels(mask: np.ndarray) -> np.ndarray:
    """Return [percolates_z, percolates_y, percolates_x] for a binary pore mask."""

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(3, dtype=np.float32)

    structure = generate_binary_structure(rank=3, connectivity=1)
    labels, _ = label(mask, structure=structure)
    result = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        low = np.take(labels, 0, axis=axis)
        high = np.take(labels, labels.shape[axis] - 1, axis=axis)
        low_ids = set(np.unique(low[low > 0]).tolist())
        high_ids = set(np.unique(high[high > 0]).tolist())
        result[axis] = float(bool(low_ids & high_ids))
    return result


class MultiScaleNoiseConsistencyDataset(BereaPatchDataset):
    """Return centered multi-scale noisy views of the same rock patch."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        view_cube_sizes: Sequence[int] = DEFAULT_CUBE_SIZES,
        view_noise_types: Sequence[str] | None = None,
        **kwargs: Any,
    ):
        self.view_cube_sizes = _as_cube_sizes(view_cube_sizes)
        self.view_noise_types = tuple(view_noise_types) if view_noise_types is not None else None
        kwargs.pop("cube_size", None)
        super().__init__(root_dir, split=split, cube_size=max(self.view_cube_sizes), **kwargs)

    @staticmethod
    def _center_crop(cube: np.ndarray, size: int) -> np.ndarray:
        starts = [(dim - size) // 2 for dim in cube.shape]
        z, y, x = starts
        return cube[z : z + size, y : y + size, x : x + size]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cube = self.get_cube(idx)
        clean = self.normalize_uint8(cube["gray"])
        target_full = (cube["binary"] == self.pore_value).astype(np.float32)

        x_views = []
        y_views = []
        percolation_views = []
        porosity_views = []
        noise_views = []
        for size in self.view_cube_sizes:
            clean_view = self._center_crop(clean, size)
            target_view = self._center_crop(target_full, size)
            noise_pool = self.view_noise_types or self.noise_types
            noise_type = str(self.rng.choice(noise_pool))
            x_view = self.add_noise(clean_view, noise_type)
            x_views.append(torch.from_numpy(x_view.copy()).float().unsqueeze(0))
            y_views.append(torch.from_numpy(target_view.copy()).float().unsqueeze(0))
            porosity_views.append(float(target_view.mean()))
            percolation_views.append(percolation_labels(target_view > 0.5))
            noise_views.append(noise_type)

        return {
            "x_views": x_views,
            "y_views": y_views,
            "coord": torch.tensor(cube["coord"], dtype=torch.long),
            "rock": cube["rock"],
            "cube_sizes": torch.tensor(self.view_cube_sizes, dtype=torch.long),
            "porosity_views": torch.tensor(porosity_views, dtype=torch.float32),
            "percolation_views": torch.tensor(np.stack(percolation_views), dtype=torch.float32),
            "noise_views": noise_views,
        }


MultiScalePatchDataset = MultiScaleNoiseConsistencyDataset
MultiRockPatchDataset = BereaPatchDataset
BereaSegmentationDataset = BereaPatchDataset
