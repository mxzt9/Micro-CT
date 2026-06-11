from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import generate_binary_structure, label
from torch.utils.data import Dataset, Sampler

from tqdm import tqdm

from .topology import cubical_persistence_summary


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

    # ПРОСТРАНСТВЕННЫЙ train/val split (вместо случайного).
    #
    # Раньше val-патчи выбирались случайно внутри того же объёма. Это давало
    # утечку: (а) соседние патчи сильно коррелированы; (б) кубы разных
    # размеров перекрываются — val-куб 64³ мог целиком лежать внутри
    # train-куба 192³; (в) «last start» патчи перекрываются с соседними.
    # Из-за этого val dice был завышен.
    #
    # Теперь val — это непрерывный слой объёма в конце оси z, единый для
    # ВСЕХ cube_size (граница не зависит от cube_size). Патч относится к val,
    # если он ЦЕЛИКОМ лежит в val-зоне; патчи, пересекающие границу,
    # отбрасываются (буферная зона, исключающая частичное перекрытие).
    if val_fraction > 0 and len(df) > 0:
        depth = int(shape[0])
        val_boundary = int(round(depth * (1.0 - val_fraction)))
        is_val = df["z"] >= val_boundary
        crosses = (~is_val) & (df["z"] + cube_size > val_boundary)
        df.loc[is_val, "split"] = "val"
        df = df.loc[~crosses].reset_index(drop=True)
    df["cube_size"] = cube_size
    if rock is not None:
        df["rock"] = rock
    return df


def percolation_labels(mask: np.ndarray) -> np.ndarray:
    """Return [percolates_z, percolates_y, percolates_x] for a binary pore mask.

    Реализация через scipy.ndimage.label (C-код): одна маркировка компонент
    на все три оси, затем проверка пересечения множеств меток на
    противоположных гранях.

    Прежний «быстрый BFS с ранним выходом» на чистом Python в худшем случае
    (нет перколяции — а это частый случай для 64³) обходил всю компоненту
    по вокселю за итерацию и был на порядки медленнее ndimage.label,
    вопреки заявленному в докстринге «~50x быстрее».
    """
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros(3, dtype=np.float32)
    if not mask.any():
        return result

    structure = generate_binary_structure(3, 1)  # 6-связность
    labels, num = label(mask, structure=structure)
    if num == 0:
        return result

    for axis in range(3):
        slices_low = [slice(None)] * 3
        slices_high = [slice(None)] * 3
        slices_low[axis] = 0
        slices_high[axis] = -1
        low_labels = np.unique(labels[tuple(slices_low)])
        high_labels = np.unique(labels[tuple(slices_high)])
        low_labels = low_labels[low_labels > 0]
        high_labels = high_labels[high_labels > 0]
        if low_labels.size and high_labels.size and np.intersect1d(low_labels, high_labels, assume_unique=True).size:
            result[axis] = 1.0

    return result


# Обратная совместимость со старым именем
percolation_labels_fast = percolation_labels


def add_aux_targets_to_index(
    df: pd.DataFrame,
    binary_volume: np.ndarray,
    cube_size: int,
    *,
    pore_value: int = 0,
    num_workers: int = 0,
    chunk_memory_mb: int = 256,
) -> pd.DataFrame:
    """Add porosity and percolation labels to an index dataframe once, during preparation.

    Кубы обрабатываются ПОТОКОВО, чанками с ограничением по памяти: в RAM
    одновременно живёт не больше ~chunk_memory_mb мегабайт кубов (плюс их
    копии в очередях воркеров на время чанка). Прежняя реализация сначала
    материализовала ВСЕ кубы списком и разом отправляла их в futures —
    на полном объёме это съедало десятки ГБ и роняло процесс по OOM.

    Args:
        df: index dataframe
        binary_volume: memmap или ndarray
        cube_size: размер куба
        pore_value: значение пор в binary (0 по умолчанию)
        num_workers: число процессов для параллельной перколяции (0 = последовательно)
        chunk_memory_mb: бюджет RAM на один чанк кубов, МБ
    """
    df = df.copy()
    n = len(df)
    porosity_values = np.zeros(n, dtype=np.float32)
    percolation_values = np.zeros((n, 3), dtype=np.float32)
    if n == 0:
        df["porosity"] = porosity_values
        df["percolates_z"] = percolation_values[:, 0]
        df["percolates_y"] = percolation_values[:, 1]
        df["percolates_x"] = percolation_values[:, 2]
        return df

    coords = [
        (int(getattr(row, "z")), int(getattr(row, "y")), int(getattr(row, "x")))
        for row in df.itertuples(index=False)
    ]
    bytes_per_cube = int(cube_size) ** 3  # bool = 1 байт/воксель
    chunk_size = max(1, int(chunk_memory_mb * 1024 * 1024 // bytes_per_cube))

    def _read_cube(i: int) -> np.ndarray:
        z, y, x = coords[i]
        cube = binary_volume[z : z + cube_size, y : y + cube_size, x : x + cube_size]
        return np.asarray(cube == pore_value, dtype=bool)

    executor = None
    if num_workers > 0 and n > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(max_workers=num_workers)

    desc = f"  aux targets {cube_size}\u00b3" + (f" ({num_workers} workers)" if executor else "")
    pbar = tqdm(total=n, desc=desc, leave=False)
    try:
        for start in range(0, n, chunk_size):
            idxs = list(range(start, min(start + chunk_size, n)))
            cubes = [_read_cube(i) for i in idxs]
            for i, cube in zip(idxs, cubes):
                porosity_values[i] = float(cube.mean())
            if executor is not None:
                map_chunksize = max(1, len(cubes) // (num_workers * 4))
                for i, result in zip(idxs, executor.map(percolation_labels, cubes, chunksize=map_chunksize)):
                    percolation_values[i] = result
                    pbar.update(1)
            else:
                for i, cube in zip(idxs, cubes):
                    percolation_values[i] = percolation_labels(cube)
                    pbar.update(1)
            del cubes  # освобождаем чанк до чтения следующего
    finally:
        if executor is not None:
            executor.shutdown()
        pbar.close()

    df["porosity"] = porosity_values
    df["percolates_z"] = percolation_values[:, 0]
    df["percolates_y"] = percolation_values[:, 1]
    df["percolates_x"] = percolation_values[:, 2]
    return df


AUX_COLUMNS = {"porosity", "percolates_z", "percolates_y", "percolates_x"}


def _csv_has_aux_columns(path: Path) -> bool:
    """Проверяет, содержит ли CSV все колонки aux_targets."""
    if not path.exists():
        return False
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        return AUX_COLUMNS.issubset(header)
    except Exception:
        return False


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
    num_workers: int = 0,
    force: bool = False,
) -> pd.DataFrame:
    """Write index_<size>.csv files for every discovered rock.

    Args:
        root_dir: корень проекта
        cube_sizes: список размеров кубов
        data_root: путь к data/ (по умолчанию root/data)
        index_root: путь к datasets/ (по умолчанию root/datasets)
        rocks: список пород (None = все)
        shape: размеры воксельного объема
        stride_by_size: stride для каждого размера куба
        val_fraction: доля валидации
        seed: seed для random split
        use_raw_gray: использовать сырые grayscale
        compute_aux_targets: вычислять porosity/percolation
        pore_value: значение поры в binary
        num_workers: число процессов для parallel percolation (0 = последовательно)
        force: перезаписывать существующие CSV даже если aux_targets уже есть
    """
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
            path = out_dir / f"index_{int(size)}.csv"

            # Пропускаем, если CSV уже существует и содержит все нужные колонки
            need_aux = compute_aux_targets
            if not force and path.exists():
                if need_aux:
                    if _csv_has_aux_columns(path):
                        records.append(pd.read_csv(path).assign(path=str(path)))
                        continue
                else:
                    # CSV есть, aux не нужны — пропускаем
                    records.append(pd.read_csv(path).assign(path=str(path)))
                    continue

            stride = stride_by_size.get(int(size)) if stride_by_size else None
            df = build_patch_index(
                spec.shape,
                int(size),
                stride=stride,
                val_fraction=val_fraction,
                seed=seed,
                rock=spec.name,
            )
            if need_aux and binary is not None:
                df = add_aux_targets_to_index(df, binary, int(size), pore_value=pore_value, num_workers=num_workers)
            df.to_csv(path, index=False)
            records.append(df.assign(path=str(path)))
        del binary  # закрыть memmap сразу, не держать страницы/хэндл до конца функции
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
        return_topology: bool = False,
        topology_cache_dir: str | Path | None = None,
        topology_max_size: int | None = 32,
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
        self.return_topology = bool(return_topology)
        self.topology_cache_dir = Path(topology_cache_dir) if topology_cache_dir is not None else self.root_dir / "outputs" / "topology_cache"
        self.topology_max_size = topology_max_size
        self._topology_memory_cache: dict[tuple[str, str, int, int, int, int, int | None], np.ndarray] = {}

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
                + ". Run Section 1 (Prepare Data) in src/full_pipeline.ipynb to create them."
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
                raise ValueError("patch index metadata is missing 'rock'; rerun Section 1 in full_pipeline.ipynb")
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
                raise ValueError("patch index metadata is missing 'cube_size'; rerun Section 1 in full_pipeline.ipynb")
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

    def _topology_cache_path(
        self,
        *,
        rock: str,
        source: str,
        cube_size: int,
        coord: tuple[int, int, int],
    ) -> Path:
        z, y, x = coord
        max_label = "full" if self.topology_max_size is None else str(int(self.topology_max_size))
        name = f"cs{cube_size}_z{z}_y{y}_x{x}_{source}_m{max_label}.npy"
        return self.topology_cache_dir / rock / name

    def _cached_topology_summary(
        self,
        volume: np.ndarray,
        *,
        rock: str,
        source: str,
        cube_size: int,
        coord: tuple[int, int, int],
    ) -> np.ndarray:
        key = (rock, source, int(cube_size), int(coord[0]), int(coord[1]), int(coord[2]), self.topology_max_size)
        if key in self._topology_memory_cache:
            return self._topology_memory_cache[key]

        path = self._topology_cache_path(rock=rock, source=source, cube_size=cube_size, coord=coord)
        if path.exists():
            value = np.load(path).astype(np.float32, copy=False)
            self._topology_memory_cache[key] = value
            return value

        # Keep PH input honest: source="raw" is derived only from grayscale, while
        # source="target" may use binary labels and must be used only as a loss target.
        value = cubical_persistence_summary(volume, max_size=self.topology_max_size).astype(np.float32, copy=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.{id(value)}.tmp.npy")
        np.save(tmp_path, value)
        tmp_path.replace(path)
        self._topology_memory_cache[key] = value
        return value

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

        sample = {
            "x": torch.from_numpy(noisy_x).unsqueeze(0),
            "y": torch.from_numpy(target).unsqueeze(0),
            "coord": torch.tensor(cube["coord"], dtype=torch.long),
            "rock": cube["rock"],
            "cube_size": torch.tensor(cube["cube_size"], dtype=torch.long),
            "porosity": torch.tensor(porosity, dtype=torch.float32),
            "percolates": torch.tensor(percolates, dtype=torch.float32),
            "noise": str(noise_type),
        }

        if self.return_topology:
            # PH рассчитывается из ЧИСТОГО grayscale (clean_x).
            #
            # Раньше PH считался из noisy_x с комментарием про «согласованность
            # train/inference», но это не работало: (а) ключ кэша не включал
            # тип/реализацию шума, поэтому с первой эпохи замораживалось PH
            # одной случайной реализации и переиспользовалось для любых входов;
            # (б) precompute_topology_cache.py считает raw-PH от чистого
            # grayscale — предзаполненный кэш всё равно подменял noisy-PH на
            # clean-PH. Детерминированный clean-PH делает кэш корректным и
            # согласован со скриптом precompute и c инференсом.
            ph_features = self._cached_topology_summary(
                clean_x,
                rock=str(cube["rock"]),
                source="raw",
                cube_size=int(cube["cube_size"]),
                coord=tuple(int(v) for v in cube["coord"]),
            )
            topology_target = self._cached_topology_summary(
                target > 0.5,
                rock=str(cube["rock"]),
                source="target",
                cube_size=int(cube["cube_size"]),
                coord=tuple(int(v) for v in cube["coord"]),
            )
            sample["ph_features"] = torch.tensor(ph_features, dtype=torch.float32)
            sample["topology_target"] = torch.tensor(topology_target, dtype=torch.float32)

        return sample


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