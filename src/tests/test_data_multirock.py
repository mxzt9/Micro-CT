from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    DEFAULT_CUBE_SIZES,
    BereaPatchDataset,
    MultiScaleNoiseConsistencyDataset,
    build_patch_index,
    discover_rock_volumes,
    percolation_labels,
    write_patch_indices,
)


def _write_raw_pair(root, rock: str, shape: tuple[int, int, int]) -> None:
    data_dir = root / "data" / rock
    data_dir.mkdir(parents=True, exist_ok=True)
    gray = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    binary = (gray % 2).astype(np.uint8)
    gray.tofile(data_dir / "grayscale_filtered.raw")
    binary.tofile(data_dir / "binary.raw")


def test_multirock_dataset_balances_rocks_and_cube_sizes(tmp_path):
    shape = (8, 8, 8)
    for rock in ("sandstone", "limestone"):
        _write_raw_pair(tmp_path, rock, shape)
        (tmp_path / "datasets" / rock).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"z": 0, "y": 0, "x": 0, "split": "train"},
            {"z": 0, "y": 0, "x": 4, "split": "train"},
        ]
    ).to_csv(tmp_path / "datasets" / "sandstone" / "index_4.csv", index=False)
    pd.DataFrame([{"z": 0, "y": 0, "x": 0, "split": "train"}]).to_csv(
        tmp_path / "datasets" / "sandstone" / "index_6.csv",
        index=False,
    )
    pd.DataFrame([{"z": 0, "y": 0, "x": 0, "split": "train"}]).to_csv(
        tmp_path / "datasets" / "limestone" / "index_4.csv",
        index=False,
    )
    pd.DataFrame([{"z": 0, "y": 0, "x": 0, "split": "train"}]).to_csv(
        tmp_path / "datasets" / "limestone" / "index_6.csv",
        index=False,
    )

    dataset = BereaPatchDataset(
        tmp_path,
        split="train",
        cube_size=[4, 6],
        shape=shape,
        noise_types=["none"],
        balance=True,
    )

    effective = dataset.df.iloc[dataset.sample_index]
    counts = effective.groupby(["rock", "cube_size"]).size().to_dict()
    assert set(counts.values()) == {2}
    assert len(dataset) == 8

    sample = dataset[0]
    assert sample["x"].shape[0] == 1
    assert sample["x"].shape == sample["y"].shape
    assert sample["rock"] in {"sandstone", "limestone"}
    assert int(sample["cube_size"]) in {4, 6}


def test_write_patch_indices_discovers_rocks_and_writes_each_size(tmp_path):
    shape = (8, 8, 8)
    for rock in ("sandstone", "limestone"):
        _write_raw_pair(tmp_path, rock, shape)

    summary = write_patch_indices(tmp_path, cube_sizes=[4, 6], shape=shape, val_fraction=0.25, seed=0)

    assert {spec.name for spec in discover_rock_volumes(tmp_path, shape=shape)} == {"sandstone", "limestone"}
    assert (tmp_path / "datasets" / "sandstone" / "index_4.csv").exists()
    assert (tmp_path / "datasets" / "limestone" / "index_6.csv").exists()
    assert set(summary["rock"]) == {"sandstone", "limestone"}
    assert set(summary["cube_size"]) == {4, 6}


def test_build_patch_index_includes_last_start_for_non_divisible_stride():
    df = build_patch_index((10, 10, 10), 6, stride=4, val_fraction=0.0)

    assert df[["z", "y", "x"]].max().tolist() == [4, 4, 4]
    assert set(df["split"]) == {"train"}


def test_default_cube_sizes_are_pooling_safe_and_percolation_labels():
    assert DEFAULT_CUBE_SIZES == (64, 128, 192)
    assert all(size % 8 == 0 for size in DEFAULT_CUBE_SIZES)

    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[:, 3, 3] = True
    assert percolation_labels(mask).tolist() == [1.0, 0.0, 0.0]


def test_multiscale_consistency_dataset_returns_centered_views(tmp_path):
    shape = (8, 8, 8)
    _write_raw_pair(tmp_path, "sandstone", shape)
    index_dir = tmp_path / "datasets" / "sandstone"
    index_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"z": 0, "y": 0, "x": 0, "split": "train"}]).to_csv(index_dir / "index_6.csv", index=False)

    dataset = MultiScaleNoiseConsistencyDataset(
        tmp_path,
        split="train",
        view_cube_sizes=[4, 6],
        shape=shape,
        noise_types=["none"],
        balance=False,
    )
    sample = dataset[0]

    assert [tuple(view.shape[-3:]) for view in sample["x_views"]] == [(4, 4, 4), (6, 6, 6)]
    assert sample["cube_sizes"].tolist() == [4, 6]
    assert sample["porosity_views"].shape == (2,)
    assert sample["percolation_views"].shape == (2, 3)


def test_dataset_recovers_missing_rock_metadata_from_index_path(tmp_path):
    shape = (8, 8, 8)
    _write_raw_pair(tmp_path, "sandstone", shape)
    index_dir = tmp_path / "datasets" / "sandstone"
    index_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"z": 0, "y": 0, "x": 0, "split": "train", "cube_size": 4},
            {"z": 0, "y": 0, "x": 4, "split": "train", "cube_size": 4},
        ]
    ).to_csv(index_dir / "index_4.csv", index=False)

    dataset = BereaPatchDataset(
        tmp_path,
        split="train",
        cube_size=[4],
        shape=shape,
        noise_types=["none"],
        balance=True,
    )

    assert "rock" in dataset.df.columns
    assert dataset.df["rock"].unique().tolist() == ["sandstone"]


def test_dataset_uses_cached_aux_targets_from_index(tmp_path):
    shape = (8, 8, 8)
    _write_raw_pair(tmp_path, "sandstone", shape)
    index_dir = tmp_path / "datasets" / "sandstone"
    index_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "z": 0,
                "y": 0,
                "x": 0,
                "split": "train",
                "cube_size": 4,
                "rock": "sandstone",
                "porosity": 0.25,
                "percolates_z": 1.0,
                "percolates_y": 0.0,
                "percolates_x": 1.0,
            }
        ]
    ).to_csv(index_dir / "index_4.csv", index=False)

    dataset = BereaPatchDataset(tmp_path, split="train", cube_size=4, shape=shape, noise_types=["none"], balance=False)
    sample = dataset[0]

    assert float(sample["porosity"]) == 0.25
    assert sample["percolates"].tolist() == [1.0, 0.0, 1.0]


def test_size_sampling_weights_reduce_expensive_size(tmp_path):
    shape = (8, 8, 8)
    _write_raw_pair(tmp_path, "sandstone", shape)
    index_dir = tmp_path / "datasets" / "sandstone"
    index_dir.mkdir(parents=True, exist_ok=True)
    for size in (4, 6):
        pd.DataFrame(
            [
                {"z": 0, "y": 0, "x": 0, "split": "train", "cube_size": size, "rock": "sandstone"},
                {"z": 0, "y": 0, "x": 1, "split": "train", "cube_size": size, "rock": "sandstone"},
                {"z": 0, "y": 1, "x": 0, "split": "train", "cube_size": size, "rock": "sandstone"},
                {"z": 1, "y": 0, "x": 0, "split": "train", "cube_size": size, "rock": "sandstone"},
            ]
        ).to_csv(index_dir / f"index_{size}.csv", index=False)

    dataset = BereaPatchDataset(
        tmp_path,
        split="train",
        cube_size=[4, 6],
        shape=shape,
        noise_types=["none"],
        balance=True,
        size_sampling_weights={4: 1.0, 6: 0.25},
    )
    effective = dataset.df.iloc[dataset.sample_index]

    counts = effective.groupby("cube_size").size().to_dict()
    assert counts[4] > counts[6]
