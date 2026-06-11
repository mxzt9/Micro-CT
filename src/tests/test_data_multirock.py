from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils.data as data_module
from utils import (
    DEFAULT_CUBE_SIZES,
    BereaPatchDataset,
    CubeSizeBatchSampler,
    MultiScaleNoiseConsistencyDataset,
    build_patch_index,
    discover_rock_volumes,
    dice_score_from_logits,
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


def test_dice_score_from_logits_averages_per_sample_not_voxels():
    logits = torch.tensor(
        [
            [[[[10.0, -10.0]]]],
            [[[[10.0, -10.0]]]],
        ]
    )
    targets = torch.tensor(
        [
            [[[[1.0, 0.0]]]],
            [[[[0.0, 1.0]]]],
        ]
    )

    assert torch.isclose(dice_score_from_logits(logits, targets), torch.tensor(0.5))


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


def test_dataset_topology_features_are_optional_and_cached(tmp_path, monkeypatch):
    shape = (8, 8, 8)
    _write_raw_pair(tmp_path, "sandstone", shape)
    index_dir = tmp_path / "datasets" / "sandstone"
    index_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"z": 0, "y": 0, "x": 0, "split": "train", "cube_size": 4, "rock": "sandstone"}]
    ).to_csv(index_dir / "index_4.csv", index=False)

    calls = []

    def fake_summary(volume, *, max_size):
        calls.append((tuple(volume.shape), max_size, bool(np.asarray(volume).dtype == np.bool_)))
        base = float(np.asarray(volume, dtype=np.float32).mean())
        return np.asarray([1.0, base, 0.5, 2.0, base + 1.0, 0.25], dtype=np.float32)

    monkeypatch.setattr(data_module, "cubical_persistence_summary", fake_summary)

    plain = BereaPatchDataset(tmp_path, split="train", cube_size=4, shape=shape, noise_types=["none"], balance=False)
    plain_sample = plain[0]
    assert "ph_features" not in plain_sample
    assert "topology_target" not in plain_sample

    dataset = BereaPatchDataset(
        tmp_path,
        split="train",
        cube_size=4,
        shape=shape,
        noise_types=["none"],
        balance=False,
        return_topology=True,
        topology_cache_dir=tmp_path / "topology_cache",
        topology_max_size=4,
    )

    first = dataset[0]
    second = dataset[0]

    assert first["ph_features"].shape == (6,)
    assert first["topology_target"].shape == (6,)
    assert torch.equal(first["ph_features"], second["ph_features"])
    assert torch.equal(first["topology_target"], second["topology_target"])
    assert len(calls) == 2
    assert len(list((tmp_path / "topology_cache").rglob("*.npy"))) == 2


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


def test_cube_size_batch_sampler_keeps_batch_shapes_stackable(tmp_path):
    shape = (8, 8, 8)
    _write_raw_pair(tmp_path, "sandstone", shape)
    index_dir = tmp_path / "datasets" / "sandstone"
    index_dir.mkdir(parents=True, exist_ok=True)
    for size in (4, 6):
        pd.DataFrame(
            [
                {"z": 0, "y": 0, "x": 0, "split": "train", "cube_size": size, "rock": "sandstone"},
                {"z": 0, "y": 0, "x": 1, "split": "train", "cube_size": size, "rock": "sandstone"},
            ]
        ).to_csv(index_dir / f"index_{size}.csv", index=False)

    dataset = BereaPatchDataset(
        tmp_path,
        split="train",
        cube_size=[4, 6],
        shape=shape,
        noise_types=["none"],
        balance=False,
    )
    sampler = CubeSizeBatchSampler(dataset, batch_size={4: 2, 6: 1}, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    batches = list(loader)

    assert [tuple(batch["x"].shape[-3:]) for batch in batches] == [(4, 4, 4), (6, 6, 6), (6, 6, 6)]
    assert [batch["x"].shape[0] for batch in batches] == [2, 1, 1]
    assert all(batch["cube_size"].unique().numel() == 1 for batch in batches)


def test_spatial_split_has_no_train_val_overlap():
    """Регрессионный тест к ликеджу: ни один val-патч не должен
    пространственно пересекаться ни с одним train-патчем (по оси z),
    в том числе между разными cube_size."""
    from utils.data import build_patch_index

    shape = (256, 128, 128)
    frames = [build_patch_index(shape, cs, val_fraction=0.25) for cs in (32, 64)]
    df = pd.concat(frames, ignore_index=True)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    assert len(train) > 0 and len(val) > 0

    train_z_end = (train["z"] + train["cube_size"]).max()
    val_z_start = val["z"].min()
    # Train-зона целиком до начала val-зоны: перекрытие невозможно
    assert train_z_end <= val_z_start


def test_percolation_labels_matches_simple_cases():
    from utils.data import percolation_labels

    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[:, 4, 4] = True  # сквозной канал вдоль z
    labels = percolation_labels(mask)
    assert labels.tolist() == [1.0, 0.0, 0.0]

    mask2 = np.zeros((8, 8, 8), dtype=bool)
    mask2[:4, 4, 4] = True  # обрывается на середине
    labels2 = percolation_labels(mask2)
    assert labels2.tolist() == [0.0, 0.0, 0.0]


def test_add_aux_targets_streaming_chunks_match_direct(tmp_path):
    """Потоковая (чанковая) версия add_aux_targets_to_index должна давать
    те же значения, что и прямой расчёт по кубу, независимо от размера чанка."""
    import numpy as np
    import pandas as pd

    from utils.data import add_aux_targets_to_index, percolation_labels

    rng = np.random.default_rng(7)
    vol = (rng.random((64, 64, 64)) > 0.55).astype(np.uint8)  # 0 = пора
    size = 32
    rows = [
        {"z": z, "y": y, "x": x, "split": "train"}
        for z in (0, 32)
        for y in (0, 32)
        for x in (0, 32)
    ]
    df = pd.DataFrame(rows)

    # chunk_memory_mb=1 → чанк больше одного, но меньше всех кубов сразу
    out = add_aux_targets_to_index(df, vol, size, pore_value=0, num_workers=0, chunk_memory_mb=1)

    for _, row in out.iterrows():
        z, y, x = int(row.z), int(row.y), int(row.x)
        cube = vol[z : z + size, y : y + size, x : x + size] == 0
        assert abs(row.porosity - cube.mean()) < 1e-6
        ref = percolation_labels(cube)
        got = np.array([row.percolates_z, row.percolates_y, row.percolates_x])
        assert np.array_equal(got, ref)
