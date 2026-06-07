from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BereaPatchDataset(Dataset):
    """Датасет патчей по grayscale и бинарному объему Berea."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        cube_size: int = 128,
        shape: tuple[int, int, int] = (1000, 1000, 1000),
        pore_value: int = 0,
        use_raw_gray: bool = False,
        noise_types: list[str] | None = None,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "data"
        self.dataset_dir = self.root_dir / "dataset_128"
        self.split = split
        self.cube_size = cube_size
        self.shape = shape
        self.pore_value = pore_value
        self.rng = np.random.default_rng(seed)

        gray_name = "Berea_2d25um_grayscale.raw" if use_raw_gray else "Berea_2d25um_grayscale_filtered.raw"
        self.gray_path = self.data_dir / gray_name
        self.binary_path = self.data_dir / "Berea_2d25um_binary.raw"

        self.gray = np.memmap(self.gray_path, dtype=np.uint8, mode="r", shape=self.shape)
        self.binary = np.memmap(self.binary_path, dtype=np.uint8, mode="r", shape=self.shape)

        index_path = self.dataset_dir / f"index_{cube_size}.csv"
        if not index_path.exists():
            index_path = self.dataset_dir / "index_128.csv"
        df = pd.read_csv(index_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

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

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def normalize_uint8(cube: np.ndarray) -> np.ndarray:
        return cube.astype(np.float32) / 255.0

    def add_noise(self, cube: np.ndarray, noise_type: str) -> np.ndarray:
        img = cube.copy().astype(np.float32)

        if noise_type == "none":
            pass
        elif noise_type == "gaussian_low":
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
        elif noise_type == "contrast_shift":
            img = img * self.rng.uniform(0.75, 1.25) + self.rng.uniform(-0.10, 0.10)
        elif noise_type == "mixed":
            img += self.rng.normal(0.0, 0.05, size=img.shape).astype(np.float32)
            img = img * self.rng.uniform(0.8, 1.2) + self.rng.uniform(-0.08, 0.08)
            prob = 0.01
            mask = self.rng.random(img.shape)
            img[mask < prob / 2] = 0.0
            img[(mask >= prob / 2) & (mask < prob)] = 1.0
        else:
            raise ValueError(f"Неизвестный тип шума: {noise_type}")

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def get_cube(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        cs = self.cube_size
        gray_cube = self.gray[z:z + cs, y:y + cs, x:x + cs]
        binary_cube = self.binary[z:z + cs, y:y + cs, x:x + cs]
        return {"gray": gray_cube, "binary": binary_cube, "coord": (z, y, x)}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cube = self.get_cube(idx)
        clean_x = self.normalize_uint8(cube["gray"])
        noise_type = self.rng.choice(self.noise_types)
        noisy_x = self.add_noise(clean_x, str(noise_type))
        target = (cube["binary"] == self.pore_value).astype(np.float32)

        return {
            "x": torch.from_numpy(noisy_x.copy()).float().unsqueeze(0),
            "y": torch.from_numpy(target.copy()).float().unsqueeze(0),
            "coord": torch.tensor(cube["coord"], dtype=torch.long),
            "noise": str(noise_type),
        }


BereaSegmentationDataset = BereaPatchDataset
