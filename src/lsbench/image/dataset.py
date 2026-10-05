from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig
from lsbench.utils.image_tools import cv2_to_torch


class ImageDataset(Dataset):
    """
    Dataset over the prepared, cropped PNG images.

    Images are expected to already live under ``prepared_datasets/<name>/``
    (produced by ``scripts/prepare_datasets.py``), each pre-cropped to the
    configured size. Only file paths are kept in memory; pixels are read
    from disk on the fly in ``__getitem__``.
    """

    def __init__(self, cfg: DictConfig):
        prepared_root = Path(__file__).resolve().parents[3] / "prepared_datasets"
        self.image_paths: list[Path] = []
        for dataset_name in cfg.datasets:
            dataset_dir = prepared_root / dataset_name
            if not dataset_dir.is_dir():
                raise FileNotFoundError(
                    f"Prepared dataset directory not found: {dataset_dir}. "
                    f"Run scripts/prepare_datasets.py first."
                )
            self.image_paths.extend(sorted(dataset_dir.glob("*.png")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.image_paths[index]))
        if image is None:
            raise ValueError(f"Failed to read image: {self.image_paths[index]}")

        image = cv2_to_torch(image)[0]

        return image
