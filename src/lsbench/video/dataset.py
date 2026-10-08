from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig


class VideoDataset(Dataset):
    """
    Dataset over the prepared, cropped videos.

    Videos are expected to already live under ``prepared_datasets/<name>/``
    (produced by ``scripts/prepare_datasets.py``) as ``NNNNNN.npy`` files holding
    uint8 RGB frames [T, H, W, 3], pre-cropped to the configured size and length.
    Only file paths are kept in memory; frames are read from disk in ``__getitem__``.
    """

    def __init__(self, cfg: DictConfig, num_samples: int | None = None):
        prepared_root = Path(__file__).resolve().parents[3] / "prepared_datasets"
        self.video_paths: list[Path] = []
        for dataset_name in cfg.datasets:
            dataset_dir = prepared_root / dataset_name
            if not dataset_dir.is_dir():
                raise FileNotFoundError(
                    f"Prepared dataset directory not found: {dataset_dir}. "
                    f"Run scripts/prepare_datasets.py first."
                )
            self.video_paths.extend(sorted(dataset_dir.glob("*.npy")))
        if num_samples is not None:
            self.video_paths = self.video_paths[:num_samples]

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        returns:
            video: [C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, float32, cpu
        """
        frames = np.load(self.video_paths[index])
        video = torch.from_numpy(frames).permute(3, 0, 1, 2)
        return video.float().div(255)
