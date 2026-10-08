"""Prepare raw datasets into a uniform, cropped PNG tree.

Each raw dataset under ``datasets/`` has a different on-disk layout. This
script extracts only the meaningful content images (skipping ground-truth
masks, captions, and macOS junk), crops them to the configured size, and
writes them as zero-padded ``NNNNNN.png`` files under ``prepared_datasets/``.

After running this, the image dataset loader only has to read the uniform
``prepared_datasets/<name>/`` folders.

Usage:
    .venv/bin/python scripts/prepare_datasets.py [config_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from lsbench.utils.crop_maximal_rectangle import crop_maximal_rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
PREPARED_ROOT = PROJECT_ROOT / "prepared_datasets"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".gif",
}


def _is_junk_file(path: Path) -> bool:
    """Return True for macOS AppleDouble metadata files (``__MACOSX`` / ``._*``)."""
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _find_images(root: Path) -> list[Path]:
    """Recursively collect real image files under ``root``, skipping junk."""
    images: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if _is_junk_file(path):
            continue
        images.append(path)
    return images


def _describe_unreadable(path: Path) -> str:
    """Give a helpful reason for why an image file could not be decoded."""
    try:
        head = path.read_bytes()[:32]
    except OSError:
        return "could not be read"
    if head.startswith(b"version https://git-lfs"):
        return "git-lfs pointer (run `git lfs pull` to fetch it)"
    return "could not be decoded"


def collect_source_images(dataset_name: str, dataset_dir: Path) -> list[Path]:
    """Return the meaningful source images for a dataset.

    Each dataset has a different layout, so we branch on the dataset name
    and only pull the real content images.
    """
    if dataset_name == "axis-v1_1k":
        # Per-category subfolders, each mixing images with .txt captions.
        return _find_images(dataset_dir)
    if dataset_name == "OCR-Quality":
        # All content images live under pics/.
        return _find_images(dataset_dir / "pics")
    if dataset_name == "Total-Text-Dataset":
        # Only Images/ holds real photos. groundtruth_* are masks,
        # txt_format is text, __MACOSX is macOS junk.
        return _find_images(dataset_dir / "Images")
    raise ValueError(f"Unknown dataset: {dataset_name!r}")


def _clear_pngs(out_dir: Path) -> None:
    """Remove previously written PNGs so re-runs stay consistent."""
    for png in out_dir.glob("*.png"):
        png.unlink()


def prepare_dataset(dataset_name: str, width: int, height: int) -> int:
    """Crop and save all meaningful images of one dataset. Returns count saved."""
    dataset_dir = DATASETS_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = PREPARED_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_pngs(out_dir)

    source_images = collect_source_images(dataset_name, dataset_dir)
    saved = 0
    for src in tqdm(source_images, desc=dataset_name, leave=False):
        image = cv2.imread(str(src))
        if image is None:
            # Skip files that are not real images (e.g. git-lfs pointers or
            # corrupt files) instead of aborting the whole run.
            print(
                f"  skipping {src.relative_to(PROJECT_ROOT)}: {_describe_unreadable(src)}"
            )
            continue
        image = crop_maximal_rectangle(image, height, width)
        cv2.imwrite(str(out_dir / f"{saved:06d}.png"), image)
        saved += 1
    return saved


def collect_source_videos(dataset_name: str, dataset_dir: Path) -> list[Path]:
    """Return the source videos for a video dataset."""
    if dataset_name == "movirec":
        return sorted((dataset_dir / "video-patches").glob("*.mp4"))
    raise ValueError(f"Unknown video dataset: {dataset_name!r}")


def _read_video(
    path: Path, height: int, width: int, num_frames: int
) -> np.ndarray | None:
    """Decode the first `num_frames` frames, cropped to size. Returns uint8 RGB [T, H, W, 3] or None if too short."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame = crop_maximal_rectangle(frame, height, width)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames) if len(frames) == num_frames else None


def prepare_video_dataset(
    dataset_name: str, width: int, height: int, num_frames: int
) -> int:
    """Crop and save all videos of one dataset as NNNNNN.npy (uint8 RGB [T, H, W, 3]). Returns count saved."""
    dataset_dir = DATASETS_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = PREPARED_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npy"):
        old.unlink()

    saved = 0
    for src in tqdm(
        collect_source_videos(dataset_name, dataset_dir), desc=dataset_name, leave=False
    ):
        video = _read_video(src, height, width, num_frames)
        if video is None:
            print(
                f"  skipping {src.relative_to(PROJECT_ROOT)}: fewer than {num_frames} frames"
            )
            continue
        np.save(out_dir / f"{saved:06d}.npy", video)
        saved += 1
    return saved


def main() -> None:
    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(PROJECT_ROOT / "configs" / "lsbench_1.0.yaml")
    )
    cfg = OmegaConf.load(config_path)
    image_cfg = cfg.image
    width = int(image_cfg.width)
    height = int(image_cfg.height)

    # Create the prepared root up front so it exists even if a dataset is empty.
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name in image_cfg.datasets:
        count = prepare_dataset(dataset_name, width, height)
        print(f"{dataset_name}: saved {count} images -> {PREPARED_ROOT / dataset_name}")

    video_cfg = cfg.get("video")
    if video_cfg is not None:
        for dataset_name in video_cfg.datasets:
            count = prepare_video_dataset(
                dataset_name,
                int(video_cfg.width),
                int(video_cfg.height),
                int(video_cfg.num_frames),
            )
            print(
                f"{dataset_name}: saved {count} videos -> {PREPARED_ROOT / dataset_name}"
            )


if __name__ == "__main__":
    main()
