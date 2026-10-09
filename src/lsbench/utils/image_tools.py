import torch
import cv2
import numpy as np

# FORMATS:
# - torch: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32
# - np: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
# - cv2: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8


def cv2_to_np(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8
    Returns:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def np_to_torch(
    image: np.ndarray, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    Returns:
        image: [1, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
    """
    image = torch.from_numpy(image).to(device).to(dtype)
    return image.unsqueeze(0).permute(0, 3, 1, 2).div(255)


def cv2_to_torch(
    image: np.ndarray, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8
    Returns:
        image: [1, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
    """
    return np_to_torch(cv2_to_np(image), device=device, dtype=dtype)
