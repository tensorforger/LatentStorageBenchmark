import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# FORMATS:
# - torch: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
# - np: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
# - cv2: [H, W, 3], pixel_range = [0, 255], channel_order = BRG, dtype = uint8


# Convertations


def torch_to_np(image: torch.Tensor) -> np.ndarray:
    """
    Args:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    Returns:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    """
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image = image.clip(0, 1)
    image = image.mul(255)
    image = image.to(torch.uint8)
    image = image.cpu()
    image = image.numpy()

    return image


def np_to_torch(
    image: np.ndarray, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
        device: target device for the tensor
        dtype: target dtype for the tensor
    Returns:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    """
    image = torch.from_numpy(image)
    image = image.to(device)
    image = image.to(dtype)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    image = image.div(255)

    return image


def cv2_to_np(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8
    Returns:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def np_to_cv2(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    Returns:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def cv2_to_torch(
    image: np.ndarray, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8
        device: target device for the tensor
        dtype: target dtype for the tensor
    Returns:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    """
    image = cv2_to_np(image)
    image = np_to_torch(image, device=device, dtype=dtype)

    return image


def torch_to_cv2(image: torch.Tensor) -> np.ndarray:
    """
    Args:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    Returns:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8
    """
    image = torch_to_np(image)
    image = np_to_cv2(image)

    return image


# Normalization


def norm_image(image: torch.Tensor) -> torch.Tensor:
    """
    Args:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    Returns:
        image: [B, 3, H, W], pixel_range = [-1.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    """
    image = image.mul(2)
    image = image.sub(1)

    return image


def denorm_image(image: torch.Tensor) -> torch.Tensor:
    """
    Args:
        image: [B, 3, H, W], pixel_range = [-1.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    Returns:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    """
    image = image.add(1)
    image = image.div(2)

    return image


# Resize


def resize_np(
    image: np.ndarray, height: int, width: int, interpolation: str = "bilinear"
) -> np.ndarray:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
        height: target height
        width: target width
        interpolation: interpolation method ("bilinear" or "nearests")
    Returns:
        image: [height, width, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    """
    image = resize_cv2(image, height=height, width=width, interpolation=interpolation)

    return image


def resize_cv2(
    image: np.ndarray, height: int, width: int, interpolation: str = "bilinear"
) -> np.ndarray:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB or BGR, dtype = uint8
        height: target height
        width: target width
        interpolation: interpolation method ("bilinear" or "nearests")
    Returns:
        image: [height, width, 3], pixel_range = [0, 255], channel_order = RGB or BGR, dtype = uint8
    """
    if interpolation == "bilinear":
        inter = cv2.INTER_LINEAR
    if interpolation == "nearests":
        inter = cv2.INTER_NEAREST

    image = cv2.resize(image, (width, height), interpolation=inter)

    return image


def resize_torch(
    image: torch.Tensor, height: int, width: int, interpolation: str = "bilinear"
) -> torch.Tensor:
    """
    Args:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0] or [-1.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
        height: target height
        width: target width
        interpolation: interpolation method (e.g., "bilinear", "nearest", "bicubic")
    Returns:
        image: [B, 3, height, width], pixel_range = [0.0, 1.0] or [-1.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    """
    image = torch.nn.functional.interpolate(image, (height, width), mode=interpolation)

    return image


# Preview


def show_np_image(image: np.ndarray) -> None:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = RGB, dtype = uint8
    Returns:
        None
    """
    plt.imshow(image)
    plt.show()


def show_torch_image(image: torch.Tensor) -> None:
    """
    Args:
        image: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB, dtype = float32 (or float16, bfloat16), device = "cuda"
    Returns:
        None
    """
    image = torch_to_np(image)
    show_np_image(image)


def show_cv2_image(image: np.ndarray) -> None:
    """
    Args:
        image: [H, W, 3], pixel_range = [0, 255], channel_order = BGR, dtype = uint8
    Returns:
        None
    """
    image = cv2_to_np(image)
    show_np_image(image)
