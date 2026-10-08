import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from lsbench.image.metrics.image_metric import ImageMetric


class SSIMMetric(ImageMetric):
    """Structural similarity (Gaussian 11x11 window, sigma 1.5), averaged per sample (higher is better, max 1)."""

    NAME = "ssim"

    def __init__(self):
        self.reset()

    def add_pair(
        self, original_image: torch.Tensor, reconstructed_image: torch.Tensor
    ) -> None:
        """
        args:
            original_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        original = original_image.detach().float().clamp(0.0, 1.0)
        reconstructed = reconstructed_image.detach().float().clamp(0.0, 1.0)

        per_sample_ssim = structural_similarity_index_measure(
            reconstructed, original, data_range=1.0, reduction="none"
        )

        self.sum_per_sample_ssim += per_sample_ssim.sum().item()
        self.num_samples += original_image.shape[0]

    def calculate_metric(
        self,
    ) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.sum_per_sample_ssim / self.num_samples

    def reset(self) -> None:
        self.sum_per_sample_ssim = 0.0
        self.num_samples = 0

    def get_metric_name(self) -> str:
        return self.NAME
