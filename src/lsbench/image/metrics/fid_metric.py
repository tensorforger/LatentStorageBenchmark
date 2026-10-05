import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from lsbench.image.metrics.image_metric import ImageMetric


class FIDMetric(ImageMetric):
    """Frechet Inception Distance between original and reconstructed image sets (lower is better)."""

    NAME = "fid"

    def __init__(self, feature: int = 2048):
        self.feature = feature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fid = FrechetInceptionDistance(
            feature=self.feature, reset_real_features=True, normalize=True
        ).to(self.device)
        self.num_samples = 0

    def add_pair(
        self, original_image: torch.Tensor, reconstructed_image: torch.Tensor
    ) -> None:
        """
        args:
            original_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        real = original_image.detach().float().clamp(0.0, 1.0).to(self.device)
        fake = reconstructed_image.detach().float().clamp(0.0, 1.0).to(self.device)
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        self.num_samples += original_image.shape[0]

    def calculate_metric(
        self,
    ) -> float:
        # FID needs at least two samples to estimate a covariance.
        if self.num_samples < 2:
            return 0.0
        return float(self.fid.compute().item())

    def reset(self) -> None:
        self.fid.reset()
        self.num_samples = 0

    def get_metric_name(self) -> str:
        return self.NAME
