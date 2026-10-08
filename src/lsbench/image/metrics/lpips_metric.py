import lpips
import torch

from lsbench.image.metrics.image_metric import ImageMetric


class LPIPSMetric(ImageMetric):
    """Learned perceptual image patch similarity, averaged per sample (lower is better)."""

    NAME = "lpips"
    _shared_networks = {}  # one network per backbone; one instance per storage would waste GPU memory

    def __init__(self, net: str = "alex"):
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if net not in LPIPSMetric._shared_networks:
            model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
            model.requires_grad_(False)
            LPIPSMetric._shared_networks[net] = model
        self.model = LPIPSMetric._shared_networks[net]
        self.reset()

    def add_pair(
        self, original_image: torch.Tensor, reconstructed_image: torch.Tensor
    ) -> None:
        """
        args:
            original_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        # lpips expects inputs in [-1, 1]
        original = (
            original_image.detach().float().clamp(0.0, 1.0).to(self.device) * 2.0 - 1.0
        )
        reconstructed = (
            reconstructed_image.detach().float().clamp(0.0, 1.0).to(self.device) * 2.0
            - 1.0
        )

        with torch.no_grad():
            per_sample_lpips = self.model(reconstructed, original)

        self.sum_per_sample_lpips += per_sample_lpips.sum().item()
        self.num_samples += original_image.shape[0]

    def calculate_metric(
        self,
    ) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.sum_per_sample_lpips / self.num_samples

    def reset(self) -> None:
        self.sum_per_sample_lpips = 0.0
        self.num_samples = 0

    def get_metric_name(self) -> str:
        return self.NAME
