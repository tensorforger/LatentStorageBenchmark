import torch
from lsbench.video.metrics.video_metric import VideoMetric


class PSNRMetric(VideoMetric):
    """Peak signal-to-noise ratio in dB of the whole video (MSE over all frames), averaged per video (higher is better)."""

    NAME = "psnr"
    _MIN_MSE = 1e-10  # caps identical videos at 100 dB instead of inf

    def __init__(self):
        self.reset()

    def add_pair(
        self, original_video: torch.Tensor, reconstructed_video: torch.Tensor
    ) -> None:
        """
        args:
            original_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        original = original_video.detach().float().clamp(0.0, 1.0)
        reconstructed = reconstructed_video.detach().float().clamp(0.0, 1.0)

        per_sample_mse = (
            ((original - reconstructed) ** 2).flatten(start_dim=1).mean(dim=1)
        )
        per_sample_psnr = -10.0 * torch.log10(per_sample_mse.clamp_min(self._MIN_MSE))

        self.sum_per_sample_psnr += per_sample_psnr.sum().item()
        self.num_samples += original_video.shape[0]

    def calculate_metric(
        self,
    ) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.sum_per_sample_psnr / self.num_samples

    def reset(self) -> None:
        self.sum_per_sample_psnr = 0.0
        self.num_samples = 0

    def get_metric_name(self) -> str:
        return self.NAME
