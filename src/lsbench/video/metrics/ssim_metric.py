import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from lsbench.video.metrics.video_metric import VideoMetric


class SSIMMetric(VideoMetric):
    """Per-frame structural similarity (Gaussian 11x11 window, sigma 1.5), averaged over frames, then over videos (higher is better, max 1)."""

    NAME = "ssim"
    _FRAME_CHUNK = 32

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
        batch_size, channels, num_frames, height, width = original_video.shape
        # [B, C, T, H, W] -> [B*T, C, H, W]
        original = (
            original_video.detach()
            .float()
            .clamp(0.0, 1.0)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, channels, height, width)
        )
        reconstructed = (
            reconstructed_video.detach()
            .float()
            .clamp(0.0, 1.0)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, channels, height, width)
        )

        per_frame_ssim = torch.cat(
            [
                structural_similarity_index_measure(
                    reconstructed[i : i + self._FRAME_CHUNK],
                    original[i : i + self._FRAME_CHUNK],
                    data_range=1.0,
                    reduction="none",
                )
                for i in range(0, original.shape[0], self._FRAME_CHUNK)
            ]
        )
        per_sample_ssim = per_frame_ssim.reshape(batch_size, num_frames).mean(dim=1)

        self.sum_per_sample_ssim += per_sample_ssim.sum().item()
        self.num_samples += batch_size

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
