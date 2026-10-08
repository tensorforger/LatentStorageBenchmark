import torch
from lsbench.video.metrics.video_metric import VideoMetric


class MSEMetric(VideoMetric):
    NAME = "mse"

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
        se = (original_video - reconstructed_video) ** 2

        per_sample_mse = se.flatten(start_dim=1).mean(dim=1)

        self.sum_per_sample_mse += per_sample_mse.sum().item()
        self.num_samples += original_video.shape[0]

    def calculate_metric(
        self,
    ) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.sum_per_sample_mse / self.num_samples

    def reset(self) -> None:
        self.sum_per_sample_mse = 0.0
        self.num_samples = 0

    def get_metric_name(self) -> str:
        return self.NAME
