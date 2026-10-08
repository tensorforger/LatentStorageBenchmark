import lpips
import torch

from lsbench.video.metrics.video_metric import VideoMetric


class LPIPSMetric(VideoMetric):
    """Per-frame learned perceptual image patch similarity, averaged over frames, then over videos (lower is better)."""

    NAME = "lpips"
    _FRAME_CHUNK = 32
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
        self, original_video: torch.Tensor, reconstructed_video: torch.Tensor
    ) -> None:
        """
        args:
            original_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        batch_size, channels, num_frames, height, width = original_video.shape
        # lpips expects [B*T, C, H, W] in [-1, 1]
        original = (
            original_video.detach()
            .float()
            .clamp(0.0, 1.0)
            .to(self.device)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, channels, height, width)
            * 2.0
            - 1.0
        )
        reconstructed = (
            reconstructed_video.detach()
            .float()
            .clamp(0.0, 1.0)
            .to(self.device)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, channels, height, width)
            * 2.0
            - 1.0
        )

        with torch.no_grad():
            per_frame_lpips = torch.cat(
                [
                    self.model(
                        reconstructed[i : i + self._FRAME_CHUNK],
                        original[i : i + self._FRAME_CHUNK],
                    ).flatten()
                    for i in range(0, original.shape[0], self._FRAME_CHUNK)
                ]
            )
        per_sample_lpips = per_frame_lpips.reshape(batch_size, num_frames).mean(dim=1)

        self.sum_per_sample_lpips += per_sample_lpips.sum().item()
        self.num_samples += batch_size

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
