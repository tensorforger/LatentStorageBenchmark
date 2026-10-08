from pathlib import Path

import torch
import torch.nn.functional as F

from lsbench.video.metrics.video_metric import VideoMetric


class FVDMetric(VideoMetric):
    """Frechet Video Distance (StyleGAN-V I3D, 400-d features) between original and reconstructed video sets (lower is better)."""

    NAME = "fvd"
    _I3D_INPUT_SIZE = 224
    _shared_detectors = {}  # one I3D per weights path; one instance per storage would exhaust GPU memory

    def __init__(self, path_to_model: str = "models/i3d/i3d_torchscript.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if path_to_model not in FVDMetric._shared_detectors:
            path = Path(path_to_model)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[4] / path
            detector = torch.jit.load(str(path), map_location=self.device).eval()
            FVDMetric._shared_detectors[path_to_model] = detector
        self.detector = FVDMetric._shared_detectors[path_to_model]
        self.reset()

    def _features(self, video: torch.Tensor) -> torch.Tensor:
        """[B, C, T, H, W] in [0, 1] -> [B, D] float64 I3D features."""
        batch_size, channels, num_frames, _, _ = video.shape
        frames = video.detach().float().clamp(0.0, 1.0).to(self.device)
        frames = frames.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, *video.shape[3:]
        )
        frames = F.interpolate(
            frames,
            size=(self._I3D_INPUT_SIZE, self._I3D_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        frames = frames.reshape(
            batch_size, num_frames, channels, self._I3D_INPUT_SIZE, self._I3D_INPUT_SIZE
        ).permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            features = self.detector(
                frames * 2.0 - 1.0, rescale=False, resize=False, return_features=True
            )
        return features.double()

    def add_pair(
        self, original_video: torch.Tensor, reconstructed_video: torch.Tensor
    ) -> None:
        """
        args:
            original_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        for features, stats in (
            (self._features(original_video), self.real),
            (self._features(reconstructed_video), self.fake),
        ):
            if stats["sum"] is None:
                dim = features.shape[1]
                stats["sum"] = torch.zeros(dim, dtype=torch.float64, device=self.device)
                stats["outer"] = torch.zeros(
                    dim, dim, dtype=torch.float64, device=self.device
                )
            stats["sum"] += features.sum(dim=0)
            stats["outer"] += features.T @ features
            stats["count"] += features.shape[0]

    @staticmethod
    def _mean_cov(stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
        n = stats["count"]
        mean = stats["sum"] / n
        cov = (stats["outer"] - n * torch.outer(mean, mean)) / (n - 1)
        return mean, cov

    def calculate_metric(
        self,
    ) -> float:
        # A covariance estimate needs at least two videos.
        if self.real["count"] < 2:
            return 0.0
        mean_real, cov_real = self._mean_cov(self.real)
        mean_fake, cov_fake = self._mean_cov(self.fake)
        # tr(sqrt(A @ B)) = sum of sqrt of the (real, non-negative) eigenvalues of A @ B
        eigenvalues = torch.linalg.eigvals(cov_real @ cov_fake).real.clamp_min(0.0)
        trace_sqrt = eigenvalues.sqrt().sum()
        fvd = (
            (mean_real - mean_fake).pow(2).sum()
            + cov_real.trace()
            + cov_fake.trace()
            - 2.0 * trace_sqrt
        )
        return float(fvd.item())

    def reset(self) -> None:
        self.real = {"sum": None, "outer": None, "count": 0}
        self.fake = {"sum": None, "outer": None, "count": 0}

    def get_metric_name(self) -> str:
        return self.NAME
