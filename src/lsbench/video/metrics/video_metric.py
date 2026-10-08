from abc import ABC, abstractmethod
import torch


class VideoMetric(ABC):
    """
    Abstract class and common interface for video metrics.
    Subclasses must define a class constant NAME equal to the config entry.
    """

    NAME: str

    @abstractmethod
    def add_pair(
        self, original_video: torch.Tensor, reconstructed_video: torch.Tensor
    ) -> None:
        """
        args:
            original_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        pass

    @abstractmethod
    def calculate_metric(
        self,
    ) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        pass
