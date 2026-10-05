from abc import ABC, abstractmethod
import torch


class ImageMetric(ABC):
    """
    Abstract class and common interface for image metrics.
    Subclasses must define a class constant NAME equal to the config entry.
    """

    NAME: str

    @abstractmethod
    def add_pair(
        self, original_image: torch.Tensor, reconstructed_image: torch.Tensor
    ) -> None:
        """
        args:
            original_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            reconstructed_image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
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
