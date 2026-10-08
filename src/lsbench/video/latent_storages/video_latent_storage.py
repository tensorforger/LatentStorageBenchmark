from abc import ABC, abstractmethod
import torch


class VideoLatentStorage(ABC):
    """
    Abstract class and common interface for video latent storages.
    Handeles device by itself.
    Subclasses must define a class constant NAME equal to the config entry.
    """

    NAME: str

    @abstractmethod
    def serialize(self, latents: torch.Tensor) -> bytes:
        """
        args:
            latents: [B, C, T, H, W], normalized to model format
        returns:
            bytes
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes, device: torch.device = "cuda") -> torch.Tensor:
        """
        args:
            data: bytes
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        pass

    @abstractmethod
    def get_storage_name(self) -> str:
        pass
