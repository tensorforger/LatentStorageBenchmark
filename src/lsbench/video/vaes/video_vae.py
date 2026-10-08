from abc import abstractmethod
import torch


class VideoVAE(torch.nn.Module):
    """
    Abstract class and common interface for all Video VAE wrappers.
    Intended for VAE inference only.
    VAE should handle by itself:
    - pixel and latent normalization
    - video and latent device and dtype handling according to weights
    Subclasses must define a class constant NAME equal to the config entry.
    """

    NAME: str

    @abstractmethod
    def encode(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        returns:
            latents: [B, C, T, H, W], normalized to model format
        """
        pass

    @abstractmethod
    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H, W], normalized to model format
        returns:
            video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
            T may exceed the encoded frame count if encode padded it (the runner crops it).
        """
        pass

    @abstractmethod
    def get_latent_channels(self) -> int:
        """
        returns:
            latent_channels of the VAE
        """
        pass

    @abstractmethod
    def get_spatial_compression(self) -> int:
        """
        returns:
            spatial_compression of the VAE
        """
        pass

    @abstractmethod
    def get_temporal_compression(self) -> int:
        """
        returns:
            temporal_compression of the VAE (frames per latent frame)
        """
        pass

    @abstractmethod
    def get_vae_name(self) -> str:
        pass
