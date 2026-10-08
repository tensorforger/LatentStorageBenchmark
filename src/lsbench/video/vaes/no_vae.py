import torch

from lsbench.video.vaes.video_vae import VideoVAE


class NoVAE(VideoVAE):
    NAME = "no_vae"

    def __init__(self, device: torch.device = "cuda"):
        super().__init__()
        self.latent_channels = 3
        self.spatial_compression = 1
        self.temporal_compression = 1

        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None, None].to(
            device
        )
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None, None].to(
            device
        )

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
        latents = video.sub(self.mean).div(self.std)

        return latents

    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H, W], normalized to model format
        returns:
            video: [B, C, T, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        video = latents.mul(self.std).add(self.mean)

        return video

    def get_latent_channels(self) -> int:
        """
        returns:
            latent_channels of the VAE
        """
        return self.latent_channels

    def get_spatial_compression(self) -> int:
        """
        returns:
            spatial_compression of the VAE
        """
        return self.spatial_compression

    def get_temporal_compression(self) -> int:
        """
        returns:
            temporal_compression of the VAE (frames per latent frame)
        """
        return self.temporal_compression

    def get_vae_name(self) -> str:
        return self.NAME
