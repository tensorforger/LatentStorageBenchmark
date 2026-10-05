import torch

from lsbench.image.vaes.image_vae import ImageVAE


class NoVAE(ImageVAE):
    def __init__(self, device: torch.device = "cuda"):
        self.latent_channels = 3
        self.spatial_compression = 1
        self.vae_name = "no_vae"

        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)

    def encode(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        latents = image.sub(self.mean).div(self.std)

        return latents

    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W], normalized to model format
        returns:
            image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        image = latents.mul(self.std).add(self.mean)

        return image

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

    def get_vae_name(self) -> str:
        return self.vae_name
