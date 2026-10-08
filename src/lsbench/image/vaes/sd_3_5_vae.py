import torch

from diffusers import AutoencoderKL

from lsbench.image.vaes.image_vae import ImageVAE


class SD35VAE(ImageVAE):
    NAME = "sd_3_5"

    def __init__(self, path_to_model: str, device: torch.device = "cuda"):
        super().__init__()

        self.model = AutoencoderKL.from_pretrained(path_to_model).to(device)
        self.model.eval()

        self.latent_channels = self.model.config.latent_channels
        self.spatial_compression = 2 ** (len(self.model.config.block_out_channels) - 1)
        self.scaling_factor = self.model.config.scaling_factor
        self.shift_factor = self.model.config.shift_factor

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

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
        x = (image * 2.0 - 1.0).to(self.dtype)
        latents = self.model.encode(x).latent_dist.mode()
        latents = (latents - self.shift_factor) * self.scaling_factor

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
        latents = latents.to(self.dtype) / self.scaling_factor + self.shift_factor
        image = self.model.decode(latents, return_dict=False)[0]

        return (image + 1.0) / 2.0

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
        return self.NAME
