import torch

from diffusers import AutoencoderKLQwenImage21

from lsbench.image.vaes.image_vae import ImageVAE


class QwenImage21VAE(ImageVAE):
    NAME = "qwen_image_2_1"

    def __init__(self, path_to_model: str, device: torch.device = "cuda"):
        super().__init__()

        self.model = AutoencoderKLQwenImage21.from_pretrained(path_to_model).to(device)
        self.model.eval()

        self.latent_channels = self.model.config.z_dim
        self.spatial_compression = self.model.spatial_compression_ratio

        shape = (1, self.latent_channels, 1, 1)
        self.register_buffer(
            "mean",
            torch.tensor(self.model.config.latents_mean).view(shape),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(self.model.config.latents_std).view(shape),
            persistent=False,
        )
        self.to(device)

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
        # The model is an RGBA video VAE: add an opaque alpha channel and a single-frame time axis.
        x = image * 2.0 - 1.0
        alpha = torch.ones_like(x[:, :1])
        x = torch.cat([x, alpha], dim=1).to(self.dtype).unsqueeze(2)
        latents = self.model.encode(x).latent_dist.mode().squeeze(2)
        latents = (latents - self.mean) / self.std

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
        latents = latents.to(self.dtype) * self.std + self.mean
        image = self.model.decode(latents.unsqueeze(2), return_dict=False)[0][:, :3, 0]

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
