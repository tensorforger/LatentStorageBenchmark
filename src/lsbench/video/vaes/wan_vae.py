import torch

from diffusers import AutoencoderKLWan

from lsbench.utils.video_tools import pad_frames_last
from lsbench.video.vaes.video_vae import VideoVAE


class WanVAE(VideoVAE):
    NAME = "wan"

    def __init__(self, path_to_model: str, device: torch.device = "cuda"):
        super().__init__()

        self.model = AutoencoderKLWan.from_pretrained(path_to_model).to(device)
        self.model.eval()

        cfg = self.model.config
        self.latent_channels = cfg.z_dim
        self.spatial_compression = cfg.scale_factor_spatial
        self.temporal_compression = cfg.scale_factor_temporal

        self.mean = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

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
        # The causal temporal VAE needs T = 1 + 4k; the tail is padded and cropped in decode output.
        video = pad_frames_last(video, self.temporal_compression, 1)
        video = (video * 2 - 1).to(self.dtype)
        latents = self.model.encode(video).latent_dist.mode()
        latents = (latents - self.mean.to(latents.dtype)) / self.std.to(latents.dtype)

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
            T may exceed the original frame count (padding from encode), caller crops it.
        """
        latents = latents.to(self.dtype)
        latents = latents * self.std.to(self.dtype) + self.mean.to(self.dtype)
        video = self.model.decode(latents, return_dict=False)[0]

        return (video + 1) / 2

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
