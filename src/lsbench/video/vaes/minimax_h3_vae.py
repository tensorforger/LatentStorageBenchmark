import torch

from diffusers import AutoencoderKLMiniMaxH3

from lsbench.utils.video_tools import pad_frames_last
from lsbench.video.vaes.video_vae import VideoVAE


class MiniMaxH3VAE(VideoVAE):
    NAME = "minimax_h3"

    def __init__(self, path_to_model: str, device: torch.device = "cuda"):
        super().__init__()

        self.model = AutoencoderKLMiniMaxH3.from_pretrained(path_to_model).to(device)
        self.model.eval()

        cfg = self.model.config
        self.latent_channels = cfg.latent_channels
        self.spatial_compression = self.model.spatial_compression_ratio
        self.temporal_compression = self.model.temporal_compression_ratio

        self.latents_mean = torch.tensor(cfg.latents_mean, device=device).view(
            1, -1, 1, 1, 1
        )
        self.latents_std = torch.tensor(cfg.latents_std, device=device).view(
            1, -1, 1, 1, 1
        )
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(
            1, 3, 1, 1, 1
        )
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(
            1, 3, 1, 1, 1
        )

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
        # Frame counts 17n + 5 map to 5n + 2 latents and decode back exactly; pad the tail otherwise.
        video = pad_frames_last(video, 17, 5)
        video = (video - self.pixel_mean) / self.pixel_std
        latents = self.model.encode(video).latent_dist.mode()
        latents = (latents - self.latents_mean) / self.latents_std

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
        latents = latents.float() * self.latents_std + self.latents_mean
        video = self.model.decode(latents, return_dict=False)[0]

        return video.float() * self.pixel_std + self.pixel_mean

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
