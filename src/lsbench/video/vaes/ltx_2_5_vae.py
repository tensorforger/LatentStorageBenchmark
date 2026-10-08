import torch

from diffusers import AutoencoderKLLTX2Video

from lsbench.utils.video_tools import pad_frames_last
from lsbench.video.vaes.video_vae import VideoVAE


class LTX25VAE(VideoVAE):
    NAME = "ltx_2_5"

    def __init__(self, path_to_model: str, device: torch.device = "cuda"):
        super().__init__()

        self.model = AutoencoderKLLTX2Video.from_pretrained(path_to_model).to(device)
        self.model.eval()

        self.latent_channels = self.model.config.latent_channels
        self.spatial_compression = self.model.spatial_compression_ratio
        self.temporal_compression = self.model.temporal_compression_ratio

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _stats(self, like: torch.Tensor):
        mean = self.model.latents_mean.view(1, -1, 1, 1, 1).to(like)
        std = self.model.latents_std.view(1, -1, 1, 1, 1).to(like)
        return mean, std, self.model.config.scaling_factor

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
        # The causal temporal VAE needs T = 1 + 8k; the tail is padded and cropped in decode output.
        video = pad_frames_last(video, self.temporal_compression, 1)
        video = (video * 2 - 1).to(self.dtype)
        latents = self.model.encode(video).latent_dist.mode()
        mean, std, scaling_factor = self._stats(latents)

        return (latents - mean) * scaling_factor / std

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
        mean, std, scaling_factor = self._stats(latents)
        latents = latents * std / scaling_factor + mean
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
