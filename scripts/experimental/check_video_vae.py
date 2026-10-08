"""Quick sanity check of a video VAE: latent statistics and roundtrip MSE on one movirec sample."""

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from lsbench.utils.component_factory import configure_vae

root = Path(__file__).resolve().parents[2]
cfg = OmegaConf.load(root / "configs" / "lsbench_1.1_fast.yaml")
entries = {e["name"]: e for e in cfg.video.vaes if not isinstance(e, str)}
name = sys.argv[1]
device = torch.device("cuda")

video = np.load(sorted((root / "prepared_datasets" / "movirec").glob("*.npy"))[0])
video = torch.from_numpy(video).permute(3, 0, 1, 2).float().div(255)[None].to(device)

vae = configure_vae(entries[name], device, "video")
with torch.no_grad():
    z = vae.encode(video)
    rec = vae.decode(z)[:, :, : video.shape[2]].float()

zf = z.float()
mse = ((rec - video) ** 2).mean().item()
print(f"{name}: video {tuple(video.shape)} -> latents {tuple(z.shape)}, dtype {z.dtype}")
print(f"latent mean {zf.mean():.3f} std {zf.std():.3f} min {zf.min():.2f} max {zf.max():.2f}")
ch_mean, ch_std = zf.mean(dim=(0, 2, 3, 4)), zf.std(dim=(0, 2, 3, 4))
print(f"per-channel mean in [{ch_mean.min():.2f}, {ch_mean.max():.2f}], std in [{ch_std.min():.2f}, {ch_std.max():.2f}]")
z0 = (zf - zf.mean()) / zf.std()
print(f"excess kurtosis {(z0 ** 4).mean().item() - 3:.2f}, frac |z|>3: {(zf.abs() > 3).float().mean():.4f} (normal: 0.0027)")
print(f"rec range [{rec.min():.3f}, {rec.max():.3f}]  MSE {mse:.6f}  PSNR {-10 * np.log10(mse):.2f} dB")
