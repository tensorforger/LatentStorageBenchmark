# Latent Storage Benchmark

**How much can VAE latents be compressed before decoded images degrade?**

> VAEs compress images, but naive `fp32` latent storage often uses more disk space than the original JPEG. This benchmark measures the quality and efficiency trade-offs of different compression strategies.

While the core purpose of this is to benchmark **Storage** efficency, it can also be used as general **VAE** benchmark (no compression case).

## Metrics

- **Quality:** MSE, FID, CLIP score, text reconstruction, ...
- **Efficiency:** disk space, read/write time, ...

## Coverage

- **VAEs:** Qwen-Image 2.1, FLUX.2, SDXL, ...; video: Wan, LTX-2.5, MiniMax-H3
- **Compression:** `fp32` (baseline), `bf16`, `fp16`, 8-bit (FP8 E4M3/E5M2, MXFP8, INT8 per-tensor/channel/group/asymmetric/MSE-clip/outlier/Hadamard/mu-law, NF8), 4-bit (NVFP4, MXFP4, INT4 group/asymmetric/MSE-clip/Hadamard/outlier, NF4, NF4 double-quant, k-means codebook), HOSVD, PARAFAC, ...

## Contents

- Result tables
- Reproduction code


# Setup


## Env

```bash
uv venv --python=3.12
source .venv/bin/activate
# adjust for your cuda version
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu134
uv pip install -r requirements.txt
uv pip install -e .


```

## Prepare dataset

```bash
cd datasets

# image datasets
git clone https://huggingface.co/datasets/refine-axis/axis-v1_1k
git clone https://huggingface.co/datasets/yunusserhat/Total-Text-Dataset
git clone https://huggingface.co/datasets/Aslan-mingye/OCR-Quality

# video datasets
git clone https://huggingface.co/datasets/madebyollin/movirec
git clone https://huggingface.co/datasets/kk12ff/GasVideo1000

uv run scripts/prepare_datasets.py configs/lsbench_1.0.yaml
```
## Download models

```bash
cd models

# image vaes
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
git clone https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1
git clone https://huggingface.co/stabilityai/sdxl-vae

# only the vae/ subfolder (full repos are huge)
# FLUX.1-schnell: Apache-2.0, FLUX.1-dev shares the same VAE
hf download black-forest-labs/FLUX.1-schnell --include "vae/*" --local-dir FLUX.1-schnell
# gated: accept the license on the model page and run `hf auth login` first
hf download stabilityai/stable-diffusion-3.5-large --include "vae/*" --local-dir stable-diffusion-3.5-large

# video vaes (vae/ subfolders only, diffusers format)
# Wan VAE (Wan2.1, Apache-2.0): https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*" --local-dir Wan2.1-T2V-1.3B-Diffusers
# LTX-2.5 video VAE: gated, accept the license on https://huggingface.co/Lightricks/LTX-2.5-Diffusers and run `hf auth login` first
hf download Lightricks/LTX-2.5-Diffusers --include "vae/*" --local-dir LTX-2.5-Diffusers
# MiniMax-H3 visual VAE (MiniMax H3 Community License): https://huggingface.co/MiniMaxAI/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --include "vae/*" --local-dir MiniMax-H3

# I3D detector for the FVD metric (StyleGAN-V torchscript weights)
mkdir -p i3d && curl -L -o i3d/i3d_torchscript.pt "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"

```

## Run

```bash
# compute all and visualize
uv run python scripts/run_benchmark.py configs/lsbench_1.0.yaml

# visualize only
uv run python scripts/make_report.py results/lsbench_1.0/
```