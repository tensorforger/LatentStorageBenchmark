# Latent Storage Benchmark

**How much can VAE latents be compressed before decoded images degrade?**

> VAEs compress images, but naive `fp32` latent storage often uses more disk space than the original JPEG. This benchmark measures the quality and efficiency trade-offs of different compression strategies.

While the core purpose of this is to benchmark **Storage** efficency, it can also be used as general **VAE** benchmark (no compression case).

## Metrics

- **Quality:** MSE, FID, CLIP score, text reconstruction, ...
- **Efficiency:** disk space, read/write time, ...

## Coverage

- **VAEs:** Qwen-Image 2.1, FLUX.2, SDXL, ...
- **Compression:** `fp32` (baseline), `bf16`, `int8`, HOSVD, PARAFAC, ...

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

```

## Run

```bash
uv run python scripts/run_benchmark.py configs/lsbench_1.0.yaml
```