# Latent Storage Benchmark

> **How much can VAE latents be compressed before decoded images degrade?**
>
> VAEs compress images, but naive `fp32` latent storage often uses more disk space than the original JPEG. This benchmark measures the quality and efficiency trade-offs of different compression strategies.

## Metrics

- **Quality:** MSE, FID, CLIP score, text reconstruction, ...
- **Efficiency:** disk space, read/write time, ...

## Coverage

- **VAEs:** Qwen-Image 2.1, FLUX.2, SDXL, ...
- **Compression:** `fp32` (baseline), `bf16`, `int8`, HOSVD, PARAFAC, ...

## Contents

- Result tables
- Reproduction code