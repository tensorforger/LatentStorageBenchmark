# AGENTS.md

Guide for AI coding assistants (and humans) extending **lsbench**: a benchmark of how well VAE latents survive storage/compression schemes (quality vs. size and I/O time). See `README.md` for results and the benchmark phases. The package lives in `src/lsbench/` and is installed editable (`import lsbench...`).

## 0. Environment

- Linux, **fish** shell (no heredocs). Use the project venv: `.venv/bin/python` (system python lacks torch/omegaconf/cv2). Setup is in `README.md`.
- There is no test suite. Verification = run a smoke test (section 4) and check that your component shows up in `results/<name>/tables/`.
- `datasets/`, `prepared_datasets/`, `models/` are git-ignored data. Never modify `datasets/`; never download models at runtime.

## 1. Architecture

Pipeline per VAE: `batch -> vae.encode -> storage.serialize -> bytes -> storage.deserialize -> vae.decode -> metrics(original, reconstruction)`.

| Abstraction | Base class | Folder (`<d>` = `image` or `video`) |
|---|---|---|
| Dataset | `ImageDataset` / `VideoDataset` (`src/lsbench/<d>/dataset.py`) | reads `prepared_datasets/<name>/` |
| VAE | `ImageVAE` / `VideoVAE` (`torch.nn.Module`) | `src/lsbench/<d>/vaes/` |
| Latent storage | `ImageLatentStorage` / `VideoLatentStorage` | `src/lsbench/<d>/latent_storages/` |
| Metric | `ImageMetric` / `VideoMetric` | `src/lsbench/<d>/metrics/` |

Tensor conventions: images `[B, 3, H, W]`, videos `[B, 3, T, H, W]`; RGB, float, range `[0, 1]`. BGR/uint8 only appear inside dataset loading/preparation. Latents are `[B, C, H, W]` / `[B, C, T, H, W]`.

Key files:
- `scripts/run_benchmark.py`: `BenchmarkRunner` encodes each batch **once**, fans out over all storages, records `bytes_per_<domain>`, `serialize_ms_per_<domain>`, `deserialize_ms_per_<domain>` and every metric through `ResultWriter.add_value(value, vae, storage, metric, domain)`. It runs every domain section present in the config (`image`, then `video`). Runs are resumable: (vae, storage) pairs whose metric and size/time rows already exist are skipped, and a VAE with nothing pending is not even loaded. Video VAEs that need padded frame counts pad in `encode`; the runner crops the reconstruction back to the original frame count.
- `scripts/prepare_datasets.py`: raw `datasets/` -> cropped `prepared_datasets/` (images: `NNNNNN.png`; videos: `NNNNNN.npy`, uint8 `[T, H, W, 3]`). Prepared data is the only thing loaders read.
- `scripts/make_report.py <results_dir>`: re-make plots from `results.parquet`. `scripts/make_ablation_plots.py`: phase-3 ablation plots (assumes FLUX.2 latent size).
- `src/lsbench/utils/`: `component_factory.py` (config -> objects), `result_writer.py` (long table -> `results.parquet` + `tables/*.md`), `report.py` (`make_plots`, Pareto fronts, summary score, per-metric `METRIC_SPECS`), `quantization.py` (shared packing/quantization helpers used by many storages), `image_tools.py`, `video_tools.py`, `crop_maximal_rectangle.py`.
- `configs/*.yaml` (OmegaConf): `benchmark_name`, optional global `num_samples` (first N samples of all datasets combined; `null` = all), and `image:` / `video:` sections with `{width, height, [num_frames], batch_size, datasets, vaes, latent_storages, metrics}`. Release configs: `lsbench_1.0` (broad search), `lsbench_1.1` (many VAEs, image + video), `lsbench_1.2` (ablation); `*_fast.yaml` are small smoke-test configs.

### How components are registered

`component_factory.py` imports every module in `lsbench/<domain>/{vaes,latent_storages,metrics}/` and picks the single subclass of the base class whose class constant `NAME` equals the config name. **Adding a file with a unique `NAME` + a config entry is the whole registration**; the runner and factory need no edits. A wrong/missing name raises an error listing the available `NAME`s. Because every module in these folders is imported at discovery time, all top-level imports must resolve.

A config entry is either a plain name (`- mse`) or a mapping with constructor kwargs (`- {name: lloyd_minmax_square, bits: 4, block: 4}`).
- VAEs: `device` is injected automatically if `__init__` has a `device` argument. A VAE entry may also set `batch_size` (overrides the domain `batch_size`; consumed by the runner, not passed to the constructor).
- Metrics: the runner builds one fresh instance per (vae, storage) via `functools.partial(cls, **kwargs)`.

## 2. Rules for adding components

General:
- One class per file, file named `<name>_<kind>.py` (`snake_case`, e.g. `flux_2_vae.py`, `nvfp4_latent_storage.py`, `ssim_metric.py`), absolute imports (`from lsbench.image.vaes.image_vae import ImageVAE`). No `__init__.py` files.
- Subclass the base class and implement **every** abstract method; copy the docstring/shape contract from the base class. Match the style of an existing implementation (`flux_2_vae.py`, `bf16_latent_storage.py`, `lloyd_minmax_square_latent_storage.py`, `mse_metric.py`).
- `NAME` is the config key: short, `snake_case`, unique within the component type and domain.
- Inference only: no grads, `eval()` mode. The runner wraps everything in `torch.no_grad()`.
- A component for the other domain is a separate file in the other folder (the image and video trees mirror each other). Image storages are not picked up by video configs.
- Add new third-party dependencies to `requirements.txt`.

### New VAE (`<d>/vaes/<name>_vae.py`, `<Name>VAE(ImageVAE | VideoVAE)`)
- Constructor: `path_to_model: str` (local path under `models/...`) and `device`; call `super().__init__()`, `.eval()`, move to device.
- The wrapper owns all pre/post-processing: map `[0, 1]` pixels to the model's range, and normalize/denormalize latents (scale/shift/BN statistics, patchify/unpatchify) so that `encode` returns **normalized latents** and `decode` accepts them and returns `[0, 1]` RGB (do not clamp; metrics clamp where needed).
- Cast inputs to the model dtype/device internally; expose `device`/`dtype` properties like `Flux2VAE`.
- Prefer deterministic encoding (`latent_dist.mode()`). `Flux2VAE` still uses `.sample()` (known quirk; changing it changes published numbers).
- Implement `get_latent_channels()`, `get_spatial_compression()` (ratio of the **returned** tensor, after any unpatchify), `get_vae_name()` (return `self.NAME`), and for video `get_temporal_compression()`. These are informational; the runner does not call them.
- Video VAEs with constrained frame counts pad the tail in `encode` (see `utils/video_tools.py::pad_frames_last`); the runner crops after `decode`.
- Add the download command to the README "Models" section.

### New latent storage (`<d>/latent_storages/<name>_latent_storage.py`)
- `serialize(latents) -> bytes` and `deserialize(bytes, device) -> Tensor` are a self-contained round trip: everything needed to decode (shape, scales, codebooks, factors) must be inside the bytes. No state kept between calls (one instance serves many batches).
- `serialize` does `.detach()` itself and handles any device; `deserialize` returns a tensor on `device` with the original shape in a dtype the VAE accepts (fp32 is always safe).
- `len(bytes)` **is** the reported size, so the bytes must be the true compressed payload including all metadata. Use `safetensors` for containers and the helpers in `utils/quantization.py` (`pack_tensors`/`unpack_tensors`, `to_square_groups`, `pack_bits`, ...). Lossy methods are expected.
- Constructor takes only hyperparameters (kwargs from the config mapping). If hyperparameters change results, `get_storage_name()` **must encode them** (`hosvd_r16`, `lloyd_minmax_square_c4_b4_r4`), otherwise result rows collide. Otherwise return `self.NAME`.
- Time is measured around `serialize`/`deserialize` with CUDA sync, so keep the work on the GPU where possible.

### New metric (`<d>/metrics/<name>_metric.py`)
- A **streaming accumulator**: `add_pair(original, reconstructed)` per batch (never buffer images), `calculate_metric() -> float`, `reset()`, `get_metric_name()`. All constructor arguments need defaults or come from the config mapping.
- Inputs are `[B, 3, ...]` in `[0, 1]` on the compute device; clamp internally if needed. Average per sample, then over all samples (see `MSEMetric`). Return a plain Python `float`, handle the empty case.
- Metrics with pretrained weights (LPIPS, FID, FVD) load them from `models/` or a library cache and share the network between instances via a class-level cache (see `LPIPSMetric._shared_networks`): the runner creates one instance per storage and would otherwise exhaust GPU memory.
- If higher is better or the range needs special handling in the summary plots, add an entry to `METRIC_SPECS` in `utils/report.py`.

## 3. Registration checklist

1. Create the file in the right folder with `NAME` set.
2. Add the name (or `{name: ..., kwargs}`) to the matching list in a config. Do not edit the finished release configs (`lsbench_1.0/1.1/1.2.yaml`) unless you intend to extend the published results; copy a config and give it a new `benchmark_name` instead.
3. Smoke test (section 4).
4. Update the README (tables of VAEs / storages / metrics, download commands) if the component is user-facing.

## 4. Running

```bash
.venv/bin/python scripts/prepare_datasets.py configs/lsbench_1.1.yaml            # one-time, writes prepared_datasets/
.venv/bin/python scripts/run_benchmark.py configs/lsbench_fast.yaml --overwrite   # smoke test, results/lsbench_fast/ (git-ignored)
.venv/bin/python scripts/run_benchmark.py configs/lsbench_1.2.yaml                # continue: skips finished rows
.venv/bin/python scripts/make_report.py results/lsbench_1.2                       # re-make plots
```

- Smoke-test new components by adding them to a copy of `configs/lsbench_fast.yaml` (5 samples) and running it with `--overwrite`.
- Default mode is continue: re-running a finished benchmark does nothing; adding entries computes only the missing rows. Results are keyed by (domain, vae, storage, metric) only, so after changing `num_samples`, datasets or other data settings use `--overwrite` or a new `benchmark_name`.
- `--overwrite` deletes the whole `results/<benchmark_name>/`. Use it only for scratch/smoke configs, never for the release results (`lsbench_1.0/1.1/1.2`) unless intended.
- Uses CUDA when available, otherwise CPU (very slow for large VAEs).
- `results/*fast/` is git-ignored; release results (`results/lsbench_1.x/`) are committed.
