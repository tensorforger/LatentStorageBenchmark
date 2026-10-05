# AGENTS.md

Latent Storage Benchmark (`lsbench`): measures how well VAE latents survive different storage/compression schemes (quality vs. disk size and I/O time). Package lives in `src/lsbench/`, installed editable (`import lsbench...`).

## 1. Architecture

Pipeline per VAE: `image -> vae.encode -> storage.serialize -> bytes -> storage.deserialize -> vae.decode -> metrics(original, reconstructed)`.

| Abstraction | Base class (abstract) | Implementations |
|---|---|---|
| Dataset | `lsbench/image/dataset.py::ImageDataset` | reads `prepared_datasets/<name>/*.png`, returns `[3,H,W]` float RGB in [0,1] |
| VAE | `lsbench/image/vaes/image_vae.py::ImageVAE` (`torch.nn.Module`) | `no_vae.py` (identity + ImageNet norm), `flux_2_vae.py` |
| Latent storage | `lsbench/image/latent_storages/image_latent_storage.py::ImageLatentStorage` | `fp32_`, `fp16_`, `bf16_latent_storage.py` (safetensors bytes) |
| Metric | `lsbench/image/metrics/image_metric.py::ImageMetric` | `mse_metric.py` |

- `scripts/run_benchmark.py::ImageBenchmarkRunner` is the orchestrator: for each batch it encodes **once**, then fans out over all storages; it records `bytes_per_image`, `serialize_ms_per_image`, `deserialize_ms_per_image` plus every metric via `ResultWriter.add_value(value, vae, storage, metric)`. Runs are resumable: `ResultWriter` loads an existing `results.parquet`, and `main()` skips every (vae, storage) whose metric + size/time rows already exist (a VAE with nothing pending is not even loaded).
- `lsbench/utils/`: `result_writer.py` (long-format table -> `results/<benchmark_name>/results.parquet` + `tables/*.md`), `report.py::make_plots`, `image_tools.py` (torch/np/cv2 conversions), `crop_maximal_rectangle.py`.
- `configs/lsbench_1.0.yaml` (OmegaConf): `benchmark_name`, optional global `num_samples` (first N images of all datasets combined in config order; `null`/absent = all), `image.{width,height,batch_size,datasets,vaes,latent_storages,metrics}`, and a `video` section (not used yet). `configs/lsbench_fast.yaml` is the same with `num_samples: 5` and its own `benchmark_name` for quick tests.
- **Components are built from the config** by `lsbench/utils/component_factory.py` (`configure_vae(entry, device)`, `configure_latent_storage(entry)`, `configure_metric_factory(entry)`). It imports every module in `lsbench/image/{vaes,latent_storages,metrics}/` and picks the single subclass whose class constant `NAME` equals the config name. `main()` has no component imports: adding a file + a config entry is the whole registration.
- Config entry is either a plain name (`- mse`) or a mapping with constructor kwargs (`- {name: flux_2, path_to_model: models/FLUX.2-klein-4B/vae}`). `device` is injected automatically into VAEs whose `__init__` has a `device` argument.
- Every VAE/storage/metric class must define `NAME = "..."` as a class constant; `get_*_name()` returns `self.NAME`.
- Only the `image` domain is implemented; `src/lsbench/video/` is empty (`ResultWriter` already has a `domain_name` arg).
- Data flow is two-stage: raw `datasets/` (never modify) -> `scripts/prepare_datasets.py` -> `prepared_datasets/` (only thing the loader reads). `models/` holds local HF checkpoints (git-cloned).

## 2. Rules for adding components

General (all components):
- One class per file, `snake_case` file named after the class, in the matching folder; import absolute (`from lsbench.image.vaes.image_vae import ImageVAE`). No `__init__.py` files are used.
- Subclass the base and implement **every** abstract method; copy the docstring/shape contract from the base class. Match the style of existing implementations (`flux_2_vae.py`, `bf16_latent_storage.py`, `mse_metric.py`).
- Class constant `NAME` is the key in the config: short, `snake_case`, unique within its component type (e.g. `flux_2`, `bf16`, `mse`). `get_*_name()` is the key in results tables/plots and returns `self.NAME` unless hyperparameters must be encoded (see storages).
- Tensor convention everywhere: `[B, C, H, W]`; images are RGB, float, range `[0, 1]`. Never use BGR/uint8 outside of dataset loading.
- Inference only: no grads, `eval()` mode, no training code. Runner already wraps in `torch.no_grad()`.
- Add heavy dependencies to `requirements.txt` (and import lazily if they are optional).

**New VAE** (`image/vaes/<name>_vae.py`, class `<Name>VAE(ImageVAE)`):
- Constructor takes `path_to_model: str` (local path under `models/<repo>/...`, never download at runtime) and `device`; call `super().__init__()`, `.eval()`, move to device. `path_to_model` comes from the config mapping entry.
- The wrapper owns all pre/post-processing: map `[0,1]` pixels to the model's expected range, and normalize/denormalize latents (scale/shift/BN stats, patchify/unpatchify) so that `encode` returns **normalized latents** and `decode` accepts them. `decode` must return `[0,1]` RGB (do not clamp; metrics/plots handle it).
- Cast inputs to the model dtype/device internally (`image.to(self.dtype)`); expose `device`/`dtype` properties like `Flux2VAE`.
- Sample deterministically if possible (`latent_dist.mode()`); `Flux2VAE` currently uses `.sample()`.
- Implement `get_latent_channels()` (channels of the returned latents), `get_spatial_compression()` (pixel/latent spatial ratio of the **returned** tensor, after any unpatchify), `get_vae_name()`.
- Download instructions go in `README.md` "Download models" (`git clone` into `models/`).

**New latent storage** (`image/latent_storages/<name>_latent_storage.py`, class `<Name>LatentStorage(ImageLatentStorage)`):
- `serialize(latents) -> bytes` and `deserialize(bytes, device) -> Tensor` must be a self-contained round trip: all info needed to decode (shape, scales, codebooks, factors) must be inside the bytes. Nothing may be kept as instance state between calls (the same instance serves many batches).
- `serialize` must `.detach().cpu()` itself; `deserialize` returns a tensor on `device` of a dtype the VAE can accept (VAE casts it; fp32 is always safe). Output shape == input shape.
- `len(bytes)` is the reported size, so the bytes must be the actual compressed payload. Use safetensors for containers (see `bf16_latent_storage.py`); lossy methods (int8, HOSVD, PARAFAC, ...) are expected, lossless are fine.
- Constructor takes only hyperparameters (passed as kwargs from a config mapping entry, e.g. `{name: hosvd, rank: 16}`). `NAME` is the class key (`hosvd`); if hyperparameters change results, `get_storage_name()` must include them (`hosvd_r16`) so table rows stay unique.

**New metric** (`image/metrics/<name>_metric.py`, class `<Name>Metric(ImageMetric)`):
- Must be a **streaming accumulator**: `add_pair(original, reconstructed)` per batch (do not buffer images), `calculate_metric() -> float`, `reset()`, `get_metric_name()`. Runner creates one instance per storage via a factory (`functools.partial(cls, **kwargs)` built by `configure_metric_factory`), so every constructor argument must have a default or come from the config mapping entry.
- Inputs are `[B,C,H,W]` RGB `[0,1]` on the compute device; clamp to `[0,1]` internally if the metric needs it. Average per-sample then over all samples (see `MSEMetric`). Return a plain Python `float`; handle the empty case.
- Metrics needing pretrained weights (CLIP, FID, OCR) load them from `models/` or a library cache in `__init__`.

Registration checklist (no edits to `scripts/run_benchmark.py` needed): (1) create the file in the right folder with `NAME` set, (2) add the name (or `{name: ..., kwargs}` mapping) to the matching list in `configs/lsbench_1.0.yaml`, (3) smoke-test with `configs/lsbench_fast.yaml --overwrite` (below); a wrong/missing name raises an error listing the available `NAME`s, (4) update the README "Coverage"/setup if needed. Any new top-level dependency must be importable at discovery time, since all modules in the folder are imported.

## 3. Running

Environment (Linux, fish shell, repo root as cwd; `uv` available):
```bash
uv venv --python=3.12 && source .venv/bin/activate      # or call .venv/bin/python directly
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu134   # match CUDA
uv pip install -r requirements.txt && uv pip install -e .
```
Always use the project venv (`.venv/bin/python`); system `python` lacks torch/omegaconf/cv2.

Data (see README for the `git clone` commands into `datasets/` and `models/`):
```bash
.venv/bin/python scripts/prepare_datasets.py configs/lsbench_1.0.yaml   # one-time, writes prepared_datasets/
```
Known data quirks: `Total-Text-Dataset/Images/Train/img61.JPG` is a git-lfs pointer (skipped with a warning; fix via `git lfs pull`); `__MACOSX/._*` files are skipped.

Benchmark and report:
```bash
.venv/bin/python scripts/run_benchmark.py configs/lsbench_1.0.yaml              # continue: skips finished rows, nothing is overwritten
.venv/bin/python scripts/run_benchmark.py configs/lsbench_1.0.yaml --overwrite  # deletes results/<benchmark_name> and starts fresh
.venv/bin/python scripts/run_benchmark.py configs/lsbench_fast.yaml --overwrite  # fast smoke test (5 samples) -> results/lsbench_fast/
.venv/bin/python scripts/make_report.py results/lsbench_1.0                       # re-make plots from results.parquet
```
- Default mode is continue: re-running a finished benchmark does nothing; adding a VAE/storage/metric to the config computes only the missing rows. Results are keyed by (vae, storage, metric) only, so after changing `num_samples`, datasets or other data settings use `--overwrite` (or a new `benchmark_name`), otherwise old and new rows get mixed.
- `--overwrite` removes the whole `results/<benchmark_name>` dir: use it only for test configs such as `lsbench_fast.yaml`, never on the full `lsbench_1.0` results unless intended.
- For smoke tests of new components, add them to `configs/lsbench_fast.yaml` (or a copy) and run the fast command; there is no test suite yet.
- Uses CUDA when available, otherwise CPU (very slow for large VAEs).
