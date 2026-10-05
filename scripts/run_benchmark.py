import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, List

import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from lsbench.image.dataset import ImageDataset
from lsbench.image.vaes.image_vae import ImageVAE
from lsbench.image.metrics.image_metric import ImageMetric
from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.image.vaes.flux_2_vae import Flux2VAE
from lsbench.image.vaes.no_vae import NoVAE

from lsbench.image.metrics.mse_metric import MSEMetric
from lsbench.image.latent_storages.fp32_latent_storage import FP32LatentStorage
from lsbench.image.latent_storages.fp16_latent_storage import FP16LatentStorage
from lsbench.image.latent_storages.bf16_latent_storage import BF16LatentStorage

from lsbench.utils.report import make_plots
from lsbench.utils.result_writer import ResultWriter


class ImageBenchmarkRunner:
    """
    Runs the benchmark for one VAE and all storages.
    Each batch is encoded once and then fanned out over storages.
    Metrics must be streaming accumulators (no image buffering), one instance per storage.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        vae: ImageVAE,
        storages: List[ImageLatentStorage],
        metric_factories: List[Callable[[], ImageMetric]],
        writer: ResultWriter,
        device: torch.device,
    ):
        self.dataloader = dataloader
        self.vae = vae
        self.storages = storages
        self.metric_factories = metric_factories
        self.writer = writer
        self.device = device

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def run(self):
        names = [s.get_storage_name() for s in self.storages]
        metrics = {n: [f() for f in self.metric_factories] for n in names}
        total_bytes = defaultdict(int)
        total_ser_s = defaultdict(float)
        total_deser_s = defaultdict(float)
        num_images = 0

        for batch in tqdm(self.dataloader, desc=self.vae.get_vae_name()):
            batch = batch.to(self.device)
            num_images += batch.shape[0]
            latents = self.vae.encode(batch)

            for storage, name in zip(self.storages, names):
                self._sync()
                t0 = perf_counter()
                serialized = storage.serialize(latents)
                self._sync()
                t1 = perf_counter()
                deserialized = storage.deserialize(serialized, device=self.device)
                self._sync()
                t2 = perf_counter()

                total_bytes[name] += len(serialized)
                total_ser_s[name] += t1 - t0
                total_deser_s[name] += t2 - t1

                reconstructed = self.vae.decode(deserialized)
                for metric in metrics[name]:
                    metric.add_pair(
                        original_image=batch, reconstructed_image=reconstructed
                    )

        vae_name = self.vae.get_vae_name()
        for name in names:
            for metric in metrics[name]:
                self.writer.add_value(
                    metric.calculate_metric(), vae_name, name, metric.get_metric_name()
                )
            self.writer.add_value(
                total_bytes[name] / num_images, vae_name, name, "bytes_per_image"
            )
            self.writer.add_value(
                1000 * total_ser_s[name] / num_images,
                vae_name,
                name,
                "serialize_ms_per_image",
            )
            self.writer.add_value(
                1000 * total_deser_s[name] / num_images,
                vae_name,
                name,
                "deserialize_ms_per_image",
            )


def main():
    cfg = OmegaConf.load(sys.argv[1])
    image_cfg: DictConfig = cfg.image

    out_dir = Path("results") / cfg.benchmark_name
    out_dir.mkdir(parents=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    writer = ResultWriter(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        ImageDataset(cfg=image_cfg), batch_size=image_cfg.batch_size, shuffle=False
    )

    vaes = [
        NoVAE(),
        Flux2VAE(path_to_model="models/FLUX.2-klein-4B/vae"),
    ]

    storages = [
        FP32LatentStorage(),
        FP16LatentStorage(),
        BF16LatentStorage(),
    ]

    metric_factories = [
        MSEMetric,
    ]

    for vae in vaes:
        runner = ImageBenchmarkRunner(
            dataloader, vae, storages, metric_factories, writer, device
        )
        runner.run()
        writer.write()  # incremental, so a crash keeps finished VAEs

    make_plots(writer.dataframe, out_dir)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
