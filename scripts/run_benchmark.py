import argparse
import shutil
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

from lsbench.utils.component_factory import (
    configure_latent_storage,
    configure_metric_factory,
    configure_vae,
)
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


RESULT_KEYS_PER_STORAGE = [
    "bytes_per_image",
    "serialize_ms_per_image",
    "deserialize_ms_per_image",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="erase previous results and start fresh (default: continue, skipping finished rows)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    image_cfg: DictConfig = cfg.image

    out_dir = Path("results") / cfg.benchmark_name
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    writer = ResultWriter(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        ImageDataset(cfg=image_cfg, num_samples=cfg.get("num_samples")),
        batch_size=image_cfg.batch_size,
        shuffle=False,
    )

    storages = [configure_latent_storage(e) for e in image_cfg.latent_storages]
    metric_factories = [configure_metric_factory(e) for e in image_cfg.metrics]

    metric_names = [f().get_metric_name() for f in metric_factories]
    required_keys = metric_names + RESULT_KEYS_PER_STORAGE

    for vae_entry in image_cfg.vaes:
        vae_name = vae_entry if isinstance(vae_entry, str) else vae_entry.name
        pending = [
            s
            for s in storages
            if not all(
                writer.has_value(vae_name, s.get_storage_name(), k)
                for k in required_keys
            )
        ]
        if not pending:
            print(f"Skipping {vae_name}: all results already computed")
            continue

        vae = configure_vae(vae_entry, device)
        runner = ImageBenchmarkRunner(
            dataloader, vae, pending, metric_factories, writer, device
        )
        runner.run()
        writer.write()  # incremental, so a crash keeps finished VAEs
        del runner, vae  # free VRAM before loading the next VAE

    make_plots(writer.dataframe, out_dir)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
