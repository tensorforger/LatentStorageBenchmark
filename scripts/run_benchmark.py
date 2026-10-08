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
from lsbench.video.dataset import VideoDataset

from lsbench.utils.component_factory import (
    configure_latent_storage,
    configure_metric_factory,
    configure_vae,
)
from lsbench.utils.report import efficiency_keys, make_plots
from lsbench.utils.result_writer import ResultWriter

DATASET_CLASSES = {"image": ImageDataset, "video": VideoDataset}


class BenchmarkRunner:
    """
    Runs the benchmark of one domain (image or video) for one VAE and all storages.
    Each batch is encoded once and then fanned out over storages.
    Metrics must be streaming accumulators (no buffering), one instance per storage.
    """

    def __init__(
        self,
        domain: str,
        dataloader: DataLoader,
        vae,
        storages: list,
        metric_factories: List[Callable],
        writer: ResultWriter,
        device: torch.device,
    ):
        self.domain = domain
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
        num_items = 0

        for batch in tqdm(
            self.dataloader, desc=f"{self.domain}/{self.vae.get_vae_name()}"
        ):
            batch = batch.to(self.device)
            num_items += batch.shape[0]
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
                if self.domain == "video":
                    # VAEs with constrained frame counts pad in encode
                    reconstructed = reconstructed[:, :, : batch.shape[2]]
                for metric in metrics[name]:
                    metric.add_pair(batch, reconstructed)

        vae_name = self.vae.get_vae_name()
        size_key, ser_key, deser_key = efficiency_keys(self.domain)
        for name in names:
            for metric in metrics[name]:
                self.writer.add_value(
                    metric.calculate_metric(),
                    vae_name,
                    name,
                    metric.get_metric_name(),
                    self.domain,
                )
            self.writer.add_value(
                total_bytes[name] / num_items, vae_name, name, size_key, self.domain
            )
            self.writer.add_value(
                1000 * total_ser_s[name] / num_items,
                vae_name,
                name,
                ser_key,
                self.domain,
            )
            self.writer.add_value(
                1000 * total_deser_s[name] / num_items,
                vae_name,
                name,
                deser_key,
                self.domain,
            )


def run_domain(
    domain: str,
    domain_cfg: DictConfig,
    num_samples: int | None,
    writer: ResultWriter,
    device: torch.device,
) -> None:
    dataset = DATASET_CLASSES[domain](cfg=domain_cfg, num_samples=num_samples)

    storages = [configure_latent_storage(e, domain) for e in domain_cfg.latent_storages]
    metric_factories = [configure_metric_factory(e, domain) for e in domain_cfg.metrics]

    metric_names = [f().get_metric_name() for f in metric_factories]
    required_keys = metric_names + list(efficiency_keys(domain))

    for vae_entry in domain_cfg.vaes:
        vae_name = vae_entry if isinstance(vae_entry, str) else vae_entry.name
        pending = [
            s
            for s in storages
            if not all(
                writer.has_value(vae_name, s.get_storage_name(), k, domain)
                for k in required_keys
            )
        ]
        if not pending:
            print(f"Skipping {domain}/{vae_name}: all results already computed")
            continue

        vae = configure_vae(vae_entry, device, domain)
        batch_size = (
            domain_cfg.batch_size
            if isinstance(vae_entry, str)
            else vae_entry.get("batch_size", domain_cfg.batch_size)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        runner = BenchmarkRunner(
            domain, dataloader, vae, pending, metric_factories, writer, device
        )
        runner.run()
        writer.write()  # incremental, so a crash keeps finished VAEs
        del runner, vae  # free VRAM before loading the next VAE


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

    out_dir = Path("results") / cfg.benchmark_name
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    writer = ResultWriter(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for domain in DATASET_CLASSES:
        if domain in cfg:
            run_domain(domain, cfg[domain], cfg.get("num_samples"), writer, device)

    make_plots(writer.dataframe, out_dir)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
