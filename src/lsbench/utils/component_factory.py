import importlib
import inspect
import pkgutil
from functools import partial
from typing import Any, Callable

import torch
from omegaconf import OmegaConf

from lsbench.image.latent_storages.image_latent_storage import ImageLatentStorage
from lsbench.image.metrics.image_metric import ImageMetric
from lsbench.image.vaes.image_vae import ImageVAE

# Config entry: "name" or {name: ..., **constructor_kwargs}.


def _parse_entry(entry) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    entry = OmegaConf.to_container(entry) if OmegaConf.is_config(entry) else dict(entry)
    return entry.pop("name"), entry


def _find_class(package_name: str, base: type, name: str) -> type:
    """Imports every module of the package and returns the `base` subclass with NAME == name."""
    package = importlib.import_module(package_name)
    found, available = [], []
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, base)
                and cls is not base
                and cls.__module__ == module.__name__
            ):
                available.append(cls.NAME)
                if cls.NAME == name:
                    found.append(cls)
    if len(found) != 1:
        raise ValueError(
            f"Expected exactly one {base.__name__} with NAME={name!r} in {package_name}, "
            f"found {len(found)}. Available: {sorted(available)}"
        )
    return found[0]


def _with_device(cls: type, kwargs: dict, device: torch.device) -> dict:
    if "device" in inspect.signature(cls.__init__).parameters:
        kwargs = {"device": device, **kwargs}
    return kwargs


def configure_vae(entry, device: torch.device) -> ImageVAE:
    name, kwargs = _parse_entry(entry)
    cls = _find_class("lsbench.image.vaes", ImageVAE, name)
    return cls(**_with_device(cls, kwargs, device))


def configure_latent_storage(entry) -> ImageLatentStorage:
    name, kwargs = _parse_entry(entry)
    return _find_class("lsbench.image.latent_storages", ImageLatentStorage, name)(
        **kwargs
    )


def configure_metric_factory(entry) -> Callable[[], ImageMetric]:
    name, kwargs = _parse_entry(entry)
    return partial(_find_class("lsbench.image.metrics", ImageMetric, name), **kwargs)
