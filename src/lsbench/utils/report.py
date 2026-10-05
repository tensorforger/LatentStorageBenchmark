from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SIZE_METRIC = "bytes_per_image"
EFFICIENCY_METRICS = {SIZE_METRIC, "serialize_ms_per_image", "deserialize_ms_per_image"}


def make_plots(df: pd.DataFrame, out_dir: str | Path) -> None:
    """
    Plots from the long-format results table:
    - plots/<domain>_<metric>_vs_size.png: quality vs storage size, one point per (vae, storage)
    - plots/<domain>_<metric>_bar.png: grouped bars, vae x storage
    """
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for domain, domain_df in df.groupby("domain"):
        wide = domain_df.pivot(
            index=["vae", "storage"], columns="metric", values="value"
        )
        for metric in wide.columns:
            if metric in EFFICIENCY_METRICS:
                continue
            _plot_bar(wide[metric], plots_dir / f"{domain}_{metric}_bar.png", metric)
            if SIZE_METRIC in wide.columns:
                _plot_vs_size(
                    wide[[SIZE_METRIC, metric]],
                    plots_dir / f"{domain}_{metric}_vs_size.png",
                    metric,
                )


def _plot_bar(series: pd.Series, path: Path, metric: str) -> None:
    ax = series.unstack("storage").plot.bar(rot=0)
    ax.set_ylabel(metric)
    ax.set_xlabel("vae")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def _plot_vs_size(wide: pd.DataFrame, path: Path, metric: str) -> None:
    fig, ax = plt.subplots()
    for vae, vae_df in wide.groupby(level="vae"):
        ax.scatter(vae_df[SIZE_METRIC], vae_df[metric], label=vae)
        for (_, storage), row in vae_df.iterrows():
            ax.annotate(storage, (row[SIZE_METRIC], row[metric]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("bytes per image")
    ax.set_ylabel(metric)
    ax.legend(title="vae")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
