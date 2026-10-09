"""Ablation plots (bits per value vs metric) of the lloyd_minmax_square features, written to <results_dir>/plots/ablation/.

Meant for results/lsbench_1.2; VALUES_PER_SAMPLE assumes the 32x128x128 FLUX.2 latents."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from lsbench.utils.report import METRIC_SPECS, MetricSpec, _label, pareto_mask

VALUES_PER_SAMPLE = 32 * 128 * 128
SIZE_KEY = "bytes_per_image"
TIME_KEYS = {"serialize_ms_per_image", "deserialize_ms_per_image"}

# Family name -> color; the order is the legend order.
FAMILIES = {
    "global minmax (baseline)": "tab:blue",
    "+ blockwise minmax": "tab:orange",
    "+ Lloyd tables": "tab:green",
}
NAIVE_COLOR = "white"
XTICKS = [2, 3, 4, 5, 6, 8, 10, 16, 32]


def family(storage: str) -> str | None:
    """Family of a storage name, None for naive storages."""
    if storage.endswith("_global"):
        return "global minmax (baseline)"
    if storage.startswith("lloyd_minmax_square_"):
        return "+ Lloyd tables"
    if storage.startswith("minmax_square_"):
        return "+ blockwise minmax"
    return None


def draw_metric(
    ax, wide: pd.DataFrame, metric: str, linestyle: str, label_below: bool = False
) -> None:
    """Pareto fronts per family, other points faded, and labeled naive storages of one metric."""
    spec = METRIC_SPECS.get(metric, MetricSpec())
    bpv = wide["bits_per_value"].to_numpy()
    y = wide[metric].to_numpy()
    fam = np.array([family(s) or "" for s in wide.index])

    for name, color in FAMILIES.items():
        sel = fam == name
        x, v = bpv[sel], y[sel]
        front = pareto_mask(x, v, spec.higher_is_better)
        ax.scatter(x[~front], v[~front], s=10, color=color, alpha=0.3)
        order = np.argsort(x[front])
        ax.plot(
            x[front][order],
            v[front][order],
            linestyle=linestyle,
            marker="o",
            color=color,
            markersize=4,
            linewidth=1.2,
        )

    naive = fam == ""
    ax.scatter(bpv[naive], y[naive], s=20, color=NAIVE_COLOR)
    seen: dict[float, int] = {}  # stack labels of points sharing an x
    for px, py, name in zip(bpv[naive], y[naive], wide.index[naive]):
        k = seen[round(px, 2)] = seen.get(round(px, 2), -1) + 1
        dy = -12 - 11 * k if label_below else 4 + 11 * k
        ax.annotate(
            name,
            (px, py),
            xytext=(4, dy),
            textcoords="offset points",
            fontsize=8,
            color=NAIVE_COLOR,
        )


def style_xaxis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in XTICKS]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("bits per value")
    ax.grid(True, which="major", alpha=0.2)


def family_handles() -> list[Line2D]:
    handles = [
        Line2D([], [], color=c, marker="o", markersize=4, label=n)
        for n, c in FAMILIES.items()
    ]
    handles.append(
        Line2D([], [], color=NAIVE_COLOR, marker="o", linestyle="", label="naive")
    )
    return handles


def plot_psnr_fid(wide: pd.DataFrame, path: Path) -> None:
    """PSNR (left axis, solid) and FID (right axis, dashed) vs bits per value."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax_fid = ax.twinx()
    draw_metric(ax, wide, "psnr", "-")
    draw_metric(ax_fid, wide, "fid", "--", label_below=True)

    style_xaxis(ax)
    ax.margins(y=0.12)
    ax_fid.margins(y=0.12)
    ax.set_ylabel(_label("psnr", True))
    ax_fid.set_ylabel(_label("fid", False))

    handles = family_handles() + [
        Line2D(
            [], [], color="gray", marker="o", markersize=4, label="PSNR (left axis)"
        ),
        Line2D(
            [],
            [],
            color="gray",
            marker="o",
            markersize=4,
            linestyle="--",
            label="FID (right axis)",
        ),
    ]
    ax.legend(handles=handles, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric(wide: pd.DataFrame, metric: str, path: Path) -> None:
    spec = METRIC_SPECS.get(metric, MetricSpec())
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_metric(ax, wide, metric, "-")
    style_xaxis(ax)
    ax.margins(y=0.12)
    ax.set_ylabel(_label(metric, spec.higher_is_better))
    ax.legend(handles=family_handles(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results/lsbench_1.2")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = pd.read_parquet(results_dir / "results.parquet")
    df = df[df["domain"] == "image"]
    out_dir = results_dir / "plots" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for vae, vae_df in df.groupby("vae"):
        wide = vae_df.pivot(index="storage", columns="metric", values="value")
        wide["bits_per_value"] = wide[SIZE_KEY] * 8 / VALUES_PER_SAMPLE
        metrics = [
            m for m in wide.columns if m not in {SIZE_KEY, "bits_per_value"} | TIME_KEYS
        ]
        with plt.style.context("dark_background"):
            if {"psnr", "fid"} <= set(wide.columns):
                plot_psnr_fid(wide, out_dir / f"{vae}_psnr_fid_vs_bits.png")
            for metric in metrics:
                plot_metric(wide, metric, out_dir / f"{vae}_{metric}_vs_bits.png")
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
