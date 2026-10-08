"""Ablation plots for lsbench_1.2: Pareto curves of the lloyd_minmax_square features vs bits per value."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsbench.utils.report import METRIC_SPECS, MetricSpec, _label, pareto_mask

VALUES_PER_SAMPLE = 32 * 128 * 128
SIZE_KEY = "bytes_per_image"
TIME_KEYS = {"serialize_ms_per_image", "deserialize_ms_per_image"}

# Family name -> color; the order is the legend order.
FAMILIES = {
    "bits + global minmax": "tab:blue",
    "+ blockwise minmax": "tab:orange",
    "+ blockwise minmax + Lloyd tables": "tab:green",
}
NAIVE_COLOR = "black"


def family(storage: str) -> str | None:
    """Family of a storage name, None for naive storages."""
    if storage.endswith("_global"):
        return "bits + global minmax"
    if storage.startswith("lloyd_minmax_square_"):
        return "+ blockwise minmax + Lloyd tables"
    if storage.startswith("minmax_square_"):
        return "+ blockwise minmax"
    return None


def plot_metric(wide: pd.DataFrame, metric: str, path: Path) -> None:
    spec = METRIC_SPECS.get(metric, MetricSpec())
    bpv = wide["bits_per_value"].to_numpy()
    y = wide[metric].to_numpy()
    fam = np.array([family(s) or "" for s in wide.index])

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, color in FAMILIES.items():
        sel = fam == name
        x, v = bpv[sel], y[sel]
        front = pareto_mask(x, v, spec.higher_is_better)
        ax.scatter(x[~front], v[~front], s=10, color=color, alpha=0.35)
        order = np.argsort(x[front])
        ax.plot(
            x[front][order],
            v[front][order],
            "-o",
            color=color,
            markersize=4,
            linewidth=1.2,
            label=name,
        )

    naive = fam == ""
    ax.scatter(bpv[naive], y[naive], s=30, color=NAIVE_COLOR, marker="s", label="naive")
    seen: dict[float, int] = {}  # stack labels of points sharing an x
    for px, py, name in zip(bpv[naive], y[naive], wide.index[naive]):
        k = seen[round(px, 2)] = seen.get(round(px, 2), -1) + 1
        ax.annotate(
            name,
            (px, py),
            xytext=(4, 4 + 11 * k),
            textcoords="offset points",
            fontsize=8,
            color=NAIVE_COLOR,
        )

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.margins(y=0.12)
    ax.set_xlabel("bits per value")
    ax.set_ylabel(_label(metric, spec.higher_is_better))
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
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
        for metric in metrics:
            plot_metric(wide, metric, out_dir / f"{vae}_{metric}_vs_bits.png")
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
