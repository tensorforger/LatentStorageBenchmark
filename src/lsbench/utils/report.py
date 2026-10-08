from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-12


def efficiency_keys(domain: str) -> tuple[str, str, str]:
    """Per-storage result keys (size, serialize time, deserialize time) of a domain."""
    return (
        f"bytes_per_{domain}",
        f"serialize_ms_per_{domain}",
        f"deserialize_ms_per_{domain}",
    )


@dataclass(frozen=True)
class MetricSpec:
    higher_is_better: bool = False
    # Values beyond this are indistinguishable in practice; clipped before scoring the summary.
    floor: float | None = None
    ceiling: float | None = None
    weight: float = 1.0


# Metrics that are not listed use MetricSpec() (lower is better, no floor, weight 1).
METRIC_SPECS: dict[str, MetricSpec] = {
    "mse": MetricSpec(floor=1e-6),
    "psnr": MetricSpec(higher_is_better=True, ceiling=60.0),  # 60 dB == mse 1e-6
    "ssim": MetricSpec(higher_is_better=True, ceiling=0.9999),
    "lpips": MetricSpec(floor=1e-4),
    "fid": MetricSpec(),
}


def make_plots(
    df: pd.DataFrame,
    out_dir: str | Path,
    specs: dict[str, MetricSpec] | None = None,
) -> None:
    """
    Plots from the long-format results table (one point per (vae, storage), Pareto frontier per vae),
    written to plots/all_vaes/ (every vae on one plot) and plots/<vae>/ (that vae only):
    - <domain>_<metric>_vs_size.png: metric vs storage size
    - <domain>_summary.png: weighted summary score vs storage size
    """
    specs = {**METRIC_SPECS, **(specs or {})}
    plots_dir = Path(out_dir) / "plots"

    for domain, domain_df in df.groupby("domain"):
        wide = domain_df.pivot(
            index=["vae", "storage"], columns="metric", values="value"
        )
        size_metric = efficiency_keys(domain)[0]
        if size_metric not in wide.columns:
            continue
        metrics = [m for m in wide.columns if m not in efficiency_keys(domain)]
        if not metrics:
            continue
        summary = summary_score(wide[metrics], specs)
        vaes = wide.index.get_level_values("vae").unique()
        for group, sel in [("all_vaes", slice(None))] + [(v, [v]) for v in vaes]:
            group_dir = plots_dir / group
            group_dir.mkdir(parents=True, exist_ok=True)
            part = wide.loc[sel]
            for metric in metrics:
                spec = specs.get(metric, MetricSpec())
                _plot_vs_size(
                    part[size_metric],
                    part[metric],
                    group_dir / f"{domain}_{metric}_vs_size.png",
                    ylabel=_label(metric, spec.higher_is_better),
                    higher_is_better=spec.higher_is_better,
                    domain=domain,
                )
            _plot_vs_size(
                part[size_metric],
                summary.loc[sel],
                group_dir / f"{domain}_summary.png",
                ylabel="summary score (higher is better)",
                higher_is_better=True,
                domain=domain,
            )


def summary_score(wide: pd.DataFrame, specs: dict[str, MetricSpec]) -> pd.Series:
    """
    Weighted mean of per-metric scores in [0, 1] (1 = best), indexed like `wide` ((vae, storage)).
    Each metric is clipped at its floor, log-transformed (metrics span decades) and min-max
    scaled within each vae, so the score compares storages of the same vae and no metric
    dominates because of its units.
    """
    scores, weights = {}, {}
    for metric in wide.columns:
        spec = specs.get(metric, MetricSpec())
        v = wide[metric].clip(lower=max(spec.floor or 0.0, EPS), upper=spec.ceiling)
        t = np.log10(v) * (-1 if spec.higher_is_better else 1)  # lower t is better
        grouped = t.groupby(level="vae")
        lo, hi = grouped.transform("min"), grouped.transform("max")
        span = (hi - lo).where(hi > lo, 1.0)
        scores[metric] = 1 - (t - lo) / span
        weights[metric] = spec.weight
    score_df = pd.DataFrame(scores)
    w = pd.Series(weights)
    valid = score_df.notna()
    return (score_df.fillna(0) * w).sum(axis=1) / (valid * w).sum(axis=1)


def pareto_mask(x: np.ndarray, y: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """True for points not dominated by another point with smaller-or-equal x and a better y."""
    y = y if higher_is_better else -y
    mask = np.zeros(len(x), dtype=bool)
    best = -np.inf
    for i in np.lexsort((-y, x)):
        if y[i] > best:
            mask[i] = True
            best = y[i]
    return mask


def _label(metric: str, higher_is_better: bool) -> str:
    return f"{metric} ({'higher' if higher_is_better else 'lower'} is better)"


def _plot_vs_size(
    size: pd.Series,
    value: pd.Series,
    path: Path,
    ylabel: str,
    higher_is_better: bool,
    domain: str,
) -> None:
    fig, ax = plt.subplots()
    for i, (vae, idx) in enumerate(size.groupby(level="vae").groups.items()):
        color = plt.get_cmap("tab10")(i % 10)
        x, y = size.loc[idx].to_numpy(), value.loc[idx].to_numpy()
        names = idx.get_level_values("storage")
        front = pareto_mask(x, y, higher_is_better)

        ax.scatter(x[~front], y[~front], s=10, color=color, alpha=0.5)
        order = np.argsort(x[front])
        fx, fy = x[front][order], y[front][order]
        ax.plot(fx, fy, "-o", color=color, markersize=5, linewidth=1, label=vae)
        for px, py, name in zip(fx, fy, names[front][order]):
            ax.annotate(
                name,
                (px, py),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=color,
                rotation=30,
                rotation_mode="anchor",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"bytes per {domain}")
    ax.set_ylabel(ylabel)
    ax.legend(title="vae", fontsize=7, title_fontsize=8, markerscale=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
