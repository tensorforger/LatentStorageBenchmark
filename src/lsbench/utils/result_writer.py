from pathlib import Path

import pandas as pd

COLUMNS = ["domain", "vae", "storage", "metric", "value"]


class ResultWriter:
    """
    Collects benchmark results (floats) in a long-format table and stores them as
    a parquet file (machine-readable) and per-metric markdown pivots (human-readable).
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.records: list[dict] = []

    def add_value(
        self,
        value: float,
        vae_name: str,
        storage_name: str,
        metric_name: str,
        domain_name: str = "image",
    ):
        self.records.append(
            {
                "domain": domain_name,
                "vae": vae_name,
                "storage": storage_name,
                "metric": metric_name,
                "value": float(value),
            }
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=COLUMNS)

    def write(self):
        tables_dir = self.out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        df = self.dataframe
        df.to_parquet(self.out_dir / "results.parquet", index=False)

        for (domain, metric), group in df.groupby(["domain", "metric"]):
            table = group.pivot(index="vae", columns="storage", values="value")
            text = f"# {domain} / {metric}\n\n{table.to_markdown(floatfmt='.6g')}\n"
            (tables_dir / f"{domain}_{metric}.md").write_text(text)
