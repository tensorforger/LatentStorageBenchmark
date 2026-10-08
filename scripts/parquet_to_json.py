#!/usr/bin/env python
"""Convert parquet results files to JSON."""

import json
from pathlib import Path
import pandas as pd


def parquet_to_json(parquet_path: str, json_path: str = None) -> None:
    """Read a parquet file and write it as JSON."""
    parquet_path = Path(parquet_path)
    if json_path is None:
        json_path = parquet_path.with_suffix(".json")

    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Converted {parquet_path} -> {json_path} ({len(records)} rows)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parquet_to_json.py <parquet_file> [json_file]")
        sys.exit(1)

    parquet_to_json(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
