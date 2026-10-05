import sys

import pandas as pd

from lsbench.utils.report import make_plots


def main():
    run_dir = sys.argv[1]
    make_plots(pd.read_parquet(f"{run_dir}/results.parquet"), run_dir)


if __name__ == "__main__":
    main()
