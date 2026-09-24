"""Compare a fresh synthetic run with the archived numerical artifacts."""

import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    artifacts = {
        "synthetic_figure3_data.csv": root / "source_data/synthetic_figure3_data.csv",
        "synthetic_shapley_replicates.csv": (
            root / "analysis_code/synthetic_outputs/synthetic_shapley_replicates.csv"
        ),
    }
    for name, archived in artifacts.items():
        expected = pd.read_csv(archived)
        actual = pd.read_csv(args.output_dir / name)
        assert_frame_equal(expected, actual, check_exact=False, atol=1e-12, rtol=1e-12)
        print(f"PASS {name}: {len(actual)} rows; all columns agree within 1e-12")
    grid = pd.read_csv(args.output_dir / "synthetic_figure3_data.csv")
    print(f"Mean sign reversals: {int(grid.naive_fixed_budget_sign_reversal.sum())}/81")
    print("Negative all-record-budget replicate ranges: "
          f"{int(grid.fixed_budget_interval_below_zero.sum())}/81")


if __name__ == "__main__":
    main()
