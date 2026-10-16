"""
Submission contract. Every file written by the pipeline (and every file downloaded by
``make gcp-fetch``) is validated here.

    python -m src.submission submissions/submission_X.csv [--raw-dir data/raw]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

SUBMISSION_COLUMNS = ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]
EXPECTED_ROWS = 5488


def opportunity_classes(raw_dir: Path) -> List[str]:
    return json.loads((Path(raw_dir) / "label_config.json").read_text())["opportunity_labels"]


def read_sample_submission(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(raw_dir) / "SampleSubmission.csv", dtype={"ID": str})


def validate_submission(sub_df: pd.DataFrame, sample_sub: pd.DataFrame, classes: List[str],
                        expected_rows: int | None = EXPECTED_ROWS) -> None:
    """Pre-flight assertions. ``expected_rows`` is 5488 for the real competition."""
    n_rows = expected_rows if expected_rows is not None else len(sample_sub)
    assert sub_df.shape == (n_rows, 4), f"Expected ({n_rows}, 4), got {sub_df.shape}"
    assert list(sub_df.columns) == SUBMISSION_COLUMNS, f"Bad columns {list(sub_df.columns)}"
    assert len(sample_sub) == n_rows, f"SampleSubmission has {len(sample_sub)} rows, expected {n_rows}"
    assert (sub_df["ID"].astype(str).values == sample_sub["ID"].astype(str).values).all(), "ID ordering mismatch!"
    assert sub_df.isna().sum().sum() == 0, "NaN values detected!"
    assert (sub_df["CLV_fuel"] >= 0).all() and (sub_df["CLV_nonfuel"] >= 0).all(), "Negative CLV predictions!"
    assert set(sub_df["Opportunity"].unique()).issubset(set(classes)), "Invalid Opportunity label!"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a Sasol submission CSV")
    ap.add_argument("path")
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args(argv)
    sample = read_sample_submission(Path(args.raw_dir))
    sub = pd.read_csv(args.path, dtype={"ID": str})
    try:
        validate_submission(sub, sample, opportunity_classes(Path(args.raw_dir)), expected_rows=len(sample))
    except AssertionError as e:
        print(f"INVALID submission {args.path}: {e}", file=sys.stderr)
        return 1
    dist = sub["Opportunity"].value_counts(normalize=True).head(5).round(3).to_dict()
    print(f"OK {args.path}: shape={sub.shape}, CLV_fuel mean={sub.CLV_fuel.mean():.4f}, "
          f"CLV_nonfuel mean={sub.CLV_nonfuel.mean():.4f}, top classes={dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
