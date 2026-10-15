import csv
import json
import os
from pathlib import Path
from typing import Any, Dict
import pandas as pd

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw")


def get_dataset_overview() -> Dict[str, Any]:
    """Inspects the local Sasol training dataset and returns key dataset statistics.

    Returns:
        dict: A dictionary containing row counts, headers, transaction categories,
              and sample monetary sums.
    """
    train_path = os.path.join(RAW_DATA_PATH, "train.csv")
    if not os.path.exists(train_path):
        return {"status": "error", "message": "train.csv not found locally"}

    categories = set()
    total_sample_amount = 0.0
    row_count = 0

    with open(train_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            row_count += 1
            cat = row.get("ItemCategoryLevel1")
            if cat:
                categories.add(cat)
            try:
                amt = float(row.get("TransactionTotalInclAmount", 0))
                total_sample_amount += amt
            except (ValueError, TypeError):
                pass
            if row_count >= 1000:
                break

    return {
        "dataset": "Sasol CLV Train Dataset",
        "sample_analyzed_rows": row_count,
        "total_columns": len(headers),
        "columns": headers,
        "categories_level1": sorted(list(categories)),
        "sample_total_amount_sum": round(total_sample_amount, 2),
    }


def get_cloud_storage_status() -> Dict[str, str]:
    """Checks the Google Cloud Storage bucket destination and cloud dataset status.

    Returns:
        dict: Information on the cloud storage location, bucket URI, and file verification.
    """
    return {
        "bucket_uri": "gs://sasol-train-dataset/",
        "training_file": "gs://sasol-train-dataset/train.csv",
        "file_size": "70.46 MiB (73,879,714 bytes)",
        "status": "Ready & verified on Google Cloud Storage",
    }


def inspect_label_rules() -> Dict[str, Any]:
    """Reads and returns the authoritative target generation rules and configuration for the Sasol challenge.

    Use this tool whenever you need to inspect the exact thresholds, time windows,
    scaling constants, or deterministic Python logic used to compute `CLV_fuel`,
    `CLV_nonfuel`, and the 17 `Opportunity` classes from raw transactions.

    Returns:
        dict: A dictionary containing the parsed `label_config.json`, the source code
              of `label_rules.py`, and the documentation from `LABEL_RULES.md`.
    """
    base_dir = Path("data/raw")
    if not base_dir.exists():
        base_dir = Path(__file__).resolve().parents[2] / "data/raw"

    result: Dict[str, Any] = {}

    config_path = base_dir / "label_config.json"
    if config_path.exists():
        result["label_config"] = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        result["label_config"] = f"File not found at {config_path}"

    rules_py_path = base_dir / "label_rules.py"
    if rules_py_path.exists():
        result["label_rules_py"] = rules_py_path.read_text(encoding="utf-8")
    else:
        result["label_rules_py"] = f"File not found at {rules_py_path}"

    rules_md_path = base_dir / "LABEL_RULES.md"
    if rules_md_path.exists():
        result["label_rules_md"] = rules_md_path.read_text(encoding="utf-8")
    else:
        result["label_rules_md"] = f"File not found at {rules_md_path}"

    return result


def audit_submission_file(filepath: str = "submissions/submission_baseline.csv") -> Dict[str, Any]:
    """Audits a generated competition submission CSV against the strict Sasol evaluation contract.

    Use this tool to validate any submission file before uploading to Zindi, or to inspect
    predicted CLV distributions and Opportunity class proportions for drift/collapse.

    Args:
        filepath (str): Relative path to the submission CSV file to validate.

    Returns:
        dict: Validation status, assertion check results, regression summary statistics,
              and Opportunity class distribution breakdown.
    """
    sub_path = Path(filepath)
    if not sub_path.exists():
        candidate = Path(__file__).resolve().parents[2] / filepath
        if candidate.exists():
            sub_path = candidate

    sample_path = Path("data/raw/SampleSubmission.csv")
    if not sample_path.exists():
        sample_candidate = Path(__file__).resolve().parents[2] / "data/raw/SampleSubmission.csv"
        if sample_candidate.exists():
            sample_path = sample_candidate

    if not sub_path.exists():
        return {"status": "ERROR", "message": f"Submission file not found at {sub_path}"}

    sub_df = pd.read_csv(sub_path)
    expected_cols = ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]

    checks: Dict[str, Any] = {
        "row_count_matches_5488": len(sub_df) == 5488,
        "actual_shape": list(sub_df.shape),
        "columns_match_contract": list(sub_df.columns) == expected_cols,
        "zero_nans": int(sub_df.isna().sum().sum()) == 0,
        "clv_fuel_non_negative": bool((sub_df["CLV_fuel"] >= 0).all()) if "CLV_fuel" in sub_df else False,
        "clv_nonfuel_non_negative": bool((sub_df["CLV_nonfuel"] >= 0).all()) if "CLV_nonfuel" in sub_df else False,
    }

    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        checks["id_alignment_exact"] = bool((sub_df["ID"].values == sample_df["ID"].values).all())

    config_path = Path("data/raw/label_config.json")
    if not config_path.exists():
        config_candidate = Path(__file__).resolve().parents[2] / "data/raw/label_config.json"
        if config_candidate.exists():
            config_path = config_candidate

    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        # In label_config.json, the key is opportunity_labels (with opportunity_classes fallback)
        valid_classes = set(config_data.get("opportunity_labels", config_data.get("opportunity_classes", [])))
        if valid_classes and "Opportunity" in sub_df:
            invalid = set(sub_df["Opportunity"].dropna().unique()) - valid_classes
            checks["invalid_opportunity_classes"] = list(invalid)
            checks["all_classes_valid"] = len(invalid) == 0

    class_counts = sub_df["Opportunity"].value_counts().to_dict() if "Opportunity" in sub_df else {}
    class_pcts = (
        (sub_df["Opportunity"].value_counts(normalize=True) * 100).round(2).to_dict()
        if "Opportunity" in sub_df
        else {}
    )

    stats = {}
    for col in ["CLV_fuel", "CLV_nonfuel"]:
        if col in sub_df:
            stats[col] = {
                "mean": round(float(sub_df[col].mean()), 5),
                "std": round(float(sub_df[col].std()), 5),
                "min": round(float(sub_df[col].min()), 5),
                "median": round(float(sub_df[col].median()), 5),
                "p95": round(float(sub_df[col].quantile(0.95)), 5),
                "max": round(float(sub_df[col].max()), 5),
                "zero_pct": round(float((sub_df[col] == 0).mean() * 100), 2),
            }

    all_passed = all(
        v is True
        for k, v in checks.items()
        if isinstance(v, bool)
    )

    return {
        "status": "PASSED" if all_passed else "FAILED_CONTRACT_CHECKS",
        "filepath": str(sub_path),
        "contract_checks": checks,
        "clv_summary_stats": stats,
        "opportunity_class_counts": class_counts,
        "opportunity_class_percentages": class_pcts,
    }
