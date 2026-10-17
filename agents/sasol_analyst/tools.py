import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict
import pandas as pd

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw")


def get_dataset_overview(sample_rows: int = 10000) -> Dict[str, Any]:
    """Inspects the local Sasol training dataset and returns key dataset statistics.

    Respects the competition data dictionary by aggregating whole basket totals
    (TransactionTotalInclAmount) strictly per unique TransactionId rather than
    summing across line items.

    Args:
        sample_rows (int): Number of line items to sample for fast overview stats.
                           Defaults to 10000.

    Returns:
        dict: Dataset statistics including unique baskets, unique customers (ID),
              categories, and deduplicated basket monetary sums.
    """
    train_path = os.path.join(RAW_DATA_PATH, "train.csv")
    if not os.path.exists(train_path):
        return {"status": "error", "message": "train.csv not found locally"}

    categories_l1 = set()
    categories_l2 = set()
    unique_baskets = {}
    unique_customers = set()
    line_total_excl = 0.0
    row_count = 0

    with open(train_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            row_count += 1
            cid = row.get("ID")
            if cid:
                unique_customers.add(cid)

            tx_id = row.get("TransactionId")
            if tx_id and tx_id not in unique_baskets:
                try:
                    unique_baskets[tx_id] = float(row.get("TransactionTotalInclAmount", 0))
                except (ValueError, TypeError):
                    pass

            try:
                line_total_excl += float(row.get("TotalExclAmount", 0))
            except (ValueError, TypeError):
                pass

            cat1 = row.get("ItemCategoryLevel1")
            if cat1:
                categories_l1.add(cat1)
            cat2 = row.get("ItemCategoryLevel2")
            if cat2:
                categories_l2.add(cat2)

            if row_count >= sample_rows:
                break

    basket_total_sum = sum(unique_baskets.values())

    return {
        "dataset": "Sasol CLV Train Dataset",
        "sample_lines_analyzed": row_count,
        "unique_baskets_in_sample": len(unique_baskets),
        "unique_customers_in_sample": len(unique_customers),
        "total_columns": len(headers),
        "columns": headers,
        "categories_level1": sorted(list(categories_l1)),
        "categories_level2_count": len(categories_l2),
        "deduplicated_basket_total_incl_sum": round(basket_total_sum, 2),
        "line_items_total_excl_sum": round(line_total_excl, 2),
        "note": "Aggregated TransactionTotalInclAmount strictly per unique TransactionId per DATA_DICTIONARY.md rules.",
    }


def get_cloud_storage_status(bucket_uri: str = "gs://sasol-train-dataset/") -> Dict[str, Any]:
    """Dynamically checks Google Cloud Storage bucket contents and artifact status using gcloud CLI.

    Executes a live query to verify bucket connectivity, list stored objects, and check sizes.

    Args:
        bucket_uri (str): Cloud Storage bucket URI to inspect. Defaults to "gs://sasol-train-dataset/".

    Returns:
        dict: Live bucket status, detected objects, total size, and raw output from GCS.
    """
    try:
        res = subprocess.run(
            ["gcloud", "storage", "ls", "--long", bucket_uri],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if res.returncode != 0:
            return {
                "status": "ERROR",
                "bucket_uri": bucket_uri,
                "error": res.stderr.strip() or "Failed to list bucket contents",
            }

        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        objects = []
        for line in lines:
            if line.startswith("TOTAL:"):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[-1].startswith("gs://"):
                objects.append({"size_bytes": parts[0], "updated": parts[1], "uri": parts[2]})
            elif len(parts) == 1 and parts[0].startswith("gs://"):
                objects.append({"uri": parts[0], "type": "prefix_or_directory"})

        raw_totals = [l for l in lines if l.startswith("TOTAL:")]

        return {
            "status": "LIVE_VERIFIED",
            "bucket_uri": bucket_uri,
            "object_count": len(objects),
            "objects": objects,
            "summary": raw_totals[0] if raw_totals else "Objects detected",
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "bucket_uri": bucket_uri,
            "error": str(e),
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


def inspect_pipeline_metrics(filepath: str = "submissions/oof_metrics_fast.json") -> Dict[str, Any]:
    """Reads and returns out-of-fold and out-of-time cross-validation metrics generated by the training pipeline.

    Use this tool to evaluate baseline/model performance, check RMSE per target,
    inspect weighted F1 on Opportunity, and review composite loss and blend weights.

    Args:
        filepath (str): Relative path to the metrics JSON file. Defaults to "submissions/oof_metrics_fast.json".

    Returns:
        dict: Parsed pipeline evaluation metrics including OOT scores, OOF scores, and model blend weights.
    """
    path = Path(filepath)
    if not path.exists():
        candidate = Path(__file__).resolve().parents[2] / filepath
        if candidate.exists():
            path = candidate

    if not path.exists():
        return {
            "status": "ERROR",
            "message": f"Metrics file not found at {path}",
            "available_metric_files": [str(p) for p in Path("submissions").glob("*metrics*.json")] if Path("submissions").exists() else [],
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "SUCCESS",
        "filepath": str(path),
        "config": data.get("config"),
        "n_features": data.get("n_features"),
        "oot_metrics": data.get("oot", {}).get("metrics", {}),
        "oof_summary": data.get("final", {}).get("oof", {}),
        "fuel_blends": data.get("final", {}).get("fuel", {}).get("blend_weights", {}),
        "nonfuel_blends": data.get("final", {}).get("nonfuel", {}).get("blend_weights", {}),
        "opportunity_blends": data.get("final", {}).get("opportunity", {}).get("blend_weights", {}),
        "submission_summary": data.get("submission", {}),
    }


def audit_submission_file(filepath: str = "submissions/submission_fast.csv") -> Dict[str, Any]:
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
