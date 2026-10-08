"""
Grandmaster Modular Pipeline Coordinator for Sasol Challenge.
Supports:
- Multi-cohort data ingestion (2024-09, 2024-12, 2025-03, 2025-06) for training augmentation
- Out-of-Time (OOT) holdout validation (2025-09)
- Multi-head modeling (Fuel, Non-fuel Hurdle, Opportunity)
- Cross-task feature conditioning
- Ensembling (LightGBM + CatBoost + XGBoost)
- Nelder-Mead F1 threshold optimization
- Strict submission verification
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

from src.features import compute_customer_features
from src.models.regressors import (
    LGBMRegressorModel,
    CatBoostRegressorModel,
    XGBoostRegressorModel,
    HurdleRegressorModel
)
from src.models.classifiers import (
    LGBMClassifierModel,
    CatBoostClassifierModel,
    XGBoostClassifierModel
)
from src.metrics import composite_metric
from src.postprocessing import optimize_thresholds, apply_thresholds
from src.utils import setup_logger, seed_everything

logger = setup_logger("sasol_pipeline")

OPPORTUNITY_CLASSES = [
    "Existing-category growth: Beverages",
    "Existing-category growth: Food & beverage",
    "Existing-category growth: Fuel",
    "Existing-category growth: Other",
    "Inactivity",
    "New category adoption: Beverages",
    "New category adoption: Braai & Ice",
    "New category adoption: Breads & rolls",
    "New category adoption: Confectionary",
    "New category adoption: Food & beverage",
    "New category adoption: Groceries",
    "New category adoption: Ice-cream",
    "New category adoption: Lubricants",
    "New category adoption: Other",
    "New category adoption: Snacking",
    "New category adoption: Tobacco",
    "Stable"
]
CLASS_MAP = {c: i for i, c in enumerate(OPPORTUNITY_CLASSES)}
INV_CLASS_MAP = {i: c for i, c in enumerate(OPPORTUNITY_CLASSES)}


def extract_or_load_features(raw_data_dir: Path, processed_dir: Path, cutoff: str, force_recompute: bool = False) -> pd.DataFrame:
    """Extracts features via Polars or loads existing CSV."""
    safe_name = cutoff.replace("-", "_")
    feat_path = processed_dir / f"features_{safe_name}.csv"
    
    if feat_path.exists() and not force_recompute:
        logger.info(f"Loading cached features for cutoff {cutoff} from {feat_path}")
        return pd.read_csv(feat_path)
        
    logger.info(f"Generating Polars features for cutoff {cutoff}...")
    tx_path = raw_data_dir / "train.csv"
    feats_pl = compute_customer_features(str(tx_path), cutoff)
    feats_df = feats_pl.to_pandas()
    feats_df.to_csv(feat_path, index=False)
    logger.info(f"Cached {len(feats_df)} customer feature rows to {feat_path}")
    return feats_df


def extract_or_load_labels(raw_data_dir: Path, processed_dir: Path, cutoff: str) -> pd.DataFrame:
    """Loads existing precomputed labels or invokes label_rules."""
    safe_name = cutoff.replace("-", "_")
    label_path = processed_dir / f"labels_{safe_name}.csv"
    if label_path.exists():
        return pd.read_csv(label_path)
    
    # If not existing, run label_rules generator
    import subprocess
    logger.info(f"Computing ground truth labels for cutoff {cutoff} via label_rules.py...")
    cmd = [
        "python", str(raw_data_dir / "label_rules.py"),
        "--transactions", str(raw_data_dir / "train.csv"),
        "--cutoff", cutoff,
        "--config", str(raw_data_dir / "label_config.json"),
        "--output", str(label_path)
    ]
    subprocess.run(cmd, check=True)
    return pd.read_csv(label_path)


def run_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    sub_dir: str = "submissions",
    sub_filename: str = "submission_colab_ensemble.csv",
    use_multi_cohort: bool = True,
    use_ensemble: bool = True,
    seed: int = 42
):
    seed_everything(seed)
    raw_p = Path(raw_dir)
    proc_p = Path(processed_dir)
    sub_p = Path(sub_dir)
    proc_p.mkdir(parents=True, exist_ok=True)
    sub_p.mkdir(parents=True, exist_ok=True)
    
    sample_sub_path = raw_p / "SampleSubmission.csv"
    if not sample_sub_path.exists():
        raise FileNotFoundError(f"Missing {sample_sub_path}")
    sample_sub = pd.read_csv(sample_sub_path)
    
    # ---------------------------------------------------------
    # 1. Prepare Training Cohorts
    # ---------------------------------------------------------
    train_cutoffs = ["2025-06-01"]
    if use_multi_cohort:
        train_cutoffs = ["2024-12-01", "2025-03-01", "2025-06-01"]
        
    val_cutoff = "2025-09-01"
    test_cutoff = "2025-12-01"
    
    logger.info(f"Training cutoffs: {train_cutoffs} | Validation cutoff: {val_cutoff}")
    
    train_frames = []
    for cutoff in train_cutoffs:
        f_df = extract_or_load_features(raw_p, proc_p, cutoff)
        l_df = extract_or_load_labels(raw_p, proc_p, cutoff)
        merged = pd.merge(f_df, l_df, on="ID", how="inner")
        merged["cohort_cutoff"] = cutoff
        train_frames.append(merged)
        
    train_df = pd.concat(train_frames, ignore_index=True)
    logger.info(f"Total training instances pooled: {len(train_df)}")
    
    val_feats = extract_or_load_features(raw_p, proc_p, val_cutoff)
    val_labels = extract_or_load_labels(raw_p, proc_p, val_cutoff)
    val_df = pd.merge(val_feats, val_labels, on="ID", how="inner")
    logger.info(f"Validation instances: {len(val_df)}")
    
    test_feats = extract_or_load_features(raw_p, proc_p, test_cutoff)
    test_df = pd.merge(sample_sub[["ID"]], test_feats, on="ID", how="left")
    logger.info(f"Test instances: {len(test_df)}")
    
    # ---------------------------------------------------------
    # 2. Feature Columns & Sanitization
    # ---------------------------------------------------------
    meta_cols = [
        "ID", "fuel_litres", "nonfuel_rand", "Opportunity",
        "winning_source_category", "winning_increment_rand",
        "CLV_fuel", "CLV_nonfuel", "cohort_cutoff"
    ]
    feature_cols = [c for c in train_df.columns if c not in meta_cols and not c.startswith("_")]
    logger.info(f"Using {len(feature_cols)} engineered features.")
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_val = val_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    y_fuel_train = train_df["CLV_fuel"].values
    y_nonfuel_train = train_df["CLV_nonfuel"].values
    y_opp_train = train_df["Opportunity"].map(CLASS_MAP).values
    
    y_fuel_val = val_df["CLV_fuel"].values
    y_nonfuel_val = val_df["CLV_nonfuel"].values
    y_opp_val = val_df["Opportunity"].map(CLASS_MAP).values
    
    # ---------------------------------------------------------
    # 3. Head 1: Fuel CLV Regressors (Ensemble)
    # ---------------------------------------------------------
    logger.info("Training Head 1: Fuel CLV Models...")
    lgb_fuel = LGBMRegressorModel({'n_estimators': 350, 'learning_rate': 0.03, 'random_state': seed})
    lgb_fuel.fit(X_train, y_fuel_train)
    p_fuel_val_lgb = lgb_fuel.predict(X_val)
    p_fuel_test_lgb = lgb_fuel.predict(X_test)
    
    if use_ensemble:
        cat_fuel = CatBoostRegressorModel({'iterations': 350, 'learning_rate': 0.03, 'random_seed': seed})
        cat_fuel.fit(X_train, y_fuel_train)
        xgb_fuel = XGBoostRegressorModel({'n_estimators': 300, 'learning_rate': 0.03, 'random_state': seed})
        xgb_fuel.fit(X_train, y_fuel_train)
        
        p_fuel_val = 0.5 * p_fuel_val_lgb + 0.3 * cat_fuel.predict(X_val) + 0.2 * xgb_fuel.predict(X_val)
        p_fuel_test = 0.5 * p_fuel_test_lgb + 0.3 * cat_fuel.predict(X_test) + 0.2 * xgb_fuel.predict(X_test)
        p_fuel_train_pred = 0.5 * lgb_fuel.predict(X_train) + 0.3 * cat_fuel.predict(X_train) + 0.2 * xgb_fuel.predict(X_train)
    else:
        p_fuel_val = p_fuel_val_lgb
        p_fuel_test = p_fuel_test_lgb
        p_fuel_train_pred = lgb_fuel.predict(X_train)
        
    p_fuel_val = np.clip(p_fuel_val, 0, None)
    p_fuel_test = np.clip(p_fuel_test, 0, None)
    p_fuel_train_pred = np.clip(p_fuel_train_pred, 0, None)
    
    # ---------------------------------------------------------
    # 4. Head 2: Non-Fuel CLV Regressors (Hurdle Ensemble)
    # ---------------------------------------------------------
    logger.info("Training Head 2: Non-Fuel CLV Two-Stage Hurdle...")
    hurdle = HurdleRegressorModel()
    hurdle.fit(X_train, y_nonfuel_train)
    p_nonfuel_val_h = hurdle.predict(X_val)
    p_nonfuel_test_h = hurdle.predict(X_test)
    p_nonfuel_train_h = hurdle.predict(X_train)
    
    if use_ensemble:
        lgb_nonfuel = LGBMRegressorModel({'n_estimators': 300, 'learning_rate': 0.03, 'random_state': seed})
        lgb_nonfuel.fit(X_train, y_nonfuel_train)
        p_nonfuel_val = 0.65 * p_nonfuel_val_h + 0.35 * lgb_nonfuel.predict(X_val)
        p_nonfuel_test = 0.65 * p_nonfuel_test_h + 0.35 * lgb_nonfuel.predict(X_test)
        p_nonfuel_train_pred = 0.65 * p_nonfuel_train_h + 0.35 * lgb_nonfuel.predict(X_train)
    else:
        p_nonfuel_val = p_nonfuel_val_h
        p_nonfuel_test = p_nonfuel_test_h
        p_nonfuel_train_pred = p_nonfuel_train_h
        
    p_nonfuel_val = np.clip(p_nonfuel_val, 0, None)
    p_nonfuel_test = np.clip(p_nonfuel_test, 0, None)
    p_nonfuel_train_pred = np.clip(p_nonfuel_train_pred, 0, None)
    
    # ---------------------------------------------------------
    # 5. Head 3: Opportunity Classifier with Cross-Conditioning
    # ---------------------------------------------------------
    logger.info("Training Head 3: Opportunity Multi-Class Classifier...")
    X_tr_opp = X_train.copy()
    X_tr_opp["cond_fuel"] = p_fuel_train_pred
    X_tr_opp["cond_nonfuel"] = p_nonfuel_train_pred
    
    X_val_opp = X_val.copy()
    X_val_opp["cond_fuel"] = p_fuel_val
    X_val_opp["cond_nonfuel"] = p_nonfuel_val
    
    X_test_opp = X_test.copy()
    X_test_opp["cond_fuel"] = p_fuel_test
    X_test_opp["cond_nonfuel"] = p_nonfuel_test
    
    lgb_opp = LGBMClassifierModel({'n_estimators': 300, 'learning_rate': 0.04, 'random_state': seed})
    lgb_opp.fit(X_tr_opp, y_opp_train)
    probs_val_lgb = lgb_opp.predict_proba(X_val_opp)
    probs_test_lgb = lgb_opp.predict_proba(X_test_opp)
    
    if use_ensemble:
        cat_opp = CatBoostClassifierModel({'iterations': 300, 'learning_rate': 0.04, 'random_seed': seed})
        cat_opp.fit(X_tr_opp, y_opp_train)
        probs_val_cat = cat_opp.predict_proba(X_val_opp)
        probs_test_cat = cat_opp.predict_proba(X_test_opp)
        
        probs_val = 0.6 * probs_val_lgb + 0.4 * probs_val_cat
        probs_test = 0.6 * probs_test_lgb + 0.4 * probs_test_cat
    else:
        probs_val = probs_val_lgb
        probs_test = probs_test_lgb
        
    # ---------------------------------------------------------
    # 6. Post-Processing: Nelder-Mead F1 Threshold Optimizer
    # ---------------------------------------------------------
    logger.info("Optimizing Multi-Class Decision Thresholds...")
    best_weights = optimize_thresholds(y_opp_val, probs_val)
    p_opp_val = apply_thresholds(probs_val, best_weights)
    p_opp_test = apply_thresholds(probs_test, best_weights)
    
    # ---------------------------------------------------------
    # 7. Evaluation & Leaderboard Metric Calculation
    # ---------------------------------------------------------
    metrics = composite_metric(y_fuel_val, p_fuel_val, y_nonfuel_val, p_nonfuel_val, y_opp_val, p_opp_val)
    logger.info("==================================================")
    logger.info("OUT-OF-TIME VALIDATION PERFORMANCE:")
    logger.info(f"-> Fuel CLV RMSE:        {metrics['rmse_fuel']:.4f}")
    logger.info(f"-> Non-Fuel CLV RMSE:    {metrics['rmse_nonfuel']:.4f}")
    logger.info(f"-> Opportunity Weighted F1: {metrics['f1_opportunity']:.4f}")
    logger.info(f"-> Composite Score:      {metrics['composite_score']:.4f}")
    logger.info("==================================================")
    
    # ---------------------------------------------------------
    # 8. Submission Export & Strict Contracts
    # ---------------------------------------------------------
    sub_df = pd.DataFrame({
        "ID": test_df["ID"],
        "CLV_fuel": p_fuel_test,
        "CLV_nonfuel": p_nonfuel_test,
        "Opportunity": [INV_CLASS_MAP[idx] for idx in p_opp_test]
    })
    
    # Assertions
    assert sub_df.shape == sample_sub.shape, f"Shape mismatch: {sub_df.shape} vs {sample_sub.shape}"
    assert list(sub_df.columns) == list(sample_sub.columns), "Column names mismatch"
    assert sub_df.isna().sum().sum() == 0, "Found NaNs in submission"
    assert (sub_df["CLV_fuel"] >= 0).all(), "Negative fuel predictions detected"
    assert (sub_df["CLV_nonfuel"] >= 0).all(), "Negative nonfuel predictions detected"
    assert set(sub_df["Opportunity"].unique()).issubset(set(OPPORTUNITY_CLASSES)), "Illegal opportunity classes detected"
    
    out_file = sub_p / sub_filename
    sub_df.to_csv(out_file, index=False)
    logger.info(f"Exported verified submission to {out_file}")
    
    return metrics, sub_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sasol Multi-Task ML Pipeline")
    parser.add_argument("--raw-dir", default="data/raw", type=str)
    parser.add_argument("--processed-dir", default="data/processed", type=str)
    parser.add_argument("--sub-dir", default="submissions", type=str)
    parser.add_argument("--sub-name", default="submission_colab_ensemble.csv", type=str)
    parser.add_argument("--no-multi-cohort", action="store_true")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    
    run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        sub_dir=args.sub_dir,
        sub_filename=args.sub_name,
        use_multi_cohort=not args.no_multi_cohort,
        use_ensemble=not args.no_ensemble,
        seed=args.seed
    )
