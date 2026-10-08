"""
High-Velocity Polars Feature Engineering Factory for Sasol Challenge.
Implements Phase 2 Competitive ML Retail & Fuel Indicators:
1. RFM & Multi-Window Aggregations [14d, 30d, 60d, 90d, 180d, Lifetime]
2. Spend & Visit Velocities
3. Inter-Purchase Interval (IPI) statistics
4. Category & Basket Dynamics (Fuel vs Non-fuel, Diversity/Entropy)
5. Temporal, Cadence & Pay-Cycle Dynamics
6. Site & Mobility Profiles
7. Spend Trajectory (Trend Slope)
"""

import math
from datetime import datetime
import numpy as np
import polars as pl


def compute_customer_features(transactions_path: str, cutoff_date: str) -> pl.DataFrame:
    """
    Computes strict, leak-free customer features evaluated strictly before cutoff_date.
    """
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    
    # Read raw transactions with schema flexibility
    df = pl.read_csv(transactions_path, infer_schema_length=0)
    
    # Parse timestamps and numeric values
    df = df.with_columns([
        pl.coalesce([
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False),
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%Z", strict=False),
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False),
        ]).alias("dt"),
        pl.col("Quantity").cast(pl.Float64, strict=False).fill_null(0.0).alias("qty"),
        pl.col("TotalExclAmount").cast(pl.Float64, strict=False).fill_null(0.0).alias("amount"),
        pl.col("ItemCategoryLevel2").fill_null("Unknown").alias("cat_l2"),
        pl.col("LineItemType").fill_null("").alias("item_type"),
        pl.col("LocationId").fill_null("Unknown").alias("loc_id"),
        pl.col("TransactionId").fill_null("Unknown").alias("txn_id"),
    ])
    
    # Filter strictly prior to cutoff
    df = df.filter(pl.col("dt") < cutoff_dt)
    
    # Compute days before cutoff and calendar signals
    df = df.with_columns([
        (cutoff_dt - pl.col("dt")).dt.total_days().alias("days_before_cutoff"),
        pl.col("dt").dt.hour().alias("hour"),
        pl.col("dt").dt.weekday().alias("weekday"),  # 1=Mon, 7=Sun
        pl.col("dt").dt.day().alias("day_of_month"),
        pl.col("dt").dt.date().alias("txn_date"),
    ])
    
    # Flags for fuel and non-fuel items
    is_fuel = (pl.col("cat_l2") == "Fuel") & (pl.col("item_type") == "Sale_Fuel")
    is_nonfuel = (pl.col("cat_l2") != "Fuel") & (~pl.col("cat_l2").is_in(["Value added services", "Print media", "Carrier Bags"]))
    
    df = df.with_columns([
        is_fuel.alias("is_fuel"),
        is_nonfuel.alias("is_nonfuel"),
        pl.when(is_fuel).then(pl.col("qty")).otherwise(0.0).alias("fuel_qty"),
        pl.when(is_fuel).then(pl.col("amount")).otherwise(0.0).alias("fuel_spend"),
        pl.when(is_nonfuel).then(pl.col("amount")).otherwise(0.0).alias("nonfuel_spend"),
    ])
    
    # ----------------------------------------------------
    # Aggregation Builders
    # ----------------------------------------------------
    aggs = [
        # Lifetime boundaries & recency
        pl.col("days_before_cutoff").min().alias("recency_days"),
        pl.col("days_before_cutoff").max().alias("tenure_days"),
        pl.col("txn_id").n_unique().alias("lifetime_txn_count"),
        pl.col("amount").sum().alias("lifetime_spend"),
        pl.col("fuel_qty").sum().alias("lifetime_fuel_litres"),
        pl.col("nonfuel_spend").sum().alias("lifetime_nonfuel_spend"),
        pl.col("cat_l2").n_unique().alias("unique_categories_lifetime"),
        pl.col("loc_id").n_unique().alias("unique_stations_lifetime"),
    ]
    
    # Multi-window RFM slices: 14d, 30d, 60d, 90d, 180d
    windows = [14, 30, 60, 90, 180]
    for w in windows:
        mask = pl.col("days_before_cutoff") <= w
        aggs.extend([
            pl.col("txn_id").filter(mask).n_unique().alias(f"freq_{w}d"),
            pl.col("amount").filter(mask).sum().alias(f"spend_{w}d"),
            pl.col("fuel_qty").filter(mask).sum().alias(f"fuel_litres_{w}d"),
            pl.col("nonfuel_spend").filter(mask).sum().alias(f"nonfuel_spend_{w}d"),
            pl.col("cat_l2").filter(mask).n_unique().alias(f"unique_cats_{w}d"),
            pl.col("loc_id").filter(mask).n_unique().alias(f"unique_locs_{w}d"),
        ])
    
    # Temporal & Cadence Dynamics
    aggs.extend([
        # Commuting rush hours (6-9 AM, 16-19 PM)
        pl.col("txn_id").filter((pl.col("hour") >= 6) & (pl.col("hour") <= 9)).n_unique().alias("morning_commute_txns"),
        pl.col("txn_id").filter((pl.col("hour") >= 16) & (pl.col("hour") <= 19)).n_unique().alias("evening_commute_txns"),
        # Weekend vs Weekday
        pl.col("txn_id").filter(pl.col("weekday") >= 6).n_unique().alias("weekend_txns"),
        pl.col("txn_id").filter(pl.col("weekday") < 6).n_unique().alias("weekday_txns"),
        # Pay-cycle (25th of month to 2nd of next month)
        pl.col("txn_id").filter((pl.col("day_of_month") >= 25) | (pl.col("day_of_month") <= 2)).n_unique().alias("paycycle_txns"),
        pl.col("amount").filter((pl.col("day_of_month") >= 25) | (pl.col("day_of_month") <= 2)).sum().alias("paycycle_spend"),
    ])
    
    # Perform primary group_by aggregation
    base_agg = df.group_by("ID").agg(aggs)
    
    # ----------------------------------------------------
    # Basket & Station Concentration (Top station loyalty)
    # ----------------------------------------------------
    station_spend = df.group_by(["ID", "loc_id"]).agg(pl.col("amount").sum().alias("station_spend"))
    primary_station = station_spend.sort(["ID", "station_spend"], descending=[False, True]).group_by("ID").first()
    primary_station = primary_station.select([
        pl.col("ID"),
        pl.col("station_spend").alias("primary_station_spend")
    ])
    
    base_agg = base_agg.join(primary_station, on="ID", how="left")
    
    # ----------------------------------------------------
    # Inter-Purchase Interval (IPI) Statistics
    # ----------------------------------------------------
    distinct_visits = df.select(["ID", "txn_date"]).unique().sort(["ID", "txn_date"])
    ipi_df = distinct_visits.with_columns([
        (pl.col("txn_date") - pl.col("txn_date").shift(1).over("ID")).dt.total_days().alias("ipi_days")
    ]).filter(pl.col("ipi_days").is_not_null())
    
    ipi_stats = ipi_df.group_by("ID").agg([
        pl.col("ipi_days").mean().alias("ipi_mean"),
        pl.col("ipi_days").std().alias("ipi_std"),
        pl.col("ipi_days").median().alias("ipi_median"),
        pl.col("ipi_days").min().alias("ipi_min"),
        pl.col("ipi_days").max().alias("ipi_max"),
    ])
    
    base_agg = base_agg.join(ipi_stats, on="ID", how="left")
    
    # ----------------------------------------------------
    # Derived Ratios, Velocities & Behavioral Classifications
    # ----------------------------------------------------
    feats = base_agg.with_columns([
        # Spend velocities
        (pl.col("spend_30d") / ((pl.col("spend_90d") / 3.0) + 1e-4)).alias("spend_velocity_30_90"),
        (pl.col("freq_14d") / ((pl.col("freq_60d") / 4.28) + 1e-4)).alias("visit_velocity_14_60"),
        (pl.col("spend_14d") / ((pl.col("spend_60d") / 4.28) + 1e-4)).alias("spend_velocity_14_60"),
        
        # Spend per visit
        (pl.col("lifetime_spend") / (pl.col("lifetime_txn_count") + 1e-4)).alias("avg_basket_spend_lifetime"),
        (pl.col("spend_90d") / (pl.col("freq_90d") + 1e-4)).alias("avg_basket_spend_90d"),
        (pl.col("spend_30d") / (pl.col("freq_30d") + 1e-4)).alias("avg_basket_spend_30d"),
        
        # Fuel vs Nonfuel ratios
        (pl.col("lifetime_nonfuel_spend") / (pl.col("lifetime_spend") + 1e-4)).alias("nonfuel_spend_ratio_lifetime"),
        (pl.col("nonfuel_spend_90d") / (pl.col("spend_90d") + 1e-4)).alias("nonfuel_spend_ratio_90d"),
        (pl.col("fuel_litres_90d") / (pl.col("freq_90d") + 1e-4)).alias("litres_per_visit_90d"),
        
        # Station Loyalty Index
        (pl.col("primary_station_spend") / (pl.col("lifetime_spend") + 1e-4)).alias("station_loyalty_index"),
        
        # Behavioral Segmentation Flags
        ((pl.col("lifetime_fuel_litres") > 0) & (pl.col("lifetime_nonfuel_spend") <= 0)).cast(pl.Int32).alias("flag_fuel_only"),
        ((pl.col("lifetime_fuel_litres") <= 0) & (pl.col("lifetime_nonfuel_spend") > 0)).cast(pl.Int32).alias("flag_store_only"),
        ((pl.col("lifetime_fuel_litres") > 0) & (pl.col("lifetime_nonfuel_spend") > 0)).cast(pl.Int32).alias("flag_dual_shopper"),
        
        # Temporal cadence ratios
        (pl.col("weekend_txns") / (pl.col("lifetime_txn_count") + 1e-4)).alias("weekend_txn_ratio"),
        ((pl.col("morning_commute_txns") + pl.col("evening_commute_txns")) / (pl.col("lifetime_txn_count") + 1e-4)).alias("commute_rush_ratio"),
        (pl.col("paycycle_txns") / (pl.col("lifetime_txn_count") + 1e-4)).alias("paycycle_txn_ratio"),
        (pl.col("paycycle_spend") / (pl.col("lifetime_spend") + 1e-4)).alias("paycycle_spend_ratio"),
        
        # Trend slope indicator: Recent 30d relative to older periods
        ((pl.col("spend_30d") - (pl.col("spend_90d") - pl.col("spend_30d")) / 2.0)).alias("spend_acceleration_delta"),
    ])
    
    # ----------------------------------------------------
    # Hugging Face Semantic Representation Extraction
    # ----------------------------------------------------
    try:
        from src.hf_embeddings import extract_customer_semantic_features
        sem_feats = extract_customer_semantic_features(transactions_path, cutoff_date)
        if sem_feats.width > 1:
            feats = feats.join(sem_feats, on="ID", how="left")
    except Exception as e:
        # Graceful fallback if HF dependencies are absent locally
        pass
    
    return feats


def generate_features(transactions_path: str, cutoff_date: str) -> pl.DataFrame:
    """Wrapper conforming to existing pipeline signature."""
    return compute_customer_features(transactions_path, cutoff_date)

