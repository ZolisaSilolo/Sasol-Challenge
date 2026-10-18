"""
Context features (experimental, FEATURE family "ctx1"). All use rows with ts < cutoff only.

* Station context: for every station, activity momentum (baskets / active customers in the last 30 d vs
  the prior 60 d), days since the station's last transaction (closures / outages), and the station
  churn rate (share of its customers active 90-180 d ago who were not seen in the last 90 d). Each
  customer gets the basket-share-weighted average and the value at their primary station.
* Fuel-type mix: diesel share of litres, premium (95) share, distinct grades, litres per fill (last 90 d).
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

ID = "ID"
CTX_VERSION = "ctx1"


def context_features(tx: pl.DataFrame, cutoff: str) -> pl.DataFrame:
    t = datetime.strptime(cutoff, "%Y-%m-%d")
    h = (tx.lazy().filter((pl.col("ts") < t) & ~pl.col("is_adjustment"))
         .with_columns(((pl.lit(t) - pl.col("ts")).dt.total_seconds() / 86400.0).alias("dbc"),
                       pl.col("LocationId").cast(pl.String).alias("loc"))
         .collect())
    b = h.group_by(ID, "TransactionId").agg(pl.col("loc").first(), pl.col("dbc").min())

    cust_last = b.group_by(ID, "loc").agg(pl.col("dbc").min().alias("last"), pl.len().alias("n"))
    st = b.group_by("loc").agg(
        pl.col("dbc").min().alias("st_days_since_last"),
        (pl.col("dbc") <= 30).sum().alias("_b30"),
        ((pl.col("dbc") > 30) & (pl.col("dbc") <= 90)).sum().alias("_b31_90"),
        pl.col(ID).filter(pl.col("dbc") <= 30).n_unique().alias("_c30"),
        pl.col(ID).filter((pl.col("dbc") > 30) & (pl.col("dbc") <= 90)).n_unique().alias("_c31_90"),
    )
    # Station churn: customers whose last visit to the station was 90-180 d ago, vs all with a visit 90-180 d ago.
    seen_mid = b.filter((pl.col("dbc") > 90) & (pl.col("dbc") <= 180)).select("loc", ID).unique()
    cust_any_last = b.group_by(ID).agg(pl.col("dbc").min().alias("any_last"))
    churn = (seen_mid.join(cust_any_last, on=ID).group_by("loc")
             .agg((pl.col("any_last") > 90).mean().alias("st_churn_rate"), pl.len().alias("st_mid_customers")))
    st = (st.join(churn, on="loc", how="left")
          .with_columns((pl.col("_b30") / (pl.col("_b31_90") / 2 + 1)).alias("st_mom_baskets"),
                        (pl.col("_c30") / (pl.col("_c31_90") / 2 + 1)).alias("st_mom_customers"))
          .drop("_b30", "_b31_90", "_c30", "_c31_90"))
    st_cols = ["st_days_since_last", "st_churn_rate", "st_mid_customers", "st_mom_baskets", "st_mom_customers"]
    j = cust_last.join(st, on="loc", how="left").with_columns((pl.col("n") / pl.col("n").sum().over(ID)).alias("w"))
    wavg = j.group_by(ID).agg([(pl.col(c) * pl.col("w")).sum().alias(f"ctx_w_{c}") for c in st_cols])
    prim = (j.sort([ID, "n", "last"], descending=[False, True, False]).group_by(ID, maintain_order=True).first()
            .select(ID, *[pl.col(c).alias(f"ctx_p_{c}") for c in st_cols],
                    (pl.col("last") - pl.col("st_days_since_last")).alias("ctx_p_cust_lag_vs_station")))

    f = (h.filter((pl.col("LineItemType").cast(pl.String) == "Sale_Fuel") & (pl.col("qty") > 0))
         .with_columns(pl.col("ItemName").cast(pl.String).alias("g")))
    fuel = f.group_by(ID).agg(
        (pl.col("qty").filter(pl.col("g").str.contains("(?i)diesel")).sum() / pl.col("qty").sum()).alias("ctx_diesel_share"),
        (pl.col("qty").filter(pl.col("g").str.contains("95")).sum() / pl.col("qty").sum()).alias("ctx_ulp95_share"),
        pl.col("g").n_unique().alias("ctx_n_grades"),
        (pl.col("qty").filter(pl.col("dbc") <= 90).sum()
         / (pl.col("TransactionId").filter(pl.col("dbc") <= 90).n_unique() + 1e-5)).alias("ctx_litres_per_fill_90"),
        pl.col("qty").filter(pl.col("dbc") <= 90).std().alias("ctx_litres_std_90"),
    )
    out = wavg.join(prim, on=ID, how="left").join(fuel, on=ID, how="left")
    return out.with_columns(pl.col(ID).cast(pl.String), *[pl.col(c).cast(pl.Float64) for c in out.columns if c != ID])
