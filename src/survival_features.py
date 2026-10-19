"""
Gap-survival churn features (FEATURE family "surv1"). Rows with ts < cutoff only.

For each customer and activity stream (any visit, fuel fill, qualifying non-fuel purchase) we take the
completed inter-visit gaps g_1..g_n (days between distinct active dates) and the current, right-censored
gap r = days since the last visit. The probability that the customer stays silent for the whole
3-month outcome window is the conditional survival

    P(dormant) = S(r + H) / S(r),   H = 91 days,

estimated per customer with empirical-Bayes shrinkage toward the population gap distribution:

    S_i(t) = (#{g_i > t} + a * S_pop(t)) / (n_i + a)

S_pop is the pooled gap survival over all customers (same cutoff, so no leakage). Also emitted: the
population-only version, the customer's own n_i, the log hazard ratio, and expected visits in the
window under a renewal approximation H / E[g].
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

ID = "ID"
SURV_VERSION = "surv1"
H = 91.0
PRIOR = 3.0


def _stream(h: pl.DataFrame, name: str) -> pl.DataFrame:
    d = h.select(ID, "date", "dbc").group_by(ID, "date").agg(pl.col("dbc").min()).sort(ID, "date")
    gaps = (d.with_columns((pl.col("date") - pl.col("date").shift(1).over(ID)).dt.total_days().cast(pl.Float64).alias("g"))
            .group_by(ID).agg(pl.col("g").drop_nulls().alias("gaps"), pl.col("dbc").min().alias("r")))
    ids = gaps[ID].to_list()
    G = gaps["gaps"].to_list()
    r = gaps["r"].to_numpy()
    pool = np.sort(np.concatenate([np.asarray(g, float) for g in G if len(g)] or [np.array([H])]))
    n_pool = len(pool)

    def s_pop(t):
        return (n_pool - np.searchsorted(pool, t, side="right")) / n_pool

    sp_r, sp_rh = s_pop(r), s_pop(r + H)
    p_pop = np.where(sp_r > 0, sp_rh / np.maximum(sp_r, 1e-9), 1.0)
    n = np.array([len(g) for g in G], float)
    gt_r = np.array([np.sum(np.asarray(g) > ri) for g, ri in zip(G, r)], float)
    gt_rh = np.array([np.sum(np.asarray(g) > ri + H) for g, ri in zip(G, r)], float)
    s_r = (gt_r + PRIOR * sp_r) / (n + PRIOR)
    s_rh = (gt_rh + PRIOR * sp_rh) / (n + PRIOR)
    p_cust = np.where(s_r > 0, s_rh / np.maximum(s_r, 1e-9), 1.0)
    mean_g = np.array([np.mean(g) if len(g) else np.nan for g in G])
    exceed = np.array([np.mean(np.asarray(g) >= ri) if len(g) else np.nan for g, ri in zip(G, r)])
    return pl.DataFrame({
        ID: ids,
        f"sv_{name}_p_dormant": np.clip(p_cust, 0, 1),
        f"sv_{name}_p_dormant_pop": np.clip(p_pop, 0, 1),
        f"sv_{name}_log_ratio": np.log(np.clip(p_cust, 1e-4, 1) / np.clip(p_pop, 1e-4, 1)),
        f"sv_{name}_n_gaps": n,
        f"sv_{name}_share_gaps_ge_r": exceed,
        f"sv_{name}_exp_visits": H / np.maximum(mean_g, 1.0),
        f"sv_{name}_r_over_mean_gap": r / np.maximum(mean_g, 1.0),
    })


def survival_features(tx: pl.DataFrame, cutoff: str, label_config: dict) -> pl.DataFrame:
    t = datetime.strptime(cutoff, "%Y-%m-%d")
    excluded = {c for c, v in label_config["category_mapping"].items() if not v["included"]}
    h = (tx.lazy().filter((pl.col("ts") < t) & ~pl.col("is_adjustment"))
         .with_columns(((pl.lit(t) - pl.col("ts")).dt.total_seconds() / 86400.0).alias("dbc"),
                       pl.col("ts").dt.date().alias("date"),
                       pl.col("ItemCategoryLevel2").cast(pl.String).alias("cat"))
         .filter(~pl.col("cat").is_in(list(excluded)))
         .collect())
    qual = (pl.col("qty") > 0) & (pl.col("amount") > 0)
    out = _stream(h.filter(qual), "any")
    fuel = h.filter((pl.col("cat") == "Fuel") & (pl.col("LineItemType").cast(pl.String) == "Sale_Fuel") & (pl.col("qty") > 0))
    nf = h.filter((pl.col("cat") != "Fuel") & qual)
    for name, d in (("fuel", fuel), ("nf", nf)):
        if d.height:
            out = out.join(_stream(d, name), on=ID, how="left")
    return out.with_columns(pl.col(ID).cast(pl.String))
