"""
Polars feature factory. All features for cutoff ``t`` use only rows with ``ts < t``.

Families
--------
* Multi-horizon RFM over [7, 14, 30, 60, 90, 180, all-time] days: baskets, active days,
  visits/week, in-scope spend, fuel litres, non-fuel spend, fuel/non-fuel basket counts.
* Label-aligned windows: previous 3 *calendar* months (the exact growth comparison
  window used by label_rules) and the same quarter one year earlier (seasonality).
* Recency (any visit, last fuel fill, last non-fuel purchase), tenure.
* Momentum: short/long ratios, 30-day-bucket linear spend slopes, IPI statistics and
  recency-to-mean-IPI (churn signal).
* Per-category signals mirroring label_config.json's included categories: all-time share,
  previous-quarter net spend, qualifying-basket count (=> untouched flag), 30d vs prior
  60d growth rate, category recency; plus Shannon entropy.
* Cadence: commute rush ratios (06-09h, 16-19h), weekend ratio, pay-cycle (25th-2nd)
  ratios, primary-station concentration, adjustment-item usage.
* Rule replay: label_rules applied to the two most recent *past* quarters (lag1 = outcome
  window [t-3m, t), lag2 = [t-6m, t-3m)), plus growth-hurdle gaps (max(prevq + R20,
  1.25 * prevq) vs the spend of earlier quarters) and monthly-spend volatility.

Definitions follow LABEL_RULES.md: adjustment items never count as purchases, the three
excluded categories are out of scope, a qualifying purchase has qty > 0 and amount > 0,
fuel litres = Quantity on Level2=Fuel & LineItemType=Sale_Fuel.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Dict, List, Sequence

import pandas as pd
import polars as pl

from src.utils import setup_logger

logger = setup_logger("sasol_features")

FEATURE_VERSION = "f4"
DEFAULT_WINDOWS = (7, 14, 30, 60, 90, 180)
EPS = 1e-5
ID = "ID"


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def included_categories(label_config: dict) -> List[str]:
    return sorted(c for c, v in label_config["category_mapping"].items() if v["included"])


def _ratio(num: str | pl.Expr, den: str | pl.Expr, den_scale: float = 1.0) -> pl.Expr:
    n = pl.col(num) if isinstance(num, str) else num
    d = pl.col(den) if isinstance(den, str) else den
    return n / (d / den_scale + EPS)


def _slope_expr(cols: Sequence[str]) -> pl.Expr:
    """OLS slope of y over x = -k (k = buckets before cutoff, so x increases toward cutoff)."""
    xs = [-float(k) for k in range(len(cols))]
    xbar = sum(xs) / len(xs)
    denom = sum((x - xbar) ** 2 for x in xs)
    return pl.sum_horizontal([pl.col(c) * ((x - xbar) / denom) for c, x in zip(cols, xs)])


KIND_CODES = {"Ineligible": 0, "Inactivity": 1, "Stable": 2, "Existing-category growth": 3, "New category adoption": 4}
GROWTH_GROUPS = ("Fuel", "Beverages", "Food & beverage")   # every other growth category maps to "Other"


def rule_per_category(h: pl.DataFrame, end: datetime, label_config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """label_rules candidate logic for outcome [end-3m, end), baseline [end-6m, end-3m), history < end-3m.

    ``h`` needs ID, TransactionId, ts, cat, amount, in_scope, qual for rows with ts < end.
    Returns (eligible IDs, per (ID, cat) frame with hist, fut_n, fut, prev, gv (growth increment
    when the category qualifies for growth, else 0) and av (adoption increment, else 0)).
    """
    cfg = label_config
    start = (pd.Timestamp(end) - pd.DateOffset(months=3)).to_pydatetime()
    base_start = (pd.Timestamp(end) - pd.DateOffset(months=6)).to_pydatetime()
    elig = (h.filter(pl.col("ts") < start).group_by(ID).agg(pl.col("TransactionId").n_unique().alias("n"))
            .filter(pl.col("n") >= cfg["minimum_historical_transactions"]).select(ID))
    s = h.filter(pl.col("in_scope") & (pl.col("ts") < end))
    per_cat = s.group_by(ID, "cat").agg(
        pl.col("TransactionId").filter(pl.col("qual") & (pl.col("ts") < start)).n_unique().alias("hist"),
        pl.col("TransactionId").filter(pl.col("qual") & (pl.col("ts") >= start)).n_unique().alias("fut_n"),
        pl.col("amount").filter(pl.col("ts") >= start).sum().alias("fut"),
        pl.col("amount").filter((pl.col("ts") >= base_start) & (pl.col("ts") < start)).sum().alias("prev"),
    ).join(elig, on=ID, how="semi")
    rel = cfg["growth_min_relative"]
    grow = ((pl.col("hist") > 0) & (pl.col("fut_n") >= cfg["growth_min_purchases"])
            & ((pl.col("fut") - pl.col("prev")) >= cfg["growth_min_rand"] - 1e-9))
    if rel is not None:
        grow &= pl.col("fut") > pl.col("prev").clip(lower_bound=0) * (1 + rel)
    adopt = ((pl.col("hist") == 0) & (pl.col("fut_n") >= cfg["adoption_min_purchases"]) & (pl.col("fut") > 0)
             & (pl.col("fut") >= cfg["adoption_min_rand"] - 1e-9))
    per_cat = per_cat.with_columns(
        pl.when(grow).then(pl.col("fut") - pl.col("prev")).otherwise(0.0).alias("gv"),
        pl.when(adopt).then(pl.col("fut")).otherwise(0.0).alias("av"),
    )
    return elig, per_cat


def rule_state(h: pl.DataFrame, end: datetime, label_config: dict, prefix: str) -> pl.DataFrame:
    """Replay label_rules.raw_labels on a *past* window: outcome [end-3m, end), baseline [end-6m, end-3m),
    history < end-3m. With end <= cutoff this uses pre-cutoff rows only, so it is leak-free.

    ``h`` must already be filtered to ts < cutoff and carry the flags built in build_features.
    Returns one row per eligible customer (>= minimum_historical_transactions baskets before end-3m).
    """
    mapping = label_config["category_mapping"]
    elig, per_cat = rule_per_category(h, end, label_config)

    def best(col: str, kind: str) -> pl.DataFrame:   # argmax per customer, ties -> first category (as numpy)
        top = (per_cat.filter(pl.col(col) > 0).sort([ID, col, "cat"], descending=[False, True, False])
               .group_by(ID, maintain_order=True).first()
               .select(ID, pl.col(col).alias(f"_{col}"), pl.col("cat").alias(f"_{col}_cat")))
        return top.with_columns(pl.col(f"_{col}_cat").replace_strict(
            {c: f"{kind}: {v[kind]}" for c, v in mapping.items()}, default=None).alias(f"_{col}_lab"))

    agg = per_cat.group_by(ID).agg(
        pl.col("fut_n").sum().alias("_fut_n"),
        (pl.col("gv") > 0).sum().alias(f"{prefix}n_cats_grown"),
        (pl.col("av") > 0).sum().alias(f"{prefix}n_cats_adopted"),
    )
    out = (elig.join(agg, on=ID, how="left").join(best("gv", "Existing-category growth"), on=ID, how="left")
           .join(best("av", "New category adoption"), on=ID, how="left")
           .with_columns(pl.col("_gv").fill_null(0.0), pl.col("_av").fill_null(0.0), pl.col("_fut_n").fill_null(0)))
    is_adopt = (pl.col("_av") > 0) & (pl.col("_av") >= pl.col("_gv"))
    is_grow = (pl.col("_gv") > 0) & ~is_adopt
    label = (pl.when(pl.col("_fut_n") == 0).then(pl.lit("Inactivity"))
             .when(is_adopt).then(pl.col("_av_lab")).when(is_grow).then(pl.col("_gv_lab"))
             .otherwise(pl.lit("Stable")))
    classes = list(label_config["opportunity_labels"])
    out = out.with_columns(label.alias("_label")).with_columns(
        pl.col("_label").replace_strict({c: i for i, c in enumerate(classes)}, default=-1).alias(f"{prefix}label_idx"),
        pl.col("_label").str.split(":").list.first().replace_strict(KIND_CODES, default=0).alias(f"{prefix}kind"),
        pl.col("_gv").alias(f"{prefix}growth_increment"),
        pl.col("_av").alias(f"{prefix}adopt_increment"),
        pl.col("_fut_n").alias(f"{prefix}qual_purchases"),
        *[(pl.col("_label") == f"Existing-category growth: {g}").cast(pl.Int8).alias(f"{prefix}is_growth__{safe_name(g)}")
          for g in GROWTH_GROUPS + ("Other",)],
    )
    return out.select([c for c in out.columns if not c.startswith("_")])


def build_features(
    tx: pl.DataFrame,
    cutoff: str,
    label_config: dict,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    slope_buckets: int = 6,
    include_cutoff_month: bool = True,
) -> pl.DataFrame:
    t = datetime.strptime(cutoff, "%Y-%m-%d")
    t_prevq = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).to_pydatetime()
    t_ly_start = (pd.Timestamp(cutoff) - pd.DateOffset(months=12)).to_pydatetime()
    t_ly_end = (pd.Timestamp(cutoff) - pd.DateOffset(months=9)).to_pydatetime()

    cats = included_categories(label_config)
    cat2 = pl.col("ItemCategoryLevel2").cast(pl.String)

    h = (
        tx.lazy()
        .filter(pl.col("ts") < t)
        .with_columns(
            ((pl.lit(t) - pl.col("ts")).dt.total_seconds() / 86400.0).alias("dbc"),
            cat2.alias("cat"),
            pl.col("ts").dt.date().alias("date"),
            pl.col("ts").dt.hour().alias("hour"),
            pl.col("ts").dt.weekday().alias("wd"),
            pl.col("ts").dt.day().alias("dom"),
        )
        .with_columns(
            (~pl.col("is_adjustment") & pl.col("cat").is_in(cats)).alias("in_scope"),
        )
        .with_columns(
            (pl.col("in_scope") & (pl.col("cat") != "Fuel")).alias("is_nf"),
            (~pl.col("is_adjustment") & (pl.col("cat") == "Fuel")
             & (pl.col("LineItemType").cast(pl.String) == "Sale_Fuel")).alias("is_fuel_line"),
            (pl.col("in_scope") & (pl.col("qty") > 0) & (pl.col("amount") > 0)).alias("qual"),
        )
        .with_columns(
            pl.when(pl.col("in_scope")).then(pl.col("amount")).otherwise(0.0).alias("spend"),
            pl.when(pl.col("is_nf")).then(pl.col("amount")).otherwise(0.0).alias("nf_spend"),
            pl.when(pl.col("is_fuel_line")).then(pl.col("qty")).otherwise(0.0).alias("litres"),
        )
    ).collect()

    if h.height == 0:
        return pl.DataFrame({ID: pl.Series([], dtype=pl.String)})

    txn = pl.col("TransactionId")
    w_all = list(windows) + [None]

    # ------------------------------------------------------------ RFM per window
    aggs: List[pl.Expr] = [
        pl.col("dbc").min().alias("recency_days"),
        pl.col("dbc").max().alias("tenure_days"),
        pl.col("dbc").filter(pl.col("is_fuel_line") & (pl.col("qty") > 0)).min().alias("recency_fuel_days"),
        pl.col("dbc").filter(pl.col("is_nf") & pl.col("qual")).min().alias("recency_nonfuel_days"),
        pl.col("LocationId").n_unique().alias("n_stations_all"),
        pl.col("ItemName").filter(pl.col("in_scope")).n_unique().alias("n_items_all"),
        (pl.col("qty") < 0).sum().alias("n_return_lines"),
        pl.col("ItemName").eq("PENSIONER DISCOUNT").sum().alias("n_pensioner_discount"),
        pl.col("ItemName").eq("WOW_PROMO").sum().alias("n_wow_promo"),
        pl.len().alias("n_lines_all"),
        # label-aligned calendar windows
        pl.col("litres").filter(pl.col("ts") >= t_prevq).sum().alias("fuel_litres_prevq"),
        pl.col("nf_spend").filter(pl.col("ts") >= t_prevq).sum().alias("nonfuel_spend_prevq"),
        pl.col("spend").filter(pl.col("ts") >= t_prevq).sum().alias("spend_prevq"),
        txn.filter(pl.col("ts") >= t_prevq).n_unique().alias("baskets_prevq"),
        pl.col("litres").filter(pl.col("ts").is_between(t_ly_start, t_ly_end, closed="left")).sum().alias("fuel_litres_ly_q"),
        pl.col("nf_spend").filter(pl.col("ts").is_between(t_ly_start, t_ly_end, closed="left")).sum().alias("nonfuel_spend_ly_q"),
        txn.filter(pl.col("ts").is_between(t_ly_start, t_ly_end, closed="left")).n_unique().alias("baskets_ly_q"),
    ]
    for w in w_all:
        sfx = f"{w}d" if w else "all"
        m = (pl.col("dbc") <= w) if w else pl.lit(True)
        aggs += [
            txn.filter(m).n_unique().alias(f"baskets_{sfx}"),
            pl.col("date").filter(m).n_unique().alias(f"active_days_{sfx}"),
            pl.col("spend").filter(m).sum().alias(f"spend_{sfx}"),
            pl.col("litres").filter(m).sum().alias(f"fuel_litres_{sfx}"),
            pl.col("nf_spend").filter(m).sum().alias(f"nonfuel_spend_{sfx}"),
            txn.filter(m & pl.col("is_fuel_line") & (pl.col("qty") > 0)).n_unique().alias(f"fuel_baskets_{sfx}"),
            txn.filter(m & pl.col("is_nf") & pl.col("qual")).n_unique().alias(f"nonfuel_baskets_{sfx}"),
        ]
        if w and w >= 30:
            aggs.append(pl.col("cat").filter(m & pl.col("qual")).n_unique().alias(f"n_cats_{sfx}"))

    # ------------------------------------------------------------ 30-day buckets for slopes
    bucket = (pl.col("dbc") // 30).cast(pl.Int32)
    for k in range(slope_buckets):
        mk = bucket == k
        aggs += [
            pl.col("spend").filter(mk).sum().alias(f"_spend_b{k}"),
            pl.col("litres").filter(mk).sum().alias(f"_litres_b{k}"),
            pl.col("nf_spend").filter(mk).sum().alias(f"_nf_b{k}"),
            txn.filter(mk).n_unique().alias(f"_visits_b{k}"),
        ]

    base = h.group_by(ID).agg(aggs)

    # ------------------------------------------------------------ basket-level cadence
    baskets = h.group_by(ID, "TransactionId").agg(
        pl.col("ts").min(), pl.col("hour").first(), pl.col("wd").first(), pl.col("dom").first(),
        pl.col("LocationId").first().cast(pl.String).alias("loc"), pl.col("spend").sum().alias("bspend"),
        pl.len().alias("lines"),
    )
    pay = (pl.col("dom") >= 25) | (pl.col("dom") <= 2)
    cadence = baskets.group_by(ID).agg(
        pl.col("hour").is_between(6, 8).mean().alias("morning_rush_ratio"),
        pl.col("hour").is_between(16, 18).mean().alias("evening_rush_ratio"),
        (pl.col("hour").is_between(6, 8) | pl.col("hour").is_between(16, 18)).mean().alias("commute_rush_ratio"),
        (pl.col("hour") >= 20).mean().alias("night_ratio"),
        (pl.col("wd") >= 6).mean().alias("weekend_ratio"),
        pay.mean().alias("paycycle_basket_ratio"),
        (pl.col("bspend").filter(pay).sum() / (pl.col("bspend").sum().abs() + EPS)).alias("paycycle_spend_ratio"),
        pl.col("bspend").mean().alias("basket_value_mean"),
        pl.col("bspend").std().alias("basket_value_std"),
        pl.col("bspend").max().alias("basket_value_max"),
        pl.col("lines").mean().alias("lines_per_basket"),
        pl.col("hour").mean().alias("hour_mean"),
        pl.col("hour").std().alias("hour_std"),
    )
    station = (
        baskets.group_by(ID, "loc").agg(pl.len().alias("n"))
        .group_by(ID).agg(
            (pl.col("n").max() / pl.col("n").sum()).alias("primary_station_share"),
            (-(pl.col("n") / pl.col("n").sum() * (pl.col("n") / pl.col("n").sum()).log()).sum()).alias("station_entropy"),
        )
    )

    # ------------------------------------------------------------ inter-purchase intervals
    ipi = (
        h.select(ID, "date").unique().sort(ID, "date")
        .with_columns((pl.col("date") - pl.col("date").shift(1).over(ID)).dt.total_days().cast(pl.Float64).alias("gap"))
        .drop_nulls("gap")
        .group_by(ID).agg(
            pl.col("gap").mean().alias("ipi_mean"), pl.col("gap").std().alias("ipi_std"),
            pl.col("gap").min().alias("ipi_min"), pl.col("gap").max().alias("ipi_max"),
            pl.col("gap").median().alias("ipi_median"), pl.col("gap").skew().alias("ipi_skew"),
            pl.col("gap").tail(3).mean().alias("ipi_last3_mean"),
        )
    )

    # ------------------------------------------------------------ per-category signals
    hc = h.filter(pl.col("in_scope"))
    cat_stats = hc.group_by(ID, "cat").agg(
        pl.col("amount").sum().alias("net_all"),
        pl.col("amount").filter(pl.col("ts") >= t_prevq).sum().alias("prevq"),
        pl.col("amount").filter(pl.col("dbc") <= 30).sum().alias("s30"),
        pl.col("amount").filter((pl.col("dbc") > 30) & (pl.col("dbc") <= 90)).sum().alias("s31_90"),
        pl.col("TransactionId").filter(pl.col("qual")).n_unique().alias("qual_baskets"),
        pl.col("dbc").filter(pl.col("qual")).min().alias("recency"),
    )
    tot = cat_stats.group_by(ID).agg(pl.col("net_all").clip(lower_bound=0).sum().alias("_pos_total"))
    cat_stats = cat_stats.join(tot, on=ID).with_columns(
        (pl.col("net_all").clip(lower_bound=0) / (pl.col("_pos_total") + EPS)).alias("share"),
        _ratio("s30", "s31_90", 2.0).alias("growth_30_vs_prior60"),
    )
    entropy = cat_stats.group_by(ID).agg(
        (-(pl.when(pl.col("share") > 0).then(pl.col("share") * pl.col("share").log()).otherwise(0.0))).sum().alias("category_entropy"),
        (pl.col("qual_baskets") > 0).sum().alias("n_cats_ever"),
    )
    cat_metrics = ["share", "prevq", "qual_baskets", "growth_30_vs_prior60", "recency"]
    wide = cat_stats.pivot(on="cat", index=ID, values=cat_metrics, aggregate_function="first")
    rename, keep = {}, [ID]
    for mtr in cat_metrics:
        for c in cats:
            src = f"{mtr}_{c}" if len(cat_metrics) > 1 else c
            dst = f"cat_{mtr}__{safe_name(c)}"
            if src in wide.columns:
                rename[src] = dst
                keep.append(src)
    wide = wide.select(keep).rename(rename)
    # Guarantee an identical schema across cohorts even if a category is unseen.
    fill_zero = {"share", "prevq", "qual_baskets"}
    for mtr in cat_metrics:
        for c in cats:
            col = f"cat_{mtr}__{safe_name(c)}"
            if col not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
            if mtr in fill_zero:
                wide = wide.with_columns(pl.col(col).cast(pl.Float64).fill_null(0.0))
    untouched = [(pl.col(f"cat_qual_baskets__{safe_name(c)}") == 0).cast(pl.Int8).alias(f"cat_untouched__{safe_name(c)}") for c in cats]
    wide = wide.with_columns(untouched)
    wide = wide.with_columns(pl.sum_horizontal([f"cat_untouched__{safe_name(c)}" for c in cats]).alias("n_cats_untouched"))

    # ------------------------------------------------------------ growth-hurdle gaps (label_rules thresholds)
    # Next quarter grows category c iff spend >= prevq + R20 and > 1.25 * max(prevq, 0). Compare that
    # hurdle with what the customer actually spent in each of the three quarters before prevq.
    q_edges = [(pd.Timestamp(cutoff) - pd.DateOffset(months=3 * k)).to_pydatetime() for k in range(5)]
    rel = label_config["growth_min_relative"] or 0.0
    hurdle = pl.max_horizontal(pl.col("q1") + label_config["growth_min_rand"],
                               pl.col("q1").clip(lower_bound=0) * (1 + rel))
    qcat = hc.group_by(ID, "cat").agg(
        [pl.col("amount").filter((pl.col("ts") >= q_edges[k + 1]) & (pl.col("ts") < q_edges[k])).sum().alias(f"q{k + 1}")
         for k in range(4)]
    ).with_columns(hurdle.alias("hurdle"), pl.max_horizontal("q2", "q3", "q4").alias("qmax"),
                   ((pl.col("q2") + pl.col("q3") + pl.col("q4")) / 3).alias("qmean"))
    qcat = qcat.with_columns((pl.col("qmax") - pl.col("hurdle")).alias("gap_max"),
                             (pl.col("qmean") - pl.col("hurdle")).alias("gap_mean"))
    is_other = ~pl.col("cat").is_in(list(GROWTH_GROUPS))
    gap_aggs: List[pl.Expr] = []
    for g in GROWTH_GROUPS:
        sg = safe_name(g)
        sel = pl.col("cat") == g
        gap_aggs += [pl.col("hurdle").filter(sel).first().alias(f"grow_hurdle__{sg}"),
                     pl.col("gap_max").filter(sel).first().alias(f"grow_gap_max__{sg}"),
                     pl.col("gap_mean").filter(sel).first().alias(f"grow_gap_mean__{sg}"),
                     (pl.col("q1") / (pl.col("q2").abs() + EPS)).filter(sel).first().alias(f"q1_over_q2__{sg}")]
    gap_aggs += [pl.col("gap_max").filter(is_other).max().alias("grow_gap_max__other"),
                 pl.col("gap_mean").filter(is_other).max().alias("grow_gap_mean__other"),
                 (pl.col("gap_max") > 0).sum().alias("n_cats_cleared_hurdle_before"),
                 pl.col("gap_max").max().alias("grow_gap_max__any")]
    gaps = qcat.group_by(ID).agg(gap_aggs)

    # ------------------------------------------------------------ replayed label states (lag 1 and 2 quarters)
    t_lag2 = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).to_pydatetime()
    lag1 = rule_state(h, t, label_config, "lag1_")
    lag2 = rule_state(h, t_lag2, label_config, "lag2_")

    # ------------------------------------------------------------ assemble + derived ratios
    f = (
        base.join(cadence, on=ID, how="left")
        .join(station, on=ID, how="left")
        .join(ipi, on=ID, how="left")
        .join(entropy, on=ID, how="left")
        .join(wide, on=ID, how="left")
        .join(gaps, on=ID, how="left")
        .join(lag1, on=ID, how="left")
        .join(lag2, on=ID, how="left")
    )
    lag_cols = [c for c in f.columns if c.startswith(("lag1_", "lag2_"))]
    f = f.with_columns(
        [pl.col(c).fill_null(-1 if c.endswith("label_idx") else 0) for c in lag_cols]
        + [(pl.col("lag1_label_idx") == pl.col("lag2_label_idx")).cast(pl.Int8).alias("lag_label_repeat"),
           (pl.col("lag1_kind") == 3).cast(pl.Int8).add((pl.col("lag2_kind") == 3).cast(pl.Int8)).alias("lag_growth_count"),
           (pl.col("lag1_kind") == 1).cast(pl.Int8).add((pl.col("lag2_kind") == 1).cast(pl.Int8)).alias("lag_inactive_count")]
    )
    # Customers with only out-of-scope history are absent from `wide`: zero activity, all untouched.
    f = f.with_columns(
        [pl.col(f"cat_{m}__{safe_name(c)}").fill_null(0.0) for m in fill_zero for c in cats]
        + [pl.col(f"cat_untouched__{safe_name(c)}").fill_null(1) for c in cats]
        + [pl.col("n_cats_untouched").fill_null(len(cats)), pl.col("n_cats_ever").fill_null(0)]
    )
    derived = [
        _ratio("spend_30d", "spend_90d", 3.0).alias("mom_spend_30_90"),
        _ratio("baskets_14d", "baskets_60d", 60 / 14).alias("mom_visits_14_60"),
        _ratio("fuel_litres_30d", "fuel_litres_90d", 3.0).alias("mom_litres_30_90"),
        _ratio("nonfuel_spend_30d", "nonfuel_spend_90d", 3.0).alias("mom_nonfuel_30_90"),
        _ratio("spend_90d", "spend_180d", 2.0).alias("mom_spend_90_180"),
        _ratio("baskets_30d", "baskets_180d", 6.0).alias("mom_visits_30_180"),
        _ratio("fuel_litres_90d", "fuel_litres_180d", 2.0).alias("mom_litres_90_180"),
        _ratio("nonfuel_spend_90d", "nonfuel_spend_180d", 2.0).alias("mom_nonfuel_90_180"),
        (pl.col("recency_days") / (pl.col("ipi_mean") + EPS)).alias("recency_to_ipi"),
        (pl.col("recency_days") / (pl.col("ipi_max") + EPS)).alias("recency_to_ipi_max"),
        _ratio("baskets_all", "tenure_days", 7.0).alias("visits_per_week_all"),
        _ratio("nonfuel_spend_all", "spend_all").alias("nonfuel_share_all"),
        _ratio("nonfuel_spend_90d", "spend_90d").alias("nonfuel_share_90d"),
        _ratio("fuel_litres_90d", "fuel_baskets_90d").alias("litres_per_fill_90d"),
        _ratio("fuel_litres_all", "fuel_baskets_all").alias("litres_per_fill_all"),
        _ratio("nonfuel_spend_all", "nonfuel_baskets_all").alias("nonfuel_per_basket_all"),
        (pl.col("fuel_litres_prevq") - pl.col("fuel_litres_ly_q")).alias("fuel_prevq_minus_ly"),
        (pl.col("nonfuel_spend_prevq") - pl.col("nonfuel_spend_ly_q")).alias("nonfuel_prevq_minus_ly"),
        (pl.col("fuel_litres_prevq").clip(lower_bound=0).log1p()).alias("log_fuel_litres_prevq"),
        (pl.col("nonfuel_spend_prevq").clip(lower_bound=0).log1p()).alias("log_nonfuel_spend_prevq"),
        _slope_expr([f"_spend_b{k}" for k in range(slope_buckets)]).alias("slope_spend_monthly"),
        _slope_expr([f"_litres_b{k}" for k in range(slope_buckets)]).alias("slope_litres_monthly"),
        _slope_expr([f"_nf_b{k}" for k in range(slope_buckets)]).alias("slope_nonfuel_monthly"),
        _slope_expr([f"_visits_b{k}" for k in range(slope_buckets)]).alias("slope_visits_monthly"),
        pl.sum_horizontal([(pl.col(f"_visits_b{k}") > 0).cast(pl.Int8) for k in range(slope_buckets)]).alias("active_months_6"),
        (pl.concat_list([f"_spend_b{k}" for k in range(slope_buckets)]).list.std()
         / (pl.concat_list([f"_spend_b{k}" for k in range(slope_buckets)]).list.mean().abs() + EPS)).alias("cv_spend_monthly"),
        (pl.concat_list([f"_visits_b{k}" for k in range(slope_buckets)]).list.std()
         / (pl.concat_list([f"_visits_b{k}" for k in range(slope_buckets)]).list.mean() + EPS)).alias("cv_visits_monthly"),
    ]
    for w in windows:
        derived.append((pl.col(f"baskets_{w}d") / (w / 7.0)).alias(f"visits_per_week_{w}d"))
    f = f.with_columns(derived)
    f = f.drop([c for c in f.columns if c.startswith("_")])
    if include_cutoff_month:
        f = f.with_columns(pl.lit(t.month, dtype=pl.Int8).alias("cutoff_month"))

    # Normalise dtypes; inf -> null so GBDTs treat them as missing.
    num_cols = [c for c in f.columns if c != ID]
    f = f.with_columns([
        pl.col(c).cast(pl.Float64).pipe(lambda e: pl.when(e.is_infinite()).then(None).otherwise(e)).alias(c)
        for c in num_cols
    ])
    return f.with_columns(pl.col(ID).cast(pl.String)).sort(ID)


def feature_columns(df: pd.DataFrame | pl.DataFrame) -> List[str]:
    return [c for c in df.columns if c != ID]
