"""Feature factory + semantic embeddings: leakage, schema stability, correctness, dedup guardrail."""

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import embeddings as emb
from src.features import build_features, included_categories, safe_name

CUT = "2025-06-01"


@pytest.fixture(scope="module")
def feats(tx, label_config):
    return build_features(tx, CUT, label_config)


def test_no_future_leakage(tx, label_config, feats):
    """Features at cutoff t must be identical whether or not rows >= t exist."""
    past_only = tx.filter(pl.col("ts") < datetime(2025, 6, 1))
    again = build_features(past_only, CUT, label_config)
    assert again.columns == feats.columns
    a, b = feats.to_pandas().set_index("ID"), again.to_pandas().set_index("ID")
    pd.testing.assert_frame_equal(a, b.loc[a.index])


def test_future_perturbation_does_not_change_features(tx, label_config, feats):
    perturbed = tx.with_columns(
        pl.when(pl.col("ts") >= datetime(2025, 6, 1)).then(pl.col("amount") * 100).otherwise(pl.col("amount")).alias("amount"))
    again = build_features(perturbed, CUT, label_config)
    pd.testing.assert_frame_equal(feats.to_pandas(), again.to_pandas())


def test_schema_identical_across_cohorts(tx, label_config, feats):
    other = build_features(tx, "2024-09-01", label_config)
    assert other.columns == feats.columns
    assert feats["ID"].is_unique().all()


def test_required_feature_families_present(feats, label_config):
    cols = set(feats.columns)
    for w in ("7d", "14d", "30d", "60d", "90d", "180d", "all"):
        assert {f"baskets_{w}", f"spend_{w}", f"fuel_litres_{w}", f"nonfuel_spend_{w}", f"active_days_{w}"} <= cols
    for c in ("recency_days", "recency_fuel_days", "recency_nonfuel_days", "mom_spend_30_90", "mom_visits_14_60",
              "slope_spend_monthly", "ipi_mean", "ipi_std", "ipi_min", "ipi_max", "ipi_skew", "recency_to_ipi",
              "category_entropy", "morning_rush_ratio", "evening_rush_ratio", "weekend_ratio",
              "paycycle_basket_ratio", "primary_station_share"):
        assert c in cols, c
    for cat in included_categories(label_config):
        s = safe_name(cat)
        assert {f"cat_share__{s}", f"cat_untouched__{s}", f"cat_growth_30_vs_prior60__{s}"} <= cols


def test_values_match_label_rule_definitions(tx, feats, label_config):
    """prev-quarter fuel litres / non-fuel spend recomputed independently with pandas."""
    d = tx.to_pandas()
    t, t0 = pd.Timestamp(CUT), pd.Timestamp(CUT) - pd.DateOffset(months=3)
    adj = d.is_adjustment
    w = d[(d.ts >= t0) & (d.ts < t) & ~adj]
    cat = w.ItemCategoryLevel2.astype(str)
    fuel = w[(cat == "Fuel") & (w.LineItemType.astype(str) == "Sale_Fuel")].groupby("ID").qty.sum()
    nf = w[cat.isin(included_categories(label_config)) & (cat != "Fuel")].groupby("ID").amount.sum()
    f = feats.to_pandas().set_index("ID")
    np.testing.assert_allclose(f.fuel_litres_prevq.reindex(fuel.index), fuel, rtol=1e-9)
    np.testing.assert_allclose(f.nonfuel_spend_prevq.reindex(nf.index), nf, rtol=1e-9)
    # Baskets all-time = distinct TransactionId before cutoff (all lines, like eligibility).
    b = d[d.ts < t].groupby("ID").TransactionId.nunique()
    np.testing.assert_array_equal(f.baskets_all.reindex(b.index).to_numpy(), b.to_numpy().astype(float))


def test_no_infinite_values(feats):
    num = feats.drop("ID").to_numpy()
    assert not np.isinf(num[~np.isnan(num)]).any()


def test_embeddings_deduplicate_texts(tx, label_config, monkeypatch, tmp_path):
    """GUARDRAIL: the encoder receives each unique item text once, never per transaction line."""
    calls = []
    real = emb._encode_tfidf

    def spy(texts):
        calls.append(list(texts))
        return real(texts)

    monkeypatch.setattr(emb, "_encode_tfidf", spy)
    out = emb.semantic_features(tx, [CUT], label_config, {"backend": "tfidf", "n_components": 8},
                                tmp_path, "fp", force_recompute=True)
    assert len(calls) == 1
    texts = calls[0]
    n_classes = len(label_config["opportunity_labels"])
    assert len(texts) == len(set(texts[:-n_classes])) + n_classes   # items unique, then 17 class descriptions
    assert len(texts) - n_classes < tx.height / 10
    assert (tmp_path / "semantic_features.parquet").exists()
    f = out[CUT]
    assert f["ID"].is_unique().all()
    assert sum(c.startswith("sem_svd") for c in f.columns) == 2 * 8
    assert sum(c.startswith("sem_aff__") for c in f.columns) == 2 * n_classes

    # Cache hit: no further encoder calls.
    emb.semantic_features(tx, [CUT], label_config, {"backend": "tfidf", "n_components": 8}, tmp_path, "fp")
    assert len(calls) == 1


def test_opportunity_descriptions_cover_all_classes(label_config):
    d = emb.opportunity_descriptions(label_config["opportunity_labels"])
    assert set(d) == set(label_config["opportunity_labels"]) and all(d.values())


@pytest.mark.parametrize("cutoff", ["2025-03-01", "2025-06-01"])
def test_rule_replay_matches_label_rules(store, tx, label_config, cutoff):
    """lag1_* replays label_rules on [t-3m, t): must equal raw_labels(t-3m) for every eligible customer."""
    feats = build_features(tx, cutoff, label_config).to_pandas().set_index("ID")
    lag_cut = str((pd.Timestamp(cutoff) - pd.DateOffset(months=3)).date())
    ref = store.rules.raw_labels(store._prepared(), lag_cut, label_config).set_index("ID")
    ref.index = ref.index.astype(str)
    classes = label_config["opportunity_labels"]
    mine = feats.lag1_label_idx[feats.lag1_label_idx >= 0].astype(int).map(lambda i: classes[i])
    assert set(mine.index) == set(ref.index)
    pd.testing.assert_series_equal(mine.sort_index(), ref.Opportunity.sort_index(), check_names=False)
    assert (feats.lag1_label_idx >= 0).sum() > 0 and (feats.lag2_label_idx >= 0).sum() > 0


def test_rule_replay_and_hurdle_features_present(feats):
    cols = set(feats.columns)
    for c in ("lag1_label_idx", "lag1_kind", "lag2_label_idx", "lag_label_repeat", "lag_growth_count",
              "grow_hurdle__fuel", "grow_gap_max__fuel", "grow_gap_max__other", "q1_over_q2__beverages",
              "cv_spend_monthly", "cv_visits_monthly"):
        assert c in cols, c
