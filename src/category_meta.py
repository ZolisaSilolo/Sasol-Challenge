"""
Per-category label-rule meta-features for the Opportunity head (and optionally the CLV heads).

The Opportunity label is a deterministic function of per-category outcomes (LABEL_RULES.md 5-7):
each included Level2 category either qualifies for growth (historical purchase, >= 1 future
purchase, delta >= R20 and future > 1.25 x previous quarter) or for adoption (no prior purchase,
positive future spend), and the qualifying category with the largest monetary increment wins.

This module
1. builds the per-category *outcomes* for a labelled cohort with the exact rule logic used by
   ``features.rule_state`` (``rule_per_category``): grow/adopt flags and increments per category;
2. fits, per frequent (kind, category) pair, a binary model P(category qualifies) and a
   positive-only regressor E[log1p(increment) | qualifies], OOF on the stack folds;
3. turns them into meta-features: per-pair probabilities and expected scores P x E, per-label
   aggregates (P(any category of that label qualifies), max expected score), the predicted
   winning label under the rules' "largest increment wins" logic, and P(no candidate) ~ Stable.

Leakage: outcomes are labels-side information and are only used as *targets*. Training rows only
ever receive OOF predictions; scoring rows receive fold-averaged predictions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from src.features import included_categories, rule_per_category, safe_name
from src.utils import SEED, setup_logger

logger = setup_logger("sasol_catmeta")
ID = "ID"
GROWTH, ADOPT = "Existing-category growth", "New category adoption"
CATOUT_VERSION = "c1"


# ============================================================================ outcomes (targets)
def flag_history(tx: pl.DataFrame, end, label_config: dict) -> pl.DataFrame:
    cats = included_categories(label_config)
    return (
        tx.lazy().filter(pl.col("ts") < end)
        .with_columns(pl.col("ItemCategoryLevel2").cast(pl.String).alias("cat"))
        .with_columns((~pl.col("is_adjustment") & pl.col("cat").is_in(cats)).alias("in_scope"))
        .with_columns((pl.col("in_scope") & (pl.col("qty") > 0) & (pl.col("amount") > 0)).alias("qual"))
        .select(ID, "TransactionId", "ts", "cat", "amount", "in_scope", "qual")
        .collect()
    )


def category_outcomes(tx: pl.DataFrame, cutoff: str, label_config: dict) -> pl.DataFrame:
    """Wide per-customer outcome frame for the window [cutoff, cutoff + 3m): one row per eligible ID with
    ``g__<cat>`` (growth increment if the category qualifies for growth, else 0) and ``a__<cat>``
    (adoption increment, else 0). Uses rows ts < cutoff + 3m, i.e. it is TARGET information."""
    end = (pd.Timestamp(cutoff) + pd.DateOffset(months=3)).to_pydatetime()
    h = flag_history(tx, end, label_config)
    elig, per_cat = rule_per_category(h, end, label_config)
    cats = included_categories(label_config)
    out = elig.with_columns(pl.col(ID).cast(pl.String))
    if per_cat.height:
        wide = per_cat.select(ID, "cat", "gv", "av").pivot(on="cat", index=ID, values=["gv", "av"],
                                                             aggregate_function="first")
        out = out.join(wide.with_columns(pl.col(ID).cast(pl.String)), on=ID, how="left")
    exprs = []
    for c in cats:
        for src, dst in (("gv", "g"), ("av", "a")):
            name = f"{src}_{c}"
            col = pl.col(name) if name in out.columns else pl.lit(0.0)
            exprs.append(col.cast(pl.Float64).fill_null(0.0).alias(f"{dst}__{c}"))
    return out.select(pl.col(ID), *exprs).sort(ID)


# ============================================================================ meta model
@dataclass
class Target:
    kind: str           # GROWTH or ADOPT
    cat: str            # original Level2 category
    label: str          # mapped Opportunity label
    col: str            # outcome column (g__cat / a__cat)

    @property
    def tag(self) -> str:
        return f"{'g' if self.kind == GROWTH else 'a'}_{safe_name(self.cat)}"


@dataclass
class CatMetaResult:
    train: np.ndarray
    others: Dict[str, np.ndarray]
    names: List[str]
    report: dict = field(default_factory=dict)


DEFAULT_PARAMS = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=100,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, verbose=-1)


def select_targets(Y: pd.DataFrame, label_config: dict, min_pos: int) -> List[Target]:
    mapping = label_config["category_mapping"]
    out = []
    for c in included_categories(label_config):
        for kind, pre in ((GROWTH, "g"), (ADOPT, "a")):
            col = f"{pre}__{c}"
            if col in Y and int((Y[col].to_numpy() > 0).sum()) >= min_pos:
                out.append(Target(kind, c, f"{kind}: {mapping[c][kind]}", col))
    return out


def fit_category_meta(X: np.ndarray, Y: pd.DataFrame, folds: np.ndarray, others: Dict[str, np.ndarray],
                      label_config: dict, classes: List[str], params: Optional[dict] = None, min_pos: int = 150,
                      seed: int = SEED, n_jobs: int = -1, weight: Optional[np.ndarray] = None) -> CatMetaResult:
    """OOF (train) / fold-averaged (others) per-category qualification probabilities + increments."""
    import lightgbm as lgb

    p = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed, "n_jobs": n_jobs}
    targets = select_targets(Y, label_config, min_pos)
    K = int(folds.max()) + 1
    n = len(X)
    t0 = time.time()
    P = np.zeros((n, len(targets)), np.float64)
    E = np.zeros((n, len(targets)), np.float64)
    Po = {o: np.zeros((len(Xo), len(targets))) for o, Xo in others.items()}
    Eo = {o: np.zeros((len(Xo), len(targets))) for o, Xo in others.items()}
    for j, t in enumerate(targets):
        inc = Y[t.col].to_numpy(dtype=np.float64)
        pos = inc > 0
        ylog = np.log1p(np.clip(inc, 0, None))
        for k in range(K):
            tr, va = folds != k, folds == k
            w = None if weight is None else weight[tr]
            clf = lgb.LGBMClassifier(objective="binary", **p).fit(X[tr], pos[tr].astype(int), sample_weight=w)
            P[va, j] = clf.predict_proba(X[va])[:, 1]
            trp = tr & pos
            reg = lgb.LGBMRegressor(objective="regression", **{**p, "min_child_samples": 20}).fit(
                X[trp], ylog[trp], sample_weight=None if weight is None else weight[trp])
            E[va, j] = reg.predict(X[va])
            for o, Xo in others.items():
                Po[o][:, j] += clf.predict_proba(Xo)[:, 1] / K
                Eo[o][:, j] += reg.predict(Xo) / K
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for j, t in enumerate(targets):
        y = (Y[t.col].to_numpy() > 0).astype(int)
        aucs[t.tag] = round(float(roc_auc_score(y, P[:, j])), 4) if 0 < y.sum() < len(y) else float("nan")
    logger.info(f"Category meta: {len(targets)} (kind, category) targets x {K} folds in {time.time() - t0:.0f}s | "
                f"OOF AUC {aucs}")
    train, names = derive_meta(P, E, targets, classes)
    oth = {o: derive_meta(Po[o], Eo[o], targets, classes)[0] for o in others}
    return CatMetaResult(train=train, others=oth, names=names,
                         report={"targets": [t.tag for t in targets], "oof_auc": aucs})


def derive_meta(P: np.ndarray, E: np.ndarray, targets: List[Target], classes: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Per-target P and expected score S = P x E[increment | qualifies]; per-label aggregates; rule-style winner."""
    n = len(P)
    S = P * np.expm1(np.clip(E, 0, 15))
    cols, names = [P, S], [f"cm_p__{t.tag}" for t in targets] + [f"cm_s__{t.tag}" for t in targets]
    labels = sorted({t.label for t in targets})
    LP = np.zeros((n, len(labels)))
    LS = np.zeros((n, len(labels)))
    for i, lab in enumerate(labels):
        idx = [j for j, t in enumerate(targets) if t.label == lab]
        LP[:, i] = 1.0 - np.prod(1.0 - P[:, idx], axis=1)
        LS[:, i] = S[:, idx].max(axis=1)
    cols += [LP, LS]
    names += [f"cm_lp__{safe_name(l)}" for l in labels] + [f"cm_ls__{safe_name(l)}" for l in labels]
    p_none = np.prod(1.0 - P, axis=1)
    srt = np.sort(LS, axis=1) if LS.shape[1] else np.zeros((n, 2))
    top = srt[:, -1] if srt.shape[1] else np.zeros(n)
    second = srt[:, -2] if srt.shape[1] > 1 else np.zeros(n)
    cmap = {c: i for i, c in enumerate(classes)}
    win = np.array([cmap[labels[i]] for i in LS.argmax(1)]) if LS.shape[1] else np.full(n, -1)
    g = np.array([t.kind == GROWTH for t in targets])
    extra = np.column_stack([p_none, P.sum(1), top, second, top - second, np.log1p(top) - np.log1p(second), win,
                             P[:, g].sum(1) if g.any() else np.zeros(n), P[:, ~g].sum(1) if (~g).any() else np.zeros(n)])
    cols.append(extra)
    names += ["cm_p_none", "cm_p_sum", "cm_top_score", "cm_second_score", "cm_margin", "cm_log_margin",
              "cm_winner_idx", "cm_p_growth_sum", "cm_p_adopt_sum"]
    return np.hstack(cols).astype(np.float32), names
