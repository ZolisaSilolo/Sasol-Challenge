"""
Multi-cohort, leak-free three-head pipeline.

Data protocol
-------------
* Cohorts: one labelled snapshot per cutoff (default: the organisers' 5 calibration
  cutoffs). Features < cutoff, targets in [cutoff, cutoff + 3 months) from label_rules.
* Out-of-time benchmark (``validation.oot``): the latest labelled cohort is held out and
  every training cohort whose outcome window overlaps it is purged. The full stack is
  fitted on the remaining cohorts and scored on the OOT cohort.
* Final fit: all labelled cohorts, 5-fold GroupKFold grouped on customer ``ID`` (a
  customer never appears on both sides of a fold, across all cohorts).

Heads
-----
1. CLV_fuel: LGBM/CatBoost/XGBoost (RMSE) + auxiliary Huber model, simplex-blended on OOF.
2. CLV_nonfuel: Branch A direct regressors + Branch B hurdle (P(z>0) x E[z|z>0]),
   simplex-blended on OOF.
3. Opportunity: base features + OOF meta-features from Heads 1-2 (z_fuel_hat,
   z_nonfuel_hat, P(z_nonfuel>0), predicted deltas vs previous-quarter actuals).
   Training rows only ever see OOF meta-features; scoring rows see the fold-averaged
   predictions. LGBM/CatBoost/XGBoost probabilities are simplex-blended (log-loss), then
   nested-OOF class multipliers are applied.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupKFold

from src.data_loader import CohortStore, purge_cohorts, resolve_cohort_cutoffs
from src.embeddings import semantic_features
from src.features import FEATURE_VERSION, build_features
from src.metrics import composite_metric, rmse
from src.models.gbdt import MultiClassifier, Regressor
from src.models.hurdle import HurdleModel
from src.postprocessing import apply_class_weights, nested_class_weights, simplex_blend_weights, weighted_f1
from src.submission import read_sample_submission, validate_submission
from src.utils import SEED, deep_update, load_config, resolve_device, seed_everything, setup_logger

logger = setup_logger("sasol_pipeline")
ID = "ID"


# =============================================================================== config
def load_pipeline_config(path: str | Path) -> dict:
    """YAML loader supporting ``extends: other.yaml`` (deep-merged, relative to the file)."""
    path = Path(path)
    cfg = load_config(str(path))
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_update(load_pipeline_config(path.parent / parent), cfg)
    return cfg


# =============================================================================== data
@dataclass
class Dataset:
    name: str
    ids: np.ndarray
    cohort: np.ndarray
    X: np.ndarray
    prevq_fuel: np.ndarray
    prevq_nonfuel: np.ndarray
    y_fuel: Optional[np.ndarray] = None
    y_nonfuel: Optional[np.ndarray] = None
    y_opp: Optional[np.ndarray] = None
    weight: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.ids)


def _feature_cache_key(cfg: dict) -> str:
    f = cfg.get("features", {})
    return hashlib.sha256(json.dumps([FEATURE_VERSION, f.get("windows"), f.get("slope_buckets")]).encode()).hexdigest()[:8]


def build_feature_frames(store: CohortStore, cutoffs: Sequence[str], cfg: dict, force: bool) -> Dict[str, pd.DataFrame]:
    fcfg = cfg.get("features", {})
    key = _feature_cache_key(cfg)
    tx = None
    frames: Dict[str, pl.DataFrame] = {}
    for c in cutoffs:
        path = store.cache_path(f"features_{key}_{c}")
        if path.exists() and not force:
            frames[c] = pl.read_parquet(path)
            continue
        tx = tx if tx is not None else store.transactions()
        t0 = time.time()
        frames[c] = build_features(tx, c, store.label_config,
                                   windows=fcfg.get("windows", (7, 14, 30, 60, 90, 180)),
                                   slope_buckets=fcfg.get("slope_buckets", 6))
        frames[c].write_parquet(path)
        logger.info(f"Features {c}: {frames[c].shape} in {time.time() - t0:.1f}s -> {path.name}")

    ecfg = cfg.get("embeddings", {})
    if ecfg.get("enabled", True):
        tx = tx if tx is not None else store.transactions()
        sem = semantic_features(tx, cutoffs, store.label_config, {k: v for k, v in ecfg.items() if k != "enabled"},
                                store.processed_dir, store.fingerprint, force, cfg.get("seed", SEED))
        frames = {c: f.join(sem[c], on=ID, how="left") for c, f in frames.items()}
    return {c: f.to_pandas() for c, f in frames.items()}


def make_dataset(name: str, frames: Dict[str, pd.DataFrame], cutoffs: Sequence[str], feature_cols: List[str],
                 classes: List[str], labels: Optional[Dict[str, pd.DataFrame]] = None,
                 ids: Optional[pd.Series] = None, decay: float = 1.0, ref_cutoff: Optional[str] = None) -> Dataset:
    parts = []
    for c in cutoffs:
        f = frames[c]
        if labels is not None:
            df = labels[c].merge(f, on=ID, how="left", validate="one_to_one")
        else:
            df = pd.DataFrame({ID: ids.astype(str).values}).merge(f, on=ID, how="left", validate="one_to_one")
        df["_cohort"] = c
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    X = df.reindex(columns=feature_cols).to_numpy(dtype=np.float32)
    ds = Dataset(name=name, ids=df[ID].astype(str).to_numpy(), cohort=df["_cohort"].to_numpy(), X=X,
                 prevq_fuel=df["fuel_litres_prevq"].fillna(0.0).to_numpy(),
                 prevq_nonfuel=df["nonfuel_spend_prevq"].fillna(0.0).to_numpy())
    if labels is not None:
        cmap = {c: i for i, c in enumerate(classes)}
        ds.y_fuel = df["CLV_fuel"].to_numpy(dtype=np.float64)
        ds.y_nonfuel = df["CLV_nonfuel"].to_numpy(dtype=np.float64)
        ds.y_opp = df["Opportunity"].map(cmap).to_numpy()
        if np.isnan(ds.y_opp.astype(float)).any():
            raise ValueError("Unknown Opportunity label encountered")
        ds.y_opp = ds.y_opp.astype(int)
        ref = pd.Timestamp(ref_cutoff or max(cutoffs))
        age_q = ((ref - pd.to_datetime(df["_cohort"])).dt.days / 91.3125).round().to_numpy()
        ds.weight = np.power(decay, age_q).astype(np.float64)
    return ds


def make_folds(ids: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    folds = np.full(len(ids), -1)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(gkf.split(np.zeros(len(ids)), groups=ids)):
        folds[va] = k
    assert (folds >= 0).all()
    return folds


# =============================================================================== CV engine
@dataclass
class CVResult:
    oof: np.ndarray                         # (n, k)
    others: Dict[str, np.ndarray]           # name -> (n_other, k), fold-averaged


def run_cv(name: str, factory: Callable[[], object], X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray],
           folds: np.ndarray, others: Dict[str, np.ndarray], predict: Callable[[object, np.ndarray], np.ndarray],
           score: Callable[[np.ndarray, np.ndarray], float], score_name: str,
           fit_mask: Optional[np.ndarray] = None) -> CVResult:
    """K-fold fit; ``fit_mask`` restricts training rows (OOF still produced for every row)."""
    K = int(folds.max()) + 1
    oof, acc = None, {}
    t0 = time.time()
    for k in range(K):
        tr, va = folds != k, folds == k
        if fit_mask is not None:
            tr &= fit_mask
        model = factory()
        model.fit(X[tr], y[tr], None if w is None else w[tr])
        p_va = np.atleast_2d(predict(model, X[va]).T).T
        if oof is None:
            oof = np.zeros((len(X), p_va.shape[1]))
            acc = {o: np.zeros((len(Xo), p_va.shape[1])) for o, Xo in others.items()}
        oof[va] = p_va
        for o, Xo in others.items():
            acc[o] += np.atleast_2d(predict(model, Xo).T).T / K
        logger.info(f"  [{name}] fold {k + 1}/{K} {score_name}={score(y[va], p_va):.5f}")
    logger.info(f"  [{name}] OOF {score_name}={score(y, oof):.5f} ({time.time() - t0:.1f}s)")
    return CVResult(oof=oof, others=acc)


def _rmse0(y, p):
    return rmse(y, np.clip(p[:, 0], 0, None))


# =============================================================================== stack
@dataclass
class StackOutput:
    fuel_oof: np.ndarray
    nonfuel_oof: np.ndarray
    opp_probs_oof: np.ndarray
    opp_pred_oof: np.ndarray
    fuel: Dict[str, np.ndarray] = field(default_factory=dict)
    nonfuel: Dict[str, np.ndarray] = field(default_factory=dict)
    opp_probs: Dict[str, np.ndarray] = field(default_factory=dict)
    opp_pred: Dict[str, np.ndarray] = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def _engine_members(head_cfg: dict) -> Dict[str, dict]:
    return {e: (p or {}) for e, p in (head_cfg.get("engines") or {}).items()}


def fit_stack(train: Dataset, others: Dict[str, Dataset], cfg: dict, classes: List[str],
              s_fuel: float, s_nonfuel: float) -> StackOutput:
    seed = cfg.get("seed", SEED)
    device = resolve_device()
    hcfg = cfg.get("heads", {})
    folds = make_folds(train.ids, cfg.get("validation", {}).get("n_splits", 5), seed)
    Xo = {k: v.X for k, v in others.items()}
    report: dict = {"n_train": len(train), "fold_sizes": np.bincount(folds).tolist(), "device": device}
    C = len(classes)

    # ------------------------------------------------------------------ Head 1: fuel
    logger.info(f"Head 1 (CLV_fuel) on {len(train):,} rows [{train.name}]")
    fcfg = hcfg.get("fuel", {})
    members: Dict[str, CVResult] = {}
    for eng, params in _engine_members(fcfg).items():
        members[eng] = run_cv(f"fuel/{eng}", lambda e=eng, p=params: Regressor(e, p, seed, device),
                              train.X, train.y_fuel, train.weight, folds, Xo, lambda m, X: m.predict(X), _rmse0, "rmse")
    aux = fcfg.get("aux_huber")
    if aux:
        members[f"huber_{aux['engine']}"] = run_cv(
            f"fuel/huber_{aux['engine']}",
            lambda: Regressor(aux["engine"], aux.get("params"), seed, device, loss="huber"),
            train.X, train.y_fuel, train.weight, folds, Xo, lambda m, X: m.predict(X), _rmse0, "rmse")
    names = list(members)
    P = np.clip(np.column_stack([members[n].oof[:, 0] for n in names]), 0, None)
    wf = simplex_blend_weights(P, train.y_fuel, "rmse")
    fuel_oof = np.clip(P @ wf, 0, None)
    fuel_o = {o: np.clip(np.column_stack([np.clip(members[n].others[o][:, 0], 0, None) for n in names]) @ wf, 0, None)
              for o in others}
    report["fuel"] = {"members": {n: rmse(train.y_fuel, P[:, i]) for i, n in enumerate(names)},
                      "blend_weights": dict(zip(names, wf.round(4).tolist())), "oof_rmse": rmse(train.y_fuel, fuel_oof)}
    logger.info(f"Head 1 blend {report['fuel']['blend_weights']} -> OOF RMSE {report['fuel']['oof_rmse']:.5f}")

    # ------------------------------------------------------------------ Head 2: non-fuel
    logger.info("Head 2 (CLV_nonfuel): Branch A direct + Branch B hurdle")
    ncfg = hcfg.get("nonfuel", {})
    members = {}
    for eng, params in _engine_members(ncfg).items():
        members[f"direct_{eng}"] = run_cv(f"nonfuel/direct_{eng}", lambda e=eng, p=params: Regressor(e, p, seed, device),
                                          train.X, train.y_nonfuel, train.weight, folds, Xo,
                                          lambda m, X: m.predict(X), _rmse0, "rmse")
    hc = ncfg.get("hurdle", {})
    hurdle = run_cv(
        "nonfuel/hurdle",
        lambda: HurdleModel(hc.get("classifier_engines", ["lgbm"]), hc.get("regressor_engines", ["catboost"]),
                            hc.get("classifier_params"), hc.get("regressor_params"), hc.get("calibrate", True),
                            seed, device),
        train.X, train.y_nonfuel, train.weight, folds, Xo, lambda m, X: m.predict(X), _rmse0, "rmse")
    members["hurdle"] = hurdle
    names = list(members)
    P = np.clip(np.column_stack([members[n].oof[:, 0] for n in names]), 0, None)
    wn = simplex_blend_weights(P, train.y_nonfuel, "rmse")
    nonfuel_oof = np.clip(P @ wn, 0, None)
    nonfuel_o = {o: np.clip(np.column_stack([np.clip(members[n].others[o][:, 0], 0, None) for n in names]) @ wn, 0, None)
                 for o in others}
    p_pos_oof, p_pos_o = hurdle.oof[:, 1], {o: hurdle.others[o][:, 1] for o in others}
    from sklearn.metrics import roc_auc_score
    pos = (train.y_nonfuel > 0).astype(int)
    auc = float(roc_auc_score(pos, p_pos_oof)) if 0 < pos.sum() < len(pos) else float("nan")
    report["nonfuel"] = {"members": {n: rmse(train.y_nonfuel, P[:, i]) for i, n in enumerate(names)},
                         "blend_weights": dict(zip(names, wn.round(4).tolist())),
                         "oof_rmse": rmse(train.y_nonfuel, nonfuel_oof), "stage2a_oof_auc": auc}
    logger.info(f"Head 2 blend {report['nonfuel']['blend_weights']} -> OOF RMSE {report['nonfuel']['oof_rmse']:.5f} "
                f"| P(z>0) AUC {auc:.4f}")

    # ------------------------------------------------------------------ Head 3: Opportunity (OOF stacking)
    def meta(z_f, z_n, p_pos, ds: Dataset) -> np.ndarray:
        y_f_hat = np.expm1(z_f * s_fuel)
        y_n_hat = np.expm1(z_n * s_nonfuel)
        return np.column_stack([
            z_f, z_n, p_pos,
            y_f_hat - ds.prevq_fuel, y_n_hat - ds.prevq_nonfuel,
            z_f - np.log1p(np.clip(ds.prevq_fuel, 0, None)) / s_fuel,
            z_n - np.log1p(np.clip(ds.prevq_nonfuel, 0, None)) / s_nonfuel,
        ]).astype(np.float32)

    # Training rows: OOF predictions ONLY. Scoring rows: fold-averaged predictions.
    X3 = np.hstack([train.X, meta(fuel_oof, nonfuel_oof, p_pos_oof, train)])
    X3o = {o: np.hstack([others[o].X, meta(fuel_o[o], nonfuel_o[o], p_pos_o[o], others[o])]) for o in others}
    logger.info(f"Head 3 (Opportunity, {C} classes) with 7 OOF meta-features")
    ocfg = hcfg.get("opportunity", {})
    w3 = train.weight.copy() if train.weight is not None else np.ones(len(train))
    power = float(ocfg.get("class_weight_power", 0.0))
    if power > 0:
        freq = np.bincount(train.y_opp, minlength=C).astype(float)
        cw = np.where(freq > 0, (freq.sum() / np.maximum(freq, 1) / C) ** power, 0.0)
        w3 = w3 * cw[train.y_opp]

    def f1_score_probs(y, p):
        return weighted_f1(y, p.argmax(1), C)

    members = {}
    for eng, params in _engine_members(ocfg).items():
        members[eng] = run_cv(f"opp/{eng}", lambda e=eng, p=params: MultiClassifier(e, C, p, seed, device),
                              X3, train.y_opp, w3, folds, X3o, lambda m, X: m.predict_proba(X),
                              f1_score_probs, "wF1(argmax)")
    names = list(members)
    PP = np.stack([members[n].oof for n in names], axis=1)                   # n x m x C
    wo = simplex_blend_weights(PP, train.y_opp, "logloss")
    probs_oof = np.tensordot(PP, wo, axes=([1], [0]))
    probs_o = {o: np.tensordot(np.stack([members[n].others[o] for n in names], axis=1), wo, axes=([1], [0]))
               for o in others}

    pcfg = cfg.get("postprocessing", {}).get("class_weights", {})
    if pcfg.get("enabled", True):
        cw_res = nested_class_weights(probs_oof, train.y_opp, folds, C, lo=pcfg.get("lo", 0.2), hi=pcfg.get("hi", 5.0),
                                      grid_size=pcfg.get("grid_size", 25), max_passes=pcfg.get("max_passes", 3),
                                      method=pcfg.get("method", "coordinate"))
        cw_vec = cw_res.weights
        cw_rep = {"f1_argmax": cw_res.f1_argmax, "f1_nested_oof": cw_res.f1_nested,
                  "used_identity": cw_res.used_identity, "weights": dict(zip(classes, cw_vec.round(3).tolist()))}
        logger.info(f"Class multipliers: argmax wF1 {cw_res.f1_argmax:.5f} -> nested-OOF wF1 {cw_res.f1_nested:.5f}"
                    f"{' (kept argmax)' if cw_res.used_identity else ''}")
        # Honest OOF prediction = nested predictions (each fold scored with weights not fitted on it).
        opp_pred_oof = np.empty(len(train), dtype=int)
        for k in np.unique(folds):
            wk = np.ones(C) if cw_res.used_identity else cw_res.fold_weights[k]
            opp_pred_oof[folds == k] = apply_class_weights(probs_oof[folds == k], wk)
    else:
        cw_vec = np.ones(C)
        cw_rep = {"enabled": False}
        opp_pred_oof = probs_oof.argmax(1)

    report["opportunity"] = {"members": {n: weighted_f1(train.y_opp, members[n].oof.argmax(1), C) for n in names},
                             "blend_weights": dict(zip(names, wo.round(4).tolist())),
                             "oof_wf1": weighted_f1(train.y_opp, opp_pred_oof, C), "class_weights": cw_rep}
    report["oof"] = composite_metric(train.y_fuel, fuel_oof, train.y_nonfuel, nonfuel_oof, train.y_opp, opp_pred_oof)
    return StackOutput(
        fuel_oof=fuel_oof, nonfuel_oof=nonfuel_oof, opp_probs_oof=probs_oof, opp_pred_oof=opp_pred_oof,
        fuel=fuel_o, nonfuel=nonfuel_o, opp_probs=probs_o,
        opp_pred={o: apply_class_weights(probs_o[o], cw_vec) for o in others}, report=report,
    )


def _log_metrics(title: str, m: dict) -> None:
    logger.info("=" * 64)
    logger.info(title)
    logger.info(f"  RMSE CLV_fuel     (w=0.3): {m['rmse_fuel']:.5f}")
    logger.info(f"  RMSE CLV_nonfuel  (w=0.3): {m['rmse_nonfuel']:.5f}")
    logger.info(f"  wF1  Opportunity  (w=0.4): {m['f1_opportunity']:.5f}")
    logger.info(f"  Composite loss (lower=better): {m['composite_loss']:.5f}")
    logger.info("=" * 64)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# =============================================================================== entrypoint
def run_pipeline(config_path: str = "configs/base_config.yaml", overrides: Optional[dict] = None,
                 force_recompute: bool = False, sub_name: Optional[str] = None,
                 metrics_name: Optional[str] = None) -> tuple[dict, pd.DataFrame]:
    t_start = time.time()
    cfg = deep_update(load_pipeline_config(config_path), overrides or {})
    seed = int(cfg.get("seed", SEED))
    seed_everything(seed)
    paths = cfg.get("paths", {})
    raw_dir, proc_dir = Path(paths.get("raw_dir", "data/raw")), Path(paths.get("processed_dir", "data/processed"))
    sub_dir = Path(paths.get("sub_dir", "submissions"))
    sub_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Config {config_path} | seed {seed} | device {resolve_device()}")

    store = CohortStore(raw_dir, proc_dir, force_recompute=force_recompute)
    lc = store.label_config
    classes = list(lc["opportunity_labels"])
    s_fuel, s_nonfuel = lc["normalization"]["CLV_fuel"], lc["normalization"]["CLV_nonfuel"]
    ccfg = cfg.get("cohorts", {})
    label_cutoffs = resolve_cohort_cutoffs(ccfg, lc)
    test_cutoff = ccfg.get("test_cutoff", lc["public_history_end_exclusive"])
    logger.info(f"Labelled cohorts: {label_cutoffs} | test cutoff: {test_cutoff}")

    labels = store.materialise(label_cutoffs)
    frames = build_feature_frames(store, label_cutoffs + [test_cutoff], cfg, force_recompute)
    feature_cols = sorted(set.intersection(*[set(f.columns) for f in frames.values()]) - {ID})
    logger.info(f"{len(feature_cols)} features per customer")

    sample = read_sample_submission(raw_dir)
    test_path = raw_dir / "test.csv"
    if test_path.exists():
        test_ids = pd.read_csv(test_path, dtype={ID: str})[ID]
        assert (test_ids.values == sample[ID].values).all(), "test.csv and SampleSubmission.csv ID order differ"
    decay = float(ccfg.get("recency_weight_decay", 1.0))
    test = make_dataset("test", frames, [test_cutoff], feature_cols, classes, ids=sample[ID])
    missing = int(np.isnan(test.X[:, feature_cols.index("baskets_all")]).sum())
    if missing:
        logger.warning(f"{missing} test customers have no history before {test_cutoff} (features are NaN)")
    report: dict = {"config": str(config_path), "seed": seed, "label_cutoffs": label_cutoffs,
                    "test_cutoff": test_cutoff, "n_features": len(feature_cols)}

    # -------------------------------------------------------------- OOT benchmark
    vcfg = cfg.get("validation", {})
    if vcfg.get("oot", True) and len(label_cutoffs) >= 2:
        oot_cut = label_cutoffs[-1]
        tr_cuts = purge_cohorts(label_cutoffs[:-1], oot_cut)
        logger.info(f"OOT benchmark: train {tr_cuts} -> evaluate {oot_cut} (purged overlap)")
        tr = make_dataset("oot-train", frames, tr_cuts, feature_cols, classes, labels, decay=decay)
        oot = make_dataset("oot", frames, [oot_cut], feature_cols, classes, labels)
        out = fit_stack(tr, {"oot": oot}, cfg, classes, s_fuel, s_nonfuel)
        m = composite_metric(oot.y_fuel, out.fuel["oot"], oot.y_nonfuel, out.nonfuel["oot"], oot.y_opp, out.opp_pred["oot"])
        m["f1_opportunity_argmax"] = weighted_f1(oot.y_opp, out.opp_probs["oot"].argmax(1), len(classes))
        _log_metrics(f"OUT-OF-TIME benchmark @ {oot_cut} ({len(oot):,} customers)", m)
        report["oot"] = {"cutoff": oot_cut, "train_cutoffs": tr_cuts, "metrics": m, "stack": out.report}

    # -------------------------------------------------------------- final fit on all labelled cohorts
    full = make_dataset("full", frames, label_cutoffs, feature_cols, classes, labels, decay=decay)
    logger.info(f"Final fit on {len(full):,} rows from {len(label_cutoffs)} cohorts "
                f"({len(np.unique(full.ids)):,} unique customers)")
    out = fit_stack(full, {"test": test}, cfg, classes, s_fuel, s_nonfuel)
    _log_metrics(f"GROUPED {cfg.get('validation', {}).get('n_splits', 5)}-FOLD OOF (all cohorts)", out.report["oof"])
    report["final"] = out.report

    # -------------------------------------------------------------- submission
    sub = pd.DataFrame({
        "ID": sample[ID].values,
        "CLV_fuel": np.clip(out.fuel["test"], 0.0, None),
        "CLV_nonfuel": np.clip(out.nonfuel["test"], 0.0, None),
        "Opportunity": np.asarray(classes, dtype=object)[out.opp_pred["test"]],
    })
    expected = cfg.get("submission", {}).get("expected_rows", 5488)
    validate_submission(sub, sample, classes, expected_rows=expected)
    sub_path = sub_dir / (sub_name or paths.get("sub_filename", "submission.csv"))
    sub.to_csv(sub_path, index=False, float_format="%.10g")
    report["submission"] = {"path": str(sub_path), "rows": len(sub),
                            "opportunity_distribution": sub["Opportunity"].value_counts(normalize=True).round(4).to_dict(),
                            "clv_fuel_mean": float(sub.CLV_fuel.mean()), "clv_nonfuel_mean": float(sub.CLV_nonfuel.mean())}
    report["runtime_seconds"] = round(time.time() - t_start, 1)
    metrics_path = sub_dir / (metrics_name or paths.get("metrics_filename", "oof_metrics.json"))
    metrics_path.write_text(json.dumps(_jsonable(report), indent=2))
    logger.info(f"Validated submission -> {sub_path} | metrics -> {metrics_path} | {report['runtime_seconds']}s")
    return report, sub
