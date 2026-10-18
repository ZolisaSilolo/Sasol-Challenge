"""Fast OOT experiment harness (single LGBM per head) for comparing data/feature/model ideas.

Usage: python scripts/exp_harness.py --exp NAME [--evals 2024-12-01,2025-06-01,2025-09-01] [--catmeta]

--catmeta fits the per-category label-rule meta models (src/category_meta.py) OOF on the training
cohorts (GroupKFold on ID) and evaluates three variants in one run: base (no meta), cm_opp (meta on
the Opportunity head only) and cm_all (meta on all three heads).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.category_meta import CATOUT_VERSION, category_outcomes, fit_category_meta  # noqa: E402
from src.data_loader import CohortStore, add_months  # noqa: E402
from src.metrics import composite_metric  # noqa: E402
from src.pipeline import build_feature_frames, load_pipeline_config, make_folds  # noqa: E402

ID = "ID"


def month_starts(a: str, b: str):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(a, b, freq="MS")]


def load(cutoffs, emb=False):
    cfg = load_pipeline_config("configs/fast_config.yaml")
    cfg["embeddings"]["enabled"] = emb
    store = CohortStore(Path("data/raw"), Path("data/processed"))
    labels = store.materialise(cutoffs)
    frames = build_feature_frames(store, cutoffs, cfg, False)
    return store, labels, frames


def load_outcomes(store, cutoffs):
    out, tx = {}, None
    for c in cutoffs:
        path = store.cache_path(f"catout_{CATOUT_VERSION}_{c}")
        if not path.exists():
            tx = tx if tx is not None else store.transactions()
            category_outcomes(tx, c, store.label_config).write_parquet(path)
        out[c] = pl.read_parquet(path).to_pandas()
    return out


def assemble(labels, frames, cuts, outcomes=None):
    parts = []
    for c in cuts:
        df = labels[c].merge(frames[c], on=ID, how="left")
        if outcomes is not None:
            df = df.merge(outcomes[c], on=ID, how="left", validate="one_to_one")
        df["_cohort"] = c
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="base")
    ap.add_argument("--evals", default="2025-09-01,2025-06-01")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--first", default="2024-05-01", help="first monthly cohort")
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--mcs", type=int, default=50)
    ap.add_argument("--colsample", type=float, default=0.5)
    ap.add_argument("--clf-n", type=int, default=None, help="classifier trees (default n//2)")
    ap.add_argument("--clf-leaves", type=int, default=None)
    ap.add_argument("--clf-mcs", type=int, default=None)
    ap.add_argument("--clf-colsample", type=float, default=None)
    ap.add_argument("--emb", action="store_true")
    ap.add_argument("--catmeta", action="store_true")
    ap.add_argument("--ctx", action="store_true", help="add src/context_features (station context, fuel grades)")
    ap.add_argument("--cm-n", type=int, default=250)
    ap.add_argument("--cm-min-pos", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    import lightgbm as lgb

    lc_cuts = ["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]
    cuts = sorted(set(lc_cuts + (month_starts(a.first, "2025-09-01") if a.monthly else [])))
    store, labels, frames = load(cuts, a.emb)
    outcomes = load_outcomes(store, cuts) if a.catmeta else None
    if a.ctx:
        from src.context_features import CTX_VERSION, context_features
        tx = None
        for c in cuts:
            path = store.cache_path(f"ctx_{CTX_VERSION}_{c}")
            if not path.exists():
                tx = tx if tx is not None else store.transactions()
                context_features(tx, c).write_parquet(path)
            frames[c] = frames[c].merge(pl.read_parquet(path).to_pandas(), on=ID, how="left")
    lc = store.label_config
    classes = lc["opportunity_labels"]
    cmap = {c: i for i, c in enumerate(classes)}
    cols = sorted(set.intersection(*[set(f.columns) for f in frames.values()]) - {ID})
    res = []
    for E in a.evals.split(","):
        tr_cuts = [c for c in cuts if add_months(c, 3) <= pd.Timestamp(E)]
        tr, te = assemble(labels, frames, tr_cuts, outcomes), assemble(labels, frames, [E], outcomes)
        age = ((pd.Timestamp(E) - pd.to_datetime(tr["_cohort"])).dt.days / 30.44).round()
        w = np.power(a.decay, age / 3.0).to_numpy()
        Xtr, Xte = tr[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32)
        p = dict(n_estimators=a.n, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.mcs,
                 subsample=0.8, subsample_freq=1, colsample_bytree=a.colsample, reg_lambda=1.0, random_state=42,
                 verbose=-1, n_jobs=a.jobs)
        pc = {**p, "n_estimators": a.clf_n or a.n // 2}
        for k, v in (("num_leaves", a.clf_leaves), ("min_child_samples", a.clf_mcs), ("colsample_bytree", a.clf_colsample)):
            if v is not None:
                pc[k] = v
        ytr = tr.Opportunity.map(cmap).to_numpy()

        variants = {"base": (Xtr, Xte, Xtr, Xte)}
        if a.catmeta:
            t0 = time.time()
            folds = make_folds(tr[ID].astype(str).to_numpy(), 5, 42)
            cm = fit_category_meta(Xtr, tr, folds, {"te": Xte}, lc, classes, params={"n_estimators": a.cm_n},
                                   min_pos=a.cm_min_pos, n_jobs=a.jobs, weight=w)
            print(f"[{a.exp}] {E} catmeta {len(cm.names)} cols in {time.time() - t0:.0f}s", flush=True)
            Mtr, Mte = np.hstack([Xtr, cm.train]), np.hstack([Xte, cm.others["te"]])
            compact = [i for i, nm in enumerate(cm.names) if not nm.startswith(("cm_p__", "cm_s__"))]
            Ktr, Kte = np.hstack([Xtr, cm.train[:, compact]]), np.hstack([Xte, cm.others["te"][:, compact]])
            variants["cm_opp"] = (Xtr, Xte, Mtr, Mte)
            variants["cm_opp_compact"] = (Xtr, Xte, Ktr, Kte)
            variants["cm_all"] = (Mtr, Mte, Mtr, Mte)
        cache = {}
        for v, (Rtr, Rte, Ctr, Cte) in variants.items():
            t0 = time.time()
            key = id(Rtr)
            if key not in cache:
                cache[key] = (lgb.LGBMRegressor(**p).fit(Rtr, tr.CLV_fuel, sample_weight=w).predict(Rte).clip(0),
                              lgb.LGBMRegressor(**p).fit(Rtr, tr.CLV_nonfuel, sample_weight=w).predict(Rte).clip(0))
            pf, pn = cache[key]
            po = lgb.LGBMClassifier(**pc).fit(Ctr, ytr, sample_weight=w)
            P = np.zeros((len(te), len(classes)))
            P[:, po.classes_] = po.predict_proba(Cte)
            m = composite_metric(te.CLV_fuel, pf, te.CLV_nonfuel, pn, te.Opportunity.map(cmap), P.argmax(1))
            m.update(eval=E, variant=v, n_train=len(tr), secs=round(time.time() - t0))
            res.append(m)
            print(f"[{a.exp}/{v}] {E} n={len(tr)} fuel={m['rmse_fuel']:.4f} nf={m['rmse_nonfuel']:.4f} "
                  f"f1={m['f1_opportunity']:.4f} comp={m['composite_loss']:.4f} ({m['secs']}s)", flush=True)
    r = pd.DataFrame(res)
    for v, g in r.groupby("variant", sort=False):
        print(f"[{a.exp}/{v}] MEAN fuel={g.rmse_fuel.mean():.4f} nf={g.rmse_nonfuel.mean():.4f} "
              f"f1={g.f1_opportunity.mean():.4f} comp={g.composite_loss.mean():.4f}")


if __name__ == "__main__":
    main()
