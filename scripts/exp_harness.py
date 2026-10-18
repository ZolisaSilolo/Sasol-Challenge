"""Fast OOT experiment harness (single LGBM per head) for comparing data/feature/model ideas.

Usage: python scripts/exp_harness.py --exp NAME [--evals 2025-09-01,2025-06-01]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import CohortStore, add_months  # noqa: E402
from src.metrics import composite_metric  # noqa: E402
from src.pipeline import build_feature_frames, load_pipeline_config  # noqa: E402

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


def assemble(labels, frames, cuts, cols):
    parts = []
    for c in cuts:
        df = labels[c].merge(frames[c], on=ID, how="left")
        df["_cohort"] = c
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="base")
    ap.add_argument("--evals", default="2025-09-01,2025-06-01")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--mcs", type=int, default=50)
    ap.add_argument("--emb", action="store_true")
    a = ap.parse_args()
    import lightgbm as lgb

    lc_cuts = ["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]
    cuts = sorted(set(lc_cuts + (month_starts("2024-05-01", "2025-09-01") if a.monthly else [])))
    store, labels, frames = load(cuts, a.emb)
    classes = store.label_config["opportunity_labels"]
    cmap = {c: i for i, c in enumerate(classes)}
    cols = sorted(set.intersection(*[set(f.columns) for f in frames.values()]) - {ID})
    res = []
    for E in a.evals.split(","):
        tr_cuts = [c for c in cuts if add_months(c, 3) <= pd.Timestamp(E)]
        tr, te = assemble(labels, frames, tr_cuts, cols), assemble(labels, frames, [E], cols)
        age = ((pd.Timestamp(E) - pd.to_datetime(tr["_cohort"])).dt.days / 30.44).round()
        w = np.power(a.decay, age / 3.0).to_numpy()
        Xtr, Xte = tr[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32)
        p = dict(n_estimators=a.n, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.mcs,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=42,
                 verbose=-1, n_jobs=4)
        t0 = time.time()
        pf = lgb.LGBMRegressor(**p).fit(Xtr, tr.CLV_fuel, sample_weight=w).predict(Xte).clip(0)
        pn = lgb.LGBMRegressor(**p).fit(Xtr, tr.CLV_nonfuel, sample_weight=w).predict(Xte).clip(0)
        ytr = tr.Opportunity.map(cmap).to_numpy()
        po = lgb.LGBMClassifier(**{**p, "n_estimators": a.n // 2}).fit(Xtr, ytr, sample_weight=w)
        P = np.zeros((len(te), len(classes)))
        P[:, po.classes_] = po.predict_proba(Xte)
        m = composite_metric(te.CLV_fuel, pf, te.CLV_nonfuel, pn, te.Opportunity.map(cmap), P.argmax(1))
        m.update(eval=E, n_train=len(tr), secs=round(time.time() - t0))
        res.append(m)
        print(f"[{a.exp}] {E} n={len(tr)} fuel={m['rmse_fuel']:.4f} nf={m['rmse_nonfuel']:.4f} "
              f"f1={m['f1_opportunity']:.4f} comp={m['composite_loss']:.4f} ({m['secs']}s)", flush=True)
    r = pd.DataFrame(res)
    print(f"[{a.exp}] MEAN fuel={r.rmse_fuel.mean():.4f} nf={r.rmse_nonfuel.mean():.4f} "
          f"f1={r.f1_opportunity.mean():.4f} comp={r.composite_loss.mean():.4f}")


if __name__ == "__main__":
    main()
