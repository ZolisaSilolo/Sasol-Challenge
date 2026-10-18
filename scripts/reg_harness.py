"""CLV regression experiments (OOT): direct vs hurdle vs OOF-blended, for both targets.

Hurdle = P(z > 0) x E[z | z > 0]. The blend weight between direct and hurdle is fitted on training OOF
(GroupKFold on ID) and applied to the OOT cohort, as the production simplex blend would.

Usage: python scripts/reg_harness.py --evals 2024-12-01,2025-06-01,2025-09-01 [--jobs 4] [--seeds 1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp_harness import assemble, load, month_starts  # noqa: E402
from src.data_loader import add_months  # noqa: E402
from src.metrics import rmse  # noqa: E402
from src.pipeline import make_folds  # noqa: E402

ID = "ID"
P = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=100, colsample_bytree=0.3,
         subsample=0.8, subsample_freq=1, reg_lambda=1.0, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="2024-12-01,2025-06-01,2025-09-01")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--exp", default="reg")
    a = ap.parse_args()
    import lightgbm as lgb

    cuts = sorted(set(["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]
                      + month_starts("2024-05-01", "2025-09-01")))
    store, labels, frames = load(cuts)
    cols = sorted(set.intersection(*[set(f.columns) for f in frames.values()]) - {ID})
    rows = []
    for E in a.evals.split(","):
        t0 = time.time()
        tr_cuts = [c for c in cuts if add_months(c, 3) <= pd.Timestamp(E)]
        tr, te = assemble(labels, frames, tr_cuts), assemble(labels, frames, [E])
        X, Xt = tr[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32)
        folds = make_folds(tr[ID].astype(str).to_numpy(), 5, 42)
        K = int(folds.max()) + 1
        for tgt in ("CLV_fuel", "CLV_nonfuel"):
            y, yt = tr[tgt].to_numpy(), te[tgt].to_numpy()
            pos = y > 0

            def models(idx, Z, seed):
                d = lgb.LGBMRegressor(**P, random_state=seed, n_jobs=a.jobs).fit(X[idx], y[idx]).predict(Z)
                c = lgb.LGBMClassifier(**P, objective="binary", random_state=seed, n_jobs=a.jobs).fit(
                    X[idx], pos[idx].astype(int)).predict_proba(Z)[:, 1]
                ip = idx & pos if idx.dtype == bool else idx
                r = lgb.LGBMRegressor(**{**P, "min_child_samples": 50}, random_state=seed, n_jobs=a.jobs).fit(
                    X[ip], y[ip]).predict(Z)
                return np.clip(d, 0, None), c * np.clip(r, 0, None)

            oof_d, oof_h = np.zeros(len(X)), np.zeros(len(X))
            te_d, te_h = np.zeros(len(Xt)), np.zeros(len(Xt))
            for s in range(a.seeds):
                for k in range(K):
                    tri, va = folds != k, folds == k
                    d, h = models(tri, X[va], 42 + s)
                    oof_d[va] += d / a.seeds
                    oof_h[va] += h / a.seeds
                d, h = models(np.ones(len(X), bool), Xt, 42 + s)
                te_d += d / a.seeds
                te_h += h / a.seeds
            ws = np.linspace(0, 1, 21)
            w = ws[int(np.argmin([rmse(y, (1 - v) * oof_d + v * oof_h) for v in ws]))]
            res = {"direct": rmse(yt, te_d), "hurdle": rmse(yt, te_h), f"blend(w_h={w:.2f})": rmse(yt, (1 - w) * te_d + w * te_h)}
            for v, r in res.items():
                rows.append({"eval": E, "target": tgt, "variant": v.split("(")[0], "rmse": r})
            print(f"[{a.exp}] {E} {tgt:<12} " + " ".join(f"{k}={v:.4f}" for k, v in res.items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(rows)
    print(f"[{a.exp}] MEAN RMSE:")
    print(r.groupby(["target", "variant"]).rmse.mean().round(4).to_string())


if __name__ == "__main__":
    main()
