"""Opportunity decision-layer experiments (OOT), mirroring the production Head 3 more closely than exp_harness.

For each eval cutoff E (train on cohorts whose outcome window ends <= E):
  1. 5-fold GroupKFold (ID) OOF + full-fit test predictions for: CLV_fuel / CLV_nonfuel regressors (-> the
     pipeline's 7 meta-features), the 17-class classifier, and three binary "gate" models:
        active = not Inactivity, cand = has a qualifying growth/adoption candidate, fuelq = Fuel qualifies for growth.
  2. Variants (wF1 on the OOT cohort; every tuned parameter is fitted on training OOF only):
        A  multiclass argmax                       B  A + class multipliers (= production)
        C  hierarchical gates x multiclass + mult  D  geometric blend of A and C (lambda on OOF) + mult
        E  multiclass on X + meta + gate OOF probabilities (stacked) + mult

Usage: python scripts/opp_harness.py --evals 2024-12-01,2025-06-01,2025-09-01 [--jobs 12]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp_harness import assemble, load, load_outcomes, month_starts  # noqa: E402
from src.data_loader import add_months  # noqa: E402
from src.metrics import rmse  # noqa: E402
from src.pipeline import make_folds  # noqa: E402
from src.postprocessing import apply_class_weights, fit_class_weights, weighted_f1  # noqa: E402

ID = "ID"
REG = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=100, colsample_bytree=0.3,
           subsample=0.8, subsample_freq=1, reg_lambda=1.0, random_state=42, verbose=-1)
CLF = {**REG, "n_estimators": 500, "min_child_samples": 200}


def cv_fit(make, X, y, folds, Xte, proba: bool, n_out: int = 1):
    """OOF on X (folds) + test prediction from a model fitted on all rows."""
    def pred(m, Z):
        if not proba:
            return m.predict(Z)[:, None]
        out = np.zeros((len(Z), n_out))
        out[:, m.classes_] = m.predict_proba(Z)
        return out
    oof = np.zeros((len(X), n_out))
    for k in range(int(folds.max()) + 1):
        tr, va = folds != k, folds == k
        oof[va] = pred(make().fit(X[tr], y[tr]), X[va])
    return oof, pred(make().fit(X, y), Xte)


def meta(zf, zn, prevf, prevn, s_f, s_n):
    return np.column_stack([zf, zn, np.expm1(zf * s_f) - prevf, np.expm1(zn * s_n) - prevn,
                            zf - np.log1p(np.clip(prevf, 0, None)) / s_f,
                            zn - np.log1p(np.clip(prevn, 0, None)) / s_n]).astype(np.float32)


def hier(P, pa, pc, i_inact, i_stable):
    """P(Inact)=1-pa, P(Stable)=pa(1-pc), candidate classes share pa*pc in the multiclass proportions."""
    Q = P.copy()
    cand = np.ones(P.shape[1], bool)
    cand[[i_inact, i_stable]] = False
    Q[:, cand] = P[:, cand] / np.clip(P[:, cand].sum(1, keepdims=True), 1e-9, None) * (pa * pc)[:, None]
    Q[:, i_inact] = 1 - pa
    Q[:, i_stable] = pa * (1 - pc)
    return Q


def geo(P, Q, lam):
    L = (1 - lam) * np.log(np.clip(P, 1e-9, 1)) + lam * np.log(np.clip(Q, 1e-9, 1))
    L = np.exp(L - L.max(1, keepdims=True))
    return L / L.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="2024-12-01,2025-06-01,2025-09-01")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--exp", default="opp")
    a = ap.parse_args()
    import lightgbm as lgb

    lc_cuts = ["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]
    cuts = sorted(set(lc_cuts + month_starts("2024-05-01", "2025-09-01")))
    store, labels, frames = load(cuts)
    outcomes = load_outcomes(store, cuts)
    lc = store.label_config
    classes = list(lc["opportunity_labels"])
    C = len(classes)
    cmap = {c: i for i, c in enumerate(classes)}
    i_inact, i_stable = cmap["Inactivity"], cmap["Stable"]
    s_f, s_n = lc["normalization"]["CLV_fuel"], lc["normalization"]["CLV_nonfuel"]
    cols = sorted(set.intersection(*[set(f.columns) for f in frames.values()]) - {ID})
    reg = lambda: lgb.LGBMRegressor(**REG, n_jobs=a.jobs)  # noqa: E731
    clf = lambda: lgb.LGBMClassifier(**CLF, n_jobs=a.jobs)  # noqa: E731
    rows = []
    for E in a.evals.split(","):
        t0 = time.time()
        tr_cuts = [c for c in cuts if add_months(c, 3) <= pd.Timestamp(E)]
        tr, te = assemble(labels, frames, tr_cuts, outcomes), assemble(labels, frames, [E], outcomes)
        X, Xt = tr[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32)
        folds = make_folds(tr[ID].astype(str).to_numpy(), 5, 42)
        y, yt = tr.Opportunity.map(cmap).to_numpy(), te.Opportunity.map(cmap).to_numpy()

        zf, zft = cv_fit(reg, X, tr.CLV_fuel.to_numpy(), folds, Xt, False)
        zn, znt = cv_fit(reg, X, tr.CLV_nonfuel.to_numpy(), folds, Xt, False)
        zf, zft, zn, znt = (np.clip(v[:, 0], 0, None) for v in (zf, zft, zn, znt))
        M = meta(zf, zn, tr.fuel_litres_prevq.fillna(0).to_numpy(), tr.nonfuel_spend_prevq.fillna(0).to_numpy(), s_f, s_n)
        Mt = meta(zft, znt, te.fuel_litres_prevq.fillna(0).to_numpy(), te.nonfuel_spend_prevq.fillna(0).to_numpy(), s_f, s_n)
        X3, X3t = np.hstack([X, M]), np.hstack([Xt, Mt])

        P, Pt = cv_fit(clf, X3, y, folds, X3t, True, C)
        gates = {"active": y != i_inact, "cand": (y != i_inact) & (y != i_stable),
                 "fuelq": tr["g__Fuel"].to_numpy() > 0}
        G, Gt = {}, {}
        for g, yy in gates.items():
            o, t = cv_fit(lambda: lgb.LGBMClassifier(**{**CLF, "objective": "binary"}, n_jobs=a.jobs), X3,
                          yy.astype(int), folds, X3t, True, 2)
            G[g], Gt[g] = o[:, 1], t[:, 1]
        # P(cand | active) for the hierarchy (cand gate is unconditional -> divide by active, clipped).
        pc, pct = np.clip(G["cand"] / np.clip(G["active"], 1e-6, None), 0, 1), np.clip(Gt["cand"] / np.clip(Gt["active"], 1e-6, None), 0, 1)
        H, Ht = hier(P, G["active"], pc, i_inact, i_stable), hier(Pt, Gt["active"], pct, i_inact, i_stable)
        lams = np.linspace(0, 1, 11)
        lam = lams[int(np.argmax([weighted_f1(y, geo(P, H, l).argmax(1), C) for l in lams]))]
        D, Dt = geo(P, H, lam), geo(Pt, Ht, lam)
        GF = np.column_stack([G[g] for g in gates]).astype(np.float32)
        GFt = np.column_stack([Gt[g] for g in gates]).astype(np.float32)
        S, St = cv_fit(clf, np.hstack([X3, GF]), y, folds, np.hstack([X3t, GFt]), True, C)

        def mult(Po, Pte):
            w = fit_class_weights(Po, y, 0.2, 5.0, 15, 2)
            return apply_class_weights(Pte, w)
        res = {"A_argmax": Pt.argmax(1), "B_mult": mult(P, Pt), "C_hier_mult": mult(H, Ht),
               f"D_geo{lam:.1f}_mult": mult(D, Dt), "E_stack_mult": mult(S, St)}
        # Prior shift: class mix drifts across cohorts (Inactivity 5% -> 27%). Fit multipliers on the most
        # recent training cohorts' OOF only (F), or on all OOF after re-weighting rows to the recent class mix (G).
        coh = tr["_cohort"].to_numpy()
        recent = np.isin(coh, sorted(set(coh))[-3:])
        w_r = fit_class_weights(P[recent], y[recent], 0.2, 5.0, 15, 2)
        res["F_mult_recent3"] = apply_class_weights(Pt, w_r)
        pr_all = np.bincount(y, minlength=C) / len(y)
        pr_rec = np.bincount(y[recent], minlength=C) / recent.sum()
        ratio = np.where(pr_all > 0, pr_rec / np.clip(pr_all, 1e-9, None), 1.0)
        Ps, Pts = P * ratio, Pt * ratio
        Ps, Pts = Ps / Ps.sum(1, keepdims=True), Pts / Pts.sum(1, keepdims=True)
        res["G_prior_mult_recent3"] = apply_class_weights(Pts, fit_class_weights(Ps[recent], y[recent], 0.2, 5.0, 15, 2))
        # H: recency-weighted training of the classifier (decay 0.85 per quarter) + multipliers on recent OOF.
        age_q = ((pd.Timestamp(E) - pd.to_datetime(tr["_cohort"])).dt.days / 91.3).to_numpy()
        wt = np.power(0.85, age_q)
        Pw, Pwt = np.zeros_like(P), None
        for k in range(int(folds.max()) + 1):
            trk, va = folds != k, folds == k
            m = clf().fit(X3[trk], y[trk], sample_weight=wt[trk])
            tmp = np.zeros((va.sum(), C)); tmp[:, m.classes_] = m.predict_proba(X3[va]); Pw[va] = tmp
        m = clf().fit(X3, y, sample_weight=wt)
        Pwt = np.zeros((len(X3t), C)); Pwt[:, m.classes_] = m.predict_proba(X3t)
        res["H_decay_argmax"] = Pwt.argmax(1)
        res["H_decay_mult_recent3"] = apply_class_weights(Pwt, fit_class_weights(Pw[recent], y[recent], 0.2, 5.0, 15, 2))
        res["H_decay_mult_all"] = mult(Pw, Pwt)
        auc = {}
        from sklearn.metrics import roc_auc_score
        yg_te = {"active": yt != i_inact, "cand": (yt != i_inact) & (yt != i_stable), "fuelq": te["g__Fuel"].to_numpy() > 0}
        for g in gates:
            auc[g] = round(roc_auc_score(yg_te[g], Gt[g]), 4)
        print(f"[{a.exp}] {E} n={len(tr)} fuel={rmse(te.CLV_fuel, zft):.4f} nf={rmse(te.CLV_nonfuel, znt):.4f} "
              f"gate AUC {auc} ({time.time() - t0:.0f}s)", flush=True)
        for v, pred in res.items():
            f1 = weighted_f1(yt, pred, C)
            rows.append({"eval": E, "variant": "D_geo_mult" if v.startswith("D_") else v, "wf1": f1})
            print(f"[{a.exp}]   {E} {v:<16} wF1={f1:.4f}", flush=True)
    r = pd.DataFrame(rows)
    print(f"[{a.exp}] MEAN wF1 by variant:")
    print(r.groupby("variant").wf1.mean().round(4).to_string())


if __name__ == "__main__":
    main()
