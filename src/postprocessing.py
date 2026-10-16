"""
Post-processing: simplex blend weights and nested-OOF class-multiplier optimisation.

Class multipliers
-----------------
Prediction rule: ``y_hat = argmax_c(w_c * P(y = c))`` with ``w in [lo, hi]^C``.
Fitting 17 multipliers on the full OOF set overfits rare classes, so
``nested_class_weights`` fits ``w^(k)`` on the K-1 folds excluding fold k, scores fold k
with it (honest nested-OOF F1), and averages the ``w^(k)`` for test inference. If the
nested F1 does not beat plain argmax, the identity vector is returned instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- metrics (fast)
def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Equivalent to sklearn f1_score(average='weighted', zero_division=0), O(n) via bincount."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    pred_cnt = cm.sum(axis=0).astype(float)
    denom = support + pred_cnt
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    total = support.sum()
    return float((f1 * support).sum() / total) if total else 0.0


def apply_class_weights(probs: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.argmax(probs * w[None, :], axis=1)


# --------------------------------------------------------------------------- blending
def simplex_blend_weights(preds: np.ndarray, y: np.ndarray, loss: str = "rmse") -> np.ndarray:
    """Non-negative weights summing to 1 minimising RMSE (preds: n x m) or log-loss (n x m x C)."""
    m = preds.shape[1]
    if m == 1:
        return np.ones(1)

    if loss == "rmse":
        def f(w):
            return float(np.sqrt(np.mean((preds @ w - y) ** 2)))
    elif loss == "logloss":
        yi = np.asarray(y, dtype=int)

        def f(w):
            p = np.tensordot(preds, w, axes=([1], [0]))
            p = np.clip(p[np.arange(len(yi)), yi], 1e-12, 1.0)
            return float(-np.mean(np.log(p)))
    else:
        raise ValueError(loss)

    w0 = np.full(m, 1.0 / m)
    res = minimize(f, w0, method="SLSQP", bounds=[(0.0, 1.0)] * m,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x if res.success else w0, 0, None)
    w = w / w.sum() if w.sum() > 0 else w0
    return w if f(w) <= f(w0) else w0


# --------------------------------------------------------------------------- class multipliers
def fit_class_weights(probs: np.ndarray, y: np.ndarray, lo: float = 0.2, hi: float = 5.0,
                      grid_size: int = 25, max_passes: int = 3, method: str = "coordinate") -> np.ndarray:
    """Coordinate ascent over a log-spaced grid (optionally refined by bounded Powell)."""
    n_classes = probs.shape[1]
    grid = np.geomspace(lo, hi, grid_size)
    w = np.ones(n_classes)
    best = weighted_f1(y, apply_class_weights(probs, w), n_classes)
    active = np.where(np.bincount(np.asarray(y, int), minlength=n_classes) > 0)[0]
    for _ in range(max_passes):
        improved = False
        for c in active:
            cur = w[c]
            for g in grid:
                w[c] = g
                s = weighted_f1(y, apply_class_weights(probs, w), n_classes)
                if s > best + 1e-12:
                    best, cur, improved = s, g, True
            w[c] = cur
        if not improved:
            break
    if method == "powell":
        res = minimize(lambda v: -weighted_f1(y, apply_class_weights(probs, v), n_classes), w,
                       method="Powell", bounds=[(lo, hi)] * n_classes, options={"maxiter": 2000, "xtol": 1e-3})
        if -res.fun > best:
            w = np.clip(res.x, lo, hi)
    return w


@dataclass
class ClassWeightResult:
    weights: np.ndarray
    f1_argmax: float
    f1_nested: float
    f1_in_sample: float
    used_identity: bool
    fold_weights: np.ndarray


def nested_class_weights(probs: np.ndarray, y: np.ndarray, folds: np.ndarray, n_classes: int,
                         lo: float = 0.2, hi: float = 5.0, grid_size: int = 25, max_passes: int = 3,
                         method: str = "coordinate", fallback_to_argmax: bool = True) -> ClassWeightResult:
    y = np.asarray(y, dtype=int)
    folds = np.asarray(folds)
    nested_pred = np.empty(len(y), dtype=int)
    ws = []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        w_k = fit_class_weights(probs[tr], y[tr], lo, hi, grid_size, max_passes, method)
        nested_pred[va] = apply_class_weights(probs[va], w_k)
        ws.append(w_k)
    W = np.vstack(ws)
    w_mean = W.mean(axis=0)
    f1_arg = weighted_f1(y, probs.argmax(1), n_classes)
    f1_nest = weighted_f1(y, nested_pred, n_classes)
    use_identity = fallback_to_argmax and f1_nest <= f1_arg
    w_final = np.ones(n_classes) if use_identity else w_mean
    return ClassWeightResult(
        weights=w_final, f1_argmax=f1_arg, f1_nested=f1_nest,
        f1_in_sample=weighted_f1(y, apply_class_weights(probs, w_final), n_classes),
        used_identity=use_identity, fold_weights=W,
    )
