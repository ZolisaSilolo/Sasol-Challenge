"""
Two-stage hurdle model for zero-inflated CLV_nonfuel (Head 2, Branch B).

Stage 2A: calibrated binary GBDT(s) for P(z > 0).
Stage 2B: GBDT regressor(s) fitted *only* on rows with z > 0, i.e. E[z | z > 0].
Prediction: E[z] = P(z > 0) * E[z | z > 0]  (RMSE-optimal combination in z-space).

``predict`` returns an (n, 3) matrix: [expected z, P(z>0), E[z | z>0]] so the pipeline can
use P(z>0) as an OOF meta-feature for Head 3.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.models.gbdt import BinaryClassifier, Regressor


class HurdleModel:
    def __init__(self, classifier_engines: Sequence[str] = ("lgbm",), regressor_engines: Sequence[str] = ("catboost",),
                 classifier_params: Optional[Dict[str, dict]] = None, regressor_params: Optional[Dict[str, dict]] = None,
                 calibrate: bool = True, seed: int = 42, device: Optional[str] = None):
        cp, rp = classifier_params or {}, regressor_params or {}
        self.classifiers = [BinaryClassifier(e, cp.get(e), seed, device, calibrate=calibrate) for e in classifier_engines]
        self.regressors = [Regressor(e, rp.get(e), seed, device) for e in regressor_engines]
        self._pos_mean: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        pos = y > 0
        for c in self.classifiers:
            c.fit(X, pos.astype(int), sample_weight)
        sw = None if sample_weight is None else sample_weight[pos]
        if pos.sum() >= 20:
            for r in self.regressors:
                r.fit(X[pos], y[pos], sw)
        else:
            self._pos_mean = float(y[pos].mean()) if pos.any() else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = np.mean([c.predict_proba(X) for c in self.classifiers], axis=0)
        if self._pos_mean is not None:
            mu = np.full(len(X), self._pos_mean)
        else:
            mu = np.clip(np.mean([r.predict(X) for r in self.regressors], axis=0), 0.0, None)
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([p * mu, p, mu])
