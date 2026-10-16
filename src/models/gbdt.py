"""
GBDT engine factory with automatic hardware placement.

* XGBoost:  ``tree_method='hist'`` + ``device='cuda'`` on GPU hosts.
* CatBoost: ``task_type='GPU'`` on GPU hosts.
* LightGBM: CPU (PyPI wheels are CPU-only); deterministic, all cores.

All engines are seeded from the pipeline seed (42). Multi-class models are wrapped so
that classes absent from a training fold still yield a full 17-column probability matrix.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from src.utils import resolve_device

ENGINES = ("lgbm", "catboost", "xgboost")

warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")


def _lgbm(task: str, params: dict, seed: int, n_classes: Optional[int]):
    import lightgbm as lgb

    base = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbose=-1, deterministic=True, force_col_wise=True)
    if task == "regression":
        base["objective"] = "regression"
    elif task == "huber":
        base.update(objective="huber", alpha=1.0)
    elif task == "binary":
        base["objective"] = "binary"
    elif task == "multiclass":
        base["objective"] = "multiclass"
    base.update(params or {})
    cls = lgb.LGBMRegressor if task in ("regression", "huber") else lgb.LGBMClassifier
    return cls(**base)


def _catboost(task: str, params: dict, seed: int, device: str, n_classes: Optional[int]):
    import catboost as cb

    base: Dict[str, Any] = dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                                random_seed=seed, verbose=0, allow_writing_files=False)
    base["loss_function"] = {"regression": "RMSE", "huber": "Huber:delta=1.0", "binary": "Logloss",
                             "multiclass": "MultiClass"}[task]
    if device == "cuda":
        base.update(task_type="GPU", devices="0")
    else:
        base.update(thread_count=-1)
    base.update(params or {})
    cls = cb.CatBoostRegressor if task in ("regression", "huber") else cb.CatBoostClassifier
    return cls(**base)


def _xgboost(task: str, params: dict, seed: int, device: str, n_classes: Optional[int]):
    import xgboost as xgb

    base = dict(n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
                device=device, random_state=seed, n_jobs=-1, verbosity=0)
    base["objective"] = {"regression": "reg:squarederror", "huber": "reg:pseudohubererror",
                         "binary": "binary:logistic", "multiclass": "multi:softprob"}[task]
    if task == "multiclass":
        base["eval_metric"] = "mlogloss"
    base.update(params or {})
    cls = xgb.XGBRegressor if task in ("regression", "huber") else xgb.XGBClassifier
    return cls(**base)


def build_estimator(engine: str, task: str, params: Optional[dict] = None, seed: int = 42,
                    device: Optional[str] = None, n_classes: Optional[int] = None):
    device = device or resolve_device()
    params = dict(params or {})
    if engine == "lgbm":
        return _lgbm(task, params, seed, n_classes)
    if engine == "catboost":
        return _catboost(task, params, seed, device, n_classes)
    if engine == "xgboost":
        return _xgboost(task, params, seed, device, n_classes)
    raise ValueError(f"Unknown engine '{engine}' (expected one of {ENGINES})")


class Regressor:
    def __init__(self, engine: str, params: Optional[dict] = None, seed: int = 42,
                 device: Optional[str] = None, loss: str = "regression"):
        self.engine, self.model = engine, build_estimator(engine, loss, params, seed, device)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.float64).ravel()


class BinaryClassifier:
    """P(y=1). Optional isotonic calibration via inner CV (sklearn CalibratedClassifierCV)."""

    def __init__(self, engine: str, params: Optional[dict] = None, seed: int = 42,
                 device: Optional[str] = None, calibrate: bool = True, calibration_cv: int = 3):
        self.engine = engine
        self.base = build_estimator(engine, "binary", params, seed, device)
        self.calibrate, self.cv, self.seed = calibrate, calibration_cv, seed
        self.model = None
        self._const: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        y = np.asarray(y).astype(int)
        if y.min() == y.max():                       # degenerate fold
            self._const = float(y[0])
            return self
        if self.calibrate and np.bincount(y).min() >= self.cv * 2:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import StratifiedKFold

            self.model = CalibratedClassifierCV(
                self.base, method="isotonic",
                cv=StratifiedKFold(self.cv, shuffle=True, random_state=self.seed))
        else:
            self.model = self.base
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._const is not None:
            return np.full(len(X), self._const)
        return np.asarray(self.model.predict_proba(X))[:, 1]


class MultiClassifier:
    """Fits on the classes present and expands probabilities back to ``n_classes`` columns."""

    def __init__(self, engine: str, n_classes: int, params: Optional[dict] = None, seed: int = 42,
                 device: Optional[str] = None):
        self.engine, self.n_classes = engine, n_classes
        self.params, self.seed, self.device = params, seed, device
        self.model = None
        self.present: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        y = np.asarray(y).astype(int)
        self.present = np.unique(y)
        if len(self.present) == 1:
            return self
        remap = np.full(self.n_classes, -1)
        remap[self.present] = np.arange(len(self.present))
        task = "multiclass" if len(self.present) > 2 else "binary"
        self.model = build_estimator(self.engine, task, self.params, self.seed, self.device)
        self.model.fit(X, remap[y], sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), self.n_classes))
        if self.model is None:
            out[:, self.present[0]] = 1.0
            return out
        out[:, self.present] = np.asarray(self.model.predict_proba(X))
        return out
