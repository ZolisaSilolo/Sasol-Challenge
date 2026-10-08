"""
Regression model wrappers for CLV targets (Head 1 & Head 2).
Supports LightGBM, CatBoost, XGBoost, and Two-Stage Hurdle Architecture.
"""

from .base import BaseModel
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb


class LGBMRegressorModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 42,
            'n_estimators': 300,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1
        }
        if params:
            default_params.update(params)
        self.model = lgb.LGBMRegressor(**default_params)
        
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


class CatBoostRegressorModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'iterations': 400,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 0
        }
        if params:
            default_params.update(params)
        self.model = cb.CatBoostRegressor(**default_params)
        
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


class XGBoostRegressorModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist'
        }
        if params:
            default_params.update(params)
        self.model = xgb.XGBRegressor(**default_params)
        
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


class HurdleRegressorModel(BaseModel):
    """
    Two-Stage Hurdle Model for zero-inflated non-fuel CLV:
    Stage A: Binary Classification P(spend > 0)
    Stage B: Regressor on positive spenders E[spend | spend > 0]
    Final = P(spend > 0) * E[spend | spend > 0]
    """
    def __init__(self, classifier_params=None, regressor_params=None):
        clf_p = {
            'objective': 'binary',
            'random_state': 42,
            'n_estimators': 250,
            'learning_rate': 0.04,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1
        }
        if classifier_params:
            clf_p.update(classifier_params)
            
        reg_p = {
            'iterations': 300,
            'learning_rate': 0.03,
            'depth': 5,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 0
        }
        if regressor_params:
            reg_p.update(regressor_params)
            
        self.classifier = lgb.LGBMClassifier(**clf_p)
        self.regressor = cb.CatBoostRegressor(**reg_p)
        
    def fit(self, X, y, **kwargs):
        y_bin = (y > 0).astype(int)
        self.classifier.fit(X, y_bin)
        
        pos_mask = (y > 0)
        if pos_mask.sum() > 5:
            X_pos = X[pos_mask] if hasattr(X, "iloc") else X[pos_mask]
            y_pos = y[pos_mask]
            self.regressor.fit(X_pos, y_pos)
        return self
            
    def predict(self, X):
        p_active = self.classifier.predict_proba(X)[:, 1]
        try:
            preds_amount = np.clip(self.regressor.predict(X), 0, None)
        except Exception:
            preds_amount = np.zeros(len(X))
        return np.clip(p_active * preds_amount, 0, None)
