"""
Multi-class classification models for Opportunity behavioral transition target (Head 3).
Supports LightGBM, CatBoost, and XGBoost.
"""

from .base import BaseModel
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import numpy as np


class LGBMClassifierModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'objective': 'multiclass',
            'random_state': 42,
            'n_estimators': 300,
            'learning_rate': 0.04,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'verbose': -1
        }
        if params:
            default_params.update(params)
        self.model = lgb.LGBMClassifier(**default_params)
        
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostClassifierModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'loss_function': 'MultiClass',
            'iterations': 400,
            'learning_rate': 0.04,
            'depth': 6,
            'random_seed': 42,
            'auto_class_weights': 'Balanced',
            'verbose': 0
        }
        if params:
            default_params.update(params)
        self.model = cb.CatBoostClassifier(**default_params)
        
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        preds = self.model.predict(X)
        if len(preds.shape) > 1:
            return preds.ravel()
        return preds
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoostClassifierModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'n_estimators': 300,
            'learning_rate': 0.04,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist'
        }
        if params:
            default_params.update(params)
        self.model = xgb.XGBClassifier(**default_params)
        
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
