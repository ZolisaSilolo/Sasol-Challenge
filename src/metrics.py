"""
Competition metrics.

Tracked components: RMSE(CLV_fuel) (weight 0.3), RMSE(CLV_nonfuel) (weight 0.3),
weighted F1(Opportunity) (weight 0.4). The organisers' exact aggregation is not
published in the repo, so we report each component plus

    composite_loss = 0.3*RMSE_fuel + 0.3*RMSE_nonfuel + 0.4*(1 - F1)   (lower is better)

which is monotone in every component and uses the stated weights.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score

WEIGHTS = {"rmse_fuel": 0.3, "rmse_nonfuel": 0.3, "f1_opportunity": 0.4}


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def composite_metric(y_fuel_true, y_fuel_pred, y_nonfuel_true, y_nonfuel_pred,
                     y_opp_true, y_opp_pred) -> Dict[str, float]:
    r_f = rmse(y_fuel_true, y_fuel_pred)
    r_n = rmse(y_nonfuel_true, y_nonfuel_pred)
    f1 = float(f1_score(y_opp_true, y_opp_pred, average="weighted", zero_division=0))
    return {
        "rmse_fuel": r_f,
        "rmse_nonfuel": r_n,
        "f1_opportunity": f1,
        "composite_loss": WEIGHTS["rmse_fuel"] * r_f + WEIGHTS["rmse_nonfuel"] * r_n
        + WEIGHTS["f1_opportunity"] * (1.0 - f1),
    }
