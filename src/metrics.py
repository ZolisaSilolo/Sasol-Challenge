import numpy as np
from sklearn.metrics import mean_squared_error, f1_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def composite_metric(y_fuel_true, y_fuel_pred, 
                    y_nonfuel_true, y_nonfuel_pred, 
                    y_opp_true, y_opp_pred,
                    s_fuel=1.0, s_nonfuel=1.0):
    
    rmse_fuel = rmse(y_fuel_true, y_fuel_pred)
    rmse_nonfuel = rmse(y_nonfuel_true, y_nonfuel_pred)
    f1_opp = f1_score(y_opp_true, y_opp_pred, average='weighted')
    
    # As per prompt, we return the components for detailed breakdown
    # Score_composite = f(RMSE_fuel, RMSE_nonfuel, Weighted_F1_opportunity)
    # The actual composition wasn't fully specified, but usually it's RMSE - F1 or similar
    score = (rmse_fuel + rmse_nonfuel) - f1_opp 
    
    return {
        'rmse_fuel': rmse_fuel,
        'rmse_nonfuel': rmse_nonfuel,
        'f1_opportunity': f1_opp,
        'composite_score': score
    }
