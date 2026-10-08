import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

def optimize_thresholds(y_true, y_probs):
    """
    Optimize F1 decision thresholds using Nelder-Mead.
    """
    n_classes = y_probs.shape[1]
    
    def loss_func(weights):
        # Apply weights to probabilities
        weighted_probs = y_probs * weights
        # Get predictions
        y_pred = np.argmax(weighted_probs, axis=1)
        # We want to maximize F1, so minimize negative F1
        score = f1_score(y_true, y_pred, average='weighted')
        return -score
        
    # Initial weights are all 1s
    initial_weights = np.ones(n_classes)
    
    # Run optimization
    result = minimize(loss_func, initial_weights, method='Nelder-Mead')
    
    return result.x

def apply_thresholds(y_probs, weights):
    weighted_probs = y_probs * weights
    return np.argmax(weighted_probs, axis=1)
