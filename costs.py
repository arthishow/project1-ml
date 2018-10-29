import numpy as np
from proj1_helpers import predict_labels

def compute_categorical_loss(y, tx, w):
    """Compute the percentage of wrongly categorized predictions."""
    pred = predict_labels(w, tx)
    e = y - pred
    loss = compute_mae(e)/2
    return loss

def compute_log_likelihood(y, tx, w):
    """Compute the log-likelihood."""
    loss = np.sum(np.log(1 + np.exp(tx@w)) - y*(tx@w))
    return loss

def compute_mse_loss(y, tx, w):
    """Compute the mean square error between y and tx multiplied by w."""
    return compute_mse(compute_error(y, tx, w))

def compute_mae_loss(y, tx, w):
    """Compute the mean absolute error between y and tx multiplied by w."""
    return compute_mae(compute_error(y, tx, w))

def compute_error(y, tx, w):
    """Compute the error"""
    e = y-tx@w.T
    return e

def compute_mse(e):
    """Compute the mean square error."""
    loss = np.mean(e**2)/2
    return loss

def compute_mae(e):
    """Compute the mean absolute error."""
    loss = np.mean(abs(e))
    return loss
