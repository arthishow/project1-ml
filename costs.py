import numpy as np
from proj1_helpers import *

def accuracy(y, tx, w):
    pred=predict_labels(w,tx)
    e=y-pred
    return (1/(2*len(y)))*sum(e**2)

def compute_mse_loss(y, tx, w):
    """Compute the mean square error."""
    return compute_mse(compute_error(y, tx, w))

def compute_mae_loss(y, tx, w):
    """Compute the mean absolute error."""
    return compute_mae(compute_error(y, tx, w))

def compute_error(y, tx, w):
    """Compute the error"""
    e = y-tx@w.T
    return e

def compute_mse(e):
    loss = np.mean(e**2)/2
    return loss

def compute_mae(e):
    loss = np.mean(abs(e))/2
    return loss
