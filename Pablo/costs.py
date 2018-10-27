# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
from proj1_helpers import *
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    pred=predict_labels(w,tx)
    e=y-pred
    return (1/(2*len(y)))*sum(e**2) #mse


def compute_rmse(y,tx,w):
    rmse = np.sqrt(compute_loss(y,tx,w))
    return rmse


''' error fcts from arthur's notebook'''

def compute_categorical_loss(y, tx, w):
    """Compute the percentage of wrongly categorized predictions."""
    pred = predict_labels(w, tx)
    e = y - pred
    N = y.shape[0]
    loss = compute_mae(e)/2
    return loss

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
    loss = np.mean(abs(e))
    return loss