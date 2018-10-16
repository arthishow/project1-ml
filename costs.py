# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
from proj1_helpers import *
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e=y-tx@np.transpose(w)
    return (1/(2*len(y)))*sum(e**2)
def accuracy(y, tx, w):
    
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    pred=predict_labels(w,tx)
    e=y-pred
    return (1/(2*len(y)))*sum(e**2)