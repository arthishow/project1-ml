# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e=y-tx@np.transpose(w)
    return (1/(2*len(y)))*sum(e**2)