# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import costs
from costs import compute_loss
from ridge_regression import ridge_regression
from build_polynomial import build_poly

from costs import compute_loss
from ridge_regression import ridge_regression
from build_polynomial import build_poly

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    m = []
    for i in range(len(k_indices)):
        if i != k:
            m =np.concatenate((m,k_indices[i]),axis=None) 
    xtest = x[k_indices[k]]
    ytest = y[k_indices[k]] 
    xtrain = x[m.astype(int)]
    ytrain = y[m.astype(int)]
    polytrain = build_poly(xtrain,degree)
    polytest = build_poly(xtest,degree)
    w = ridge_regression(ytrain,polytrain,lambda_)
    loss_tr = compute_loss(ytrain,polytrain,w)
    loss_te = compute_loss(ytest,polytest,w)
    return loss_tr, loss_te