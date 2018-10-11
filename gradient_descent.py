# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y- tx @ np.transpose(w)
    return (-1/len(y)) * np.transpose(tx) @ e

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad=compute_gradient(y,tx,w)
        w=w-gamma*grad
        # store w and loss
    return w
