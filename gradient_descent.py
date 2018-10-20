import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = compute_error(y, tx, w)
    N = y.shape[0]
    grad = -(tx.T@e)/N
    return (grad, e)
