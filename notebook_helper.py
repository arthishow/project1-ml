import numpy as np
from helper import build_poly
from implementations import least_squares
from costs import compute_categorical_loss

def fast_build_poly(x, poly_degree_minus_one, degree):
    """Fast polynomial basis functions for input data x to be used in loops only."""
    return np.c_[poly_degree_minus_one, np.power(x, degree)]

def split_data(x, y, ratio, seed=23):
    """Split the given datasets based on the split ratio."""
    np.random.seed(seed)
    N = y.shape[0]
    indices = np.random.permutation(N)
    index_split = int(np.floor(ratio * N))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def build_k_indices(y, k_fold, seed=23):
    """Build k indices for k-fold cross validation."""
    N = y.shape[0]
    interval = int(N / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, degree):
    """Cross-validation using least squares."""
    loss_te = 0
    loss_tr = 0

    for i in range(k):
        x_te = x[k_indices[i]]
        y_te = y[k_indices[i]]
        remaining_indices = np.delete(np.arange(y.shape[0]), k_indices[i])
        x_tr = x[remaining_indices]
        y_tr = y[remaining_indices]

        tx_te = build_poly(x_te, degree)
        tx_tr = build_poly(x_tr, degree)
        weights_tr, loss = least_squares(y_tr, tx_tr)

        loss_tr += loss
        loss_te += compute_mse_loss(y_te, tx_te, weights_tr)

    loss_te = loss_te/k
    loss_tr = loss_tr/k

    return loss_tr, loss_te
